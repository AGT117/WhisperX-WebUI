import os
import gc
import shutil
import tempfile
import traceback
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
from config.settings import DEVICE, COMPUTE_TYPE, BATCH_SIZE, HF_TOKEN, AUDIO_SEPARATOR_HOME

import torch
import pandas as pd
import whisperx
from pyannote.audio import Pipeline
from whisperx.diarize import assign_word_speakers
from audio_separator.separator import Separator

class FullPipelineEngine:
    def __init__(self):
        self.transcribe_model = None
        self.align_model = None
        self.diarize_model = None
        self.current_model_size = None
        self.current_compute_type = None 

    def _clear_gpu(self):
        """强制执行垃圾回收并清空 CUDA 缓存"""
        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self):
        """卸载所有模型实例并释放显存"""
        print("执行资源释放...")
        if self.transcribe_model: del self.transcribe_model
        if self.align_model: del self.align_model
        if self.diarize_model: del self.diarize_model
        self.transcribe_model = None
        self.align_model = None
        self.diarize_model = None
        self._clear_gpu()
        print("显存已重置")

    def _convert_to_wav(self, input_path: str) -> str:
        """预处理：将输入媒体转换为 16k 单声道 WAV 格式"""
        import subprocess
        print(f"音频标准化处理中...")
        temp_dir = tempfile.gettempdir()
        temp_name = f"whisperx_base_{os.getpid()}_{hash(input_path)}.wav"
        output_wav = os.path.join(temp_dir, temp_name)
        
        if os.path.exists(output_wav):
            try: os.remove(output_wav)
            except OSError: pass
            
        try:
            cmd = ['ffmpeg', '-y', '-i', input_path, '-ac', '1', '-ar', '16000', output_wav]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_wav
        except Exception as e:
            raise RuntimeError(f"FFmpeg 处理失败: {e}")

    def _isolate_vocals(self, input_path: str) -> str:
        """
        执行人声分离 (BS-RoFormer)
        """
        print(f"运行 BS-RoFormer 分离模型...")

        temp_out_dir = Path(tempfile.gettempdir()) / f"roformer_out_{os.getpid()}_{hash(input_path)}"
        temp_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            sep = Separator(
                log_level=logging.ERROR,
                model_file_dir=str(AUDIO_SEPARATOR_HOME),
                output_dir=str(temp_out_dir),
                output_format="wav",
                output_single_stem="Vocals"
            )

            # 加载指定模型检查点
            model_filename = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            sep.load_model(model_filename=model_filename)

            output_files = sep.separate(input_path)

            if not output_files:
                raise RuntimeError("分离过程未生成输出文件")

            vocals_path = temp_out_dir / output_files[0]
            print(f"人声提取完成: {vocals_path.name}")
            
            # 格式标准化：44.1k Stereo -> 16k Mono
            final_wav = temp_out_dir / "final_whisper_ready.wav"
            import subprocess
            cmd = [
                'ffmpeg', '-y', 
                '-i', str(vocals_path), 
                '-ac', '1',
                '-ar', '16000',
                str(final_wav)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return str(final_wav)

        except Exception as e:
            print(f"BS-RoFormer 执行异常: {e}")
            traceback.print_exc()
            raise e

    def run_pipeline(
        self, 
        audio_path: str, 
        model_size: str = "large-v2", 
        lang: Optional[str] = None, 
        enable_diarization: bool = False, 
        min_speakers: Optional[int] = None, 
        max_speakers: Optional[int] = None,
        vad_onset: float = 0.5,           
        initial_prompt: Optional[str] = None,
        compute_type: str = "float16",
        enable_demucs: bool = False 
    ) -> Tuple[Any, str]:
        
        temp_files_to_clean = []
        
        try:
            processing_audio = None
            
            # 1. 预处理阶段
            if enable_demucs:
                processing_audio = self._isolate_vocals(audio_path)
                temp_files_to_clean.append(processing_audio)
                temp_files_to_clean.append(str(Path(processing_audio).parent))
            else:
                processing_audio = self._convert_to_wav(audio_path)
                temp_files_to_clean.append(processing_audio)
            
            # 2. 转录阶段 (ASR)
            print(f"1. 执行转录 ({model_size} | {compute_type})...")
            
            asr_options = {
                "initial_prompt": initial_prompt, 
                "hotwords": None,
            }
            
            vad_options = {
                "vad_onset": vad_onset, 
                "vad_offset": 0.363 
            }
            
            # 检查是否需要重新加载模型
            needs_reload = (
                self.transcribe_model is None or 
                self.current_model_size != model_size or
                self.current_compute_type != compute_type
            )

            if needs_reload:
                print(f"加载模型权重: {model_size} ({compute_type})...")
                self._clear_gpu()
                self.transcribe_model = whisperx.load_model(
                    model_size, 
                    DEVICE, 
                    compute_type=compute_type, 
                    language=lang,
                    asr_options=asr_options, 
                    vad_options=vad_options    
                )
                self.current_model_size = model_size
                self.current_compute_type = compute_type
            
            audio = whisperx.load_audio(processing_audio)
            
            result = self.transcribe_model.transcribe(
                audio, 
                batch_size=BATCH_SIZE
            )
            
            # 3. 强对齐阶段 (Alignment)
            print("2. 执行音素级对齐...")
            if result["segments"]:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=DEVICE
                )
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
                )
                del model_a
                self._clear_gpu()
            print("转录与对齐完成")

            # 4. 说话人区分阶段 (Diarization)
            if enable_diarization:
                if not HF_TOKEN:
                    return result["segments"], "Error: HF_TOKEN 未配置"

                print("3. 执行说话人聚类 (Pyannote)...")
                try:
                    self.diarize_model = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=HF_TOKEN
                    )
                except Exception as e:
                    print(f"Pyannote 初始化失败: {e}")
                    return result["segments"], f"Pyannote Init Failed: {e}"

                self.diarize_model.to(torch.device(DEVICE))

                diarize_segments = self.diarize_model(
                    processing_audio, 
                    min_speakers=min_speakers, 
                    max_speakers=max_speakers
                )
                
                print("合并聚类结果...")
                # 兼容 pyannote 不同版本返回类型：
                # - 旧版直接返回 Annotation（具有 itertracks）
                # - 新版返回 DiarizeOutput（包含 speaker_diarization 属性）
                if hasattr(diarize_segments, 'itertracks'):
                    tracks_iter = diarize_segments.itertracks(yield_label=True)
                elif hasattr(diarize_segments, 'speaker_diarization'):
                    tracks_iter = diarize_segments.speaker_diarization.itertracks(
                        yield_label=True
                    )
                else:
                    raise RuntimeError('Unsupported diarization output type')

                diarize_df = pd.DataFrame(
                    tracks_iter,
                    columns=['segment', 'label', 'speaker']
                )
                diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
                diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
                
                result = assign_word_speakers(diarize_df, result)
                
                del self.diarize_model
                self.diarize_model = None
                self._clear_gpu()

            return result["segments"], "Success"

        except Exception as e:
            print(f"管道执行异常: {e}")
            traceback.print_exc()
            return [], f"Pipeline Exception: {str(e)}"
            
        finally:
            # 清理临时文件
            for path_str in temp_files_to_clean:
                try:
                    p = Path(path_str)
                    if p.is_file(): os.remove(p)
                    elif p.is_dir(): shutil.rmtree(p)
                except OSError: pass