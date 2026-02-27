import os
import gc
import subprocess
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

from src.core.utils import filter_hallucinated_segments

logger = logging.getLogger(__name__)

# FFmpeg 最大执行时间 (秒)
FFMPEG_TIMEOUT = 600

class FullPipelineEngine:
    def __init__(self):
        self.transcribe_model = None
        self.align_model = None
        self.diarize_model = None
        self.current_model_size = None
        self.current_compute_type = None
        # 用于跟踪 ASR 运行时参数，变化时重载模型
        self._current_initial_prompt = None
        self._current_vad_onset = None
        self._current_lang = None

    def _clear_gpu(self):
        """强制执行垃圾回收并清空 CUDA 缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self):
        """卸载所有模型实例并释放显存"""
        logger.info("执行资源释放...")
        if self.transcribe_model: del self.transcribe_model
        if self.align_model: del self.align_model
        if self.diarize_model: del self.diarize_model
        self.transcribe_model = None
        self.align_model = None
        self.diarize_model = None
        self._current_initial_prompt = None
        self._current_vad_onset = None
        self._current_lang = None
        self._clear_gpu()
        logger.info("显存已重置")

    def _run_ffmpeg(self, cmd: list, description: str = "FFmpeg") -> None:
        """统一的 FFmpeg 调用，带超时和错误输出捕获"""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT
            )
            if result.returncode != 0:
                stderr_snippet = (result.stderr or "")[:500]
                raise RuntimeError(f"{description} 失败 (code={result.returncode}): {stderr_snippet}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{description} 执行超时 (超过 {FFMPEG_TIMEOUT}s)")

    def _convert_to_wav(self, input_path: str) -> str:
        """预处理：将输入媒体转换为 16k 单声道 WAV 格式"""
        logger.info("音频标准化处理中...")
        # 使用 tempfile 生成唯一文件名，避免冲突
        fd, output_wav = tempfile.mkstemp(suffix=".wav", prefix="whisperx_base_")
        os.close(fd)
            
        cmd = ['ffmpeg', '-y', '-i', input_path, '-ac', '1', '-ar', '16000', output_wav]
        self._run_ffmpeg(cmd, "音频格式转换")
        return output_wav

    def _isolate_vocals(self, input_path: str) -> str:
        """
        执行人声分离 (BS-RoFormer)
        """
        logger.info("运行 BS-RoFormer 分离模型...")

        temp_out_dir = Path(tempfile.mkdtemp(prefix="roformer_out_"))

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
            logger.info(f"人声提取完成: {vocals_path.name}")
            
            # 格式标准化：44.1k Stereo -> 16k Mono
            final_wav = temp_out_dir / "final_whisper_ready.wav"
            cmd = [
                'ffmpeg', '-y', 
                '-i', str(vocals_path), 
                '-ac', '1',
                '-ar', '16000',
                str(final_wav)
            ]
            self._run_ffmpeg(cmd, "人声音频格式转换")
            
            return str(final_wav)

        except Exception as e:
            logger.error(f"BS-RoFormer 执行异常: {e}")
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
        enable_demucs: bool = False,
        hallucination_filter: bool = True,
        hallucination_threshold: float = 0.35,
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
            logger.info(f"1. 执行转录 ({model_size} | {compute_type})...")
            
            asr_options = {
                "initial_prompt": initial_prompt, 
                "hotwords": None,
                # 仅保留 condition_on_previous_text 防止幻觉自我强化
                # no_speech_threshold / log_prob_threshold / compression_ratio_threshold
                # 这些参数对歌曲/音乐场景过于激进（歌声会被误判为非语音而丢弃），
                # 因此不在 ASR 层面做过滤，完全依赖后处理的 filter_hallucinated_segments() 函数。
                "condition_on_previous_text": False,
            }
            
            vad_options = {
                "vad_onset": vad_onset, 
                "vad_offset": 0.363 
            }
            
            # 检查是否需要重新加载模型（包含 ASR 运行时参数变化）
            needs_reload = (
                self.transcribe_model is None or 
                self.current_model_size != model_size or
                self.current_compute_type != compute_type or
                self._current_initial_prompt != initial_prompt or
                self._current_vad_onset != vad_onset or
                self._current_lang != lang
            )

            if needs_reload:
                logger.info(f"加载模型权重: {model_size} ({compute_type})...")
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
                self._current_initial_prompt = initial_prompt
                self._current_vad_onset = vad_onset
                self._current_lang = lang
            
            audio = whisperx.load_audio(processing_audio)
            
            result = self.transcribe_model.transcribe(
                audio, 
                batch_size=BATCH_SIZE
            )
            
            # 3. 强对齐阶段 (Alignment)
            logger.info("2. 执行音素级对齐...")
            if result["segments"]:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=DEVICE
                )
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
                )
                del model_a
                self._clear_gpu()
            logger.info("转录与对齐完成")

            # 3.5 幻觉过滤阶段 (Hallucination Filtering)
            if hallucination_filter:
                logger.info("2.5 执行幻觉过滤...")
                result["segments"] = filter_hallucinated_segments(
                    result["segments"],
                    confidence_threshold=hallucination_threshold,
                )

            # 4. 说话人区分阶段 (Diarization)
            if enable_diarization:
                if not HF_TOKEN:
                    return result["segments"], "Error: HF_TOKEN 未配置"

                logger.info("3. 执行说话人聚类 (Pyannote)...")
                try:
                    # 缓存 diarize 模型，避免每次重新加载
                    if self.diarize_model is None:
                        self.diarize_model = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=HF_TOKEN
                        )
                        self.diarize_model.to(torch.device(DEVICE))
                except Exception as e:
                    logger.error(f"Pyannote 初始化失败: {e}")
                    return result["segments"], f"Pyannote Init Failed: {e}"

                diarize_segments = self.diarize_model(
                    processing_audio, 
                    min_speakers=min_speakers, 
                    max_speakers=max_speakers
                )
                
                logger.info("合并聚类结果...")
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
                
                # 不再每次删除 diarize_model，保留缓存供下次复用
                self._clear_gpu()

            return result["segments"], "Success"

        except Exception as e:
            logger.error(f"管道执行异常: {e}")
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