import os
import gc
import subprocess
import shutil
import tempfile
import traceback
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Callable, Dict
from config.settings import DEVICE, COMPUTE_TYPE, BATCH_SIZE, HF_TOKEN, AUDIO_SEPARATOR_HOME
from config.settings import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, is_llm_configured

import torch
import pandas as pd
import whisperx
from pyannote.audio import Pipeline
from whisperx.diarize import assign_word_speakers
from audio_separator.separator import Separator

from src.core.utils import filter_hallucinated_segments
from src.core.llm_processor import LLMProcessor
from src.core.data_cleaner import DataCleaner, CleaningConfig, compute_segment_snr, load_blacklist, rule_g_blacklist_filter, rule_g_blacklist_soft_label
from src.core.corpus_exporter import compute_advanced_features, export_jsonl, merge_jsonl_files

logger = logging.getLogger(__name__)

# FFmpeg 最大执行时间 (秒)
FFMPEG_TIMEOUT = 600


_RULE_H_REASON_TEXT = {
    'ad': '商业推广/内容无关',
    'emo': '情感表达不足',
    'asr': '疑似识别幻觉',
    'keep': '保守保留',
    'override': '规则标记被LLM推翻',
    'other': '其他语义原因',
    'unchecked': 'API未返回(降级保留)',
}


def _format_rule_h_reason(reason_code: str) -> str:
    text = _RULE_H_REASON_TEXT.get(reason_code, _RULE_H_REASON_TEXT['other'])
    return f"LLM语义过滤({text})"


def _pick_override_removed_rule(seg: dict) -> str:
    """
    在 LLM 接管模式下，为被 LLM 最终删除的段落确定主归因规则。

    若命中过滤软标签，则优先归因给对应前置规则；
    否则归因为 rule_h（纯语义判定删除）。
    """
    if seg.get('flag_g_blacklist'):
        return 'rule_g'
    if seg.get('flag_c_length_ratio'):
        return 'rule_c'
    if seg.get('flag_d_context_island'):
        return 'rule_d'
    if seg.get('flag_e_low_info'):
        return 'rule_e'
    return 'rule_h'

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
                cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
                encoding='utf-8', errors='replace',
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

    def _notify(self, callback: Optional[Callable], fraction: float, desc: str):
        """安全调用进度回调"""
        if callback:
            try:
                callback(fraction, desc)
            except Exception:
                pass

    def _cancel_status_if_requested(
        self,
        stop_event: Optional[Any],
        stage_desc: str = "",
    ) -> Optional[str]:
        """检查是否收到取消信号，命中时返回统一状态文本。"""
        if stop_event is None:
            return None
        try:
            if stop_event.is_set():
                status = "Cancelled: 用户手动停止"
                if stage_desc:
                    status = f"{status} ({stage_desc})"
                logger.info(status)
                return status
        except Exception:
            return None
        return None

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
        # 幻觉过滤模式: "code" = 代码规则 | "llm" = LLM 判断 | "off" = 关闭
        hallucination_mode: str = "code",
        hallucination_threshold: float = 0.35,
        # LLM 功能参数
        llm_enabled: bool = False,
        llm_mode: str = "segmentation",
        llm_target_lang: Optional[str] = None,
        # 进度回调: callable(fraction: float, description: str)
        progress_callback: Optional[Callable] = None,
        stop_event: Optional[Any] = None,
    ) -> Tuple[Any, str]:
        
        temp_files_to_clean = []
        
        try:
            cancel_status = self._cancel_status_if_requested(stop_event, "启动前")
            if cancel_status:
                return [], cancel_status

            processing_audio = None
            
            # 1. 预处理阶段
            self._notify(progress_callback, 0.0, "音频预处理中...")
            cancel_status = self._cancel_status_if_requested(stop_event, "预处理前")
            if cancel_status:
                return [], cancel_status

            if enable_demucs:
                processing_audio = self._isolate_vocals(audio_path)
                temp_files_to_clean.append(processing_audio)
                temp_files_to_clean.append(str(Path(processing_audio).parent))
            else:
                processing_audio = self._convert_to_wav(audio_path)
                temp_files_to_clean.append(processing_audio)
            self._notify(progress_callback, 0.10, "预处理完成")

            cancel_status = self._cancel_status_if_requested(stop_event, "ASR前")
            if cancel_status:
                return [], cancel_status
            
            # 2. 转录阶段 (ASR)
            self._notify(progress_callback, 0.10, "加载 ASR 模型...")
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
            
            self._notify(progress_callback, 0.20, "模型就绪，正在转录...")
            audio = whisperx.load_audio(processing_audio)
            
            result = self.transcribe_model.transcribe(
                audio, 
                batch_size=BATCH_SIZE
            )
            self._notify(progress_callback, 0.45, "转录完成")

            cancel_status = self._cancel_status_if_requested(stop_event, "对齐前")
            if cancel_status:
                return result.get("segments", []), cancel_status
            
            # 3. 强对齐阶段 (Alignment)
            self._notify(progress_callback, 0.45, "执行音素级对齐...")
            import time as _time_module
            t_align_start = _time_module.time()
            logger.info("2. 执行音素级对齐...")
            if result["segments"]:
                logger.info(f"   待对齐 segments: {len(result['segments'])} 条")
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=DEVICE
                )
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
                )
                del model_a
                self._clear_gpu()
            t_align_elapsed = _time_module.time() - t_align_start
            self._notify(progress_callback, 0.60, f"对齐完成 ({t_align_elapsed:.1f}s)")
            logger.info(f"转录与对齐完成 (对齐耗时: {t_align_elapsed:.1f}s)")

            cancel_status = self._cancel_status_if_requested(stop_event, "清洗前")
            if cancel_status:
                return result.get("segments", []), cancel_status

            # 3.5 幻觉过滤阶段
            if hallucination_mode == "code":
                self._notify(progress_callback, 0.60, "代码规则幻觉过滤...")
                logger.info("2.5 执行代码规则幻觉过滤...")
                result["segments"] = filter_hallucinated_segments(
                    result["segments"],
                    confidence_threshold=hallucination_threshold,
                )
            elif hallucination_mode == "llm":
                if is_llm_configured():
                    self._notify(progress_callback, 0.60, "LLM 幻觉过滤...")
                    logger.info("2.5 执行 LLM 幻觉过滤...")
                    try:
                        llm = LLMProcessor(
                            api_base=LLM_API_BASE,
                            api_key=LLM_API_KEY,
                            model=LLM_MODEL,
                            max_context_tokens=LLM_MAX_TOKENS,
                            temperature=LLM_TEMPERATURE,
                        )
                        result["segments"] = llm.filter_hallucinations(
                            result["segments"]
                        )
                        logger.info("LLM 幻觉过滤完成")
                    except Exception as e:
                        logger.error(f"LLM 幻觉过滤失败，回退到代码规则: {e}")
                        result["segments"] = filter_hallucinated_segments(
                            result["segments"],
                            confidence_threshold=hallucination_threshold,
                        )
                else:
                    logger.warning("LLM 未配置，回退到代码规则幻觉过滤")
                    result["segments"] = filter_hallucinated_segments(
                        result["segments"],
                        confidence_threshold=hallucination_threshold,
                    )
            # hallucination_mode == "off" → 不做任何过滤
            self._notify(progress_callback, 0.70, "幻觉过滤完成")

            # 3.6 LLM 智能断句/翻译（可选）
            if llm_enabled and is_llm_configured():
                self._notify(progress_callback, 0.70, f"LLM 智能处理 ({llm_mode})...")
                logger.info(f"3. LLM 智能处理 ({LLM_MODEL} | {llm_mode})...")
                try:
                    llm = LLMProcessor(
                        api_base=LLM_API_BASE,
                        api_key=LLM_API_KEY,
                        model=LLM_MODEL,
                        max_context_tokens=LLM_MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                    )
                    result["segments"] = llm.process_segments(
                        result["segments"],
                        mode=llm_mode,
                        target_lang=llm_target_lang,
                    )
                    logger.info("LLM 处理完成")
                except ImportError:
                    logger.warning("openai 库未安装，跳过 LLM 处理")
                except Exception as e:
                    logger.error(f"LLM 处理失败，使用原始断句: {e}")
            elif llm_enabled and not is_llm_configured():
                logger.warning("LLM 已启用但未配置 API，请编辑 config/llm_config.json")
            self._notify(progress_callback, 0.85, "LLM 处理完成")

            cancel_status = self._cancel_status_if_requested(stop_event, "说话人分离前")
            if cancel_status:
                return result.get("segments", []), cancel_status

            # 4. 说话人区分阶段 (Diarization)
            if enable_diarization:
                self._notify(progress_callback, 0.85, "说话人聚类...")
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

                # 将已加载的 numpy 音频转为 pyannote 接受的 waveform dict，
                # 避免重复读取音频文件，减少 I/O 开销
                waveform_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # (1, T)
                diarize_input = {"waveform": waveform_tensor, "sample_rate": 16000}

                # 构建聚类参数 —— 当 min == max 时直接用 num_speakers 跳过搜索
                diarize_kwargs = {}
                if (min_speakers is not None and max_speakers is not None
                        and min_speakers == max_speakers):
                    diarize_kwargs["num_speakers"] = min_speakers
                    logger.info(f"  已知说话人数={min_speakers}，跳过最优聚类搜索")
                else:
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers

                diarize_segments = self.diarize_model(
                    diarize_input, **diarize_kwargs
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

            cancel_status = self._cancel_status_if_requested(stop_event, "导出前")
            if cancel_status:
                return result.get("segments", []), cancel_status

            self._notify(progress_callback, 1.0, "处理完成")
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

    def run_corpus_pipeline(
        self,
        audio_path: str,
        output_dir: str,
        cleaning_config: Optional[CleaningConfig] = None,
        model_size: str = "large-v3",
        lang: Optional[str] = None,
        vad_onset: float = 0.5,
        initial_prompt: Optional[str] = None,
        compute_type: str = "float16",
        enable_demucs: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        export_jsonl_flag: bool = True,
        metadata: Optional[dict] = None,
        audio_display_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        stop_event: Optional[Any] = None,
    ) -> Tuple[list, dict, str]:
        """
        语料库构建专用管道：ASR → 说话人分离 → 数据清洗 → 特征计算 → JSONL 导出。
        
        Args:
            audio_path:       输入音频路径
            output_dir:       输出目录
            cleaning_config:  数据清洗配置
            其他参数同 run_pipeline
            
        Returns:
            (enriched_segments, cleaning_stats, status_message)
        """
        import time
        t_start = time.time()
        t_prev = t_start
        try:
            cancel_status = self._cancel_status_if_requested(stop_event, "语料任务启动前")
            if cancel_status:
                return [], {}, cancel_status

            # 0. 中文标点引导：如果用户未指定 prompt 且目标是中文，自动注入标点引导
            effective_prompt = initial_prompt
            if not effective_prompt and lang in ('zh', None):
                effective_prompt = (
                    "以下是普通话的句子。"
                    "请注意添加标点符号：逗号，句号。问号？感叹号！"
                )

            # 1. 调用核心 ASR + 说话人分离管道
            self._notify(progress_callback, 0.0, "执行 ASR 管道...")
            segments, status = self.run_pipeline(
                audio_path=audio_path,
                model_size=model_size,
                lang=lang,
                enable_diarization=True,  # 语料库模式强制启用说话人分离
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                vad_onset=vad_onset,
                initial_prompt=effective_prompt,
                compute_type=compute_type,
                enable_demucs=enable_demucs,
                hallucination_mode="code",
                hallucination_threshold=0.35,
                progress_callback=lambda frac, desc: self._notify(
                    progress_callback, frac * 0.6, desc
                ) if progress_callback else None,
                stop_event=stop_event,
            )

            if status.startswith("Cancelled"):
                return [], {}, status

            if status != "Success":
                return [], {}, f"ASR 管道失败: {status}"
            
            t_asr_total = time.time() - t_prev
            logger.info(f"[ASR耗时] {t_asr_total:.1f}s，共产生 {len(segments)} 段")
            t_prev = time.time()

            cancel_status = self._cancel_status_if_requested(stop_event, "清洗前")
            if cancel_status:
                return [], {}, cancel_status

            # 1.5 SNR 计算（如果启用了规则 F）
            cfg = cleaning_config or CleaningConfig()
            if cfg.enable_snr_filter:
                self._notify(progress_callback, 0.55, "计算信噪比...")
                try:
                    import whisperx as _wx
                    snr_audio = _wx.load_audio(audio_path)
                    compute_segment_snr(snr_audio, segments, sample_rate=16000)
                    del snr_audio
                except Exception as e:
                    logger.warning(f"SNR 计算失败，跳过规则F: {e}")
            
            t_snr = time.time() - t_prev
            if t_snr > 0.1:
                logger.info(f"[SNR耗时] {t_snr:.1f}s")
            t_prev = time.time()

            # 2. 数据清洗
            self._notify(progress_callback, 0.60, "执行数据清洗...")
            
            override_mode = cfg.use_llm_override_mode
            logger.info(
                f"[清洗参数] LLM接管模式={override_mode} | "
                f"规则D: enable={cfg.enable_context_island_filter}, "
                f"orphan_window={cfg.orphan_window_sec}s, min_segments={cfg.min_segments_to_apply_d} | "
                f"规则G: enable={cfg.enable_blacklist_filter}"
            )
            
            cleaner = DataCleaner(cfg)
            if override_mode:
                # 接管模式：硬过滤(A/B/F) → 软标签(C/E/D) → 特征 → G(软) → LLM(总裁决)
                cleaned_segments = cleaner.hard_filter(segments)
                cleaned_segments = cleaner.soft_label(cleaned_segments)
            else:
                # 传统模式：C/D/E 直接删除
                cleaned_segments, _ = cleaner.clean(segments)
            stats = dict(cleaner.stats)
            stats['_removed_segments'] = list(cleaner.removed_segments)
            
            t_clean = time.time() - t_prev
            logger.info(f"[清洗耗时] {t_clean:.1f}s，{len(segments)} → {len(cleaned_segments)} 段")
            t_prev = time.time()

            cancel_status = self._cancel_status_if_requested(stop_event, "特征计算前")
            if cancel_status:
                return [], stats, cancel_status

            # 3. 高级特征计算
            self._notify(progress_callback, 0.75, "计算高级特征...")
            enriched = compute_advanced_features(cleaned_segments)
            
            t_feature = time.time() - t_prev
            logger.info(f"[特征耗时] {t_feature:.1f}s")
            t_prev = time.time()

            cancel_status = self._cancel_status_if_requested(stop_event, "规则G前")
            if cancel_status:
                return enriched, stats, cancel_status

            # 3.5 规则 G: 黑名单/脱轨内容过滤（需要 conversation_id，故在特征计算之后执行）
            if cfg.enable_blacklist_filter:
                self._notify(progress_callback, 0.80, "黑名单过滤...")
                blacklist = load_blacklist(cfg.blacklist_path)
                if override_mode:
                    # 接管模式：仅标记，不删除，不做上下文清除
                    enriched, rule_g_flagged = rule_g_blacklist_soft_label(enriched, blacklist)
                    stats['rule_g_flagged'] = rule_g_flagged
                else:
                    before_rule_g = enriched
                    enriched, rule_g_count = rule_g_blacklist_filter(
                        enriched, blacklist, context_purge=cfg.blacklist_context_purge
                    )
                    stats['rule_g_removed'] = rule_g_count
                    stats['output_count'] = stats.get('output_count', 0) - rule_g_count
                    kept_ids = {id(seg) for seg in enriched}
                    removed_samples = [seg for seg in before_rule_g if id(seg) not in kept_ids]
                    stats['rule_g_removed_samples'] = removed_samples
                    # 记录 Rule G 移除的段落到通用列表
                    for seg in removed_samples:
                        stats['_removed_segments'].append(
                            {
                                **seg,
                                '_removed_by': 'rule_g',
                                '_reason': '黑名单命中',
                                '_reason_code': 'blacklist',
                            }
                        )
            
            t_filter = time.time() - t_prev
            if t_filter > 0.1:
                logger.info(f"[黑名单过滤耗时] {t_filter:.1f}s")
            t_prev = time.time()

            # 3.6 规则 H: LLM 语义校验（需要 conversation_id，故在特征计算之后执行）
            if cfg.enable_llm_semantic_filter and is_llm_configured():
                self._notify(progress_callback, 0.82, "LLM 语义过滤...")
                try:
                    llm = LLMProcessor(
                        api_base=LLM_API_BASE,
                        api_key=LLM_API_KEY,
                        model=LLM_MODEL,
                        max_context_tokens=LLM_MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                    )
                    before_rule_h = enriched
                    enriched, rule_h_stats = llm.semantic_filter(
                        enriched,
                        emotion_threshold=cfg.llm_emotion_threshold,
                        concurrency=cfg.llm_concurrency,
                        override_mode=override_mode,
                        progress_callback=lambda frac, desc: self._notify(
                            progress_callback, 0.82 + frac * 0.03, desc
                        ) if progress_callback else None,
                    )
                    stats['rule_h_removed'] = rule_h_stats.get('removed_count', 0)
                    stats['rule_h_unchecked'] = rule_h_stats.get('unchecked_count', 0)
                    if override_mode:
                        stats['rule_h_overridden'] = rule_h_stats.get('overridden_count', 0)
                    stats['output_count'] = len(enriched)

                    reason_by_cid = {
                        item.get('conversation_id'): item.get('reason_code', 'other')
                        for item in rule_h_stats.get('details', [])
                        if isinstance(item, dict) and item.get('conversation_id') is not None
                    }

                    # 记录 Rule H 移除的段落。
                    # 在 LLM 接管模式下，会把带 flag 的删除样本重归因到 C/D/E/G，
                    # rule_h_removed 仅统计“无前置规则标签”的纯语义删除。
                    if override_mode:
                        stats['rule_h_removed'] = 0
                        override_removed_split: Dict[str, int] = {
                            'rule_c': 0,
                            'rule_d': 0,
                            'rule_e': 0,
                            'rule_g': 0,
                            'rule_h': 0,
                        }

                    kept_h_ids = {id(seg) for seg in enriched}
                    for seg in before_rule_h:
                        if id(seg) not in kept_h_ids:
                            cid = seg.get('conversation_id')
                            reason_code = str(reason_by_cid.get(cid, 'other'))
                            removed_by = 'rule_h'
                            if override_mode:
                                removed_by = _pick_override_removed_rule(seg)
                                override_removed_split[removed_by] += 1
                                if removed_by != 'rule_h':
                                    stats[f'{removed_by}_removed'] = stats.get(f'{removed_by}_removed', 0) + 1
                                else:
                                    stats['rule_h_removed'] += 1

                            stats['_removed_segments'].append(
                                {
                                    **seg,
                                    '_removed_by': removed_by,
                                    '_reason': _format_rule_h_reason(reason_code),
                                    '_reason_code': reason_code,
                                }
                            )

                    if override_mode:
                        logger.info(
                            "[LLM语义过滤·归因拆分] "
                            f"C={override_removed_split['rule_c']}, "
                            f"D={override_removed_split['rule_d']}, "
                            f"E={override_removed_split['rule_e']}, "
                            f"G={override_removed_split['rule_g']}, "
                            f"H(纯语义)={override_removed_split['rule_h']}"
                        )

                    t_llm = time.time() - t_prev
                    logger.info(
                        f"[LLM语义过滤耗时] {t_llm:.1f}s, "
                        f"移除 {rule_h_stats.get('removed_count', 0)} 段, "
                        f"未检查 {rule_h_stats.get('unchecked_count', 0)} 段"
                        + (f", 接管翻转 {rule_h_stats.get('overridden_count', 0)} 段" if override_mode else "")
                    )
                except Exception as e:
                    logger.warning(f"LLM 语义过滤失败，跳过规则H: {e}")
                    stats['rule_h_removed'] = 0
                    stats['rule_h_unchecked'] = len(enriched)
                t_prev = time.time()
            elif cfg.enable_llm_semantic_filter and not is_llm_configured():
                logger.warning("[规则H] LLM 语义过滤已启用但未配置 API，跳过")

            cancel_status = self._cancel_status_if_requested(stop_event, "导出前")
            if cancel_status:
                return enriched, stats, cancel_status

            # 4. JSONL 导出
            if export_jsonl_flag:
                self._notify(progress_callback, 0.85, "导出 JSONL...")
                # 使用显示名称而非临时路径
                display_name = audio_display_name or Path(audio_path).name
                file_stem = Path(display_name).stem
                jsonl_path = str(Path(output_dir) / f"{file_stem}.jsonl")
                export_jsonl(
                    segments=enriched,
                    output_path=jsonl_path,
                    audio_source=display_name,
                    metadata=metadata,
                )

            self._notify(progress_callback, 1.0, "语料库处理完成")
            return enriched, stats, "Success"

        except Exception as e:
            logger.error(f"语料库管道异常: {e}")
            traceback.print_exc()
            return [], {}, f"Corpus Pipeline Exception: {str(e)}"