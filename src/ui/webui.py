import json
import time
import os
import sys
import subprocess
import csv
import io
import threading
import queue
import logging
import hashlib
from html import escape
from collections import deque
from datetime import datetime
import gradio as gr
from pathlib import Path
from src.core.engine import FullPipelineEngine
from src.core.utils import generate_srt, format_transcript_for_display
from src.core.data_cleaner import DataCleaner, CleaningConfig
from src.core.corpus_exporter import compute_advanced_features, export_jsonl, merge_jsonl_files
from config.settings import OUTPUT_DIR, HF_TOKEN, is_llm_configured, LLM_MODEL

logger = logging.getLogger(__name__)

# 语言 → Whisper initial_prompt 映射
LANG_PROMPT_MAP = {
    "zh": "以下是普通话的句子，请添加标点符号。",
    "en": "The following is an English sentence. Please add punctuation.",
    "ja": "以下は日本語の文章です。句読点を追加してください。",
    "ko": "다음은 한국어 문장입니다. 구두점을 추가해 주세요.",
}

REMOVED_REASON_CODE_FALLBACK = {
    'rule_a': 'conf',
    'rule_b': 'overlap',
    'rule_c': 'length',
    'rule_d': 'context',
    'rule_e': 'low_info',
    'rule_f': 'snr',
    'rule_g': 'blacklist',
    'rule_h': 'other',
}

# 引擎实例化
engine = FullPipelineEngine()
# 并发保护锁：防止多个 Gradio 请求同时操作模型状态
_engine_lock = threading.Lock()
# 语料库任务停止信号（每次任务独立 Event，避免跨任务干扰）
_corpus_stop_state_lock = threading.Lock()
_active_corpus_stop_event = None
# 语料库任务防重入：避免重复点击“开始构建”导致并发任务
_corpus_task_guard_lock = threading.Lock()
_corpus_task_running = threading.Event()
# ETA 历史文件写入锁：避免并发任务覆盖历史样本
_eta_history_lock = threading.Lock()


def _set_active_corpus_stop_event(stop_event: threading.Event):
    global _active_corpus_stop_event
    with _corpus_stop_state_lock:
        _active_corpus_stop_event = stop_event


def _clear_active_corpus_stop_event(stop_event: threading.Event):
    global _active_corpus_stop_event
    with _corpus_stop_state_lock:
        if _active_corpus_stop_event is stop_event:
            _active_corpus_stop_event = None


def _signal_active_corpus_stop() -> bool:
    with _corpus_stop_state_lock:
        stop_event = _active_corpus_stop_event
    if stop_event is None:
        return False
    try:
        stop_event.set()
        return True
    except Exception:
        return False

def process_batch_task(
    file_paths, model_size, lang, 
    enable_diar, min_spk, max_spk, 
    vad_onset, initial_prompt, compute_type, enable_demucs, 
    release_memory, export_srt,
    custom_output_path,
    hallucination_mode, hallucination_threshold,
    llm_enabled, llm_mode, llm_target_lang,
    progress=gr.Progress(),
):
    """WebUI 批处理回调函数"""
    if not file_paths:
        yield "警告: 请先上传文件"
        return
    
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    # 路径解析逻辑
    try:
        if custom_output_path and custom_output_path.strip():
            save_dir = Path(custom_output_path.strip())
        else:
            save_dir = OUTPUT_DIR
        
        # 确保目标目录存在，如果不存在则自动创建
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"已自动创建输出目录: {save_dir}")
    except Exception as e:
        yield f"错误: 自定义路径无效 ({str(e)})，请检查路径格式。"
        return

    total_files = len(file_paths)
    log_buffer = ""      

    min_spk = int(min_spk) if min_spk else None
    max_spk = int(max_spk) if max_spk else None
    
    initial_prompt = initial_prompt.strip() if initial_prompt else None

    # 映射中文 Radio 标签 → 内部代码
    _hall_map = {"代码规则过滤": "code", "LLM 智能过滤": "llm", "关闭": "off"}
    hallucination_mode = _hall_map.get(hallucination_mode, "code")

    # 批处理循环
    for i, file_path in enumerate(file_paths):
        current_index = i + 1
        input_path = Path(file_path)
        file_stem = input_path.stem
        
        # 计算当前文件在总进度中的基准和范围
        file_base = i / total_files
        file_range = 1.0 / total_files
        
        def make_progress_callback(base, rng, idx, total, stem):
            """创建闭包以捕获当前循环变量"""
            def _cb(fraction, desc):
                overall = base + fraction * rng
                progress(overall, desc=f"[{idx}/{total}] {stem}: {desc}")
            return _cb
        
        progress_callback = make_progress_callback(
            file_base, file_range, current_index, total_files, file_stem
        )
        
        mode_info = "[人声分离模式]" if enable_demucs else "[标准转录模式]"
        status_msg = f"[进度: {current_index}/{total_files}] 正在处理: {file_stem} ... {mode_info} (VAD阈值: {vad_onset})"
        current_log = status_msg + "\n" + log_buffer
        yield current_log

        start_time = time.time()
        
        try:
            # 执行核心管道（加锁保护）
            with _engine_lock:
                segments, status = engine.run_pipeline(
                    audio_path=file_path, 
                    model_size=model_size, 
                    lang=lang, 
                    enable_diarization=enable_diar, 
                    min_speakers=min_spk, 
                    max_speakers=max_spk,
                    vad_onset=vad_onset,        
                    initial_prompt=initial_prompt,
                    compute_type=compute_type,
                    enable_demucs=enable_demucs,
                    hallucination_mode=hallucination_mode or "code",
                    hallucination_threshold=hallucination_threshold,
                    llm_enabled=llm_enabled,
                    llm_mode=llm_mode if llm_mode else "segmentation",
                    llm_target_lang=llm_target_lang if llm_target_lang else None,
                    progress_callback=progress_callback,
                )

            # 结构化错误检测：通过状态码前缀判断，而非字符串包含
            if status != "Success":
                raise RuntimeError(status)

            # 结果持久化
            json_path = save_dir / f"{file_stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            if export_srt:
                srt_path = save_dir / f"{file_stem}.srt"
                srt_content = generate_srt(segments)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

            # 更新 UI 日志
            duration = time.time() - start_time
            display_text = format_transcript_for_display(segments)
            
            file_log = (
                f"[完成: {current_index}/{total_files}] {file_stem} (耗时: {duration:.2f}s)\n"
                f"输出路径: {save_dir}\n"
                f"{'-'*30}\n"
                f"{display_text}\n"
                f"{'='*30}\n\n"
            )
            log_buffer = file_log + log_buffer
            yield log_buffer

        except Exception as e:
            error_log = (
                f"[失败: {current_index}/{total_files}] {file_stem}\n"
                f"原因: {str(e)}\n"
                f"{'='*30}\n\n"
            )
            log_buffer = error_log + log_buffer
            logger.error(f"Task Failed: {e}")
            yield log_buffer

    if release_memory:
        progress(0.95, desc="正在释放显存...")
        yield "正在执行资源释放...\n" + log_buffer
        with _engine_lock:
            engine.unload_all()
        progress(1.0, desc="全部完成")
        yield "所有任务执行完毕，显存已释放。\n" + log_buffer
    else:
        progress(1.0, desc="全部完成")
        yield "所有任务执行完毕。\n" + log_buffer


def _normalize_corpus_step(desc: str, model_size: str, compute_type: str) -> str:
    """将底层回调描述归一化为单行展示的步骤名。"""
    if not desc:
        return ""

    text = desc.strip()
    if not text:
        return ""

    if "转录" in text or "ASR 管道" in text:
        return f"执行转录 ({model_size} | {compute_type})..."
    if "对齐" in text:
        return "执行音素级对齐..."
    if "说话人" in text:
        return "执行说话人聚类..."
    if "数据清洗" in text:
        return "执行数据清洗..."
    if "高级特征" in text:
        return "计算高级特征..."
    if "黑名单" in text:
        return "执行黑名单过滤..."
    if "LLM 语义过滤" in text:
        return "执行 LLM 语义过滤..."
    if "信噪比" in text:
        return "计算信噪比..."
    if "导出 JSONL" in text:
        return "导出 JSONL..."
    if "预处理" in text:
        return "执行音频预处理..."
    # 仅在明确的“单文件完成”信号时才映射为 100%，
    # 避免把子流程的“处理完成/对齐完成”误判为整文件完成。
    if "完成当前音频处理" in text:
        return "完成当前音频处理"

    return text.rstrip("。")


def _format_duration_hhmmss(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS。"""
    total_seconds = max(int(seconds), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _render_corpus_progress_panel(
    total_files: int,
    current_index: int,
    current_file: str,
    current_file_progress: float,
    completed_files: int,
    current_step: str,
    elapsed_seconds: float = 0.0,
    current_eta_seconds: float = -1.0,
    eta_seconds: float = -1.0,
    success_files: int = 0,
    failed_files: int = 0,
    status_note: str = "",
) -> str:
    """渲染语料库构建右侧单一进度面板（双进度条 + 单行步骤）。"""
    safe_total = max(int(total_files), 0)
    safe_current_index = min(max(int(current_index), 0), safe_total) if safe_total > 0 else 0
    safe_completed = min(max(int(completed_files), 0), safe_total) if safe_total > 0 else 0
    safe_file_progress = min(max(float(current_file_progress), 0.0), 1.0)

    overall_units = float(safe_completed)
    if safe_total > 0 and safe_current_index > safe_completed:
        overall_units += safe_file_progress
    overall_progress = min(max((overall_units / safe_total) if safe_total > 0 else 0.0, 0.0), 1.0)

    file_bar_width = int(round(safe_file_progress * 100))
    overall_bar_width = int(round(overall_progress * 100))

    if current_step and current_step.strip():
        step_html = f"<div class='corpus-step'>{escape(current_step.strip())}</div>"
    else:
        step_html = "<div class='corpus-step-empty'>等待步骤回调...</div>"

    current_file_display = escape(current_file) if current_file else "-"
    elapsed_text = _format_duration_hhmmss(elapsed_seconds)
    current_eta_text = _format_duration_hhmmss(current_eta_seconds) if current_eta_seconds >= 0 else "计算中"
    eta_text = _format_duration_hhmmss(eta_seconds) if eta_seconds >= 0 else "计算中"
    if status_note and status_note.strip():
        status_note_html = f"<div class='corpus-alert'>{escape(status_note.strip())}</div>"
    else:
        status_note_html = ""

    return f"""
<div style="border: 1px solid #d7dbe2; border-radius: 10px; padding: 14px; background: #fbfcff;">
  <style>
    .corpus-title {{ font-size: 16px; font-weight: 600; margin-bottom: 10px; color: #1f2a44; }}
    .corpus-subtitle {{ font-size: 13px; font-weight: 600; margin: 10px 0 6px 0; color: #31466f; }}
        .corpus-step {{ font-size: 12px; line-height: 1.5; color: #3d4a63; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .corpus-step-empty {{ font-size: 12px; color: #7b8799; }}
    .corpus-track {{ width: 100%; height: 14px; background: #e6ebf2; border-radius: 999px; overflow: hidden; }}
    .corpus-fill-file {{ height: 100%; background: linear-gradient(90deg, #3b82f6, #2563eb); }}
    .corpus-fill-overall {{ height: 100%; background: linear-gradient(90deg, #22c55e, #16a34a); }}
    .corpus-meta {{ margin-top: 6px; font-size: 12px; color: #2f3d56; }}
        .corpus-status {{ margin-top: 12px; padding: 8px 10px; border-radius: 8px; background: #eef3fb; color: #2c3b56; font-size: 12px; }}
    .corpus-alert {{ margin-top: 8px; padding: 8px 10px; border-radius: 8px; background: #fff5e8; color: #9a4e00; font-size: 12px; font-weight: 600; }}
  </style>
  <div class="corpus-title">语料构建实时进度</div>

  <div class="corpus-subtitle">当前音频步骤</div>
    {step_html}
  <div style="margin-top: 8px;" class="corpus-track">
    <div class="corpus-fill-file" style="width: {file_bar_width}%;"></div>
  </div>
    <div class="corpus-meta">文件: [{current_index}/{safe_total}] {current_file_display}</div>

  <div class="corpus-subtitle" style="margin-top: 14px;">总体处理进度</div>
  <div class="corpus-track">
    <div class="corpus-fill-overall" style="width: {overall_bar_width}%;"></div>
  </div>
    <div class="corpus-meta">已完成: {safe_completed}/{safe_total} | 成功: {success_files} | 失败: {failed_files}</div>

    <div class="corpus-status">已处理时间: {elapsed_text} | 当前处理剩余: {current_eta_text} | 总体处理剩余: {eta_text}</div>
    {status_note_html}
</div>
"""


def process_corpus_task(
    file_paths, path_input,
    model_size, lang,
    min_spk, max_spk,
    vad_onset, initial_prompt, compute_type, enable_demucs,
    release_memory,
    custom_output_path,
    resume_enabled,
    audio_path_prefix,
    # 清洗规则参数
    enable_rule_a, confidence_threshold,
    enable_rule_b, overlap_min_duration,
    enable_rule_c,
    enable_rule_d, orphan_window, min_segments_d,
    enable_rule_e, low_info_duration, low_info_chars,
    enable_rule_f, snr_threshold,
    enable_rule_g, blacklist_path, blacklist_context_purge,
    enable_rule_h, llm_emotion_threshold, llm_concurrency, use_llm_override,
    # 数据集交付与可视化
    enable_materialize, materialize_output_dir, materialize_max_clips,
    enable_analysis, analysis_output_dir,
    # 图表生成选项
    chart_funnel, chart_speed, chart_delay, chart_robustness,
    chart_llm_analysis,
    progress=gr.Progress(),
):
    """语料库构建批处理回调函数（含断点保护与续跑）"""
    with _corpus_task_guard_lock:
        if _corpus_task_running.is_set():
            yield _render_corpus_progress_panel(
                total_files=0,
                current_index=0,
                current_file="",
                current_file_progress=0.0,
                completed_files=0,
                current_step="已有语料构建任务在运行，请勿重复启动。",
                elapsed_seconds=0.0,
                eta_seconds=-1.0,
                status_note="任务已在运行",
            )
            return
        _corpus_task_running.set()

    def _release_corpus_task_guard():
        with _corpus_task_guard_lock:
            _corpus_task_running.clear()

    task_stop_event = threading.Event()
    _set_active_corpus_stop_event(task_stop_event)

    AUDIO_EXTS = {
        '.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac',
        '.mp4', '.mkv', '.avi', '.mov', '.webm'
    }

    def _strip_quotes(s: str) -> str:
        s = s.strip()
        quote_pairs = [
            ('"', '"'), ("'", "'"), ('\u201c', '\u201d'), ('\u2018', '\u2019'),
            ('\u300c', '\u300d'), ('\uff02', '\uff02')
        ]
        for lq, rq in quote_pairs:
            if s.startswith(lq) and s.endswith(rq) and len(s) >= 2:
                s = s[len(lq):-len(rq)].strip()
        return s

    def _build_audio_display_name(file_path: str, original_path: str) -> str:
        input_path = Path(file_path)
        if original_path:
            return original_path
        if audio_path_prefix and str(audio_path_prefix).strip():
            prefix_clean = _strip_quotes(str(audio_path_prefix))
            prefix_path = Path(prefix_clean)
            if prefix_path.suffix.lower() in AUDIO_EXTS:
                prefix_path = prefix_path.parent
            return str(prefix_path / input_path.name)
        return input_path.name

    def _atomic_write_json(path: Path, payload: dict):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _safe_read_json(path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _load_removed_segments(path: Path):
        rows = []
        if not path.exists():
            return rows
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
                    except Exception:
                        continue
        except Exception:
            return []
        return rows

    def _probe_media_duration_sec(file_path: str) -> float:
        """使用 ffprobe 获取媒体时长（秒），失败时返回 0。"""
        if not file_path:
            return 0.0
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                return 0.0
            lines = (result.stdout or "").strip().splitlines()
            if not lines:
                return 0.0
            duration = float(lines[0].strip())
            return duration if duration > 0 else 0.0
        except Exception:
            return 0.0

    # 收集待处理文件
    all_files = []
    if file_paths:
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        for fp in file_paths:
            all_files.append((fp, None))

    if path_input and path_input.strip():
        for raw_line in path_input.strip().splitlines():
            line = _strip_quotes(raw_line)
            if not line:
                continue
            p = Path(line)
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                all_files.append((str(p), str(p)))
            elif p.is_dir():
                for f in sorted(p.rglob('*')):
                    if f.suffix.lower() in AUDIO_EXTS:
                        all_files.append((str(f), str(f)))
            else:
                logger.warning(f"[路径跳过] 无效或不支持: {line}")

    if not all_files:
        yield _render_corpus_progress_panel(
            total_files=0,
            current_index=0,
            current_file="",
            current_file_progress=0.0,
            completed_files=0,
            current_step="警告: 未找到可处理的音视频文件，请检查路径与格式。",
            elapsed_seconds=0.0,
            eta_seconds=-1.0,
        )
        _release_corpus_task_guard()
        return

    file_states = []
    for i, (file_path, original_path) in enumerate(all_files):
        input_path = Path(file_path)
        audio_duration_sec = _probe_media_duration_sec(str(file_path))
        file_states.append({
            'idx': i + 1,
            'source_path': str(file_path),
            'original_path': str(original_path) if original_path else "",
            'file_name': input_path.name,
            'file_stem': input_path.stem,
            'audio_display_name': _build_audio_display_name(str(file_path), str(original_path) if original_path else ""),
            'audio_duration_sec': float(audio_duration_sec),
            'status': 'pending',
            'attempts': 0,
            'error': '',
        })

    input_signature = hashlib.sha256(
        "\n".join([f"{s['idx']}|{s['audio_display_name']}" for s in file_states]).encode("utf-8")
    ).hexdigest()

    # 构建清洗配置
    cleaning_config = CleaningConfig(
        enable_confidence_filter=enable_rule_a,
        confidence_threshold=confidence_threshold,
        enable_overlap_filter=enable_rule_b,
        overlap_min_duration=float(overlap_min_duration),
        enable_length_ratio_filter=enable_rule_c,
        enable_context_island_filter=enable_rule_d,
        orphan_window_sec=float(orphan_window),
        min_segments_to_apply_d=int(min_segments_d),
        enable_low_info_filter=enable_rule_e,
        low_info_max_duration=low_info_duration,
        low_info_max_chars=int(low_info_chars),
        enable_snr_filter=enable_rule_f,
        min_snr_db=float(snr_threshold),
        enable_blacklist_filter=enable_rule_g,
        blacklist_path=blacklist_path.strip() if blacklist_path else "",
        blacklist_context_purge=blacklist_context_purge,
        enable_llm_semantic_filter=enable_rule_h,
        llm_emotion_threshold=int(llm_emotion_threshold),
        llm_concurrency=int(llm_concurrency),
        use_llm_override_mode=bool(use_llm_override),
    )

    min_spk = int(min_spk) if min_spk else None
    max_spk = int(max_spk) if max_spk else None
    initial_prompt = initial_prompt.strip() if initial_prompt else None

    try:
        if custom_output_path and custom_output_path.strip():
            user_path = Path(custom_output_path.strip())
        else:
            user_path = OUTPUT_DIR / "corpus_work"
    except Exception as e:
        yield _render_corpus_progress_panel(
            total_files=0,
            current_index=0,
            current_file="",
            current_file_progress=0.0,
            completed_files=0,
            current_step=f"错误: 输出路径无效 ({str(e)})",
            elapsed_seconds=0.0,
            eta_seconds=-1.0,
        )
        _release_corpus_task_guard()
        return

    ETA_HISTORY_MAX_SAMPLES = 20
    eta_history_root = user_path.parent if user_path.name.startswith("run_") else user_path
    eta_history_path = eta_history_root / "eta_history.txt"
    eta_profile_key = f"model={model_size}|compute={compute_type}|demucs={'on' if enable_demucs else 'off'}"

    def _load_eta_history_samples():
        with _eta_history_lock:
            payload = _safe_read_json(eta_history_path)
        if not isinstance(payload, dict):
            return [], []
        profiles = payload.get("profiles", {})
        if not isinstance(profiles, dict):
            return [], []
        entry = profiles.get(eta_profile_key, {})
        if not isinstance(entry, dict):
            return [], []

        proc_samples = []
        rtf_samples = []

        raw_structured = entry.get("samples", [])
        if isinstance(raw_structured, list):
            for item in raw_structured:
                if not isinstance(item, dict):
                    continue
                try:
                    proc_sec = float(item.get("proc_sec", 0.0) or 0.0)
                    audio_sec = float(item.get("audio_sec", 0.0) or 0.0)
                except Exception:
                    continue
                if proc_sec > 0:
                    proc_samples.append(proc_sec)
                if proc_sec > 0 and audio_sec > 0:
                    rtf_samples.append(proc_sec / audio_sec)

        # 兼容旧格式：samples_sec 仅包含处理时长
        raw_legacy = entry.get("samples_sec", [])
        if isinstance(raw_legacy, list):
            for v in raw_legacy:
                try:
                    fv = float(v)
                    if fv > 0:
                        proc_samples.append(fv)
                except Exception:
                    continue

        return proc_samples[-ETA_HISTORY_MAX_SAMPLES:], rtf_samples[-ETA_HISTORY_MAX_SAMPLES:]

    def _append_eta_history_sample(proc_sec: float, audio_sec: float):
        if proc_sec <= 0 or audio_sec <= 0:
            return
        try:
            with _eta_history_lock:
                payload = _safe_read_json(eta_history_path)
                if not isinstance(payload, dict):
                    payload = {"version": 1, "profiles": {}}

                profiles = payload.get("profiles", {})
                if not isinstance(profiles, dict):
                    profiles = {}

                entry = profiles.get(eta_profile_key, {})
                if not isinstance(entry, dict):
                    entry = {}

                raw_structured = entry.get("samples", [])
                if not isinstance(raw_structured, list):
                    raw_structured = []

                cleaned_structured = []
                for item in raw_structured:
                    if not isinstance(item, dict):
                        continue
                    try:
                        old_proc = float(item.get("proc_sec", 0.0) or 0.0)
                        old_audio = float(item.get("audio_sec", 0.0) or 0.0)
                    except Exception:
                        continue
                    if old_proc > 0 and old_audio > 0:
                        cleaned_structured.append({
                            "proc_sec": old_proc,
                            "audio_sec": old_audio,
                            "rtf": old_proc / old_audio,
                        })

                cleaned_structured.append({
                    "proc_sec": float(proc_sec),
                    "audio_sec": float(audio_sec),
                    "rtf": float(proc_sec) / float(audio_sec),
                })
                cleaned_structured = cleaned_structured[-ETA_HISTORY_MAX_SAMPLES:]

                entry.update({
                    "model_size": model_size,
                    "compute_type": compute_type,
                    "enable_demucs": bool(enable_demucs),
                    "samples": cleaned_structured,
                    # 保留旧字段，便于向后兼容
                    "samples_sec": [x["proc_sec"] for x in cleaned_structured],
                })
                profiles[eta_profile_key] = entry

                payload["version"] = 1
                payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
                payload["max_samples"] = ETA_HISTORY_MAX_SAMPLES
                payload["profiles"] = profiles

                eta_history_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_json(eta_history_path, payload)
        except Exception as e:
            logger.warning(f"写入 eta_history.txt 失败: {e}")

    def _find_resumable_run(search_root: Path):
        if search_root.name.startswith("run_"):
            candidates = [search_root]
        else:
            try:
                candidates = sorted(
                    [d for d in search_root.glob("run_*") if d.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except Exception:
                candidates = []

        for candidate in candidates:
            cp = candidate / "progress_snapshot.json"
            snap = _safe_read_json(cp)
            if not snap:
                continue
            if str(snap.get("status", "")).lower() == "completed":
                continue
            if snap.get("input_signature") != input_signature:
                continue
            if int(snap.get("total_files", -1)) != len(file_states):
                continue
            return candidate, snap
        return None, None

    resume_mode = False
    resume_snapshot = None
    if bool(resume_enabled):
        candidate_dir, candidate_snap = _find_resumable_run(user_path)
        if candidate_dir and candidate_snap:
            save_dir = candidate_dir
            resume_snapshot = candidate_snap
            resume_mode = True
        else:
            base_dir = user_path.parent if user_path.name.startswith("run_") else user_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = base_dir / f"run_{timestamp}"
    else:
        base_dir = user_path.parent if user_path.name.startswith("run_") else user_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = base_dir / f"run_{timestamp}"

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        yield _render_corpus_progress_panel(
            total_files=0,
            current_index=0,
            current_file="",
            current_file_progress=0.0,
            completed_files=0,
            current_step=f"错误: 输出目录不可用 ({str(e)})",
            elapsed_seconds=0.0,
            eta_seconds=-1.0,
        )
        _release_corpus_task_guard()
        return

    quality_log_path = save_dir / "日志.txt"
    terminal_log_path = save_dir / "terminal.log"
    quality_stats_path = save_dir / "quality_stats.json"
    removed_all_path = save_dir / "removed_segments.jsonl"
    checkpoint_path = save_dir / "progress_snapshot.json"

    # 状态容器
    total_files = len(file_states)
    task_started_at = time.time()
    recent_file_durations = deque(maxlen=ETA_HISTORY_MAX_SAMPLES)
    recent_rtf_values = deque(maxlen=ETA_HISTORY_MAX_SAMPLES)
    _loaded_proc_samples, _loaded_rtf_samples = _load_eta_history_samples()
    for s in _loaded_proc_samples:
        recent_file_durations.append(s)
    for r in _loaded_rtf_samples:
        recent_rtf_values.append(r)
    log_buffer = ""
    jsonl_files = []
    total_stats = {
        'total_input': 0, 'total_output': 0,
        'rule_a': 0, 'rule_b': 0, 'rule_c': 0,
        'rule_d': 0, 'rule_e': 0, 'rule_f': 0, 'rule_g': 0,
        'rule_h': 0, 'rule_h_unchecked': 0, 'rule_h_overridden': 0,
        'rule_c_flagged': 0, 'rule_d_flagged': 0,
        'rule_e_flagged': 0, 'rule_g_flagged': 0,
        'failed_files': 0, 'success_files': 0,
    }
    per_file_stats_map = {}
    all_removed_segments = []
    stop_requested = False
    checkpoint_created_at = datetime.now().isoformat(timespec='seconds')

    if resume_mode and isinstance(resume_snapshot, dict):
        checkpoint_created_at = str(
            resume_snapshot.get("created_at", datetime.now().isoformat(timespec='seconds'))
        )
        snap_states = resume_snapshot.get("file_states", [])
        if isinstance(snap_states, list):
            snap_map = {}
            for item in snap_states:
                if isinstance(item, dict):
                    snap_map[item.get('idx')] = item
            for state in file_states:
                old = snap_map.get(state['idx'])
                if not old:
                    continue
                state['status'] = str(old.get('status', 'pending'))
                state['attempts'] = int(old.get('attempts', 0) or 0)
                state['error'] = str(old.get('error', ''))

        snap_total_stats = resume_snapshot.get("total_stats")
        if isinstance(snap_total_stats, dict):
            for key in total_stats:
                try:
                    total_stats[key] = int(snap_total_stats.get(key, total_stats[key]))
                except Exception:
                    pass

        snap_jsonl = resume_snapshot.get("jsonl_files", [])
        if isinstance(snap_jsonl, list):
            jsonl_files = [str(p) for p in snap_jsonl]

        snap_log = resume_snapshot.get("log_buffer", "")
        if isinstance(snap_log, str):
            log_buffer = snap_log

        snap_per_file = resume_snapshot.get("per_file", [])
        if isinstance(snap_per_file, list):
            for row in snap_per_file:
                if isinstance(row, dict):
                    try:
                        idx = int(row.get('idx'))
                        per_file_stats_map[idx] = row
                    except Exception:
                        continue

        all_removed_segments = _load_removed_segments(removed_all_path)

    def _upsert_per_file_stat(row: dict):
        try:
            idx = int(row.get('idx'))
        except Exception:
            return
        per_file_stats_map[idx] = row

    def _sorted_per_file_stats():
        return [per_file_stats_map[k] for k in sorted(per_file_stats_map.keys())]

    def _count_completed_files() -> int:
        return sum(1 for s in file_states if s.get('status') in ('success', 'failed'))

    def _sync_outcome_counts():
        total_stats['success_files'] = sum(1 for s in file_states if s.get('status') == 'success')
        total_stats['failed_files'] = sum(1 for s in file_states if s.get('status') == 'failed')

    def _append_removed_segments(new_segments: list, source_file: str):
        if not new_segments:
            return
        try:
            with open(removed_all_path, "a", encoding="utf-8") as f:
                for seg in new_segments:
                    removed_by = seg.get('_removed_by', 'unknown')
                    reason_code = seg.get('_reason_code', '')
                    if not reason_code:
                        reason_code = REMOVED_REASON_CODE_FALLBACK.get(removed_by, 'other')
                    item = {
                        'text': seg.get('text', ''),
                        'speaker_id': seg.get('speaker', ''),
                        'start': seg.get('start', 0.0),
                        'end': seg.get('end', 0.0),
                        'duration': seg.get('end', 0.0) - seg.get('start', 0.0),
                        'removed_by': removed_by,
                        'reason': seg.get('_reason', ''),
                        'reason_code': reason_code,
                        'source_file': source_file,
                    }
                    all_removed_segments.append(item)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"写入 removed_segments.jsonl 失败: {e}")

    def _write_checkpoint(run_status: str, current_index: int = 0, current_file: str = "", current_step: str = ""):
        _sync_outcome_counts()
        payload = {
            'version': 1,
            'created_at': checkpoint_created_at,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'status': run_status,
            'run_dir': str(save_dir),
            'input_signature': input_signature,
            'total_files': total_files,
            'completed_files': _count_completed_files(),
            'current_index': int(current_index),
            'current_file': str(current_file),
            'current_step': str(current_step),
            'total_stats': total_stats,
            'per_file': _sorted_per_file_stats(),
            'jsonl_files': sorted(list(set(jsonl_files))),
            'log_buffer': log_buffer,
            'file_states': file_states,
        }
        try:
            _atomic_write_json(checkpoint_path, payload)
        except Exception as e:
            logger.warning(f"写入 progress_snapshot.json 失败: {e}")

    terminal_handler = None
    root_logger = logging.getLogger()
    try:
        terminal_handler = logging.FileHandler(
            terminal_log_path,
            mode="a" if resume_mode else "w",
            encoding="utf-8",
        )
        terminal_handler.setLevel(logging.INFO)
        terminal_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root_logger.addHandler(terminal_handler)
        logger.info("[TerminalLog] 语料构建开始 | 输出目录: %s | 续跑: %s", save_dir, resume_mode)
    except Exception as e:
        logger.warning(f"初始化 terminal.log 失败: {e}")

    for state in file_states:
        if state.get('status') == 'success':
            p = save_dir / f"{state.get('file_stem', '')}.jsonl"
            if p.exists():
                jsonl_files.append(str(p))

    jsonl_files = sorted(list(set(jsonl_files)))
    _sync_outcome_counts()

    def _compute_elapsed_eta(done_count: int, current_progress: float, include_current: bool = True):
        elapsed = max(time.time() - task_started_at, 0.0)
        current_unit = 0.0
        if include_current and done_count < total_files:
            current_unit = min(max(float(current_progress), 0.0), 1.0)
        remain_ratio = max(1.0 - current_unit, 0.0) if include_current else 0.0

        current_eta = -1.0

        if include_current:
            running_audio_sec = 0.0
            for s in file_states:
                if str(s.get('status', 'pending')) != 'running':
                    continue
                try:
                    running_audio_sec = float(s.get('audio_duration_sec', 0.0) or 0.0)
                except Exception:
                    running_audio_sec = 0.0
                break

            if running_audio_sec > 0 and len(recent_rtf_values) > 0:
                avg_rtf = sum(recent_rtf_values) / len(recent_rtf_values)
                current_eta = avg_rtf * running_audio_sec * remain_ratio
            elif len(recent_file_durations) > 0:
                avg_file_sec = sum(recent_file_durations) / len(recent_file_durations)
                current_eta = avg_file_sec * remain_ratio

            if current_unit >= 1.0:
                current_eta = 0.0
        elif done_count >= total_files:
            current_eta = 0.0

        # 优先使用 RTF（处理时长/音频时长）与剩余音频时长估算 ETA
        if len(recent_rtf_values) > 0:
            remaining_audio_sec = 0.0
            has_audio_duration = False
            for s in file_states:
                try:
                    audio_sec = float(s.get('audio_duration_sec', 0.0) or 0.0)
                except Exception:
                    audio_sec = 0.0
                if audio_sec <= 0:
                    continue

                has_audio_duration = True
                status = str(s.get('status', 'pending'))
                if status in ('success', 'failed', 'stopped'):
                    continue
                if status == 'running':
                    remaining_audio_sec += audio_sec * max(remain_ratio, 0.0)
                else:
                    remaining_audio_sec += audio_sec

            if has_audio_duration:
                avg_rtf = sum(recent_rtf_values) / len(recent_rtf_values)
                eta = avg_rtf * max(remaining_audio_sec, 0.0)
                return elapsed, (max(current_eta, 0.0) if current_eta >= 0 else -1.0), max(eta, 0.0)

        if total_files > 0 and len(recent_file_durations) > 0:
            remain_units = max(float(total_files) - float(done_count) - current_unit, 0.0)
            avg_file_sec = sum(recent_file_durations) / len(recent_file_durations)
            eta = avg_file_sec * remain_units
            if current_eta < 0 and include_current:
                current_eta = avg_file_sec * remain_ratio
            return elapsed, (max(current_eta, 0.0) if current_eta >= 0 else -1.0), max(eta, 0.0)

        units = float(max(done_count, 0))
        if include_current and done_count < total_files:
            units += current_unit
        if total_files <= 0 or units <= 0:
            return elapsed, (max(current_eta, 0.0) if current_eta >= 0 else -1.0), -1.0
        remain = max(float(total_files) - units, 0.0)
        eta = (elapsed / units) * remain
        if current_eta < 0 and include_current:
            current_eta = (elapsed / units) * remain_ratio
        return elapsed, (max(current_eta, 0.0) if current_eta >= 0 else -1.0), max(eta, 0.0)

    def _render_live_panel(
        current_index: int,
        current_file: str,
        current_file_progress: float,
        completed_files_now: int,
        current_step: str,
        include_current: bool = True,
        status_note: str = "",
    ):
        elapsed_seconds, current_eta_seconds, eta_seconds = _compute_elapsed_eta(
            done_count=completed_files_now,
            current_progress=current_file_progress,
            include_current=include_current,
        )
        return _render_corpus_progress_panel(
            total_files=total_files,
            current_index=current_index,
            current_file=current_file,
            current_file_progress=current_file_progress,
            completed_files=completed_files_now,
            current_step=current_step,
            elapsed_seconds=elapsed_seconds,
            current_eta_seconds=current_eta_seconds,
            eta_seconds=eta_seconds,
            success_files=total_stats['success_files'],
            failed_files=total_stats['failed_files'],
            status_note=status_note,
        )

    def _step_rank(step_text: str) -> int:
        text = (step_text or "").strip()
        if not text:
            return 0
        if "预处理" in text:
            return 1
        if "加载 ASR 模型" in text:
            return 2
        if "转录" in text:
            return 3
        if "对齐" in text:
            return 4
        if "说话人" in text:
            return 5
        if "数据清洗" in text:
            return 6
        if "高级特征" in text:
            return 7
        if "黑名单" in text:
            return 8
        if "LLM 语义过滤" in text:
            return 9
        if "导出 JSONL" in text:
            return 10
        if "完成当前音频处理" in text or "构建完成" in text:
            return 11
        return 0

    def _step_progress_floor(step_text: str) -> float:
        text = (step_text or "").strip()
        if not text:
            return 0.0
        if "预处理" in text:
            return 0.05
        if "加载 ASR 模型" in text:
            return 0.08
        if "转录" in text:
            return 0.12
        if "对齐" in text:
            return 0.27
        if "说话人" in text:
            return 0.51
        if "数据清洗" in text:
            return 0.60
        if "高级特征" in text:
            return 0.75
        if "黑名单" in text:
            return 0.80
        if "LLM 语义过滤" in text:
            return 0.82
        if "导出 JSONL" in text:
            return 0.85
        if "完成当前音频处理" in text or "构建完成" in text:
            return 1.0
        return 0.0

    def _promote_step(current_step_text: str, candidate_step_text: str) -> str:
        candidate = (candidate_step_text or "").strip()
        if not candidate:
            return current_step_text
        current = (current_step_text or "").strip()
        if not current:
            return candidate

        cur_rank = _step_rank(current)
        cand_rank = _step_rank(candidate)

        # 已终止状态不再被普通阶段覆盖
        if "用户已停止" in current and "用户已停止" not in candidate:
            return current

        # 候选步骤无法识别时，仅在当前步骤也无法识别时才覆盖
        if cand_rank == 0:
            return candidate if cur_rank == 0 else current

        # 识别到阶段时，按单调推进更新，避免旧阶段回写覆盖新阶段
        return candidate if cand_rank >= cur_rank else current

    step_bridge = {"step": ""}

    class _StepSyncLogHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return
            normalized = _normalize_corpus_step(str(msg), model_size, compute_type)
            if normalized:
                step_bridge["step"] = normalized

    step_sync_handler = _StepSyncLogHandler(level=logging.INFO)
    try:
        root_logger.addHandler(step_sync_handler)
    except Exception:
        step_sync_handler = None

    try:
        _write_checkpoint("running", 0, "", "准备开始")

        initial_note = "断点续跑已启用" if bool(resume_enabled) else ""
        initial_step = "从断点恢复并继续处理..." if resume_mode else "准备开始语料库构建..."
        yield _render_live_panel(0, "", 0.0, _count_completed_files(), initial_step, include_current=False, status_note=initial_note)

        resume_anchor_index = 0
        resume_anchor_step = ""
        if resume_mode and isinstance(resume_snapshot, dict):
            try:
                resume_anchor_index = int(resume_snapshot.get('current_index', 0) or 0)
            except Exception:
                resume_anchor_index = 0
            resume_anchor_step = str(resume_snapshot.get('current_step', '') or '').strip()

        for state in file_states:
            current_index = int(state['idx'])
            file_stem = str(state.get('file_stem', ''))
            file_path = str(state.get('source_path', ''))
            original_path = str(state.get('original_path', ''))
            audio_display_name = str(state.get('audio_display_name', file_stem))

            if state.get('status') == 'success':
                continue

            if task_stop_event.is_set():
                stop_requested = True
                break

            if not file_path or (not Path(file_path).exists()):
                err = f"输入文件不存在: {file_path}"
                state['status'] = 'failed'
                state['error'] = err
                _upsert_per_file_stat({
                    'idx': current_index,
                    'file_name': state.get('file_name', file_stem),
                    'file_stem': file_stem,
                    'status': 'failed',
                    'duration_sec': 0.0,
                    'input': 0,
                    'kept': 0,
                    'removed': 0,
                    'error': err,
                })
                _write_checkpoint("running", current_index, file_stem, err)
                yield _render_live_panel(
                    current_index,
                    file_stem,
                    0.0,
                    _count_completed_files(),
                    f"处理失败: {err}",
                    include_current=False,
                )
                continue

            resume_step_hint = ""
            if (
                resume_mode
                and current_index == resume_anchor_index
                and resume_anchor_step
                and resume_anchor_step not in ("处理中", "准备开始", "用户已停止", "语料库构建完成")
            ):
                resume_step_hint = resume_anchor_step

            current_step = resume_step_hint if resume_step_hint else "等待任务开始..."
            current_file_progress = 0.0
            current_file_progress = max(current_file_progress, _step_progress_floor(current_step))
            update_queue = queue.Queue()

            state['status'] = 'running'
            state['attempts'] = int(state.get('attempts', 0)) + 1
            state['error'] = ''
            mode_info = "[人声分离]" if enable_demucs else "[标准]"
            status_msg = f"正在处理: {file_stem} {mode_info}"
            if not resume_step_hint:
                current_step = status_msg
            _write_checkpoint("running", current_index, file_stem, current_step)

            def progress_callback(fraction, desc):
                nonlocal current_file_progress, current_step
                frac = min(max(float(fraction), 0.0), 1.0)
                current_file_progress = max(current_file_progress, frac)

                normalized_step = _normalize_corpus_step(desc, model_size, compute_type)
                candidate_step = ""
                if normalized_step:
                    candidate_step = normalized_step
                elif desc:
                    candidate_step = str(desc).strip()

                if candidate_step:
                    current_step = _promote_step(current_step, candidate_step)

                current_file_progress = max(current_file_progress, _step_progress_floor(current_step))

                update_queue.put((current_file_progress, current_step))

            logger.info(status_msg)
            yield _render_live_panel(
                current_index,
                file_stem,
                current_file_progress,
                _count_completed_files(),
                current_step,
                include_current=True,
            )

            start_time = time.time()
            stop_notice_sent = False

            try:
                worker_result = {}
                worker_error = {}

                def _worker_run():
                    try:
                        with _engine_lock:
                            worker_result['value'] = engine.run_corpus_pipeline(
                                audio_path=file_path,
                                output_dir=str(save_dir),
                                cleaning_config=cleaning_config,
                                model_size=model_size,
                                lang=lang,
                                vad_onset=vad_onset,
                                initial_prompt=initial_prompt,
                                compute_type=compute_type,
                                enable_demucs=enable_demucs,
                                min_speakers=min_spk,
                                max_speakers=max_spk,
                                export_jsonl_flag=True,
                                metadata={'source_file': state.get('file_name', file_stem)},
                                audio_display_name=audio_display_name,
                                progress_callback=progress_callback,
                                stop_event=task_stop_event,
                            )
                    except Exception as worker_exc:
                        worker_error['value'] = worker_exc

                worker = threading.Thread(target=_worker_run, daemon=True)
                worker.start()
                last_panel_refresh = time.time()

                while worker.is_alive():
                    if task_stop_event.is_set() and not stop_notice_sent:
                        stop_requested = True
                        stop_notice_sent = True
                        current_step = "用户已停止，正在安全终止当前任务..."
                        yield _render_live_panel(
                            current_index,
                            file_stem,
                            current_file_progress,
                            _count_completed_files(),
                            current_step,
                            include_current=True,
                            status_note="用户已停止",
                        )

                    updated = False
                    while True:
                        try:
                            current_file_progress, current_step = update_queue.get_nowait()
                            updated = True
                        except queue.Empty:
                            break

                    bridged_step = step_bridge.get("step", "")
                    promoted_step = _promote_step(current_step, bridged_step)
                    if promoted_step != current_step:
                        current_step = promoted_step
                        updated = True

                    floored_progress = max(current_file_progress, _step_progress_floor(current_step))
                    if floored_progress > current_file_progress:
                        current_file_progress = floored_progress
                        updated = True

                    now_ts = time.time()
                    if updated or (now_ts - last_panel_refresh >= 1.0):
                        _write_checkpoint("running", current_index, file_stem, current_step)
                        yield _render_live_panel(
                            current_index,
                            file_stem,
                            current_file_progress,
                            _count_completed_files(),
                            current_step,
                            include_current=True,
                        )
                        last_panel_refresh = now_ts
                    worker.join(timeout=0.15)

                while True:
                    try:
                        current_file_progress, current_step = update_queue.get_nowait()
                    except queue.Empty:
                        break

                if 'value' in worker_error:
                    raise worker_error['value']

                enriched, stats, status = worker_result['value']

                if status.startswith("Cancelled"):
                    stop_requested = True
                    state['status'] = 'stopped'
                    state['error'] = status
                    duration = time.time() - start_time
                    _upsert_per_file_stat({
                        'idx': current_index,
                        'file_name': state.get('file_name', file_stem),
                        'file_stem': file_stem,
                        'status': 'stopped',
                        'duration_sec': round(duration, 3),
                        'input': int(stats.get('input_count', 0)),
                        'kept': int(stats.get('output_count', 0)),
                        'removed': int(stats.get('input_count', 0) - stats.get('output_count', 0)),
                        'error': status,
                    })
                    _write_checkpoint("stopped", current_index, file_stem, "用户已停止")
                    yield _render_live_panel(
                        current_index,
                        file_stem,
                        current_file_progress,
                        _count_completed_files(),
                        "用户已停止",
                        include_current=False,
                        status_note="用户已停止",
                    )
                    break

                if "Success" not in status:
                    raise RuntimeError(status)

                total_stats['total_input'] += stats.get('input_count', 0)
                total_stats['total_output'] += stats.get('output_count', 0)
                total_stats['rule_a'] += stats.get('rule_a_removed', 0)
                total_stats['rule_b'] += stats.get('rule_b_removed', 0)
                total_stats['rule_c'] += stats.get('rule_c_removed', 0)
                total_stats['rule_d'] += stats.get('rule_d_removed', 0)
                total_stats['rule_e'] += stats.get('rule_e_removed', 0)
                total_stats['rule_f'] += stats.get('rule_f_removed', 0)
                total_stats['rule_g'] += stats.get('rule_g_removed', 0)
                total_stats['rule_h'] += stats.get('rule_h_removed', 0)
                total_stats['rule_h_unchecked'] += stats.get('rule_h_unchecked', 0)
                total_stats['rule_h_overridden'] += stats.get('rule_h_overridden', 0)
                total_stats['rule_c_flagged'] += stats.get('rule_c_flagged', 0)
                total_stats['rule_d_flagged'] += stats.get('rule_d_flagged', 0)
                total_stats['rule_e_flagged'] += stats.get('rule_e_flagged', 0)
                total_stats['rule_g_flagged'] += stats.get('rule_g_flagged', 0)

                _append_removed_segments(
                    stats.get('_removed_segments', []),
                    str(state.get('file_name', file_stem)),
                )

                jsonl_path = save_dir / f"{file_stem}.jsonl"
                if jsonl_path.exists():
                    jsonl_files.append(str(jsonl_path))
                    jsonl_files = sorted(list(set(jsonl_files)))

                duration = time.time() - start_time
                removed = stats.get('input_count', 0) - stats.get('output_count', 0)
                _upsert_per_file_stat({
                    'idx': current_index,
                    'file_name': state.get('file_name', file_stem),
                    'file_stem': file_stem,
                    'status': 'success',
                    'duration_sec': round(duration, 3),
                    'input': int(stats.get('input_count', 0)),
                    'kept': int(stats.get('output_count', 0)),
                    'removed': int(removed),
                })
                state['status'] = 'success'
                state['error'] = ''

                file_log = (
                    f"✓ [{current_index}/{total_files}] {file_stem}  ({duration:.1f}s)\n"
                    f"  保留 {stats.get('output_count',0)} 段 / 移除 {removed} 段\n"
                )
                logger.info(file_log.strip())
                log_buffer = file_log + log_buffer

                sample_sec = max(duration, 0.0)
                recent_file_durations.append(sample_sec)
                audio_sec = float(state.get('audio_duration_sec', 0.0) or 0.0)
                if sample_sec > 0 and audio_sec > 0:
                    recent_rtf_values.append(sample_sec / audio_sec)
                    _append_eta_history_sample(sample_sec, audio_sec)
                _write_checkpoint("running", current_index, file_stem, "完成当前音频处理")
                yield _render_live_panel(
                    current_index,
                    file_stem,
                    1.0,
                    _count_completed_files(),
                    "完成当前音频处理",
                    include_current=False,
                )

                if task_stop_event.is_set():
                    stop_requested = True
                    _write_checkpoint("stopped", current_index, file_stem, "用户已停止")
                    break

            except Exception as e:
                if task_stop_event.is_set() or "Cancelled" in str(e):
                    stop_requested = True
                    duration = time.time() - start_time
                    state['status'] = 'stopped'
                    state['error'] = str(e)
                    _upsert_per_file_stat({
                        'idx': current_index,
                        'file_name': state.get('file_name', file_stem),
                        'file_stem': file_stem,
                        'status': 'stopped',
                        'duration_sec': round(duration, 3),
                        'input': 0,
                        'kept': 0,
                        'removed': 0,
                        'error': str(e),
                    })
                    _write_checkpoint("stopped", current_index, file_stem, "用户已停止")
                    yield _render_live_panel(
                        current_index,
                        file_stem,
                        current_file_progress,
                        _count_completed_files(),
                        "用户已停止",
                        include_current=False,
                        status_note="用户已停止",
                    )
                    break

                duration = time.time() - start_time
                state['status'] = 'failed'
                state['error'] = str(e)
                _upsert_per_file_stat({
                    'idx': current_index,
                    'file_name': state.get('file_name', file_stem),
                    'file_stem': file_stem,
                    'status': 'failed',
                    'duration_sec': round(duration, 3),
                    'input': 0,
                    'kept': 0,
                    'removed': 0,
                    'error': str(e),
                })

                error_log = (
                    f"✗ [{current_index}/{total_files}] {file_stem}\n"
                    f"  失败原因: {str(e)}\n"
                )
                log_buffer = error_log + log_buffer
                logger.error(f"Corpus Task Failed: {e}")
                _write_checkpoint("running", current_index, file_stem, f"处理失败: {str(e)}")
                yield _render_live_panel(
                    current_index,
                    file_stem,
                    current_file_progress,
                    _count_completed_files(),
                    f"处理失败: {str(e)}",
                    include_current=False,
                )

        _sync_outcome_counts()

        final_dataset = None
        if (not stop_requested) and len(jsonl_files) >= 2:
            merged_path = save_dir / "dataset_merged.jsonl"
            merge_jsonl_files(jsonl_files, str(merged_path))
            final_dataset = "dataset_merged.jsonl"
        elif len(jsonl_files) == 1:
            final_dataset = Path(jsonl_files[0]).name

        stats_report = _format_corpus_stats(total_stats, total_files, str(save_dir), final_dataset)

        total_removed = total_stats['total_input'] - total_stats['total_output']
        keep_rate = (total_stats['total_output'] / total_stats['total_input']) if total_stats['total_input'] > 0 else 0.0
        quality_stats_payload = {
            'version': 1,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'summary': {
                'raw': int(total_stats['total_input']),
                'kept': int(total_stats['total_output']),
                'removed': int(total_removed),
                'keep_rate': round(keep_rate, 6),
                'success_files': int(total_stats['success_files']),
                'failed_files': int(total_stats['failed_files']),
                'total_files': int(total_files),
                'rule_a': int(total_stats['rule_a']),
                'rule_b': int(total_stats['rule_b']),
                'rule_c': int(total_stats['rule_c']),
                'rule_d': int(total_stats['rule_d']),
                'rule_e': int(total_stats['rule_e']),
                'rule_f': int(total_stats['rule_f']),
                'rule_g': int(total_stats['rule_g']),
                'rule_h': int(total_stats['rule_h']),
                'rule_h_unchecked': int(total_stats['rule_h_unchecked']),
                'rule_h_overridden': int(total_stats['rule_h_overridden']),
                'rule_c_flagged': int(total_stats['rule_c_flagged']),
                'rule_d_flagged': int(total_stats['rule_d_flagged']),
                'rule_e_flagged': int(total_stats['rule_e_flagged']),
                'rule_g_flagged': int(total_stats['rule_g_flagged']),
            },
            'per_file': _sorted_per_file_stats(),
        }
        try:
            with open(quality_stats_path, "w", encoding="utf-8") as f:
                json.dump(quality_stats_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写入 quality_stats.json 失败: {e}")

        if all_removed_segments:
            try:
                with open(removed_all_path, "w", encoding="utf-8") as f:
                    for item in all_removed_segments:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"导出清洗段落汇总失败: {e}")

        post_logs = []
        dataset_jsonl_path = (save_dir / final_dataset) if final_dataset else None
        if (not stop_requested) and enable_materialize and dataset_jsonl_path and dataset_jsonl_path.exists():
            try:
                materialize_script = Path(__file__).resolve().parents[2] / "scripts" / "materialize_corpus.py"
                mat_out = materialize_output_dir.strip() if materialize_output_dir and materialize_output_dir.strip() not in ("{auto}", "") else str(save_dir / "dataset")
                cmd = [
                    sys.executable,
                    str(materialize_script),
                    "--input-jsonl", str(dataset_jsonl_path),
                    "--output-dir", mat_out,
                ]
                max_clips = int(materialize_max_clips) if materialize_max_clips else 0
                if max_clips > 0:
                    cmd += ["--max-clips", str(max_clips)]

                audio_source_dirs = set()
                for fp, orig in all_files:
                    if orig:
                        audio_source_dirs.add(str(Path(orig).parent))
                    else:
                        audio_source_dirs.add(str(Path(fp).parent))
                for adir in audio_source_dirs:
                    cmd += ["--audio-dir", adir]

                logger.info("[后处理] 数据物化命令: %s", subprocess.list2cmdline(cmd))
                result = subprocess.run(cmd, capture_output=True, text=True)
                stdout_text = (result.stdout or "").strip()
                stderr_text = (result.stderr or "").strip()
                if stdout_text:
                    logger.info("[后处理][物化][stdout]\n%s", stdout_text)
                if stderr_text:
                    logger.warning("[后处理][物化][stderr]\n%s", stderr_text)
                if result.returncode == 0:
                    post_logs.append("[后处理] 数据物化完成")
                else:
                    post_logs.append(f"[后处理] 数据物化失败: {result.stderr.strip() or result.stdout.strip()}")
            except Exception as e:
                post_logs.append(f"[后处理] 数据物化异常: {e}")

        if (not stop_requested) and enable_analysis and dataset_jsonl_path and dataset_jsonl_path.exists():
            try:
                analysis_script = Path(__file__).resolve().parents[2] / "scripts" / "analyze_corpus.py"
                ana_out = analysis_output_dir.strip() if analysis_output_dir and analysis_output_dir.strip() not in ("{auto}", "") else str(save_dir / "analysis")
                cmd = [
                    sys.executable,
                    str(analysis_script),
                    "--merged-jsonl", str(dataset_jsonl_path),
                    "--quality-stats", str(quality_stats_path),
                    "--output-dir", ana_out,
                ]
                if chart_funnel:
                    cmd.append("--enable-funnel")
                if chart_speed:
                    cmd.append("--enable-speed-histogram")
                if chart_delay:
                    cmd.append("--enable-interaction-delay")
                if chart_robustness:
                    cmd.append("--enable-robustness")
                if chart_llm_analysis:
                    cmd.append("--enable-llm-analysis")

                logger.info("[后处理] 可视化分析命令: %s", subprocess.list2cmdline(cmd))
                result = subprocess.run(cmd, capture_output=True, text=True)
                stdout_text = (result.stdout or "").strip()
                stderr_text = (result.stderr or "").strip()
                if stdout_text:
                    logger.info("[后处理][分析][stdout]\n%s", stdout_text)
                if stderr_text:
                    logger.warning("[后处理][分析][stderr]\n%s", stderr_text)
                if result.returncode == 0:
                    post_logs.append("[后处理] 可视化分析完成")
                else:
                    post_logs.append(f"[后处理] 可视化分析失败: {result.stderr.strip() or result.stdout.strip()}")
            except Exception as e:
                post_logs.append(f"[后处理] 可视化分析异常: {e}")

        should_release_memory = bool(release_memory) or bool(stop_requested)
        if should_release_memory:
            release_desc = "停止后释放显存..." if stop_requested else "释放显存..."
            progress(0.95, desc=release_desc)
            with _engine_lock:
                engine.unload_all()

        completed_files = _count_completed_files()
        if stop_requested:
            progress(1.0, desc="任务已停止")
            final_log = f"任务被用户手动停止。已完成 {completed_files}/{total_files} 个文件。\n" + log_buffer
            run_status = "stopped"
        else:
            progress(1.0, desc="语料库构建完成")
            final_log = f"全部 {total_files} 个文件处理完毕。\n" + log_buffer
            run_status = "completed"

        if post_logs:
            final_log = final_log + "\n" + "\n".join(post_logs) + "\n"

        human_log = final_log + "\n" + stats_report + "\n"
        try:
            with open(quality_log_path, "w", encoding="utf-8") as f:
                f.write(human_log)
        except Exception as e:
            logger.warning(f"写入 日志.txt 失败: {e}")

        try:
            with open(terminal_log_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write("[UI摘要]\n")
                f.write(human_log)
        except Exception as e:
            logger.warning(f"写入 terminal.log 失败: {e}")

        _write_checkpoint(
            run_status,
            completed_files if stop_requested else total_files,
            "已停止" if stop_requested else "全部文件",
            "用户已停止" if stop_requested else "语料库构建完成",
        )

        yield _render_corpus_progress_panel(
            total_files=total_files,
            current_index=completed_files if stop_requested else total_files,
            current_file="已停止" if stop_requested else "全部文件",
            current_file_progress=1.0,
            completed_files=completed_files if stop_requested else total_files,
            current_step=(
                f"用户已停止。当前已完成 {completed_files}/{total_files}，成功 {total_stats['success_files']}，失败 {total_stats['failed_files']}。"
                if stop_requested
                else f"语料库构建完成。成功 {total_stats['success_files']}，失败 {total_stats['failed_files']}。"
            ),
            elapsed_seconds=max(time.time() - task_started_at, 0.0),
            current_eta_seconds=0.0,
            eta_seconds=0.0 if not stop_requested else -1.0,
            success_files=total_stats['success_files'],
            failed_files=total_stats['failed_files'],
            status_note="用户已停止" if stop_requested else "",
        )
    finally:
        _clear_active_corpus_stop_event(task_stop_event)
        if step_sync_handler:
            try:
                root_logger.removeHandler(step_sync_handler)
            except Exception:
                pass
        if terminal_handler:
            try:
                root_logger.removeHandler(terminal_handler)
            except Exception:
                pass
            try:
                terminal_handler.close()
            except Exception:
                pass
        _release_corpus_task_guard()


def _format_corpus_stats(stats: dict, total_files: int, output_dir: str, final_dataset: str = None) -> str:
    """格式化语料库清洗统计报告（与构建日志区分，专注数据质量分析）"""
    total_input = stats['total_input']
    total_output = stats['total_output']
    total_removed = total_input - total_output
    keep_rate = (total_output / total_input * 100) if total_input > 0 else 0

    # 各规则移除占比
    rule_details = [
        ('A 置信度过滤', stats['rule_a']),
        ('B 重叠音过滤', stats['rule_b']),
        ('C 长度匹配度', stats['rule_c']),
        ('D 上下文孤岛', stats['rule_d']),
        ('E 低信息量', stats['rule_e']),
        ('F 信噪比', stats['rule_f']),
        ('G 黑名单', stats['rule_g']),
        ('H LLM语义', stats['rule_h']),
    ]

    lines = [
        "┌───────────── 数据质量报告 ─────────────┐",
        f"│ 文件: {stats['success_files']} 成功 / {stats['failed_files']} 失败 (共 {total_files})",
        "├────────────────────────────────────────┤",
        f"│ 原始段落:  {total_input:>6}",
        f"│ 保留段落:  {total_output:>6}  ({keep_rate:.1f}%)",
        f"│ 移除段落:  {total_removed:>6}",
        "├────────── 清洗规则命中明细 ────────────┤",
    ]
    for name, count in rule_details:
        pct = (count / total_removed * 100) if total_removed > 0 else 0
        bar_len = int(pct / 5)  # 每5%一格，最多20格
        bar = '█' * bar_len + '░' * (20 - bar_len)
        lines.append(f"│ {name:<10} {count:>4}  {bar} {pct:>5.1f}%")
    lines.append("├────────────────────────────────────────┤")
    # 规则 H 特殊状态
    rule_h_unchecked = stats.get('rule_h_unchecked', 0)
    if rule_h_unchecked > 0:
        lines.append(f"│ ⚠ 规则H未检查: {rule_h_unchecked} 段 (API降级保留)")
    # LLM 接管模式统计
    total_flagged = (stats.get('rule_c_flagged', 0) + stats.get('rule_d_flagged', 0)
                     + stats.get('rule_e_flagged', 0) + stats.get('rule_g_flagged', 0))
    if total_flagged > 0:
        lines.append("├──────── LLM 接管模式统计 ──────────────┤")
        for label, key in [('C 长度匹配', 'rule_c_flagged'), ('D 上下文孤岛', 'rule_d_flagged'),
                           ('E 低信息量', 'rule_e_flagged'), ('G 黑名单', 'rule_g_flagged')]:
            val = stats.get(key, 0)
            if val:
                lines.append(f"│ 🏷 {label} 标记: {val} 段")
        overridden = stats.get('rule_h_overridden', 0)
        lines.append(f"│ ↩ LLM 翻转保留: {overridden} 段 (规则标记→LLM保留)")
        lines.append("│ 注: LLM裁决删除已按 C/D/E/G 重新归因，H仅统计纯语义删除")
    lines.append(f"│ 输出目录: {output_dir}")
    if final_dataset:
        lines.append(f"│ 数据集文件: {final_dataset}")
    lines.append("└────────────────────────────────────────┘")
    return "\n".join(lines)


def _na_gpu_payload(index: int, name: str) -> dict:
    return {
        "index": int(index),
        "name": str(name),
        "temperature": None,
        "fan": {"speed": None},
        "utilization": {"gpu": None, "memory": None},
        "memory": {"total": None, "used": None},
        "power": {"draw": None, "limit": None},
        "clocks": {"graphics": None},
    }


def _parse_gpu_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "N/A", "[N/A]", "Not Supported", "Unknown Error", "-"):
        return None
    try:
        number = float(text)
    except Exception:
        return None
    if number.is_integer():
        return int(number)
    return number


def get_gpu_stats():
    """获取 GPU 监控数据。CPU 模式返回 N/A 占位，不返回 Mock 数据。"""
    response = {
        "hasNvidiaSmi": False,
        "isMock": False,
        "cuda_available": False,
        "gpus": [],
    }

    try:
        import torch
    except Exception as e:
        response["error"] = f"PyTorch 不可用: {e}"
        response["gpus"] = [_na_gpu_payload(0, "CUDA Unavailable (CPU Mode)")]
        return response

    cuda_available = bool(torch.cuda.is_available())
    response["cuda_available"] = cuda_available
    if not cuda_available:
        response["error"] = "CUDA 不可用，当前运行在 CPU 模式。"
        response["gpus"] = [_na_gpu_payload(0, "CUDA Unavailable (CPU Mode)")]
        return response

    gpu_names = {}
    gpu_count = 0
    try:
        gpu_count = int(torch.cuda.device_count())
        for idx in range(gpu_count):
            gpu_names[idx] = str(torch.cuda.get_device_name(idx))
    except Exception:
        gpu_count = 0

    query_fields = [
        "index",
        "name",
        "temperature.gpu",
        "fan.speed",
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.used",
        "clocks.current.graphics",
        "power.draw",
        "power.limit",
        "enforced.power.limit",
        "power.max_limit",
        "power.default_limit",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]

    parsed_gpus = []
    result = None
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        response["error"] = "未找到 nvidia-smi，部分 GPU 指标不可用。"
    except Exception as e:
        response["error"] = f"调用 nvidia-smi 失败: {e}"

    if result is not None and result.returncode == 0:
        response["hasNvidiaSmi"] = True
        reader = csv.reader(io.StringIO(result.stdout or ""))
        for row in reader:
            if len(row) < 14:
                continue

            idx_num = _parse_gpu_number(row[0])
            gpu_index = int(idx_num) if idx_num is not None else len(parsed_gpus)
            gpu_name = row[1].strip() if row[1].strip() else gpu_names.get(gpu_index, f"GPU {gpu_index}")

            power_limit_candidates = [
                _parse_gpu_number(row[10]),
                _parse_gpu_number(row[11]),
                _parse_gpu_number(row[12]),
                _parse_gpu_number(row[13]),
            ]
            power_limit_value = next((v for v in power_limit_candidates if v is not None), None)

            parsed_gpus.append(
                {
                    "index": gpu_index,
                    "name": gpu_name,
                    "temperature": _parse_gpu_number(row[2]),
                    "fan": {"speed": _parse_gpu_number(row[3])},
                    "utilization": {
                        "gpu": _parse_gpu_number(row[4]),
                        "memory": _parse_gpu_number(row[5]),
                    },
                    "memory": {
                        "total": _parse_gpu_number(row[6]),
                        "used": _parse_gpu_number(row[7]),
                    },
                    "clocks": {"graphics": _parse_gpu_number(row[8])},
                    "power": {
                        "draw": _parse_gpu_number(row[9]),
                        "limit": power_limit_value,
                    },
                }
            )
    elif result is not None and result.returncode != 0 and "error" not in response:
        err = (result.stderr or result.stdout or "").strip()
        response["error"] = err or "nvidia-smi 调用失败。"

    if not parsed_gpus:
        fallback_count = gpu_count if gpu_count > 0 else 1
        parsed_gpus = [
            _na_gpu_payload(i, gpu_names.get(i, f"GPU {i}"))
            for i in range(fallback_count)
        ]
    elif gpu_count > len(parsed_gpus):
        existing = {int(item.get("index", -1)) for item in parsed_gpus}
        for idx in range(gpu_count):
            if idx not in existing:
                parsed_gpus.append(_na_gpu_payload(idx, gpu_names.get(idx, f"GPU {idx}")))

    parsed_gpus.sort(key=lambda item: int(item.get("index", 0)))
    response["gpus"] = parsed_gpus
    return response


def _get_gpu_stats_json() -> str:
    """将 GPU 状态编码为 JSON 字符串，供 gr.HTML(value=..., every=...) 轮询刷新。"""
    try:
        return json.dumps(get_gpu_stats(), ensure_ascii=False)
    except Exception as e:
        fallback = {
            "hasNvidiaSmi": False,
            "isMock": False,
            "cuda_available": None,
            "gpus": [
                {
                    "index": 0,
                    "name": "GPU Status Unavailable",
                    "temperature": None,
                    "fan": {"speed": None},
                    "utilization": {"gpu": None, "memory": None},
                    "memory": {"total": None, "used": None},
                    "power": {"draw": None, "limit": None},
                    "clocks": {"graphics": None},
                }
            ],
            "error": f"GPU 状态获取失败: {e}",
        }
        return json.dumps(fallback, ensure_ascii=False)

# GPU 监视器组件，定期调用 get_gpu_stats 获取数据并刷新显示。前端 JS 负责解析 JSON 并渲染成表格或图表。
class GPUMonitor(gr.HTML):
    def __init__(self, update_interval=1000, show_last_updated=True, **kwargs):
        html_template = """
        <div class="gpu-monitor-container">
            <div class="gpu-header">
                <h1 class="gpu-title">GPU监视器</h1>
                ${show_last_updated === 'true' ? `<div class="gpu-time">Last updated: <span id="gpu-time-val">Loading...</span></div>` : ''}
            </div>
            <div id="gpu-alert"></div>
            <div id="gpu-root" class="gpu-grid"></div>
        </div>
        """

        css_template = """
            .gpu-monitor-container {
                font-family: "Inter", sans-serif;
                width: 100%;
            }

            .gpu-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .gpu-title {
                font-size: 1.125rem;
                margin: 0;
                font-weight: 500;
                color: var(--body-text-color);
            }

            .gpu-time {
                font-size: 0.75rem;
                color: var(--body-text-color-subdued);
            }

            .gpu-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 1rem;
            }

            .gpu-card {
                background-color: var(--background-fill-secondary);
                border-radius: 0.75rem;
                box-shadow: var(--shadow-drop-md);
                border: 1px solid var(--border-color-primary);
                overflow: hidden;
                transition: box-shadow 0.3s ease;
            }

            .gpu-card:hover {
                box-shadow: var(--shadow-drop-lg);
            }

            .gpu-card-header {
                background-color: var(--background-fill-primary);
                padding: 0.75rem 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--border-color-primary);
            }

            .gpu-name {
                font-weight: 600;
                color: var(--body-text-color);
                margin: 0;
                font-size: 0.9rem;
            }

            .gpu-badge {
                padding: 0.125rem 0.5rem;
                background-color: var(--background-fill-secondary);
                border-radius: 9999px;
                font-size: 0.75rem;
                color: var(--body-text-color);
            }

            .gpu-body {
                padding: 1rem;
            }

            .gpu-metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }

            .metric-item {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.75rem;
            }

            .metric-label {
                font-size: 0.75rem;
                color: var(--body-text-color-subdued);
                margin: 0;
            }

            .metric-value {
                font-size: 0.875rem;
                font-weight: 500;
                margin: 0;
                color: var(--body-text-color);
            }

            .metric-na {
                color: var(--body-text-color-subdued) !important;
            }

            .progress-header {
                display: flex;
                align-items: center;
                width: 100%;
                margin-bottom: 0.25rem;
            }

            .progress-left {
                display: flex;
                align-items: center;
                justify-content: flex-start;
                text-align: left;
                gap: 0.25rem;
                flex: 1;
                min-width: 0;
            }

            .progress-value {
                margin-left: auto;
                font-size: 0.75rem;
                color: var(--body-text-color);
                text-align: right;
                flex: 0 0 auto;
            }

            .progress-track {
                width: 100%;
                background-color: var(--neutral-200);
                border-radius: 9999px;
                height: 0.35rem;
                margin-bottom: 0.75rem;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                border-radius: 9999px;
                transition:
                    width 0.3s ease,
                    background-color 0.3s ease;
            }

            .progress-footer {
                font-size: 0.75rem;
                color: var(--body-text-color-subdued);
                margin-top: -0.5rem;
            }

            .gpu-footer {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                padding-top: 0.5rem;
                margin-top: 0.5rem;
                border-top: 1px solid var(--border-color-primary);
            }

            .text-emerald-500 {
                color: #10b981 !important;
                stroke: #10b981 !important;
            }

            .bg-emerald-500 {
                background-color: #10b981 !important;
            }

            .text-amber-500 {
                color: #f59e0b !important;
                stroke: #f59e0b !important;
            }

            .bg-amber-500 {
                background-color: #f59e0b !important;
            }

            .text-rose-500 {
                color: #f43f5e !important;
                stroke: #f43f5e !important;
            }

            .bg-rose-500 {
                background-color: #f43f5e !important;
            }

            .text-blue-500 {
                color: #3b82f6 !important;
                stroke: #3b82f6 !important;
            }

            .bg-blue-500 {
                background-color: #3b82f6 !important;
            }

            .text-purple-500 {
                color: #a855f7 !important;
                stroke: #a855f7 !important;
            }

            .icon-muted {
                color: var(--body-text-color-subdued) !important;
                stroke: var(--body-text-color-subdued) !important;
            }

            .icon {
                width: 16px;
                height: 16px;
                stroke-width: 2.2;
            }

            .metric-icon-wrap {
                position: relative;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 16px;
                height: 16px;
                margin-right: 4px;
                flex: 0 0 16px;
            }

            .metric-icon-wrap .icon {
                position: absolute;
                inset: 0;
            }

            .metric-icon-fallback {
                font-size: 12px;
                line-height: 1;
                font-weight: 700;
            }

            .lucide-ready .metric-icon-fallback {
                display: none;
            }

            .gpu-fill-na {
                background-color: #94a3b8 !important;
            }

            .alert-box {
                background-color: rgba(245, 158, 11, 0.1);
                border: 1px solid #f59e0b;
                color: #d97706;
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                font-size: 0.875rem;
            }
        """

        js_on_load = """
        const root = element.querySelector("#gpu-root");
        const timeVal = element.querySelector("#gpu-time-val");
        const alertBox = element.querySelector("#gpu-alert");
        const NA_TEXT = "N/A";
        const LUCIDE_SCRIPT_ID = "gpu-monitor-lucide-script";
        const LUCIDE_SRC = "https://cdn.jsdelivr.net/npm/lucide@0.468.0/dist/umd/lucide.min.js";

        const FALLBACK_NA_DATA = {
            hasNvidiaSmi: false,
            isMock: false,
            cuda_available: null,
            gpus: [
                {
                    index: 0,
                    name: "GPU Status Unavailable",
                    temperature: null,
                    fan: { speed: null },
                    utilization: { gpu: null, memory: null },
                    memory: { total: null, used: null },
                    power: { draw: null, limit: null },
                    clocks: { graphics: null },
                },
            ],
        };

        const toNumber = (value) => {
            if (value === null || value === undefined || value === "") return null;
            const num = Number(value);
            return Number.isFinite(num) ? num : null;
        };

        const formatMemory = (mb, disabled = false) => {
            const v = toNumber(mb);
            if (disabled || v === null || v < 0) return NA_TEXT;
            return v >= 1024 ? `${(v / 1024).toFixed(1)} GB` : `${Math.round(v)} MB`;
        };

        const formatValue = (value, { suffix = "", digits = 0, disabled = false } = {}) => {
            const v = toNumber(value);
            if (disabled || v === null || Number.isNaN(v)) return NA_TEXT;
            if (digits > 0) return `${v.toFixed(digits)}${suffix}`;
            return `${Math.round(v)}${suffix}`;
        };

        const getUtilColor = (val, disabled = false) => {
            if (disabled) return "gpu-fill-na";
            const v = toNumber(val);
            if (v === null) return "gpu-fill-na";
            return v < 30 ? "bg-emerald-500" : v < 70 ? "bg-amber-500" : "bg-rose-500";
        };

        const getTempClass = (temp, disabled = false) => {
            if (disabled) return "metric-na";
            const t = toNumber(temp);
            if (t === null) return "metric-na";
            return t < 50 ? "text-emerald-500" : t < 80 ? "text-amber-500" : "text-rose-500";
        };

        const getPercentPayload = (value, disabled = false) => {
            if (disabled) return { text: NA_TEXT, width: 0 };
            const v = toNumber(value);
            if (v === null) return { text: NA_TEXT, width: 0 };
            const clamped = Math.max(0, Math.min(100, v));
            return { text: `${clamped.toFixed(1)}%`, width: clamped };
        };

        const normalizeGpu = (gpu, index, disabled = false) => {
            const base = gpu || {};
            return {
                index: Number.isFinite(Number(base.index)) ? Number(base.index) : index,
                name: base.name || `GPU ${index}`,
                temperature: disabled ? null : toNumber(base.temperature),
                fan: { speed: disabled ? null : toNumber(base?.fan?.speed) },
                utilization: {
                    gpu: disabled ? null : toNumber(base?.utilization?.gpu),
                    memory: disabled ? null : toNumber(base?.utilization?.memory),
                },
                memory: {
                    total: disabled ? null : toNumber(base?.memory?.total),
                    used: disabled ? null : toNumber(base?.memory?.used),
                },
                power: {
                    draw: disabled ? null : toNumber(base?.power?.draw),
                    limit: disabled ? null : toNumber(base?.power?.limit),
                },
                clocks: {
                    graphics: disabled ? null : toNumber(base?.clocks?.graphics),
                },
            };
        };

        const getRuntimeProps = () => {
            if (props && typeof props === "object") {
                const nested = props.props && typeof props.props === "object" ? props.props : {};
                return { ...props, ...nested };
            }
            return {};
        };

        const parseDataFromProps = () => {
            const runtimeProps = getRuntimeProps();
            const candidates = [runtimeProps?.value, props?.value];
            for (const raw of candidates) {
                if (raw && typeof raw === "object") {
                    return raw;
                }
                if (typeof raw === "string" && raw.trim()) {
                    try {
                        const parsed = JSON.parse(raw);
                        if (parsed && typeof parsed === "object") {
                            return parsed;
                        }
                    } catch {
                        // Ignore JSON parse errors and continue trying other candidates.
                    }
                }
            }
            return null;
        };

        const parseAnyPayloadToObject = (raw) => {
            if (raw && typeof raw === "object") {
                return raw;
            }
            if (typeof raw === "string" && raw.trim()) {
                try {
                    const parsed = JSON.parse(raw);
                    if (parsed && typeof parsed === "object") {
                        return parsed;
                    }
                } catch {
                    return null;
                }
            }
            return null;
        };

        const fetchDataFromRunApi = async () => {
            try {
                const response = await fetch("/gradio_api/run/_get_gpu_stats_json", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ data: [] }),
                    cache: "no-store",
                });
                if (!response.ok) {
                    return null;
                }

                const payload = await response.json();
                if (!payload || !Array.isArray(payload.data) || payload.data.length === 0) {
                    return null;
                }

                return parseAnyPayloadToObject(payload.data[0]);
            } catch (error) {
                console.error("Fetch /gradio_api/run/_get_gpu_stats_json error:", error);
                return null;
            }
        };

        const ensureLucideLoaded = async () => {
            if (window.lucide) {
                return window.lucide;
            }

            if (window.__gpuMonitorLucidePromise) {
                return window.__gpuMonitorLucidePromise;
            }

            window.__gpuMonitorLucidePromise = new Promise((resolve, reject) => {
                const finishIfReady = () => {
                    if (window.lucide) {
                        resolve(window.lucide);
                        return true;
                    }
                    return false;
                };

                if (finishIfReady()) {
                    return;
                }

                let script = document.getElementById(LUCIDE_SCRIPT_ID);
                if (!script) {
                    script = document.createElement("script");
                    script.id = LUCIDE_SCRIPT_ID;
                    script.src = LUCIDE_SRC;
                    script.async = true;
                    document.head.appendChild(script);
                }

                script.addEventListener("load", () => {
                    if (!finishIfReady()) {
                        reject(new Error("Lucide loaded but window.lucide is unavailable"));
                    }
                }, { once: true });

                script.addEventListener("error", () => {
                    reject(new Error("Failed to load Lucide script"));
                }, { once: true });
            });

            return window.__gpuMonitorLucidePromise;
        };

        async function initLucide() {
            try {
                await ensureLucideLoaded();
                if (window.lucide) {
                    window.lucide.createIcons({ root: root });
                    root.classList.add("lucide-ready");
                    return true;
                }
            } catch (error) {
                console.warn("Lucide init failed:", error);
            }

            root.classList.remove("lucide-ready");
            return false;
        }

        const iconHtml = (name, iconClass, fallbackChar, extraStyle = "") =>
            `<span class="metric-icon-wrap" style="${extraStyle}"><i data-lucide="${name}" class="icon ${iconClass}"></i><span class="metric-icon-fallback ${iconClass}">${fallbackChar}</span></span>`;

        const generateGPUCard = (gpu, disabled = false) => {
            const utilMain = getPercentPayload(gpu?.utilization?.gpu, disabled);
            const memoryTotal = toNumber(gpu?.memory?.total);
            const memoryUsed = toNumber(gpu?.memory?.used);
            const memPercent =
                !disabled && memoryTotal && memoryTotal > 0 && memoryUsed !== null
                    ? getPercentPayload((memoryUsed / memoryTotal) * 100, false)
                    : { text: NA_TEXT, width: 0 };

            return `
                <div class="gpu-card" id="gpu-card-${gpu.index}">
                    <div class="gpu-card-header">
                        <h2 class="gpu-name">${gpu.name}</h2>
                        <span class="gpu-badge"># ${gpu.index}</span>
                    </div>
                    <div class="gpu-body">
                        <div class="gpu-metrics-grid">
                            <div>
                                <div class="metric-item">
                                    ${iconHtml("thermometer", getTempClass(gpu.temperature, disabled), "🌡")}
                                    <div><p class="metric-label">温度</p><p class="metric-value ${getTempClass(gpu.temperature, disabled)}">${formatValue(gpu.temperature, { suffix: "°C", digits: 0, disabled })}</p></div>
                                </div>
                                <div class="metric-item">
                                    ${iconHtml("fan", disabled ? "icon-muted" : "text-blue-500", "🌀")}
                                    <div><p class="metric-label">风扇转速</p><p class="metric-value ${disabled ? "metric-na" : "text-blue-500"}">${formatValue(gpu?.fan?.speed, { suffix: "%", digits: 0, disabled })}</p></div>
                                </div>
                            </div>
                            <div>
                                <div class="progress-header"><div class="progress-left">${iconHtml("cpu", "icon-muted", "🖥")}<p class="metric-label">GPU负载</p></div><span class="progress-value ${disabled ? "metric-na" : ""}">${utilMain.text}</span></div>
                                <div class="progress-track"><div class="progress-fill ${getUtilColor(gpu?.utilization?.gpu, disabled)}" style="width: ${utilMain.width}%"></div></div>
                                <div class="progress-header" style="margin-top: 12px;"><div class="progress-left">${iconHtml("hard-drive", disabled ? "icon-muted" : "text-blue-500", "💾")}<p class="metric-label">显存</p></div><span class="progress-value ${disabled ? "metric-na" : ""}">${memPercent.text}</span></div>
                                <div class="progress-track"><div class="progress-fill ${disabled ? "gpu-fill-na" : "bg-blue-500"}" style="width: ${memPercent.width}%"></div></div>
                                <p class="progress-footer ${disabled ? "metric-na" : ""}">${formatMemory(gpu?.memory?.used, disabled)} / ${formatMemory(gpu?.memory?.total, disabled)}</p>
                            </div>
                        </div>
                        <div class="gpu-footer">
                            <div class="metric-item" style="margin:0">${iconHtml("clock", disabled ? "icon-muted" : "text-purple-500", "⏱")}<div><p class="metric-label">核心频率</p><p class="metric-value ${disabled ? "metric-na" : ""}">${formatValue(gpu?.clocks?.graphics, { suffix: " MHz", digits: 0, disabled })}</p></div></div>
                            <div class="metric-item" style="margin:0">${iconHtml("zap", disabled ? "icon-muted" : "text-amber-500", "⚡")}<div><p class="metric-label">功耗</p><p class="metric-value ${disabled ? "metric-na" : ""}">${formatValue(gpu?.power?.draw, { suffix: "W", digits: 1, disabled })} <span style="font-size:0.7rem; color:var(--body-text-color-subdued)">/ ${formatValue(gpu?.power?.limit, { suffix: "W", digits: 0, disabled })}</span></p></div></div>
                        </div>
                    </div>
                </div>
            `;
        };

        async function fetchAndUpdate() {
            let data = null;
            let backendConnected = false;
            const apiData = await fetchDataFromRunApi();
            if (apiData && typeof apiData === "object") {
                data = apiData;
                backendConnected = true;
            }

            // 若 run API 暂时不可用，回退到 props.value。
            if (!backendConnected) {
                data = parseDataFromProps();
                backendConnected = !!(data && typeof data === "object");
            }

            if (!data || typeof data !== "object") {
                data = FALLBACK_NA_DATA;
            }

            const isCpuMode = backendConnected && data.cuda_available === false;
            const disableAllMetrics = !backendConnected || isCpuMode;
            const sourceGpus =
                Array.isArray(data.gpus) && data.gpus.length > 0 ? data.gpus : FALLBACK_NA_DATA.gpus;
            const gpus = sourceGpus.map((gpu, index) => normalizeGpu(gpu, index, disableAllMetrics));

            root.innerHTML = gpus.map((gpu) => generateGPUCard(gpu, disableAllMetrics)).join("");
            await initLucide();

            if (timeVal) {
                timeVal.innerText = new Date().toLocaleTimeString();
            }

            const alertLines = [];
            if (!backendConnected) {
                alertLines.push("GPU 数据服务未连接，请等待页面初始化或刷新重试。");
            } else if (isCpuMode) {
                alertLines.push("CUDA 不可用，当前为 CPU 模式。");
            }
            if (data.error) {
                alertLines.push(data.error);
            }
            if (alertLines.length > 0) {
                alertBox.innerHTML = `<div class="alert-box">${alertLines.join("<br/>")}</div>`;
            } else {
                alertBox.innerHTML = "";
            }
        }

        fetchAndUpdate();
        const parsedInterval = Number(getRuntimeProps().update_interval);
        const intervalMs = Number.isFinite(parsedInterval) && parsedInterval > 0 ? parsedInterval : 1000;
        const intervalId = setInterval(fetchAndUpdate, intervalMs);
        return () => clearInterval(intervalId);
        """

        super().__init__(
            value=_get_gpu_stats_json,
            every=max(float(update_interval) / 1000.0, 0.2),
            html_template=html_template,
            css_template=css_template,
            js_on_load=js_on_load,
            update_interval=update_interval,
            show_last_updated=str(show_last_updated).lower(),
            **kwargs,
        )

    def api_info(self):
        return {"type": "null"}

def create_ui():
    """Gradio 界面构建"""
    with gr.Blocks(title="WhisperX 语料库构建平台") as app:
        gr.Markdown("## WhisperX 语料库构建平台")
        
        if not HF_TOKEN:
            gr.Warning("警告: 未检测到 HF_TOKEN，说话人分离功能将不可用。")

        with gr.Tabs():
            # ============================================================
            #  Tab 1: 原有字幕提取功能
            # ============================================================
            with gr.TabItem("字幕提取"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="文件批处理输入",
                            type="filepath",
                            file_types=["audio", "video"],
                            file_count="multiple",
                            height=120
                        )
                        
                        with gr.Tabs():
                            with gr.TabItem("基础配置"):
                                model_sel = gr.Dropdown(
                                    ["base", "small", "medium", "large-v2", "large-v3"], 
                                    value="large-v3", 
                                    label="Whisper 模型架构",
                                    info="模型越大识别精度越高，但显存占用和推理耗时也会显著增加。8GB 显存及以上推荐使用 large-v3。"
                                )
                                lang_sel = gr.Dropdown(
                                    [None, "zh", "en", "ja", "ko"], 
                                    value=None, 
                                    label="源语言设置",
                                    info="若不确定，请保持为空以启用自动检测。指定语言可略微提升准确率。"
                                )
                                
                                gr.Markdown("---")
                                enable_demucs = gr.Checkbox(
                                    label="启用人声分离预处理 (BS-RoFormer)", 
                                    value=False,
                                    info="在识别前将人声从背景音乐或噪音中分离。适用于歌曲或高噪环境，会增加总处理时间，但对于提高识别率很有帮助。"
                                )
                                
                                prompt_input = gr.Textbox(
                                    label="上下文提示词 (Prompt)",
                                    info="提供给模型的风格引导或专有名词参考。选择语言后会自动填入对应提示词，也可手动修改。",
                                    value="",
                                    placeholder="请输入提示词...",
                                    lines=3,
                                    interactive=True 
                                )

                                def _on_lang_change(selected_lang):
                                    """语言切换时自动填入对应的 Whisper 提示词"""
                                    return LANG_PROMPT_MAP.get(selected_lang, "")

                                lang_sel.change(
                                    fn=_on_lang_change,
                                    inputs=[lang_sel],
                                    outputs=[prompt_input],
                                )

                            with gr.TabItem("推理参数"):
                                compute_type_sel = gr.Dropdown(
                                    ["int8", "float16", "float32"],
                                    value="float16",
                                    label="量化精度",
                                    info="float16 为 GPU 标准精度；int8 可大幅降低显存占用但可能轻微损失精度；float32 仅用于 CPU 模式。"
                                )
                                vad_onset_slider = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.35, step=0.05,
                                    label="VAD 触发阈值 (Onset)",
                                    info="语音活动检测的灵敏度。数值越高越严格（减少幻觉），数值越低越灵敏（保留更多细节）。默认 0.35。"
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**幻觉过滤 (Hallucination Filter)**")
                                hallucination_mode_radio = gr.Radio(
                                    ["代码规则过滤", "LLM 智能过滤", "关闭"],
                                    value="关闭",
                                    label="幻觉过滤模式",
                                    info="代码规则: 基于置信度+模式匹配+时间异常(0 API 费用) | LLM 智能: 由大模型判断(需配置 llm_config.json) | 关闭: 不过滤"
                                )
                                hallucination_threshold_slider = gr.Slider(
                                    minimum=0.1, maximum=0.8, value=0.35, step=0.05,
                                    label="置信度阈值 (仅代码规则模式)",
                                    info="平均词置信度低于此值且零分词占比过高的段落将被移除。推荐 0.3-0.4。"
                                )
                                
                            with gr.TabItem("说话人区分"):
                                enable_diar = gr.Checkbox(
                                    label="启用说话人聚类 (Diarization)", 
                                    value=False,
                                    info="识别并区分音频中的不同说话人。需要有效的 HuggingFace Token。"
                                )
                                with gr.Row():
                                    min_spk = gr.Number(label="最小说话人数", value=1, precision=0)
                                    max_spk = gr.Number(label="最大说话人数", value=5, precision=0)

                            with gr.TabItem("LLM 智能处理"):
                                _llm_status = f"✅ 已配置 ({LLM_MODEL})" if is_llm_configured() else "❌ 未配置"
                                gr.Markdown(
                                    f"**API 状态: {_llm_status}**\n\n"
                                    "API 配置请编辑 `config/llm_config.json` 文件，修改后重启程序生效。\n"
                                    "支持 OpenAI / DeepSeek / Ollama 等 OpenAI 兼容接口。"
                                )
                                llm_enabled_cb = gr.Checkbox(
                                    label="启用 LLM 后处理",
                                    value=False,
                                    info="在 ASR 识别后调用 LLM 进行智能断句与翻译。需先配置 llm_config.json。",
                                    interactive=is_llm_configured(),
                                )
                                gr.Markdown("---")
                                llm_mode_sel = gr.Radio(
                                    ["segmentation", "translation", "both"],
                                    value="segmentation",
                                    label="处理模式",
                                    info="segmentation: 仅智能断句 | translation: 仅翻译 | both: 断句 + 翻译（双语字幕）",
                                )
                                llm_target_lang_input = gr.Textbox(
                                    label="翻译目标语言",
                                    value="简体中文",
                                    placeholder="English / 简体中文 / 日本語",
                                    info="仅在模式包含翻译时生效。请使用自然语言描述目标语种。",
                                    max_lines=1,
                                )

                        with gr.Accordion("输出与资源管理", open=False):
                            export_srt = gr.Checkbox(label="同步导出 SRT 字幕文件", value=True)
                            release_mem = gr.Checkbox(
                                label="任务完成后释放显存", 
                                value=True,
                                info="任务结束后卸载模型并执行垃圾回收，释放 GPU 资源给其他应用。取消勾选能够提升批处理时候的效率。"
                            )

                        with gr.Row():
                            btn_submit = gr.Button("开始处理任务", variant="primary", scale=2)
                            btn_stop = gr.Button("终止任务", variant="stop", scale=1)

                    with gr.Column(scale=2):
                        text_out = gr.TextArea(
                            label="系统运行日志", 
                            lines=30,
                            placeholder="任务进度与详细日志将显示在此处..."
                        )
                        
                        with gr.Row(variant="panel"):
                            output_dir_input = gr.Textbox(
                                label="输出目录 (Output Directory)", 
                                value=str(OUTPUT_DIR),
                                info="选择或输入结果保存路径",
                                scale=5,
                                max_lines=1,
                                interactive=True
                            )
                            btn_browse = gr.Button("选择文件夹", scale=1)
                            btn_open = gr.Button("打开", scale=1, variant="secondary")

                        def browse_folder_action():
                            try:
                                import tkinter as tk
                                from tkinter import filedialog
                                root = tk.Tk()
                                root.withdraw()
                                root.attributes('-topmost', True)
                                selected_path = filedialog.askdirectory()
                                root.destroy()
                                if selected_path:
                                    return str(Path(selected_path))
                                else:
                                    return gr.update()
                            except (ImportError, Exception) as e:
                                logger.warning(f"文件夹选择器不可用: {e}。请直接在输出目录框中手动输入路径。")
                                return gr.update()

                        def open_folder_action(custom_path):
                            path = custom_path.strip() if custom_path else str(OUTPUT_DIR)
                            path_obj = Path(path)
                            if not path_obj.exists():
                                try:
                                    path_obj.mkdir(parents=True, exist_ok=True)
                                except Exception:
                                    return 
                            if sys.platform == "win32":
                                os.startfile(path)
                            elif sys.platform == "darwin":
                                subprocess.call(["open", path])
                            else:
                                subprocess.call(["xdg-open", path])

                        btn_browse.click(browse_folder_action, inputs=None, outputs=output_dir_input)
                        btn_open.click(open_folder_action, inputs=[output_dir_input], outputs=None)

                process_event = btn_submit.click(
                    process_batch_task, 
                    inputs=[
                        file_input, model_sel, lang_sel, 
                        enable_diar, min_spk, max_spk, 
                        vad_onset_slider, prompt_input, compute_type_sel, 
                        enable_demucs, 
                        release_mem, export_srt,
                        output_dir_input,
                        hallucination_mode_radio, hallucination_threshold_slider,
                        llm_enabled_cb, llm_mode_sel, llm_target_lang_input,
                    ], 
                    outputs=[text_out], 
                    show_progress="full" 
                )
                btn_stop.click(fn=None, inputs=None, outputs=None, cancels=[process_event])

            # ============================================================
            #  Tab 2: 语料库构建控制台
            # ============================================================
            with gr.TabItem("语料库构建"):
                gr.Markdown(
                    "### 语料库构建控制台\n"
                    "从音视频批量构建高质量结构化语料库（JSONL 格式），内置六项数据清洗规则与高级特征计算。"
                )
                with gr.Row():
                    # ---------- 左侧：输入与配置 ----------
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("数据输入"):
                                gr.Markdown(
                                    "**方式一：拖拽上传**（适合少量或文件）"
                                )
                                corpus_file_input = gr.File(
                                    label="拖拽上传音视频",
                                    type="filepath",
                                    file_types=["audio", "video"],
                                    file_count="multiple",
                                    height=80,
                                )
                                corpus_audio_prefix = gr.Textbox(
                                    label="音频路径前缀 (仅拖拽上传时生效)",
                                    placeholder="例如: E:\\音频\\example",
                                    info="拖拽上传时 Gradio 无法获取原始路径。设此项后 JSONL 中 audio_path = 前缀/文件名。路径输入时自动忽略。",
                                    max_lines=1,
                                )
                                gr.Markdown("---")
                                gr.Markdown(
                                    "**方式二：路径输入**（支持文件和文件夹混合输入，每行一个路径）"
                                )
                                corpus_path_input = gr.Textbox(
                                    label="文件/文件夹路径",
                                    placeholder="每行一个路径，支持混合输入：\n"
                                                "E:\\音频\\example.mp3\n"
                                                "E:\\音频\\example\\（输入文件夹将自动递归扫描）",
                                    info="可同时输入文件路径和文件夹路径。文件夹会递归扫描所有音视频。支持中文路径和特殊字符。",
                                    lines=3,
                                    max_lines=10,
                                )
                            
                            with gr.TabItem("ASR 配置"):
                                corpus_model_sel = gr.Dropdown(
                                    ["base", "small", "medium", "large-v2", "large-v3"],
                                    value="large-v3",
                                    label="Whisper 模型",
                                )
                                corpus_lang_sel = gr.Dropdown(
                                    [None, "zh", "en", "ja", "ko"],
                                    value=None,
                                    label="源语言",
                                )
                                corpus_compute_sel = gr.Dropdown(
                                    ["int8", "float16", "float32"],
                                    value="float16",
                                    label="量化精度",
                                )
                                corpus_vad_slider = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.35, step=0.05,
                                    label="VAD 阈值",
                                )
                                corpus_prompt = gr.Textbox(
                                    label="提示词 (Prompt)",
                                    placeholder="留空时，中文语音将自动注入标点引导 prompt",
                                    info="选择语言后会自动填入对应提示词，也可手动修改。中文场景留空即可自动添加标点引导。",
                                    lines=2,
                                )

                                corpus_lang_sel.change(
                                    fn=lambda lang: LANG_PROMPT_MAP.get(lang, ""),
                                    inputs=[corpus_lang_sel],
                                    outputs=[corpus_prompt],
                                )
                                corpus_demucs = gr.Checkbox(
                                    label="启用人声分离 (BS-RoFormer)", value=False,
                                )
                                with gr.Row():
                                    corpus_min_spk = gr.Number(label="最小说话人数", value=1, precision=0)
                                    corpus_max_spk = gr.Number(label="最大说话人数", value=6, precision=0)

                            with gr.TabItem("清洗规则"):
                                gr.Markdown("**逐条启用/禁用数据清洗规则，并调节对应阈值。**")
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 A: 置信度过滤**")
                                rule_a_cb = gr.Checkbox(label="启用", value=True)
                                rule_a_threshold = gr.Slider(
                                    minimum=0.3, maximum=1.0, value=0.6, step=0.05,
                                    label="置信度阈值",
                                    info="低于此值的句子将被丢弃。语料库构建推荐 0.5-0.7（比字幕更宽松）。",
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 B: 重叠音过滤**")
                                rule_b_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="检测多人同时说话的片段，仅移除较短的一方（保留主要内容）。",
                                )
                                rule_b_min_overlap = gr.Slider(
                                    minimum=0.1, maximum=3.0, value=0.5, step=0.1,
                                    label="最小重叠时长 (秒)",
                                    info="重叠 < 此值视为对齐误差忽略。播客/访谈推荐 0.5-1.0。",
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 C: 长度匹配度**")
                                rule_c_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="音频时长与文本字数比例极度失调（吞音/乱码）时判定为噪音。",
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 D: 上下文孤岛过滤**")
                                rule_d_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="移除在时间窗口内无其他说话人出现的孤立片段。单人播客/全文仅一人时自动跳过。",
                                )
                                rule_d_window = gr.Slider(
                                    minimum=30, maximum=600, value=120, step=30,
                                    label="孤岛检测窗口 (秒)",
                                    info="在前后各 N 秒内查找不同说话人，找不到则视为孤岛。窗口越大越宽松。",
                                )
                                rule_d_min_segments = gr.Slider(
                                    minimum=2, maximum=20, value=6, step=1,
                                    label="安全阈值 (最少段数)",
                                    info="清洗后剩余段数 ≤ 此值时自动跳过规则D，防止清空。",
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 E: 低信息量剔除**")
                                rule_e_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="剔除「嗯」「啊」「对对对」等无意义单字附和。",
                                )
                                rule_e_duration = gr.Slider(
                                    minimum=0.5, maximum=5.0, value=1.5, step=0.5,
                                    label="最大持续时间 (秒)",
                                )
                                rule_e_chars = gr.Slider(
                                    minimum=1, maximum=10, value=4, step=1,
                                    label="最大字符数",
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 F: 信噪比检测**")
                                rule_f_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="基于音频信噪比(SNR)过滤低质量片段。启用后会额外加载一次音频进行分析。",
                                )
                                rule_f_threshold = gr.Slider(
                                    minimum=3, maximum=30, value=10, step=1,
                                    label="最低 SNR (dB)",
                                    info="低于此信噪比的片段将被丢弃。推荐 8-15 dB。",
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("**规则 G: 黑名单/脱轨内容过滤**")
                                rule_g_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="基于关键词识别广告口播、品牌推广、引流话术等“数据毒药”，避免污染训练语料。",
                                )
                                rule_g_blacklist_path = gr.Textbox(
                                    label="黑名单词库路径",
                                    placeholder="留空使用默认词库 (config/blacklist_words.txt)",
                                    info="每行一个关键词，# 开头为注释。可自定义针对特定领域的过滤词库。",
                                    max_lines=1,
                                )
                                rule_g_context_purge = gr.Checkbox(
                                    label="连带清除相邻上下文", value=True,
                                    info="命中黑名单后，将同一 conversation_id 的相邻句子一并丢弃，防止广告语境污染。",
                                )

                                gr.Markdown("---")
                                gr.Markdown("**规则 H: LLM 语义校验（API）**")
                                rule_h_cb = gr.Checkbox(
                                    label="启用", value=True,
                                    info="通过大语言模型进行深度语义清洗：软广识别、情感丰富度评分、幻觉检测。需在 config/llm_config.json 中配置 API。",
                                )
                                rule_h_emotion_threshold = gr.Slider(
                                    minimum=1, maximum=5, value=1, step=1,
                                    label="情感得分阈值",
                                    info="LLM 对每组对话评分 1-5 分，低于此值的组将被丢弃。1=最宽松(几乎不过滤)，5=最严格(仅保留强共情内容)。",
                                )
                                rule_h_concurrency = gr.Slider(
                                    minimum=1, maximum=10, value=3, step=1,
                                    label="API 并发数",
                                    info="同时发送的 API 请求数量。提高并发可加速处理，但需注意 API 速率限制。",
                                )
                                gr.Markdown("---")
                                gr.Markdown("**LLM 深度接管模式（消融实验）**")
                                rule_llm_override = gr.Checkbox(
                                    label="启用 LLM 深度接管",
                                    value=True,
                                    info="将规则 C/D/E/G 降级为软标签，由 LLM 做最终裁决。需同时启用规则 H。用于消融实验对比。",
                                )

                            with gr.TabItem("数据物化与可视化"):
                                gr.Markdown("**阶段一：语料物化（JSONL → WAV 切片）**")
                                corpus_enable_materialize = gr.Checkbox(
                                    label="处理完成后自动切片导出数据集",
                                    value=True,
                                    info="读取最终数据集 JSONL，按时间戳切割为标准 wav 片段并生成 metadata.jsonl。",
                                )
                                corpus_materialize_out = gr.Textbox(
                                    label="物化输出目录",
                                    value="{auto}",
                                    placeholder="留空自动使用 项目输出目录/dataset",
                                    max_lines=1,
                                )
                                corpus_materialize_max = gr.Number(
                                    label="物化条数上限（0=全部）",
                                    value=0,
                                    precision=0,
                                )

                                gr.Markdown("---")
                                gr.Markdown("**阶段二：可视化分析（论文图表）**")
                                corpus_enable_analysis = gr.Checkbox(
                                    label="处理完成后自动生成图表",
                                    value=True,
                                    info="生成多种学术质量的可视化图表。",
                                )
                                corpus_analysis_out = gr.Textbox(
                                    label="图表输出目录",
                                    value="{auto}",
                                    placeholder="留空自动使用 项目输出目录/analysis",
                                    max_lines=1,
                                )
                                
                                gr.Markdown("**图表生成选项：**")
                                with gr.Row():
                                    corpus_chart_funnel = gr.Checkbox(
                                        label="① 清洗漏斗图", value=True, scale=1
                                    )
                                    corpus_chart_speed = gr.Checkbox(
                                        label="② 语速分布图", value=True, scale=1
                                    )
                                    corpus_chart_delay = gr.Checkbox(
                                        label="③ 交互延迟散点图", value=True, scale=1
                                    )
                                
                                with gr.Row():
                                    corpus_chart_robustness = gr.Checkbox(
                                        label="④ 多源鲁棒性对比", value=True, scale=1
                                    )
                                    corpus_chart_llm_analysis = gr.Checkbox(
                                        label="⑤ LLM语义分析图", value=True, scale=1,
                                        info="需启用规则H后才有数据"
                                    )

                        with gr.Accordion("输出与资源", open=True):
                            corpus_release_mem = gr.Checkbox(
                                label="完成后释放显存", value=True,
                            )
                            corpus_resume_enabled = gr.Checkbox(
                                label="启用断点续跑",
                                value=True,
                                info="中断/报错后会写入 progress_snapshot.json；再次启动时可从匹配任务继续。",
                            )

                        with gr.Row():
                            corpus_btn_start = gr.Button("开始构建语料库", variant="primary", scale=2)
                            corpus_btn_stop = gr.Button("终止", variant="stop", scale=1)

                    # ---------- 右侧：进度面板 ----------
                    with gr.Column(scale=2):
                        corpus_gpu_monitor = GPUMonitor(
                            update_interval=1000,
                            show_last_updated=True,
                            label="GPU监视器",
                        )

                        corpus_progress_panel = gr.HTML(
                            label="构建进度",
                            value=_render_corpus_progress_panel(
                                total_files=0,
                                current_index=0,
                                current_file="",
                                current_file_progress=0.0,
                                completed_files=0,
                                current_step="等待开始...",
                                elapsed_seconds=0.0,
                                eta_seconds=-1.0,
                                success_files=0,
                                failed_files=0,
                            ),
                        )
                        
                        with gr.Row(variant="panel"):
                            corpus_output_dir = gr.Textbox(
                                label="项目输出目录（主文件夹）",
                                value=str(OUTPUT_DIR / "corpus_work"),
                                info="包含 JSONL、日志、dataset、analysis 等所有输出的统一项目文件夹",
                                scale=5,
                                max_lines=1,
                                interactive=True,
                            )
                            corpus_btn_browse = gr.Button("选择文件夹", scale=1)
                            corpus_btn_open = gr.Button("打开", scale=1, variant="secondary")

                        def corpus_browse():
                            try:
                                import tkinter as tk
                                from tkinter import filedialog
                                root = tk.Tk()
                                root.withdraw()
                                root.attributes('-topmost', True)
                                selected = filedialog.askdirectory()
                                root.destroy()
                                return str(Path(selected)) if selected else gr.update()
                            except Exception:
                                return gr.update()

                        def corpus_open(p):
                            path = p.strip() if p else str(OUTPUT_DIR / "corpus_work")
                            Path(path).mkdir(parents=True, exist_ok=True)
                            if sys.platform == "win32":
                                os.startfile(path)
                            elif sys.platform == "darwin":
                                subprocess.call(["open", path])
                            else:
                                subprocess.call(["xdg-open", path])

                        corpus_btn_browse.click(corpus_browse, outputs=corpus_output_dir)
                        corpus_btn_open.click(corpus_open, inputs=[corpus_output_dir])

                def corpus_stop_action():
                    _signal_active_corpus_stop()
                    return _render_corpus_progress_panel(
                        total_files=1,
                        current_index=0,
                        current_file="等待当前步骤终止",
                        current_file_progress=0.0,
                        completed_files=0,
                        current_step="用户已停止，正在安全终止当前任务...",
                        elapsed_seconds=0.0,
                        eta_seconds=-1.0,
                        success_files=0,
                        failed_files=0,
                        status_note="用户已停止",
                    )

                corpus_event = corpus_btn_start.click(
                    process_corpus_task,
                    inputs=[
                        corpus_file_input, corpus_path_input,
                        corpus_model_sel, corpus_lang_sel,
                        corpus_min_spk, corpus_max_spk,
                        corpus_vad_slider, corpus_prompt, corpus_compute_sel, corpus_demucs,
                        corpus_release_mem,
                        corpus_output_dir,
                        corpus_resume_enabled,
                        corpus_audio_prefix,
                        # 清洗规则参数
                        rule_a_cb, rule_a_threshold,
                        rule_b_cb, rule_b_min_overlap,
                        rule_c_cb,
                        rule_d_cb, rule_d_window, rule_d_min_segments,
                        rule_e_cb, rule_e_duration, rule_e_chars,
                        rule_f_cb, rule_f_threshold,
                        rule_g_cb, rule_g_blacklist_path, rule_g_context_purge,
                        rule_h_cb, rule_h_emotion_threshold, rule_h_concurrency, rule_llm_override,
                        # 数据集交付与可视化
                        corpus_enable_materialize, corpus_materialize_out, corpus_materialize_max,
                        corpus_enable_analysis, corpus_analysis_out,
                        # 图表生成选项
                        corpus_chart_funnel, corpus_chart_speed,
                        corpus_chart_delay, corpus_chart_robustness,
                        corpus_chart_llm_analysis,
                    ],
                    outputs=[corpus_progress_panel],
                    show_progress="hidden",
                )
                corpus_btn_stop.click(
                    fn=corpus_stop_action,
                    inputs=None,
                    outputs=[corpus_progress_panel],
                    queue=False,
                    show_progress="hidden",
                )
    
    return app