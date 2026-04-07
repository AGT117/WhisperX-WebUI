import json
import time
import os
import sys
import subprocess
import threading
import logging
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

# 引擎实例化
engine = FullPipelineEngine()
# 并发保护锁：防止多个 Gradio 请求同时操作模型状态
_engine_lock = threading.Lock()

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


def process_corpus_task(
    file_paths, path_input,
    model_size, lang,
    min_spk, max_spk,
    vad_onset, initial_prompt, compute_type, enable_demucs,
    release_memory,
    custom_output_path,
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
    enable_analysis, analysis_output_dir, analysis_font_path, analysis_with_rule_g_wordcloud,
    # 图表生成选项
    chart_quality_report, chart_funnel, chart_speed, chart_delay, chart_poison, chart_robustness,
    chart_llm_analysis,
    progress=gr.Progress(),
):
    """语料库构建批处理回调函数"""
    
    AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac',
                  '.mp4', '.mkv', '.avi', '.mov', '.webm'}
    
    def _strip_quotes(s: str) -> str:
        """去除路径首尾的各种引号（ASCII 双引号、单引号、中文引号等）"""
        s = s.strip()
        quote_pairs = [('"', '"'), ("'", "'"), ('\u201c', '\u201d'), ('\u2018', '\u2019'),
                       ('\u300c', '\u300d'), ('\uff02', '\uff02')]
        for lq, rq in quote_pairs:
            if s.startswith(lq) and s.endswith(rq) and len(s) >= 2:
                s = s[len(lq):-len(rq)].strip()
        return s
    
    # 收集所有待处理文件
    all_files = []       # (file_path, original_path) 元组列表
    
    # 从文件上传获取（Gradio 上传会产生临时路径，无法恢复原始磁盘路径）
    if file_paths:
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        for fp in file_paths:
            all_files.append((fp, None))
    
    # 从路径文本框获取：支持文件路径、文件夹路径，每行一个
    if path_input and path_input.strip():
        for raw_line in path_input.strip().splitlines():
            line = _strip_quotes(raw_line)
            if not line:
                continue
            p = Path(line)
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                # 单文件路径：保留完整原始路径
                all_files.append((str(p), str(p)))
            elif p.is_dir():
                # 文件夹路径：递归扫描
                for f in sorted(p.rglob('*')):
                    if f.suffix.lower() in AUDIO_EXTS:
                        all_files.append((str(f), str(f)))
            else:
                # 路径不存在或格式不支持
                logger.warning(f"[路径跳过] 无效或不支持: {line}")
    
    if not all_files:
        yield "警告: 未找到可处理的音视频文件。\n请检查：\n  1. 路径是否存在（支持文件或文件夹）\n  2. 文件格式是否为 mp3/wav/flac/m4a/mp4 等", ""
        return
    
    # 输出路径（带时间戳隔离）
    try:
        if custom_output_path and custom_output_path.strip():
            base_dir = Path(custom_output_path.strip())
        else:
            base_dir = OUTPUT_DIR / "corpus_work"
        
        # 生成时间戳文件夹：run_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = base_dir / f"run_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        yield f"错误: 输出路径无效 ({str(e)})", ""
        return

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

    total_files = len(all_files)
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
    removed_g_records = []
    all_removed_segments = []  # 所有规则移除的段落汇总

    for i, (file_path, original_path) in enumerate(all_files):
        current_index = i + 1
        input_path = Path(file_path)
        file_stem = input_path.stem
        # audio_display_name: 用于 JSONL 的 audio_path 字段
        # 路径输入来源 → 完整原始路径
        # 拖拽上传来源 → 有前缀时智能拼接，否则仅文件名
        if original_path:
            audio_display_name = original_path
        elif audio_path_prefix and audio_path_prefix.strip():
            prefix_clean = _strip_quotes(audio_path_prefix)
            prefix_path = Path(prefix_clean)
            # 智能判断：如果前缀是一个文件路径（带音频/视频后缀），取其父目录
            if prefix_path.suffix.lower() in AUDIO_EXTS:
                prefix_path = prefix_path.parent
            audio_display_name = str(prefix_path / input_path.name)
        else:
            audio_display_name = input_path.name

        file_base = i / total_files
        file_range = 1.0 / total_files

        def make_progress_callback(base, rng, idx, total, stem):
            def _cb(fraction, desc):
                overall = base + fraction * rng
                progress(overall, desc=f"[{idx}/{total}] {stem}: {desc}")
            return _cb

        progress_callback = make_progress_callback(
            file_base, file_range, current_index, total_files, file_stem
        )

        mode_info = "[人声分离]" if enable_demucs else "[标准]"
        status_msg = f"[{current_index}/{total_files}] 正在处理: {file_stem} {mode_info}"
        yield status_msg + "\n" + log_buffer, "处理中..."

        start_time = time.time()

        try:
            with _engine_lock:
                enriched, stats, status = engine.run_corpus_pipeline(
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
                    metadata={'source_file': input_path.name},
                    audio_display_name=audio_display_name,
                    progress_callback=progress_callback,
                )

            if "Success" not in status:
                raise RuntimeError(status)

            # 记录统计
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
            total_stats['success_files'] += 1

            # 采集规则 G 被移除样本（用于后续词云）
            for seg in stats.get('rule_g_removed_samples', []):
                removed_g_records.append({
                    'text': seg.get('text', ''),
                    'speaker_id': seg.get('speaker', ''),
                    'start': seg.get('start', 0.0),
                    'end': seg.get('end', 0.0),
                    'duration': seg.get('duration', 0.0),
                    'conversation_id': seg.get('conversation_id', None),
                    'metadata': {'source_file': input_path.name},
                })

            # 采集所有被清洗规则移除的段落（汇总导出用）
            for seg in stats.get('_removed_segments', []):
                all_removed_segments.append({
                    'text': seg.get('text', ''),
                    'speaker_id': seg.get('speaker', ''),
                    'start': seg.get('start', 0.0),
                    'end': seg.get('end', 0.0),
                    'duration': seg.get('end', 0.0) - seg.get('start', 0.0),
                    'removed_by': seg.get('_removed_by', 'unknown'),
                    'reason': seg.get('_reason', ''),
                    'source_file': input_path.name,
                })

            jsonl_path = save_dir / f"{file_stem}.jsonl"
            if jsonl_path.exists():
                jsonl_files.append(str(jsonl_path))

            duration = time.time() - start_time
            removed = stats.get('input_count', 0) - stats.get('output_count', 0)
            file_log = (
                f"✓ [{current_index}/{total_files}] {file_stem}  ({duration:.1f}s)\n"
                f"  保留 {stats.get('output_count',0)} 段 / 移除 {removed} 段\n"
            )
            log_buffer = file_log + log_buffer
            yield log_buffer, "处理中..."

        except Exception as e:
            total_stats['failed_files'] += 1
            error_log = (
                f"✗ [{current_index}/{total_files}] {file_stem}\n"
                f"  失败原因: {str(e)}\n"
            )
            log_buffer = error_log + log_buffer
            logger.error(f"Corpus Task Failed: {e}")
            yield log_buffer, "处理中..."

    # 合并所有 JSONL 为最终数据集（仅多文件时生成合并文件）
    final_dataset = None
    if len(jsonl_files) >= 2:
        merged_path = save_dir / "dataset_merged.jsonl"
        merge_jsonl_files(jsonl_files, str(merged_path))
        final_dataset = "dataset_merged.jsonl"
    elif len(jsonl_files) == 1:
        final_dataset = Path(jsonl_files[0]).name

    # 生成统计报告
    stats_report = _format_corpus_stats(total_stats, total_files, str(save_dir), final_dataset)

    # 写入质量日志文件（可供分析脚本直接读取）
    quality_log_path = save_dir / "日志.txt"
    try:
        with open(quality_log_path, "w", encoding="utf-8") as f:
            f.write(log_buffer)
            f.write("\n")
            f.write(stats_report)
    except Exception as e:
        logger.warning(f"写入质量日志失败: {e}")

    # 可选：导出规则 G 拦截样本
    removed_g_path = None
    if analysis_with_rule_g_wordcloud and removed_g_records:
        removed_g_path = save_dir / "rule_g_removed.jsonl"
        try:
            with open(removed_g_path, "w", encoding="utf-8") as f:
                for item in removed_g_records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"导出规则G拦截样本失败: {e}")

    # 导出所有被清洗掉的段落汇总（方便查阅清洗内容）
    if all_removed_segments:
        removed_all_path = save_dir / "removed_segments.jsonl"
        try:
            with open(removed_all_path, "w", encoding="utf-8") as f:
                for item in all_removed_segments:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"[导出] 被清洗段落汇总: {len(all_removed_segments)} 条 → {removed_all_path}")
        except Exception as e:
            logger.warning(f"导出清洗段落汇总失败: {e}")

    # 可选：数据集物化（切片导出 WAV）
    post_logs = []
    dataset_jsonl_path = (save_dir / final_dataset) if final_dataset else None
    if enable_materialize and dataset_jsonl_path and dataset_jsonl_path.exists():
        try:
            materialize_script = Path(__file__).resolve().parents[2] / "scripts" / "materialize_corpus.py"
            # 如果物化输出为 {auto} 或为空，自动使用主输出目录/dataset
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

            # 自动收集音频源目录，传递给物化脚本以便查找音频文件
            audio_source_dirs = set()
            for fp, orig in all_files:
                if orig:
                    audio_source_dirs.add(str(Path(orig).parent))
                else:
                    audio_source_dirs.add(str(Path(fp).parent))
            for adir in audio_source_dirs:
                cmd += ["--audio-dir", adir]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                post_logs.append("[后处理] 数据物化完成")
            else:
                post_logs.append(f"[后处理] 数据物化失败: {result.stderr.strip() or result.stdout.strip()}")
        except Exception as e:
            post_logs.append(f"[后处理] 数据物化异常: {e}")

    # 可选：可视化分析（漏斗图/语速直方图/词云）
    if enable_analysis and dataset_jsonl_path and dataset_jsonl_path.exists():
        try:
            analysis_script = Path(__file__).resolve().parents[2] / "scripts" / "analyze_corpus.py"
            # 如果分析输出为 {auto} 或为空，自动使用主输出目录/analysis
            ana_out = analysis_output_dir.strip() if analysis_output_dir and analysis_output_dir.strip() not in ("{auto}", "") else str(save_dir / "analysis")
            cmd = [
                sys.executable,
                str(analysis_script),
                "--merged-jsonl", str(dataset_jsonl_path),
                "--quality-log", str(quality_log_path),
                "--output-dir", ana_out,
            ]
            
            # 添加图表生成选项
            if chart_quality_report:
                cmd.append("--enable-quality-report")
            if chart_funnel:
                cmd.append("--enable-funnel")
            if chart_speed:
                cmd.append("--enable-speed-histogram")
            if chart_delay:
                cmd.append("--enable-interaction-delay")
            if chart_poison:
                cmd.append("--enable-poison-keywords")
            if chart_robustness:
                cmd.append("--enable-robustness")
            if analysis_with_rule_g_wordcloud:
                cmd.append("--enable-wordcloud")
            if chart_llm_analysis:
                cmd.append("--enable-llm-analysis")
            
            if analysis_with_rule_g_wordcloud and removed_g_path and removed_g_path.exists():
                cmd += ["--removed-g-jsonl", str(removed_g_path)]
            if analysis_font_path and analysis_font_path.strip():
                cmd += ["--font-path", analysis_font_path.strip()]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                post_logs.append("[后处理] 可视化分析完成")
            else:
                post_logs.append(f"[后处理] 可视化分析失败: {result.stderr.strip() or result.stdout.strip()}")
        except Exception as e:
            post_logs.append(f"[后处理] 可视化分析异常: {e}")

    if release_memory:
        progress(0.95, desc="释放显存...")
        with _engine_lock:
            engine.unload_all()

    progress(1.0, desc="语料库构建完成")
    final_log = f"全部 {total_files} 个文件处理完毕。\n" + log_buffer
    if post_logs:
        final_log = final_log + "\n" + "\n".join(post_logs) + "\n"
    yield final_log, stats_report


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
    lines.append(f"│ 输出目录: {output_dir}")
    if final_dataset:
        lines.append(f"│ 数据集文件: {final_dataset}")
    lines.append("└────────────────────────────────────────┘")
    return "\n".join(lines)

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
                                    label="启用", value=False,
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
                                    label="连带清除相邻上下文", value=False,
                                    info="命中黑名单后，将同一 conversation_id 的相邻句子一并丢弃，防止广告语境污染。",
                                )

                                gr.Markdown("---")
                                gr.Markdown("**规则 H: LLM 语义校验（API）**")
                                rule_h_cb = gr.Checkbox(
                                    label="启用", value=False,
                                    info="通过大语言模型进行深度语义清洗：软广识别、情感丰富度评分、幻觉检测。需在 config/llm_config.json 中配置 API。",
                                )
                                rule_h_emotion_threshold = gr.Slider(
                                    minimum=1, maximum=5, value=3, step=1,
                                    label="情感得分阈值",
                                    info="LLM 对每组对话评分 1-5 分，低于此值的组将被丢弃。1=最宽松(几乎不过滤)，5=最严格(仅保留强共情内容)。推荐 2-3。",
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
                                    value=False,
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
                                corpus_analysis_font = gr.Textbox(
                                    label="中文字体路径（可选）",
                                    value="C:/Windows/Fonts/msyh.ttc",
                                    max_lines=1,
                                )
                                
                                gr.Markdown("**图表生成选项：**")
                                with gr.Row():
                                    corpus_chart_quality_report = gr.Checkbox(
                                        label="① 质量报告", value=True, scale=1
                                    )
                                    corpus_chart_funnel = gr.Checkbox(
                                        label="② 清洗漏斗图", value=True, scale=1
                                    )
                                    corpus_chart_speed = gr.Checkbox(
                                        label="③ 语速分布图", value=True, scale=1
                                    )
                                
                                with gr.Row():
                                    corpus_chart_delay = gr.Checkbox(
                                        label="④ 交互延迟散点图", value=False, scale=1
                                    )
                                    corpus_chart_poison = gr.Checkbox(
                                        label="⑤ 毒药关键词排行", value=False, scale=1
                                    )
                                    corpus_chart_robustness = gr.Checkbox(
                                        label="⑥ 多源鲁棒性对比", value=False, scale=1
                                    )
                                
                                corpus_analysis_wordcloud = gr.Checkbox(
                                    label="⑦ 规则G词云", value=False,
                                    info="需要提供中文字体路径"
                                )
                                corpus_chart_llm_analysis = gr.Checkbox(
                                    label="⑧ LLM语义分析图", value=False,
                                    info="需启用规则H后才有数据"
                                )

                        with gr.Accordion("输出与资源", open=True):
                            corpus_release_mem = gr.Checkbox(
                                label="完成后释放显存", value=True,
                            )

                        with gr.Row():
                            corpus_btn_start = gr.Button("开始构建语料库", variant="primary", scale=2)
                            corpus_btn_stop = gr.Button("终止", variant="stop", scale=1)

                    # ---------- 右侧：日志与统计 ----------
                    with gr.Column(scale=2):
                        corpus_log = gr.TextArea(
                            label="构建日志",
                            lines=22,
                            placeholder="语料库构建进度与日志...",
                        )
                        corpus_stats_box = gr.TextArea(
                            label="清洗统计报告",
                            lines=16,
                            placeholder="构建完成后将在此显示统计...",
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

                corpus_event = corpus_btn_start.click(
                    process_corpus_task,
                    inputs=[
                        corpus_file_input, corpus_path_input,
                        corpus_model_sel, corpus_lang_sel,
                        corpus_min_spk, corpus_max_spk,
                        corpus_vad_slider, corpus_prompt, corpus_compute_sel, corpus_demucs,
                        corpus_release_mem,
                        corpus_output_dir,
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
                        corpus_enable_analysis, corpus_analysis_out, corpus_analysis_font, corpus_analysis_wordcloud,
                        # 图表生成选项
                        corpus_chart_quality_report, corpus_chart_funnel, corpus_chart_speed,
                        corpus_chart_delay, corpus_chart_poison, corpus_chart_robustness,
                        corpus_chart_llm_analysis,
                    ],
                    outputs=[corpus_log, corpus_stats_box],
                    show_progress="full",
                )
                corpus_btn_stop.click(fn=None, cancels=[corpus_event])
    
    return app