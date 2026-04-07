#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_name(value: str) -> str:
    text = value.strip() if value else "UNKNOWN"
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    return text[:48] if text else "UNKNOWN"


def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _cut_with_ffmpeg(audio_path: Path, start: float, end: float, output_path: Path) -> Tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffmpeg failed"
    return True, ""


def _resolve_audio_path(audio_path_value: str, jsonl_dir: Path,
                        extra_dirs: Optional[List[Path]] = None) -> Path:
    """
    多策略解析音频路径，依次尝试：
    1. 原始路径（绝对路径直接使用）
    2. resolve() 解析后的路径（消除符号链接等）
    3. 相对于 JSONL 所在目录的路径
    4. 在用户指定的额外目录中按文件名搜索
    5. 在 JSONL 所在目录中按文件名递归搜索（最多向上2层）
    """
    if not audio_path_value:
        return Path(audio_path_value)

    # 策略 1: 原始路径
    p = Path(audio_path_value)
    if p.exists():
        return p

    # 策略 2: resolve
    try:
        resolved = p.resolve()
        if resolved.exists():
            return resolved
    except Exception:
        pass

    filename = p.name

    # 策略 3: 相对于 JSONL 目录
    relative_try = jsonl_dir / filename
    if relative_try.exists():
        return relative_try

    # 策略 4: 在用户指定的额外搜索目录中查找
    if extra_dirs:
        for d in extra_dirs:
            candidate = d / filename
            if candidate.exists():
                return candidate
        # 在额外目录中递归搜索
        for d in extra_dirs:
            if d.is_dir():
                for found in d.rglob(filename):
                    if found.is_file():
                        return found

    # 策略 5: 在 JSONL 目录的上级目录中按文件名搜索（最多向上2层）
    search_root = jsonl_dir
    for _ in range(3):
        for found in search_root.rglob(filename):
            if found.is_file():
                return found
        search_root = search_root.parent

    # 全部失败，返回原始路径（调用方会报 not found）
    return p


def materialize_dataset(input_jsonl: Path, output_dir: Path, max_clips: int = 0,
                        audio_dirs: Optional[List[str]] = None) -> None:
    records = _load_jsonl(input_jsonl)
    if not records:
        raise RuntimeError(f"输入文件为空: {input_jsonl}")

    if max_clips > 0:
        records = records[:max_clips]

    jsonl_dir = input_jsonl.resolve().parent

    # 构建额外搜索目录列表
    extra_dirs: List[Path] = []
    if audio_dirs:
        for d in audio_dirs:
            dp = Path(d)
            if dp.is_dir():
                extra_dirs.append(dp)
            elif dp.is_file():
                extra_dirs.append(dp.parent)
    # 自动从记录中提取 audio_path 的父目录作为候选搜索路径
    seen_parents = set()
    for rec in records:
        ap = rec.get("audio_path", "")
        if ap:
            parent = Path(ap).parent
            parent_str = str(parent)
            if parent_str not in seen_parents:
                seen_parents.add(parent_str)
                if parent.is_dir() and parent not in extra_dirs:
                    extra_dirs.append(parent)
    dataset_root = output_dir
    audio_dir = dataset_root / "audio"
    metadata_path = dataset_root / "metadata.jsonl"
    fail_path = dataset_root / "failed_segments.jsonl"

    dataset_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    
    # 为每个源文件维护独立的计数器
    source_file_counters = {}

    with metadata_path.open("w", encoding="utf-8") as metadata_file, fail_path.open("w", encoding="utf-8") as fail_file:
        for index, record in enumerate(records, start=1):
            audio_path_value = record.get("audio_path", "")
            source_audio = _resolve_audio_path(audio_path_value, jsonl_dir, extra_dirs)
            speaker_id = _safe_name(record.get("speaker_id", "SPEAKER"))
            
            # 从 metadata 中提取源文件名，用于为切片命名
            source_filename = ""
            if "metadata" in record and isinstance(record["metadata"], dict):
                source_filename = record["metadata"].get("source_file", "")
            if not source_filename and audio_path_value:
                source_filename = Path(audio_path_value).stem  # 如果没有 metadata，从 audio_path 推断
            
            # 为每个源文件单独计数
            if source_filename not in source_file_counters:
                source_file_counters[source_filename] = 0
            source_file_counters[source_filename] += 1
            source_file_seq = source_file_counters[source_filename]

            start = float(record.get("start", 0.0))
            end = float(record.get("end", 0.0))
            if end <= start:
                fail_count += 1
                fail_payload = {
                    "index": index,
                    "reason": "invalid_time_range",
                    "start": start,
                    "end": end,
                    "record": record,
                }
                fail_file.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
                continue

            # 切片文件名格式：{源文件名}_{说话人}_{源文件内序号}.wav
            clip_name = f"{source_filename}_{speaker_id}_{source_file_seq:05d}.wav" if source_filename else f"{speaker_id}_seg{source_file_seq:05d}.wav"
            clip_path = audio_dir / clip_name

            if not source_audio.exists():
                fail_count += 1
                fail_payload = {
                    "index": index,
                    "reason": "source_audio_not_found",
                    "audio_path": audio_path_value,
                    "record": record,
                }
                fail_file.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
                continue

            ok, error_message = _cut_with_ffmpeg(source_audio, start, end, clip_path)
            if not ok:
                fail_count += 1
                fail_payload = {
                    "index": index,
                    "reason": "ffmpeg_cut_failed",
                    "error": error_message,
                    "audio_path": audio_path_value,
                    "record": record,
                }
                fail_file.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
                continue

            # 白名单构建最终训练记录，排除绝对路径和调试字段
            output_record = {
                "clip_path": str(Path("audio") / clip_name),
                "text": record.get("text", ""),
                "speaker_id": record.get("speaker_id", ""),
                "start": start,
                "end": end,
                "duration": record.get("duration", round(end - start, 3)),
                "words_per_second": record.get("words_per_second"),
                "turn_delay": record.get("turn_delay"),
                "interaction_type": record.get("interaction_type", "normal"),
                "conversation_id": record.get("conversation_id", 0),
                "contains_overlap": record.get("contains_overlap", False),
                "confidence_score": record.get("confidence_score", 0.0),
            }
            # LLM 语义标签
            if "emotion_score" in record:
                output_record["emotion_score"] = record["emotion_score"]
                output_record["is_ad"] = record.get("is_ad", False)
                output_record["hallucination_risk"] = record.get("hallucination_risk", False)
                output_record["llm_status"] = record.get("llm_status", "unchecked")
            # 仅保留 source_file 的元数据
            if isinstance(record.get("metadata"), dict):
                output_record["metadata"] = {
                    "source_file": record["metadata"].get("source_file", "")
                }
            metadata_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            success_count += 1

    print(f"[完成] 输入段数: {len(records)}")
    print(f"[完成] 成功切片: {success_count}")
    print(f"[完成] 失败段数: {fail_count}")
    print(f"[输出] 元数据: {metadata_path}")
    print(f"[输出] 音频目录: {audio_dir}")
    if fail_count > 0:
        print(f"[输出] 失败明细: {fail_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 dataset_merged.jsonl 物化为标准音频数据集")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="outputs/corpus/dataset_merged.jsonl",
        help="输入 JSONL 路径（默认 outputs/corpus/dataset_merged.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/My_Podcast_Dataset",
        help="输出数据集目录（默认 outputs/My_Podcast_Dataset）",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="仅处理前 N 条（0 表示处理全部）",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        action="append",
        default=None,
        help="额外的音频搜索目录（可多次指定），当 JSONL 中的 audio_path 失效时在此目录中查找",
    )
    args = parser.parse_args()

    materialize_dataset(
        Path(args.input_jsonl), Path(args.output_dir),
        max_clips=args.max_clips, audio_dirs=args.audio_dir,
    )


if __name__ == "__main__":
    main()
