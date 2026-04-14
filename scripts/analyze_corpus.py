#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

# Windows 优先级最高的中文字体配置策略
if sys.platform == "win32":
    # 尝试多个备选字体，按优先级排列
    windows_fonts = [
        'SimHei',  # 黑体（最稳定）
        'Microsoft YaHei',  # 微软雅黑
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
        'SimSun',  # 宋体
    ]
    # 检查哪个字体在系统中可用
    available_font = None
    for font_name in windows_fonts:
        try:
            fm.findfont(fm.FontProperties(family=font_name))
            available_font = font_name
            break
        except Exception:
            continue
    
    if available_font:
        matplotlib.rcParams['font.sans-serif'] = available_font
    else:
        # 如果系统字体都不可用，使用操作系统路径加载
        font_paths = [
            'C:\\Windows\\Fonts\\simhei.ttf',
            'C:\\Windows\\Fonts\\msyh.ttf',
            'C:\\Windows\\Fonts\\msyh.ttc',
        ]
        for font_path in font_paths:
            if Path(font_path).exists():
                fm.addfont(font_path)
                matplotlib.rcParams['font.sans-serif'] = Path(font_path).stem
                break
else:
    # Mac/Linux 字体配置
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 图表输出文件名（中文）
CHART_FILENAMES = {
    "funnel": "清洗漏斗图.png",
    "speed_histogram": "语速分布图.png",
    "interaction_delay": "交互延迟散点图.png",
    "robustness": "多源鲁棒性对比图.png",
    "llm_emotion": "LLM语句情感分析图.png",
}


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_quality_stats(path: Path) -> Optional[Dict]:
    """读取结构化质量统计文件（图表唯一数据源）。"""
    if not path.exists():
        print(f"[跳过] 未找到 quality_stats.json: {path}")
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print(f"[跳过] quality_stats.json 解析失败: {e}")
        return None

    if not isinstance(payload, dict):
        print("[跳过] quality_stats.json 格式错误：根对象必须为 JSON object")
        return None

    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        print("[跳过] quality_stats.json 格式错误：summary 必须为 object")
        return None

    def _safe_int(value, default=0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    stats = {
        "raw": _safe_int(summary.get("raw", 0)),
        "kept": _safe_int(summary.get("kept", 0)),
        "rule_a": _safe_int(summary.get("rule_a", 0)),
        "rule_b": _safe_int(summary.get("rule_b", 0)),
        "rule_c": _safe_int(summary.get("rule_c", 0)),
        "rule_d": _safe_int(summary.get("rule_d", 0)),
        "rule_e": _safe_int(summary.get("rule_e", 0)),
        "rule_f": _safe_int(summary.get("rule_f", 0)),
        "rule_g": _safe_int(summary.get("rule_g", 0)),
        "rule_h": _safe_int(summary.get("rule_h", 0)),
        "success_files": _safe_int(summary.get("success_files", 0)),
        "failed_files": _safe_int(summary.get("failed_files", 0)),
    }

    if stats["raw"] <= 0:
        print("[跳过] quality_stats.json 缺少有效 raw 统计")
        return None

    raw_per_file = payload.get("per_file", [])
    per_file = raw_per_file if isinstance(raw_per_file, list) else []
    return {"summary": stats, "per_file": per_file}


def _draw_funnel(stats: Dict[str, int], output_path: Path) -> None:
    stages = ["原始段落"]
    values = [max(stats.get("raw", 0), 0)]

    current = values[0]
    removal_order = [
        ("A后", "rule_a"),
        ("B后", "rule_b"),
        ("C后", "rule_c"),
        ("D后", "rule_d"),
        ("E后", "rule_e"),
        ("F后", "rule_f"),
        ("G后", "rule_g"),
        ("H后", "rule_h"),
    ]
    for label, key in removal_order:
        current = max(current - max(stats.get(key, 0), 0), 0)
        stages.append(label)
        values.append(current)

    # 最终保留值以结构化统计为准
    stages.append("最终保留")
    values.append(max(stats.get("kept", 0), 0))

    fig, axis = plt.subplots(figsize=(12, 7))
    colors = ["#4C78A8"] + ["#72B7B2"] * (len(stages) - 2) + ["#54A24B"]
    bars = axis.bar(range(len(stages)), values, color=colors, edgecolor="black", linewidth=1)

    for idx, bar in enumerate(bars):
        val = values[idx]
        axis.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.015, str(val),
                  ha="center", va="bottom", fontsize=10, weight="bold")

    axis.set_xticks(range(len(stages)))
    axis.set_xticklabels(stages)
    axis.set_ylabel("段落数量", fontsize=12, weight="bold")
    axis.set_title("清洗漏斗图（各阶段段落保留变化）", fontsize=14, weight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    raw = max(stats.get("raw", 0), 0)
    kept = max(stats.get("kept", 0), 0)
    keep_pct = (kept / raw * 100) if raw > 0 else 0
    axis.text(0.5, -0.12, f"原始: {raw} | 最终保留: {kept} ({keep_pct:.1f}%)",
              transform=axis.transAxes, ha="center", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_speed_histogram(rows: List[Dict], output_path: Path) -> bool:
    import numpy as np

    speed_values: List[float] = []
    for row in rows:
        # 兼容新旧字段：优先使用 words_per_second，回退到 speech_rate
        speed = row.get("words_per_second")
        if speed is None:
            speed = row.get("speech_rate")
        if isinstance(speed, (int, float)) and speed > 0:
            speed_values.append(float(speed))

    if not speed_values:
        print("[跳过] 无 words_per_second/speech_rate 数据，无法生成语速分布图")
        return False

    fig, axis = plt.subplots(figsize=(12, 7))
    bins = min(30, max(10, int(len(speed_values) ** 0.5 * 2)))
    axis.hist(speed_values, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.75)

    avg_speed = float(np.mean(speed_values))
    comfort_min, comfort_max = 4.0, 7.0
    axis.axvline(avg_speed, color="red", linestyle="-", linewidth=2.5,
                 label=f"平均语速: {avg_speed:.2f} 字/秒", zorder=3)
    axis.axvline(comfort_min, color="blue", linestyle="--", linewidth=2,
                 label=f"舒适区间下限: {comfort_min} 字/秒", zorder=3)
    axis.axvline(comfort_max, color="blue", linestyle="--", linewidth=2,
                 label=f"舒适区间上限: {comfort_max} 字/秒", zorder=3)
    axis.axvspan(comfort_min, comfort_max, alpha=0.1, color="blue", label="人类自然沟通舒适区")

    comfort_count = sum(1 for v in speed_values if comfort_min <= v <= comfort_max)
    comfort_pct = (comfort_count / len(speed_values) * 100) if speed_values else 0

    axis.set_title("语速分布直方图（人类自然沟通特征分析）", fontsize=14, weight="bold")
    axis.set_xlabel("语速（字/秒）", fontsize=12, weight="bold")
    axis.set_ylabel("段落数量", fontsize=12, weight="bold")
    axis.legend(loc="upper right", fontsize=10)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    stats_info = (
        f"总段落: {len(speed_values)} | 舒适区间内: {comfort_count} ({comfort_pct:.1f}%) | "
        f"最小: {min(speed_values):.2f} | 最大: {max(speed_values):.2f}"
    )
    axis.text(0.5, -0.12, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_interaction_delay(rows: List[Dict], output_path: Path) -> bool:
    """绘制交互延迟散点图（conversation turn-taking分析）"""
    import numpy as np
    
    # 提取 turn_delay 和 interaction_type
    delays = []
    types = []
    indices = []
    
    speaker_ids = set()

    for idx, row in enumerate(rows):
        delay = row.get("turn_delay")
        itype = row.get("interaction_type", "normal")
        speaker = row.get("speaker_id") or row.get("speaker")
        if speaker:
            speaker_ids.add(str(speaker))
        if delay is not None and isinstance(delay, (int, float)):
            delays.append(float(delay))
            types.append(itype if itype else "normal")
            indices.append(idx + 1)
    
    if not delays:
        if rows and len(speaker_ids) == 1:
            print("[跳过] 交互延迟图未生成：检测到单人说话场景，turn_delay 全为 null")
        else:
            print("[跳过] 无有效 turn_delay 数据，无法生成交互延迟图")
        return False
    
    # 定义颜色映射
    type_colors = {
        "long_narrative": "#1f77b4",  # 蓝色：长篇大论
        "short_reply": "#ff7f0e",     # 橙色：快速回复
        "overlap": "#d62728",          # 红色：重叠
        "normal": "#2ca02c",           # 绿色：正常
    }
    
    fig, axis = plt.subplots(figsize=(14, 8))
    
    # 按类型分组绘制
    for itype in set(types):
        mask = [t == itype for t in types]
        x_vals = [i for i, m in zip(indices, mask) if m]
        y_vals = [d for d, m in zip(delays, mask) if m]
        color = type_colors.get(itype, "#7f7f7f")
        label = f"{itype} (n={len(x_vals)})"
        axis.scatter(x_vals, y_vals, alpha=0.6, s=50, color=color, label=label, edgecolors="black", linewidth=0.5)
    
    # 添加平均延迟线
    avg_delay = np.mean(delays)
    axis.axhline(avg_delay, color="red", linestyle="--", linewidth=2, label=f"平均延迟: {avg_delay:.0f}ms")
    
    # 添加理想对话延迟区间（200-800ms为自然对话范围）
    natural_min, natural_max = 200, 800
    axis.axhspan(natural_min, natural_max, alpha=0.1, color="green", label="自然对话延迟区间")
    
    axis.set_xlabel("段落序号", fontsize=12, weight="bold")
    axis.set_ylabel("交互延迟（毫秒）", fontsize=12, weight="bold")
    axis.set_title("交互延迟散点图（对话自然度分析）", fontsize=14, weight="bold")
    axis.legend(loc="upper right", fontsize=10)
    axis.grid(True, alpha=0.3)
    
    # 添加统计信息
    long_narrative_count = sum(1 for t in types if t == "long_narrative")
    short_reply_count = sum(1 for t in types if t == "short_reply")
    overlap_count = sum(1 for t in types if t == "overlap")
    stats_info = f"总计: {len(delays)} | 长篇: {long_narrative_count} | 快速: {short_reply_count} | 重叠: {overlap_count} | 均值: {avg_delay:.0f}ms | 范围: {min(delays):.0f}-{max(delays):.0f}ms"
    axis.text(0.5, -0.12, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_robustness_chart(per_file_stats: List[Dict], output_path: Path) -> bool:
    """绘制多数据源鲁棒性对比柱状图"""
    if not per_file_stats:
        print("[跳过] quality_stats.json 中没有 per_file 文件级统计")
        return False

    import numpy as np

    # 按原始索引排序
    normalized = []
    for item in per_file_stats:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "failed":
            continue
        idx = int(item.get("idx", 999))
        kept = int(item.get("kept", 0))
        removed = int(item.get("removed", 0))
        name = item.get("file_stem") or item.get("file_name") or f"file_{idx}"
        normalized.append({"idx": idx, "name": str(name), "kept": kept, "removed": removed})

    if not normalized:
        print("[跳过] quality_stats.json 没有可用于鲁棒性图的成功文件记录")
        return False

    sorted_files = sorted(normalized, key=lambda x: x.get("idx", 999))

    filenames = [Path(item["name"]).stem for item in sorted_files]  # 仅保留文件名（不含路径）
    kept_values = [item["kept"] for item in sorted_files]
    removed_values = [item["removed"] for item in sorted_files]
    
    fig, axis = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(filenames))
    width = 0.6
    
    # 堆叠柱状图
    bars1 = axis.bar(x, kept_values, width, label="保留段落", color="#2ca02c", edgecolor="black", linewidth=1)
    bars2 = axis.bar(x, removed_values, width, bottom=kept_values, label="移除段落", color="#d62728", edgecolor="black", linewidth=1)
    
    # 在每个柱子上标注数值和保留率
    for idx, stats in enumerate(sorted_files):
        kept = stats["kept"]
        removed = stats["removed"]
        total = kept + removed
        retention_rate = (kept / total * 100) if total > 0 else 0
        
        # 保留段落标签
        axis.text(idx, kept/2, str(kept), ha="center", va="center", fontsize=9, weight="bold", color="white")
        # 移除段落标签
        axis.text(idx, kept + removed/2, str(removed), ha="center", va="center", fontsize=9, weight="bold", color="white")
        # 保留率标签（在柱子顶部）
        axis.text(idx, total + 5, f"{retention_rate:.1f}%", ha="center", va="bottom", fontsize=10, weight="bold")
    
    axis.set_xlabel("音频文件", fontsize=12, weight="bold")
    axis.set_ylabel("段落数量", fontsize=12, weight="bold")
    axis.set_title("多数据源鲁棒性对比（不同源音频的清洗效果差异）", fontsize=14, weight="bold")
    axis.set_xticks(x)
    axis.set_xticklabels(filenames, rotation=45, ha="right")
    axis.legend(fontsize=11)
    axis.grid(axis="y", alpha=0.3)
    
    # 添加全局统计
    total_kept = sum(kept_values)
    total_removed = sum(removed_values)
    total_all = total_kept + total_removed
    global_retention = (total_kept / total_all * 100) if total_all > 0 else 0
    stats_info = f"总计: {total_all} | 保留: {total_kept} ({global_retention:.1f}%) | 移除: {total_removed} ({100-global_retention:.1f}%) | 文件数: {len(filenames)}"
    axis.text(0.5, -0.2, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_llm_analysis(rows: List[Dict], output_dir: Path) -> bool:
    """
    绘制 LLM 规则 H 相关的可视化分析（情感评分分布）。
    生成图：LLM语句情感分析图.png
    """
    import numpy as np

    # 筛选有 LLM 标签的数据
    llm_rows = [r for r in rows if r.get("llm_status") == "checked"]
    if not llm_rows:
        print("[跳过] 未找到 LLM 语义标签数据（llm_status=checked），无法生成 LLM 分析图")
        return False

    # ── 图1: 情感评分分布直方图 ──
    emotion_scores = [int(r.get("emotion_score", 0)) for r in llm_rows if r.get("emotion_score") is not None]
    if emotion_scores:
        fig, axis = plt.subplots(figsize=(10, 7))
        score_counts = Counter(emotion_scores)
        scores = sorted(score_counts.keys())
        counts = [score_counts[s] for s in scores]
        
        # 颜色映射：1=冷色(差) → 5=暖色(好)
        color_map = {1: "#d62728", 2: "#ff7f0e", 3: "#bcbd22", 4: "#2ca02c", 5: "#1f77b4"}
        bar_colors = [color_map.get(s, "#7f7f7f") for s in scores]
        
        bars = axis.bar(scores, counts, color=bar_colors, edgecolor="black", linewidth=1.2, width=0.7)
        
        for bar, count in zip(bars, counts):
            axis.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                      str(count), ha="center", va="bottom", fontsize=12, weight="bold")
        
        avg_score = np.mean(emotion_scores)
        axis.axvline(avg_score, color="red", linestyle="--", linewidth=2, label=f"平均情感评分: {avg_score:.2f}")
        
        axis.set_xlabel("情感丰富度评分 (1-5)", fontsize=12, weight="bold")
        axis.set_ylabel("会话组数量", fontsize=12, weight="bold")
        axis.set_title("LLM 情感评分分布（规则H: 情感丰富度分析）", fontsize=14, weight="bold")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xticklabels(["1\n(念稿)", "2\n(平淡)", "3\n(一般)", "4\n(丰富)", "5\n(强共情)"])
        axis.legend(fontsize=11)
        axis.grid(axis="y", alpha=0.3)
        
        # 统计信息
        total = len(emotion_scores)
        high_emo = sum(1 for s in emotion_scores if s >= 4)
        high_pct = (high_emo / total * 100) if total > 0 else 0
        stats_info = f"已评估: {total} 组 | 均分: {avg_score:.2f} | 高情感(≥4): {high_emo} ({high_pct:.1f}%)"
        axis.text(0.5, -0.15, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
        
        fig.tight_layout()
        llm_output = output_dir / CHART_FILENAMES["llm_emotion"]
        fig.savefig(llm_output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[输出] 情感评分分布: {llm_output}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="语料库可视化分析：漏斗图 / 语速直方图 / 交互延迟 / 鲁棒性 / LLM分析")
    parser.add_argument("--merged-jsonl", default="outputs/corpus/dataset_merged.jsonl", help="清洗后合并数据")
    parser.add_argument("--quality-stats", default="outputs/corpus/quality_stats.json", help="质量统计JSON（图表唯一数据源）")
    parser.add_argument("--output-dir", default="outputs/analysis", help="图表输出目录")
    
    # 图表生成选项
    parser.add_argument("--enable-funnel", action="store_true", default=True, help="生成清洗漏斗图")
    parser.add_argument("--enable-speed-histogram", action="store_true", default=True, help="生成语速分布直方图")
    parser.add_argument("--enable-interaction-delay", action="store_true", help="生成交互延迟散点图")
    parser.add_argument("--enable-robustness", action="store_true", help="生成多源鲁棒性对比柱状图")
    parser.add_argument("--enable-llm-analysis", action="store_true", help="生成LLM语义分析图（情感评分分布）")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = Path(args.merged_jsonl)
    rows = _load_jsonl(merged_path)
    print(f"[读取] merged rows: {len(rows)}")

    # 解析结构化质量统计
    quality_bundle = _load_quality_stats(Path(args.quality_stats))
    quality_stats = quality_bundle["summary"] if quality_bundle else None
    quality_per_file = quality_bundle["per_file"] if quality_bundle else []

    # 1) 漏斗图
    if args.enable_funnel and quality_stats:
        funnel_path = output_dir / CHART_FILENAMES["funnel"]
        _draw_funnel(quality_stats, funnel_path)
        print(f"[输出] 漏斗图: {funnel_path}")

    # 2) 语速直方图
    if args.enable_speed_histogram:
        histogram_path = output_dir / CHART_FILENAMES["speed_histogram"]
        ok = _draw_speed_histogram(rows, histogram_path)
        if ok:
            print(f"[输出] 语速直方图: {histogram_path}")

    # 3) 交互延迟散点图
    if args.enable_interaction_delay:
        try:
            interaction_path = output_dir / CHART_FILENAMES["interaction_delay"]
            ok = _draw_interaction_delay(rows, interaction_path)
            if ok:
                print(f"[输出] 交互延迟图: {interaction_path}")
        except Exception as e:
            print(f"[跳过] 交互延迟图生成失败: {e}")

    # 4) 多数据源鲁棒性对比
    if args.enable_robustness:
        try:
            robustness_path = output_dir / CHART_FILENAMES["robustness"]
            ok = _draw_robustness_chart(quality_per_file, robustness_path)
            if ok:
                print(f"[输出] 鲁棒性对比图: {robustness_path}")
        except Exception as e:
            print(f"[跳过] 鲁棒性对比图生成失败: {e}")

    # 5) LLM 语义分析图
    if args.enable_llm_analysis:
        try:
            ok = _draw_llm_analysis(rows, output_dir)
            if not ok:
                print("[跳过] LLM 分析图未生成（无 LLM 标签数据）")
        except Exception as e:
            print(f"[跳过] LLM 分析图生成失败: {e}")


if __name__ == "__main__":
    main()
