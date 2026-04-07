#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
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

try:
    from wordcloud import WordCloud
except Exception:  # pragma: no cover
    WordCloud = None


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_quality_log(path: Path) -> Optional[Dict[str, int]]:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")

    def _extract(pattern: str) -> int:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    stats = {
        "raw": _extract(r"原始段落:\s*(\d+)"),
        "kept": _extract(r"保留段落:\s*(\d+)"),
        "rule_a": _extract(r"A\s+置信度过滤\s+(\d+)"),
        "rule_b": _extract(r"B\s+重叠音过滤\s+(\d+)"),
        "rule_c": _extract(r"C\s+长度匹配度\s+(\d+)"),
        "rule_d": _extract(r"D\s+上下文孤岛\s+(\d+)"),
        "rule_e": _extract(r"E\s+低信息量\s+(\d+)"),
        "rule_f": _extract(r"F\s+信噪比\s+(\d+)"),
        "rule_g": _extract(r"G\s+黑名单\s+(\d+)"),
        "rule_h": _extract(r"H\s+LLM语义\s+(\d+)"),
    }
    
    # 提取文件处理统计（从"│ 文件: X 成功 / Y 失败"这样的行）
    file_match = re.search(r"文件:\s*(\d+)\s+成功\s*/\s*(\d+)\s+失败", text)
    if file_match:
        stats["success_files"] = int(file_match.group(1))
        stats["failed_files"] = int(file_match.group(2))
    else:
        stats["success_files"] = 0
        stats["failed_files"] = 0

    if stats["raw"] <= 0:
        return None
    return stats


def _draw_funnel(stats: Dict[str, int], output_path: Path) -> None:
    removal_order = [
        ("规则A: 置信度过滤", stats.get("rule_a", 0)),
        ("规则B: 重叠音过滤", stats.get("rule_b", 0)),
        ("规则C: 长度匹配度", stats.get("rule_c", 0)),
        ("规则D: 上下文孤岛", stats.get("rule_d", 0)),
        ("规则E: 低信息量", stats.get("rule_e", 0)),
        ("规则F: 信噪比", stats.get("rule_f", 0)),
        ("规则G: 黑名单/脱轨", stats.get("rule_g", 0)),
        ("规则H: LLM语义", stats.get("rule_h", 0)),
    ]

    stage_names = ["原始数据"]
    stage_values = [stats["raw"]]
    stage_labels = [f"{stats['raw']}"]

    current = stats["raw"]
    removed_cumsum = 0
    total_removed = sum(removed for _, removed in removal_order)
    
    for rule_name, removed in removal_order:
        removed_cumsum += removed
        if removed > 0:
            pct = (removed / total_removed * 100) if total_removed > 0 else 0
            stage_names.append(f"{rule_name}\n(移除{removed}, {pct:.1f}%)")
        else:
            stage_names.append(f"{rule_name}\n(0)")
        current = max(current - removed, 0)
        stage_values.append(current)
        stage_labels.append(f"{current}")

    fig, axis = plt.subplots(figsize=(12, 9))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    bars = axis.barh(stage_names, stage_values, color=colors[:len(stage_names)])
    axis.invert_yaxis()
    axis.set_xlabel("段落数量", fontsize=12, weight="bold")
    axis.set_ylabel("处理阶段", fontsize=12, weight="bold")
    axis.set_title("语料库清洗漏斗图（多层级数据质量过滤）", fontsize=14, weight="bold")
    
    # 在每个柱子上标注数值
    for idx, (bar, label) in enumerate(zip(bars, stage_labels)):
        width = bar.get_width()
        axis.text(width + stats["raw"] * 0.01, bar.get_y() + bar.get_height()/2, 
                  f"{label}", ha="left", va="center", fontsize=10, weight="bold")
    
    # 添加总体统计信息
    keep_pct = (stats["kept"] / stats["raw"] * 100) if stats["raw"] > 0 else 0
    stats_text = f"原始: {stats['raw']} | 保留: {stats['kept']} ({keep_pct:.1f}%) | 移除: {total_removed} ({100-keep_pct:.1f}%)"
    axis.text(0.5, -0.1, stats_text, transform=axis.transAxes, ha="center", fontsize=11,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_speed_histogram(rows: List[Dict], output_path: Path) -> None:
    speed_values = [float(item.get("words_per_second")) for item in rows if item.get("words_per_second") is not None]
    if not speed_values:
        raise RuntimeError("未找到 words_per_second 字段，无法绘制语速分布图")

    fig, axis = plt.subplots(figsize=(12, 7))
    n, bins, patches = axis.hist(speed_values, bins=40, color="#70ad47", edgecolor="white", alpha=0.8)
    
    # 计算平均语速
    avg_speed = sum(speed_values) / len(speed_values)
    
    # 添加舒适语速区间虚线（4.0-7.0 字/秒为人类自然沟通范围）
    comfort_min, comfort_max = 4.0, 7.0
    axis.axvline(avg_speed, color="red", linestyle="-", linewidth=2.5, label=f"平均语速: {avg_speed:.2f} 字/秒", zorder=3)
    axis.axvline(comfort_min, color="blue", linestyle="--", linewidth=2, label=f"舒适区间下限: {comfort_min} 字/秒", zorder=3)
    axis.axvline(comfort_max, color="blue", linestyle="--", linewidth=2, label=f"舒适区间上限: {comfort_max} 字/秒", zorder=3)
    
    # 填充舒适区间背景
    axis.axvspan(comfort_min, comfort_max, alpha=0.1, color="blue", label="人类自然沟通舒适区")
    
    # 统计在舒适区间内的比例
    comfort_count = sum(1 for v in speed_values if comfort_min <= v <= comfort_max)
    comfort_pct = (comfort_count / len(speed_values) * 100) if speed_values else 0
    
    axis.set_title("语速分布直方图（人类自然沟通特征分析）", fontsize=14, weight="bold")
    axis.set_xlabel("语速（字/秒）", fontsize=12, weight="bold")
    axis.set_ylabel("段落数量", fontsize=12, weight="bold")
    axis.legend(loc="upper right", fontsize=10)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    
    # 添加统计信息
    stats_info = f"总段落: {len(speed_values)} | 舒适区间内: {comfort_count} ({comfort_pct:.1f}%) | 最小: {min(speed_values):.2f} | 最大: {max(speed_values):.2f}"
    axis.text(0.5, -0.12, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_interaction_delay(rows: List[Dict], output_path: Path) -> None:
    """绘制交互延迟散点图（conversation turn-taking分析）"""
    import numpy as np
    
    # 提取 turn_delay 和 interaction_type
    delays = []
    types = []
    indices = []
    
    for idx, row in enumerate(rows):
        delay = row.get("turn_delay")
        itype = row.get("interaction_type", "normal")
        if delay is not None and isinstance(delay, (int, float)):
            delays.append(float(delay))
            types.append(itype if itype else "normal")
            indices.append(idx + 1)
    
    if not delays:
        # 如果没有 turn_delay 数据，跳过
        return
    
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


def _draw_quality_report(stats: Dict[str, int], output_path: Path, output_dir: str = "", dataset_file: str = "") -> None:
    """绘制数据质量报告图表"""
    fig, axis = plt.subplots(figsize=(12, 10))
    axis.axis("off")
    
    # 计算清洗数据
    total_removed = stats.get("rule_a", 0) + stats.get("rule_b", 0) + stats.get("rule_c", 0) + \
                    stats.get("rule_d", 0) + stats.get("rule_e", 0) + stats.get("rule_f", 0) + \
                    stats.get("rule_g", 0) + stats.get("rule_h", 0)
    raw = stats.get("raw", 0)
    kept = stats.get("kept", 0)
    success_files = stats.get("success_files", 0)
    failed_files = stats.get("failed_files", 0)
    total_files = success_files + failed_files
    
    # 准备文本内容
    y_pos = 0.98
    line_height = 0.035
    
    # 标题
    axis.text(0.5, y_pos, "数据质量报告", fontsize=18, weight="bold", ha="center")
    y_pos -= line_height * 1.5
    
    # 文件统计
    axis.text(0.05, y_pos, f"文件: {success_files} 成功 / {failed_files} 失败 (共 {total_files})", 
              fontsize=11)
    y_pos -= line_height * 2
    
    # 分割线
    axis.plot([0.05, 0.95], [y_pos, y_pos], 'k-', lw=0.5)
    y_pos -= line_height
    
    # 段落统计
    kept_pct = (kept / raw * 100) if raw > 0 else 0
    axis.text(0.05, y_pos, f"原始段落:      {raw:6d}", fontsize=11, weight="bold")
    y_pos -= line_height
    axis.text(0.05, y_pos, f"保留段落:      {kept:6d}  ({kept_pct:5.1f}%)", fontsize=11, weight="bold", color="#4CAF50")
    y_pos -= line_height
    axis.text(0.05, y_pos, f"移除段落:      {total_removed:6d}", fontsize=11, weight="bold", color="#F44336")
    y_pos -= line_height * 2
    
    # 分割线
    axis.plot([0.05, 0.95], [y_pos, y_pos], 'k-', lw=0.5)
    y_pos -= line_height
    
    # 清洗规则明细
    axis.text(0.05, y_pos, "清洗规则命中明细", fontsize=12, weight="bold")
    y_pos -= line_height * 1.5
    
    rule_info = [
        ("A 置信度过滤", stats.get("rule_a", 0)),
        ("B 重叠音过滤", stats.get("rule_b", 0)),
        ("C 长度匹配度", stats.get("rule_c", 0)),
        ("D 上下文孤岛", stats.get("rule_d", 0)),
        ("E 低信息量", stats.get("rule_e", 0)),
        ("F 信噪比", stats.get("rule_f", 0)),
        ("G 黑名单", stats.get("rule_g", 0)),
        ("H LLM语义", stats.get("rule_h", 0)),
    ]
    
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
    
    for idx, (rule_name, removed_count) in enumerate(rule_info):
        pct = (removed_count / total_removed * 100) if total_removed > 0 else 0
        
        # 规则名称和数字
        axis.text(0.05, y_pos, f"{rule_name:12s}", fontsize=10)
        axis.text(0.30, y_pos, f"{removed_count:6d}", fontsize=10, weight="bold")
        
        # 进度条
        bar_width = pct / 100 * 0.5  # 进度条最大宽度 0.5
        bar_color = colors[idx]
        if bar_width > 0:
            axis.barh(y_pos - 0.007, bar_width, height=0.018, left=0.40, color=bar_color, alpha=0.8)
        
        # 百分比
        axis.text(0.92, y_pos, f"{pct:5.1f}%", fontsize=10, ha="right")
        y_pos -= line_height
    
    y_pos -= line_height
    
    # 分割线
    axis.plot([0.05, 0.95], [y_pos, y_pos], 'k-', lw=0.5)
    y_pos -= line_height
    
    # 输出信息
    output_dir_text = output_dir if output_dir else "N/A"
    dataset_file_text = dataset_file if dataset_file else "N/A"
    axis.text(0.05, y_pos, f"输出目录: {output_dir_text}", fontsize=9)
    y_pos -= line_height
    axis.text(0.05, y_pos, f"数据集文件: {dataset_file_text}", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_poison_keywords(removed_g_rows: List[Dict], output_path: Path) -> bool:
    """绘制'数据毒药'关键词频率图（规则G被移除内容的语义特征分析）"""
    if not removed_g_rows:
        print("[跳过] removed_g 数据为空，无法生成毒药关键词图")
        return False
    
    import numpy as np
    
    # 提取并统计中文关键词
    token_counter = Counter()
    for row in removed_g_rows:
        text = str(row.get("text", ""))
        # 提取连续的中文字符序列（长度2以上）
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            token_counter[token] += 1
    
    if not token_counter:
        print("[跳过] 毒药关键词统计为空")
        return False
    
    # 取频率最高的20个关键词
    top_keywords = dict(token_counter.most_common(20))
    keywords = list(top_keywords.keys())
    frequencies = list(top_keywords.values())
    
    fig, axis = plt.subplots(figsize=(14, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(keywords)))
    bars = axis.barh(keywords, frequencies, color=colors, edgecolor="black", linewidth=1)
    
    # 在每个柱子上标注数值
    for bar, freq in zip(bars, frequencies):
        width = bar.get_width()
        axis.text(width + 1, bar.get_y() + bar.get_height()/2, f"{freq}次", 
                  ha="left", va="center", fontsize=10, weight="bold")
    
    axis.set_xlabel("出现频次", fontsize=12, weight="bold")
    axis.set_title(f"'数据毒药'关键词排行（规则G拦截的脏数据语义特征分析）", fontsize=14, weight="bold")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.3)
    
    # 添加统计信息
    total_removed = sum(frequencies)
    unique_keywords = len(token_counter)
    stats_info = f"被规则G拦截段落: {len(removed_g_rows)} | 唯一关键词: {unique_keywords} | 关键词总出现数: {total_removed}"
    axis.text(0.5, -0.12, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _parse_per_file_stats(quality_log_path: Path) -> Dict[str, Dict[str, int]]:
    """从日志中解析每个文件的保留/移除统计"""
    per_file = {}
    
    if not quality_log_path.exists():
        return per_file
    
    text = quality_log_path.read_text(encoding="utf-8", errors="ignore")
    
    # 匹配日志中的文件处理记录：✓ [1/N] filename  (耗时Xs) 保留 X 段 / 移除 Y 段
    pattern = r"✓\s+\[(\d+)/\d+\]\s+(.+?)\s+\(\d+\.\d+s\)\s+保留\s+(\d+)\s+段\s+/\s+移除\s+(\d+)\s+段"
    for match in re.finditer(pattern, text):
        file_idx = int(match.group(1))
        filename = match.group(2).strip()
        kept = int(match.group(3))
        removed = int(match.group(4))
        per_file[filename] = {"kept": kept, "removed": removed, "idx": file_idx}
    
    return per_file


def _draw_robustness_chart(quality_log_path: Path, output_path: Path) -> bool:
    """绘制多数据源鲁棒性对比柱状图"""
    per_file_stats = _parse_per_file_stats(quality_log_path)
    
    if not per_file_stats:
        print("[跳过] 无法从日志中解析文件级统计数据")
        return False
    
    import numpy as np
    
    # 按原始索引排序
    sorted_files = sorted(per_file_stats.items(), key=lambda x: x[1].get("idx", 999))
    
    filenames = [Path(name).stem for name, _ in sorted_files]  # 仅保留文件名（不含路径）
    kept_values = [stats["kept"] for _, stats in sorted_files]
    removed_values = [stats["removed"] for _, stats in sorted_files]
    
    fig, axis = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(filenames))
    width = 0.6
    
    # 堆叠柱状图
    bars1 = axis.bar(x, kept_values, width, label="保留段落", color="#2ca02c", edgecolor="black", linewidth=1)
    bars2 = axis.bar(x, removed_values, width, bottom=kept_values, label="移除段落", color="#d62728", edgecolor="black", linewidth=1)
    
    # 在每个柱子上标注数值和保留率
    for idx, (file, stats) in enumerate(sorted_files):
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


def _draw_rule_g_wordcloud(
    removed_g_rows: List[Dict],
    output_path: Path,
    font_path: Optional[str] = None,
) -> bool:
    if WordCloud is None:
        print("[跳过] 未安装 wordcloud，无法生成词云图")
        return False

    if not removed_g_rows:
        print("[跳过] removed_g 数据为空，无法生成规则 G 词云图")
        return False

    token_counter = Counter()
    for row in removed_g_rows:
        text = str(row.get("text", ""))
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}", text):
            token_counter[token] += 1

    if not token_counter:
        print("[跳过] removed_g 文本分词后为空")
        return False

    cloud = WordCloud(
        width=1400,
        height=900,
        background_color="white",
        font_path=font_path,
        max_words=180,
        collocations=False,
    )
    cloud.generate_from_frequencies(token_counter)
    cloud.to_file(str(output_path))
    return True


def _draw_llm_analysis(rows: List[Dict], output_dir: Path) -> bool:
    """
    绘制 LLM 规则 H 相关的可视化分析（情感评分分布 + 广告/幻觉标记统计）。
    生成两张图：emotion_distribution.png 和 llm_label_summary.png
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
        fig.savefig(output_dir / "emotion_distribution.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[输出] 情感评分分布: {output_dir / 'emotion_distribution.png'}")

    # ── 图2: LLM 标签汇总饼图 ──
    ad_count = sum(1 for r in llm_rows if r.get("is_ad"))
    hallucination_count = sum(1 for r in llm_rows if r.get("hallucination_risk"))
    clean_count = sum(1 for r in llm_rows if not r.get("is_ad") and not r.get("hallucination_risk"))
    unchecked_count = sum(1 for r in rows if r.get("llm_status") != "checked")
    
    labels = []
    sizes = []
    colors_pie = []
    
    if clean_count > 0:
        labels.append(f"清洁段落\n({clean_count})")
        sizes.append(clean_count)
        colors_pie.append("#2ca02c")
    if ad_count > 0:
        labels.append(f"软广/无关\n({ad_count})")
        sizes.append(ad_count)
        colors_pie.append("#d62728")
    if hallucination_count > 0:
        labels.append(f"幻觉风险\n({hallucination_count})")
        sizes.append(hallucination_count)
        colors_pie.append("#ff7f0e")
    if unchecked_count > 0:
        labels.append(f"未检查\n({unchecked_count})")
        sizes.append(unchecked_count)
        colors_pie.append("#7f7f7f")
    
    if sizes:
        fig, axis = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = axis.pie(
            sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "black", "linewidth": 1}
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_weight("bold")
        
        axis.set_title("LLM 语义标签分布（规则H: 内容质量全景）", fontsize=14, weight="bold")
        
        total_all = len(rows)
        checked = len(llm_rows)
        stats_info = f"总段落: {total_all} | 已评估: {checked} | 未评估: {unchecked_count}"
        axis.text(0.5, -0.05, stats_info, transform=axis.transAxes, ha="center", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
        
        fig.tight_layout()
        fig.savefig(output_dir / "llm_label_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[输出] LLM 标签汇总: {output_dir / 'llm_label_summary.png'}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="语料库可视化分析：漏斗图 / 语速直方图 / 规则G词云")
    parser.add_argument("--merged-jsonl", default="outputs/corpus/dataset_merged.jsonl", help="清洗后合并数据")
    parser.add_argument("--quality-log", default="outputs/corpus/日志.txt", help="质量报告日志（用于漏斗图）")
    parser.add_argument("--removed-g-jsonl", default="", help="规则 G 被移除样本 JSONL（可选）")
    parser.add_argument("--output-dir", default="outputs/analysis", help="图表输出目录")
    parser.add_argument("--font-path", default="", help="中文字体路径（词云中文建议设置）")
    
    # 图表生成选项
    parser.add_argument("--enable-quality-report", action="store_true", default=True, help="生成质量报告")
    parser.add_argument("--enable-funnel", action="store_true", default=True, help="生成清洗漏斗图")
    parser.add_argument("--enable-speed-histogram", action="store_true", default=True, help="生成语速分布直方图")
    parser.add_argument("--enable-interaction-delay", action="store_true", help="生成交互延迟散点图")
    parser.add_argument("--enable-poison-keywords", action="store_true", help="生成数据毒药关键词排行")
    parser.add_argument("--enable-robustness", action="store_true", help="生成多源鲁棒性对比柱状图")
    parser.add_argument("--enable-wordcloud", action="store_true", help="生成规则G词云")
    parser.add_argument("--enable-llm-analysis", action="store_true", help="生成LLM语义分析图（情感评分分布 + 标签汇总）")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = Path(args.merged_jsonl)
    rows = _load_jsonl(merged_path)
    print(f"[读取] merged rows: {len(rows)}")

    # 解析质量日志
    quality_stats = _parse_quality_log(Path(args.quality_log))

    # 1) 数据质量报告
    if args.enable_quality_report and quality_stats:
        quality_path = output_dir / "quality_report.png"
        _draw_quality_report(quality_stats, quality_path, output_dir=args.output_dir, dataset_file="dataset_merged.jsonl")
        print(f"[输出] 质量报告: {quality_path}")
    
    # 2) 漏斗图
    if args.enable_funnel and quality_stats:
        funnel_path = output_dir / "cleaning_funnel.png"
        _draw_funnel(quality_stats, funnel_path)
        print(f"[输出] 漏斗图: {funnel_path}")

    # 3) 语速直方图
    if args.enable_speed_histogram:
        histogram_path = output_dir / "speech_rate_histogram.png"
        _draw_speed_histogram(rows, histogram_path)
        print(f"[输出] 语速直方图: {histogram_path}")

    # 4) 交互延迟散点图
    if args.enable_interaction_delay:
        try:
            interaction_path = output_dir / "interaction_delay.png"
            _draw_interaction_delay(rows, interaction_path)
            print(f"[输出] 交互延迟图: {interaction_path}")
        except Exception as e:
            print(f"[跳过] 交互延迟图生成失败: {e}")

    # 5) 多数据源鲁棒性对比
    if args.enable_robustness:
        try:
            robustness_path = output_dir / "robustness_comparison.png"
            ok = _draw_robustness_chart(Path(args.quality_log), robustness_path)
            if ok:
                print(f"[输出] 鲁棒性对比图: {robustness_path}")
        except Exception as e:
            print(f"[跳过] 鲁棒性对比图生成失败: {e}")

    # 6) 规则 G 词云和毒药关键词（需提供 removed_g 样本）
    removed_path_text = args.removed_g_jsonl.strip()
    if removed_path_text:
        removed_rows = _load_jsonl(Path(removed_path_text))
        
        # 毒药关键词柱状图
        if args.enable_poison_keywords:
            try:
                poison_path = output_dir / "poison_keywords.png"
                ok = _draw_poison_keywords(removed_rows, poison_path)
                if ok:
                    print(f"[输出] 毒药关键词图: {poison_path}")
            except Exception as e:
                print(f"[跳过] 毒药关键词图生成失败: {e}")
        
        # 规则G词云
        if args.enable_wordcloud:
            cloud_path = output_dir / "rule_g_wordcloud.png"
            ok = _draw_rule_g_wordcloud(
                removed_rows,
                cloud_path,
                font_path=args.font_path.strip() or None,
            )
            if ok:
                print(f"[输出] 规则G词云: {cloud_path}")
    else:
        print("[跳过] 未提供 --removed-g-jsonl，词云和毒药关键词图未生成")

    # 8) LLM 语义分析图
    if args.enable_llm_analysis:
        try:
            ok = _draw_llm_analysis(rows, output_dir)
            if not ok:
                print("[跳过] LLM 分析图未生成（无 LLM 标签数据）")
        except Exception as e:
            print(f"[跳过] LLM 分析图生成失败: {e}")


if __name__ == "__main__":
    main()
