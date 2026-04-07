"""
数据清洗模块：领域驱动的语料库质量控制规则引擎。

实现毕设文档中定义的八条清洗规则：
- 规则 A: 置信度过滤 — 丢弃低置信度句子
- 规则 B: 重叠音过滤 — 检测并丢弃多人同时说话片段
- 规则 C: 长度匹配度 — 音频长度与文本字数比例失调检测
- 规则 D: 上下文孤岛过滤 — 过滤在时间窗口内完全无其他说话人出现的孤立片段
- 规则 E: 低信息量剔除 — 剔除无意义单字附和
- 规则 F: 信噪比检测 — 基于音频特征筛选劣质片段（简化实现）
- 规则 G: 黑名单/脱轨内容过滤 — 基于关键词识别广告、推广等无关内容，连带清除相邻上下文
- 规则 H: LLM 语义校验 — 通过大语言模型进行软广识别、情感评分、幻觉检测（在 engine.py 中调用 LLMProcessor）
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """清洗规则的可配置阈值参数"""

    # 规则 A: 置信度过滤
    enable_confidence_filter: bool = True
    confidence_threshold: float = 0.6

    # 规则 B: 重叠音过滤
    enable_overlap_filter: bool = True
    overlap_max_gap: float = 0.0  # 两个说话人片段重叠的最大允许间隔(秒), <=0 表示严格重叠
    overlap_min_duration: float = 0.5  # 重叠时长 >= 此值才判定为真正的多人同时说话（对齐精度容差）

    # 规则 C: 长度匹配度
    enable_length_ratio_filter: bool = True
    # CJK: 每秒应有 0.5~25 个字; 拉丁: 每秒应有 0.3~12 个词
    min_chars_per_sec_cjk: float = 0.5
    max_chars_per_sec_cjk: float = 25.0
    min_words_per_sec_latin: float = 0.3
    max_words_per_sec_latin: float = 12.0
    length_ratio_min_duration: float = 1.5  # 时长低于此值的片段跳过语速检查（太短不可靠）

    # 规则 D: 上下文孤岛过滤（孤立片段检测）
    enable_context_island_filter: bool = True
    orphan_window_sec: float = 120.0  # 在前后各 N 秒内寻找不同说话人，找不到则视为孤立
    min_segments_to_apply_d: int = 6  # 剩余段数 <= 此值时跳过规则D（安全阈值）

    # 规则 E: 低信息量剔除
    enable_low_info_filter: bool = True
    low_info_max_duration: float = 1.5  # 持续时间(秒)低于此值
    low_info_max_chars: int = 4  # 且文本字符数低于此值

    # 规则 F: 信噪比检测 (简化：基于 segment 特征)
    enable_snr_filter: bool = False  # 默认关闭，需音频分析支持
    min_snr_db: float = 10.0

    # 规则 G: 黑名单/脱轨内容过滤
    enable_blacklist_filter: bool = True
    blacklist_path: str = ""  # 黑名单词库文件路径，空则使用默认路径
    blacklist_context_purge: bool = True  # 是否连带清除同 conversation_id 的相邻上下文

    # 规则 H: LLM 语义校验
    enable_llm_semantic_filter: bool = False  # 默认关闭，需配置 LLM API
    llm_emotion_threshold: int = 3  # 情感得分 (1-5) 低于此值的组将被丢弃
    llm_concurrency: int = 3  # 并发 API 请求数

    # LLM 深度接管模式（消融实验开关）
    # True: C/D/E/G 降级为软标签，由 LLM 做最终裁决
    # False: 传统模式，C/D/E/G 硬拦截
    use_llm_override_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# 低信息量词汇表（中日英）
_LOW_INFO_PATTERNS = re.compile(
    r'^[\s]*(嗯+|啊+|哦+|呃+|噢+|哈+|嘿+|呵+|唔+|额+|'
    r'对+|是+|好+|行+|嗯嗯+|对对+|好好+|是是+|'
    r'うん+|ああ+|えー+|はい+|ええ+|'
    r'hmm+|uh+|um+|ah+|oh+|yeah+|yes+|no+|ok+|okay+|right+|'
    r'mm+|mhm+|huh+)[\s]*$',
    re.IGNORECASE
)


def _is_cjk_text(text: str) -> bool:
    """判断文本是否以 CJK 字符为主"""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff' or
                    '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff' or
                    '\uac00' <= ch <= '\ud7af')
    non_space = sum(1 for ch in text if not ch.isspace())
    return non_space > 0 and (cjk_count / non_space) > 0.3


def _calc_avg_confidence(segment: dict) -> float:
    """计算 segment 所有词的平均置信度"""
    words = segment.get('words', [])
    scores = [w.get('score', 0.0) for w in words if 'score' in w]
    if not scores:
        return 1.0  # 无置信度信息时默认保留
    return sum(scores) / len(scores)


def _get_text_length_metric(text: str) -> tuple:
    """返回 (长度值, 是否CJK)"""
    is_cjk = _is_cjk_text(text)
    if is_cjk:
        # CJK: 按字符数
        length = sum(1 for ch in text if not ch.isspace())
    else:
        # 拉丁: 按词数
        length = len(text.split())
    return length, is_cjk


def compute_segment_snr(
    audio: np.ndarray,
    segments: List[dict],
    sample_rate: int = 16000,
) -> None:
    """
    为每个 segment 计算信噪比(SNR)并写入 'snr_db' 字段（原地修改）。
    
    方法：将音频分帧，取最安静的 10% 帧的平均功率作为噪声底线，
    与每段的信号功率对比得到 SNR。

    Args:
        audio:       16kHz 单声道 numpy float32 音频数组
        segments:    segment 列表（会被原地添加 'snr_db' 字段）
        sample_rate: 采样率
    """
    if len(audio) == 0 or not segments:
        return

    # 1. 全局帧级功率统计 → 估计噪声底线
    frame_size = int(0.025 * sample_rate)   # 25ms 帧
    hop = int(0.010 * sample_rate)          # 10ms 步长

    frame_powers = []
    for start_idx in range(0, len(audio) - frame_size, hop):
        frame = audio[start_idx:start_idx + frame_size]
        power = float(np.mean(frame ** 2))
        frame_powers.append(power)

    if not frame_powers:
        return

    frame_powers_sorted = sorted(frame_powers)
    # 取最安静的 10% 作为噪声估计
    noise_n = max(1, int(len(frame_powers_sorted) * 0.10))
    noise_power = float(np.mean(frame_powers_sorted[:noise_n]))
    noise_power = max(noise_power, 1e-10)  # 防止除零

    # 2. 逐段计算 SNR
    for seg in segments:
        seg_start = seg.get('start', 0.0)
        seg_end = seg.get('end', 0.0)

        start_sample = max(0, int(seg_start * sample_rate))
        end_sample = min(len(audio), int(seg_end * sample_rate))

        if end_sample - start_sample < frame_size:
            seg['snr_db'] = 0.0
            continue

        segment_audio = audio[start_sample:end_sample]
        signal_power = float(np.mean(segment_audio ** 2))
        signal_power = max(signal_power, 1e-10)

        snr = 10.0 * np.log10(signal_power / noise_power)
        seg['snr_db'] = round(float(snr), 1)

    logger.info(
        f"[SNR计算] 为 {len(segments)} 段计算信噪比完成 "
        f"(噪声底线: {10*np.log10(noise_power):.1f} dB)"
    )


class DataCleaner:
    """
    语料库数据清洗引擎。
    
    接受 WhisperX 输出的 segments 列表，按配置的规则链进行过滤，
    返回清洗后的 segments 和清洗统计报告。
    """

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            # 硬拦截计数
            'rule_a_removed': 0,
            'rule_b_removed': 0,
            'rule_f_removed': 0,
            # 传统模式（硬拦截）或 LLM 模式（软标签）计数
            'rule_c_removed': 0,
            'rule_c_flagged': 0,
            'rule_d_removed': 0,
            'rule_d_flagged': 0,
            'rule_e_removed': 0,
            'rule_e_flagged': 0,
            'rule_g_removed': 0,
            'rule_g_flagged': 0,
            # LLM 裁决计数
            'rule_h_removed': 0,
            'rule_h_unchecked': 0,
        }
        # 被各规则移除的片段（含移除原因）
        self.removed_segments: List[dict] = []

    def hard_filter(self, segments: List[dict]) -> List[dict]:
        """
        Stage 1: 物理硬阻断 — 执行规则 A、B、F，不达标者直接丢弃。
        这些是基于物理信号质量的规则，与语义无关，无需 LLM 介入。
        
        Args:
            segments: WhisperX 对齐+说话人分离后的 segment 列表
            
        Returns:
            通过硬过滤的 segments
        """
        self.stats = {k: 0 for k in self.stats}
        self.stats['input_count'] = len(segments)
        self.removed_segments = []

        result = list(segments)

        if self.config.enable_confidence_filter:
            result = self._rule_a_confidence(result)

        if self.config.enable_overlap_filter:
            result = self._rule_b_overlap(result)

        if self.config.enable_snr_filter:
            result = self._rule_f_snr(result)

        hard_removed = self.stats['input_count'] - len(result)
        if hard_removed > 0:
            logger.info(
                f"[Stage1·硬阻断] {self.stats['input_count']} → {len(result)} 段 "
                f"(A={self.stats['rule_a_removed']}, B={self.stats['rule_b_removed']}, F={self.stats['rule_f_removed']})"
            )
        return result

    def soft_label(self, segments: List[dict]) -> List[dict]:
        """
        Stage 2: 启发式软标签 — 执行规则 C、D、E。
        
        行为取决于 use_llm_override_mode:
        - False（传统模式）: 不达标者直接丢弃（与旧版行为一致）
        - True（LLM 接管模式）: 不达标者仅注入 flag_* 标签，保留供 LLM 裁决
        
        Args:
            segments: 已通过 hard_filter 的 segment 列表
            
        Returns:
            处理后的 segments（传统模式下数量减少，LLM 模式下数量不变但携带 flag）
        """
        result = list(segments)

        if self.config.enable_length_ratio_filter:
            result = self._rule_c_length_ratio(result)

        if self.config.enable_low_info_filter:
            result = self._rule_e_low_info(result)

        # 规则 D 需要在 C/E 之后执行（依赖说话人交替模式）
        if self.config.enable_context_island_filter:
            result = self._rule_d_context_island(result)

        self.stats['output_count'] = len(result)

        # 日志汇总
        override = self.config.use_llm_override_mode
        mode_label = "LLM接管" if override else "传统"
        removed_total = self.stats['input_count'] - self.stats['output_count']
        flagged_total = sum(self.stats.get(f'rule_{r}_flagged', 0) for r in ('c', 'd', 'e'))

        logger.info(
            f"[数据清洗·{mode_label}] 输入 {self.stats['input_count']} 段 → "
            f"输出 {self.stats['output_count']} 段 "
            f"(硬拦截移除 {self.stats['rule_a_removed']+self.stats['rule_b_removed']+self.stats['rule_f_removed']}, "
            f"软规则移除 {self.stats['rule_c_removed']+self.stats['rule_d_removed']+self.stats['rule_e_removed']}, "
            f"软标签标记 {flagged_total})"
        )
        for rule_key in ['rule_a', 'rule_b', 'rule_f', 'rule_c', 'rule_d', 'rule_e', 'rule_g', 'rule_h']:
            removed = self.stats.get(f'{rule_key}_removed', 0)
            flagged = self.stats.get(f'{rule_key}_flagged', 0)
            if removed > 0:
                logger.info(f"  - {rule_key.upper()}: 移除 {removed} 段")
            if flagged > 0:
                logger.info(f"  - {rule_key.upper()}: 标记 {flagged} 段 (待LLM裁决)")

        return result

    def clean(self, segments: List[dict]) -> tuple:
        """
        执行全部清洗规则链（兼容旧接口）。
        内部依次调用 hard_filter() + soft_label()。
        
        Returns:
            (cleaned_segments, stats_dict)
        """
        result = self.hard_filter(segments)
        result = self.soft_label(result)
        return result, dict(self.stats)

    # ── 规则 A: 置信度过滤 ──────────────────────────────────
    def _rule_a_confidence(self, segments: List[dict]) -> List[dict]:
        """丢弃平均置信度低于阈值的句子"""
        threshold = self.config.confidence_threshold
        filtered = []
        for seg in segments:
            avg_conf = _calc_avg_confidence(seg)
            if avg_conf >= threshold:
                filtered.append(seg)
            else:
                self.stats['rule_a_removed'] += 1
                self.removed_segments.append({**seg, '_removed_by': 'rule_a', '_reason': f'置信度={avg_conf:.3f}'})
                logger.debug(
                    f"[规则A] 移除 (置信度={avg_conf:.3f}): "
                    f"\"{seg.get('text', '')[:50]}\""
                )
        return filtered

    # ── 规则 B: 重叠音过滤 ──────────────────────────────────
    def _rule_b_overlap(self, segments: List[dict]) -> List[dict]:
        """
        检测并丢弃多人同时说话(Overlap)的片段。

        改进逻辑（相比旧版）：
        1. 只有重叠时长 >= overlap_min_duration 才判定为真正重叠
           （WhisperX 对齐精度有 ±0.3s 的误差，微小重叠不算真重叠）
        2. 发现真重叠时，只移除较短的那个片段，保留较长的
           （播客中抢话时，只丢掉短插话，保留主要内容）
        3. 修正了旧版的统计计数 bug
        """
        if len(segments) < 2:
            return segments

        n = len(segments)
        remove_indices = set()
        min_overlap = self.config.overlap_min_duration

        for i in range(n):
            if i in remove_indices:
                continue
            a = segments[i]
            a_start = a.get('start', 0)
            a_end = a.get('end', 0)
            a_speaker = a.get('speaker', '')
            a_dur = a_end - a_start

            for j in range(i + 1, n):
                if j in remove_indices:
                    continue
                b = segments[j]
                b_speaker = b.get('speaker', '')
                if a_speaker == b_speaker:
                    continue  # 同一说话人不算重叠

                b_start = b.get('start', 0)
                b_end = b.get('end', 0)
                b_dur = b_end - b_start

                # 计算实际重叠时长
                overlap_start = max(a_start, b_start)
                overlap_end = min(a_end, b_end)
                overlap_duration = max(0.0, overlap_end - overlap_start)

                if overlap_duration >= min_overlap:
                    # 移除较短的片段（保留更多内容）
                    victim = j if b_dur <= a_dur else i
                    remove_indices.add(victim)
                    victim_seg = segments[victim]
                    self.removed_segments.append({**victim_seg, '_removed_by': 'rule_b', '_reason': f'重叠{overlap_duration:.2f}s'})
                    logger.debug(
                        f"[规则B] 移除重叠音 (重叠{overlap_duration:.2f}s): "
                        f"\"{victim_seg.get('text', '')[:50]}\""
                    )
                    # 如果 i 自身被移除，跳出内层循环
                    if victim == i:
                        break

        self.stats['rule_b_removed'] = len(remove_indices)
        return [seg for idx, seg in enumerate(segments) if idx not in remove_indices]

    # ── 规则 C: 长度匹配度 ─────────────────────────────────
    def _rule_c_length_ratio(self, segments: List[dict]) -> List[dict]:
        """
        音频长度与文本字数比例极度失调的，判定为噪音。
        LLM 接管模式下仅标记 flag_c_length_ratio，不删除。
        """
        override = self.config.use_llm_override_mode
        filtered = []
        min_dur = self.config.length_ratio_min_duration
        for seg in segments:
            text = seg.get('text', '').strip()
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = max(end - start, 0.01)

            if duration < min_dur:
                filtered.append(seg)
                continue

            text_len, is_cjk = _get_text_length_metric(text)
            rate = text_len / duration

            if is_cjk:
                min_rate = self.config.min_chars_per_sec_cjk
                max_rate = self.config.max_chars_per_sec_cjk
            else:
                min_rate = self.config.min_words_per_sec_latin
                max_rate = self.config.max_words_per_sec_latin

            if rate < min_rate or rate > max_rate:
                if override:
                    seg['flag_c_length_ratio'] = True
                    self.stats['rule_c_flagged'] += 1
                    filtered.append(seg)
                    logger.debug(
                        f"[规则C·标记] (速率={rate:.2f}/s): \"{text[:50]}\""
                    )
                else:
                    self.stats['rule_c_removed'] += 1
                    self.removed_segments.append({**seg, '_removed_by': 'rule_c', '_reason': f'速率={rate:.2f}/s'})
                    logger.debug(
                        f"[规则C] 移除 (速率={rate:.2f}/s, 时长={duration:.2f}s): "
                        f"\"{text[:50]}\""
                    )
            else:
                filtered.append(seg)
        return filtered

    # ── 规则 D: 上下文孤岛过滤 ─────────────────────────────
    def _rule_d_context_island(self, segments: List[dict]) -> List[dict]:
        """
        过滤在时间窗口内完全没有其他说话人出现的孤立片段。
        LLM 接管模式下仅标记 flag_d_context_island，不删除。
        """
        override = self.config.use_llm_override_mode
        n = len(segments)
        window = self.config.orphan_window_sec
        min_segs = self.config.min_segments_to_apply_d
        
        if n <= min_segs:
            logger.info(
                f"[规则D] 剩余 {n} 段 <= 安全阈值 {min_segs}，跳过孤岛过滤"
            )
            return segments

        all_speakers = set(seg.get('speaker', '') for seg in segments)
        all_speakers.discard('')
        if len(all_speakers) <= 1:
            logger.info("[规则D] 全文仅一个说话人，跳过孤岛过滤")
            return segments

        logger.info(
            f"[规则D] 启用孤岛过滤 | 参数: window={window}s, min_segments={min_segs}, "
            f"说话人数={len(all_speakers)}, 总段数={n}"
        )

        filtered = []

        for i, seg in enumerate(segments):
            cur_speaker = seg.get('speaker', '')
            cur_start = seg.get('start', 0)
            cur_end = seg.get('end', 0)
            cur_mid = (cur_start + cur_end) / 2.0

            min_distance_to_other = float('inf')
            
            for j, other in enumerate(segments):
                if i == j:
                    continue
                other_speaker = other.get('speaker', '')
                if other_speaker == cur_speaker or other_speaker == '':
                    continue
                other_start = other.get('start', 0)
                other_end = other.get('end', 0)
                other_mid = (other_start + other_end) / 2.0

                distance = abs(cur_mid - other_mid)
                min_distance_to_other = min(min_distance_to_other, distance)

            if min_distance_to_other > window:
                if override:
                    seg['flag_d_context_island'] = True
                    self.stats['rule_d_flagged'] += 1
                    filtered.append(seg)
                    logger.debug(
                        f"[规则D·标记] [{cur_speaker}] "
                        f"\"{seg.get('text', '')[:40]}...\" "
                        f"(距最近其他说话人 {min_distance_to_other:.1f}s > {window}s)"
                    )
                else:
                    self.stats['rule_d_removed'] += 1
                    self.removed_segments.append({**seg, '_removed_by': 'rule_d', '_reason': f'孤岛距离{min_distance_to_other:.1f}s'})
                    logger.debug(
                        f"[规则D] 移除孤岛段 | [说话人:{cur_speaker}] "
                        f"\"{seg.get('text', '')[:40]}...\" "
                        f"(距离最近其他说话人 {min_distance_to_other:.1f}s > {window}s)"
                    )
            else:
                filtered.append(seg)

        return filtered

    # ── 规则 E: 低信息量剔除 ────────────────────────────────
    def _rule_e_low_info(self, segments: List[dict]) -> List[dict]:
        """
        剔除持续时间极短且仅包含"嗯"、"啊"、"对对对"等无意义单字附和的无效交互。
        LLM 接管模式下仅标记 flag_e_low_info，不删除（自然附和可能是拟人化特征）。
        """
        override = self.config.use_llm_override_mode
        filtered = []
        for seg in segments:
            text = seg.get('text', '').strip()
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = end - start

            is_short = duration <= self.config.low_info_max_duration
            is_few_chars = len(text.replace(' ', '')) <= self.config.low_info_max_chars
            is_low_info = bool(_LOW_INFO_PATTERNS.match(text))

            if is_short and is_few_chars and is_low_info:
                if override:
                    seg['flag_e_low_info'] = True
                    self.stats['rule_e_flagged'] += 1
                    filtered.append(seg)
                    logger.debug(
                        f"[规则E·标记] \"{text}\" ({duration:.2f}s)"
                    )
                else:
                    self.stats['rule_e_removed'] += 1
                    self.removed_segments.append({**seg, '_removed_by': 'rule_e', '_reason': f'低信息{duration:.2f}s'})
                    logger.debug(
                        f"[规则E] 移除低信息: \"{text}\" ({duration:.2f}s)"
                    )
            else:
                filtered.append(seg)
        return filtered

    # ── 规则 F: 信噪比检测 ─────────────────────────────────
    def _rule_f_snr(self, segments: List[dict]) -> List[dict]:
        """
        基于 segment 附加的信噪比特征进行过滤。
        注意：此功能需要预先对音频进行 SNR 分析并写入 segment 的 'snr_db' 字段。
        如果 segment 无 snr_db 字段，则跳过该规则。
        """
        filtered = []
        for seg in segments:
            snr = seg.get('snr_db')
            if snr is not None and snr < self.config.min_snr_db:
                self.stats['rule_f_removed'] += 1
                self.removed_segments.append({**seg, '_removed_by': 'rule_f', '_reason': f'SNR={snr:.1f}dB'})
                logger.debug(
                    f"[规则F] 移除低SNR ({snr:.1f}dB): "
                    f"\"{seg.get('text', '')[:50]}\""
                )
            else:
                filtered.append(seg)
        return filtered


# ── 规则 G: 黑名单/脱轨内容过滤（独立函数，在特征计算后执行）────────

def load_blacklist(path: str = "") -> List[str]:
    """
    加载黑名单词库文件。
    
    Args:
        path: 词库文件路径。空字符串则使用默认路径 config/blacklist_words.txt
        
    Returns:
        关键词列表（已去重、去空、忽略注释行）
    """
    from pathlib import Path as _Path

    if not path:
        # 默认路径：项目根目录/config/blacklist_words.txt
        default = _Path(__file__).resolve().parent.parent.parent / "config" / "blacklist_words.txt"
        path = str(default)

    p = _Path(path)
    if not p.is_file():
        logger.warning(f"[规则G] 黑名单文件不存在: {path}")
        return []

    keywords = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            keywords.append(line)

    keywords = list(dict.fromkeys(keywords))  # 去重保序
    logger.info(f"[规则G] 加载黑名单词库: {len(keywords)} 个关键词 ← {p.name}")
    return keywords


def rule_g_blacklist_filter(
    segments: List[dict],
    blacklist: List[str],
    context_purge: bool = True,
) -> tuple:
    """
    规则 G: 基于黑名单关键词过滤广告/推广/脱轨内容。
    
    此规则在 compute_advanced_features() 之后执行，因为需要 conversation_id 字段。
    
    过滤逻辑：
    1. 扫描每条 segment 的 text，命中任意黑名单关键词则标记为"毒数据"
    2. 如果 context_purge=True, 将与命中句子相邻且同 conversation_id 的上下文一并丢弃
       （"相邻" = 在 segments 列表中与命中段连续且 conversation_id 相同）
    
    Args:
        segments:      带有 conversation_id 的 enriched segment 列表
        blacklist:     黑名单关键词列表
        context_purge: 是否连带清除同 conversation_id 的相邻上下文
        
    Returns:
        (filtered_segments, rule_g_removed_count)
    """
    if not blacklist or not segments:
        return segments, 0

    n = len(segments)

    # 第一步：标记命中段的索引
    hit_indices = set()
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        for keyword in blacklist:
            if keyword in text:
                hit_indices.add(i)
                logger.debug(
                    f"[规则G] 命中黑名单 \"{keyword}\": "
                    f"\"{text[:60]}...\""
                )
                break  # 一个关键词命中即可

    if not hit_indices:
        return segments, 0

    # 第二步：如果开启上下文清除，扩展丢弃范围
    purge_indices = set(hit_indices)

    if context_purge:
        for hit_idx in sorted(hit_indices):
            hit_conv_id = segments[hit_idx].get('conversation_id')
            if hit_conv_id is None:
                continue

            # 向前扩展：连续相邻且同 conversation_id
            j = hit_idx - 1
            while j >= 0 and segments[j].get('conversation_id') == hit_conv_id:
                purge_indices.add(j)
                j -= 1

            # 向后扩展：连续相邻且同 conversation_id
            j = hit_idx + 1
            while j < n and segments[j].get('conversation_id') == hit_conv_id:
                purge_indices.add(j)
                j += 1

    # 第三步：过滤
    filtered = [seg for i, seg in enumerate(segments) if i not in purge_indices]
    removed_count = len(purge_indices)

    if removed_count > 0:
        direct_hits = len(hit_indices)
        context_hits = removed_count - direct_hits
        logger.info(
            f"[规则G] 黑名单过滤: 直接命中 {direct_hits} 段"
            + (f" + 上下文连带 {context_hits} 段" if context_hits > 0 else "")
            + f" = 共移除 {removed_count} 段"
        )

    return filtered, removed_count


def rule_g_blacklist_soft_label(
    segments: List[dict],
    blacklist: List[str],
) -> tuple:
    """
    规则 G 软标签模式：仅标记命中黑名单的片段，不删除、不做上下文清除。
    LLM 接管模式下使用，由 LLM 最终决定是否保留。

    Returns:
        (segments, flagged_count) — segments 原地注入 flag_g_blacklist 字段
    """
    if not blacklist or not segments:
        return segments, 0

    flagged = 0
    for seg in segments:
        text = seg.get('text', '')
        for keyword in blacklist:
            if keyword in text:
                seg['flag_g_blacklist'] = True
                flagged += 1
                logger.debug(
                    f"[规则G·标记] 命中黑名单 \"{keyword}\": \"{text[:60]}...\""
                )
                break

    if flagged > 0:
        logger.info(f"[规则G·软标签] 标记 {flagged} 段命中黑名单")
    return segments, flagged
