"""
JSONL 语料库导出模块：将清洗后的 segments 转换为标准化的训练数据格式。

输出字段说明（对应毕设文档第五章）：
- audio_path:         音频文件路径
- text:               转录文本
- speaker_id:         说话人编号
- confidence_score:   文本识别置信度（词级平均）
- duration:           当前语句持续时间（秒）
- words_per_second:   语速特征 — 每秒字/词数（CJK按字数，拉丁按词数）
- turn_delay:         回合延迟 — Speaker A 说完到 Speaker B 开口的毫秒数（超过阈值则 null）
- interaction_type:   交互类型 — "long_narrative" / "short_reply" / "quick_exchange" / "normal"
- conversation_id:    会话编号 — 同一段连续对话共享相同 ID
- contains_overlap:   是否与前一句存在时间重叠（抢话标记）
- start:              起始时间戳（秒）
- end:                结束时间戳（秒）
- emotion_score:      （可选）LLM 语义过滤 — 情感丰富度评分 (1-5)
- is_ad:              （可选）LLM 语义过滤 — 是否为软广/无关内容
- hallucination_risk: （可选）LLM 语义过滤 — 是否存在 ASR 幻觉风险
- llm_status:         （可选）LLM 语义过滤 — "checked" / "unchecked"
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── 交互类型判定参数 ────────────────────────────────────────
# 长段叙述: 单条 segment 持续时间超过此值(秒)
LONG_NARRATIVE_DURATION = 15.0
# 短句回复: 单条 segment 持续时间低于此值(秒)
SHORT_REPLY_DURATION = 5.0
# 快问快答: 滑动窗口内平均每轮持续时间低于此值(秒)
QUICK_EXCHANGE_AVG_DURATION = 5.0
# 快问快答: 滑动窗口内平均 turn_delay 低于此值(毫秒)
QUICK_EXCHANGE_AVG_DELAY_MS = 2000.0

# ── 上下文断裂阈值 ─────────────────────────────────────────
# turn_delay 超过此值(毫秒)视为上下文断裂，设为 null
CONTEXT_BREAK_THRESHOLD_MS = 5000.0
# 连续段间距超过此值(秒)时递增 conversation_id
CONVERSATION_GAP_SEC = 10.0


def _calc_segment_confidence(segment: dict) -> float:
    """计算 segment 所有词的平均置信度"""
    words = segment.get('words', [])
    scores = [w.get('score', 0.0) for w in words if 'score' in w]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _is_cjk_text(text: str) -> bool:
    """判断文本是否以 CJK 字符为主"""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff' or
                    '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff' or
                    '\uac00' <= ch <= '\ud7af')
    non_space = sum(1 for ch in text if not ch.isspace())
    return non_space > 0 and (cjk_count / non_space) > 0.3


def _calc_words_per_second(text: str, duration: float) -> Optional[float]:
    """计算语速: CJK 按字符数/秒, 拉丁按词数/秒"""
    if duration <= 0:
        return None
    text = text.strip()
    if not text:
        return 0.0
    if _is_cjk_text(text):
        count = sum(1 for ch in text if not ch.isspace())
    else:
        count = len(text.split())
    return round(count / duration, 2)


def compute_advanced_features(segments: List[dict]) -> List[dict]:
    """
    为每个 segment 计算高级特征字段：
    - confidence_score        置信度
    - duration                持续时间
    - words_per_second        语速
    - turn_delay (ms)         回合延迟（超过阈值自动置 null）
    - interaction_type        交互类型
    - conversation_id         会话编号
    - contains_overlap        是否存在抢话重叠
    
    Args:
        segments: 清洗后的 segment 列表
        
    Returns:
        带有高级特征字段的 segment 列表
    """
    enriched = []
    conversation_id = 0

    for i, seg in enumerate(segments):
        entry = dict(seg)  # 浅拷贝

        # ── 基础特征 ──────────────────────────────────────
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        duration = round(end - start, 3)
        text = seg.get('text', '').strip()
        entry['duration'] = duration
        entry['confidence_score'] = _calc_segment_confidence(seg)
        entry['words_per_second'] = _calc_words_per_second(text, duration)

        # ── 回合延迟 + 抢话检测 ──────────────────────────
        contains_overlap = False
        if i > 0:
            prev_end = segments[i - 1].get('end', 0.0)
            prev_speaker = segments[i - 1].get('speaker', '')
            cur_speaker = seg.get('speaker', '')

            # 抢话检测: 当前句 start < 上一句 end
            if start < prev_end:
                contains_overlap = True

            # 计算与上一段的时间间距（秒），用于 conversation_id
            gap_sec = start - prev_end

            if prev_speaker != cur_speaker:
                delay_ms = round((start - prev_end) * 1000, 1)
                delay_ms = max(delay_ms, 0.0)
                # 超过阈值 → 上下文断裂，设为 null
                if delay_ms > CONTEXT_BREAK_THRESHOLD_MS:
                    entry['turn_delay'] = None
                else:
                    entry['turn_delay'] = delay_ms
            else:
                entry['turn_delay'] = None

            # 会话编号: 间距超过阈值或不同说话人间距过大 → 新会话
            if gap_sec > CONVERSATION_GAP_SEC:
                conversation_id += 1
        else:
            entry['turn_delay'] = None

        entry['contains_overlap'] = contains_overlap
        entry['conversation_id'] = conversation_id
        enriched.append(entry)

    # 交互类型标注（需要全局上下文）
    _annotate_interaction_types(enriched)

    return enriched


def _annotate_interaction_types(segments: List[dict]):
    """
    标注每个 segment 的交互类型:
    - "long_narrative":  长段叙述 — 单条 segment 持续 > 15s
    - "short_reply":     短句回复 — 单条 segment 持续 < 5s
    - "quick_exchange":  快问快答 — 多人快速交替的窗口
    - "normal":          常规对话 — 以上都不满足
    """
    n = len(segments)
    if n == 0:
        return

    # 第一遍：根据单条 segment 的持续时间标注
    for seg in segments:
        dur = seg.get('duration', 0.0)
        if dur >= LONG_NARRATIVE_DURATION:
            seg['interaction_type'] = 'long_narrative'
        elif dur <= SHORT_REPLY_DURATION:
            seg['interaction_type'] = 'short_reply'
        else:
            seg['interaction_type'] = None  # 待定

    # 第二遍：检测快问快答窗口（滑动窗口 5 个 segments）
    window_size = 5
    for i in range(n - window_size + 1):
        window = segments[i:i + window_size]
        delays = [s.get('turn_delay') for s in window
                  if s.get('turn_delay') is not None]
        durations = [s.get('duration', 0) for s in window]
        speakers = [s.get('speaker', '') for s in window]
        unique_speakers = len(set(speakers))

        if (
            unique_speakers >= 2
            and len(delays) >= 2
            and sum(durations) / len(durations) <= QUICK_EXCHANGE_AVG_DURATION
            and sum(delays) / len(delays) <= QUICK_EXCHANGE_AVG_DELAY_MS
        ):
            for k in range(i, i + window_size):
                # quick_exchange 优先级高于 short_reply
                if segments[k].get('interaction_type') in (None, 'short_reply'):
                    segments[k]['interaction_type'] = 'quick_exchange'

    # 未标注的默认为 normal
    for seg in segments:
        if seg.get('interaction_type') is None:
            seg['interaction_type'] = 'normal'


def export_jsonl(
    segments: List[dict],
    output_path: str,
    audio_source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    将处理后的 segments 导出为标准化 JSONL 训练文件。
    
    Args:
        segments:     带有高级特征的 segment 列表
        output_path:  JSONL 输出文件路径
        audio_source: 源音频文件路径（写入每条记录）
        metadata:     额外元数据（如视频标题、发布时间等）
        
    Returns:
        输出文件的绝对路径
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output, 'w', encoding='utf-8') as f:
        for seg in segments:
            record = {
                'audio_path': audio_source,
                'text': seg.get('text', '').strip(),
                'speaker_id': seg.get('speaker', ''),
                'confidence_score': seg.get('confidence_score', 0.0),
                'start': seg.get('start', 0.0),
                'end': seg.get('end', 0.0),
                'duration': seg.get('duration', 0.0),
                'words_per_second': seg.get('words_per_second'),
                'turn_delay': seg.get('turn_delay'),
                'interaction_type': seg.get('interaction_type', 'normal'),
                'conversation_id': seg.get('conversation_id', 0),
                'contains_overlap': seg.get('contains_overlap', False),
            }
            # 附加 LLM 语义标签（规则 H 注入）
            if 'emotion_score' in seg:
                record['emotion_score'] = seg['emotion_score']
                record['is_ad'] = seg.get('is_ad', False)
                record['hallucination_risk'] = seg.get('hallucination_risk', False)
                record['llm_status'] = seg.get('llm_status', 'unchecked')
            # 附加软标签 flags（LLM 接管模式消融分析用）
            for fk in ('flag_c_length_ratio', 'flag_d_context_island',
                        'flag_e_low_info', 'flag_g_blacklist'):
                if seg.get(fk):
                    record[fk] = True
            # 附加元数据
            if metadata:
                record['metadata'] = metadata

            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    logger.info(f"[JSONL导出] 写入 {count} 条记录 → {output}")
    return str(output.resolve())


def merge_jsonl_files(
    input_paths: List[str],
    output_path: str,
) -> str:
    """
    合并多个 JSONL 文件为一个数据集文件。
    用于批量处理后的数据集汇总。
    
    Args:
        input_paths: JSONL 文件路径列表
        output_path: 合并输出路径
        
    Returns:
        合并后文件的绝对路径
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    with open(output, 'w', encoding='utf-8') as out_f:
        for path in input_paths:
            try:
                with open(path, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + '\n')
                            total += 1
            except Exception as e:
                logger.error(f"[JSONL合并] 读取失败: {path} — {e}")

    logger.info(f"[JSONL合并] 合并 {len(input_paths)} 个文件, 共 {total} 条 → {output}")
    return str(output.resolve())
