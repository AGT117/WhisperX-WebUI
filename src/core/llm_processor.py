"""
LLM 集成模块：利用大语言模型实现智能字幕断句与翻译。

核心思路：
- 从 WhisperX 对齐结果中提取词级时间戳
- 将纯文本发送给 LLM 进行断句（和可选翻译）
- 将 LLM 返回的行级文本 **映射回** 词级时间戳
- 自动分批处理以适应 LLM 上下文窗口限制

支持的模式：
- segmentation : 仅智能断句
- translation  : 仅翻译（保留原文断句逻辑后追加译文）
- both         : 断句 + 翻译（双语字幕）
"""

import json
import re
import time
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── Token 估算 ──────────────────────────────────────────────────
_CHARS_PER_TOKEN_CJK = 1.5   # CJK 字符约 1.5 字/token
_CHARS_PER_TOKEN_LATIN = 4.0 # 拉丁字符约 4 字/token


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or
        (0x3400 <= cp <= 0x4DBF) or
        (0x3040 <= cp <= 0x309F) or
        (0x30A0 <= cp <= 0x30FF) or
        (0xAC00 <= cp <= 0xD7AF)
    )


def _estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数"""
    cjk = sum(1 for c in text if _is_cjk(c))
    latin = len(text) - cjk
    return max(1, int(cjk / _CHARS_PER_TOKEN_CJK + latin / _CHARS_PER_TOKEN_LATIN))


# ── Prompt 模板 ─────────────────────────────────────────────────

_SYS_SEGMENTATION = """\
你是专业字幕断句助手。将连续语音识别文本重新断句为自然的字幕行。

规则：
1. **不增删改**原文任何文字、标点或符号
2. 每行 CJK 控制在 12-25 字，英文控制在 40-80 字符
3. 在语义自然处断句（句末、从句边界、语气词后）
4. 返回 **纯 JSON 数组**，每个元素是一行字幕文本
5. 不要输出任何 JSON 以外的内容（无解释、无 markdown）

示例输入：
今天天气真好我们一起去公园散步吧你觉得怎么样

示例输出：
["今天天气真好","我们一起去公园散步吧","你觉得怎么样"]"""

_SYS_TRANSLATION = """\
你是专业字幕翻译助手。将语音识别文本翻译为{target_lang}。

规则：
1. 保留原文不修改，在翻译中体现自然断句
2. 返回 **纯 JSON 数组**，每个元素是对象: {{"text": "原文句", "translation": "译文"}}
3. 所有原文必须完整覆盖，不遗漏
4. 不要输出任何 JSON 以外的内容

示例输入：
今天天气真好我们一起去公园散步吧

示例输出：
[{{"text":"今天天气真好","translation":"The weather is great today"}},{{"text":"我们一起去公园散步吧","translation":"Let's go for a walk in the park"}}]"""

_SYS_BOTH = """\
你是专业字幕编辑与翻译助手。将语音识别文本断句并翻译为{target_lang}。

规则：
1. **不增删改**原文任何文字
2. 每行 CJK 控制在 12-25 字
3. 在语义自然处断句
4. 返回 **纯 JSON 数组**，每个元素: {{"text": "原文句", "translation": "译文"}}
5. 所有原文必须完整覆盖
6. 不要输出任何 JSON 以外的内容

示例输入：
今日は天気がいいですね一緒に公園を散歩しましょう

示例输出：
[{{"text":"今日は天気がいいですね","translation":"今天天气真好呢"}},{{"text":"一緒に公園を散歩しましょう","translation":"一起去公园散步吧"}}]"""

_SYS_SEMANTIC_FILTER = """\
你是一个严谨的语言学与语料库清洗专家。你的任务是分析以下提取自播客/访谈的对话文本，判断其是否适合用于训练"日常情感陪伴与拟人化AI"。
请严格以 JSON 格式输出评估结果，包含以下字段：
1. "is_ad_or_irrelevant" (bool): 是否包含商业软广、播客赞助鸣谢、或刻板的单向访谈。
2. "emotion_score" (int, 1-5): 情感丰富度。1为毫无感情/念稿，5为情绪饱满/强共情。
3. "hallucination_risk" (bool): 是否存在明显的语音识别错误导致的逻辑不通。
4. "reason" (string): 简短的判决理由（20字以内）。

输入格式：JSON 数组，每个元素有 "id"（会话组编号）和 "texts"（该组所有句子的数组）。

规则：
1. 返回 **纯 JSON 数组**，每个元素是对象: {"id": 组编号, "is_ad_or_irrelevant": bool, "emotion_score": int, "hallucination_risk": bool, "reason": "..."}
2. 必须对每个输入组都给出评估，不可遗漏
3. 无法确定时，给出保守评估（保留而非丢弃）
4. 不要输出任何 JSON 以外的内容（无解释、无 markdown）"""

_SYS_SEMANTIC_FILTER_OVERRIDE = """\
你是一个严谨的语言学与语料库清洗专家。你的任务是作为最终裁决者，分析以下对话文本是否适合用于训练"日常情感陪伴与拟人化AI"。

每组输入可能携带前置规则引擎的标记（flags），这些标记说明了该组被怀疑存在问题的原因：
- flag_c_length_ratio: 语速异常（太快或太慢），可能是识别噪音
- flag_d_context_island: 上下文孤立，周围没有对话互动
- flag_e_low_info: 极短的附和词（如"嗯""啊"），可能无信息量
- flag_g_blacklist: 命中了广告/推广黑名单关键词

你需要综合判断：这些标记是否准确？有些被标记的内容实际上可能是有价值的（比如自然附和体现情感、孤立段落但内容优质等）。

请输出以下字段：
1. "is_ad_or_irrelevant" (bool): 是否包含商业软广、播客赞助鸣谢、或刻板的单向访谈。
2. "emotion_score" (int, 1-5): 情感丰富度。1为毫无感情/念稿，5为情绪饱满/强共情。
3. "hallucination_risk" (bool): 是否存在明显的语音识别错误导致的逻辑不通。
4. "final_decision" (bool): **最终裁决**——true=保留该组, false=丢弃该组。请综合所有信息做出判断。
5. "reason" (string): 简短的判决理由（20字以内）。

输入格式：JSON 数组，每个元素有 "id"（会话组编号）、"texts"（句子数组）、"flags"（被触发的标记列表，可能为空）。

规则：
1. 返回 **纯 JSON 数组**，每个元素: {"id": 组编号, "is_ad_or_irrelevant": bool, "emotion_score": int, "hallucination_risk": bool, "final_decision": bool, "reason": "..."}
2. 必须对每个输入组都给出评估，不可遗漏
3. 对于无标记(flags为空)的组：通常应保留(final_decision=true)，除非内容确实有问题
4. 对于有标记的组：仔细审核标记是否合理，LLM 有权推翻规则引擎的判断
5. 不要输出任何 JSON 以外的内容（无解释、无 markdown）"""

_SYS_HALLUCINATION = """\
你是语音识别质量审核助手。判断以下语音识别片段是否为"幻觉"（即模型虚构的、实际音频中不存在的文本）。

常见幻觉特征：
- 与上下文完全无关的广告/频道推广语（如"请订阅""感谢观看"）
- 同一短语大量无意义重复
- 明显不属于该音频内容的模板化文本（字幕署名、版权声明等）
- 纯音乐/静音段被错误转录为文字

输入格式：JSON 数组，每个元素有 "id"（序号）和 "text"（识别文本）。

规则：
1. 返回 **纯 JSON 数组**，仅包含你判断为 **真实有效** 的 id 列表
2. 被你排除的 id 即为幻觉
3. 无法确定时保留（宁可多保留也不误删）
4. 不要输出任何 JSON 以外的内容

示例输入：
[{{"id":1,"text":"今天天气真好"}},{{"id":2,"text":"请订阅我的频道"}},{{"id":3,"text":"我们一起去公园"}}]

示例输出：
[1,3]"""


# ── 核心处理器 ───────────────────────────────────────────────────

class LLMProcessor:
    """
    LLM 字幕后处理器。

    通过 OpenAI 兼容 API 调用大语言模型对 WhisperX 输出进行
    智能断句和（可选）翻译。支持 OpenAI / DeepSeek / Ollama /
    vLLM 等所有兼容接口。
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_context_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        # 延迟导入，仅在实际使用时才需安装 openai
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "LLM 功能需要 openai 库，请执行: pip install openai"
            )

        # 确保 api_base 以 / 结尾时兼容
        self.client = OpenAI(base_url=api_base.rstrip("/"), api_key=api_key)
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature

    # ── 公共接口 ────────────────────────────────────────────

    def process_segments(
        self,
        segments: list,
        mode: str = "segmentation",
        target_lang: Optional[str] = None,
    ) -> list:
        """
        主入口：对 WhisperX segments 进行 LLM 智能处理。

        Args:
            segments:    WhisperX 对齐后的 segment 列表（含 words）
            mode:        "segmentation" | "translation" | "both"
            target_lang: 翻译目标语言（mode 含翻译时必填）

        Returns:
            处理后的 segment 列表，每个元素:
            {"start", "end", "text", "speaker"[, "translation"]}
        """
        all_words = self._flatten_words(segments)
        if not all_words:
            logger.warning("[LLM] 无可处理的词数据，跳过")
            return segments

        batches = self._create_batches(all_words)
        logger.info(
            f"[LLM] 共 {len(all_words)} 词 → {len(batches)} 批 "
            f"(ctx={self.max_context_tokens}, model={self.model})"
        )

        refined: list = []
        for idx, batch in enumerate(batches):
            logger.info(
                f"[LLM] 批次 {idx + 1}/{len(batches)} "
                f"({len(batch)} 词)"
            )
            try:
                result = self._process_batch(batch, mode, target_lang)
                refined.extend(result)
            except Exception as e:
                logger.error(
                    f"[LLM] 批次 {idx + 1} 失败: {e}，回退到原始文本"
                )
                refined.append(self._fallback_segment(batch))

        return refined

    def filter_hallucinations(self, segments: list) -> list:
        """
        使用 LLM 判断并过滤幻觉 segment。

        将所有 segment 的文本发送给 LLM，由其判断哪些是真实内容。
        自动分批以适应上下文窗口限制。

        Args:
            segments: WhisperX 对齐后的 segment 列表

        Returns:
            过滤后仅保留 LLM 判断为真实的 segments
        """
        if not segments:
            return segments

        # 构建带 id 的文本列表
        indexed = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                # 从 words 拼接
                words = seg.get("words", [])
                text = " ".join(w.get("word", "") for w in words).strip()
            indexed.append({"id": i, "text": text})

        # 按 token 预算分批
        sys_overhead = 600  # hallucination prompt 较长
        response_ratio = 0.3
        budget = int(
            (self.max_context_tokens - sys_overhead) * (1 - response_ratio)
        )
        budget = max(budget, 200)

        batches: List[List[dict]] = []
        cur_batch: List[dict] = []
        cur_tokens = 0
        for item in indexed:
            # JSON 格式开销 ~20 token per item
            item_tok = _estimate_tokens(item["text"]) + 20
            if cur_tokens + item_tok > budget and cur_batch:
                batches.append(cur_batch)
                cur_batch = []
                cur_tokens = 0
            cur_batch.append(item)
            cur_tokens += item_tok
        if cur_batch:
            batches.append(cur_batch)

        logger.info(
            f"[LLM·幻觉过滤] {len(segments)} 段 → {len(batches)} 批"
        )

        keep_ids: set = set()
        for idx, batch in enumerate(batches):
            logger.info(
                f"[LLM·幻觉过滤] 批次 {idx + 1}/{len(batches)} ({len(batch)} 段)"
            )
            try:
                user_prompt = json.dumps(batch, ensure_ascii=False)
                raw = self._call_api(_SYS_HALLUCINATION, user_prompt)
                parsed = self._extract_json(raw)

                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, int):
                            keep_ids.add(item)
                        elif isinstance(item, dict):
                            keep_ids.add(item.get("id", -1))
                else:
                    # 解析异常，保留该批所有
                    keep_ids.update(item["id"] for item in batch)
            except Exception as e:
                logger.error(
                    f"[LLM·幻觉过滤] 批次 {idx + 1} 失败: {e}，保留全部"
                )
                keep_ids.update(item["id"] for item in batch)

        filtered = [seg for i, seg in enumerate(segments) if i in keep_ids]
        removed = len(segments) - len(filtered)
        if removed > 0:
            logger.info(
                f"[LLM·幻觉过滤] 移除 {removed} 段，保留 {len(filtered)} 段"
            )
        return filtered

    def semantic_filter(
        self,
        segments: list,
        emotion_threshold: int = 3,
        concurrency: int = 3,
        override_mode: bool = False,
        progress_callback=None,
    ) -> tuple:
        """
        规则 H: LLM 语义校验 — 对语料进行情感丰富度、软广识别、幻觉检测。

        按 conversation_id 分组打包发送，结合上下文判断。
        使用线程池实现并发 API 调用。

        Args:
            segments:          带有 conversation_id 的 enriched segment 列表
            emotion_threshold: 情感得分阈值 (1-5)，低于此值的组将被丢弃
            concurrency:       并发 API 请求数
            override_mode:     LLM 接管模式 — 传入 flag 字段，由 LLM 做 final_decision
            progress_callback: 可选进度回调 callable(fraction, desc)

        Returns:
            (filtered_segments, stats_dict)
            stats_dict 包含: removed_count, unchecked_count, overridden_count, details
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        if not segments:
            return segments, {'removed_count': 0, 'unchecked_count': 0, 'overridden_count': 0, 'details': []}

        # 选择系统 Prompt
        sys_prompt = _SYS_SEMANTIC_FILTER_OVERRIDE if override_mode else _SYS_SEMANTIC_FILTER

        # 收集每组的 flags（接管模式下）
        _FLAG_KEYS = ('flag_c_length_ratio', 'flag_d_context_island', 'flag_e_low_info', 'flag_g_blacklist')

        # 1. 按 conversation_id 分组
        groups = {}
        group_flags = {}  # cid -> set of flag names
        for seg in segments:
            cid = seg.get('conversation_id', 0)
            groups.setdefault(cid, []).append(seg)
            if override_mode:
                flags = group_flags.setdefault(cid, set())
                for fk in _FLAG_KEYS:
                    if seg.get(fk):
                        flags.add(fk)

        sorted_cids = sorted(groups.keys())
        logger.info(
            f"[LLM·语义过滤] {len(segments)} 段 → {len(sorted_cids)} 组 "
            f"(emotion_threshold={emotion_threshold}, concurrency={concurrency})"
        )

        # 2. 构建 API 请求批次（按 token 预算分组）
        sys_overhead = 800
        response_ratio = 0.35
        budget = int(
            (self.max_context_tokens - sys_overhead) * (1 - response_ratio)
        )
        budget = max(budget, 500)

        api_batches = []  # 每个元素: [(cid, texts), ...]
        cur_batch = []
        cur_tokens = 0

        for cid in sorted_cids:
            segs = groups[cid]
            texts = [s.get('text', '').strip() for s in segs if s.get('text', '').strip()]
            combined = " ".join(texts)
            item_tokens = _estimate_tokens(combined) + 30  # JSON 格式开销

            if cur_tokens + item_tokens > budget and cur_batch:
                api_batches.append(cur_batch)
                cur_batch = []
                cur_tokens = 0

            cur_batch.append((cid, texts))
            cur_tokens += item_tokens

        if cur_batch:
            api_batches.append(cur_batch)

        logger.info(f"[LLM·语义过滤] {len(sorted_cids)} 组 → {len(api_batches)} 个API批次")

        # 3. 同步调用单批次
        def _call_batch(batch_items):
            """调用 LLM 评估一个批次，返回 {cid: result_dict}"""
            input_data = []
            for cid, texts in batch_items:
                item = {"id": cid, "texts": texts}
                if override_mode:
                    item["flags"] = sorted(group_flags.get(cid, []))
                input_data.append(item)

            user_prompt = json.dumps(input_data, ensure_ascii=False)
            results = {}

            try:
                raw = self._call_api(sys_prompt, user_prompt)
                parsed = self._extract_json(raw)

                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            item_id = item.get("id")
                            if item_id is not None:
                                result = {
                                    "is_ad_or_irrelevant": bool(item.get("is_ad_or_irrelevant", False)),
                                    "emotion_score": int(item.get("emotion_score", 3)),
                                    "hallucination_risk": bool(item.get("hallucination_risk", False)),
                                    "reason": str(item.get("reason", ""))[:50],
                                    "llm_status": "checked",
                                }
                                if override_mode:
                                    result["final_decision"] = bool(item.get("final_decision", True))
                                results[item_id] = result
            except Exception as e:
                logger.error(f"[LLM·语义过滤] API批次失败: {e}")

            # 未返回结果的组标记为 UNCHECKED（降级保留）
            for cid, _ in batch_items:
                if cid not in results:
                    results[cid] = {
                        "is_ad_or_irrelevant": False,
                        "emotion_score": 3,
                        "hallucination_risk": False,
                        "reason": "API未返回",
                        "llm_status": "unchecked",
                    }

            return results

        # 4. 使用线程池并发调用
        all_results = {}  # cid -> result_dict
        total_batches = len(api_batches)

        def _run_all():
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_call_batch, batch): idx
                           for idx, batch in enumerate(api_batches)}
                for future in futures:
                    idx = futures[future]
                    try:
                        batch_results = future.result()
                        all_results.update(batch_results)
                    except Exception as e:
                        logger.error(f"[LLM·语义过滤] 线程异常 (批次{idx}): {e}")
                    if progress_callback:
                        done = len(all_results)
                        progress_callback(done / max(len(sorted_cids), 1),
                                          f"语义过滤 {done}/{len(sorted_cids)} 组")

        # 在同步上下文中运行
        _run_all()

        # 5. 根据评估结果过滤 + 注入语义标签
        filtered = []
        removed_count = 0
        unchecked_count = 0
        overridden_count = 0
        details = []

        for cid in sorted_cids:
            result = all_results.get(cid, {
                "is_ad_or_irrelevant": False,
                "emotion_score": 3,
                "hallucination_risk": False,
                "reason": "未评估",
                "llm_status": "unchecked",
            })

            if result.get("llm_status") == "unchecked":
                unchecked_count += len(groups[cid])

            details.append({"conversation_id": cid, **result})

            # 判定是否过滤
            if override_mode:
                # 接管模式：由 LLM 的 final_decision 决定
                should_remove = not result.get("final_decision", True)
                # UNCHECKED 的组降级保留
                if result.get("llm_status") == "unchecked":
                    should_remove = False
                # 统计被 flag 但被 LLM 保留的翻转次数
                has_flags = bool(group_flags.get(cid))
                if has_flags and not should_remove and result.get("llm_status") == "checked":
                    overridden_count += len(groups[cid])
            else:
                # 传统模式：按规则组合判定
                should_remove = (
                    result.get("is_ad_or_irrelevant", False)
                    or result.get("emotion_score", 3) < emotion_threshold
                    or result.get("hallucination_risk", False)
                )
                if result.get("llm_status") == "unchecked":
                    should_remove = False

            if should_remove:
                removed_count += len(groups[cid])
                for seg in groups[cid]:
                    logger.debug(
                        f"[规则H] 移除 (cid={cid}, "
                        f"ad={result.get('is_ad_or_irrelevant')}, "
                        f"emo={result.get('emotion_score')}, "
                        f"reason={result.get('reason')}): "
                        f"\"{seg.get('text', '')[:40]}\""
                    )
            else:
                # 注入语义标签到每个 segment
                for seg in groups[cid]:
                    seg['emotion_score'] = result.get('emotion_score', 3)
                    seg['is_ad'] = result.get('is_ad_or_irrelevant', False)
                    seg['hallucination_risk'] = result.get('hallucination_risk', False)
                    seg['llm_status'] = result.get('llm_status', 'unchecked')
                    filtered.append(seg)

        logger.info(
            f"[LLM·语义过滤] 完成: 保留 {len(filtered)} 段, "
            f"移除 {removed_count} 段, 未检查 {unchecked_count} 段"
        )

        stats = {
            'removed_count': removed_count,
            'unchecked_count': unchecked_count,
            'overridden_count': overridden_count,
            'details': details,
        }
        return filtered, stats

    # ── 内部方法 ────────────────────────────────────────────

    def _flatten_words(self, segments: list) -> list:
        """从 segments 提取词级列表，保留 speaker"""
        words = []
        for seg in segments:
            spk = seg.get("speaker", "")
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    entry = dict(w)
                    if spk:
                        entry["speaker"] = spk
                    words.append(entry)
        return words

    def _create_batches(self, words: list) -> List[list]:
        """
        按 token 预算将词列表切分为多批。

        预算分配：
        - system prompt ≈ 500 tokens
        - 响应预留 40%
        - 剩余给 user prompt（即词文本）
        """
        sys_overhead = 500
        response_ratio = 0.4
        budget = int(
            (self.max_context_tokens - sys_overhead) * (1 - response_ratio)
        )
        budget = max(budget, 200)  # 保底

        batches: List[list] = []
        cur_batch: list = []
        cur_tokens = 0

        for w in words:
            w_text = w.get("word", "")
            # 每词额外 ~3 token 开销（索引/空格）
            w_tok = _estimate_tokens(w_text) + 3

            if cur_tokens + w_tok > budget and cur_batch:
                batches.append(cur_batch)
                cur_batch = []
                cur_tokens = 0

            cur_batch.append(w)
            cur_tokens += w_tok

        if cur_batch:
            batches.append(cur_batch)

        return batches

    def _process_batch(
        self,
        words: list,
        mode: str,
        target_lang: Optional[str],
    ) -> list:
        """处理单批词数据"""
        # 1. 拼接纯文本（CJK 不加空格，拉丁加空格）
        plain_text = self._words_to_text(words)

        # 2. 构建提示词
        sys_prompt = self._build_system_prompt(mode, target_lang)
        user_prompt = plain_text

        # 3. 调用 LLM
        raw_response = self._call_api(sys_prompt, user_prompt)

        # 4. 解析 JSON
        parsed = self._extract_json(raw_response)

        # 5. 映射回时间戳
        if mode == "segmentation":
            lines = self._normalize_segmentation(parsed)
            return self._match_lines_to_words(lines, words)
        else:
            pairs = self._normalize_translation(parsed)
            return self._match_pairs_to_words(pairs, words)

    def _words_to_text(self, words: list) -> str:
        """将词列表拼接为纯文本（自动处理 CJK）"""
        if not words:
            return ""
        parts = [words[0].get("word", "").strip()]
        for i in range(1, len(words)):
            prev = words[i - 1].get("word", "").strip()
            curr = words[i].get("word", "").strip()
            if not prev or not curr:
                parts.append(curr)
                continue
            p_tail, c_head = prev[-1], curr[0]
            if _is_cjk(p_tail) or _is_cjk(c_head):
                parts.append(curr)
            else:
                parts.append(" " + curr)
        return "".join(parts)

    def _build_system_prompt(
        self, mode: str, target_lang: Optional[str]
    ) -> str:
        if mode == "segmentation":
            return _SYS_SEGMENTATION
        elif mode == "translation":
            lang = target_lang or "English"
            return _SYS_TRANSLATION.format(target_lang=lang)
        else:  # both
            lang = target_lang or "English"
            return _SYS_BOTH.format(target_lang=lang)

    def _call_api(
        self, system_prompt: str, user_prompt: str, max_retries: int = 3
    ) -> str:
        """调用 LLM，带指数退避重试"""
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                content = resp.choices[0].message.content
                if content:
                    return content.strip()
                raise ValueError("LLM 返回空内容")
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"[LLM] API 调用失败 ({attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM API 连续 {max_retries} 次失败: {last_exc}")

    # ── JSON 解析 ───────────────────────────────────────────

    def _extract_json(self, text: str) -> Any:
        """
        从 LLM 响应中提取 JSON。

        处理常见情况：
        - 纯 JSON
        - ```json ... ``` 代码块
        - 前后有多余文字
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json 代码块
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试找到第一个 [ 到最后一个 ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从 LLM 响应中解析 JSON: {text[:200]}")

    def _normalize_segmentation(self, parsed: Any) -> List[str]:
        """规范化断句模式的 JSON → 字符串列表"""
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    result.append(item.get("text", str(item)))
                else:
                    result.append(str(item))
            return [s for s in result if s.strip()]
        raise ValueError(f"期望 JSON 数组，收到: {type(parsed)}")

    def _normalize_translation(
        self, parsed: Any
    ) -> List[Dict[str, str]]:
        """规范化翻译模式的 JSON → [{text, translation}] 列表"""
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict):
                    result.append(
                        {
                            "text": item.get("text", ""),
                            "translation": item.get("translation", ""),
                        }
                    )
                elif isinstance(item, str):
                    result.append({"text": item, "translation": ""})
            return [p for p in result if p["text"].strip()]
        raise ValueError(f"期望 JSON 数组，收到: {type(parsed)}")

    # ── 时间戳映射 ──────────────────────────────────────────

    def _match_lines_to_words(
        self, lines: List[str], words: list
    ) -> list:
        """
        将 LLM 返回的文本行映射回词级时间戳。

        使用贪心字符匹配：逐字符扫描每行文本，消耗对应的词。
        """
        segments = []
        ptr = 0  # 词指针

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            target_chars = re.sub(r"\s+", "", line_stripped)
            seg_start = ptr
            matched_len = 0

            while ptr < len(words) and matched_len < len(target_chars):
                w_text = re.sub(
                    r"\s+", "", words[ptr].get("word", "")
                )
                matched_len += len(w_text)
                ptr += 1

            if seg_start < ptr:
                segments.append(
                    {
                        "start": words[seg_start].get("start", 0.0),
                        "end": words[ptr - 1].get("end", 0.0),
                        "text": line_stripped,
                        "speaker": words[seg_start].get("speaker", ""),
                    }
                )

        # 处理 LLM 可能遗漏的尾部词
        if ptr < len(words):
            tail_text = self._words_to_text(words[ptr:])
            if tail_text.strip():
                segments.append(
                    {
                        "start": words[ptr].get("start", 0.0),
                        "end": words[-1].get("end", 0.0),
                        "text": tail_text.strip(),
                        "speaker": words[ptr].get("speaker", ""),
                    }
                )

        return segments

    def _match_pairs_to_words(
        self, pairs: List[Dict[str, str]], words: list
    ) -> list:
        """同 _match_lines_to_words，但附带翻译"""
        segments = []
        ptr = 0

        for pair in pairs:
            text = pair["text"].strip()
            translation = pair.get("translation", "").strip()
            if not text:
                continue

            target_chars = re.sub(r"\s+", "", text)
            seg_start = ptr
            matched_len = 0

            while ptr < len(words) and matched_len < len(target_chars):
                w_text = re.sub(
                    r"\s+", "", words[ptr].get("word", "")
                )
                matched_len += len(w_text)
                ptr += 1

            if seg_start < ptr:
                seg = {
                    "start": words[seg_start].get("start", 0.0),
                    "end": words[ptr - 1].get("end", 0.0),
                    "text": text,
                    "speaker": words[seg_start].get("speaker", ""),
                }
                if translation:
                    seg["translation"] = translation
                segments.append(seg)

        if ptr < len(words):
            tail_text = self._words_to_text(words[ptr:])
            if tail_text.strip():
                segments.append(
                    {
                        "start": words[ptr].get("start", 0.0),
                        "end": words[-1].get("end", 0.0),
                        "text": tail_text.strip(),
                        "speaker": words[ptr].get("speaker", ""),
                    }
                )

        return segments

    def _fallback_segment(self, words: list) -> dict:
        """API 失败时的回退：保留原始词拼接"""
        return {
            "start": words[0].get("start", 0.0),
            "end": words[-1].get("end", 0.0),
            "text": self._words_to_text(words),
            "words": words,
            "speaker": words[0].get("speaker", ""),
        }
