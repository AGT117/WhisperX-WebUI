"""
LLM Processor 单元测试。

使用 Mock 模拟 OpenAI API 调用，测试核心逻辑：
- Token 估算
- 批次划分
- JSON 解析
- 文本 → 词时间戳映射
- 断句模式 / 翻译模式 / 双语模式
- 错误回退
"""
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_processor import (
    LLMProcessor,
    _estimate_tokens,
    _is_cjk,
)


# ── 辅助数据 ────────────────────────────────────────────────────

def _make_words(word_list):
    """从 [(text, start, end), ...] 创建词列表"""
    return [
        {"word": w, "start": s, "end": e, "score": 0.9}
        for w, s, e in word_list
    ]


def _make_segments(word_list, speaker=""):
    """构造含 words 的 segment 列表"""
    seg = {"words": _make_words(word_list), "text": ""}
    if speaker:
        seg["speaker"] = speaker
    return [seg]


# ── 基础工具测试 ─────────────────────────────────────────────────

class TestEstimateTokens:
    def test_english_text(self):
        tokens = _estimate_tokens("Hello world, this is a test.")
        assert 5 <= tokens <= 15

    def test_cjk_text(self):
        tokens = _estimate_tokens("今天天气真好我们一起去公园散步")
        assert 8 <= tokens <= 20

    def test_empty(self):
        assert _estimate_tokens("") == 1  # max(1, ...)


class TestIsCjk:
    def test_chinese(self):
        assert _is_cjk("中") is True

    def test_hiragana(self):
        assert _is_cjk("あ") is True

    def test_latin(self):
        assert _is_cjk("A") is False

    def test_digit(self):
        assert _is_cjk("1") is False


# ── Mock OpenAI Client ──────────────────────────────────────────

class MockOpenAI:
    """模拟 OpenAI 客户端"""
    def __init__(self, responses=None):
        self.responses = responses or []
        self._call_count = 0
        self.chat = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, **kwargs):
        idx = min(self._call_count, len(self.responses) - 1)
        self._call_count += 1
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = self.responses[idx]
        return resp


def _make_processor(responses, max_tokens=4096):
    """创建使用 Mock 客户端的 LLMProcessor"""
    with patch("src.core.llm_processor.LLMProcessor.__init__", lambda self, **kw: None):
        proc = LLMProcessor.__new__(LLMProcessor)
        proc.client = MockOpenAI(responses)
        proc.model = "test-model"
        proc.max_context_tokens = max_tokens
        proc.temperature = 0.3
    return proc


# ── JSON 解析测试 ────────────────────────────────────────────────

class TestExtractJson:
    def setup_method(self):
        self.proc = _make_processor([])

    def test_plain_json(self):
        result = self.proc._extract_json('["hello", "world"]')
        assert result == ["hello", "world"]

    def test_code_block(self):
        text = '```json\n["hello", "world"]\n```'
        result = self.proc._extract_json(text)
        assert result == ["hello", "world"]

    def test_with_surrounding_text(self):
        text = 'Sure! Here is the result:\n["hello", "world"]\nHope it helps.'
        result = self.proc._extract_json(text)
        assert result == ["hello", "world"]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="无法从 LLM"):
            self.proc._extract_json("not json at all")


# ── 批次划分测试 ────────────────────────────────────────────────

class TestCreateBatches:
    def test_small_input_single_batch(self):
        proc = _make_processor([], max_tokens=4096)
        words = _make_words([("hello", 0, 1), ("world", 1, 2)])
        batches = proc._create_batches(words)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_large_input_multiple_batches(self):
        proc = _make_processor([], max_tokens=200)
        # 很多词，小上下文 → 应该分多批
        words = _make_words([(f"word{i}", i, i + 1) for i in range(100)])
        batches = proc._create_batches(words)
        assert len(batches) > 1
        # 验证所有词都被分配到某个批次
        total = sum(len(b) for b in batches)
        assert total == 100


# ── 词文本拼接测试 ──────────────────────────────────────────────

class TestWordsToText:
    def setup_method(self):
        self.proc = _make_processor([])

    def test_cjk_no_spaces(self):
        words = _make_words([("今天", 0, 1), ("天气", 1, 2), ("真好", 2, 3)])
        assert self.proc._words_to_text(words) == "今天天气真好"

    def test_latin_with_spaces(self):
        words = _make_words([("Hello", 0, 1), ("world", 1, 2)])
        assert self.proc._words_to_text(words) == "Hello world"

    def test_mixed(self):
        words = _make_words([("Hello", 0, 1), ("世界", 1, 2)])
        assert self.proc._words_to_text(words) == "Hello世界"

    def test_empty(self):
        assert self.proc._words_to_text([]) == ""


# ── 时间戳映射测试 ──────────────────────────────────────────────

class TestMatchLinesToWords:
    def setup_method(self):
        self.proc = _make_processor([])

    def test_simple_segmentation(self):
        words = _make_words([
            ("今天", 0.0, 0.5), ("天气", 0.5, 1.0), ("真好", 1.0, 1.5),
            ("我们", 2.0, 2.5), ("一起", 2.5, 3.0), ("去", 3.0, 3.2),
        ])
        lines = ["今天天气真好", "我们一起去"]
        segs = self.proc._match_lines_to_words(lines, words)
        assert len(segs) == 2
        assert segs[0]["start"] == 0.0
        assert segs[0]["end"] == 1.5
        assert segs[0]["text"] == "今天天气真好"
        assert segs[1]["start"] == 2.0
        assert segs[1]["end"] == 3.2

    def test_english_segmentation(self):
        words = _make_words([
            ("Hello", 0.0, 0.5), ("world.", 0.5, 1.0),
            ("How", 1.5, 2.0), ("are", 2.0, 2.3), ("you?", 2.3, 2.8),
        ])
        lines = ["Hello world.", "How are you?"]
        segs = self.proc._match_lines_to_words(lines, words)
        assert len(segs) == 2
        assert segs[0]["text"] == "Hello world."
        assert segs[1]["start"] == 1.5

    def test_tail_words_become_extra_segment(self):
        words = _make_words([
            ("一", 0, 1), ("二", 1, 2), ("三", 2, 3), ("四", 3, 4),
        ])
        lines = ["一二"]  # LLM 只返回了部分
        segs = self.proc._match_lines_to_words(lines, words)
        assert len(segs) == 2  # "一二" + "三四"(tail)
        assert segs[1]["text"] == "三四"

    def test_empty_lines_skipped(self):
        words = _make_words([("ok", 0, 1)])
        lines = ["", "  ", "ok"]
        segs = self.proc._match_lines_to_words(lines, words)
        assert len(segs) == 1


# ── 翻译 pair 映射测试 ─────────────────────────────────────────

class TestMatchPairsToWords:
    def setup_method(self):
        self.proc = _make_processor([])

    def test_translation_mapping(self):
        words = _make_words([
            ("今天", 0.0, 0.5), ("天气", 0.5, 1.0), ("真好", 1.0, 1.5),
            ("我们", 2.0, 2.5), ("去", 2.5, 3.0),
        ])
        pairs = [
            {"text": "今天天气真好", "translation": "The weather is great"},
            {"text": "我们去", "translation": "Let's go"},
        ]
        segs = self.proc._match_pairs_to_words(pairs, words)
        assert len(segs) == 2
        assert segs[0]["translation"] == "The weather is great"
        assert segs[0]["start"] == 0.0
        assert segs[1]["translation"] == "Let's go"
        assert segs[1]["start"] == 2.0


# ── 端到端 process_segments 测试 ───────────────────────────────

class TestProcessSegments:
    def test_segmentation_mode(self):
        """断句模式端到端"""
        llm_response = json.dumps(["今天天气真好", "我们一起去"])
        proc = _make_processor([llm_response])

        segments = _make_segments([
            ("今天", 0.0, 0.5), ("天气", 0.5, 1.0), ("真好", 1.0, 1.5),
            ("我们", 2.0, 2.5), ("一起", 2.5, 3.0), ("去", 3.0, 3.2),
        ])

        result = proc.process_segments(segments, mode="segmentation")
        assert len(result) == 2
        assert result[0]["text"] == "今天天气真好"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.5
        assert result[1]["text"] == "我们一起去"

    def test_translation_mode(self):
        """翻译模式端到端"""
        llm_response = json.dumps([
            {"text": "今天天气真好", "translation": "Great weather today"},
            {"text": "我们一起去", "translation": "Let's go together"},
        ])
        proc = _make_processor([llm_response])

        segments = _make_segments([
            ("今天", 0.0, 0.5), ("天气", 0.5, 1.0), ("真好", 1.0, 1.5),
            ("我们", 2.0, 2.5), ("一起", 2.5, 3.0), ("去", 3.0, 3.2),
        ])

        result = proc.process_segments(
            segments, mode="translation", target_lang="English"
        )
        assert len(result) == 2
        assert result[0].get("translation") == "Great weather today"
        assert result[1].get("translation") == "Let's go together"

    def test_both_mode(self):
        """断句+翻译模式端到端"""
        llm_response = json.dumps([
            {"text": "今日は天気がいいですね", "translation": "今天天气真好呢"},
            {"text": "一緒に公園を散歩しましょう", "translation": "一起去公园散步吧"},
        ])
        proc = _make_processor([llm_response])

        segments = _make_segments([
            ("今日は", 0.0, 0.5), ("天気が", 0.5, 1.0), ("いいですね", 1.0, 1.5),
            ("一緒に", 2.0, 2.5), ("公園を", 2.5, 3.0),
            ("散歩しましょう", 3.0, 3.5),
        ])

        result = proc.process_segments(
            segments, mode="both", target_lang="简体中文"
        )
        assert len(result) == 2
        assert "translation" in result[0]

    def test_fallback_on_error(self):
        """API 返回错误时回退到原始文本"""
        proc = _make_processor(["INVALID RESPONSE"])

        segments = _make_segments([
            ("hello", 0.0, 0.5), ("world", 0.5, 1.0),
        ])

        result = proc.process_segments(segments, mode="segmentation")
        # 应该有一个回退 segment
        assert len(result) >= 1
        # 回退 segment 包含所有文本
        full_text = " ".join(s.get("text", "") for s in result)
        assert "hello" in full_text
        assert "world" in full_text

    def test_empty_segments(self):
        """空输入直接返回"""
        proc = _make_processor([])
        result = proc.process_segments([], mode="segmentation")
        assert result == []

    def test_speaker_preserved(self):
        """说话人标签在映射后保留"""
        llm_response = json.dumps(["hello world"])
        proc = _make_processor([llm_response])

        segments = [{
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "score": 0.9},
                {"word": "world", "start": 0.5, "end": 1.0, "score": 0.9},
            ],
            "text": "hello world",
            "speaker": "SPEAKER_01",
        }]

        result = proc.process_segments(segments, mode="segmentation")
        assert result[0].get("speaker") == "SPEAKER_01"


# ── Normalize 测试 ──────────────────────────────────────────────

class TestNormalize:
    def setup_method(self):
        self.proc = _make_processor([])

    def test_normalize_segmentation_strings(self):
        parsed = ["line 1", "line 2"]
        result = self.proc._normalize_segmentation(parsed)
        assert result == ["line 1", "line 2"]

    def test_normalize_segmentation_dicts(self):
        """某些 LLM 可能返回 dict 而不是纯字符串"""
        parsed = [{"text": "line 1"}, {"text": "line 2"}]
        result = self.proc._normalize_segmentation(parsed)
        assert result == ["line 1", "line 2"]

    def test_normalize_segmentation_empty_filtered(self):
        parsed = ["hello", "", "  ", "world"]
        result = self.proc._normalize_segmentation(parsed)
        assert result == ["hello", "world"]

    def test_normalize_translation(self):
        parsed = [
            {"text": "hello", "translation": "你好"},
            {"text": "world", "translation": "世界"},
        ]
        result = self.proc._normalize_translation(parsed)
        assert len(result) == 2
        assert result[0]["translation"] == "你好"

    def test_normalize_translation_string_fallback(self):
        """纯字符串元素应该有空翻译"""
        parsed = ["hello world"]
        result = self.proc._normalize_translation(parsed)
        assert result[0]["text"] == "hello world"
        assert result[0]["translation"] == ""


# ── 多批次处理测试 ──────────────────────────────────────────────

class TestMultiBatch:
    def test_two_batches(self):
        """当上下文很小时，应分为多批并最终合并结果"""
        responses = [
            json.dumps(["第一批内容"]),
            json.dumps(["第二批内容"]),
        ]
        proc = _make_processor(responses, max_tokens=200)
        
        # 生成足够多的词以触发两个批次
        word_data = [(f"词{i}", float(i), float(i + 1)) for i in range(50)]
        segments = _make_segments(word_data)

        result = proc.process_segments(segments, mode="segmentation")
        # 应该有来自两个批次的结果
        assert len(result) >= 2


# ── LLM 幻觉过滤测试 ───────────────────────────────────────────

class TestFilterHallucinations:
    def test_keeps_valid_segments(self):
        """LLM 返回所有 id → 全部保留"""
        response = json.dumps([0, 1, 2])
        proc = _make_processor([response])

        segments = [
            {"text": "今天天气真好", "words": [], "start": 0, "end": 1},
            {"text": "我们一起去", "words": [], "start": 1, "end": 2},
            {"text": "公园散步吧", "words": [], "start": 2, "end": 3},
        ]
        result = proc.filter_hallucinations(segments)
        assert len(result) == 3

    def test_removes_hallucination(self):
        """LLM 排除了 id=1 → 只保留 0 和 2"""
        response = json.dumps([0, 2])
        proc = _make_processor([response])

        segments = [
            {"text": "今天天气真好", "words": [], "start": 0, "end": 1},
            {"text": "请订阅我的频道", "words": [], "start": 1, "end": 2},
            {"text": "公园散步吧", "words": [], "start": 2, "end": 3},
        ]
        result = proc.filter_hallucinations(segments)
        assert len(result) == 2
        assert result[0]["text"] == "今天天气真好"
        assert result[1]["text"] == "公园散步吧"

    def test_fallback_on_api_error(self):
        """API 返回无效内容时保留全部"""
        proc = _make_processor(["INVALID NOT JSON!!!"])

        segments = [
            {"text": "hello", "words": [], "start": 0, "end": 1},
            {"text": "world", "words": [], "start": 1, "end": 2},
        ]
        result = proc.filter_hallucinations(segments)
        # 应该保留全部（回退策略）
        assert len(result) == 2

    def test_empty_segments(self):
        """空输入直接返回"""
        proc = _make_processor([])
        assert proc.filter_hallucinations([]) == []

    def test_text_from_words_when_no_text(self):
        """当 segment 没有 text 字段时从 words 拼接"""
        response = json.dumps([0])
        proc = _make_processor([response])

        segments = [
            {"words": [{"word": "hello"}, {"word": "world"}], "start": 0, "end": 1},
        ]
        result = proc.filter_hallucinations(segments)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
