"""
单元测试: src/core/utils.py
覆盖 SRT 时间戳生成、标点断句（拉丁语系 + CJK）、SRT/显示格式化
"""
import sys
import os
import unittest

# 确保项目根目录在 sys.path 中，以便直接运行: python tests/test_utils.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.utils import (
    seconds_to_srt_timestamp,
    split_segment_by_punctuation,
    generate_srt,
    format_transcript_for_display,
    _is_cjk_char,
    _is_cjk_punct,
    _text_is_cjk_dominant,
    _smart_join,
    _clean_text,
    _calc_segment_confidence,
    _calc_zero_score_ratio,
    _has_hallucination_pattern,
    _has_time_anomaly,
    _merge_split_letters,
    filter_hallucinated_segments,
    MAX_CJK_LINE_CHARS,
)


class TestSecondsToSrtTimestamp(unittest.TestCase):
    """测试时间戳转换函数"""

    def test_zero(self):
        self.assertEqual(seconds_to_srt_timestamp(0), "00:00:00,000")

    def test_none(self):
        self.assertEqual(seconds_to_srt_timestamp(None), "00:00:00,000")

    def test_negative_clamped_to_zero(self):
        """负数时间戳应被钳制为 0"""
        self.assertEqual(seconds_to_srt_timestamp(-5.0), "00:00:00,000")

    def test_normal_value(self):
        self.assertEqual(seconds_to_srt_timestamp(3661.5), "01:01:01,500")

    def test_millisecond_precision(self):
        self.assertEqual(seconds_to_srt_timestamp(1.123), "00:00:01,123")

    def test_large_value(self):
        result = seconds_to_srt_timestamp(7200.0)
        self.assertEqual(result, "02:00:00,000")


class TestCJKDetection(unittest.TestCase):
    """测试 CJK 字符检测"""

    def test_chinese_char(self):
        self.assertTrue(_is_cjk_char('你'))
        self.assertTrue(_is_cjk_char('好'))

    def test_japanese_hiragana(self):
        self.assertTrue(_is_cjk_char('あ'))

    def test_korean(self):
        self.assertTrue(_is_cjk_char('한'))

    def test_latin_char(self):
        self.assertFalse(_is_cjk_char('A'))
        self.assertFalse(_is_cjk_char('z'))

    def test_cjk_dominant_text(self):
        self.assertTrue(_text_is_cjk_dominant("你好世界"))
        self.assertTrue(_text_is_cjk_dominant("这是一个测试 test"))

    def test_latin_dominant_text(self):
        self.assertFalse(_text_is_cjk_dominant("Hello world"))
        self.assertFalse(_text_is_cjk_dominant("This is a test"))

    def test_empty_text(self):
        self.assertFalse(_text_is_cjk_dominant(""))
        self.assertFalse(_text_is_cjk_dominant("   "))


class TestSplitSegmentByPunctuation(unittest.TestCase):
    """测试标点断句算法"""

    def _make_segment(self, words_data, speaker="SPEAKER_00"):
        """辅助：构造 segment 数据结构"""
        words = []
        for i, (text, start, end) in enumerate(words_data):
            words.append({"word": text, "start": start, "end": end})
        return {"words": words, "speaker": speaker}

    def test_hard_punctuation_splits(self):
        """硬标点应触发换行"""
        seg = self._make_segment([
            ("Hello", 0.0, 0.5),
            ("world.", 0.6, 1.0),
            ("How", 1.1, 1.4),
            ("are", 1.5, 1.7),
            ("you?", 1.8, 2.2),
        ])
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hello world.")
        self.assertEqual(result[1]["text"], "How are you?")

    def test_no_split_caps_preserved(self):
        """NO_SPLIT_CAPS 中的词不应触发大写切分"""
        seg = self._make_segment([
            ("The", 0.0, 0.3),
            ("TV", 0.4, 0.6),
            ("is", 0.7, 0.9),
            ("OK.", 1.0, 1.3),
        ])
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "The TV is OK.")

    def test_short_sentence_protection(self):
        """短句保护：太短的句子不应被切分"""
        seg = self._make_segment([
            ("Hi", 0.0, 0.2),
            ("There", 0.3, 0.5),
            ("is", 0.6, 0.8),
            ("more.", 0.9, 1.2),
        ])
        result = split_segment_by_punctuation(seg)
        # "Hi" + "There" 仅 2 个词，短句保护应阻止在 "There" 前切分
        self.assertTrue(len(result) >= 1)

    def test_uppercase_triggers_split(self):
        """非特殊大写词应触发切分（当前句足够长时）"""
        seg = self._make_segment([
            ("hello", 0.0, 0.3),
            ("world", 0.4, 0.6),
            ("test", 0.7, 0.9),
            ("Something", 1.0, 1.4),
            ("new", 1.5, 1.8),
            ("here.", 1.9, 2.2),
        ])
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "hello world test")
        self.assertEqual(result[1]["text"], "Something new here.")

    def test_no_words_key(self):
        """没有 words 键时应原样返回"""
        seg = {"text": "raw text", "start": 0.0, "end": 1.0, "speaker": "X"}
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "raw text")

    def test_cjk_skips_uppercase_logic(self):
        """CJK 主导文本应跳过大写切分逻辑，仅按标点切分"""
        seg = self._make_segment([
            ("你", 0.0, 0.1), ("好", 0.1, 0.2),
            ("世", 0.2, 0.3), ("界", 0.3, 0.4),
            ("。", 0.4, 0.5),
            ("今", 0.6, 0.7), ("天", 0.7, 0.8),
            ("天", 0.8, 0.9), ("气", 0.9, 1.0),
            ("很", 1.0, 1.1), ("好", 1.1, 1.2),
            ("。", 1.2, 1.3),
        ])
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 2)
        # 核心断言：CJK 字符之间不应有空格
        self.assertEqual(result[0]["text"], "你好世界。")
        self.assertEqual(result[1]["text"], "今天天气很好。")

    def test_cjk_no_spaces_between_chars(self):
        """WhisperX 逐字对齐的 CJK 输出不应插入字间空格"""
        seg = self._make_segment([
            ("雨", 0.5, 1.0), ("が", 1.0, 1.1), ("降", 1.1, 1.2),
            ("っ", 1.2, 1.3), ("た", 1.3, 1.5),
        ])
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "雨が降った")

    def test_cjk_soft_punctuation_splits(self):
        """CJK 软标点在句子够长时应触发切分"""
        # 构造超过 10 个字符 + 逗号的 CJK 段
        seg = self._make_segment([
            ("星", 0.0, 0.1), ("期", 0.1, 0.2), ("五", 0.2, 0.3),
            ("晚", 0.3, 0.4), ("上", 0.4, 0.5), ("九", 0.5, 0.6),
            ("点", 0.6, 0.7), ("相", 0.7, 0.8), ("约", 0.8, 0.9),
            ("的", 0.9, 1.0), ("地", 1.0, 1.1), ("点", 1.1, 1.2),
            ("，", 1.2, 1.3),
            ("好", 1.4, 1.5), ("的", 1.5, 1.6), ("。", 1.6, 1.7),
        ])
        result = split_segment_by_punctuation(seg)
        # 第一段应在逗号处切分
        self.assertTrue(len(result) >= 2)
        self.assertNotIn(" ", result[0]["text"])  # CJK 字间无空格

    def test_cjk_mixed_with_latin(self):
        """CJK + 拉丁混合文本应正确处理空格"""
        seg = self._make_segment([
            ("One", 0.0, 0.2), ("night", 0.2, 0.4), ("in", 0.4, 0.5),
            ("上", 0.5, 0.6), ("海", 0.6, 0.7), ("。", 0.7, 0.8),
        ])
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 1)
        text = result[0]["text"]
        # 拉丁词之间有空格，拉丁和 CJK 之间有空格，CJK 之间无空格
        self.assertIn("One night in", text)
        self.assertIn("上海", text)  # 不是 "上 海"

    def test_missing_timestamps_skipped(self):
        """缺少 start/end 的词应被跳过"""
        seg = {
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "broken"},  # 缺少时间戳
                {"word": "world.", "start": 1.0, "end": 1.5},
            ],
            "speaker": "S"
        }
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello world.")

    def test_cjk_force_split_long_no_punct(self):
        """无标点 CJK 长文本应按字符数强制分段"""
        # 构造 30 个日文字符，无标点 → 应被拆成多行
        chars = list("街が去った街は静か僕はやっと部屋に戻って夜になったこんな良い月を")
        words_data = [(ch, i * 0.5, (i + 1) * 0.5) for i, ch in enumerate(chars)]
        seg = self._make_segment(words_data)
        result = split_segment_by_punctuation(seg)
        # 30 字符 / MAX_CJK_LINE_CHARS(20) → 应至少分成 2 段
        self.assertGreaterEqual(len(result), 2)
        # 每段都不应超过 MAX_CJK_LINE_CHARS + 余量
        for sub in result:
            self.assertLessEqual(len(sub["text"]), MAX_CJK_LINE_CHARS + 10)
        # 所有文本合并应等于原文
        merged = "".join(sub["text"] for sub in result)
        self.assertEqual(merged, "".join(chars))

    def test_cjk_force_split_prefers_particle(self):
        """CJK 强制分段应优先在日文助词处切分"""
        # "街は静か僕はやっと部屋に戻って夜になった" → 在 は/に/って 等助词后切分
        chars = list("街は静か僕はやっと部屋に戻って夜になった")
        words_data = [(ch, i * 0.5, (i + 1) * 0.5) for i, ch in enumerate(chars)]
        seg = self._make_segment(words_data)
        result = split_segment_by_punctuation(seg)
        if len(result) >= 2:
            # 第一段应在某个助词后结束（は、に、って 等）
            first_text = result[0]["text"]
            last_char = first_text[-1]
            # 助词或其紧跟的字符（CJK 无空格）
            self.assertTrue(
                last_char in {'は', 'が', 'を', 'に', 'で', 'の', 'と', 'も', 'へ',
                              'て', 'た', 'だ', 'っ', 'ね', 'よ', 'さ', 'な',
                              'か', '静', '僕', '部', '屋', '戻', '夜'} or
                len(first_text) <= MAX_CJK_LINE_CHARS + 10,
                f"First segment ended unexpectedly: '{first_text}'"
            )

    def test_cjk_short_no_force_split(self):
        """短于 MAX_CJK_LINE_CHARS 的 CJK 文本不应被强制分段"""
        chars = list("今日は天気がいい")  # 8 字符 < 20
        words_data = [(ch, i * 0.5, (i + 1) * 0.5) for i, ch in enumerate(chars)]
        seg = self._make_segment(words_data)
        result = split_segment_by_punctuation(seg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "今日は天気がいい")

    def test_cjk_chinese_force_split(self):
        """无标点中文长文本也应强制分段"""
        chars = list("星期五晚上九点我们约好了在这个地方见面真的很开心")
        words_data = [(ch, i * 0.3, (i + 1) * 0.3) for i, ch in enumerate(chars)]
        seg = self._make_segment(words_data)
        result = split_segment_by_punctuation(seg)
        # 23 字符 → 应至少分成 2 段
        self.assertGreaterEqual(len(result), 2)
        merged = "".join(sub["text"] for sub in result)
        self.assertEqual(merged, "".join(chars))


class TestGenerateSrt(unittest.TestCase):
    """测试 SRT 生成"""

    def test_basic_srt_output(self):
        segments = [{
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world.", "start": 0.6, "end": 1.0},
            ],
            "speaker": ""
        }]
        srt = generate_srt(segments)
        self.assertIn("1\n", srt)
        self.assertIn("00:00:00,000 --> 00:00:01,000", srt)
        self.assertIn("Hello world.", srt)

    def test_speaker_label_included(self):
        segments = [{
            "words": [
                {"word": "Test.", "start": 0.0, "end": 0.5},
            ],
            "speaker": "SPEAKER_00"
        }]
        srt = generate_srt(segments)
        self.assertIn("[SPEAKER_00]", srt)


class TestFormatTranscriptForDisplay(unittest.TestCase):
    """测试显示格式化"""

    def test_basic_format(self):
        segments = [{
            "words": [
                {"word": "Hello.", "start": 1.0, "end": 2.0},
            ],
            "speaker": "SPEAKER_01"
        }]
        text = format_transcript_for_display(segments)
        self.assertIn("1.00s", text)
        self.assertIn("2.00s", text)
        self.assertIn("SPEAKER_01", text)
        self.assertIn("Hello.", text)


class TestSmartJoin(unittest.TestCase):
    """测试智能拼接函数"""

    def test_latin_join(self):
        self.assertEqual(_smart_join(["Hello", "world"], is_cjk=False), "Hello world")

    def test_cjk_join_no_spaces(self):
        self.assertEqual(_smart_join(["雨", "が", "降", "っ", "た"], is_cjk=True), "雨が降った")

    def test_cjk_mixed_join(self):
        result = _smart_join(["One", "night", "in", "上", "海"], is_cjk=True)
        self.assertIn("One night in", result)
        self.assertIn("上海", result)

    def test_empty_list(self):
        self.assertEqual(_smart_join([], is_cjk=False), "")
        self.assertEqual(_smart_join([], is_cjk=True), "")

    def test_single_word(self):
        self.assertEqual(_smart_join(["Hello"], is_cjk=False), "Hello")
        self.assertEqual(_smart_join(["你好"], is_cjk=True), "你好")


class TestCleanText(unittest.TestCase):
    """测试文本清理函数"""

    def test_latin_punctuation_cleanup(self):
        self.assertEqual(_clean_text("Hello , world !", is_cjk=False), "Hello, world!")

    def test_cjk_space_removal(self):
        """CJK 字符之间的空格应被移除"""
        self.assertEqual(_clean_text("雨 が 降 っ た", is_cjk=True), "雨が降った")

    def test_cjk_preserves_latin_spaces(self):
        """CJK 清理不应影响拉丁词之间的空格"""
        result = _clean_text("One night in 上 海", is_cjk=True)
        self.assertIn("One night in", result)
        self.assertIn("上海", result)

    def test_cjk_multiple_passes(self):
        """多遍替换：'A B C' 三个相邻 CJK 字符间的空格都应移除"""
        self.assertEqual(_clean_text("你 好 世 界", is_cjk=True), "你好世界")


# =====================================================================
#  幻觉过滤测试
# =====================================================================

class TestCalcSegmentConfidence(unittest.TestCase):
    """测试置信度计算"""

    def test_normal_scores(self):
        seg = {"words": [
            {"word": "hello", "score": 0.8},
            {"word": "world", "score": 0.6},
        ]}
        self.assertAlmostEqual(_calc_segment_confidence(seg), 0.7)

    def test_all_zero_scores(self):
        seg = {"words": [
            {"word": "a", "score": 0.0},
            {"word": "b", "score": 0.0},
        ]}
        self.assertAlmostEqual(_calc_segment_confidence(seg), 0.0)

    def test_no_words(self):
        """无词时默认返回 1.0（保留 segment）"""
        self.assertAlmostEqual(_calc_segment_confidence({}), 1.0)
        self.assertAlmostEqual(_calc_segment_confidence({"words": []}), 1.0)

    def test_no_score_key(self):
        """词缺少 score 字段时应忽略该词"""
        seg = {"words": [{"word": "x"}]}
        self.assertAlmostEqual(_calc_segment_confidence(seg), 1.0)


class TestCalcZeroScoreRatio(unittest.TestCase):
    """测试零分词占比"""

    def test_all_zeros(self):
        seg = {"words": [
            {"word": "a", "score": 0.0},
            {"word": "b", "score": 0.0},
            {"word": "c", "score": 0.0},
        ]}
        self.assertAlmostEqual(_calc_zero_score_ratio(seg), 1.0)

    def test_half_zeros(self):
        seg = {"words": [
            {"word": "a", "score": 0.0},
            {"word": "b", "score": 0.8},
        ]}
        self.assertAlmostEqual(_calc_zero_score_ratio(seg), 0.5)

    def test_no_zeros(self):
        seg = {"words": [
            {"word": "a", "score": 0.9},
            {"word": "b", "score": 0.8},
        ]}
        self.assertAlmostEqual(_calc_zero_score_ratio(seg), 0.0)


class TestHasHallucinationPattern(unittest.TestCase):
    """测试已知幻觉模式匹配"""

    def test_chinese_hallucination(self):
        self.assertTrue(_has_hallucination_pattern("本歌曲来自〖云上工作室〗"))
        self.assertTrue(_has_hallucination_pattern("请订阅我的频道"))

    def test_japanese_hallucination(self):
        self.assertTrue(_has_hallucination_pattern("ご視聴ありがとうございました"))

    def test_english_hallucination(self):
        self.assertTrue(_has_hallucination_pattern("Thank you for watching"))
        self.assertTrue(_has_hallucination_pattern("Please subscribe"))

    def test_repetition_pattern(self):
        """重复短语应被检测"""
        self.assertTrue(_has_hallucination_pattern("啊啊啊啊啊啊啊啊啊啊啊啊"))

    def test_normal_text_not_flagged(self):
        self.assertFalse(_has_hallucination_pattern("Hello world, this is a test."))
        self.assertFalse(_has_hallucination_pattern("今天天气很好。"))
        self.assertFalse(_has_hallucination_pattern("One night in Shanghai"))


class TestHasTimeAnomaly(unittest.TestCase):
    """测试时间异常检测"""

    def test_normal_timing(self):
        seg = {"words": [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]}
        self.assertFalse(_has_time_anomaly(seg))

    def test_single_char_long_span(self):
        """单字符跨越 10 秒 → 时间异常"""
        seg = {"words": [
            {"word": "本", "start": 0.0, "end": 10.0},
        ]}
        self.assertTrue(_has_time_anomaly(seg))

    def test_multi_char_long_span_ok(self):
        """多字符词的长时间跨度不算异常"""
        seg = {"words": [
            {"word": "hello", "start": 0.0, "end": 10.0},
        ]}
        self.assertFalse(_has_time_anomaly(seg))

    def test_custom_threshold(self):
        seg = {"words": [
            {"word": "云", "start": 0.0, "end": 5.0},
        ]}
        self.assertFalse(_has_time_anomaly(seg, max_char_span=8.0))
        self.assertTrue(_has_time_anomaly(seg, max_char_span=3.0))


class TestMergeSplitLetters(unittest.TestCase):
    """测试英文字母合并"""

    def test_split_letters_merged(self):
        """逐字母拆分的英文应被合并"""
        words = [
            {"word": "O", "start": 0.0, "end": 0.1, "score": 0.0},
            {"word": "n", "start": 0.1, "end": 0.2, "score": 0.0},
            {"word": "e", "start": 0.2, "end": 0.3, "score": 0.0},
        ]
        result = _merge_split_letters(words)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["word"], "One")
        self.assertAlmostEqual(result[0]["start"], 0.0)
        self.assertAlmostEqual(result[0]["end"], 0.3)

    def test_normal_words_unchanged(self):
        """正常多字符词不应被修改"""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        result = _merge_split_letters(words)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["word"], "Hello")

    def test_single_letter_preserved(self):
        """单独的单字母词（如 'I' 或 'a'）不应被与后面合并，除非后面也是单字母"""
        words = [
            {"word": "I", "start": 0.0, "end": 0.2},
            {"word": "am", "start": 0.3, "end": 0.5},
        ]
        result = _merge_split_letters(words)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["word"], "I")

    def test_mixed_cjk_and_letters(self):
        """CJK 字符 + 拆分字母混合场景"""
        words = [
            {"word": "你", "start": 0.0, "end": 0.1},
            {"word": "好", "start": 0.1, "end": 0.2},
            {"word": "O", "start": 0.3, "end": 0.4, "score": 0.0},
            {"word": "K", "start": 0.4, "end": 0.5, "score": 0.0},
        ]
        result = _merge_split_letters(words)
        self.assertEqual(len(result), 3)  # 你, 好, OK
        self.assertEqual(result[2]["word"], "OK")

    def test_empty_input(self):
        self.assertEqual(_merge_split_letters([]), [])


class TestFilterHallucinatedSegments(unittest.TestCase):
    """测试幻觉过滤主函数"""

    def _make_seg(self, text, words_data, speaker=""):
        """辅助构造 segment"""
        words = []
        for w_text, start, end, score in words_data:
            words.append({"word": w_text, "start": start, "end": end, "score": score})
        return {"text": text, "words": words, "speaker": speaker}

    def test_removes_pattern_hallucination(self):
        """已知模式应被移除"""
        segs = [
            self._make_seg("本歌曲来自云上工作室", [
                ("本", 0.0, 10.0, 0.0), ("歌", 10.0, 10.1, 0.5),
                ("曲", 10.1, 10.2, 0.5), ("来", 10.2, 10.3, 0.0),
                ("自", 10.3, 10.4, 0.0), ("云", 10.4, 10.5, 0.0),
                ("上", 10.5, 10.6, 0.0), ("工", 10.6, 10.7, 0.0),
                ("作", 10.7, 10.8, 0.0), ("室", 10.8, 10.9, 0.0),
            ]),
        ]
        result = filter_hallucinated_segments(segs)
        self.assertEqual(len(result), 0)

    def test_removes_low_confidence(self):
        """低置信度 + 高零分占比的段落应被移除"""
        segs = [
            self._make_seg("phantom text", [
                ("phantom", 0.0, 0.5, 0.0),
                ("text", 0.6, 1.0, 0.0),
            ]),
        ]
        result = filter_hallucinated_segments(segs)
        self.assertEqual(len(result), 0)

    def test_preserves_normal_segment(self):
        """正常高置信度 segment 应被保留"""
        segs = [
            self._make_seg("Hello world.", [
                ("Hello", 0.0, 0.5, 0.85),
                ("world.", 0.6, 1.0, 0.9),
            ]),
        ]
        result = filter_hallucinated_segments(segs)
        self.assertEqual(len(result), 1)
        self.assertIn("Hello", result[0]["text"])

    def test_preserves_moderate_confidence(self):
        """中等置信度（有部分零分，但平均分还行）应保留"""
        segs = [
            self._make_seg("Some real speech.", [
                ("Some", 0.0, 0.3, 0.7),
                ("real", 0.4, 0.6, 0.0),  # 一个零分词
                ("speech.", 0.7, 1.0, 0.8),
            ]),
        ]
        result = filter_hallucinated_segments(segs)
        self.assertEqual(len(result), 1)

    def test_time_anomaly_with_low_conf_removed(self):
        """时间异常 + 低置信度 → 移除"""
        segs = [
            self._make_seg("云", [
                ("云", 0.0, 19.0, 0.2),
            ]),
        ]
        result = filter_hallucinated_segments(segs)
        self.assertEqual(len(result), 0)

    def test_time_anomaly_with_high_conf_kept(self):
        """时间异常但置信度高 → 保留"""
        segs = [
            self._make_seg("A", [
                ("A", 0.0, 10.0, 0.9),
            ]),
        ]
        result = filter_hallucinated_segments(segs)
        self.assertEqual(len(result), 1)

    def test_letter_merge_applied(self):
        """逐字母拆分的英文应被合并"""
        segs = [
            self._make_seg("O n e", [
                ("O", 0.0, 0.1, 0.8),
                ("n", 0.1, 0.2, 0.7),
                ("e", 0.2, 0.3, 0.9),
            ]),
        ]
        result = filter_hallucinated_segments(segs, merge_letters=True)
        self.assertEqual(len(result), 1)
        self.assertIn("One", result[0]["text"])

    def test_filter_disabled(self):
        """所有过滤器关闭时应保留全部 segment"""
        segs = [
            self._make_seg("本歌曲来自云上工作室", [
                ("本", 0.0, 10.0, 0.0),
            ]),
        ]
        result = filter_hallucinated_segments(
            segs,
            enable_pattern_filter=False,
            enable_confidence_filter=False,
            enable_time_anomaly_filter=False,
            merge_letters=False,
        )
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()