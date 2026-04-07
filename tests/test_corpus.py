"""
单元测试: 数据清洗模块 & JSONL 导出模块
覆盖七条清洗规则和高级特征计算。
"""
import sys
import os
import json
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_cleaner import DataCleaner, CleaningConfig, compute_segment_snr, rule_g_blacklist_filter
from src.core.corpus_exporter import (
    compute_advanced_features,
    export_jsonl,
    merge_jsonl_files,
)


def _make_seg(text, start, end, speaker="SPEAKER_00", words=None, score=0.9):
    """辅助: 构造 segment"""
    if words is None:
        words = [{"word": text, "start": start, "end": end, "score": score}]
    return {
        "text": text,
        "start": start,
        "end": end,
        "speaker": speaker,
        "words": words,
    }


class TestRuleAConfidence(unittest.TestCase):
    """规则 A: 置信度过滤"""

    def test_high_confidence_kept(self):
        segs = [_make_seg("你好世界", 0, 2, score=0.95)]
        cfg = CleaningConfig(
            enable_confidence_filter=True, confidence_threshold=0.8,
            enable_overlap_filter=False, enable_length_ratio_filter=False,
            enable_context_island_filter=False, enable_low_info_filter=False,
            enable_snr_filter=False,
        )
        cleaner = DataCleaner(cfg)
        result, stats = cleaner.clean(segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(stats['rule_a_removed'], 0)

    def test_low_confidence_removed(self):
        segs = [_make_seg("乱码", 0, 2, score=0.3)]
        cfg = CleaningConfig(
            enable_confidence_filter=True, confidence_threshold=0.8,
            enable_overlap_filter=False, enable_length_ratio_filter=False,
            enable_context_island_filter=False, enable_low_info_filter=False,
            enable_snr_filter=False,
        )
        cleaner = DataCleaner(cfg)
        result, stats = cleaner.clean(segs)
        self.assertEqual(len(result), 0)
        self.assertEqual(stats['rule_a_removed'], 1)


class TestRuleBOverlap(unittest.TestCase):
    """规则 B: 重叠音过滤"""

    def _only_rule_b(self, **kwargs):
        return CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=True,
            enable_length_ratio_filter=False, enable_context_island_filter=False,
            enable_low_info_filter=False, enable_snr_filter=False,
            **kwargs,
        )

    def test_significant_overlap_removes_shorter(self):
        """重叠 >= 0.5s 时，移除较短的片段"""
        segs = [
            _make_seg("你好世界我是主持人", 0, 5, speaker="A"),      # 5s，长
            _make_seg("嗯哈", 2, 4, speaker="B"),                    # 2s，短，与A重叠2s
        ]
        cfg = self._only_rule_b(overlap_min_duration=0.5)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 1)            # 保留A
        self.assertEqual(result[0]['speaker'], 'A')
        self.assertEqual(stats['rule_b_removed'], 1)

    def test_micro_overlap_ignored(self):
        """微小重叠 (< 0.5s) 视为对齐误差，不移除"""
        segs = [
            _make_seg("你好", 0, 3, speaker="A"),
            _make_seg("世界", 2.8, 5, speaker="B"),  # 仅重叠0.2s
        ]
        cfg = self._only_rule_b(overlap_min_duration=0.5)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 2)  # 两个都保留
        self.assertEqual(stats['rule_b_removed'], 0)

    def test_non_overlapping_kept(self):
        segs = [
            _make_seg("你好", 0, 2, speaker="A"),
            _make_seg("世界", 3, 5, speaker="B"),
        ]
        cfg = self._only_rule_b()
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 2)

    def test_same_speaker_overlap_ignored(self):
        """同一说话人的重叠不算重叠"""
        segs = [
            _make_seg("你好", 0, 3, speaker="A"),
            _make_seg("世界", 2, 5, speaker="A"),  # 同说话人重叠
        ]
        cfg = self._only_rule_b(overlap_min_duration=0.5)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 2)


class TestRuleELowInfo(unittest.TestCase):
    """规则 E: 低信息量剔除"""

    def test_short_filler_removed(self):
        segs = [_make_seg("嗯嗯", 0, 0.5)]
        cfg = CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=False,
            enable_length_ratio_filter=False, enable_context_island_filter=False,
            enable_low_info_filter=True, enable_snr_filter=False,
            low_info_max_duration=1.5, low_info_max_chars=4,
        )
        cleaner = DataCleaner(cfg)
        result, stats = cleaner.clean(segs)
        self.assertEqual(len(result), 0)
        self.assertEqual(stats['rule_e_removed'], 1)

    def test_long_filler_kept(self):
        """即使是填充词但持续时间长，也保留"""
        segs = [_make_seg("嗯嗯", 0, 5)]
        cfg = CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=False,
            enable_length_ratio_filter=False, enable_context_island_filter=False,
            enable_low_info_filter=True, enable_snr_filter=False,
        )
        cleaner = DataCleaner(cfg)
        result, stats = cleaner.clean(segs)
        self.assertEqual(len(result), 1)

    def test_meaningful_short_kept(self):
        """有意义的短句不被误杀"""
        segs = [_make_seg("好的明天见", 0, 1)]
        cfg = CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=False,
            enable_length_ratio_filter=False, enable_context_island_filter=False,
            enable_low_info_filter=True, enable_snr_filter=False,
        )
        cleaner = DataCleaner(cfg)
        result, stats = cleaner.clean(segs)
        self.assertEqual(len(result), 1)


class TestRuleCLengthRatio(unittest.TestCase):
    """规则 C: 长度匹配度"""

    def _only_rule_c(self, **kwargs):
        return CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=False,
            enable_length_ratio_filter=True, enable_context_island_filter=False,
            enable_low_info_filter=False, enable_snr_filter=False,
            **kwargs,
        )

    def test_very_short_segment_skipped(self):
        """极短片段 (< length_ratio_min_duration) 不做速率检查"""
        segs = [_make_seg("嗯", 0, 0.5)]  # 1 char / 0.5s = 2 chars/s — 但时长太短应跳过
        cfg = self._only_rule_c(length_ratio_min_duration=1.5)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 1)  # 保留

    def test_normal_speech_rate_kept(self):
        """正常中文语速 ~5 字/秒应保留"""
        segs = [_make_seg("我今天去超市买了很多东西", 0, 3)]  # 11 chars / 3s ≈ 3.67
        cfg = self._only_rule_c()
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 1)

    def test_extreme_rate_removed(self):
        """极端语速 (文本极多但时长极短) 应被移除"""
        long_text = "我" * 100
        segs = [_make_seg(long_text, 0, 2)]  # 100 chars / 2s = 50 chars/s
        cfg = self._only_rule_c(max_chars_per_sec_cjk=25.0)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 0)
        self.assertEqual(stats['rule_c_removed'], 1)


class TestRuleDContextIsland(unittest.TestCase):
    """规则 D: 上下文孤岛过滤（孤立片段检测）"""

    def _only_rule_d(self, **kwargs):
        return CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=False,
            enable_length_ratio_filter=False, enable_context_island_filter=True,
            enable_low_info_filter=False, enable_snr_filter=False,
            **kwargs,
        )

    def test_safety_threshold_skips(self):
        """段数 <= min_segments_to_apply_d 时跳过规则D"""
        segs = [
            _make_seg("你好", 0, 2, speaker="A"),
            _make_seg("世界", 3, 5, speaker="A"),
            _make_seg("再见", 6, 8, speaker="A"),
        ]
        cfg = self._only_rule_d(min_segments_to_apply_d=6)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 3)  # 全部保留
        self.assertEqual(stats['rule_d_removed'], 0)

    def test_single_speaker_skips(self):
        """全文仅一个说话人时跳过"""
        segs = [_make_seg(f"段落{i}", i * 3, i * 3 + 2, speaker="A") for i in range(10)]
        cfg = self._only_rule_d(min_segments_to_apply_d=2)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 10)

    def test_nearby_multi_speaker_kept(self):
        """多说话人且相邻的片段应保留"""
        segs = [
            _make_seg("你好", 0, 2, speaker="A"),
            _make_seg("你好啊", 3, 5, speaker="B"),
            _make_seg("再见", 6, 8, speaker="A"),
            _make_seg("再见啊", 9, 11, speaker="B"),
            _make_seg("是的", 12, 14, speaker="A"),
            _make_seg("对的", 15, 17, speaker="B"),
            _make_seg("好的", 18, 20, speaker="A"),
        ]
        cfg = self._only_rule_d(orphan_window_sec=120, min_segments_to_apply_d=2)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 7)  # 全部保留

    def test_truly_isolated_segment_removed(self):
        """远离所有其他说话人的片段应被移除"""
        segs = [
            _make_seg("对话开始", 0, 2, speaker="A"),
            _make_seg("你好", 3, 5, speaker="B"),
            _make_seg("再见", 6, 8, speaker="A"),
            _make_seg("是的", 9, 11, speaker="B"),
            _make_seg("对的", 12, 14, speaker="A"),
            _make_seg("好的", 15, 17, speaker="B"),
            _make_seg("完全孤立", 500, 502, speaker="C"),  # 距离最近的不同说话人 >120s
        ]
        cfg = self._only_rule_d(orphan_window_sec=120, min_segments_to_apply_d=2)
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 6)  # 孤立的C被移除
        self.assertEqual(stats['rule_d_removed'], 1)


class TestAdvancedFeatures(unittest.TestCase):
    """高级特征字段计算"""

    def test_turn_delay_calculation(self):
        segs = [
            _make_seg("你好吗", 0, 2, speaker="A"),
            _make_seg("我很好", 2.5, 4, speaker="B"),
        ]
        enriched = compute_advanced_features(segs)
        self.assertIsNone(enriched[0]['turn_delay'])
        self.assertAlmostEqual(enriched[1]['turn_delay'], 500.0, places=0)

    def test_turn_delay_context_break(self):
        """turn_delay 超过阈值应设为 null（上下文断裂）"""
        segs = [
            _make_seg("第一段", 0, 2, speaker="A"),
            _make_seg("第二段", 100, 102, speaker="B"),  # 间隔 98 秒
        ]
        enriched = compute_advanced_features(segs)
        self.assertIsNone(enriched[1]['turn_delay'])

    def test_duration_calculation(self):
        segs = [_make_seg("测试", 1.0, 3.5)]
        enriched = compute_advanced_features(segs)
        self.assertAlmostEqual(enriched[0]['duration'], 2.5)

    def test_words_per_second_cjk(self):
        """CJK 文本 words_per_second 按字符数计算"""
        segs = [_make_seg("我今天去超市买了很多东西", 0, 3)]  # 12 chars / 3s = 4.0
        enriched = compute_advanced_features(segs)
        wps = enriched[0]['words_per_second']
        self.assertIsNotNone(wps)
        self.assertAlmostEqual(wps, 12 / 3, places=1)

    def test_words_per_second_latin(self):
        """拉丁文本 words_per_second 按词数计算"""
        segs = [_make_seg("hello world how are you", 0, 2.5)]  # 5 words / 2.5s = 2.0
        enriched = compute_advanced_features(segs)
        wps = enriched[0]['words_per_second']
        self.assertIsNotNone(wps)
        self.assertAlmostEqual(wps, 2.0, places=1)

    def test_conversation_id_increments_on_gap(self):
        """段落间距超过阈值时 conversation_id 递增"""
        segs = [
            _make_seg("段落1", 0, 2, speaker="A"),
            _make_seg("段落2", 3, 5, speaker="B"),
            _make_seg("段落3", 50, 52, speaker="A"),  # 间距 45s > 10s
            _make_seg("段落4", 53, 55, speaker="B"),
        ]
        enriched = compute_advanced_features(segs)
        self.assertEqual(enriched[0]['conversation_id'], 0)
        self.assertEqual(enriched[1]['conversation_id'], 0)
        self.assertEqual(enriched[2]['conversation_id'], 1)
        self.assertEqual(enriched[3]['conversation_id'], 1)

    def test_contains_overlap_detection(self):
        """检测到前后句时间重叠时 contains_overlap=True"""
        segs = [
            _make_seg("段落1", 0, 5, speaker="A"),
            _make_seg("段落2", 3, 7, speaker="B"),  # start 3 < prev end 5 → 重叠
            _make_seg("段落3", 10, 12, speaker="A"),  # start 10 > prev end 7 → 无重叠
        ]
        enriched = compute_advanced_features(segs)
        self.assertFalse(enriched[0]['contains_overlap'])
        self.assertTrue(enriched[1]['contains_overlap'])
        self.assertFalse(enriched[2]['contains_overlap'])

    def test_interaction_type_long_narrative(self):
        """单条 segment 持续 > 15s → long_narrative"""
        segs = [_make_seg("非常长的叙述型片段", 0, 20, speaker="A")]
        enriched = compute_advanced_features(segs)
        self.assertEqual(enriched[0]['interaction_type'], 'long_narrative')

    def test_interaction_type_short_reply(self):
        """单条 segment 持续 < 5s 且不在快问快答窗口 → short_reply"""
        segs = [
            _make_seg("一段较长的话", 0, 10, speaker="A"),
            _make_seg("嗯", 11, 12, speaker="B"),  # 1s，短回复
        ]
        enriched = compute_advanced_features(segs)
        self.assertEqual(enriched[1]['interaction_type'], 'short_reply')

    def test_interaction_type_normal(self):
        """5s < duration < 15s → normal"""
        segs = [_make_seg("一段正常长度的话", 0, 8, speaker="A")]
        enriched = compute_advanced_features(segs)
        self.assertEqual(enriched[0]['interaction_type'], 'normal')

    def test_confidence_in_output(self):
        segs = [_make_seg("test", 0, 1, score=0.85)]
        enriched = compute_advanced_features(segs)
        self.assertAlmostEqual(enriched[0]['confidence_score'], 0.85, places=2)


class TestJSONLExport(unittest.TestCase):
    """JSONL 导出"""

    def test_export_and_read(self):
        segs = compute_advanced_features([
            _make_seg("你好", 0, 2, speaker="A"),
            _make_seg("你好啊", 2.5, 4, speaker="B"),
        ])
        
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode='w') as f:
            tmp_path = f.name

        try:
            export_jsonl(segs, tmp_path, audio_source="test.wav")
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 2)
            
            record = json.loads(lines[0])
            # 原有字段
            self.assertIn('audio_path', record)
            self.assertIn('text', record)
            self.assertIn('speaker_id', record)
            self.assertIn('confidence_score', record)
            self.assertIn('duration', record)
            self.assertIn('turn_delay', record)
            self.assertIn('interaction_type', record)
            self.assertEqual(record['audio_path'], 'test.wav')
            # 新增字段
            self.assertIn('words_per_second', record)
            self.assertIn('conversation_id', record)
            self.assertIn('contains_overlap', record)
            self.assertIsInstance(record['conversation_id'], int)
            self.assertIsInstance(record['contains_overlap'], bool)
        finally:
            os.unlink(tmp_path)

    def test_merge_jsonl(self):
        tmpdir = tempfile.mkdtemp()
        files = []
        for i in range(3):
            path = os.path.join(tmpdir, f"test_{i}.jsonl")
            with open(path, 'w') as f:
                f.write(json.dumps({"text": f"line_{i}"}) + '\n')
            files.append(path)

        merged = os.path.join(tmpdir, "merged.jsonl")
        merge_jsonl_files(files, merged)
        
        with open(merged, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

        # cleanup
        import shutil
        shutil.rmtree(tmpdir)


class TestSNRComputation(unittest.TestCase):
    """信噪比计算 & 规则 F"""

    def test_snr_annotates_segments(self):
        """compute_segment_snr 应为每个段写入 snr_db 字段"""
        sr = 16000
        # 生成 10 秒音频：前 5s 安静，后 5s 响亮
        quiet = np.random.randn(5 * sr).astype(np.float32) * 0.001
        loud = np.random.randn(5 * sr).astype(np.float32) * 0.5
        audio = np.concatenate([quiet, loud])

        segs = [
            {"text": "安静", "start": 1.0, "end": 4.0},  # 在安静区
            {"text": "响亮", "start": 6.0, "end": 9.0},  # 在响亮区
        ]
        compute_segment_snr(audio, segs, sample_rate=sr)
        self.assertIn('snr_db', segs[0])
        self.assertIn('snr_db', segs[1])
        # 响亮段的 SNR 应远高于安静段
        self.assertGreater(segs[1]['snr_db'], segs[0]['snr_db'])

    def test_rule_f_filters_low_snr(self):
        """规则 F 应移除 SNR 低于阈值的段"""
        segs = [
            _make_seg("好的", 0, 2),
            _make_seg("噪音", 3, 5),
        ]
        segs[0]['snr_db'] = 20.0   # 高 SNR
        segs[1]['snr_db'] = 5.0    # 低 SNR

        cfg = CleaningConfig(
            enable_confidence_filter=False, enable_overlap_filter=False,
            enable_length_ratio_filter=False, enable_context_island_filter=False,
            enable_low_info_filter=False,
            enable_snr_filter=True, min_snr_db=10.0,
        )
        result, stats = DataCleaner(cfg).clean(segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], '好的')
        self.assertEqual(stats['rule_f_removed'], 1)


class TestRuleGBlacklist(unittest.TestCase):
    """规则 G: 黑名单/脱轨内容过滤"""

    def _make_enriched(self, texts_and_conv_ids):
        """辅助: 构造带 conversation_id 的 segments"""
        segs = []
        for i, (text, conv_id) in enumerate(texts_and_conv_ids):
            segs.append({
                'text': text,
                'start': i * 10.0,
                'end': i * 10.0 + 5.0,
                'speaker': 'A' if i % 2 == 0 else 'B',
                'conversation_id': conv_id,
            })
        return segs

    def test_no_blacklist_returns_all(self):
        """空黑名单不过滤任何内容"""
        segs = self._make_enriched([
            ("你好世界", 0),
            ("精籹水推荐", 0),
        ])
        result, count = rule_g_blacklist_filter(segs, [])
        self.assertEqual(len(result), 2)
        self.assertEqual(count, 0)

    def test_direct_hit_removed(self):
        """命中黑名单的句子被移除"""
        segs = self._make_enriched([
            ("今天天气真好", 0),
            ("下单立减20元", 1),
            ("我们继续聊天", 2),
        ])
        result, count = rule_g_blacklist_filter(segs, ["下单"], context_purge=False)
        self.assertEqual(len(result), 2)
        self.assertEqual(count, 1)
        self.assertEqual(result[0]['text'], "今天天气真好")
        self.assertEqual(result[1]['text'], "我们继续聊天")

    def test_context_purge_same_conversation(self):
        """开启上下文清除时，同 conversation_id 的相邻句子一并丢弃"""
        segs = self._make_enriched([
            ("正常对话1", 0),
            ("正常对话2", 0),
            # conversation_id=1 是广告区块
            ("下面插播一条广告", 1),
            ("精籹水限时优惠", 1),   # 命中 "精籹水" + "限时优惠"
            ("赶紧下单哦", 1),           # 命中 "下单"
            # 回到正常对话
            ("好了我们继续", 2),
            ("聊到哪儿了", 2),
        ])
        result, count = rule_g_blacklist_filter(
            segs, ["精籹水", "下单", "限时优惠"], context_purge=True
        )
        # conversation_id=1 的 3 条全部被清除（包括未直接命中的 "下面插播一条广告"）
        self.assertEqual(len(result), 4)
        self.assertEqual(count, 3)
        texts = [s['text'] for s in result]
        self.assertIn("正常对话1", texts)
        self.assertIn("好了我们继续", texts)
        self.assertNotIn("精籹水限时优惠", texts)

    def test_context_purge_does_not_cross_conversation(self):
        """上下文清除不会跨越不同的 conversation_id"""
        segs = self._make_enriched([
            ("正常对话", 0),
            ("请点击链接购买", 1),  # 命中 "点击链接"
            ("回到正常话题", 2),
        ])
        result, count = rule_g_blacklist_filter(
            segs, ["点击链接"], context_purge=True
        )
        # 只移除 conversation_id=1 的 1 条，conv 0 和 2 不受影响
        self.assertEqual(len(result), 2)
        self.assertEqual(count, 1)

    def test_no_hit_returns_all(self):
        """无命中时全部保留"""
        segs = self._make_enriched([
            ("今天天气真好", 0),
            ("我们去吃饭吧", 0),
        ])
        result, count = rule_g_blacklist_filter(segs, ["下单", "优惠券"])
        self.assertEqual(len(result), 2)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
