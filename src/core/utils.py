import math
import re
import logging
import unicodedata

logger = logging.getLogger(__name__)

# 定义强制换行的标点
SPLIT_PUNCTUATION = {'.', '?', '!', '。', '？', '！'}
# 定义软换行标点
SOFT_PUNCTUATION = {',', '，', ';', '；', ':', '：'}

# 【新增】不触发换行的特殊大写单词集合 (缩写、专有名词)
# 你可以在这里随意添加单词，例如 "TV", "APP", "API", "OK" 等
NO_SPLIT_CAPS = {
    "I", "I'm", "I'll", "I've", "I'd",  # 第一人称及其缩写
    "TV", "OK", "ID", "PC", "APP", "API", 
    "USA", "UK", "CEO", "DNA", "FBI", "CIA", "DIY", "SOS", "VIP"
}

# 最小单词/字符数阈值 (防止过短的行)
MIN_WORDS = 2       # 拉丁语系最少单词数
MIN_CJK_CHARS = 6   # CJK 最少字符数

# CJK 无标点时的最大行字符数（超过则强制分段）
# 用于处理日文歌词等缺少标点的长文本块
MAX_CJK_LINE_CHARS = 20

# 日文助词集合 —— 作为强制分段时的首选断句位置
# 在这些助词之后切分最为自然（は/が/を 等标记语法边界）
_JP_PARTICLES = {'は', 'が', 'を', 'に', 'で', 'の', 'と', 'も', 'へ', 'から', 'まで', 'より', 'って', 'ね', 'よ', 'さ', 'な', 'て', 'た', 'だ'}

# CJK Unicode 范围检测
def _is_cjk_char(ch: str) -> bool:
    """判断字符是否为中日韩表意文字（含假名、韩文音节）"""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or      # CJK Unified Ideographs
        (0x3400 <= cp <= 0x4DBF) or      # CJK Extension A
        (0x20000 <= cp <= 0x2A6DF) or    # CJK Extension B
        (0xF900 <= cp <= 0xFAFF) or      # CJK Compatibility Ideographs
        (0x3000 <= cp <= 0x303F) or      # CJK Symbols and Punctuation
        (0x3040 <= cp <= 0x309F) or      # Hiragana
        (0x30A0 <= cp <= 0x30FF) or      # Katakana
        (0xAC00 <= cp <= 0xD7AF)         # Hangul Syllables
    )

def _is_cjk_punct(ch: str) -> bool:
    """判断字符是否为 CJK 全角标点"""
    return ch in {'。', '？', '！', '，', '；', '：', '、', '…', '—',
                  '「', '」', '『', '』', '（', '）', '【', '】', '《', '》'}

def _text_is_cjk_dominant(text: str) -> bool:
    """判断文本是否以 CJK 字符为主（超过 30% 即判定为 CJK 文本）"""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if _is_cjk_char(ch))
    non_space = sum(1 for ch in text if not ch.isspace())
    return non_space > 0 and (cjk_count / non_space) > 0.3

# =====================================================================
#  幻觉过滤 (Hallucination Filtering)
# =====================================================================

# 常见 Whisper 幻觉模式（正则）
_HALLUCINATION_PATTERNS = [
    # 中文常见幻觉
    r'本歌曲来自',
    r'云上工作室',
    r'请订阅',
    r'感谢收看',
    r'感谢观看',
    r'谢谢大家',
    r'字幕制作',
    r'字幕.*提供',
    r'字幕by',
    r'请关注',
    r'欢迎订阅',
    r'版权所有',
    r'翻译.*校对',
    # 日文常见幻觉
    r'ご視聴ありがとうございました',
    r'チャンネル登録',
    r'お疲れ様でした',
    r'字幕は.*です',
    # 英文常见幻觉
    r'(?i)thank(?:s| you) for watching',
    r'(?i)please subscribe',
    r'(?i)like and subscribe',
    r'(?i)subtitles by',
    r'(?i)captions by',
    r'(?i)transcribed by',
    # Whisper 特有重复幻觉
    r'(.{2,}?)\1{3,}',  # 同一短语重复 4 次以上
]

_HALLUCINATION_RE = [re.compile(p) for p in _HALLUCINATION_PATTERNS]


def _calc_segment_confidence(segment: dict) -> float:
    """计算 segment 所有词的平均置信度分数"""
    words = segment.get('words', [])
    scores = [w.get('score', 0.0) for w in words if 'score' in w]
    if not scores:
        return 1.0  # 无分数信息时保留
    return sum(scores) / len(scores)


def _calc_zero_score_ratio(segment: dict) -> float:
    """计算 segment 中零分词的占比"""
    words = segment.get('words', [])
    scored = [w for w in words if 'score' in w]
    if not scored:
        return 0.0
    zeros = sum(1 for w in scored if w.get('score', 0.0) < 1e-6)
    return zeros / len(scored)


def _has_hallucination_pattern(text: str) -> bool:
    """检查文本是否匹配已知的 Whisper 幻觉模式"""
    for pattern in _HALLUCINATION_RE:
        if pattern.search(text):
            return True
    return False


def _has_time_anomaly(segment: dict, max_char_span: float = 8.0) -> bool:
    """
    检测时间异常：单个字符的词跨越了不合理的时间范围。
    例如 "本" 占据 10 秒、"云" 占据 19 秒 → 明显是幻觉。
    """
    words = segment.get('words', [])
    for w in words:
        if 'start' not in w or 'end' not in w:
            continue
        word_text = w.get('word', '').strip()
        duration = w['end'] - w['start']
        # 单个字符持续超过阈值
        if len(word_text) <= 1 and duration > max_char_span:
            return True
    return False


def _merge_split_letters(words: list) -> list:
    """
    合并被逐字母拆分的英文单词。
    WhisperX 在处理 CJK 音频中的英文时，会将 "One" 拆分为 "O", "n", "e" 三个独立 word。
    本函数检测连续的单字母 ASCII 词并将它们合并回完整单词。
    """
    if not words:
        return words
    
    merged = []
    i = 0
    while i < len(words):
        w = words[i]
        w_text = w.get('word', '').strip()
        
        # 检测单个 ASCII 字母（非 CJK）
        if len(w_text) == 1 and w_text.isascii() and w_text.isalpha():
            # 收集连续的单字母序列
            letter_run = [w]
            j = i + 1
            while j < len(words):
                next_text = words[j].get('word', '').strip()
                if len(next_text) == 1 and next_text.isascii() and next_text.isalpha():
                    letter_run.append(words[j])
                    j += 1
                else:
                    break
            
            # 只有连续 2 个以上单字母时才合并（单个字母可能是 "I" 或 "a"）
            if len(letter_run) >= 2:
                merged_word = "".join(lr.get('word', '').strip() for lr in letter_run)
                # 取首尾字母的时间区间，取所有分数的均值
                scores = [lr.get('score', 0.0) for lr in letter_run if 'score' in lr]
                merged_entry = {
                    'word': merged_word,
                    'start': letter_run[0].get('start', 0.0),
                    'end': letter_run[-1].get('end', 0.0),
                }
                if scores:
                    merged_entry['score'] = sum(scores) / len(scores)
                merged.append(merged_entry)
                i = j
                continue
        
        merged.append(w)
        i += 1
    
    return merged


def filter_hallucinated_segments(
    segments: list,
    confidence_threshold: float = 0.35,
    zero_score_ratio_threshold: float = 0.5,
    max_char_span: float = 8.0,
    enable_pattern_filter: bool = True,
    enable_confidence_filter: bool = True,
    enable_time_anomaly_filter: bool = True,
    merge_letters: bool = True,
) -> list:
    """
    对 WhisperX 对齐后的 segments 进行幻觉过滤和文本修复。
    
    过滤策略：
    1. 已知幻觉模式匹配（正则表达式黑名单）
    2. 置信度过滤（平均分 + 零分词占比双重判定）
    3. 时间异常检测（单字符跨越不合理时长）
    4. 逐字母英文合并修复
    
    Args:
        segments: WhisperX 返回的 segment 列表
        confidence_threshold: 平均置信度低于此值的 segment 将被移除（默认 0.35）
        zero_score_ratio_threshold: 零分词占比超过此值的 segment 将被移除（默认 0.5）
        max_char_span: 单字符允许的最大时间跨度（秒），超过则判定为异常
        enable_pattern_filter: 是否启用已知幻觉模式匹配
        enable_confidence_filter: 是否启用置信度过滤
        enable_time_anomaly_filter: 是否启用时间异常检测
        merge_letters: 是否合并被拆分的英文字母
    
    Returns:
        过滤和修复后的 segments 列表
    """
    filtered = []
    removed_count = 0
    
    for seg in segments:
        text = seg.get('text', '').strip()
        words = seg.get('words', [])
        
        # --- 逐字母英文合并 ---
        if merge_letters and words:
            seg = dict(seg)  # 浅拷贝以免修改原始数据
            seg['words'] = _merge_split_letters(words)
            # 重建 text（合并后的词重新拼接）
            is_cjk = _text_is_cjk_dominant(text)
            rebuilt_words = [w.get('word', '').strip() for w in seg['words'] if w.get('word', '').strip()]
            if rebuilt_words:
                seg['text'] = _clean_text(_smart_join(rebuilt_words, is_cjk), is_cjk)
                text = seg['text']
        
        # --- 1. 已知幻觉模式 ---
        if enable_pattern_filter and _has_hallucination_pattern(text):
            logger.info(f"[幻觉过滤·模式匹配] 移除: \"{text[:60]}...\"")
            removed_count += 1
            continue
        
        # --- 2. 置信度过滤（双指标判定：平均分 + 零分占比） ---
        if enable_confidence_filter and seg.get('words'):
            avg_conf = _calc_segment_confidence(seg)
            zero_ratio = _calc_zero_score_ratio(seg)
            
            # 必须同时满足：平均分低 AND 零分词比例高，才移除
            # 这样可以避免误杀正常的低分段（如背景噪音中的真实语音）
            if avg_conf < confidence_threshold and zero_ratio > zero_score_ratio_threshold:
                logger.info(
                    f"[幻觉过滤·低置信] 移除: \"{text[:60]}\" "
                    f"(avg={avg_conf:.3f}, zero_ratio={zero_ratio:.1%})"
                )
                removed_count += 1
                continue
        
        # --- 3. 时间异常检测 ---
        if enable_time_anomaly_filter and _has_time_anomaly(seg, max_char_span):
            avg_conf = _calc_segment_confidence(seg)
            # 时间异常 + 置信度不高 → 移除
            if avg_conf < 0.5:
                logger.info(
                    f"[幻觉过滤·时间异常] 移除: \"{text[:60]}\" "
                    f"(avg_conf={avg_conf:.3f})"
                )
                removed_count += 1
                continue
        
        filtered.append(seg)
    
    if removed_count > 0:
        logger.info(f"[幻觉过滤] 共移除 {removed_count} 个可疑段落，保留 {len(filtered)} 段")
    
    return filtered


def _smart_join(words_list: list, is_cjk: bool = False) -> str:
    """
    智能拼接词列表：
    - 始终智能处理 CJK 字符：相邻 CJK 字符之间不加空格
    - 拉丁词之间正常空格拼接
    - 混合文本中 CJK 与拉丁之间保留空格
    """
    if not words_list:
        return ""
    
    result = [words_list[0]]
    for i in range(1, len(words_list)):
        prev = words_list[i - 1]
        curr = words_list[i]
        if not prev or not curr:
            result.append(curr)
            continue
        prev_tail = prev[-1]
        curr_head = curr[0]
        # 前一个词尾或当前词头是 CJK 字符/标点 → 不加空格
        if (_is_cjk_char(prev_tail) or _is_cjk_punct(prev_tail) or
            _is_cjk_char(curr_head) or _is_cjk_punct(curr_head)):
            result.append(curr)
        else:
            result.append(" " + curr)
    return "".join(result)

def _clean_text(text: str, is_cjk: bool = False) -> str:
    """清理文本中多余的空格（始终同时处理 CJK 和拉丁标点）"""
    text = text.strip()
    
    # 移除 CJK 字符之间的空格（保留 CJK 与拉丁之间的空格）
    # 例: "雨 が 降 っ た" → "雨が降った"
    # 但: "Hello 世界" → "Hello 世界" (保留)
    _cjk_range = r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff01-\uff60]'
    _cjk_pattern = re.compile(
        f'({_cjk_range})\\s+({_cjk_range})'
    )
    prev = None
    while prev != text:
        prev = text
        text = _cjk_pattern.sub(r'\1\2', text)
    
    # 拉丁语系：移除标点前的多余空格
    text = re.sub(r'\s+([,.?!;:])', r'\1', text)
    return text

def seconds_to_srt_timestamp(seconds: float) -> str:
    if seconds is None: return "00:00:00,000"
    seconds = max(0.0, seconds)  # 防止负数时间戳
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_rem = seconds % 60
    secs = int(seconds_rem)
    millis = int((seconds_rem - secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def split_segment_by_punctuation(segment):
    """
    核心算法：
    1. 标点符号决定换行 (优先级最高)
    2. 大写字母辅助断句，但排除 NO_SPLIT_CAPS 中的特殊词（仅拉丁语系）
    3. 保留短句合并功能 (MIN_WORDS / MIN_CJK_CHARS)
    4. CJK 文本自动检测：跳过大写逻辑，仅按标点断句，不插入字间空格
    5. CJK 字符数强制分段：当无标点文本超过 MAX_CJK_LINE_CHARS 字符时，
       优先在日文助词后切分，次选在 CJK 字符边界处切分
    """
    sub_segments = []
    
    if 'words' in segment:
        words = segment['words']
        current_text = []
        current_start = None
        
        # 预判断整个 segment 是否为 CJK 文本
        all_text = " ".join(w.get('word', '') for w in words)
        is_cjk = _text_is_cjk_dominant(all_text)
        
        # 根据文本类型选择短句保护阈值
        min_units = MIN_CJK_CHARS if is_cjk else MIN_WORDS
        
        for i, word_obj in enumerate(words):
            if 'start' not in word_obj or 'end' not in word_obj:
                continue
                
            w_text = word_obj['word'].strip()
            
            if current_start is None:
                current_start = word_obj['start']
            
            # --- 1. 智能大写判断 (前置切分逻辑) --- 仅对非 CJK 文本启用
            is_upper_start = False
            
            if not is_cjk:
                # A. 必须是大写字母开头
                if (i > 0) and w_text and w_text[0].isupper():
                    is_upper_start = True
                    
                    # B. 【特殊词表】检查是否在不换行名单中
                    # 去除可能的标点 (例如 "TV," -> "TV") 再进行匹配
                    clean_word = w_text.rstrip(".,?!;:")
                    if clean_word in NO_SPLIT_CAPS or w_text.startswith("I'"):
                        is_upper_start = False

                    # C. 【标点优先】
                    # 如果这个大写词本身带着标点 (如 "AI," "English.")，说明它是上一句的结尾！
                    # 绝对不能在它前面切，要等加进去后，由后面的标点逻辑切。
                    if w_text[-1] in SPLIT_PUNCTUATION or w_text[-1] in SOFT_PUNCTUATION:
                        is_upper_start = False

            # D. 【短句保护】如果上一行太短，强制合并，不切分
            if is_upper_start and len(current_text) <= min_units:
                is_upper_start = False

            # -> 执行切分 (结算上一句)
            if is_upper_start and current_text:
                full_text = _clean_text(_smart_join(current_text, is_cjk), is_cjk)
                
                sub_segments.append({
                    "start": current_start,
                    "end": words[i-1]['end'], # 结束在上一词
                    "text": full_text,
                    "speaker": segment.get("speaker", "")
                })
                
                current_text = [w_text]
                current_start = word_obj['start']
                continue

            # --- 2. 常规累积 & 标点切分 (后置切分逻辑) ---
            current_text.append(w_text)
            
            # 检查标点
            is_last_word = (i == len(words) - 1)
            has_hard_punct = w_text and w_text[-1] in SPLIT_PUNCTUATION
            
            # 用 _smart_join 计算实际文本长度（CJK 不含多余空格）
            current_str = _smart_join(current_text, is_cjk)
            current_len = len(current_str)
            
            # CJK 文本按实际字符长度判断；拉丁语系按字符串长度
            soft_threshold = 10 if is_cjk else 15
            has_soft_punct = (w_text and w_text[-1] in SOFT_PUNCTUATION) and (current_len > soft_threshold)

            # --- 3. CJK 字符数强制分段 (无标点歌词场景) ---
            # 策略：
            #   a) 接近 MAX 时（MAX-3），如果当前词是日文助词 → 提前切分（语法边界更自然）
            #   b) 达到 MAX 时，在任何 CJK 字符处切分（硬回退，保证中文也能分段）
            force_cjk_split = False
            if is_cjk and not is_last_word and not has_hard_punct:
                # 优先：接近阈值时遇到日文助词 → 在语法边界处提前切分
                if current_len >= MAX_CJK_LINE_CHARS - 3 and w_text in _JP_PARTICLES:
                    force_cjk_split = True
                # 回退：达到阈值时在任何 CJK 字符处切分
                elif current_len >= MAX_CJK_LINE_CHARS and w_text and _is_cjk_char(w_text[0]):
                    force_cjk_split = True

            # 短句保护（仅对软标点生效，硬标点始终触发切分）
            is_long_enough = len(current_text) > min_units

            # -> 执行切分 (结算当前句)
            # 硬标点无条件切分；软标点需要句子超过阈值才切；CJK 强制分段
            if is_last_word or has_hard_punct or (is_long_enough and has_soft_punct) or force_cjk_split:
                full_text = _clean_text(current_str, is_cjk)

                if full_text:
                    sub_segments.append({
                        "start": current_start,
                        "end": word_obj['end'],
                        "text": full_text,
                        "speaker": segment.get("speaker", "")
                    })
                
                current_text = []
                current_start = None
                
    else:
        sub_segments.append(segment)

    return sub_segments

# --- 下面两个函数保持不变，仅作占位 ---
def generate_srt(segments: list) -> str:
    srt_content = []
    counter = 1
    for seg in segments:
        split_segs = split_segment_by_punctuation(seg)
        for sub_seg in split_segs:
            text = sub_seg['text'].strip()
            if not text: continue
            start = seconds_to_srt_timestamp(sub_seg['start'])
            end = seconds_to_srt_timestamp(sub_seg['end'])
            speaker = sub_seg.get('speaker', '')
            if speaker: text = f"[{speaker}] {text}"
            block = f"{counter}\n{start} --> {end}\n{text}\n"
            srt_content.append(block)
            counter += 1
    return "\n".join(srt_content)

def format_transcript_for_display(segments: list) -> str:
    lines = []
    for seg in segments:
        split_segs = split_segment_by_punctuation(seg)
        for sub_seg in split_segs:
            start = f"{sub_seg['start']:.2f}"
            end = f"{sub_seg['end']:.2f}"
            speaker = sub_seg.get('speaker', 'Unknown')
            text = sub_seg['text'].strip()
            if text:
                lines.append(f"[{start}s -> {end}s] [{speaker}]: {text}")
    return "\n".join(lines)