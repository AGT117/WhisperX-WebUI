import math
import re

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

# 最小单词数阈值 (防止过短的行)
MIN_WORDS = 2

def seconds_to_srt_timestamp(seconds: float) -> str:
    if seconds is None: return "00:00:00,000"
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
    2. 大写字母辅助断句，但排除 NO_SPLIT_CAPS 中的特殊词
    3. 保留短句合并功能 (MIN_WORDS)
    """
    sub_segments = []
    
    if 'words' in segment:
        words = segment['words']
        current_text = []
        current_start = None
        
        for i, word_obj in enumerate(words):
            if 'start' not in word_obj or 'end' not in word_obj:
                continue
                
            w_text = word_obj['word'].strip()
            
            if current_start is None:
                current_start = word_obj['start']
            
            # --- 1. 智能大写判断 (前置切分逻辑) ---
            is_upper_start = False
            
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
            if is_upper_start and len(current_text) <= MIN_WORDS:
                is_upper_start = False

            # -> 执行切分 (结算上一句)
            if is_upper_start and current_text:
                full_text = " ".join(current_text).strip()
                full_text = re.sub(r'\s+([,.?!;:])', r'\1', full_text)
                
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
            
            # 软标点：句子要够长才切
            current_str = " ".join(current_text)
            current_len = len(current_str)
            has_soft_punct = (w_text and w_text[-1] in SOFT_PUNCTUATION) and (current_len > 15)

            # 短句保护
            is_long_enough = len(current_text) > MIN_WORDS

            # -> 执行切分 (结算当前句)
            if is_last_word or (is_long_enough and (has_hard_punct or has_soft_punct)):
                full_text = current_str.strip()
                full_text = re.sub(r'\s+([,.?!;:])', r'\1', full_text)

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