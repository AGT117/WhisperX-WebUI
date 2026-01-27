import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- 项目路径配置 ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

# 子目录定义
HF_HOME = MODELS_DIR / "huggingface"
TORCH_HOME = MODELS_DIR / "torch"
CACHE_HOME = MODELS_DIR / "cache"
AUDIO_SEPARATOR_HOME = MODELS_DIR / "audio-separator"

# 初始化目录结构
for p in [MODELS_DIR, HF_HOME, TORCH_HOME, CACHE_HOME, AUDIO_SEPARATOR_HOME, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- 环境变量注入 ---
# 必须在导入 torch/whisperx 之前设置，以确保模型下载至指定目录

# Hugging Face 缓存路径
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME)

# Torch 缓存路径
os.environ["TORCH_HOME"] = str(TORCH_HOME)

# 通用缓存路径 (用于 Wav2Vec2 等对齐模型)
os.environ["XDG_CACHE_HOME"] = str(CACHE_HOME)

# --- 全局参数 ---
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda"

# 计算精度配置
# 8GB VRAM 建议 float16；若显存不足可切换至 int8
COMPUTE_TYPE = "float16" 

# 批处理大小
# 根据显存容量调整，8GB VRAM 建议值为 4
BATCH_SIZE = 4