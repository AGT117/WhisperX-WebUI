import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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

# 设备自动检测：优先使用 CUDA，无 GPU 时回退到 CPU
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    logger.warning("未检测到可用的 CUDA 设备，将使用 CPU 模式（速度极慢）")
    return "cpu"

DEVICE = os.getenv("DEVICE", _detect_device())

# 计算精度配置
# 8GB VRAM 建议 float16；若显存不足可切换至 int8；CPU 模式使用 float32
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "float32")

# 批处理大小
# 根据显存容量调整，8GB VRAM 建议值为 4
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))