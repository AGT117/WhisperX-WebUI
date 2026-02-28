# WhisperX WebUI

基于 WhisperX 的本地化音频转录与字幕生成工具。

集成人声分离（BS-RoFormer）、音素级强制对齐、说话人聚类（Pyannote）与 LLM 智能后处理，通过 Gradio 网页界面进行操作，支持批量处理音频/视频文件，导出 JSON 与 SRT 字幕。

---

## 功能概述

- **语音识别 (ASR)**：基于 WhisperX + Faster-Whisper，支持音素级强制对齐，时间轴精确到词级别。
- **人声分离**：内置 BS-RoFormer 预处理，自动分离人声与背景音乐/噪声，提升嘈杂环境下的识别质量。
- **说话人聚类**：基于 Pyannote 的 Speaker Diarization，自动区分多个发言者。
- **幻觉过滤**：三种模式可选 — 代码规则过滤（置信度 + 模式匹配 + 时间异常检测）、LLM 智能过滤、关闭。
- **LLM 智能处理**：接入大语言模型（OpenAI / DeepSeek / Ollama 等兼容接口），实现智能断句与翻译（支持双语字幕）。自动分批处理以适应模型上下文长度限制。
- **CJK 文本优化**：针对中文、日文、韩文的专用分段算法，支持时间间隙检测、功能词/助词边界、字数控制三重策略。
- **批量处理**：多文件队列处理，自动生成 JSON 与 SRT 输出。
- **可执行文件**：可通过 PyInstaller 打包为轻量级 .exe 启动器，双击即可运行。

---

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.10+ | 推荐 3.11 |
| FFmpeg | 音频格式转换，须在系统 PATH 中可用 |
| NVIDIA 显卡驱动 + CUDA 12.x | GPU 加速（无 N 卡可用 CPU 模式，速度极慢） |
| Git | 安装 WhisperX 时需要 |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/AGT117/WhisperX-WebUI.git
cd WhisperX-WebUI
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入必要信息：

```ini
HF_TOKEN=hf_你的Token
```

> 说话人聚类功能需要 Hugging Face Token，并在 Hugging Face 网站接受以下模型的用户协议：
> - pyannote/speaker-diarization-3.1
> - pyannote/segmentation-3.0

### 5. 配置 LLM（可选）

编辑 `config/llm_config.json`，填入 API 信息：

```json
{
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "sk-...",
    "model": "deepseek-chat",
    "max_context_tokens": 8192,
    "temperature": 0.3
}
```

支持所有 OpenAI 兼容接口：OpenAI、DeepSeek、Ollama（本地 `http://localhost:11434/v1`）等。

### 6. 启动

```bash
python main.py
```

启动后浏览器自动打开 `http://127.0.0.1:7860`。

也可使用`WhisperX-WebUI.exe` 双击启动。

---

## 项目结构

```
WhisperX-WebUI/
├── main.py                     # 程序入口，兼容性补丁，启动 WebUI
├── requirements.txt            # 依赖清单
├── .env.example                # 环境变量配置模板
├── config/
│   ├── settings.py             # 全局配置（路径、设备、精度、LLM 加载）
│   └── llm_config.json.example         # LLM API 配置文件模板
├── src/
│   ├── core/
│   │   ├── engine.py           # 核心管道（ASR、对齐、幻觉过滤、LLM、聚类）
│   │   ├── utils.py            # 文本处理、CJK 分段、SRT 生成、幻觉过滤规则
│   │   └── llm_processor.py    # LLM 集成（断句、翻译、幻觉判断）
│   └── ui/
│       └── webui.py            # Gradio 界面
├── tests/
│   ├── test_utils.py           # 文本处理单元测试
│   └── test_llm_processor.py   # LLM 模块单元测试
├── models/                     # 模型缓存目录（自动下载）
├── outputs/                    # 默认输出目录
└── demo/                       # 演示文件
```

---

## 处理管道

```
输入音频 → [人声分离] → ASR 转录 → 音素级对齐 → [幻觉过滤] → [LLM 断句/翻译] → [说话人聚类] → JSON + SRT 输出
```

方括号表示可选步骤。

---

## 配置说明

### 环境变量（.env）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HF_TOKEN` | Hugging Face Token（说话人聚类必需） | 无 |
| `DEVICE` | 运行设备 | 自动检测（cuda / cpu） |
| `COMPUTE_TYPE` | 推理精度（float16 / int8 / float32） | GPU: float16, CPU: float32 |
| `BATCH_SIZE` | 转录批处理大小 | 4 |

### LLM 配置（config/llm_config.json）

| 字段 | 说明 |
|------|------|
| `api_base` | OpenAI 兼容 API 地址 |
| `api_key` | API 密钥 |
| `model` | 模型 ID |
| `max_context_tokens` | 上下文窗口大小（自动分批） |
| `temperature` | 生成温度（0.0-1.0） |

---

## 许可证

MIT License