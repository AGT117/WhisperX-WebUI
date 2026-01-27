import sys
import os
import warnings
import logging
import config.settings as settings

# --- 1. 噪音抑制 ---
def suppress_noise():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    warnings.filterwarnings("ignore", message=".*torchaudio._backend.*") 
    warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
    warnings.filterwarnings("ignore", message=".*In 2.9, this function.*")
    warnings.filterwarnings("ignore", message=".*Model was trained with.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
    warnings.filterwarnings("ignore", message=".*The 'use_auth_token' argument.*")
    
    logging.getLogger("whisperx").setLevel(logging.ERROR)
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

suppress_noise()

# --- 2. 导入核心依赖 ---
import torch
import huggingface_hub
try:
    import huggingface_hub.file_download
    import pyannote.audio.core.inference
    from pyannote.audio import Pipeline # 必须在此导入
except ImportError:
    pass

from dotenv import load_dotenv

# --- 3. 核心补丁 ---
def apply_compatibility_patches():
    print("正在注入系统底层补丁 (VAD + Diarization)...")
    load_dotenv()
    
    # --- Patch A: PyTorch 权重检查 ---
    try:
        from omegaconf import DictConfig, ListConfig
        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    except ImportError: pass

    _original_load = torch.load
    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False}) if 'weights_only' in kwargs else _original_load(*args, **kwargs)

    # --- 通用参数转换逻辑 ---
    def _shim_kwargs(kwargs):
        if 'use_auth_token' in kwargs:
            token_val = kwargs.pop('use_auth_token')
            if 'token' not in kwargs and token_val is not None:
                kwargs['token'] = token_val
        return kwargs

    # --- Patch B: HuggingFace 下载补丁 ---
    if hasattr(huggingface_hub, 'hf_hub_download'):
        _orig_hf = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = lambda *a, **k: _orig_hf(*a, **_shim_kwargs(k))
    
    if hasattr(huggingface_hub, 'snapshot_download'):
        _orig_snap = huggingface_hub.snapshot_download
        huggingface_hub.snapshot_download = lambda *a, **k: _orig_snap(*a, **_shim_kwargs(k))

    # --- Patch C: Pyannote Inference 补丁 (修复 VAD) ---
    try:
        _orig_inf_init = pyannote.audio.core.inference.Inference.__init__
        def _inf_init_patch(self, *args, **kwargs):
            return _orig_inf_init(self, *args, **_shim_kwargs(kwargs))
        pyannote.audio.core.inference.Inference.__init__ = _inf_init_patch
    except: pass

    # --- Patch D: Pyannote Pipeline 补丁 (修复说话人聚类) 关键新增 ---
    try:
        from pyannote.audio import Pipeline
        _orig_from_pretrained = Pipeline.from_pretrained
        
        @classmethod
        def _from_pretrained_patch(cls, checkpoint_path, **kwargs):
            # 彻底移除或重命名 use_auth_token
            new_kwargs = _shim_kwargs(kwargs)
            return _orig_from_pretrained.__func__(cls, checkpoint_path, **new_kwargs)
            
        Pipeline.from_pretrained = _from_pretrained_patch
        print("已修复 Diarization Pipeline 兼容性")
    except Exception as e:
        print(f"Pipeline 补丁注入失败: {e}")

    # --- Patch E: 回退实现 AudioDecoder / AudioSamples / AudioStreamMetadata ---
    try:
        import pyannote.audio.core.io as py_io
        if not hasattr(py_io, 'AudioDecoder'):
            try:
                import torchaudio
                from io import IOBase
                from pathlib import Path

                class AudioStreamMetadata:
                    def __init__(self, sample_rate: int, num_frames: int):
                        self.sample_rate = sample_rate
                        self.duration_seconds_from_header = num_frames / sample_rate

                class AudioSamples:
                    def __init__(self, data, sample_rate: int):
                        self.data = data
                        self.sample_rate = sample_rate

                class AudioDecoder:
                    def __init__(self, source):
                        self.source = source
                        if isinstance(source, (str, Path)):
                            waveform, sr = torchaudio.load(str(source))
                        elif isinstance(source, IOBase):
                            # torchaudio may accept file-like objects depending on backend
                            waveform, sr = torchaudio.load(source)
                        else:
                            raise ValueError('Unsupported audio source for fallback AudioDecoder')
                        self._waveform = waveform
                        self._sr = sr
                        self.metadata = AudioStreamMetadata(sr, waveform.shape[1])

                    def get_all_samples(self):
                        return AudioSamples(self._waveform, self._sr)

                    def get_samples_played_in_range(self, start: float, end: float):
                        s = int(round(start * self._sr))
                        e = int(round(end * self._sr))
                        data = self._waveform[:, s:e]
                        return AudioSamples(data, self._sr)

                py_io.AudioDecoder = AudioDecoder
                py_io.AudioSamples = AudioSamples
                py_io.AudioStreamMetadata = AudioStreamMetadata
                print('已注入 pyannote AudioDecoder 回退实现 (使用 torchaudio)')
            except Exception as inner_e:
                print('注入 AudioDecoder 回退失败:', inner_e)
    except Exception:
        pass

    print("系统补丁应用完成")

apply_compatibility_patches()

# --- 4. 启动 WebUI ---
from src.ui.webui import create_ui

if __name__ == "__main__":
    print(f"\n服务启动序列初始化...")
    print(f"根目录: {settings.ROOT_DIR}")
    app = create_ui()
    print("服务就绪，正在启动 WebUI...\n")
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)