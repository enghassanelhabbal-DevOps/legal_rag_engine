from __future__ import annotations

import platform
import sys

import faiss
import torch

print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("FAISS:", getattr(faiss, "__version__", "unknown"))
print("torch.version.cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  VRAM GB: {props.total_memory / 1024**3:.2f}")
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported(i)}")
        print(f"  Capability: {props.major}.{props.minor}")
