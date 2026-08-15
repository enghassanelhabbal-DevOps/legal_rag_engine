from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from legal_ai.core import RuntimeConfigLite


def _parse_dotenv(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"')
        env[key] = val
    return env


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


@dataclass
class Config:
    # Minimal set of config values used by the refactor bootstrap
    runtime_device: str = "auto"
    runtime_gpu_id: int = 0
    runtime_precision: str = "auto"
    dense_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    faiss_index_path: str = "artifacts_portable/faiss.index"
    embeddings_cache: str = "artifacts_portable/embeddings.json"
    llm_backend: str = "qwen_transformers"


def load_config(env_path: Optional[str] = None, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a .env file, optional YAML, and environment variables.

    Precedence (highest -> lowest): os.environ > .env file > yaml file
    """
    cfg: Dict[str, Any] = {}
    if yaml_path:
        cfg.update(_load_yaml(Path(yaml_path)))
    if env_path:
        cfg.update(_parse_dotenv(Path(env_path)))
    # Finally, overlay actual environment variables for runtime overrides
    for k, v in os.environ.items():
        cfg[k] = v
    return cfg


def load_runtime_config(cfg_map: Optional[Dict[str, Any]] = None) -> RuntimeConfigLite:
    """Create a RuntimeConfigLite from a loaded config map or environment.
    """
    cfg_map = cfg_map or load_config(env_path=".env")
    device = cfg_map.get("RUNTIME_DEVICE", cfg_map.get("runtime_device", "auto"))
    gpu_id = int(cfg_map.get("RUNTIME_GPU_ID", cfg_map.get("runtime_gpu_id", 0)))
    precision = cfg_map.get("RUNTIME_PRECISION", cfg_map.get("runtime_precision", "auto"))
    dense_batch = int(cfg_map.get("DENSE_BATCH_SIZE", cfg_map.get("dense_batch_size", 32)))
    rerank_batch = int(cfg_map.get("RERANK_BATCH_SIZE", cfg_map.get("rerank_batch_size", 32)))
    return RuntimeConfigLite(device=device, gpu_id=gpu_id, precision=precision, dense_batch_size=dense_batch, rerank_batch_size=rerank_batch)


if __name__ == "__main__":
    # Simple CLI for debugging
    import json
    cfg_map = load_config(env_path=".env")
    print(json.dumps(cfg_map, ensure_ascii=False, indent=2))
