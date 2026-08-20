"""Lightweight repo audit script for Phase 1.

Usage:
  python scripts\repo_audit.py

This script is safe: it inspects files and prints a short summary. It does not execute heavy model loading.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_FILES = [
    "legal_rag_engine.py",
    "qwen_transformers_backend.py",
    "app_qwen_m2200.py",
    "llm_backend_template.py",
    "check_gpu.py",
    "README_portable_windows_linux.md",
]


def main():
    print("Repository root:", ROOT)
    print("Python executable:", os.sys.executable)
    print("Checking key files:")
    for f in KEY_FILES:
        p = ROOT / f
        print(f" - {f}: {'FOUND' if p.exists() else 'MISSING'}")

    # Quick size stats for repo top-level
    total_files = 0
    total_bytes = 0
    for p in ROOT.rglob('*'):
        if p.is_file():
            total_files += 1
            try:
                total_bytes += p.stat().st_size
            except Exception:
                pass
    print(f"Total files: {total_files:,}, total bytes approx: {total_bytes:,}")

    # Warn about .venv presence
    venv = ROOT / '.venv'
    if venv.exists():
        print('\nWARNING: .venv directory found in repository root. Do not commit virtual environment contents.')

    print('\nSuggested next actions:')
    print(' - Create legal_ai/core.py with core dataclasses and Protocols (non-breaking).')
    print(' - Add .env.example and config loader in legal_ai/config.')
    print(' - Add tests/ with at least normalization unit tests.')


if __name__ == '__main__':
    main()
