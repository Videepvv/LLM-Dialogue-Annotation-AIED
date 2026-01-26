#!/usr/bin/env python3
"""
Download LLMs from Hugging Face for inference.

Usage:
    python download_models.py --model llama-8b --output-dir ./models
    python download_models.py --all --output-dir ./models
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login
import os

MODELS = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "gemma-9b": "google/gemma-2-9b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
}


def download_model(model_name: str, output_dir: Path):
    """Download a single model."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODELS.keys())}")
        return
    
    repo_id = MODELS[model_name]
    local_dir = output_dir / model_name
    
    print(f"Downloading {model_name} ({repo_id})...")
    print(f"  Destination: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Complete!")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models")
    parser.add_argument("--model", type=str, help="Model name to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--login", action="store_true", help="Login to HuggingFace first")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.login:
        print("Logging in to HuggingFace...")
        login()
    
    if args.all:
        for model_name in MODELS:
            download_model(model_name, output_dir)
    elif args.model:
        download_model(args.model, output_dir)
    else:
        print("Specify --model <name> or --all")
        print(f"Available models: {list(MODELS.keys())}")


if __name__ == "__main__":
    main()
