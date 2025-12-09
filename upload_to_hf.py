#!/usr/bin/env python3
"""Upload JSONL dataset to Hugging Face Hub."""

from datasets import Dataset
from huggingface_hub import HfApi
import json

def load_jsonl(file_path: str) -> list[dict]:
    """Load JSONL file and return list of records."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    # Configuration
    input_file = "data/251208_132307_filtered.jsonl"
    repo_id = "aigise/gemini_251208_132307_filtered"

    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} records")

    # Create dataset
    dataset = Dataset.from_list(data)
    print(f"Dataset created with {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")

    # Upload to Hugging Face
    print(f"\nUploading to {repo_id}...")
    dataset.push_to_hub(repo_id, private=False)
    print(f"Done! Dataset available at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()