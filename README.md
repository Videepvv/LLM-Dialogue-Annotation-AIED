# LLM Dialogue Annotation for Educational Discourse

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for automatically annotating educational dialogue using Large Language Models (LLMs), as presented in our AIED 2026 paper.

## 📋 Overview

We evaluate LLM performance on three educational dialogue annotation tasks:

| Dataset | Task | Labels | Samples |
|---------|------|--------|---------|
| **TalkMoves** | Teacher talk moves | 7 classes | 150,918 |
| **TalkMoves** | Student talk moves | 5 classes | 52,683 |
| **DELI** | Dialogue type & target | 3 + 6 classes | 14,003 |
| **Weights Task** | CPS facets | 3 binary facets | 2,400 |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n llm-annotation python=3.11 -y
conda activate llm-annotation

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models from Hugging Face

```bash
# Login to Hugging Face (required for Llama models)
huggingface-cli login

# Download models (example)
python scripts/download_models.py --model llama-8b --output-dir ./models
```

### 3. Download Datasets

```bash
# TalkMoves
git clone https://github.com/AshishJumbo/TalkMoves.git data/TalkMoves

# DELI - Contact authors or download from source
# CPS Weights Task - Included in data/ directory
```

### 4. Run Inference

```bash
# TalkMoves annotation
python src/run_talkmoves.py --model llama-8b --data_type teacher --n_samples 1000

# DELI annotation
python src/run_deli.py --model llama-8b --n_samples 1000

# CPS annotation
python src/run_cps.py --model llama-8b --n_samples 1000
```

### 5. Calculate Metrics

```bash
# Calculate Kappa scores
python src/calculate_kappa.py --dataset talkmoves

# Generate analysis report
python src/analysis.py --all
```

## 📁 Repository Structure

```
LLM-Dialogue-Annotation-AIED/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── run_talkmoves.py      # TalkMoves inference
│   ├── run_deli.py           # DELI inference
│   ├── run_cps.py            # CPS inference
│   ├── calculate_kappa.py    # Kappa calculation
│   └── analysis.py           # Per-label analysis
├── data/                     # Datasets (download separately)
│   ├── TalkMoves/            # TalkMoves dataset
│   ├── DeliData/             # DELI dataset
│   └── WTD/                  # Weights Task dataset
├── results/                  # Output results
│   └── .gitkeep
├── scripts/                  # Utility scripts
│   ├── download_models.py    # Model download helper
│   └── run_all_models.sh     # Batch inference script
└── configs/                  # Configuration files
    └── models.yaml           # Model configurations
```

## 🔧 Supported Models

| Model | HuggingFace ID | Size |
|-------|---------------|------|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | ~16GB |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | ~14GB |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | ~15GB |
| Gemma-2-9B | `google/gemma-2-9b-it` | ~18GB |
| Phi-3.5 | `microsoft/Phi-3.5-mini-instruct` | ~8GB |

## 📊 Results

### Cohen's Kappa Scores

| Dataset | Task | Best Model | Kappa |
|---------|------|------------|-------|
| TalkMoves | Teacher | Qwen-7B | 0.190 |
| TalkMoves | Student | Llama-8B | 0.446 |
| DELI | Type | Llama-8B | 0.649 |
| DELI | Target | Llama-8B | 0.594 |

### Key Findings

1. **LLMs struggle with pedagogically nuanced labels** (Revoicing, Press for Reasoning)
2. **Surface form vs. function**: Models succeed on labels with explicit markers but fail when the same surface form serves multiple functions
3. **Teacher moves harder than student moves**: κ gap of ~0.25

## 📝 Usage Examples

### Single Utterance Annotation

```python
from src.annotator import DialogueAnnotator

annotator = DialogueAnnotator(model="llama-8b")

# Teacher talk move
result = annotator.classify_teacher(
    context="Student: The answer is 42",
    utterance="Why do you think that?"
)
print(result)  # {"category": 3, "move": 6, "label": "Press for Reasoning"}
```

### Batch Processing

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor(model="qwen-7b", dataset="talkmoves")
results = processor.run(split="test", output_path="results/output.json")
```

## 🧮 Annotation Schema

### TalkMoves Teacher Labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | Other | Non-instructional talk |
| 1 | Keep Together | Managing attention/turns |
| 2 | Students Relate | Prompting peer engagement |
| 3 | Revoicing | Repeating student contributions |
| 4 | Press Accuracy | Checking correctness |
| 5 | Press Reasoning | Asking for explanations |
| 6 | Challenge | Questioning student ideas |

### DELI Type Labels

| ID | Label | Description |
|----|-------|-------------|
| -1 | None | Off-topic/greetings |
| 0 | Probing | Questions provoking discussion |
| 1 | NPD | Non-probing deliberation |

## 📖 Citation

```bibtex
@inproceedings{author2026llm,
  title={Evaluating LLMs for Educational Dialogue Annotation},
  author={Author, Name},
  booktitle={Proceedings of AIED 2026},
  year={2026}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- TalkMoves dataset: [Suresh et al., 2022](https://github.com/AshishJumbo/TalkMoves)
- DELI dataset: [Karadzhov et al., 2021](https://github.com/GT-SALT/DELI)
