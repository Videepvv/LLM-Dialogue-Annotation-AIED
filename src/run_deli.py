#!/usr/bin/env python3
"""
DELI Dataset Annotation Inference

Two-stage approach for collaborative problem-solving dialogue:
- Stage 1: Type classification (None, Probing, NPD)
- Stage 2: Target classification (Solution, Reasoning, Moderation, Agree, Disagree)

Usage:
    python run_deli.py --model llama-8b --n_samples 1000
"""

import argparse
import json
import re
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATA_PATH = Path("./data/DeliData/delidata_train.csv")
DEFAULT_RESULTS_DIR = Path("./results/DELI")
DEFAULT_MODELS_DIR = Path("./models")

MODELS = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-9b": "google/gemma-2-9b-it",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
}

# Encoding mappings
TYPE_ENCODING = {"None": -1, "0": -1, "Probing": 0, "NPD": 1, "Non-probing-deliberation": 1}
TARGET_ENCODING = {"None": 0, "0": 0, "Solution": 1, "Reasoning": 2, "Moderation": 3, "Agree": 4, "Disagree": 5}

TYPE_DECODE = {-1: "None", 0: "Probing", 1: "NPD"}
TARGET_DECODE = {0: "None", 1: "Solution", 2: "Reasoning", 3: "Moderation", 4: "Agree", 5: "Disagree"}

# ============================================================================
# System Prompts
# ============================================================================

TYPE_SYSTEM_PROMPT = """You are an expert in analyzing collaborative problem-solving dialogue.

## Background: The Wason Card Selection Task

Groups of 3-5 participants work together online to solve the Wason Card Selection Task—a classic test of logical reasoning. 

**The Task:**
Participants are shown 4 cards, each with a letter on one side and a number on the other. Only one side is visible. They must decide which cards to flip to TEST a rule like: "All cards with a vowel on one side have an even number on the other side."

Example cards: A, K, 4, 7
- Correct answer: Flip A (to check if vowel has even) and 7 (to check if odd has no vowel)
- Common mistake: Flipping 4 (doesn't test the rule—consonants can have even numbers)

## Task: Classify Dialogue Type

Classify each utterance into ONE of these three types:

**-1 = None**: Greetings, off-topic remarks, hesitation cues.
- Examples: "Hello", "hmm...", "Thanks!", "lol"

**0 = Probing**: Questions that PROVOKE discussion WITHOUT introducing novel information.
- Examples: "What did everybody put?", "Why did you choose 6?", "Do we all agree?"

**1 = NPD (Non-Probing Deliberation)**: Statements useful for the task—discussing solutions, reasoning, or expressing positions.
- Examples: "I think the answer is A because we need to check vowels", "I put 6 and S", "Yes I agree"

Output ONLY: {"type": <integer>, "reasoning": "brief explanation"}"""


TARGET_SYSTEM_PROMPT = """You are an expert in analyzing collaborative problem-solving dialogue.

## Task: Classify Target/Role

Given an utterance classified as Probing or NPD, classify its TARGET:

**1 = Solution**: Discusses which cards to flip or proposes answers
- "I think the answer is 7 and A", "What cards did you pick?"

**2 = Reasoning**: Provides justification or asks for reasoning
- "Because A is a vowel", "Why did you think that?"

**3 = Moderation**: Coordinates discussion dynamics
- "Let's vote", "Do we all agree?", "Final answer?"

**4 = Agree** (NPD only): Expressing agreement
- "yes", "I agree", "sounds good"

**5 = Disagree** (NPD only): Expressing disagreement
- "No", "I don't think so"

Output ONLY: {"target": <integer>, "reasoning": "brief explanation"}"""


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str, models_dir: Path, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    local_path = models_dir / model_name
    model_path = local_path if local_path.exists() else MODELS.get(model_name, model_name)
    
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Load DELI dialogue data, filtering to MESSAGE type only."""
    df = pd.read_csv(data_path)
    df = df[df['message_type'] == 'MESSAGE'].reset_index(drop=True)
    return df


def get_conversation_history(df: pd.DataFrame, current_idx: int) -> list:
    """Get previous turns from the same group as context."""
    current_row = df.iloc[current_idx]
    current_group = current_row['group_id']
    
    history = []
    for i in range(max(0, current_idx - 20), current_idx):  # Limit to last 20
        row = df.iloc[i]
        if row['group_id'] == current_group:
            history.append(f"[{row['origin']}]: {row['clean_text']}")
    
    return history


def get_ground_truth(row: pd.Series) -> dict:
    """Extract and encode ground truth annotations."""
    raw_type = str(row['annotation_type']) if pd.notna(row['annotation_type']) else "0"
    raw_target = str(row['annotation_target']) if pd.notna(row['annotation_target']) else "0"
    
    return {
        "type": TYPE_ENCODING.get(raw_type, -1),
        "target": TARGET_ENCODING.get(raw_target, 0),
    }


# ============================================================================
# Inference Functions
# ============================================================================

def format_type_prompt(history: list, utterance: str, participant: str) -> str:
    history_str = "\n".join(history) if history else "(First message in group)"
    return f"""Conversation history:
{history_str}

Current utterance to classify:
[{participant}]: {utterance}

Classify the TYPE: -1=None, 0=Probing, 1=NPD

Output ONLY: {{"type": <integer>, "reasoning": "brief explanation"}}"""


def format_target_prompt(history: list, utterance: str, participant: str, utterance_type: int) -> Optional[str]:
    if utterance_type == -1:
        return None
    
    history_str = "\n".join(history) if history else "(First message)"
    type_name = "Probing" if utterance_type == 0 else "NPD"
    targets = "1=Solution, 2=Reasoning, 3=Moderation" + (", 4=Agree, 5=Disagree" if utterance_type == 1 else "")
    
    return f"""Conversation history:
{history_str}

Current utterance (Type: {type_name}):
[{participant}]: {utterance}

Classify TARGET: {targets}

Output ONLY: {{"target": <integer>, "reasoning": "brief explanation"}}"""


def parse_json_with_reasoning(output: str, field: str, default: int) -> tuple:
    """Parse integer field and reasoning from JSON output."""
    reasoning = ""
    json_match = re.search(r'\{[^{}]*\}', output)
    if json_match:
        try:
            result = json.loads(json_match.group())
            reasoning = result.get("reasoning", "")
            if field in result:
                return int(result[field]), reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    pattern = rf'"{field}"\s*:\s*(-?\d+)'
    match = re.search(pattern, output)
    if match:
        return int(match.group(1)), reasoning
    
    return default, reasoning


def run_inference(model, tokenizer, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.6, max_new_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DELI Annotation Inference")
    parser.add_argument("--model", type=str, default="llama-8b")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print(f"DELI Annotation | Model: {args.model} | Samples: {args.n_samples}")
    print("=" * 60)
    
    df = load_data(Path(args.data_path))
    print(f"Dataset: {len(df)} messages")
    
    model, tokenizer = load_model(args.model, Path(args.models_dir), args.device)
    
    results = []
    type_correct, target_correct, total = 0, 0, 0
    
    end_idx = min(args.start_idx + args.n_samples, len(df))
    
    for i in range(args.start_idx, end_idx):
        row = df.iloc[i]
        utterance = row['clean_text']
        participant = row['origin']
        history = get_conversation_history(df, i)
        ground_truth = get_ground_truth(row)
        
        # Stage 1: Type
        type_prompt = format_type_prompt(history, utterance, participant)
        type_output = run_inference(model, tokenizer, TYPE_SYSTEM_PROMPT, type_prompt, args.temperature)
        pred_type, type_reasoning = parse_json_with_reasoning(type_output, "type", -1)
        
        # Stage 2: Target
        pred_target = 0
        target_reasoning = ""
        if pred_type != -1:
            target_prompt = format_target_prompt(history, utterance, participant, pred_type)
            if target_prompt:
                target_output = run_inference(model, tokenizer, TARGET_SYSTEM_PROMPT, target_prompt, args.temperature)
                pred_target, target_reasoning = parse_json_with_reasoning(target_output, "target", 0)
        
        type_match = pred_type == ground_truth["type"]
        target_match = pred_target == ground_truth["target"]
        if type_match: type_correct += 1
        if target_match: target_correct += 1
        total += 1
        
        result = {
            "idx": i,
            "group_id": row['group_id'],
            "participant": participant,
            "utterance": utterance,
            "gt_type": ground_truth["type"],
            "gt_target": ground_truth["target"],
            "pred_type": pred_type,
            "pred_target": pred_target,
            "type_reasoning": type_reasoning,
            "target_reasoning": target_reasoning,
            "type_match": type_match,
            "target_match": target_match,
        }
        results.append(result)
        
        sample_num = i - args.start_idx + 1
        if sample_num % 10 == 0 or sample_num <= 5:
            print(f"  [{sample_num}/{args.n_samples}] GT: {ground_truth['type']},{ground_truth['target']} | Pred: {pred_type},{pred_target}")
        
        if sample_num % 50 == 0:
            with open(results_dir / f"deli_{args.model}.json", "w") as f:
                json.dump(results, f, indent=2)
    
    output_path = results_dir / f"deli_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Type accuracy:   {type_correct}/{total} ({100*type_correct/total:.1f}%)")
    print(f"Target accuracy: {target_correct}/{total} ({100*target_correct/total:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
