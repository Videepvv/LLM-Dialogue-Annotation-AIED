#!/usr/bin/env python3
"""
Run ablation tests for DELI annotation.

Tests combinations of:
- Context history: 0, 3, 5, 10 turns
- Chain-of-thought: on/off

Two-stage classification:
1. Type (None, Probing, NPD)
2. Target (Reasoning, Solution, Agree, etc.) if Type != None

Usage:
    python run_ablation_deli.py --model llama-3b --history_window 5 --cot on
"""

import argparse
import json
import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import cohen_kappa_score

# ============================================================================
# Configuration
# ============================================================================

# Use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
ABLATION_DATA_PATH = SCRIPT_DIR / "data/ablation/deli_ablation_200.csv"
FULL_DATA_PATH = SCRIPT_DIR / "data/GoldenData/DeliData/delidata_train.csv"
RESULTS_DIR = SCRIPT_DIR / "results/ablation"
MODELS_DIR = Path("/data/open-weight-llms/models")

MODELS = {
    "llama-8b": "llama-8b",
    "llama-70b": "llama-70b",
    "llama-3.2-11b": "llama-3.2-11b",
    "qwen-7b": "qwen-7b",
    "qwen-14b": "qwen-14b",
    "qwen-32b": "qwen-32b",
    "qwen-72b": "qwen-72b",
    "gemma-9b": "gemma-9b",
    "mistral-7b": "mistral-7b",
    "mistral-nemo": "mistral-nemo",
    "mixtral-8x7b": "mixtral-8x7b",
    "deepseek-v2-lite": "deepseek-v2-lite",
}

HISTORY_WINDOWS = [0, 3, 5, 10, -1]
COT_OPTIONS = [True, False]

TYPE_DECODE = {-1: "None", 0: "Probing", 1: "NPD"}
TARGET_DECODE = {0: "None", 1: "Solution", 2: "Reasoning", 3: "Moderation", 4: "Agree", 5: "Disagree"}
TYPE_ENCODE = {v: k for k, v in TYPE_DECODE.items()}
TARGET_ENCODE = {v: k for k, v in TARGET_DECODE.items()}

# System Prompts
TYPE_SYSTEM_PROMPT = """You are an expert in analyzing collaborative problem-solving dialogue.

Classify the user's utterance into one of these types:

**-1 = None**: Not task-related (social chat, off-topic) or simple acknowledgments like "ok".
- Examples: "Hi", "I'm tired", "Ok", "Cool"

**0 = Probing**: Asking questions to understand others' thinking or the task.
- Examples: "What did everybody put?", "Why did you choose 6?", "Do we all agree?"

**1 = NPD (Non-Probing Deliberation)**: Statements useful for the task—discussing solutions, reasoning, or expressing positions.
- Examples: "I think the answer is A because we need to check vowels", "I put 6 and S", "Yes I agree"

"""

TARGET_SYSTEM_PROMPT = """Now classify the specific FUNCTION (Target) of this contribution.

**0 = None**: Generic or unclear task talk.
**1 = Solution**: Proposing or stating a solution/answer.
- Examples: "I put 4", "The answer is K"
**2 = Reasoning**: Explaining WHY, analyzing rules, or evaluating logic.
- Examples: "Because the rule says even numbers have vowels", "If we flip A it might be odd"
**3 = Moderation**: Managing the group process or task flow.
- Examples: "Let's move to the next one", "We have 2 minutes left", "Wait so we agree?"
**4 = Agree**: Explicitly agreeing with an idea.
- Examples: "Yes", "I agree", "Same here"
**5 = Disagree**: Explicitly disagreeing or correcting.
- Examples: "No that's wrong", "I don't think so", "But A is a vowel"

"""

def load_model(model_name: str, device: str = "auto"):
    model_folder = MODELS.get(model_name, model_name)
    model_path = MODELS_DIR / model_folder
    
    if not model_path.exists():
        raise ValueError(f"Model not found: {model_path}")
    
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

def get_conversation_history(df: pd.DataFrame, current_idx: int, window: int = 5) -> list:
    if window == 0:
        return []
    
    # DELI groups messages by group_id
    row = df.iloc[current_idx]
    group_id = row['group_id']
    
    # Get indices of same group up to current_idx
    group_rows = df[(df['group_id'] == group_id) & (df.index < current_idx)]
    
    # Take last N
    if window == -1:
        history_rows = group_rows # All previous in group
    else:
        history_rows = group_rows.tail(window)
    
    history = []
    for _, r in history_rows.iterrows():
        p = str(r['origin'])[:4]  # Truncate long IDs
        t = str(r['clean_text'])
        history.append(f"{p}: {t}")
        
    return history

def get_ground_truth(row: pd.Series):
    annot_type = row.get("annotation_type", "None")
    annot_target = row.get("annotation_target", "None")
    
    if annot_type == "Non-probing-deliberation":
        type_int = 1
    elif annot_type == "Probing":
        type_int = 0
    else:
        type_int = -1
        
    target_int = TARGET_ENCODE.get(annot_target, 0)
    
    return type_int, target_int

def format_type_prompt(history: list, utterance: str, participant: str, use_cot: bool) -> str:
    base = TYPE_SYSTEM_PROMPT
    
    if history:
        base += "Dialogue History:\n" + "\n".join(history) + "\n\n"
    else:
        base += "Dialogue History: (No prior turns)\n\n"
        
    base += f"Current Utterance ({participant}): {utterance}\n\n"
    
    if use_cot:
        base += """Output ONLY JSON:
{"type": <integer>, "reasoning": "brief explanation"}"""
    else:
        base += """Output ONLY JSON:
{"type": <integer>}"""
    
    return base

def format_target_prompt(history: list, utterance: str, participant: str, p_type: int, use_cot: bool) -> str:
    base = TARGET_SYSTEM_PROMPT
    
    if history:
        base += "Dialogue History:\n" + "\n".join(history) + "\n\n"
    
    base += f"Current Utterance ({participant}): {utterance}\n"
    base += f"Classified Type: {TYPE_DECODE.get(p_type, 'Unknown')}\n\n"
    
    if use_cot:
        base += """Output ONLY JSON:
{"target": <integer>, "reasoning": "brief explanation"}"""
    else:
        base += """Output ONLY JSON:
{"target": <integer>}"""
    
    return base

def run_inference(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def parse_json(output: str, field: str, default: int, use_cot: bool) -> dict:
    # Try regex json
    json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            val = data.get(field)
            if val is not None:
                return {
                    "val": int(val),
                    "reasoning": data.get("reasoning", "") if use_cot else "",
                    "success": True
                }
        except:
            pass
            
    # Fallback to finding simple integer
    # Look for "type": 1 or just 1
    int_match = re.search(fr'"{field}"\s*:\s*(-?\d+)', output)
    if int_match:
        return {"val": int(int_match.group(1)), "reasoning": "", "success": True}
        
    return {"val": default, "reasoning": "", "success": False}

def run_ablation_config(model, tokenizer, ablation_df, full_df, history_window: int, use_cot: bool):
    results = []
    gt_types = []
    pred_types = []
    gt_targets = []
    pred_targets = []
    
    parse_failures = 0
    
    for _, row in ablation_df.iterrows():
        original_idx = row['original_index']
        full_row = full_df.iloc[original_idx]
        
        participant = str(full_row.get("origin", "User"))[:4]
        utterance = str(full_row.get("clean_text", ""))
        
        # Ground truth
        gt_type, gt_target = get_ground_truth(full_row)
        
        # History
        history = get_conversation_history(full_df, original_idx, history_window)
        
        # Stage 1: Type
        type_prompt = format_type_prompt(history, utterance, participant, use_cot)
        type_out = run_inference(model, tokenizer, type_prompt)
        parsed_type = parse_json(type_out, "type", -1, use_cot)
        pred_type = parsed_type["val"]
        pred_type = max(-1, min(1, pred_type)) # clip
        
        if not parsed_type["success"]:
            parse_failures += 1
            
        stage_2_reasoning = ""
        pred_target = 0
        
        # Stage 2: Target (if type != -1)
        if pred_type != -1:
            target_prompt = format_target_prompt(history, utterance, participant, pred_type, use_cot)
            target_out = run_inference(model, tokenizer, target_prompt)
            parsed_target = parse_json(target_out, "target", 0, use_cot)
            pred_target = parsed_target["val"]
            pred_target = max(0, min(5, pred_target)) # clip
            stage_2_reasoning = parsed_target["reasoning"]
            
            if not parsed_target["success"]:
                parse_failures += 1
        
        # Store for metrics
        gt_types.append(gt_type)
        pred_types.append(pred_type)
        
        # Target metric: Only compare if GT type != None (fair comparison given 2-stage)
        # OR we can compare all, with 0 as default
        gt_targets.append(gt_target)
        pred_targets.append(pred_target)
        
        results.append({
            "original_index": int(original_idx),
            "utterance": utterance,
            "gt_type": gt_type,
            "pred_type": pred_type,
            "gt_target": gt_target,
            "pred_target": pred_target,
            "reasoning_type": parsed_type["reasoning"],
            "reasoning_target": stage_2_reasoning
        })
        
        if len(results) % 20 == 0:
            print(f"  Progress: {len(results)}/{len(ablation_df)}")
            
    # Metrics
    kappa_type = cohen_kappa_score(gt_types, pred_types)
    kappa_target = cohen_kappa_score(gt_targets, pred_targets)
    
    # Combined metric? Just average for now or report both
    
    return {
        "results": results,
        "kappa_type": kappa_type,
        "kappa_target": kappa_target,
        "parse_failure_rate": parse_failures / (len(results) * 2), # approx denominator
        "total": len(results)
    }

def main():
    parser = argparse.ArgumentParser(description="DELI Ablation Testing")
    parser.add_argument("--model", type=str, default="llama-8b", choices=list(MODELS.keys()))
    parser.add_argument("--history_window", type=int, default=None)
    parser.add_argument("--cot", type=str, default=None)
    parser.add_argument("--config", type=str, default="single", choices=["single", "all"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"DELI Ablation Test | Model: {args.model}")
    print("=" * 60)
    
    ablation_df = pd.read_csv(ABLATION_DATA_PATH)
    full_df = pd.read_csv(FULL_DATA_PATH)
    print(f"Ablation samples: {len(ablation_df)}")
    
    model, tokenizer = load_model(args.model, args.device)
    
    if args.config == "all":
        configs = [(h, c) for h in HISTORY_WINDOWS for c in COT_OPTIONS]
    else:
        history = args.history_window if args.history_window is not None else 5
        cot = args.cot == "on" if args.cot else True
        configs = [(history, cot)]
        
    all_results = {}
    
    for history_window, use_cot in configs:
        config_name = f"h{history_window}_cot{'on' if use_cot else 'off'}"
        print(f"\nConfig: history={history_window}, cot={use_cot}")
        
        result = run_ablation_config(model, tokenizer, ablation_df, full_df, history_window, use_cot)
        
        all_results[config_name] = {
            "history_window": history_window,
            "use_cot": use_cot,
            "kappa_type": result["kappa_type"],
            "kappa_target": result["kappa_target"],
            "parse_failure_rate": result["parse_failure_rate"]
        }
        
        output_file = RESULTS_DIR / f"deli_{args.model}_{config_name}.json"
        with open(output_file, "w") as f:
            json.dump(result["results"], f, indent=2)
            
        print(f"  Kappa Type: {result['kappa_type']:.3f}")
        print(f"  Kappa Target: {result['kappa_target']:.3f}")
        
    summary_path = RESULTS_DIR / f"deli_{args.model}_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "configs": all_results}, f, indent=2)
        
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
