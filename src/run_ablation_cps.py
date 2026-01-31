#!/usr/bin/env python3
"""
Run ablation tests for CPS annotation.

Tests combinations of:
- Context history: 0, 3, 5, 10 turns
- Chain-of-thought: on/off

Usage:
    python run_ablation_cps.py --model llama-3b --config all
    python run_ablation_cps.py --model llama-3b --history_window 5 --cot on
"""

import argparse
import json
import re
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

# Use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
ABLATION_DATA_PATH = SCRIPT_DIR / "data/ablation/cps_ablation_198.csv"
FULL_DATA_PATH = SCRIPT_DIR / "data/GoldenData/WTD/OOCPS_aied.csv"
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

# Ablation configurations
HISTORY_WINDOWS = [0, 3, 5, 10, -1]
COT_OPTIONS = [True, False]


def load_model(model_name: str, device: str = "auto"):
    """Load model from local path."""
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


def get_conversation_history(df: pd.DataFrame, current_idx: int, window: int = 5) -> str:
    """Get conversation history from the FULL dataset using original index."""
    if window == 0:
        return ""
    
    # If window is -1, take all history (start from 0, or start of session/group)
    # Note: CPS doesn't explicitly guarantee session sorting in simple row indexing
    # but based on previous exploration, it seems sequential.
    # To be safe for 'max', we should ideally respect session boundaries.
    
    if window == -1:
        # For simple sequential assumption:
        # We need to find the start of THIS conversation
        row = df.iloc[current_idx]
        group_id = row.get("Session_ID") or row.get("Group_x") # Try to identify group
        
        if group_id:
            # Get all previous rows in this group
            # Optimize: assuming dataframe is sorted by group/time
            start_idx = current_idx
            while start_idx > 0:
                prev_row = df.iloc[start_idx-1]
                prev_group = prev_row.get("Session_ID") or prev_row.get("Group_x")
                if prev_group != group_id:
                    break
                start_idx -= 1
        else:
            # Fallback if no group ID: look back 50 turns? Or just 0?
            start_idx = max(0, current_idx - 50) 
            
    else:
        start_idx = max(0, current_idx - window)
    
    history_rows = df.iloc[start_idx:current_idx]
    
    lines = []
    for _, row in history_rows.iterrows():
        participant = row.get("Participant", "?")
        text = row.get("Transcript", "")
        lines.append(f"[P{participant}]: {text}")
    
    return "\n".join(lines) if lines else ""


def get_ground_truth_facets(row: pd.Series) -> list:
    """Extract ground truth binary facet labels from row."""
    csk_cols = [c for c in row.index if 'CPS_CONST' in c]
    nc_cols = [c for c in row.index if 'CPS_NEG' in c]
    mtf_cols = [c for c in row.index if 'CPS_MAINTAIN' in c]
    
    csk = 1 if any(pd.notna(row[c]) and row[c] not in [0, '0', ''] for c in csk_cols) else 0
    nc = 1 if any(pd.notna(row[c]) and row[c] not in [0, '0', ''] for c in nc_cols) else 0
    mtf = 1 if any(pd.notna(row[c]) and row[c] not in [0, '0', ''] for c in mtf_cols) else 0
    
    return [csk, nc, mtf]


def format_prompt(history: str, utterance: str, participant: str, use_cot: bool) -> str:
    """Format prompt with/without CoT."""
    base = """You are annotating collaborative problem-solving dialogue.

For each utterance, classify into 3 binary facets (1=present, 0=absent):
- CSK (Constructing Shared Knowledge): sharing info, confirming understanding
- NC (Negotiation & Coordination): reasoning, questioning, strategizing
- MTF (Maintaining Team Function): suggestions, compliments, support

"""
    
    if history:
        base += f"Dialogue history:\n{history}\n\n"
    else:
        base += "Dialogue history: (No prior turns)\n\n"
    
    base += f"Current utterance [P{participant}]: {utterance}\n\n"
    
    if use_cot:
        base += """Output JSON with reasoning:
{"CPS_Label": [CSK, NC, MTF], "reasoning": "brief explanation"}"""
    else:
        base += """Output JSON only:
{"CPS_Label": [CSK, NC, MTF]}"""
    
    return base


def parse_model_output(output: str, use_cot: bool) -> dict:
    """Parse model output to extract CPS labels."""
    # Try to find JSON in output
    json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            label = parsed.get("CPS_Label", [0, 0, 0])
            if isinstance(label, list) and len(label) == 3:
                return {
                    "CPS_Label": [int(x) for x in label],
                    "reasoning": parsed.get("reasoning", "") if use_cot else "",
                    "parse_success": True
                }
        except:
            pass
    
    # Fallback: try to find array pattern
    array_match = re.search(r'\[(\d),\s*(\d),\s*(\d)\]', output)
    if array_match:
        return {
            "CPS_Label": [int(array_match.group(i)) for i in range(1, 4)],
            "reasoning": "",
            "parse_success": True
        }
    
    return {"CPS_Label": [0, 0, 0], "reasoning": "", "parse_success": False}


def run_inference(model, tokenizer, prompt, max_new_tokens=128):
    """Run inference with the model."""
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


from sklearn.metrics import cohen_kappa_score
import numpy as np

def run_ablation_config(model, tokenizer, ablation_df, full_df, history_window: int, use_cot: bool):
    """Run one ablation configuration."""
    results = []
    all_gt = []
    all_pred = []
    parse_failures = 0
    
    for _, row in ablation_df.iterrows():
        original_idx = row['original_index']
        full_row = full_df.iloc[original_idx]
        
        utterance = str(full_row.get("Transcript", ""))
        participant = str(full_row.get("Participant", ""))
        
        # Get history from FULL dataset
        history = get_conversation_history(full_df, original_idx, history_window)
        
        # Format prompt
        prompt = format_prompt(history, utterance, participant, use_cot)
        
        # Run inference
        output = run_inference(model, tokenizer, prompt)
        parsed = parse_model_output(output, use_cot)
        
        # Get ground truth
        gt_facets = get_ground_truth_facets(full_row)
        pred_facets = parsed["CPS_Label"]
        
        if not parsed["parse_success"]:
            parse_failures += 1
        
        match = pred_facets == gt_facets
        
        all_gt.append(gt_facets)
        all_pred.append(pred_facets)
        
        results.append({
            "original_index": int(original_idx),
            "utterance": utterance,
            "ground_truth": gt_facets,
            "predicted": pred_facets,
            "match": match,
            "parse_success": parsed["parse_success"],
            "reasoning": parsed["reasoning"],
        })
        
        if len(results) % 20 == 0:
            print(f"  Progress: {len(results)}/{len(ablation_df)}")
            
    # Calculate Kappa for each facet
    gt_array = np.array(all_gt)
    pred_array = np.array(all_pred)
    
    kappas = []
    for i, facet in enumerate(["CSK", "NC", "MTF"]):
        k = cohen_kappa_score(gt_array[:, i], pred_array[:, i])
        kappas.append(k)
        
    avg_kappa = np.mean(kappas)
    
    return {
        "results": results,
        "kappa": avg_kappa,
        "kappas_per_facet": {"CSK": kappas[0], "NC": kappas[1], "MTF": kappas[2]},
        "parse_failure_rate": parse_failures / len(results) if len(results) > 0 else 0,
        "total": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="CPS Ablation Testing")
    parser.add_argument("--model", type=str, default="llama-8b", choices=list(MODELS.keys()))
    parser.add_argument("--history_window", type=int, default=None, help="Specific history window (0, 3, 5, 10)")
    parser.add_argument("--cot", type=str, default=None, choices=["on", "off"], help="Specific CoT setting")
    parser.add_argument("--config", type=str, default="single", choices=["single", "all"], 
                        help="Run single config or all ablation combinations")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"CPS Ablation Test | Model: {args.model}")
    print("=" * 60)
    
    # Load data
    ablation_df = pd.read_csv(ABLATION_DATA_PATH)
    full_df = pd.read_csv(FULL_DATA_PATH)
    print(f"Ablation samples: {len(ablation_df)}")
    print(f"Full dataset: {len(full_df)}")
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Determine configurations to run
    if args.config == "all":
        configs = [(h, c) for h in HISTORY_WINDOWS for c in COT_OPTIONS]
    else:
        history = args.history_window if args.history_window is not None else 5
        cot = args.cot == "on" if args.cot else True
        configs = [(history, cot)]
    
    all_results = {}
    
    for history_window, use_cot in configs:
        config_name = f"h{history_window}_cot{'on' if use_cot else 'off'}"
        print(f"\n{'='*60}")
        print(f"Config: history={history_window}, cot={use_cot}")
        print("=" * 60)
        
        result = run_ablation_config(model, tokenizer, ablation_df, full_df, history_window, use_cot)
        
        all_results[config_name] = {
            "history_window": history_window,
            "use_cot": use_cot,
            "kappa": result["kappa"],
            "kappas_per_facet": result["kappas_per_facet"],
            "parse_failure_rate": result["parse_failure_rate"],
            "total_samples": result["total"],
        }
        
        # Save individual config results
        config_path = RESULTS_DIR / f"cps_{args.model}_{config_name}.json"
        with open(config_path, "w") as f:
            json.dump(result["results"], f, indent=2)
        
        print(f"  Kappa: {result['kappa']:.3f} (CSK:{result['kappas_per_facet']['CSK']:.2f}, NC:{result['kappas_per_facet']['NC']:.2f}, MTF:{result['kappas_per_facet']['MTF']:.2f})")
        print(f"  Parse failures: {100*result['parse_failure_rate']:.1f}%")
        print(f"  Saved to: {config_path}")
    
    # Save summary
    summary_path = RESULTS_DIR / f"cps_{args.model}_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "configs": all_results
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Ablation Summary:")
    print("=" * 60)
    for config, metrics in all_results.items():
        print(f"  {config}: Kappa={metrics['kappa']:.3f}, ParseFail={100*metrics['parse_failure_rate']:.1f}%")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
