#!/usr/bin/env python3
"""
CPS (Collaborative Problem Solving) Weights Task Annotation

Annotates dialogue with 3 binary CPS facets:
- CSK: Constructing Shared Knowledge
- NC: Negotiation & Coordination  
- MTF: Maintaining Team Function

Usage:
    python run_cps.py --model llama-8b --n_samples 1000
"""

import argparse
import json
import re
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATA_PATH = Path("./data/WTD/OOCPS_aied.csv")
DEFAULT_GUIDELINES_PATH = Path("./data/WTD/annotationGuideLine_v3.txt")
DEFAULT_RESULTS_DIR = Path("./results/CPS")
DEFAULT_MODELS_DIR = Path("./models")

MODELS = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-9b": "google/gemma-2-9b-it",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
}

# CPS columns for ground truth
CPS_COLUMNS = [
    # CSK
    "CPS_CONST_SharesU_Situation", "CPS_CONST_SharesU_CorrectSolutions",
    "CPS_CONST_SharesU_IncorrectSolutions", "CPS_CONST_EstablishesCG_Confirms",
    "CPS_CONST_EstablishesCG_Interrupts",
    # NC
    "CPS_NEG_Responds_Reasons", "CPS_NEG_Responds_QuestionsOthers",
    "CPS_NEG_Responds_Responds", "CPS_NEG_MonitorsE_Results",
    "CPS_NEG_MonitorsE_Strategizes", "CPS_NEG_MonitorsE_Save",
    "CPS_NEG_MonitorsE_GivingUp",
    # MTF
    "CPS_MAINTAIN_Initiative_Suggestions", "CPS_MAINTAIN_Initiative_Compliments",
    "CPS_MAINTAIN_Initiative_Criticizes", "CPS_MAINTAIN_FulfillsR_Support",
    "CPS_MAINTAIN_FulfillsR_Apologizes", "CPS_MAINTAIN_FulfillsR_InitiatesOffTopic",
    "CPS_MAINTAIN_FulfillsR_JoinsOffTopic",
]


def load_model(model_name: str, models_dir: Path, device: str = "auto"):
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


def load_guidelines(path: Path) -> str:
    if not path.exists():
        return get_default_guidelines()
    with open(path, "r") as f:
        return f.read().strip()


def get_default_guidelines() -> str:
    return """## CPS Annotation Guidelines

### CSK (Constructing Shared Knowledge)
Utterances where the speaker shares understanding about the task situation, proposes solutions, or establishes common ground.
- Examples: "I think we should move this weight here", "The lever is balanced now"

### NC (Negotiation & Coordination)
Utterances involving reasoning, questioning others, responding to requests, or monitoring progress.
- Examples: "Why do you think that?", "Let me try a different approach", "Did that work?"

### MTF (Maintaining Team Function)
Utterances that support team dynamics—suggestions, compliments, criticism, or off-topic chat.
- Examples: "Good idea!", "That won't work", "Sorry about that", "lol"

For each utterance, mark 1 if present, 0 if absent for each facet.
Output: {"CPS_Label": [CSK, NC, MTF], "reasoning": "brief explanation"}"""


def get_conversation_history(df: pd.DataFrame, current_idx: int, window: int = 5) -> str:
    # -1 means use all available history
    start_idx = 0 if window == -1 else max(0, current_idx - window)
    history = []
    for i in range(start_idx, current_idx):
        row = df.iloc[i]
        participant = row.get("Participant", "?")
        utterance = row.get("Transcript", "")
        history.append(f"[Participant {participant}]: {utterance}")
    return "\n".join(history)


def get_ground_truth_facets(row: pd.Series) -> list:
    csk, nc, mtf = 0, 0, 0
    for col in CPS_COLUMNS:
        if col in row and row[col] == 1:
            if "CPS_CONST_" in col:
                csk = 1
            elif "CPS_NEG_" in col:
                nc = 1
            elif "CPS_MAINTAIN_" in col:
                mtf = 1
    return [csk, nc, mtf]


def format_prompt(guidelines: str, history: str, utterance: str, participant: str) -> str:
    return f"""{guidelines}

---

Dialogue history:
{history if history else "(No prior turns)"}

Current Utterance: [Participant {participant}]: {utterance}

Determine the CPS categories. Output [0, 0, 0] if the utterance is neutral/off-topic.

Output: {{"CPS_Label": [CSK, NC, MTF], "reasoning": "brief explanation"}}"""


def parse_model_output(output: str) -> dict:
    json_match = re.search(r'\{[^{}]*"CPS_Label"[^{}]*\}', output, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if "CPS_Label" in result and isinstance(result["CPS_Label"], list):
                return result
        except json.JSONDecodeError:
            pass
    
    array_match = re.search(r'\[([01]),\s*([01]),\s*([01])\]', output)
    if array_match:
        return {
            "CPS_Label": [int(array_match.group(1)), int(array_match.group(2)), int(array_match.group(3))],
            "reasoning": "Parsed from array"
        }
    
    return {"CPS_Label": [0, 0, 0], "reasoning": "Failed to parse"}


def run_inference(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="CPS Annotation Inference")
    parser.add_argument("--model", type=str, default="llama-8b")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--history_window", type=int, default=5)
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--guidelines_path", type=str, default=str(DEFAULT_GUIDELINES_PATH))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print(f"CPS Annotation | Model: {args.model} | Samples: {args.n_samples}")
    print("=" * 60)
    
    guidelines = load_guidelines(Path(args.guidelines_path))
    df = pd.read_csv(args.data_path)
    print(f"Dataset: {len(df)} utterances")
    
    model, tokenizer = load_model(args.model, Path(args.models_dir), args.device)
    
    results = []
    correct = 0
    total = 0
    
    end_idx = min(args.start_idx + args.n_samples, len(df))
    
    for i in range(args.start_idx, end_idx):
        row = df.iloc[i]
        utterance = row.get("Transcript", "")
        participant = int(row.get("Participant", 0))
        
        history = get_conversation_history(df, i, args.history_window)
        prompt = format_prompt(guidelines, history, utterance, participant)
        
        output = run_inference(model, tokenizer, prompt)
        parsed = parse_model_output(output)
        
        gt_facets = get_ground_truth_facets(row)
        pred_facets = parsed.get("CPS_Label", [0, 0, 0])
        
        match = pred_facets == gt_facets if len(pred_facets) == 3 else False
        if match:
            correct += 1
        total += 1
        
        result = {
            "idx": int(i),
            "utterance": str(utterance),
            "ground_truth": gt_facets,
            "predicted": pred_facets,
            "reasoning": str(parsed.get("reasoning", "")),
            "match": match,
        }
        results.append(result)
        
        sample_num = i - args.start_idx + 1
        if sample_num % 10 == 0 or sample_num <= 5:
            print(f"  [{sample_num}/{args.n_samples}] GT: {gt_facets} | Pred: {pred_facets}")
        
        if sample_num % 50 == 0:
            with open(results_dir / f"cps_{args.model}.json", "w") as f:
                json.dump(results, f, indent=2)
    
    output_path = results_dir / f"cps_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Exact match: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
