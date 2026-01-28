#!/usr/bin/env python3
"""
Run ablation tests for TalkMoves annotation.

Tests combinations of:
- Context history: 0 (current only), 1 (prev + current)
- Chain-of-thought: on/off

Two-stage classification:
1. Category
2. Specific Move (within category)

Usage:
    python run_ablation_talkmoves.py --model llama-3b --cot on --data_type teacher
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

DATA_ROOT = Path("/s/babbage/h/nobackup/nblancha/public-datasets/ilideep/LLM-Dialogue-Annotation-AIED/data/ablation")
RESULTS_DIR = Path("/s/babbage/h/nobackup/nblancha/public-datasets/ilideep/LLM-Dialogue-Annotation-AIED/results/ablation")
MODELS_DIR = Path("/s/babbage/h/nobackup/nblancha/public-datasets/openWeightLLMs")

MODELS = {
    "llama-3b": "meta-llama_Llama-3.2-3B-Instruct-HF",
    "llama-1b": "meta-llama_Llama-3.2-1B-Instruct-HF",
    "llama-8b": "meta-llama_Llama-3.1-8B-Instruct-HF",
    "qwen-3b": "Qwen_Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen_Qwen2.5-7B-Instruct",
    "gemma-2b": "google_gemma-2-2b-it",
    "phi-3.5": "microsoft_Phi-3.5-mini-instruct",
}

HISTORY_WINDOWS = [0, 1]
COT_OPTIONS = [True, False]

# Prompts
TEACHER_CATEGORY_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the General Category of the Teacher's Utterance

Given a teacher utterance in a math classroom, classify which general category it belongs to:

**0 = Other**: Non-instructional, classroom management, or simple affirmations.
- Examples: "Okay", "Right", "Sit down", "Take out your books"

**1 = Learning Community**: Fostering a supportive learning environment or managing participation.
- Managing turns, attention, yes/no questions, asking students to agree/disagree, restating student words
- Examples: "Go ahead", "Do you agree?", "Can you repeat that?", "Seven?"

**2 = Content Knowledge**: Utterances requesting mathematical answers or procedures
- Asking for answers, definitions, problem-solving steps
- Examples: "What is 6 times 6?", "How did you solve it?", "What does x stand for?"

**3 = Rigorous Thinking**: Utterances that push students to think deeply, explain reasoning, or connect ideas.
- Pressing for reasoning, asking "Why?", connecting representations
- Examples: "Why does that work?", "How are these two methods similar?", "Can you prove it?"

"""

TEACHER_MOVE_LC_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the Specific Talk Move (Learning Community)

The teacher's utterance falls under **Learning Community**. Further classify it:

**0 = Other**: General management within this category.

**1 = Keeping Everyone Together**: Ensuring students are following along or orienting them to others' ideas.
- "Did everyone hear that?", "Who agrees with Sarah?", "Can you say that louder?"

**2 = Getting Students to Relate**: Asking students to relate their work to others.
- "How is your idea like his?", "Do you have a different way?"

**3 = Restating**: Repeating student words EXACTLY, without any additions or changes
- S: "An exponent" → T: "Exponent" (exact repeat)
- S: "Four million two" → T: "Four million two" (exact repeat)

**5 = Revoicing**: Repeating student words but adding clarity, technical terms, or expanding.
- S: "It goes up" → T: "So the line is increasing?" (revoicing with math term)

"""

TEACHER_MOVE_CK_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the Specific Talk Move (Content Knowledge)

The teacher's utterance falls under **Content Knowledge**. Further classify it:

**4 = Pressing for Accuracy**: Asking for specific answers, facts, or procedural steps.
- "What is 5 + 5?", "What is the next step?", "Is that a triangle?"

"""

TEACHER_MOVE_RT_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the Specific Talk Move (Rigorous Thinking)

The teacher's utterance falls under **Rigorous Thinking**. Further classify it:

**6 = Pressing for Reasoning**: Asking for justifications, explanations, or underlying logic.
- "Why did you do that?", "How do you know?", "Can you explain your thinking?"

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

def format_category_prompt(text_a: str, text_b: str, history_window: int, use_cot: bool) -> str:
    base = TEACHER_CATEGORY_SYSTEM 
    
    if history_window > 0 and pd.notna(text_a):
        base += f"Previous Utterance: {text_a}\n"
        
    base += f"Teacher Utterance: {text_b}\n\n"
    
    if use_cot:
        base += "Output ONLY JSON:\n{\"category\": <integer>, \"reasoning\": \"brief explanation\"}"
    else:
        base += "Output ONLY JSON:\n{\"category\": <integer>}"
        
    return base

def format_move_prompt(text_a: str, text_b: str, category: int, history_window: int, use_cot: bool) -> str:
    if category == 1:
        base = TEACHER_MOVE_LC_SYSTEM
    elif category == 2:
        base = TEACHER_MOVE_CK_SYSTEM
    elif category == 3:
        base = TEACHER_MOVE_RT_SYSTEM
    else:
        return None

    if history_window > 0 and pd.notna(text_a):
        base += f"Previous Utterance: {text_a}\n"
        
    base += f"Teacher Utterance: {text_b}\n\n"
    
    if use_cot:
        base += "Output ONLY JSON:\n{\"move\": <integer>, \"reasoning\": \"brief explanation\"}"
    else:
        base += "Output ONLY JSON:\n{\"move\": <integer>}"
        
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
            
    int_match = re.search(fr'"{field}"\s*:\s*(-?\d+)', output)
    if int_match:
        return {"val": int(int_match.group(1)), "reasoning": "", "success": True}
        
    return {"val": default, "reasoning": "", "success": False}

def run_ablation_config(model, tokenizer, ablation_df, history_window: int, use_cot: bool):
    results = []
    gt_labels = []
    pred_labels = []
    parse_failures = 0
    
    for _, row in ablation_df.iterrows():
        original_idx = row['original_index']
        text_a = row['text_a']
        text_b = row['text_b']
        gt_label = int(row['labels'])
        
        # Stage 1: Category
        cat_prompt = format_category_prompt(text_a, text_b, history_window, use_cot)
        cat_out = run_inference(model, tokenizer, cat_prompt)
        parsed_cat = parse_json(cat_out, "category", 0, use_cot)
        pred_cat = max(0, min(3, parsed_cat["val"]))
        
        if not parsed_cat["success"]:
            parse_failures += 1
            
        pred_move = 0
        
        # Stage 2: Move (if logic applies)
        # Note: In teacher data, Cat 0 -> Move 0
        if pred_cat in [1, 2, 3]:
            move_prompt = format_move_prompt(text_a, text_b, pred_cat, history_window, use_cot)
            if move_prompt:
                move_out = run_inference(model, tokenizer, move_prompt)
                parsed_move = parse_json(move_out, "move", 0, use_cot)
                pred_move = parsed_move["val"]
                if not parsed_move["success"]:
                    parse_failures += 1
        
        # Determine final label (simplified mapping for ablation)
        # If predicted Move is valid for the Category, use it. Else use encoded logic or 0.
        final_pred = pred_move if pred_move != 0 else 0
        if pred_cat == 2 and pred_move == 0: final_pred = 4 # Press for Accuracy defaults
        
        gt_labels.append(gt_label)
        pred_labels.append(final_pred)
        
        results.append({
            "original_index": int(original_idx),
            "text": text_b,
            "gt_label": gt_label,
            "pred_label": final_pred,
            "pred_cat": pred_cat,
            "pred_move": pred_move,
            "reasoning": parsed_cat["reasoning"]
        })
        
        if len(results) % 20 == 0:
            print(f"  Progress: {len(results)}/{len(ablation_df)}")
            
    kappa = cohen_kappa_score(gt_labels, pred_labels)
    
    return {
        "results": results,
        "kappa": kappa,
        "parse_failure_rate": parse_failures / (len(results) * 1.5), # Approx
        "total": len(results)
    }

def main():
    parser = argparse.ArgumentParser(description="TalkMoves Ablation Testing")
    parser.add_argument("--model", type=str, default="llama-3b", choices=list(MODELS.keys()))
    parser.add_argument("--history_window", type=int, default=None)
    parser.add_argument("--cot", type=str, default=None)
    parser.add_argument("--config", type=str, default="single", choices=["single", "all"])
    parser.add_argument("--data_type", type=str, default="teacher", choices=["teacher"]) # Student logic TODO
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"TalkMoves ({args.data_type}) Ablation Test | Model: {args.model}")
    print("=" * 60)
    
    ablation_file = DATA_ROOT / f"talkmoves_{args.data_type}_ablation_200.csv"
    ablation_df = pd.read_csv(ablation_file)
    print(f"Ablation samples: {len(ablation_df)}")
    
    model, tokenizer = load_model(args.model, args.device)
    
    if args.config == "all":
        configs = [(h, c) for h in HISTORY_WINDOWS for c in COT_OPTIONS]
    else:
        history = args.history_window if args.history_window is not None else 1
        cot = args.cot == "on" if args.cot else True
        configs = [(history, cot)]
        
    all_results = {}
    
    for history_window, use_cot in configs:
        config_name = f"h{history_window}_cot{'on' if use_cot else 'off'}"
        print(f"\nConfig: history={history_window}, cot={use_cot}")
        
        result = run_ablation_config(model, tokenizer, ablation_df, history_window, use_cot)
        
        all_results[config_name] = {
            "history_window": history_window,
            "use_cot": use_cot,
            "kappa": result["kappa"],
            "parse_failure_rate": result["parse_failure_rate"]
        }
        
        output_file = RESULTS_DIR / f"talkmoves_{args.data_type}_{args.model}_{config_name}.json"
        with open(output_file, "w") as f:
            json.dump(result["results"], f, indent=2)
            
        print(f"  Kappa: {result['kappa']:.3f}")
        
    summary_path = RESULTS_DIR / f"talkmoves_{args.data_type}_{args.model}_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "configs": all_results}, f, indent=2)
        
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
