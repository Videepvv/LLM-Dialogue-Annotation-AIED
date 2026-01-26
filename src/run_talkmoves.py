#!/usr/bin/env python3
"""
TalkMoves Dataset Annotation Inference

Annotates teacher and student talk moves using LLMs with a two-stage approach:
- Stage 1: Classify accountability category (Other, Learning Community, Content Knowledge, Rigorous Thinking)
- Stage 2: Classify specific talk move within that category

Based on Accountable Talk theory (Michaels et al., 2008).

Usage:
    python run_talkmoves.py --model llama-8b --data_type teacher --n_samples 1000
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
# Configuration - Update these paths for your setup
# ============================================================================

# Default paths (override with command line arguments)
DEFAULT_DATA_DIR = Path("./data/TalkMoves/data")
DEFAULT_RESULTS_DIR = Path("./results/TalkMoves")
DEFAULT_MODELS_DIR = Path("./models")

# Model name mapping to HuggingFace IDs
MODELS = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "gemma-9b": "google/gemma-2-9b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
}

# ============================================================================
# Label Mappings
# ============================================================================

TEACHER_LABELS = {
    0: "Other", 
    1: "Keeping Everyone Together", 
    2: "Getting Students to Relate",
    3: "Restating", 
    4: "Pressing for Accuracy", 
    5: "Revoicing", 
    6: "Pressing for Reasoning"
}

STUDENT_LABELS = {
    0: "Other", 
    1: "Relating to Another Student", 
    2: "Asking for More Information",
    3: "Making a Claim", 
    4: "Providing Evidence"
}

CATEGORY_LABELS = {
    0: "Other", 
    1: "Learning Community", 
    2: "Content Knowledge", 
    3: "Rigorous Thinking"
}

# ============================================================================
# System Prompts
# ============================================================================

TEACHER_CATEGORY_SYSTEM = """You are an expert in analyzing K-12 mathematics classroom discourse.

## Background: Teacher Talk Moves
Teacher talk moves are discussion strategies that promote students' equitable participation. They are organized into three accountability categories from Accountable Talk theory.

## Task: Classify the Accountability Category

Given a teacher utterance in a math classroom, classify which category it belongs to:

**0 = Other**: Non-instructional talk (logistics, transitions, off-topic)
- Examples: "Good morning", "Open your books", "You can work quietly"

**1 = Learning Community**: Utterances that keep students engaged and relating to each other
- Managing turns, attention, yes/no questions, asking students to agree/disagree, restating student words
- Examples: "Go ahead", "Do you agree?", "Can you repeat that?", "Seven?"

**2 = Content Knowledge**: Utterances requesting mathematical answers or procedures
- Asking for answers, definitions, problem-solving steps
- Examples: "What is 6 times 6?", "How did you solve it?", "What does x stand for?"

**3 = Rigorous Thinking**: Utterances pushing for deeper reasoning
- Revoicing (repeating + adding), asking "why", requesting justification
- Examples: "So you're saying...", "Why do you think that?", "Can you explain your reasoning?"

Output ONLY: {"category": <integer>, "reasoning": "brief explanation"}"""


TEACHER_MOVE_LC_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the Specific Talk Move (Learning Community)

The teacher utterance has been classified as "Learning Community" (focused on engagement and participation). Now classify the specific talk move:

**1 = Keeping Everyone Together**: Managing turns, attention, yes/no questions, call-and-response
- "Go ahead", "Shelly?", "Seven?", "Can you repeat that?", "Are you finished?"

**2 = Getting Students to Relate**: Asking students to comment on or agree/disagree with peer ideas
- "Do you agree?", "What do you think about what she said?", "Did anyone else get that?"

**3 = Restating**: Repeating student words EXACTLY, without any additions or changes
- S: "An exponent" → T: "Exponent" (exact repeat)
- S: "Four million two" → T: "Four million two" (exact repeat)

Output ONLY: {"move": <integer>, "reasoning": "brief explanation"}"""


TEACHER_MOVE_RT_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the Specific Talk Move (Rigorous Thinking)

The teacher utterance has been classified as "Rigorous Thinking" (pushing for deeper understanding). Now classify the specific talk move:

**5 = Revoicing**: Repeating what a student said AND adding to or changing the wording
- S: "It had two" → T: "So instead of one flat edge, it had two" (adds context)
- S: "An exponent" → T: "THE exponent" (changes article)
- S: "Company B" → T: "It's Company B because that charges $2 per minute" (adds reasoning)

**6 = Pressing for Reasoning**: Asking for justification, "why" questions, explanations
- "Why do you think that?", "Can you explain?", "How do you know?", "What's your reasoning?"

Output ONLY: {"move": <integer>, "reasoning": "brief explanation"}"""


STUDENT_CATEGORY_SYSTEM = """You are an expert in analyzing K-12 mathematics classroom discourse.

## Background: Student Talk Moves
Student talk moves reflect how students engage in mathematical discussions. They are organized into accountability categories.

## Task: Classify the Accountability Category

Given a student utterance in a math classroom, classify which category it belongs to:

**0 = Other**: Non-instructional, off-topic, simple yes/no, logistics
- Examples: "Yes", "No", "Okay", "I'm done", greetings

**1 = Learning Community**: Relating to peers or asking for help
- Responding to another student's idea, asking for clarification
- Examples: "I agree with her", "I don't understand", "Can you explain?"

**2 = Content Knowledge**: Making mathematical claims or stating answers
- Providing an answer, describing simple procedures, disagreeing with teacher
- Examples: "12", "I multiplied 8 and 5", "Y is the number of cars"

**3 = Rigorous Thinking**: Providing evidence or reasoning
- Multi-step explanations, "because" statements, noticing patterns, if/then logic
- Examples: "I divided 6 by 2 because...", "It's going up by 5 each time", "If you double it..."

Output ONLY: {"category": <integer>, "reasoning": "brief explanation"}"""


STUDENT_MOVE_LC_SYSTEM = """You are an expert in K-12 mathematics classroom discourse.

## Task: Classify the Specific Talk Move (Learning Community)

The student utterance has been classified as "Learning Community". Now classify:

**1 = Relating to Another Student**: Responding to or building on a peer's idea
- "I agree with what she said", "That's what I got too", "I did it differently than him"

**2 = Asking for More Information**: Requesting clarification or expressing confusion
- "I don't understand", "Can you explain?", "What do you mean?", "I'm confused"

Output ONLY: {"move": <integer>, "reasoning": "brief explanation"}"""


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str, models_dir: Path, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from local path or HuggingFace."""
    
    # Check for local model first
    local_path = models_dir / model_name
    if local_path.exists():
        model_path = local_path
    else:
        # Use HuggingFace ID
        model_path = MODELS.get(model_name, model_name)
    
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

def load_data(data_dir: Path, data_type: str, split: str) -> pd.DataFrame:
    """Load TalkMoves data from TSV file."""
    filename = f"{split}_{data_type}.tsv"
    filepath = data_dir / filename
    print(f"Loading data from: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t')
    df = df.fillna('')
    df['labels'] = df['labels'].astype(float).astype(int)
    return df


# ============================================================================
# Inference Functions
# ============================================================================

def format_category_prompt(text_a: str, text_b: str, data_type: str) -> str:
    """Format prompt for Stage 1 category classification."""
    context = f"Previous context: {text_a}" if text_a else "(Start of conversation)"
    speaker = "Teacher" if data_type == "teacher" else "Student"
    
    return f"""{context}

Current {speaker.lower()} utterance to classify:
"{text_b}"

Classify the accountability CATEGORY of this {speaker.lower()} utterance.
0 = Other (non-instructional)
1 = Learning Community (engagement, relating to peers)
2 = Content Knowledge (math answers, claims)
3 = Rigorous Thinking (reasoning, evidence)

Output ONLY: {{"category": <integer>, "reasoning": "brief explanation"}}"""


def format_move_prompt(text_a: str, text_b: str, category: int, data_type: str) -> Optional[str]:
    """Format prompt for Stage 2 specific move classification."""
    if category == 0:
        return None  # No second stage for "Other"
    
    context = f"Previous context: {text_a}" if text_a else "(Start of conversation)"
    speaker = "Teacher" if data_type == "teacher" else "Student"
    cat_name = CATEGORY_LABELS[category]
    
    if data_type == "teacher":
        if category == 1:  # Learning Community
            options = "1=Keeping Everyone Together, 2=Getting Students to Relate, 3=Restating"
        elif category == 2:  # Content Knowledge
            return None  # Only one option: Pressing for Accuracy (4)
        else:  # Rigorous Thinking
            options = "5=Revoicing, 6=Pressing for Reasoning"
    else:  # student
        if category == 1:  # Learning Community
            options = "1=Relating to Another Student, 2=Asking for More Info"
        elif category == 2:  # Content Knowledge
            return None  # Only one option: Making a Claim (3)
        else:  # Rigorous Thinking
            return None  # Only one option: Providing Evidence (4)
    
    return f"""{context}

Current {speaker.lower()} utterance (Category: {cat_name}):
"{text_b}"

Classify the specific talk move.
Options: {options}

Output ONLY: {{"move": <integer>, "reasoning": "brief explanation"}}"""


def get_move_system_prompt(category: int, data_type: str) -> Optional[str]:
    """Get the appropriate system prompt for Stage 2."""
    if data_type == "teacher":
        if category == 1:
            return TEACHER_MOVE_LC_SYSTEM
        elif category == 3:
            return TEACHER_MOVE_RT_SYSTEM
    else:
        if category == 1:
            return STUDENT_MOVE_LC_SYSTEM
    return None


def parse_json_with_reasoning(output: str, field: str, default) -> tuple:
    """Parse integer field and reasoning from JSON output."""
    reasoning = ""
    json_match = re.search(r'\{[^{}]*\}', output)
    if json_match:
        try:
            result = json.loads(json_match.group())
            reasoning = result.get("reasoning", "")
            if field in result:
                val = result[field]
                return (int(val) if val is not None else default), reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    pattern = rf'"{field}"\s*:\s*(-?\d+)'
    match = re.search(pattern, output)
    if match:
        return int(match.group(1)), reasoning
    
    return default, reasoning


def category_to_final_label(category: int, move: Optional[int], data_type: str) -> int:
    """Map category + move back to final label."""
    if category == 0:
        return 0  # Other
    
    if data_type == "teacher":
        if category == 1:  # Learning Community
            return move if move in [1, 2, 3] else 1
        elif category == 2:  # Content Knowledge
            return 4  # Pressing for Accuracy
        elif category == 3:  # Rigorous Thinking
            return move if move in [5, 6] else 5
    else:  # student
        if category == 1:  # Learning Community
            return move if move in [1, 2] else 1
        elif category == 2:  # Content Knowledge
            return 3  # Making a Claim
        elif category == 3:  # Rigorous Thinking
            return 4  # Providing Evidence
    
    return 0  # Fallback


def run_inference(model, tokenizer, system_prompt: str, user_prompt: str,
                  temperature: float = 0.6, max_new_tokens: int = 128) -> str:
    """Run model inference."""
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
    parser = argparse.ArgumentParser(description="TalkMoves Annotation Inference")
    parser.add_argument("--model", type=str, default="llama-8b", help="Model name")
    parser.add_argument("--data_type", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print(f"TalkMoves Annotation")
    print(f"Model: {args.model} | Type: {args.data_type} | Split: {args.split}")
    print(f"Samples: {args.n_samples} | Start: {args.start_idx}")
    print("=" * 60)
    
    # Load data and model
    df = load_data(data_dir, args.data_type, args.split)
    print(f"Dataset: {len(df)} utterances")
    
    model, tokenizer = load_model(args.model, models_dir, args.device)
    
    # Select category system prompt
    cat_system = TEACHER_CATEGORY_SYSTEM if args.data_type == "teacher" else STUDENT_CATEGORY_SYSTEM
    
    print("Running inference...")
    
    results = []
    correct = 0
    total = 0
    
    end_idx = min(args.start_idx + args.n_samples, len(df))
    
    for i in range(args.start_idx, end_idx):
        row = df.iloc[i]
        text_a = str(row['text_a'])
        text_b = str(row['text_b'])
        gt_label = int(row['labels'])
        
        # Stage 1: Category
        cat_prompt = format_category_prompt(text_a, text_b, args.data_type)
        cat_output = run_inference(model, tokenizer, cat_system, cat_prompt, args.temperature)
        pred_cat, cat_reasoning = parse_json_with_reasoning(cat_output, "category", 0)
        pred_cat = max(0, min(3, pred_cat)) if pred_cat is not None else 0
        
        # Stage 2: Specific Move
        pred_move = None
        move_reasoning = ""
        move_system = get_move_system_prompt(pred_cat, args.data_type)
        
        if move_system:
            move_prompt = format_move_prompt(text_a, text_b, pred_cat, args.data_type)
            if move_prompt:
                move_output = run_inference(model, tokenizer, move_system, move_prompt, args.temperature)
                pred_move, move_reasoning = parse_json_with_reasoning(move_output, "move", None)
        
        # Map to final label
        pred_label = category_to_final_label(pred_cat, pred_move, args.data_type)
        
        is_correct = pred_label == gt_label
        if is_correct:
            correct += 1
        total += 1
        
        result = {
            "idx": i,
            "text_a": text_a[:100] + "..." if len(text_a) > 100 else text_a,
            "text_b": text_b,
            "gt_label": gt_label,
            "pred_category": pred_cat,
            "pred_move": pred_move,
            "pred_label": pred_label,
            "category_reasoning": cat_reasoning,
            "move_reasoning": move_reasoning,
            "correct": is_correct,
        }
        results.append(result)
        
        # Progress
        sample_num = i - args.start_idx + 1
        if sample_num % 10 == 0 or sample_num <= 5:
            labels = TEACHER_LABELS if args.data_type == "teacher" else STUDENT_LABELS
            print(f"  [{sample_num}/{args.n_samples}] GT: {gt_label} | Pred: {pred_label}")
        
        # Checkpoint
        if sample_num % 50 == 0:
            output_path = results_dir / f"talkmoves_{args.data_type}_{args.model}.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
    
    # Final save
    output_path = results_dir / f"talkmoves_{args.data_type}_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
