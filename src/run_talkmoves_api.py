#!/usr/bin/env python3
"""
TalkMoves Dataset Annotation Inference - API Version

Annotates teacher and student talk moves using frontier APIs with a two-stage approach:
- Stage 1: Classify accountability category (Other, Learning Community, Content Knowledge, Rigorous Thinking)
- Stage 2: Classify specific talk move within that category

Usage:
    python run_talkmoves_api.py --model gpt-4o --data_type teacher --n_samples 100 --api_key YOUR_KEY
    python run_talkmoves_api.py --model gemini-2.0-flash --data_type student --n_samples 100 --api_key YOUR_KEY
    python run_talkmoves_api.py --model claude-sonnet-4-20250514 --data_type teacher --n_samples 100 --api_key YOUR_KEY
"""

import argparse
import json
import re
import time
import pandas as pd
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATA_DIR = Path("./data/TalkMoves/data")
DEFAULT_RESULTS_DIR = Path("./results/TalkMoves")

# Model configurations
MODELS = {
    # OpenAI
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "gpt-4-turbo": {"provider": "openai", "model_id": "gpt-4-turbo"},
    "gpt-3.5-turbo": {"provider": "openai", "model_id": "gpt-3.5-turbo"},
    # Google Gemini
    "gemini-2.0-flash": {"provider": "google", "model_id": "gemini-2.0-flash"},
    "gemini-1.5-flash": {"provider": "google", "model_id": "gemini-1.5-flash"},
    "gemini-1.5-pro": {"provider": "google", "model_id": "gemini-1.5-pro"},
    # Anthropic Claude
    "claude-sonnet-4-20250514": {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
    "claude-3-5-sonnet": {"provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022"},
    "claude-3-haiku": {"provider": "anthropic", "model_id": "claude-3-haiku-20240307"},
    "claude-3-opus": {"provider": "anthropic", "model_id": "claude-3-opus-20240229"},
}

# ============================================================================
# Label Mappings (same as local version)
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
# System Prompts (same as local version)
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
# API Clients
# ============================================================================

def call_openai(api_key: str, model_id: str, system_prompt: str, user_prompt: str, 
                temperature: float = 0.6, max_tokens: int = 128) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_google(api_key: str, model_id: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.6, max_tokens: int = 128) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    
    response = model.generate_content(user_prompt)
    return response.text


def call_anthropic(api_key: str, model_id: str, system_prompt: str, user_prompt: str,
                   temperature: float = 0.6, max_tokens: int = 128) -> str:
    """Call Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return response.content[0].text


def call_api(provider: str, api_key: str, model_id: str, system_prompt: str, 
             user_prompt: str, temperature: float = 0.6, max_tokens: int = 128) -> str:
    """Route to appropriate API based on provider."""
    if provider == "openai":
        return call_openai(api_key, model_id, system_prompt, user_prompt, temperature, max_tokens)
    elif provider == "google":
        return call_google(api_key, model_id, system_prompt, user_prompt, temperature, max_tokens)
    elif provider == "anthropic":
        return call_anthropic(api_key, model_id, system_prompt, user_prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Data Loading (same as local version)
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
# Prompt Functions (same as local version)
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TalkMoves Annotation Inference (API Version)")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--data_type", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls")
    args = parser.parse_args()
    
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return
    
    model_config = MODELS[args.model]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print(f"TalkMoves Annotation (API)")
    print(f"Model: {args.model} | Provider: {provider}")
    print(f"Type: {args.data_type} | Split: {args.split}")
    print(f"Samples: {args.n_samples} | Start: {args.start_idx}")
    print("=" * 60)
    
    df = load_data(data_dir, args.data_type, args.split)
    print(f"Dataset: {len(df)} utterances")
    
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
        
        try:
            # Stage 1: Category
            cat_prompt = format_category_prompt(text_a, text_b, args.data_type)
            cat_output = call_api(provider, args.api_key, model_id, cat_system, cat_prompt, args.temperature)
            pred_cat, cat_reasoning = parse_json_with_reasoning(cat_output, "category", 0)
            pred_cat = max(0, min(3, pred_cat)) if pred_cat is not None else 0
            
            # Stage 2: Specific Move
            pred_move = None
            move_reasoning = ""
            move_system = get_move_system_prompt(pred_cat, args.data_type)
            
            if move_system:
                move_prompt = format_move_prompt(text_a, text_b, pred_cat, args.data_type)
                if move_prompt:
                    move_output = call_api(provider, args.api_key, model_id, move_system, move_prompt, args.temperature)
                    pred_move, move_reasoning = parse_json_with_reasoning(move_output, "move", None)
        except Exception as e:
            print(f"  API Error at idx {i}: {e}")
            pred_cat = 0
            pred_move = None
            cat_reasoning = f"API Error: {str(e)}"
            move_reasoning = ""
        
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
        
        sample_num = i - args.start_idx + 1
        if sample_num % 10 == 0 or sample_num <= 5:
            print(f"  [{sample_num}/{args.n_samples}] GT: {gt_label} | Pred: {pred_label}")
        
        if sample_num % 50 == 0:
            output_path = results_dir / f"talkmoves_{args.data_type}_{args.model}.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        time.sleep(args.delay)
    
    output_path = results_dir / f"talkmoves_{args.data_type}_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
