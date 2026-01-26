#!/usr/bin/env python3
"""
CPS (Collaborative Problem Solving) Weights Task Annotation - API Version

Annotates dialogue with 3 binary CPS facets using frontier model APIs:
- GPT (OpenAI)
- Gemini (Google)
- Claude (Anthropic)

Usage:
    python run_cps_api.py --model gpt-4o --n_samples 100 --api_key YOUR_KEY
    python run_cps_api.py --model gemini-2.0-flash --n_samples 100 --api_key YOUR_KEY
    python run_cps_api.py --model claude-sonnet-4-20250514 --n_samples 100 --api_key YOUR_KEY
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

DEFAULT_DATA_PATH = Path("./data/WTD/OOCPS_aied.csv")
DEFAULT_GUIDELINES_PATH = Path("./data/WTD/annotationGuideLine_v3.txt")
DEFAULT_RESULTS_DIR = Path("./results/CPS")

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
    "gemini-3-flash-preview": {"provider": "google", "model_id": "gemini-3-flash-preview"},
    # Anthropic Claude
    "claude-sonnet-4-20250514": {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
    "claude-3-5-sonnet": {"provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022"},
    "claude-3-haiku": {"provider": "anthropic", "model_id": "claude-3-haiku-20240307"},
    "claude-3-opus": {"provider": "anthropic", "model_id": "claude-3-opus-20240229"},
    "claude-haiku-4.5": {"provider": "anthropic", "model_id": "claude-3-5-haiku-20241022"},
    "claude-haiku-4-5": {"provider": "anthropic", "model_id": "claude-3-5-haiku-20241022"},
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


# ============================================================================
# API Clients
# ============================================================================

def call_openai(api_key: str, model_id: str, system_prompt: str, user_prompt: str, 
                temperature: float = 0.6, max_tokens: int = 512) -> str:
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
                temperature: float = 0.6, max_tokens: int = 512) -> str:
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
                   temperature: float = 0.6, max_tokens: int = 512) -> str:
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
             user_prompt: str, temperature: float = 0.6, max_tokens: int = 512) -> str:
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
# Data & Prompt Functions (same as local version)
# ============================================================================

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
    start_idx = max(0, current_idx - window)
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


def format_prompt(guidelines: str, history: str, utterance: str, participant: str) -> tuple:
    """Return (system_prompt, user_prompt) tuple."""
    system_prompt = guidelines
    
    user_prompt = f"""Dialogue history:
{history if history else "(No prior turns)"}

Current Utterance: [Participant {participant}]: {utterance}

Determine the CPS categories. For each facet, use 1 if present, 0 if absent.
Output [0, 0, 0] if the utterance is neutral/off-topic.

IMPORTANT: Use only integers 0 or 1 in the array, not text labels.

Output format: {{"CPS_Label": [0 or 1, 0 or 1, 0 or 1], "reasoning": "brief explanation"}}

Example outputs:
- {{"CPS_Label": [1, 0, 0], "reasoning": "Shares task understanding"}}
- {{"CPS_Label": [0, 1, 1], "reasoning": "Coordinates and supports team"}}
- {{"CPS_Label": [0, 0, 0], "reasoning": "Off-topic or neutral"}}"""
    
    return system_prompt, user_prompt


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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CPS Annotation Inference (API Version)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o, gemini-1.5-flash, claude-3-5-sonnet)")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the model provider")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--history_window", type=int, default=5)
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--guidelines_path", type=str, default=str(DEFAULT_GUIDELINES_PATH))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds")
    args = parser.parse_args()
    
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return
    
    model_config = MODELS[args.model]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print(f"CPS Annotation (API) | Model: {args.model} | Provider: {provider}")
    print(f"Samples: {args.n_samples}")
    print("=" * 60)
    
    guidelines = load_guidelines(Path(args.guidelines_path))
    df = pd.read_csv(args.data_path)
    print(f"Dataset: {len(df)} utterances")
    
    results = []
    correct = 0
    total = 0
    
    end_idx = min(args.start_idx + args.n_samples, len(df))
    
    for i in range(args.start_idx, end_idx):
        row = df.iloc[i]
        utterance = row.get("Transcript", "")
        participant = int(row.get("Participant", 0))
        
        history = get_conversation_history(df, i, args.history_window)
        system_prompt, user_prompt = format_prompt(guidelines, history, utterance, participant)
        
        try:
            output = call_api(provider, args.api_key, model_id, system_prompt, user_prompt, args.temperature)
            parsed = parse_model_output(output)
            raw_output = output
        except Exception as e:
            print(f"  API Error at idx {i}: {e}")
            parsed = {"CPS_Label": [0, 0, 0], "reasoning": f"API Error: {str(e)}"}
            raw_output = f"API Error: {str(e)}"
        
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
            "raw_output": raw_output,
            "match": match,
        }
        results.append(result)
        
        sample_num = i - args.start_idx + 1
        if sample_num % 10 == 0 or sample_num <= 5:
            print(f"  [{sample_num}/{args.n_samples}] GT: {gt_facets} | Pred: {pred_facets}")
        
        if sample_num % 50 == 0:
            with open(results_dir / f"cps_{args.model}.json", "w") as f:
                json.dump(results, f, indent=2)
        
        time.sleep(args.delay)
    
    output_path = results_dir / f"cps_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Exact match: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
