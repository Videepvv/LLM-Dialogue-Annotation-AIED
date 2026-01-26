#!/usr/bin/env python3
"""
Calculate Cohen's Kappa and per-label metrics for all datasets.

Usage:
    python calculate_kappa.py --dataset talkmoves
    python calculate_kappa.py --dataset deli
    python calculate_kappa.py --dataset cps
    python calculate_kappa.py --all
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("./results")

# Label maps
TALKMOVES_TEACHER = {
    0: "Other", 1: "Keep Together", 2: "Students Relate", 3: "Revoicing",
    4: "Press Accuracy", 5: "Press Reasoning", 6: "Challenge"
}

TALKMOVES_STUDENT = {
    0: "Other", 1: "Relate", 2: "Ask for Info", 3: "Provide Info", 4: "Explain"
}

DELI_TYPE = {-1: "None", 0: "Probing", 1: "NPD"}
DELI_TARGET = {0: "None", 1: "Solution", 2: "Reasoning", 3: "Moderation", 4: "Agree", 5: "Disagree"}


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def compute_per_label_kappa(gt_labels, pred_labels, label_id) -> float:
    """Compute one-vs-rest kappa for a specific label."""
    gt_binary = [1 if x == label_id else 0 for x in gt_labels]
    pred_binary = [1 if x == label_id else 0 for x in pred_labels]
    
    if len(set(gt_binary)) == 1 or len(set(pred_binary)) == 1:
        return 0.0
    
    return cohen_kappa_score(gt_binary, pred_binary)


def compute_metrics(gt_labels, pred_labels, label_map) -> pd.DataFrame:
    """Compute per-label metrics."""
    results = []
    
    for label_id, label_name in sorted(label_map.items()):
        gt_count = sum(1 for x in gt_labels if x == label_id)
        pred_count = sum(1 for x in pred_labels if x == label_id)
        tp = sum(1 for i in range(len(gt_labels)) if gt_labels[i] == label_id and pred_labels[i] == label_id)
        
        precision = tp / pred_count if pred_count > 0 else 0
        recall = tp / gt_count if gt_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        kappa = compute_per_label_kappa(gt_labels, pred_labels, label_id)
        
        results.append({
            'Label': label_name,
            'Support': gt_count,
            'TP': tp,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'Kappa': round(kappa, 4)
        })
    
    return pd.DataFrame(results)


def analyze_talkmoves(results_dir: Path):
    """Analyze TalkMoves results."""
    print("\n" + "=" * 60)
    print("TALKMOVES ANALYSIS")
    print("=" * 60)
    
    talkmoves_dir = results_dir / "TalkMoves"
    if not talkmoves_dir.exists():
        print("No TalkMoves results found")
        return
    
    all_results = []
    
    for task_type, label_map in [("teacher", TALKMOVES_TEACHER), ("student", TALKMOVES_STUDENT)]:
        for filepath in sorted(talkmoves_dir.glob(f"talkmoves_{task_type}_*.json")):
            model = filepath.stem.replace(f"talkmoves_{task_type}_", "")
            data = load_json(filepath)
            
            if len(data) < 100:
                continue
            
            gt = [r["gt_label"] for r in data]
            pred = [r["pred_label"] for r in data]
            
            overall_kappa = cohen_kappa_score(gt, pred)
            accuracy = sum(1 for r in data if r["correct"]) / len(data)
            
            print(f"\n{task_type.upper()} - {model}")
            print(f"  N={len(data):,} | Kappa={overall_kappa:.4f} | Accuracy={accuracy:.2%}")
            
            df = compute_metrics(gt, pred, label_map)
            df['Model'] = model
            df['Task'] = task_type
            all_results.append(df)
            
            print(df[['Label', 'Support', 'Precision', 'Recall', 'F1', 'Kappa']].to_string(index=False))
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = results_dir / "talkmoves_metrics.csv"
        combined.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


def analyze_deli(results_dir: Path):
    """Analyze DELI results."""
    print("\n" + "=" * 60)
    print("DELI ANALYSIS")
    print("=" * 60)
    
    deli_dir = results_dir / "DELI"
    if not deli_dir.exists():
        print("No DELI results found")
        return
    
    all_results = []
    
    for filepath in sorted(deli_dir.glob("deli_*.json")):
        model = filepath.stem.replace("deli_", "").replace("_v2", "")
        data = load_json(filepath)
        
        if len(data) < 100:
            continue
        
        print(f"\nModel: {model} (N={len(data):,})")
        
        # Type
        gt_type = [r["gt_type"] for r in data]
        pred_type = [r["pred_type"] for r in data]
        type_kappa = cohen_kappa_score(gt_type, pred_type)
        
        print(f"  Type Kappa: {type_kappa:.4f}")
        df_type = compute_metrics(gt_type, pred_type, DELI_TYPE)
        df_type['Model'] = model
        df_type['Task'] = 'Type'
        all_results.append(df_type)
        
        # Target
        gt_target = [r["gt_target"] for r in data]
        pred_target = [r["pred_target"] for r in data]
        target_kappa = cohen_kappa_score(gt_target, pred_target)
        
        print(f"  Target Kappa: {target_kappa:.4f}")
        df_target = compute_metrics(gt_target, pred_target, DELI_TARGET)
        df_target['Model'] = model
        df_target['Task'] = 'Target'
        all_results.append(df_target)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = results_dir / "deli_metrics.csv"
        combined.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


def analyze_cps(results_dir: Path):
    """Analyze CPS results."""
    print("\n" + "=" * 60)
    print("CPS ANALYSIS")
    print("=" * 60)
    
    cps_dir = results_dir / "CPS"
    if not cps_dir.exists():
        # Try main results dir
        cps_files = list(results_dir.glob("cps_*.json")) + list(results_dir.glob("inference_v3_*.json"))
    else:
        cps_files = list(cps_dir.glob("cps_*.json"))
    
    if not cps_files:
        print("No CPS results found")
        return
    
    facet_names = {0: "CSK", 1: "NC", 2: "MTF"}
    all_results = []
    
    for filepath in sorted(cps_files):
        model = filepath.stem.replace("cps_", "").replace("inference_v3_", "")
        data = load_json(filepath)
        
        if len(data) < 50:
            continue
        
        print(f"\nModel: {model} (N={len(data):,})")
        
        for facet_idx, facet_name in facet_names.items():
            gt = []
            pred = []
            for r in data:
                gt_val = r.get("ground_truth", [])
                pred_val = r.get("predicted", [])
                if len(gt_val) > facet_idx and len(pred_val) > facet_idx:
                    try:
                        gt.append(int(gt_val[facet_idx]))
                        pred.append(int(pred_val[facet_idx]))
                    except (TypeError, ValueError):
                        continue
            
            if len(gt) < 10:
                continue
            
            kappa = cohen_kappa_score(gt, pred) if len(set(gt)) > 1 and len(set(pred)) > 1 else 0
            tp = sum(1 for i in range(len(gt)) if gt[i] == 1 and pred[i] == 1)
            support = sum(gt)
            pred_count = sum(pred)
            
            precision = tp / pred_count if pred_count > 0 else 0
            recall = tp / support if support > 0 else 0
            
            print(f"  {facet_name}: Kappa={kappa:.4f}, Precision={precision:.3f}, Recall={recall:.3f}")
            
            all_results.append({
                'Label': facet_name,
                'Model': model,
                'Support': support,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'Kappa': round(kappa, 4)
            })
    
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = results_dir / "cps_metrics.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate Kappa and metrics")
    parser.add_argument("--dataset", type=str, choices=["talkmoves", "deli", "cps", "all"], default="all")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.dataset in ["talkmoves", "all"]:
        analyze_talkmoves(results_dir)
    
    if args.dataset in ["deli", "all"]:
        analyze_deli(results_dir)
    
    if args.dataset in ["cps", "all"]:
        analyze_cps(results_dir)


if __name__ == "__main__":
    main()
