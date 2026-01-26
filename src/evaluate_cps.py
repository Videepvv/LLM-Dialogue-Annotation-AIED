#!/usr/bin/env python3
"""
Comprehensive CPS Annotation Evaluation Script

This script evaluates LLM annotation results for the CPS (Collaborative Problem Solving)
Weights Task dataset. It computes:
- Overall Kappa, Precision, Recall, F1 for each facet
- Per-label (facet) Kappa analysis
- Confusion matrices
- Sankey diagrams showing misclassification flows
- Detailed misclassification analysis with examples

Usage:
    python evaluate_cps.py --input <path_to_json> --output <output_dir>
    python evaluate_cps.py --input results/CPS/cps_gpt-4o-mini.json --output reports/

Author: Auto-generated for LLM-Dialogue-Annotation-AIED project
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing plotly for Sankey diagrams
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Sankey diagrams will not be generated.")
    print("Install with: pip install plotly kaleido")

# ============================================================================
# Configuration
# ============================================================================

FACET_NAMES = {0: "CSK", 1: "NC", 2: "MTF"}
FACET_FULL_NAMES = {
    0: "Constructing Shared Knowledge (CSK)",
    1: "Negotiation & Coordination (NC)", 
    2: "Maintaining Team Function (MTF)"
}


def load_json(path: Path) -> list:
    """Load JSON data from file."""
    with open(path) as f:
        return json.load(f)


def extract_facet_labels(data: list, facet_idx: int) -> tuple:
    """
    Extract ground truth and predicted labels for a specific facet.
    
    Returns:
        tuple: (ground_truth_list, predicted_list, valid_indices)
    """
    gt_labels = []
    pred_labels = []
    valid_indices = []
    
    for i, record in enumerate(data):
        gt_val = record.get("ground_truth", [])
        pred_val = record.get("predicted", [])
        
        if len(gt_val) > facet_idx and len(pred_val) > facet_idx:
            try:
                gt = int(gt_val[facet_idx])
                pred = int(pred_val[facet_idx])
                gt_labels.append(gt)
                pred_labels.append(pred)
                valid_indices.append(i)
            except (TypeError, ValueError):
                continue
    
    return gt_labels, pred_labels, valid_indices


def compute_metrics(gt: list, pred: list) -> dict:
    """Compute all metrics for binary classification."""
    n = len(gt)
    
    # Count TPs, FPs, FNs, TNs
    tp = sum(1 for i in range(n) if gt[i] == 1 and pred[i] == 1)
    fp = sum(1 for i in range(n) if gt[i] == 0 and pred[i] == 1)
    fn = sum(1 for i in range(n) if gt[i] == 1 and pred[i] == 0)
    tn = sum(1 for i in range(n) if gt[i] == 0 and pred[i] == 0)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / n if n > 0 else 0
    
    # Kappa
    kappa = 0.0
    if len(set(gt)) > 1 and len(set(pred)) > 1:
        kappa = cohen_kappa_score(gt, pred)
    
    return {
        'n': n,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'kappa': kappa,
        'support_positive': sum(gt),
        'support_negative': n - sum(gt),
        'pred_positive': sum(pred),
        'pred_negative': n - sum(pred)
    }


def create_confusion_matrix_plot(gt: list, pred: list, facet_name: str, output_path: Path):
    """Create and save a confusion matrix heatmap."""
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.title(f'Confusion Matrix: {facet_name}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm


def create_multilabel_confusion_matrix(data: list, output_path: Path) -> np.ndarray:
    """
    Create a confusion matrix for all 8 possible label combinations.
    Labels are represented as strings like "[0,0,0]", "[1,0,0]", etc.
    """
    # Generate all possible label combinations
    label_combos = []
    for csk in [0, 1]:
        for nc in [0, 1]:
            for mtf in [0, 1]:
                label_combos.append(f"[{csk},{nc},{mtf}]")
    
    # Extract GT and Pred as string labels
    gt_labels = []
    pred_labels = []
    
    for record in data:
        gt_val = record.get("ground_truth", [])
        pred_val = record.get("predicted", [])
        
        if len(gt_val) >= 3 and len(pred_val) >= 3:
            try:
                gt_str = f"[{int(gt_val[0])},{int(gt_val[1])},{int(gt_val[2])}]"
                pred_str = f"[{int(pred_val[0])},{int(pred_val[1])},{int(pred_val[2])}]"
                gt_labels.append(gt_str)
                pred_labels.append(pred_str)
            except (TypeError, ValueError):
                continue
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix as sklearn_cm
    cm = sklearn_cm(gt_labels, pred_labels, labels=label_combos)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_combos,
                yticklabels=label_combos)
    plt.xlabel('Predicted Label [CSK, NC, MTF]', fontsize=12)
    plt.ylabel('Ground Truth Label [CSK, NC, MTF]', fontsize=12)
    plt.title('Multi-Label Confusion Matrix: All CPS Facet Combinations', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm, label_combos, gt_labels, pred_labels


def create_multilabel_sankey(gt_labels: list, pred_labels: list, label_combos: list, output_path: Path) -> bool:
    """Create a Sankey diagram showing flows between all label combinations."""
    if not PLOTLY_AVAILABLE:
        return False
    
    from collections import Counter
    
    # Count transitions
    transitions = Counter(zip(gt_labels, pred_labels))
    
    # Filter to show only significant flows (>1% or at least 5 samples)
    min_flow = max(5, int(len(gt_labels) * 0.01))
    
    # Create node indices
    gt_nodes = [f"GT: {lbl}" for lbl in label_combos]
    pred_nodes = [f"Pred: {lbl}" for lbl in label_combos]
    all_nodes = gt_nodes + pred_nodes
    
    sources = []
    targets = []
    values = []
    colors = []
    
    for (gt, pred), count in transitions.items():
        if count >= min_flow:
            gt_idx = label_combos.index(gt)
            pred_idx = label_combos.index(pred) + len(label_combos)  # Offset for pred nodes
            sources.append(gt_idx)
            targets.append(pred_idx)
            values.append(count)
            # Green for correct, red for misclassification
            if gt == pred:
                colors.append('rgba(50, 180, 50, 0.5)')
            else:
                colors.append('rgba(220, 80, 80, 0.4)')
    
    if not sources:
        return False
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
        )
    )])
    
    fig.update_layout(
        title_text="Multi-Label Classification Flow: [CSK, NC, MTF]",
        font_size=10,
        width=1200,
        height=800
    )
    
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    
    try:
        fig.write_image(str(output_path))
    except Exception:
        pass
    
    return True


def create_sankey_diagram(gt: list, pred: list, facet_name: str, output_path: Path) -> bool:
    """Create a Sankey diagram showing classification flows."""
    if not PLOTLY_AVAILABLE:
        return False
    
    # Count transitions
    tn = sum(1 for i in range(len(gt)) if gt[i] == 0 and pred[i] == 0)
    fp = sum(1 for i in range(len(gt)) if gt[i] == 0 and pred[i] == 1)
    fn = sum(1 for i in range(len(gt)) if gt[i] == 1 and pred[i] == 0)
    tp = sum(1 for i in range(len(gt)) if gt[i] == 1 and pred[i] == 1)
    
    # Define nodes: Ground Truth (left), Predictions (right)
    node_labels = [
        f"GT: Negative\n(n={sum(1 for g in gt if g == 0)})",
        f"GT: Positive\n(n={sum(1 for g in gt if g == 1)})",
        f"Pred: Negative\n(n={sum(1 for p in pred if p == 0)})",
        f"Pred: Positive\n(n={sum(1 for p in pred if p == 1)})"
    ]
    
    # Define links
    # Source: 0 = GT Negative, 1 = GT Positive
    # Target: 2 = Pred Negative, 3 = Pred Positive
    sources = [0, 0, 1, 1]
    targets = [2, 3, 2, 3]
    values = [tn, fp, fn, tp]
    
    # Colors for links
    link_colors = [
        'rgba(50, 150, 50, 0.4)',   # TN: green
        'rgba(220, 50, 50, 0.4)',   # FP: red
        'rgba(220, 50, 50, 0.4)',   # FN: red
        'rgba(50, 150, 50, 0.4)'    # TP: green
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=["#2ecc71", "#3498db", "#2ecc71", "#3498db"]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=[f"TN: {tn}", f"FP: {fp}", f"FN: {fn}", f"TP: {tp}"]
        )
    )])
    
    fig.update_layout(
        title_text=f"Classification Flow: {facet_name}",
        font_size=12,
        width=800,
        height=500
    )
    
    # Save as HTML (works everywhere) and try PNG
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    
    try:
        fig.write_image(str(output_path))
    except Exception as e:
        print(f"  Note: Could not save Sankey as PNG ({e}). HTML version saved.")
    
    return True


def get_misclassification_examples(data: list, gt: list, pred: list, 
                                    valid_indices: list, facet_name: str,
                                    n_examples: int = 5) -> dict:
    """
    Get examples of misclassifications.
    
    Returns dict with:
        - false_positives: examples where GT=0, Pred=1
        - false_negatives: examples where GT=1, Pred=0
    """
    fp_examples = []
    fn_examples = []
    
    for i, idx in enumerate(valid_indices):
        if gt[i] == 0 and pred[i] == 1:  # False Positive
            record = data[idx]
            fp_examples.append({
                'idx': record.get('idx', idx),
                'utterance': record.get('utterance', 'N/A'),
                'reasoning': record.get('reasoning', 'N/A')
            })
        elif gt[i] == 1 and pred[i] == 0:  # False Negative
            record = data[idx]
            fn_examples.append({
                'idx': record.get('idx', idx),
                'utterance': record.get('utterance', 'N/A'),
                'reasoning': record.get('reasoning', 'N/A')
            })
    
    return {
        'false_positives': fp_examples[:n_examples],
        'false_negatives': fn_examples[:n_examples],
        'fp_count': len(fp_examples),
        'fn_count': len(fn_examples)
    }


def generate_markdown_report(data: list, model_name: str, output_dir: Path) -> Path:
    """Generate a comprehensive markdown report."""
    
    report_lines = []
    
    report_lines.append(f"# CPS Annotation Evaluation Report")
    report_lines.append(f"")
    report_lines.append(f"**Model**: `{model_name}`")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Total Samples**: {len(data):,}")
    report_lines.append(f"")
    
    # Overall metrics table
    report_lines.append(f"## Overall Metrics Summary")
    report_lines.append(f"")
    report_lines.append(f"| Facet | N | Kappa | Precision | Recall | F1 | Accuracy |")
    report_lines.append(f"|-------|---|-------|-----------|--------|----|---------| ")
    
    all_metrics = {}
    for facet_idx, facet_name in FACET_NAMES.items():
        gt, pred, valid_idx = extract_facet_labels(data, facet_idx)
        metrics = compute_metrics(gt, pred)
        all_metrics[facet_name] = {
            'metrics': metrics,
            'gt': gt,
            'pred': pred,
            'valid_idx': valid_idx
        }
        
        report_lines.append(
            f"| {facet_name} | {metrics['n']:,} | {metrics['kappa']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['accuracy']:.4f} |"
        )
    
    report_lines.append(f"")
    
    # Multi-label confusion matrix section
    report_lines.append(f"---")
    report_lines.append(f"")
    report_lines.append(f"## Multi-Label Confusion Matrix")
    report_lines.append(f"")
    report_lines.append(f"This shows how all 8 possible label combinations `[CSK, NC, MTF]` get confused with each other.")
    report_lines.append(f"")
    
    # Generate multi-label confusion matrix
    ml_cm_path = output_dir / "confusion_matrix_multilabel.png"
    cm_result = create_multilabel_confusion_matrix(data, ml_cm_path)
    cm, label_combos, gt_labels, pred_labels = cm_result
    
    report_lines.append(f"![Multi-Label Confusion Matrix](confusion_matrix_multilabel.png)")
    report_lines.append(f"")
    
    # Multi-label Sankey
    ml_sankey_path = output_dir / "sankey_multilabel.png"
    if create_multilabel_sankey(gt_labels, pred_labels, label_combos, ml_sankey_path):
        report_lines.append(f"### Classification Flow (Multi-Label)")
        report_lines.append(f"")
        report_lines.append(f"Green flows indicate correct predictions, red flows indicate misclassifications.")
        report_lines.append(f"")
        report_lines.append(f"![Multi-Label Sankey Diagram](sankey_multilabel.png)")
        report_lines.append(f"")
    
    # Per-facet detailed analysis
    report_lines.append(f"---")
    report_lines.append(f"")
    report_lines.append(f"## Detailed Per-Facet Analysis")
    report_lines.append(f"")
    
    for facet_idx, facet_name in FACET_NAMES.items():
        full_name = FACET_FULL_NAMES[facet_idx]
        m = all_metrics[facet_name]
        metrics = m['metrics']
        gt = m['gt']
        pred = m['pred']
        valid_idx = m['valid_idx']
        
        report_lines.append(f"### {full_name}")
        report_lines.append(f"")
        
        # Distribution info
        report_lines.append(f"**Label Distribution:**")
        report_lines.append(f"- Ground Truth: {metrics['support_positive']:,} positive ({metrics['support_positive']/metrics['n']*100:.1f}%), "
                           f"{metrics['support_negative']:,} negative ({metrics['support_negative']/metrics['n']*100:.1f}%)")
        report_lines.append(f"- Predicted: {metrics['pred_positive']:,} positive ({metrics['pred_positive']/metrics['n']*100:.1f}%), "
                           f"{metrics['pred_negative']:,} negative ({metrics['pred_negative']/metrics['n']*100:.1f}%)")
        report_lines.append(f"")
        
        # Confusion matrix details
        report_lines.append(f"**Confusion Matrix Breakdown:**")
        report_lines.append(f"| | Predicted Negative | Predicted Positive |")
        report_lines.append(f"|---|---|---|")
        report_lines.append(f"| **GT Negative** | TN: {metrics['tn']:,} | FP: {metrics['fp']:,} |")
        report_lines.append(f"| **GT Positive** | FN: {metrics['fn']:,} | TP: {metrics['tp']:,} |")
        report_lines.append(f"")
        
        # Save confusion matrix plot
        cm_path = output_dir / f"confusion_matrix_{facet_name}.png"
        create_confusion_matrix_plot(gt, pred, full_name, cm_path)
        report_lines.append(f"![Confusion Matrix - {facet_name}](confusion_matrix_{facet_name}.png)")
        report_lines.append(f"")
        
        # Sankey diagram
        sankey_path = output_dir / f"sankey_{facet_name}.png"
        if create_sankey_diagram(gt, pred, full_name, sankey_path):
            report_lines.append(f"![Sankey Diagram - {facet_name}](sankey_{facet_name}.png)")
            report_lines.append(f"")
        
        report_lines.append(f"")
    
    # Misclassification Analysis
    report_lines.append(f"---")
    report_lines.append(f"")
    report_lines.append(f"## Misclassification Analysis")
    report_lines.append(f"")
    
    for facet_idx, facet_name in FACET_NAMES.items():
        full_name = FACET_FULL_NAMES[facet_idx]
        m = all_metrics[facet_name]
        gt = m['gt']
        pred = m['pred']
        valid_idx = m['valid_idx']
        
        examples = get_misclassification_examples(data, gt, pred, valid_idx, facet_name)
        
        report_lines.append(f"### {full_name}")
        report_lines.append(f"")
        report_lines.append(f"- **Total False Positives (FP)**: {examples['fp_count']:,}")
        report_lines.append(f"- **Total False Negatives (FN)**: {examples['fn_count']:,}")
        report_lines.append(f"")
        
        if examples['false_positives']:
            report_lines.append(f"#### False Positive Examples (GT=0, Pred=1)")
            report_lines.append(f"")
            for ex in examples['false_positives']:
                report_lines.append(f"**Sample {ex['idx']}:**")
                report_lines.append(f"> {ex['utterance']}")
                report_lines.append(f"")
                report_lines.append(f"*Model Reasoning*: {ex['reasoning']}")
                report_lines.append(f"")
        
        if examples['false_negatives']:
            report_lines.append(f"#### False Negative Examples (GT=1, Pred=0)")
            report_lines.append(f"")
            for ex in examples['false_negatives']:
                report_lines.append(f"**Sample {ex['idx']}:**")
                report_lines.append(f"> {ex['utterance']}")
                report_lines.append(f"")
                report_lines.append(f"*Model Reasoning*: {ex['reasoning']}")
                report_lines.append(f"")
        
        report_lines.append(f"")
    
    # Summary statistics
    report_lines.append(f"---")
    report_lines.append(f"")
    report_lines.append(f"## Summary Statistics")
    report_lines.append(f"")
    
    # Compute overall match rate
    match_count = sum(1 for record in data if record.get('match', False))
    match_rate = match_count / len(data) if len(data) > 0 else 0
    
    report_lines.append(f"- **Exact Match Rate** (all 3 facets match): {match_rate:.2%} ({match_count:,}/{len(data):,})")
    report_lines.append(f"")
    
    # Best/worst performing facets
    kappas = [(name, all_metrics[name]['metrics']['kappa']) for name in FACET_NAMES.values()]
    kappas_sorted = sorted(kappas, key=lambda x: x[1], reverse=True)
    
    report_lines.append(f"**Facet Performance Ranking (by Kappa):**")
    for i, (name, kappa) in enumerate(kappas_sorted, 1):
        report_lines.append(f"{i}. {name}: κ = {kappa:.4f}")
    report_lines.append(f"")
    
    # Parse failure analysis
    parse_failures = sum(1 for record in data if record.get('reasoning', '') == 'Failed to parse')
    parse_failure_rate = parse_failures / len(data) if len(data) > 0 else 0
    report_lines.append(f"**Parse Failures**: {parse_failures:,} ({parse_failure_rate:.1%})")
    report_lines.append(f"")
    
    # Write report
    report_path = output_dir / f"evaluation_report_{model_name}.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CPS annotation results with comprehensive metrics and visualizations"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the JSON results file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for report and figures. Defaults to same directory as input."
    )
    parser.add_argument(
        "--n-examples", "-n",
        type=int,
        default=5,
        help="Number of misclassification examples to show per category"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output directory - now includes model subfolder
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / "reports"
    
    # Extract model name from filename
    model_name = input_path.stem.replace("cps_", "")
    
    # Create model-specific subfolder
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"CPS Annotation Evaluation")
    print(f"=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_name}")
    print()
    
    # Load data
    data = load_json(input_path)
    print(f"Loaded {len(data):,} samples")
    print()
    
    # Quick summary to console
    print("Quick Metrics Summary:")
    print("-" * 40)
    for facet_idx, facet_name in FACET_NAMES.items():
        gt, pred, _ = extract_facet_labels(data, facet_idx)
        metrics = compute_metrics(gt, pred)
        print(f"{facet_name}: κ={metrics['kappa']:.4f}, P={metrics['precision']:.4f}, "
              f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    print()
    
    # Generate full report
    print("Generating report...")
    report_path = generate_markdown_report(data, model_name, output_dir)
    print(f"✓ Report saved to: {report_path}")
    print()
    
    print("=" * 60)
    print("Evaluation complete!")
    print(f"View the full report at: {report_path}")


if __name__ == "__main__":
    main()
