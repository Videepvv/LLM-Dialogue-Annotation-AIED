#!/usr/bin/env python3
"""
Create stratified ablation test samples for each dataset.

Sampling Strategy:
- Stratified sampling based on label distribution
- Preserves index/position so context can be retrieved from original dataset
- Minority class oversampling for rare labels

Usage:
    python create_ablation_samples.py --output_dir data/ablation
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset paths (relative to repo root)
DATA_ROOT = Path("/s/babbage/h/nobackup/nblancha/public-datasets/ilideep/AutomaticAnnotations/Data")


def create_cps_sample(n_samples: int = 200) -> pd.DataFrame:
    """
    Create stratified sample from CPS dataset.
    Stratifies on multi-label combination [CSK, NC, MTF].
    """
    cps_path = DATA_ROOT / "WTD" / "OOCPS_aied.csv"
    df = pd.read_csv(cps_path)
    
    # Define facet columns
    csk_cols = [c for c in df.columns if 'CPS_CONST' in c]
    nc_cols = [c for c in df.columns if 'CPS_NEG' in c]
    mtf_cols = [c for c in df.columns if 'CPS_MAINTAIN' in c]
    
    # Create binary facet indicators
    for name, cols in [('CSK', csk_cols), ('NC', nc_cols), ('MTF', mtf_cols)]:
        df[f'{name}_binary'] = df[cols].apply(
            lambda row: 1 if any(pd.notna(v) and v not in [0, '0', ''] for v in row) else 0, 
            axis=1
        )
    
    # Create composite label for stratification
    df['strata'] = df['CSK_binary'].astype(str) + df['NC_binary'].astype(str) + df['MTF_binary'].astype(str)
    
    # Stratified sample
    np.random.seed(RANDOM_SEED)
    
    strata_counts = df['strata'].value_counts()
    sample_indices = []
    
    for stratum, count in strata_counts.items():
        stratum_df = df[df['strata'] == stratum]
        # Sample proportionally, with minimum of 2 per stratum
        n_stratum = max(2, int(n_samples * count / len(df)))
        n_stratum = min(n_stratum, len(stratum_df))  # Can't sample more than available
        sampled = stratum_df.sample(n=n_stratum, random_state=RANDOM_SEED)
        sample_indices.extend(sampled.index.tolist())
    
    # Trim to exact sample size if needed
    if len(sample_indices) > n_samples:
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[:n_samples]
    
    sample_df = df.loc[sample_indices].copy()
    sample_df['original_index'] = sample_indices  # Preserve for context retrieval
    
    print(f"CPS: Sampled {len(sample_df)} utterances from {len(df)} total")
    print(f"  Strata distribution: {sample_df['strata'].value_counts().to_dict()}")
    
    return sample_df


def create_deli_sample(n_samples: int = 200) -> pd.DataFrame:
    """
    Create stratified sample from DELI dataset.
    Stratifies on annotation_type + oversamples rare targets.
    """
    deli_path = DATA_ROOT / "DeliData" / "delidata_train.csv"
    df = pd.read_csv(deli_path)
    
    # Filter to MESSAGE types only (not SYSTEM/INITIAL)
    df = df[df['message_type'] == 'MESSAGE'].copy()
    
    # Create strata from annotation_type
    df['strata'] = df['annotation_type'].fillna('None').astype(str)
    
    # Stratified sample
    np.random.seed(RANDOM_SEED)
    
    strata_counts = df['strata'].value_counts()
    sample_indices = []
    
    for stratum, count in strata_counts.items():
        stratum_df = df[df['strata'] == stratum]
        # Sample proportionally, minimum 5 per stratum
        n_stratum = max(5, int(n_samples * count / len(df)))
        n_stratum = min(n_stratum, len(stratum_df))
        sampled = stratum_df.sample(n=n_stratum, random_state=RANDOM_SEED)
        sample_indices.extend(sampled.index.tolist())
    
    # Oversample rare targets (Disagree)
    disagree_df = df[df['annotation_target'] == 'Disagree']
    if len(disagree_df) > 0:
        n_extra = min(10, len(disagree_df))  # Add up to 10 Disagree samples
        extra = disagree_df.sample(n=n_extra, random_state=RANDOM_SEED)
        sample_indices.extend(extra.index.tolist())
    
    # Remove duplicates and trim
    sample_indices = list(set(sample_indices))
    if len(sample_indices) > n_samples:
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[:n_samples]
    
    sample_df = df.loc[sample_indices].copy()
    sample_df['original_index'] = sample_indices
    
    print(f"DELI: Sampled {len(sample_df)} messages from {len(df)} MESSAGE types")
    print(f"  Type distribution: {sample_df['strata'].value_counts().to_dict()}")
    print(f"  Target distribution: {sample_df['annotation_target'].value_counts().to_dict()}")
    
    return sample_df


def create_talkmoves_sample(data_type: str = "teacher", n_samples: int = 200) -> pd.DataFrame:
    """
    Create stratified sample from TalkMoves dataset.
    Oversamples minority classes.
    """
    talkmoves_path = DATA_ROOT / "TalkMoves" / "data" / f"train_{data_type}.tsv"
    df = pd.read_csv(talkmoves_path, sep='\t')
    
    # Stratify on labels
    df['strata'] = df['labels'].fillna(-1).astype(int).astype(str)
    
    np.random.seed(RANDOM_SEED)
    
    strata_counts = df['strata'].value_counts()
    sample_indices = []
    
    # Minority classes to oversample
    if data_type == "teacher":
        minority_labels = ['2', '3', '6']  # Labels with <3% representation
    else:
        minority_labels = ['2']  # Only label 2 is rare for students
    
    for stratum, count in strata_counts.items():
        stratum_df = df[df['strata'] == stratum]
        
        # Base proportional sample
        n_stratum = max(5, int(n_samples * count / len(df)))
        
        # Oversample minority classes (2x)
        if stratum in minority_labels:
            n_stratum = min(n_stratum * 2, len(stratum_df))
        
        n_stratum = min(n_stratum, len(stratum_df))
        sampled = stratum_df.sample(n=n_stratum, random_state=RANDOM_SEED)
        sample_indices.extend(sampled.index.tolist())
    
    # Trim to target size
    sample_indices = list(set(sample_indices))
    if len(sample_indices) > n_samples:
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[:n_samples]
    
    sample_df = df.loc[sample_indices].copy()
    sample_df['original_index'] = sample_indices
    
    print(f"TalkMoves {data_type}: Sampled {len(sample_df)} from {len(df)} total")
    print(f"  Label distribution: {sample_df['strata'].value_counts().to_dict()}")
    
    return sample_df


def generate_report(samples: dict, output_dir: Path):
    """Generate a summary report of the ablation samples."""
    report_path = output_dir / "ablation_samples_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Ablation Test Samples Summary\n\n")
        f.write(f"Generated with random seed: {RANDOM_SEED}\n\n")
        
        for name, df in samples.items():
            f.write(f"## {name}\n\n")
            f.write(f"- **Sample size**: {len(df)}\n")
            f.write(f"- **Strata distribution**:\n")
            for stratum, count in df['strata'].value_counts().items():
                f.write(f"  - `{stratum}`: {count} ({100*count/len(df):.1f}%)\n")
            f.write("\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Create ablation test samples")
    parser.add_argument("--output_dir", type=str, 
                        default="/s/babbage/h/nobackup/nblancha/public-datasets/ilideep/LLM-Dialogue-Annotation-AIED/data/ablation")
    parser.add_argument("--cps_n", type=int, default=200)
    parser.add_argument("--deli_n", type=int, default=200)
    parser.add_argument("--talkmoves_teacher_n", type=int, default=200)
    parser.add_argument("--talkmoves_student_n", type=int, default=150)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Creating Ablation Test Samples")
    print("=" * 60 + "\n")
    
    samples = {}
    
    # CPS
    cps_sample = create_cps_sample(args.cps_n)
    cps_sample.to_csv(output_dir / f"cps_ablation_{len(cps_sample)}.csv", index=False)
    samples['CPS'] = cps_sample
    
    print()
    
    # DELI
    deli_sample = create_deli_sample(args.deli_n)
    deli_sample.to_csv(output_dir / f"deli_ablation_{len(deli_sample)}.csv", index=False)
    samples['DELI'] = deli_sample
    
    print()
    
    # TalkMoves Teacher
    teacher_sample = create_talkmoves_sample("teacher", args.talkmoves_teacher_n)
    teacher_sample.to_csv(output_dir / f"talkmoves_teacher_ablation_{len(teacher_sample)}.csv", index=False)
    samples['TalkMoves_Teacher'] = teacher_sample
    
    print()
    
    # TalkMoves Student
    student_sample = create_talkmoves_sample("student", args.talkmoves_student_n)
    student_sample.to_csv(output_dir / f"talkmoves_student_ablation_{len(student_sample)}.csv", index=False)
    samples['TalkMoves_Student'] = student_sample
    
    # Generate report
    generate_report(samples, output_dir)
    
    print("\n" + "=" * 60)
    print("Done! Files saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
