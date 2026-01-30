"""
Script to download the DeliData dataset from HuggingFace and save as CSV.

DeliData is a dataset for analyzing deliberation in multi-party problem-solving contexts.
It contains ~14,000 utterances from 500 group dialogues solving the Wason card selection task.

Annotation Schema (hierarchical with 3 levels):
- annotation_type: Probing, Non-probing deliberation, or None
- annotation_target: Moderation, Reasoning, Solution, Agree, or Disagree  
- annotation_additional: partial_solution, complete_solution, specific_referee, solution_summary, or consider_opposite

Source: https://huggingface.co/datasets/gkaradzhov/DeliData
Paper: https://aclanthology.org/DeliData
"""

from datasets import load_dataset
import pandas as pd
import os

def main():
    print("Downloading DeliData dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset("gkaradzhov/DeliData")
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Get the output directory (same as script location)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert each split to CSV
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        print(f"  Number of records: {len(split_data)}")
        
        # Convert to pandas DataFrame
        df = split_data.to_pandas()
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"delidata_{split_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        
        # Print column info
        print(f"  Columns: {list(df.columns)}")
        
        # Show annotation label distributions
        if 'annotation_type' in df.columns:
            print(f"\n  annotation_type distribution:")
            print(df['annotation_type'].value_counts().to_string().replace('\n', '\n    '))
        
        if 'annotation_target' in df.columns:
            print(f"\n  annotation_target distribution:")
            print(df['annotation_target'].value_counts().to_string().replace('\n', '\n    '))
            
        if 'annotation_additional' in df.columns:
            print(f"\n  annotation_additional distribution:")
            print(df['annotation_additional'].value_counts().to_string().replace('\n', '\n    '))
    
    # Also create a combined CSV with all data
    if len(dataset) > 1:
        print("\nCreating combined CSV with all splits...")
        combined_dfs = []
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            df['split'] = split_name
            combined_dfs.append(df)
        
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_path = os.path.join(output_dir, "delidata_all.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined data saved to: {combined_path}")
        print(f"Total records: {len(combined_df)}")
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main()
