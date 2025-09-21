#!/usr/bin/env python3

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import argparse
import pandas as pd
from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset with neurosymbolic features')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, help='Path to output CSV file')
    parser.add_argument('--func_column', type=str, default='func', help='Name of function code column')
    
    args = parser.parse_args()
    
    if args.output_csv is None:
        args.output_csv = args.input_csv.replace('.csv', '_with_neuro.csv')
    
    print(f"Loading dataset from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")
    
    if args.func_column not in df.columns:
        raise ValueError(f"Column '{args.func_column}' not found in dataset")
    
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in dataset")
    
    print("Processing neurosymbolic features...")
    extractor = NeurosymbolicFeatureExtractor()
    
    if args.func_column != 'func':
        df = df.rename(columns={args.func_column: 'func'})
    
    processed_df = extractor.process_dataset(args.input_csv, args.output_csv)
    
    print(f"Generated neurosymbolic features for {len(processed_df)} samples")
    print(f"Sample feature length: {len(eval(processed_df['neuro'].iloc[0]))}")
    
    processed_df.to_csv(args.output_csv, index=False)
    print(f"Saved processed dataset to: {args.output_csv}")

if __name__ == "__main__":

    main()
