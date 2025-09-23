#!/usr/bin/env python3

# import sys
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import argparse
from src.preprocessing.neurosymbolic_extractor import process_dataset

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset with neurosymbolic features')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, help='Output CSV file path (default: input_preprocessed.csv)')
    
    args = parser.parse_args()
    
    if args.output_csv is None:
        args.output_csv = args.input_csv.replace('.csv', '_preprocessed.csv')
    
    print(f"Preprocessing {args.input_csv}...")
    print(f"Output will be saved to {args.output_csv}")
    
    try:
        df = process_dataset(args.input_csv, args.output_csv)
        print(f"Preprocessing completed successfully!")
        print(f"Processed {len(df)} samples")
        print(f"Sample neurosymbolic feature: {df['neuro'].iloc[0][:100]}...")
    except Exception as e:
        print(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()




