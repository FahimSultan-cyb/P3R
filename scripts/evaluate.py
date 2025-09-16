#!/usr/bin/env python3

import sys
import os

# Add the parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.models.p3r_model import P3RHeadGateModel
from src.data.dataset import CodeDataset, create_collate_fn
from src.evaluation.metrics import calculate_comprehensive_metrics, print_metrics_summary
from src.evaluation.space_metrics import SpaceMissionEvaluator, KSPMissionSimulator

def evaluate_model(args):
    device = torch.device(args.device if args.device != 'auto' else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    test_df = pd.read_csv(args.test_data)
    print(f"Test samples: {len(test_df)}")
    
    model = P3RHeadGateModel().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    collate_fn = create_collate_fn(model.tokenizer)
    test_dataset = CodeDataset(test_df, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    print("Running evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            chunks = batch['chunks'].to(device)
            full_code = batch['full_code'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(chunks, full_code, attention_mask)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    if args.space_metrics:
        print("Running space mission evaluation...")
        mission_eval = SpaceMissionEvaluator()
        ksp_sim = KSPMissionSimulator()
        
        mission_profiles = mission_eval.generate_mission_profile(len(test_df))
        dit_scores = mission_eval.calculate_dit_scores(mission_profiles)
        
        ksp_sim.run_simulation()
        ksp_impact = ksp_sim.analyze_impact()
        
        dit_values = [score['overall_score'] for score in dit_scores]
        metrics['space_dit_score'] = sum(dit_values) / len(dit_values)
        metrics['ksp_orbital_efficiency'] = ksp_impact['orbital_efficiency']
        metrics['ksp_fuel_efficiency'] = ksp_impact['fuel_efficiency']
    
    print("\n=== EVALUATION RESULTS ===")
    print_metrics_summary(metrics)
    
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (int, float, bool)) else str(v) 
                      for k, v in metrics.items() if k != 'cm'}, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='P3R Model Evaluation')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='models/p3r_headgate_model1.pth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--space_metrics', action='store_true')
    parser.add_argument('--output_file', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main()
