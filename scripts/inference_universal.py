#!/usr/bin/env python3

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from src.models.p3r_stage2_model import P3RStage2Model
from src.data.dataset import CodeDataset, create_collate_fn
from src.evaluation.metrics import calculate_comprehensive_metrics, print_metrics_summary
from src.evaluation.space_metrics import SpaceMissionEvaluator, KSPMissionSimulator
from src.visualization.dashboard import create_ksp_dashboard

def main():
    parser = argparse.ArgumentParser(description='Universal P3R Inference')
    parser.add_argument('--test_data', type=str, required=True, help='Test CSV file')
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name')
    parser.add_argument('--stage1_path', type=str, required=True, help='Stage 1 classifier path')
    parser.add_argument('--stage2_path', type=str, required=True, help='Stage 2 P3R model path')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Pretrained model: {args.model_name}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading test dataset...")
    test_df = pd.read_csv(args.test_data)
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:\n{test_df['label'].value_counts()}")
    
    print("Initializing P3R Stage 2 model...")
    model = P3RStage2Model(
        model_name=args.model_name,
        stage1_classifier_path=args.stage1_path
    ).to(device)
    
    print("Loading Stage 2 weights...")
    checkpoint = torch.load(args.stage2_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    trainable_params, total_params = model.count_parameters()
    print(f"Trainable: {trainable_params:,} | Total: {total_params:,}")
    print(f"Efficiency: {trainable_params/total_params:.1%}")
    
    collate_fn = create_collate_fn(model.tokenizer)
    test_dataset = CodeDataset(test_df, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("Running inference...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx+1}/{len(test_loader)}")
            
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
    
    print("Calculating metrics...")
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    print("Running space mission evaluation...")
    try:
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
        metrics['ksp_thermal_stability'] = ksp_impact['thermal_stability']
        
    except Exception as e:
        print(f"Warning: Space metrics evaluation failed: {e}")
    
    print("\n=== P3R INFERENCE RESULTS ===")
    print_metrics_summary(metrics)
    
    print("Exporting results...")
    try:
        ksp_csv_path = ksp_sim.export_data(args.output_dir)
        
        if ksp_csv_path and os.path.exists(ksp_csv_path):
            print("Creating mission dashboard...")
            dashboard_path = os.path.join(args.output_dir, "mission_dashboard.png")
            create_ksp_dashboard(ksp_csv_path, dashboard_path)
            print(f"Dashboard saved: {dashboard_path}")
        
    except Exception as e:
        print(f"Warning: Dashboard creation failed: {e}")
    
    results_summary = {
        'pretrained_model': args.model_name,
        'test_samples': len(test_df),
        'accuracy': float(metrics['acc']),
        'f1_score': float(metrics['f1']),
        'precision': float(metrics['prec']),
        'recall': float(metrics['rec']),
        'roc_auc': float(metrics['roc_auc']),
        'space_dit_score': float(metrics.get('space_dit_score', 0)),
        'ksp_orbital_efficiency': float(metrics.get('ksp_orbital_efficiency', 0)),
        'parameter_efficiency': f"{trainable_params/total_params:.1%}",
        'trainable_parameters': trainable_params,
        'total_parameters': total_params
    }
    
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults exported to: {args.output_dir}")
    print("Universal P3R inference completed successfully!")

if __name__ == "__main__":
    main()
