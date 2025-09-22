#!/usr/bin/env python3

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from src.models.universal_p3r import UniversalP3RModel
from src.data.universal_dataset import Stage2Dataset, create_stage2_collate_fn
from src.evaluation.metrics import calculate_comprehensive_metrics, print_metrics_summary
from src.evaluation.space_metrics import SpaceMissionEvaluator, KSPMissionSimulator
from src.visualization.dashboard import create_ksp_dashboard
from src.preprocessing.neurosymbolic_extractor import preprocess_dataset

def main():
    parser = argparse.ArgumentParser(description='Two-Stage P3R Inference')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained P3R model')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to stage 1 classifier')
    parser.add_argument('--model_name', type=str, default='microsoft/unixcoder-base', 
                       help='HuggingFace model name used in training')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--preprocess', action='store_true', help='Run neurosymbolic preprocessing on test data')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Using model: {args.model_name}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.preprocess:
        print("Running neurosymbolic preprocessing on test data...")
        preprocessed_path = args.test_data.replace('.csv', '_preprocessed.csv')
        test_df = preprocess_dataset(args.test_data, preprocessed_path)
    else:
        test_df = pd.read_csv(args.test_data)
    
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:\n{test_df['label'].value_counts()}")
    
    print("\nInitializing P3R model...")
    model = UniversalP3RModel(
        model_name=args.model_name,
        num_prompts=4,
        prompt_length=8,
        stage=2
    ).to(device)
    
    print("Loading stage 1 classifier...")
    if not os.path.exists(args.classifier_path):
        raise FileNotFoundError(f"Classifier file not found: {args.classifier_path}")
    
    model.load_stage1_classifier(args.classifier_path)
    
    print("Loading stage 2 P3R model...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    trainable_params, total_params = model.count_parameters()
    print(f"Trainable: {trainable_params:,} | Total: {total_params:,}")
    print(f"Efficiency: {trainable_params/total_params:.1%}")
    
    collate_fn = create_stage2_collate_fn(model.tokenizer)
    test_dataset = Stage2Dataset(test_df, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    print("\nRunning inference...")
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
    
    print("\nCalculating metrics...")
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    print("\nRunning space mission evaluation...")
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
    
    print("\n=== TWO-STAGE P3R INFERENCE RESULTS ===")
    print_metrics_summary(metrics)
    
    print("\nExporting results...")
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
        'test_samples': len(test_df),
        'model_name': args.model_name,
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
    print("Two-Stage P3R Inference completed successfully!")

if __name__ == "__main__":
    main()
