#!/usr/bin/env python3

import sys
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir)) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)



script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Default dynamic paths
default_model_path = os.path.join(project_root, 'models', 'p3r_headgate_model1.pth')
default_classifier_path = os.path.join(project_root, 'models', 'symbolic_classifier1n.pth')
default_output_dir = os.path.join(project_root, 'results')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=default_model_path, 
                    help='Path to the model file')
parser.add_argument('--classifier_path', type=str, default=default_classifier_path, 
                    help='Path to the classifier file')
parser.add_argument('--output_dir', type=str, default=default_output_dir, 
                    help='Directory to save results')




import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


from src.models import P3RHeadGateModel
from src.data.dataset import CodeDataset, create_collate_fn
from src.evaluation.metrics import calculate_comprehensive_metrics, print_metrics_summary
from src.evaluation.space_metrics import SpaceMissionEvaluator, KSPMissionSimulator
from src.visualization.dashboard import create_ksp_dashboard


def main():
    parser = argparse.ArgumentParser(description='P3R Model Inference')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test CSV file')
    # parser.add_argument('--model_path', type=str, default='/content/models/p3r_headgate_model1.pth')
    # parser.add_argument('--classifier_path', type=str, default='/content/models/symbolic_classifier1n.pth')
    # parser.add_argument('--output_dir', type=str, default='/content/results/')
    parser.add_argument('--model_path', type=str, default=default_model_path, 
                    help='Path to the model file')
    parser.add_argument('--classifier_path', type=str, default=default_classifier_path, 
                    help='Path to the classifier file')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, 
                    help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Device selection
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test dataset...")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    test_df = pd.read_csv(args.test_data)
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:\n{test_df['label'].value_counts()}")
    
    # Initialize model
    print("\nInitializing model...")
    try:
        model = P3RHeadGateModel().to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Load trained weights
    print("Loading trained weights...")
    if not os.path.exists(args.model_path):
        print(f"Warning: Model file not found at {args.model_path}")
        print("Please run: python scripts/download_models.py")
        return
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # Model info
    trainable_params, total_params = model.count_parameters()
    print(f"Trainable: {trainable_params:,} | Total: {total_params:,}")
    print(f"Efficiency: {trainable_params/total_params:.1%}")
    
    # Prepare data
    collate_fn = create_collate_fn(model.tokenizer)
    test_dataset = CodeDataset(test_df, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    # Run inference
    print("\nRunning inference...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    try:
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
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    # Space mission evaluation
    print("\nRunning space mission evaluation...")
    try:
        mission_eval = SpaceMissionEvaluator()
        ksp_sim = KSPMissionSimulator()
        
        mission_profiles = mission_eval.generate_mission_profile(len(test_df))
        dit_scores = mission_eval.calculate_dit_scores(mission_profiles)
        
        ksp_sim.run_simulation()
        ksp_impact = ksp_sim.analyze_impact()
        
        # Add space metrics
        dit_values = [score['overall_score'] for score in dit_scores]
        metrics['space_dit_score'] = sum(dit_values) / len(dit_values)
        metrics['ksp_orbital_efficiency'] = ksp_impact['orbital_efficiency']
        metrics['ksp_fuel_efficiency'] = ksp_impact['fuel_efficiency']
        metrics['ksp_thermal_stability'] = ksp_impact['thermal_stability']
        
    except Exception as e:
        print(f"Warning: Space metrics evaluation failed: {e}")
    
    # Print results
    print("\n=== INFERENCE RESULTS ===")
    print_metrics_summary(metrics)
    
    # Export KSP data and create dashboard
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
    
    # Save results summary
    results_summary = {
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
    try:
        import json
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Summary saved: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save summary: {e}")
    
    print(f"\nResults exported to: {args.output_dir}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()


