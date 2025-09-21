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

from src.models.corrected_universal_p3r_model import UniversalP3RModel
from src.training.corrected_two_stage_trainer import CompactSymbolicClassifier
from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor
from src.data.dataset import CodeDataset, create_collate_fn
from src.evaluation.metrics import calculate_comprehensive_metrics, print_metrics_summary
from src.evaluation.space_metrics import SpaceMissionEvaluator, KSPMissionSimulator
from src.visualization.dashboard import create_ksp_dashboard

def main():
    parser = argparse.ArgumentParser(description='Corrected P3R Model Inference')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--model_name', type=str, required=True, help='CodePTM model name')
    parser.add_argument('--stage1_model', type=str, default='models/stage1_symbolic_classifier.pth')
    parser.add_argument('--stage2_model', type=str, default='models/stage2_p3r_model.pth')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    
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
    print(f"Using CodePTM: {args.model_name}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading test dataset...")
    test_df = pd.read_csv(args.test_data)
    print(f"Test samples: {len(test_df)}")
    
    if 'neuro' not in test_df.columns:
        print("Extracting neurosymbolic features...")
        extractor = NeurosymbolicFeatureExtractor()
        test_df = extractor.process_dataset(test_df)
    
    print("Loading P3R model architecture...")
    try:
        model = UniversalP3RModel(args.model_name).to(device)
        
        print("Loading Stage 1 CompactSymbolicClassifier...")
        stage1_classifier = CompactSymbolicClassifier(model.embed_dim, 2).to(device)
        stage1_checkpoint = torch.load(args.stage1_model, map_location=device)
        stage1_classifier.load_state_dict(stage1_checkpoint)
        stage1_classifier.eval()
        
        for param in stage1_classifier.parameters():
            param.requires_grad = False
        
        model.classifier = stage1_classifier
        
        print("Loading Stage 2 P3R components...")
        stage2_checkpoint = torch.load(args.stage2_model, map_location=device)
        model.load_state_dict(stage2_checkpoint, strict=False)
        model.eval()
        
        trainable_params, total_params = model.count_parameters()
        print(f"Model loaded - Trainable: {trainable_params:,} | Total: {total_params:,}")
        print(f"Parameter Efficiency: {trainable_params/total_params:.1%}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Running inference with corrected P3R architecture...")
    collate_fn = create_collate_fn(model.tokenizer)
    test_dataset = CodeDataset(test_df, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
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
                
                logits = model.predict(chunks, full_code, attention_mask)
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    print("Calculating metrics...")
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    print("\n" + "="*50)
    print("CORRECTED P3R INFERENCE RESULTS")
    print("="*50)
    print("Architecture: Frozen CodePTM + Frozen Stage1 + Trainable P3R")
    print("="*50)
    print_metrics_summary(metrics)
    
    print("\nRunning aerospace mission evaluation...")
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
        
        print(f"\nAerospace Metrics:")
        print(f"Space DIT Score: {metrics['space_dit_score']:.4f}")
        print(f"KSP Orbital Efficiency: {metrics['ksp_orbital_efficiency']:.4f}")
        
    except Exception as e:
        print(f"Warning: Aerospace metrics failed: {e}")
    
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
    
    stage1_params = sum(p.numel() for p in stage1_classifier.parameters())
    p3r_params = trainable_params - stage1_params
    
    results_summary = {
        'architecture': 'Corrected P3R Two-Stage',
        'model_name': args.model_name,
        'test_samples': len(test_df),
        'accuracy': float(metrics['acc']),
        'f1_score': float(metrics['f1']),
        'precision': float(metrics['prec']),
        'recall': float(metrics['rec']),
        'roc_auc': float(metrics['roc_auc']),
        'space_dit_score': float(metrics.get('space_dit_score', 0)),
        'parameter_breakdown': {
            'frozen_backbone': total_params - trainable_params,
            'frozen_stage1_classifier': stage1_params,
            'trainable_p3r_components': p3r_params,
            'total_parameters': total_params,
            'parameter_efficiency': f"{trainable_params/total_params:.1%}"
        }
    }
    
    summary_path = os.path.join(args.output_dir, "corrected_p3r_results.json")
    try:
        import json
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Results summary saved: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save summary: {e}")
    
    print(f"\nResults exported to: {args.output_dir}")
    print("Corrected P3R inference completed!")

if __name__ == "__main__":

    main()
