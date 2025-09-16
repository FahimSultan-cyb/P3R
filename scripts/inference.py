import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.models.p3r_model import P3RHeadGateModel
from src.data.dataset import CodeDataset, create_collate_fn
from src.evaluation.metrics import calculate_comprehensive_metrics, print_metrics_summary
from src.evaluation.space_metrics import SpaceMissionEvaluator, KSPMissionSimulator
from src.visualization.dashboard import create_ksp_dashboard

def main():
    parser = argparse.ArgumentParser(description='P3R Model Inference')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--model_path', type=str, default='models/p3r_headgate_model1.pth')
    parser.add_argument('--classifier_path', type=str, default='models/symbolic_classifier1n.pth')
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading test dataset...")
    test_df = pd.read_csv(args.test_data)
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:\n{test_df['label'].value_counts()}")
    
    print("\nInitializing model...")
    model = P3RHeadGateModel().to(device)
    
    print("Loading trained weights...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    trainable_params, total_params = model.count_parameters()
    print(f"Trainable: {trainable_params:,} | Total: {total_params:,}")
    print(f"Efficiency: {trainable_params/total_params:.1%}")
    
    collate_fn = create_collate_fn(model.tokenizer)
    test_dataset = CodeDataset(test_df, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    print("\nRunning inference...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    print("\n=== INFERENCE RESULTS ===")
    print_metrics_summary(metrics)
    
    print("\nExporting results...")
    ksp_csv_path = ksp_sim.export_data(args.output_dir)
    
    if ksp_csv_path:
        print("Creating mission dashboard...")
        dashboard_path = os.path.join(args.output_dir, "mission_dashboard.png")
        create_ksp_dashboard(ksp_csv_path, dashboard_path)
        print(f"Dashboard saved: {dashboard_path}")
    
    results_summary = {
        'test_samples': len(test_df),
        'accuracy': metrics['acc'],
        'f1_score': metrics['f1'],
        'space_dit_score': metrics['space_dit_score'],
        'ksp_orbital_efficiency': metrics['ksp_orbital_efficiency'],
        'parameter_efficiency': f"{trainable_params/total_params:.1%}"
    }
    
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults exported to: {args.output_dir}")
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()