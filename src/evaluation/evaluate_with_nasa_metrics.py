import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.space_metrics import NASAMetricsCalculator, SpacecraftSimulator, export_nasa_results
from visualization.dashboard import create_nasa_dashboard

def evaluate_with_nasa_metrics(model, test_loader, device, output_dir="results"):
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Running model evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if 'chunks' in batch:
                chunks = batch['chunks'].to(device)
                full_code = batch['full_code'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(chunks, full_code, attention_mask)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask)
            
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    print("\nCalculating NASA mission-critical metrics...")
    nasa_calculator = NASAMetricsCalculator()
    
    metrics = nasa_calculator.compute_comprehensive_metrics(
        all_labels, all_predictions, all_probabilities
    )
    
    print("\nRunning spacecraft mission simulation...")
    simulator = SpacecraftSimulator()
    
    mission_states = simulator.run_mission_simulation(
        all_predictions, all_probabilities, all_labels, duration_hours=720
    )
    
    mission_success = simulator.calculate_mission_success_probability(mission_states)
    metrics['mission_success_probability'] = mission_success
    
    print("\nExporting results...")
    results_dir = export_nasa_results(metrics, mission_states, output_dir)
    
    mission_csv = os.path.join(results_dir, sorted([f for f in os.listdir(results_dir) if f.startswith('mission_timeline')])[-1])
    
    print("Creating NASA dashboard visualization...")
    dashboard_path = os.path.join(output_dir, "nasa_dashboard.png")
    create_nasa_dashboard(metrics, mission_csv, dashboard_path)
    
    print(f"\n{'='*60}")
    print("NASA SOFTWARE VALIDATION RESULTS")
    print(f"{'='*60}")
    
    print("\nDetection Performance:")
    for metric, value in metrics['detection_metrics'].items():
        print(f"  {metric.capitalize():15s}: {value:.4f}")
    
    print("\nVulnerability Analysis:")
    for severity, count in metrics['vulnerability_counts'].items():
        print(f"  {severity.capitalize():15s}: {count}")
    
    print("\nNASA Mission Metrics:")
    print(f"  Risk Score:     {metrics['mission_risk_score']:.4f}")
    print(f"  Assurance:      {metrics['software_assurance_level']}")
    print(f"  Readiness:      {metrics['mission_readiness_score']:.4f}")
    print(f"  Success Prob:   {metrics['mission_success_probability']:.4f}")
    
    if mission_states:
        final = mission_states[-1]
        print("\nFinal Spacecraft State:")
        print(f"  Altitude:       {final['altitude_km']:.2f} km")
        print(f"  Velocity:       {final['velocity_mps']:.2f} m/s")
        print(f"  Power:          {final['power_w']:.2f} W")
        print(f"  Temperature:    {final['temperature_k']:.2f} K")
        print(f"  Link Avail:     {final['link_availability']:.2%}")
    
    print(f"\n{'='*60}")
    print(f"Results exported to: {results_dir}")
    print(f"Dashboard saved to: {dashboard_path}")
    print(f"{'='*60}\n")
    
    model.train()
    return metrics, mission_states

def evaluate_model(model_path, test_data_path, model_class, tokenizer, device, output_dir="results"):
    print(f"Loading model from {model_path}...")
    
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    
    from torch.utils.data import DataLoader
    
    if hasattr(model, 'tokenizer'):
        from training.dataset import CodeDataset
        test_dataset = CodeDataset(test_df, model.tokenizer)
        
        def collate_fn(batch):
            max_chunks = max(item['chunks'].size(0) for item in batch)
            padded_chunks = []
            full_codes = []
            attention_masks = []
            labels = []
            
            for item in batch:
                chunks = item['chunks']
                if chunks.size(0) < max_chunks:
                    padding = torch.full((max_chunks - chunks.size(0), chunks.size(1)), 
                                        model.tokenizer.pad_token_id, dtype=torch.long)
                    chunks = torch.cat([chunks, padding], dim=0)
                padded_chunks.append(chunks)
                full_codes.append(item['full_code'])
                attention_masks.append(item['attention_mask'])
                labels.append(item['label'])
            
            return {
                'chunks': torch.stack(padded_chunks),
                'full_code': torch.stack(full_codes),
                'attention_mask': torch.stack(attention_masks),
                'label': torch.stack(labels)
            }
        
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    else:
        from training.dataset import SimpleCodeDataset
        test_dataset = SimpleCodeDataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    metrics, mission_states = evaluate_with_nasa_metrics(
        model, test_loader, device, output_dir
    )
    
    return metrics, mission_states
