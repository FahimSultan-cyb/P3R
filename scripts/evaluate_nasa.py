import torch
import pandas as pd
import argparse
import os
import sys



from src.models.p3r_model import P3RHeadGateModel
from src.data.dataset import CodeDataset
from torch.utils.data import DataLoader
from src.evaluation.space_metrics import NASAMetricsCalculator, SpacecraftSimulator, export_nasa_results

def main():
    parser = argparse.ArgumentParser(description='NASA Software Validation Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output', type=str, default='results/nasa_analysis', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("\nTo get a trained model, you need to:")
        print("1. Train a model using: python src/training/train.py")
        print("   OR")
        print("2. Download pretrained model from your releases/checkpoints")
        return
    
    print(f"Loading model from: {args.model}")
    
    model = P3RHeadGateModel().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"Loading test data from: {args.data}")
    test_df = pd.read_csv(args.data)
    print(f"Test samples: {len(test_df)}")
    
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
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    print("\nRunning NASA metrics evaluation...")
    model.eval()
    
    import numpy as np
    from tqdm import tqdm
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            chunks = batch['chunks'].to(device)
            full_code = batch['full_code'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(chunks, full_code, attention_mask)
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
    results_dir = export_nasa_results(metrics, mission_states, args.output)
    
    print(f"\n{'='*70}")
    print("NASA SOFTWARE VALIDATION RESULTS")
    print(f"{'='*70}")
    
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
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
