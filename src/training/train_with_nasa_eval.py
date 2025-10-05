import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import sys



from src.evaluation.space_metrics import NASAMetricsCalculator, SpacecraftSimulator, export_nasa_results
from src.visualization.dashboard import create_nasa_dashboard

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        
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
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
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
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    nasa_calculator = NASAMetricsCalculator()
    metrics = nasa_calculator.compute_comprehensive_metrics(
        all_labels, all_predictions, all_probabilities
    )
    
    return avg_loss, metrics

def train_model_with_nasa_validation(
    model, 
    train_loader, 
    val_loader,
    device,
    epochs=10,
    learning_rate=2e-5,
    output_dir="checkpoints",
    eval_every_n_epochs=2
):
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    best_readiness = 0.0
    training_history = []
    
    print(f"\n{'='*60}")
    print("TRAINING WITH NASA SOFTWARE VALIDATION")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        if (epoch + 1) % eval_every_n_epochs == 0 or epoch == epochs - 1:
            print("\nRunning NASA validation metrics...")
            
            val_loss, metrics = validate_epoch(model, val_loader, criterion, device)
            
            print(f"\nValidation Loss: {val_loss:.4f}")
            print(f"Detection Metrics:")
            print(f"  Accuracy:  {metrics['detection_metrics']['accuracy']:.4f}")
            print(f"  Precision: {metrics['detection_metrics']['precision']:.4f}")
            print(f"  Recall:    {metrics['detection_metrics']['recall']:.4f}")
            print(f"  F1:        {metrics['detection_metrics']['f1_score']:.4f}")
            
            print(f"\nNASA Mission Metrics:")
            print(f"  Risk Score: {metrics['mission_risk_score']:.4f}")
            print(f"  Assurance:  {metrics['software_assurance_level']}")
            print(f"  Readiness:  {metrics['mission_readiness_score']:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'metrics': metrics
            })
            
            readiness_score = metrics['mission_readiness_score']
            
            if readiness_score > best_readiness:
                best_readiness = readiness_score
                checkpoint_path = os.path.join(output_dir, f"best_model_epoch{epoch+1}.pth")
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'readiness_score': readiness_score
                }, checkpoint_path)
                
                print(f"\n✓ Best model saved (Readiness: {readiness_score:.4f})")
    
    final_checkpoint = os.path.join(output_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history
    }, final_checkpoint)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Readiness Score: {best_readiness:.4f}")
    print(f"Final model saved to: {final_checkpoint}")
    print(f"{'='*60}\n")
    
    return training_history

def full_nasa_evaluation(model, test_loader, device, output_dir="results/final_eval"):
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\nRunning comprehensive NASA evaluation...")
    
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
    
    import numpy as np
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    nasa_calculator = NASAMetricsCalculator()
    metrics = nasa_calculator.compute_comprehensive_metrics(
        all_labels, all_predictions, all_probabilities
    )
    
    simulator = SpacecraftSimulator()
    mission_states = simulator.run_mission_simulation(
        all_predictions, all_probabilities, all_labels, duration_hours=720
    )
    
    mission_success = simulator.calculate_mission_success_probability(mission_states)
    metrics['mission_success_probability'] = mission_success
    
    results_dir = export_nasa_results(metrics, mission_states, output_dir)
    
    mission_csv = os.path.join(
        results_dir, 
        sorted([f for f in os.listdir(results_dir) if f.startswith('mission_timeline')])[-1]
    )
    
    dashboard_path = os.path.join(output_dir, "nasa_dashboard.png")
    create_nasa_dashboard(metrics, mission_csv, dashboard_path)
    
    print(f"\n{'='*70}")
    print("FINAL NASA SOFTWARE VALIDATION RESULTS")
    print(f"{'='*70}")
    
    print("\nDetection Performance:")
    for metric, value in metrics['detection_metrics'].items():
        print(f"  {metric.capitalize():15s}: {value:.4f}")
    
    print("\nVulnerability Distribution:")
    for severity, count in metrics['vulnerability_counts'].items():
        print(f"  {severity.capitalize():15s}: {count}")
    
    print("\nNASA Mission-Critical Metrics:")
    print(f"  Mission Risk Score:        {metrics['mission_risk_score']:.4f}")
    print(f"  Software Assurance Level:  {metrics['software_assurance_level']}")
    print(f"  Mission Readiness Score:   {metrics['mission_readiness_score']:.4f}")
    print(f"  Mission Success Probability: {metrics['mission_success_probability']:.4f}")
    
    if mission_states:
        final = mission_states[-1]
        print("\nFinal Spacecraft State:")
        print(f"  Orbital Altitude:     {final['altitude_km']:.2f} km")
        print(f"  Orbital Velocity:     {final['velocity_mps']:.2f} m/s")
        print(f"  Power Available:      {final['power_w']:.2f} W")
        print(f"  Power Efficiency:     {final['power_efficiency']:.2%}")
        print(f"  Battery Health:       {final['battery_health']:.2%}")
        print(f"  Temperature:          {final['temperature_k']:.2f} K")
        print(f"  Thermal Margin:       {final['thermal_margin_k']:.2f} K")
        print(f"  Signal Strength:      {final['signal_strength_db']:.2f} dB")
        print(f"  Data Rate:            {final['data_rate_mbps']:.2f} Mbps")
        print(f"  Link Availability:    {final['link_availability']:.2%}")
    
    print(f"\n{'='*70}")
    print(f"Complete results exported to: {results_dir}")
    print(f"Dashboard visualization: {dashboard_path}")
    print(f"{'='*70}\n")
    
    return metrics, mission_states
