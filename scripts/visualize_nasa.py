import argparse
import os
import sys
import json



from src.visualization.dashboard import create_nasa_dashboard

def main():
    parser = argparse.ArgumentParser(description='NASA Aerospace Metrics Visualization')
    parser.add_argument('--data', type=str, required=True, help='Path to mission timeline CSV')
    parser.add_argument('--metrics', type=str, required=True, help='Path to NASA metrics JSON')
    parser.add_argument('--output', type=str, required=True, help='Output path for dashboard PNG')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return
    
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        return
    
    print(f"Loading metrics from: {args.metrics}")
    with open(args.metrics, 'r') as f:
        metrics_data = json.load(f)
    
    metrics = {
        'detection_metrics': metrics_data['detection_performance'],
        'vulnerability_counts': metrics_data['vulnerability_analysis'],
        'mission_risk_score': metrics_data['nasa_mission_metrics']['risk_score'],
        'software_assurance_level': metrics_data['nasa_mission_metrics']['assurance_level'],
        'mission_readiness_score': metrics_data['nasa_mission_metrics']['readiness_score']
    }
    
    print(f"Loading mission data from: {args.data}")
    print(f"Creating NASA dashboard...")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    dashboard_path = create_nasa_dashboard(metrics, args.data, args.output)
    
    print(f"\n{'='*60}")
    print(f"Dashboard created successfully!")
    print(f"Saved to: {dashboard_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
