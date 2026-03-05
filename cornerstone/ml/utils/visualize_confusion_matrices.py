#!/usr/bin/env python3
"""
Visualize confusion matrices from ensemble evaluation results.

This script loads existing ensemble evaluation results and generates
confusion matrix visualizations for each time window.
"""

import json
import os
import argparse
from plot_ensemble_metrics import plot_confusion_matrices
from path_manager import PathManager
from config import TASK_CONFIG, DATA_CONFIG

def load_ensemble_results(metrics_file: str) -> dict:
    """Load ensemble evaluation results from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Metrics file not found: {metrics_file}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in metrics file: {metrics_file}")
        return {}

def visualize_confusion_matrices(metrics_file: str = None, output_dir: str = None) -> None:
    """
    Visualize confusion matrices for all windows in ensemble evaluation results.
    
    Args:
        metrics_file: Path to ensemble metrics JSON file (optional)
        output_dir: Output directory for plots (optional)
    """
    # Initialize path manager
    path_manager = PathManager()
    
    # Use provided metrics file or default
    if metrics_file is None:
        metrics_file = path_manager.get_metrics_path(f"{TASK_CONFIG['name']}_ensemble_metrics.json")
    
    # Use provided output directory or default
    if output_dir is None:
        output_dir = path_manager.get_plot_path("")
    
    print(f"📁 Loading ensemble results from: {metrics_file}")
    print(f"📁 Saving confusion matrices to: {output_dir}")
    
    # Load ensemble results
    ensemble_results = load_ensemble_results(metrics_file)
    if not ensemble_results:
        print("❌ No ensemble results found. Please run ensemble evaluation first.")
        return
    
    # Generate confusion matrices for each window
    for window in ensemble_results.keys():
        print(f"\n📊 Generating confusion matrices for {window} window...")
        
        window_results = ensemble_results[window]
        
        # Check if results contain confusion matrix data
        if 'individual_folds' not in window_results or 'ensemble_metrics' not in window_results:
            print(f"⚠️ Skipping {window}: Missing required data structure")
            continue
        
        # Check if ensemble metrics contain confusion matrix
        if 'confusion_matrix' not in window_results['ensemble_metrics']:
            print(f"⚠️ Skipping {window}: No ensemble confusion matrix data")
            continue
        
        # Generate plot
        plot_filename = f"{TASK_CONFIG['name']}_{window}_confusion_matrices.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        try:
            plot_confusion_matrices(window_results, plot_path, window)
            print(f"✅ Confusion matrices saved to: {plot_path}")
        except Exception as e:
            print(f"❌ Error generating confusion matrices for {window}: {e}")
    
    print(f"\n🎉 Confusion matrix visualization completed!")
    print(f"📁 Check the plots directory: {output_dir}")

def main():
    """Main function to handle command line arguments and run visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize confusion matrices from ensemble evaluation results"
    )
    parser.add_argument(
        "--metrics-file", 
        type=str, 
        help="Path to ensemble metrics JSON file (default: auto-detect)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for plots (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Initialize path manager
    path_manager = PathManager()
    
    # Print configuration
    print("🔧 Confusion Matrix Visualization Configuration:")
    print(f"   Task: {TASK_CONFIG['name']}")
    print(f"   Dataset: {path_manager.get_dataset_name()}")
    print(f"   Model: {path_manager.get_model_type()}")
    
    # Run visualization
    visualize_confusion_matrices(args.metrics_file, args.output_dir)

if __name__ == "__main__":
    main() 