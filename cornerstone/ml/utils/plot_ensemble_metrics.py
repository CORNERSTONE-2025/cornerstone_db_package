import json
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.path_manager import PathManager
from config import TASK_CONFIG, DATA_CONFIG

def plot_ensemble_test_metrics(metrics: list, save_path: str, ensemble_metrics: dict = None) -> None:
    """Plot ensemble test metrics with true fold-based boxplots."""
    if ensemble_metrics is None:
        # Try to load metrics from file
        try:
            path_manager = PathManager()
            metrics_path = os.path.join(path_manager.task_dir, "metrics", f"{TASK_CONFIG['name']}_ensemble_simple_transformer.json")
            with open(metrics_path, 'r') as f:
                ensemble_metrics = json.load(f)
        except FileNotFoundError:
            print(f"Error: Ensemble metrics file not found at {metrics_path}")
            print("Available metrics directory:", os.path.join(path_manager.task_dir, "metrics"))
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file at {metrics_path}")
            return
    
    # Set style
    plt.style.use('bmh')
    
    # Create a single figure with adjusted size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for each metric
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']  # Green, Blue, Red, Orange, Purple, Teal
    
    # Calculate spacing between metrics
    n_metrics = len(metrics)
    metric_spacing = 0.3  # Increased spacing for better separation
    
    # Prepare data for boxplots
    boxplot_data = []
    positions = []
    labels = []
    ensemble_values = []  # Store ensemble values for overlay
    
    # Extract data for each window and metric
    windows = list(ensemble_metrics.keys())
    for i, window in enumerate(windows):
        window_data = ensemble_metrics[window]
        if not isinstance(window_data, dict) or 'individual_folds' not in window_data:
            continue
            
        for j, metric in enumerate(metrics):
            # Extract fold values for this metric
            fold_values = []
            for fold_data in window_data['individual_folds']:
                if metric in fold_data:
                    fold_values.append(fold_data[metric])
            
            # Only add data if we have fold values
            if fold_values:
                boxplot_data.append(fold_values)
                # Position calculation: base position + offset for metric
                # Center the metrics around each window position
                pos = i + 1 + (j - (n_metrics-1)/2) * metric_spacing
                positions.append(pos)
                labels.append(f"{metric}")
                
                # Get ensemble value for overlay
                ensemble_val = None
                if 'ensemble_metrics' in window_data and metric in window_data['ensemble_metrics']:
                    ensemble_val = window_data['ensemble_metrics'][metric]
                ensemble_values.append(ensemble_val)
    
    if not boxplot_data:
        print("\nNo fold data available for plotting")
        return
    
    # Create boxplot with increased width
    box = ax.boxplot(boxplot_data, positions=positions, widths=0.25,
                    patch_artist=True, showfliers=True, showmeans=True)
    
    # Customize boxplot appearance with different colors for each metric
    for i, patch in enumerate(box['boxes']):
        metric_idx = i % n_metrics
        patch.set_facecolor(colors[metric_idx])
        patch.set_alpha(0.7)
    
    # Add ensemble values as horizontal lines or triangle markers
    for i, (data, pos, ensemble_val) in enumerate(zip(boxplot_data, positions, ensemble_values)):
        if ensemble_val is not None:
            # Add horizontal line for ensemble value (spanning the boxplot width)
            line_width = 0.2  # Width of the horizontal line
            ax.plot([pos - line_width, pos + line_width], [ensemble_val, ensemble_val], 
                   color='black', linestyle='--', linewidth=2, alpha=0.8)
            # Add triangle marker
            ax.scatter(pos, ensemble_val, color='black', marker='^', s=100, zorder=5, 
                      label='Ensemble' if i == 0 else "")
    
    # Add fold mean values as text
    for i, data in enumerate(boxplot_data):
        mean_val = np.mean(data)
        # Position text higher above the boxplot
        ax.text(positions[i], mean_val + 0.02, f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize plot
    ax.set_title('Fold-based Metrics Distribution by Window', fontsize=14, pad=15)
    ax.set_xlabel('Window', fontsize=12, labelpad=10)
    ax.set_ylabel('Score', fontsize=12, labelpad=10)
    
    # Set x-axis ticks and labels with descriptive window names
    window_labels = [DATA_CONFIG["window_descriptions"].get(win, win) for win in windows]
    ax.set_xticks(range(1, len(windows) + 1))
    ax.set_xticklabels(window_labels)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add a light background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Add legend for metrics and ensemble
    metric_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7) 
                     for i in range(len(metrics))]
    legend_elements = metric_patches + [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Ensemble')]
    legend_labels = metrics + ['Ensemble']
    
    ax.legend(legend_elements, legend_labels, 
             title='Metrics', loc='upper right', bbox_to_anchor=(1.02, 1))
    
    # DEBUG: Print ensemble and boxplot TSS values for each window
    for i, window in enumerate(windows):
        if 'ensemble_metrics' in ensemble_metrics[window]:
            print(f'Window: {window}')
            print('  Ensemble TSS:', ensemble_metrics[window]['ensemble_metrics'].get('TSS'))
            print('  Boxplot TSS:', [fold.get('TSS') for fold in ensemble_metrics[window]['individual_folds']])

    # Adjust layout to prevent text overlap
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(ensemble_results: dict, save_path: str, window: str) -> None:
    """
    Plot confusion matrices for ensemble evaluation.
    
    Args:
        ensemble_results: Dictionary containing ensemble evaluation results
        save_path: Path to save the confusion matrix plot
        window: Time window being evaluated
    """
    # Set style
    plt.style.use('bmh')
    
    # Create subplots for individual folds and ensemble
    n_folds = len(ensemble_results.get('individual_folds', []))
    n_cols = min(3, n_folds + 1)  # Max 3 columns
    n_rows = (n_folds + 1 + n_cols - 1) // n_cols  # Calculate rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot individual fold confusion matrices
    for fold_idx in range(n_folds):
        if fold_idx < len(axes_flat):
            ax = axes_flat[fold_idx]
            
            # Get confusion matrix data for this fold
            fold_data = ensemble_results['individual_folds'][fold_idx]
            if 'confusion_matrix' in fold_data:
                cm = fold_data['confusion_matrix']
                threshold = ensemble_results['thresholds']['fold_thresholds'][fold_idx]
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['No Flare', 'Flare'],
                           yticklabels=['No Flare', 'Flare'])
                ax.set_title(f'Fold {fold_idx} (threshold: {threshold:.3f})', fontsize=10)
                ax.set_xlabel('Predicted', fontsize=9)
                ax.set_ylabel('Actual', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No CM data\nfor Fold {fold_idx}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Fold {fold_idx}', fontsize=10)
    
    # Plot ensemble confusion matrix
    ensemble_idx = min(n_folds, len(axes_flat) - 1)
    if ensemble_idx < len(axes_flat):
        ax = axes_flat[ensemble_idx]
        
        # Get ensemble confusion matrix data
        ensemble_data = ensemble_results['ensemble_metrics']
        if 'confusion_matrix' in ensemble_data:
            cm = ensemble_data['confusion_matrix']
            threshold = ensemble_results['thresholds']['ensemble_threshold']
            strategy = ensemble_results['thresholds']['strategy']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                       xticklabels=['No Flare', 'Flare'],
                       yticklabels=['No Flare', 'Flare'])
            ax.set_title(f'Ensemble ({strategy})\nthreshold: {threshold:.3f}', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No ensemble\nCM data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ensemble', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_folds + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Confusion Matrices - {window} Window', fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

