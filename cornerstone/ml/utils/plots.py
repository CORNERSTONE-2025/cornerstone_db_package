# utils/plots.py
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from .config import DATA_CONFIG
import json
from glob import glob
    
def plot_boxplot(summary_metrics, metric_name="TSS", save_path=None):
    """
    Plot boxplots of metrics across folds for each window.
    
    Args:
        summary_metrics: Dictionary containing metrics for each window
        metric_name: Name of the metric to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    
    for window, window_data in summary_metrics.items():
        fold_metrics = []
        for fold_data in window_data["folds"]:
            # Try different metric locations
            value = None
            if "val_metrics_best_epoch" in fold_data:
                value = fold_data["val_metrics_best_epoch"].get(metric_name)
            elif "test_metrics" in fold_data:
                value = fold_data["test_metrics"].get(metric_name)
            elif "cv_metrics" in fold_data:
                value = fold_data["cv_metrics"].get(metric_name)
            
            if value is not None:
                fold_metrics.append(value)
                
        if fold_metrics:
            data.append(fold_metrics)
            # Use descriptive window name if available
            window_desc = DATA_CONFIG().get("window_descriptions", {}).get(window, window)
            labels.append(window_desc)
    
    plt.boxplot(data, labels=labels)
    plt.title(f"{metric_name} Distribution Across Folds")
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_curves(summary_metrics, save_path=None):
    os.makedirs("results", exist_ok=True)
    for window, data in summary_metrics.items():
        plt.figure(figsize=(10, 6))
        plotted = False
        for fold_data in data["folds"]:
            fold = fold_data["fold"]
            train_loss = fold_data.get("train_history", [])
            val_loss = fold_data.get("val_history", [])
            if train_loss and val_loss:
                plt.plot(train_loss, label=f"Fold {fold} Train", linestyle="--")
                plt.plot(val_loss, label=f"Fold {fold} Val")
                plotted = True
        if plotted:
            # Use descriptive window name if available
            window_desc = DATA_CONFIG().get("window_descriptions", {}).get(window, window)
            plt.title(f"Training and Validation Loss - {window_desc}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_path:
                try:
                    # Create separate file for each window
                    save_dir = os.path.dirname(save_path)
                    save_name = os.path.basename(save_path)
                    name, ext = os.path.splitext(save_name)
                    window_save_path = os.path.join(save_dir, f"{name}_{window}{ext}")
                    plt.savefig(window_save_path, bbox_inches='tight', dpi=300)
                    print(f"Saved training curves for {window} to {window_save_path}")
                except Exception as e:
                    print(f"Error saving plot for {window}: {e}")
                    # Fallback to simple filename
                    fallback_path = f"training_curves_{window}.png"
                    plt.savefig(fallback_path, bbox_inches='tight', dpi=300)
                    print(f"Saved fallback plot for {window} to {fallback_path}")
            else:
                # If no save_path provided, save with window name
                fallback_path = f"training_curves_{window}.png"
                plt.savefig(fallback_path, bbox_inches='tight', dpi=300)
                print(f"Saved training curves for {window} to {fallback_path}")
        plt.close()

def plot_metric_curves(summary_metrics, metric_name="TSS", save_path=None):
    os.makedirs("results", exist_ok=True)
    for window, data in summary_metrics.items():
        plt.figure(figsize=(10, 6))
        plotted = False
        for fold_data in data["folds"]:
            fold = fold_data["fold"]
            metric_curve = fold_data.get("val_metric_history", {}).get(metric_name, [])
            if metric_curve:
                plt.plot(metric_curve, label=f"Fold {fold}")
                plotted = True
        if plotted:
            # Use descriptive window name if available
            window_desc = DATA_CONFIG().get("window_descriptions", {}).get(window, window)
            plt.title(f"Validation {metric_name} over Epochs - {window_desc}")
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_path:
                try:
                    # Create separate file for each window
                    save_dir = os.path.dirname(save_path)
                    save_name = os.path.basename(save_path)
                    name, ext = os.path.splitext(save_name)
                    window_save_path = os.path.join(save_dir, f"{name}_{window}{ext}")
                    plt.savefig(window_save_path, bbox_inches='tight', dpi=300)
                    print(f"Saved {metric_name} curves for {window} to {window_save_path}")
                except Exception as e:
                    print(f"Error saving {metric_name} plot for {window}: {e}")
                    # Fallback to simple filename
                    fallback_path = f"{metric_name}_curves_{window}.png"
                    plt.savefig(fallback_path, bbox_inches='tight', dpi=300)
                    print(f"Saved fallback {metric_name} plot for {window} to {fallback_path}")
            else:
                # If no save_path provided, save with window name
                fallback_path = f"{metric_name}_curves_{window}.png"
                plt.savefig(fallback_path, bbox_inches='tight', dpi=300)
                print(f"Saved {metric_name} curves for {window} to {fallback_path}")
        plt.close()

def plot_test_metrics(summary_metrics, selected_metrics=["TSS", "HSS", "F1"], save_path=None):

    if selected_metrics is None:
        selected_metrics = ["TSS", "HSS", "F1"]

    windows = list(summary_metrics.keys())
    os.makedirs("results", exist_ok=True)

    # Gather values - use test_metrics from the best fold
    data = {}
    for metric in selected_metrics:
        data[metric] = []
        for win in windows:
            # Get test metrics from the best fold
            best_fold_idx = summary_metrics[win].get("best_fold", 0)
            if "folds" in summary_metrics[win] and best_fold_idx < len(summary_metrics[win]["folds"]):
                best_fold = summary_metrics[win]["folds"][best_fold_idx]
                value = best_fold.get("test_metrics", {}).get(metric, 0.0)
            else:
                value = 0.0
            data[metric].append(value)

    x = np.arange(len(windows))
    width = 0.2  # width of each bar
    spacing = width * len(selected_metrics)

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(selected_metrics):
        plt.bar(x + i * width - spacing / 2 + width / 2, data[metric], width, label=metric)

    # Use descriptive window names for x-axis labels
    window_labels = [DATA_CONFIG().get("window_descriptions", {}).get(win, win) for win in windows]
    plt.xticks(x, window_labels)
    plt.title("Test Set Performance Metrics")
    plt.xlabel("Time Window")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def sk_plot_boxplot(summary_metrics,model_name = "ROCKET", metric_name="TSS", save_path=None):
    """
    Plot boxplots of metrics across folds for each window.
    
    Args:
        summary_metrics: Dictionary containing metrics for each window
        metric_name: Name of the metric to plot
        save_path: Path to save the plot
    """
    labels = []
    windows = []

    # Strip the extension so the window/split suffix goes before it
    save_base, ext = os.path.splitext(save_path)
    ext = ext or ".png"

    for window, window_data in summary_metrics.items():

        # These MUST be reset per window
        data = []
        data_test = []
        Cs = []

        for C_fold in window_data:
            C = C_fold["C"]
            Cs.append(C)

            # Collect fold metrics for this C
            fold_vals = [f["val_metrics"][metric_name] for f in C_fold["folds"]]
            fold_vals_test = [f["test_metrics"][metric_name] for f in C_fold["folds"]]

            data.append(fold_vals)
            data_test.append(fold_vals_test)

        # ----- SAVE VALIDATION BOXPLOT FOR THIS WINDOW -----
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=[str(c) for c in Cs])
        plt.title(f"{metric_name} Across Cs (Validation) – Window {window}")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(f"{save_base}_{window}_validation{ext}")
        plt.close()

        # ----- SAVE TEST BOXPLOT FOR THIS WINDOW -----
        plt.figure(figsize=(10, 6))
        plt.boxplot(data_test, labels=[str(c) for c in Cs])
        plt.title(f"{metric_name} Across Cs (Test) – Window {window}")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(f"{save_base}_{window}_test{ext}")
        plt.close()

