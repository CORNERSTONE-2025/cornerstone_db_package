# utils/training.py
import os
import json
import copy
import torch
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix 
from torch.utils.data import DataLoader
from scipy.special import expit as sigmoid
from .metrics import compute_metrics, convert_to_serializable
from .datasets import FlareHDF5Dataset
from .losses import create_loss_function
from .threshold_optimizer import create_threshold_optimizer
from .calibration import create_calibrator, prior_adjust
from .eval_io import collect_logits, reliability_ece, save_predictions_csv, expected_confusion
from .metrics_manager import MetricsManager

PT_MODELS = [
    "LSTM", "INCEPTIONTIME", "TRANSFORMER", "ONEFITSALL",
    "Autoformer", "DLinear", "FEDformer",
    "LightTS", "Nonstationary_Transformer",
    "Pyraformer", "Reformer", "TimesNet", "Informer"
]

SK_MODELS = [
    "ROCKET", "MINIROCKET", "XGBOOST"
]

ALL_MODELS = PT_MODELS + SK_MODELS
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, device="cuda", patience=5, lr_scheduler_config=None, threshold_config=None, early_stopping_config=None):
    """
    Trains a model with early stopping based on validation loss and learning rate scheduling.
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: Optimizer instance (e.g., Adam).
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        num_epochs: Maximum number of training epochs.
        device: Device to train on ("cuda" or "cpu").
        patience: Number of epochs to wait for improvement before early stopping.
        lr_scheduler_config: Dictionary containing learning rate scheduler configuration.
        threshold_config: Dictionary containing threshold optimization configuration.

    Returns:
        model: Trained model (with best validation loss).
        metrics: Validation metrics from the last evaluated epoch (stop epoch).
        train_history: List of training losses per epoch.
        val_history: List of validation losses per epoch.
        val_metric_history: Validation metrics tracked per epoch.
        best_epoch: Epoch where the best model was found.
        lr_history: List of learning rates used during training.
        threshold_history: List of optimal thresholds per epoch (if enabled).
        best_threshold: Optimal threshold at the best epoch (if enabled).
        monitor_history: List of monitored values per epoch.
        monitor_name_history: List of monitor names per epoch (for auditing).
        metrics_best_epoch: Validation metrics at the best_epoch (from history).

    """
    
    # Initialize early stopping configuration
    if early_stopping_config is None:
        early_stopping_config = {"monitor": "val_loss", "mode": "min", "min_delta": 0.0, "patience": patience}
    
    # Helper function for safe AUC computation 
    from sklearn.metrics import roc_auc_score 
    def safe_auc(y_true, y_scores):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return None
        return roc_auc_score(y_true, y_scores)
    
    # Extract early stopping parameters
    monitor = early_stopping_config.get("monitor", "val_loss").lower()
    mode = early_stopping_config.get("mode", "min").lower()
    min_delta = float(early_stopping_config.get("min_delta", 0.0))
    es_patience = int(early_stopping_config.get("patience", patience))
    
    model = model.to(device)
    best_model_state = None
    best_epoch = -1
    best_threshold = 0.5  # Default threshold
    wait = 0  # counter for early stopping
    
    # Initialize monitor tracking
    best_monitor = float("inf") if mode == "min" else -float("inf")
    train_history, val_history = [], []
    val_metric_history = {"TSS": [], "F1": [], "AUC": [], "MCC": [], "HSS": []}
    lr_history = []
    threshold_history = []
    monitor_history = []
    monitor_name_history = []  # Track which metric actually drove scheduling/ES
    
    # Initialize threshold optimizer if enabled
    threshold_optimizer = None
    if threshold_config and threshold_config.get("enabled", False):
        threshold_optimizer = create_threshold_optimizer(
            n_thresholds=threshold_config.get("n_thresholds", 100),
            threshold_range=tuple(threshold_config.get("threshold_range", [0.0, 1.0]))
        )
        print(f"Threshold optimization enabled - optimizing for {threshold_config.get('metric', 'TSS')}")
    
    # Initialize learning rate scheduler if config is provided
    scheduler = None
    if lr_scheduler_config:
        if lr_scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_config["mode"],
                factor=lr_scheduler_config["factor"],
                patience=lr_scheduler_config["patience"],
                min_lr=lr_scheduler_config["min_lr"]
            )
        elif lr_scheduler_config["type"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=lr_scheduler_config.get("T_max", num_epochs),
                eta_min=lr_scheduler_config["eta_min"]
            )
    
    for epoch in range(num_epochs):
        # ------------------------------
        # Training Phase
        # ------------------------------
        model.train()
        epoch_train_losses = []
        
        # Set epoch for curriculum learning if applicable
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        
        for X, mask, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X, mask, y = X.to(device), mask.to(device), y.to(device) 
            optimizer.zero_grad()  
            logits = model(X, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        train_history.append(float(np.mean(epoch_train_losses)))

        # ------------------------------
        # Validation Phase
        # ------------------------------
        model.eval()
        epoch_val_losses, y_true, y_probs = [], [], []
        with torch.no_grad():
            for X, mask, y in val_loader:
                X, mask, y = X.to(device), mask.to(device), y.to(device)
                logits = model(X, mask)
                loss = criterion(logits, y)
                epoch_val_losses.append(loss.item())
                
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                probs = torch.nan_to_num(probs, nan=0.0).cpu().numpy()
                y_true.extend(y.cpu().numpy())
                y_probs.extend(probs)
        
        val_loss = float(np.mean(epoch_val_losses))
        val_history.append(val_loss)

        # ------------------------------
        # Threshold Optimization
        # ------------------------------
        # Force threshold optimization when monitoring thresholded metrics
        monitor_is_thresholded = monitor in {"tss", "f1", "mcc", "hss"}
        do_thr_opt_this_epoch = (
            threshold_optimizer and
            (threshold_config.get("optimize_every_epoch", False) or monitor_is_thresholded or epoch == num_epochs - 1)
        )
        
        # Convert to numpy arrays and normalize shapes
        y_true_np = np.asarray(y_true).ravel()
        y_probs_np = np.asarray(y_probs).ravel()
        
        current_threshold = 0.5  # Default threshold
        if do_thr_opt_this_epoch:
            
            # Optimize threshold
            metric_kwargs = {}
            if threshold_config.get("metric") == "precision_at_recall":
                metric_kwargs["min_recall"] = threshold_config.get("min_recall", 0.8)
            elif threshold_config.get("metric") == "f1_at_precision":
                metric_kwargs["min_precision"] = threshold_config.get("min_precision", 0.8)
            
            threshold_result = threshold_optimizer.optimize_threshold(
                y_true_np, y_probs_np, 
                metric=threshold_config.get("metric", "TSS"),
                **metric_kwargs
            )
            
            current_threshold = threshold_result["optimal_threshold"]
            optimal_metrics = threshold_result["optimal_metrics"]
            
            print(f"🎯 Epoch {epoch+1} - Optimal threshold: {current_threshold:.4f} "
                  f"(metric: {threshold_result['optimal_metric_value']:.4f})")
            
            # Use optimal metrics for this epoch
            metrics = optimal_metrics
        else:
            # Use default threshold of 0.5
            y_pred = (y_probs_np >= current_threshold).astype(int)
            metrics = compute_metrics(y_true_np, y_pred)
        
        # Ensure AUC is always available for logging/history
        auc_epoch = safe_auc(y_true_np, y_probs_np)
        metrics["AUC"] = float(auc_epoch) if auc_epoch is not None else float("nan")
        
        threshold_history.append(current_threshold)

        # ------------------------------
        # Compute Validation Metrics
        # ------------------------------
        # Add debugging information
        y_pred = (y_probs_np >= current_threshold).astype(int)
        print(f"📊 Debug - Pos samples: {sum(y_true_np)}/{len(y_true_np)} = {sum(y_true_np)/len(y_true_np)*100:.1f}%")
        print(f"📊 Debug - Pred pos: {sum(y_pred)}/{len(y_pred)} = {sum(y_pred)/len(y_pred)*100:.1f}%")
        print(f"📊 Debug - Prob range: [{np.min(y_probs_np):.4f}, {np.max(y_probs_np):.4f}], mean: {np.mean(y_probs_np):.4f}")
        print(f"📊 Debug - Threshold: {current_threshold:.4f}")
        #########################################################
        # Add curriculum phase information
        if hasattr(criterion, 'current_epoch'):
            if criterion.current_epoch < criterion.warmup_epochs:
                phase = "BCE Only"
            elif criterion.current_epoch < criterion.warmup_epochs + criterion.transition_epochs:
                progress = (criterion.current_epoch - criterion.warmup_epochs) / criterion.transition_epochs
                focal_weight = progress * 0.3
                phase = f"Transition ({focal_weight:.1%} focal)"
            else:
                phase = "Blend (70% BCE, 30% focal)"
            print(f"📚 Curriculum Phase: {phase}")
        #########################################################
        for key in val_metric_history:
            val_metric_history[key].append(float(metrics[key]))
            
        # ------------------------------
        # Compute Monitor Value for Early Stopping
        # ------------------------------
        # Decide what to monitor this epoch
        if monitor == "val_loss":
            monitor_value = val_loss
        elif monitor == "auc":
            monitor_value = safe_auc(y_true_np, y_probs_np)  # uses probabilities, not thresholded labels
        else:
            # metrics dict already has TSS/F1/MCC/HSS for this epoch
            # Use lower-case keys to be robust
            key_map = {"tss": "TSS", "f1": "F1", "mcc": "MCC", "hss": "HSS"}
            monitor_value = metrics.get(key_map.get(monitor, ""), None)

        # Fall back only for this epoch (don't mutate global settings)
        eff_monitor = monitor
        eff_mode = mode
        used_fallback = False
        if monitor_value is None:
            monitor_value = val_loss
            eff_monitor = "val_loss"
            eff_mode = "min"
            used_fallback = True
        
        # Track monitor history
        monitor_history.append(float(monitor_value))
        monitor_name_history.append(eff_monitor)  # Track which metric actually drove decisions

        # ------------------------------
        # Learning Rate Scheduling
        # ------------------------------
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # step on the same metric we're monitoring this epoch
                scheduler.step(monitor_value)
                
                # Check for mode consistency (optional warning)
                if hasattr(scheduler, 'mode'):
                    scheduler_mode = scheduler.mode
                    if (eff_mode == "min" and scheduler_mode != "min") or (eff_mode == "max" and scheduler_mode != "max"):
                        print(f"⚠️ Warning: LR scheduler mode ({scheduler_mode}) doesn't match monitor mode ({eff_mode})")
            else:
                scheduler.step()
        
        # Track learning rate after scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Print validation metrics for the current epoch (including monitored value)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, "
            f"TSS: {metrics['TSS']:.4f}, F1: {metrics['F1']:.4f}, "
            f"AUC: {metrics.get('AUC', float('nan')):.4f}, MCC: {metrics['MCC']:.4f}, HSS: {metrics['HSS']:.4f}, "
            f"LR: {current_lr:.2e}, Threshold: {current_threshold:.4f}, "
            f"MON({eff_monitor}): {monitor_value:.6f}")

        # ------------------------------
        # Early Stopping Check
        # ------------------------------
        if used_fallback:
            # Log and SKIP updating wait/best on fallback epoch
            print(f"ℹ️ Skipping early-stopping update this epoch because '{monitor}' was undefined; fell back to '{eff_monitor}'.")
        else:
            improved = (monitor_value < best_monitor - min_delta) if eff_mode == "min" else (monitor_value > best_monitor + min_delta)

            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                best_threshold = current_threshold
                best_model_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= es_patience:
                    print(f"Early stopping (monitor={eff_monitor}, mode={eff_mode}) at epoch {epoch + 1} "
                          f"with best {eff_monitor}={best_monitor:.6f}")
                    break  # stop if no improvement for 'patience' epochs

    # Load best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Normalize best_epoch if no improvement was ever recorded
    if best_epoch == -1:
        best_epoch = len(val_history) - 1
        if threshold_history:
            best_threshold = float(threshold_history[best_epoch])
    
    # Compute best-epoch metrics for symmetry with CV training
    metrics_best_epoch = {k: float(val_metric_history[k][best_epoch]) for k in val_metric_history}
        
    return model, metrics, train_history, val_history, val_metric_history, best_epoch, lr_history, threshold_history, best_threshold, monitor_history, monitor_name_history, metrics_best_epoch

def compute_tss(y_true, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr - fpr, (tn, fp, fn, tp)


def pt_generate_plots( summary_metrics, OUTPUT_CONFIG, path_manager, TASK_CONFIG):
    from .plots import (
        plot_boxplot, plot_training_curves,
        plot_metric_curves, plot_test_metrics
    )

    if OUTPUT_CONFIG["plot_boxplot"]:
        for metric in OUTPUT_CONFIG["plot_metrics"]:
            plot_boxplot(
                summary_metrics, metric_name=metric,
                save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_{metric}_boxplot.png")
            )

    if OUTPUT_CONFIG["plot_training_curves"]:
        plot_training_curves(
            summary_metrics,
            save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_loss_curves.png")
        )

    if OUTPUT_CONFIG["plot_metric_curves"]:
        for metric in OUTPUT_CONFIG["plot_metrics"]:
            plot_metric_curves(
                summary_metrics, metric_name=metric,
                save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_{metric}_curve.png")
            )

    if OUTPUT_CONFIG["plot_test_metrics"]:
        plot_test_metrics(
            summary_metrics, selected_metrics=OUTPUT_CONFIG["plot_metrics"],
            save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_test_metrics.png")
        )


def sk_generate_plots( summary_metrics, OUTPUT_CONFIG, path_manager, TASK_CONFIG):
    from .plots import (
        sk_plot_boxplot 
    )

    if OUTPUT_CONFIG["plot_boxplot"]:
        for metric in OUTPUT_CONFIG["plot_metrics"]:
            sk_plot_boxplot(
                summary_metrics, metric_name=metric,
                save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_{metric}_boxplot.png")
            )
    '''
    if OUTPUT_CONFIG["plot_training_curves"]:
        plot_training_curves(
            summary_metrics,
            save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_loss_curves.png")
        )

    if OUTPUT_CONFIG["plot_metric_curves"]:
        for metric in OUTPUT_CONFIG["plot_metrics"]:
            plot_metric_curves(
                summary_metrics, metric_name=metric,
                save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_{metric}_curve.png")
            )

    if OUTPUT_CONFIG["plot_test_metrics"]:
        plot_test_metrics(
            summary_metrics, selected_metrics=OUTPUT_CONFIG["plot_metrics"],
            save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_test_metrics.png")
        )
    '''
        
def run_cv_training(model_name   , window_map, base_path="", output_config=None, 
                    device="cuda", patience=5, num_epochs=20, n_folds=5, 
                    model_fn=None, batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                    path_manager=None, metrics_file="./metrics/summary_metrics.json",
                    task_config=None, lr_scheduler_config=None, loss_config=None,
                    threshold_config=None, calibration_config=None,
                    early_stopping_config=None, clearml_task = None, 
                    regularization_params =None):
    
    metrics_manager = MetricsManager(path_manager)
    save_models=output_config["save_models"]
    data_prefix=task_config["data_prefix"]
    generate_plot = [v for v in output_config.values() if isinstance(v, bool)]
     
    if  model_name in PT_MODELS: 
        summary_metrics = run_cv_training_pt(window_map, base_path, save_models, device, patience, num_epochs, n_folds, model_fn, batch_size, learning_rate, weight_decay, path_manager, metrics_file, data_prefix, lr_scheduler_config, loss_config, threshold_config, calibration_config, early_stopping_config, clearml_task = clearml_task)
        # Save metrics and plots 
        metrics_manager.save_metrics(summary_metrics, metrics_file)
        print(f"Training completed — results saved to: {path_manager.get_results_structure()}")
        if generate_plot :
            pt_generate_plots(summary_metrics, output_config, path_manager, task_config)
        return summary_metrics
    elif model_name in SK_MODELS: 
        summary_metrics = run_cv_training_sk(window_map, base_path, save_models, n_folds,  model_fn, metrics_file, data_prefix, model_name = model_name, path_manager=path_manager, clearml_task = clearml_task , regularization_params = regularization_params)
        # Save metrics and plots 
        metrics_manager.save_metrics(summary_metrics, metrics_file)
        print(f"Training completed — results saved to: {path_manager.get_results_structure()}")
        if generate_plot :  
            sk_generate_plots(summary_metrics, output_config, path_manager, task_config)
        return summary_metrics
    raise ValueError(
        f"Unsupported model_name '{model_name}'. "
        "Valid options are: {ALL_MODELS}"
    )

def safe_auc(y_true, y_scores):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_scores)

def HSS(y_true, y_pred):
    # Confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    numerator = 2 * (TP * TN - FP * FN)
    denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator
                
def run_cv_training_sk(window_map, base_path="", save_models=True, n_folds=None, model_fn=None,metrics_file="./metrics/summary_metrics.json", data_prefix="M+_flare", model_name = None,path_manager = None , clearml_task=None, regularization_params = None ):
    import os, joblib, json
    import cupy as cp
    from sklearn.metrics import f1_score
    from sklearn.metrics import matthews_corrcoef
  

    Cs = regularization_params
    print(Cs)
    summary_metrics = {}    
    
    # Optional: get ClearML logger
    logger = clearml_task.get_logger() if clearml_task else None
    
    for label, steps in window_map.items():
        results = []
        # Fitted models per C, kept in memory: only those of the selected C are saved
        fold_models_by_C = {}
        print(f"\nRunning CV for window: {label} ({steps} timesteps)")

        for C in Cs:
            fold_metrics = []  
            c_step = 0 
            print(f"\nRegularization parameter: {C}")
            for fold in range(n_folds):
 
                print(f"\nFold {fold + 1}/{n_folds}")
                metrics = {}
                metrics_val = {}
                metrics_test = {}
                # Load training and validation datasets 
                train_dataset = FlareHDF5Dataset(os.path.join(base_path, f"{data_prefix}_cv_{fold}_train.h5"), timesteps=steps)
                val_dataset = FlareHDF5Dataset(os.path.join(base_path, f"{data_prefix}_cv_{fold}_val.h5"), timesteps=steps)


                X_train = train_dataset.X.numpy() 
                X_train = np.nan_to_num(X_train, nan=0.0)  
                mask_train = train_dataset.mask.numpy() 
                y_train = train_dataset.Y.numpy().astype(int)
                
                X_val = val_dataset.X.numpy()
                mask_val = val_dataset.mask.numpy() 
                y_val = val_dataset.Y.numpy().astype(int)
                
                #X_train, mask_train, y_train = train_dataset.get_numpy_with_mask()
                #X_val, mask_val, y_val = val_dataset.get_numpy_with_mask()

                counts = np.bincount(y_train) 
                rate = counts[0]/counts[1]
                
                model = model_fn(steps)
                #
                if model_name == "XGBOOST":
                    model.clf.set_params(scale_pos_weight = rate, reg_lambda =C)
                else: 
                    model.classifier.set_params(C=C)
                model.fit(X_train, y_train, mask=mask_train)
                #print("All tested C values:", model.classifier.Cs_)
                #print("Mean cross-validation scores for each C:", model.classifier.score[1] )


                #y_val_proba = model.predict_proba(X_val, mask=mask_val)[:,1]
                #y_val_pred = (y_val_proba >= 0.5).astype(int)
                
                y_train_pred = model.predict(X_train, mask = mask_train)
                y_train_proba = model.predict_proba(X_train, mask = mask_train)
                p_pos = y_train_proba[:, 1]
                
                
                eps = 1e-10
    

                # Clip probabilities for numerical stability
                p = np.clip(p_pos, eps, 1 - eps)

                # Compute binary cross-entropy loss
                loss_train = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))

                #print("Training loss:", float(loss))


                
                tss_train, cm_train = compute_tss(y_train, y_train_pred)
                
                
                
                #print(f"Fold {fold} -> TSS: {tss_train:.4f}, Confusion: {cm_train}")
                
    
                ''' 
                y_int = y_train.astype(int)
                counts = np.bincount(y_int)
                print("Number of 0s:", counts[0])
                print("Number of 1s:", counts[1])
                '''
                
                y_val_pred = model.predict(X_val, mask = mask_val)
                # Compute metrics
                #auc_val = roc_auc_score(y_val, y_val_proba)
                #loss_val = log_loss(y_val, y_val_proba) 
                tss_val, cm_val = compute_tss(y_val, y_val_pred)
                #print(f"Fold val {fold} -> TSS: {tss_val:.4f}, Confusion: {cm_val}")

                y_val_proba = model.predict_proba(X_val, mask = mask_val)
                p_val = y_val_proba[:, 1]
                
                
                eps = 1e-10

                # Convert y and p_pos to CuPy and float32
                '''y_val = cp.asarray(y_val, dtype=cp.float32)
                p_val = cp.asarray(p_pos_val, dtype=cp.float32)
                '''
                val_loss = -np.mean(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val))
                
                
                F1_train= f1_score(y_train, y_train_pred)
                AUC_train = safe_auc(y_train, y_train_pred)
                MCC_train = matthews_corrcoef(y_train, y_train_pred)
                HSS_train = HSS(y_train, y_train_pred)
                
                
                F1_val= f1_score(y_val, y_val_pred)
                AUC_val = safe_auc(y_val, y_val_pred)
                MCC_val = matthews_corrcoef(y_val, y_val_pred)
                HSS_val = HSS(y_val, y_val_pred)
                
                
                metrics['TSS'] = tss_train
                metrics['F1'] = F1_train
                metrics['AUC'] = AUC_train
                metrics['MCC'] = MCC_train
                metrics['HSS'] = HSS_train
                
                
                metrics_val['TSS'] = tss_val
                metrics_val['F1'] = F1_val
                metrics_val['AUC'] = AUC_val
                metrics_val['MCC'] = MCC_val
                metrics_val['HSS'] = HSS_val 
                
                fold_metrics_val = {
                    "TSS": tss_val,
                    "F1": F1_val,
                    "AUC": AUC_val,
                    "MCC": MCC_val,
                    "HSS": HSS_val
                    }
                
                test_dataset = FlareHDF5Dataset(os.path.join(base_path, f"{data_prefix}_test.h5"), timesteps=steps)
                X_test = test_dataset.X.numpy()
                y_test = test_dataset.Y.numpy().astype(int) 
                mask_test = test_dataset.mask.numpy()
                y_test_pred = model.predict(X_test, mask = mask_test)
                
                
                F1_test= f1_score(y_test, y_test_pred)
                AUC_test = safe_auc(y_test, y_test_pred)
                MCC_test = matthews_corrcoef(y_test, y_test_pred)
                HSS_test = HSS(y_test, y_test_pred)
                tss_test, cm_test = compute_tss(y_test, y_test_pred)
                
                y_test_proba = model.predict_proba(X_test, mask = mask_test)
                p_test = y_test_proba[:, 1]
                test_loss = -np.mean(y_test * np.log(p_test) + (1 - y_test) * np.log(1 - p_test))
               
                metrics_test['TSS'] = tss_test
                metrics_test['F1'] = F1_test
                metrics_test['AUC'] = AUC_test
                metrics_test['MCC'] = MCC_test
                metrics_test['HSS'] = HSS_test
                 
                fold_metrics_test = {
                    "TSS": tss_test,
                    "F1": F1_test,
                    "AUC": AUC_test,
                    "MCC": MCC_test,
                    "HSS": HSS_test
                    }
                  
                
                print(f"Train ==> Loss: {loss_train:.4f}, "
                    f" TSS: {metrics['TSS']:.4f}, F1: {metrics['F1']:.4f},"
                    f" AUC: {metrics['AUC']:.4f}, MCC: {metrics['MCC']:.4f},"
                    f" HSS: {metrics['HSS']:.4f}.") 
            
                print(f"Validation ==> Loss: {val_loss:.4f}, "
                    f" TSS: {metrics_val['TSS']:.4f}, F1: {metrics_val['F1']:.4f},"
                    f" AUC: {metrics_val['AUC']:.4f}, MCC: {metrics_val['MCC']:.4f},"
                    f" HSS: {metrics_val['HSS']:.4f}.") 
                 
                print(f"Test ==> Loss: {test_loss:.4f}, "
                    f" TSS: {metrics_test['TSS']:.4f}, F1: {metrics_test['F1']:.4f},"
                    f" AUC: {metrics_test['AUC']:.4f}, MCC: {metrics_test['MCC']:.4f},"
                    f" HSS: {metrics_test['HSS']:.4f}.") 
                    


                
                fold_metrics.append({
                    "fold": fold,  
                    "val_metrics": fold_metrics_val,
                    "test_metrics": fold_metrics_test
                }) 

                fold_models_by_C.setdefault(C, []).append(model)

            
            
                # ------------------------------
                # CLEARML SCALAR REPORTING BLOCK
                # ------------------------------

                if logger is not None:
                    
                    C_value = float(C)

                    # Report train/val/test metrics for this fold
                    for split_name, split_metrics in [
                        ("train", metrics),
                        ("val", metrics_val),
                        ("test", metrics_test)
                    ]:
                        for metric_name, metric_val in split_metrics.items():
                            logger.report_vector(
                                title=f"{label} | {split_name.upper()} | Fold {fold}",
                                series=metric_name,
                                values=float(metric_val),
                                xaxis=[C_value],
                                iteration=c_step
                            )

                    # Report losses separately
                    logger.report_vector(
                        title=f"{label} | LOSS | Fold {fold}",
                        series="train_loss",
                        values=float(loss_train),
                        xaxis=[C_value], 
                        iteration=c_step
                    )
                    logger.report_vector(
                        title=f"{label} | LOSS | Fold {fold}",
                        series="val_loss",
                        values=float(val_loss),
                        xaxis=[C_value], 
                        iteration=c_step
                    )
                    logger.report_vector(
                        title=f"{label} | LOSS | Fold {fold}",
                        series="test_loss",
                        values=float(test_loss),
                        xaxis=[C_value], 
                        iteration=c_step
                    )


                c_step = c_step+1
            results.append({
                "C": C,
                "time_window": label,
                "folds": fold_metrics
            }) 
            
 
             
            mon = "val_loss"
            md = "min" 
            
            # Robust best-fold selection that handles fallbacks correctly
            key_map = {"tss": "TSS", "f1": "F1", "mcc": "MCC", "hss": "HSS"}

            def fold_best_value_for_selection(m, mon, md):
                # Return a scalar "best" value for this fold consistent with the monitor
                if mon == "val_loss":
                    series = np.asarray(m["val_history"], dtype=float)
                    return float(np.min(series) if md == "min" else np.max(series))
                elif mon == "auc":
                    series = np.asarray(m["val_metric_history"]["AUC"], dtype=float)
                    finite = np.isfinite(series)
                    if finite.any():
                        # AUC comparable across folds; don't mix with other metrics
                        return float(np.max(series[finite]) if md == "max" else np.min(series[finite]))
                    else:
                        # No finite AUC: ensure this fold can't win
                        return float("-inf") if md == "max" else float("+inf")
                else:
                    k = key_map.get(mon, None)
                    if k is None:
                        # unknown monitor -> default to val_loss (min)
                        series = np.asarray(m["val_history"], dtype=float)
                        return float(np.min(series))
                    series = np.asarray(m["val_metric_history"][k], dtype=float)
                    return float(np.max(series) if md == "max" else np.min(series))
                    # Convert to array-like summary
                 
            summary_metrics[label] =   results
      
            if path_manager:
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                # Convert all numpy types to Python native types
                serializable_metrics = convert_to_serializable(summary_metrics)
                with open(metrics_file, "w") as f:
                    json.dump(serializable_metrics, f, indent=2)
             
                 

        # Extract all C values
        all_Cs = sorted(set(entry["C"] for entry in results))

        # Mean test metrics per C: reported only, never used to select C
        mean_TSS = {
            C: np.mean([
                fold["test_metrics"]["TSS"]
                for entry in results if entry["C"] == C
                for fold in entry["folds"]
            ])
            for C in all_Cs
        }

        mean_HSS = {
            C: np.mean([
                fold["test_metrics"]["HSS"]
                for entry in results if entry["C"] == C
                for fold in entry["folds"]
            ])
            for C in all_Cs
        }

        mean_F1 = {
            C: np.mean([
                fold["test_metrics"]["F1"]
                for entry in results if entry["C"] == C
                for fold in entry["folds"]
            ])
            for C in all_Cs
        }

        mean_AUC = {
            C: np.mean([
                fold["test_metrics"]["AUC"]
                for entry in results if entry["C"] == C
                for fold in entry["folds"]
            ])
            for C in all_Cs
        }

        mean_MCC = {
            C: np.mean([
                fold["test_metrics"]["MCC"]
                for entry in results if entry["C"] == C
                for fold in entry["folds"]
            ])
            for C in all_Cs
        }

        # Mean validation TSS per C: used to select C, so the test set stays untouched
        mean_val_TSS = {
            C: np.mean([
                fold["val_metrics"]["TSS"]
                for entry in results if entry["C"] == C
                for fold in entry["folds"]
            ])
            for C in all_Cs
        }

        # Select best C by mean validation TSS
        best_C = max(mean_val_TSS, key=mean_val_TSS.get)

        # Best fold of the selected C (by validation TSS): saved as the "_best" model used by run_test
        best_C_folds = next(entry["folds"] for entry in results if entry["C"] == best_C)
        best_fold = int(np.argmax([fold["val_metrics"]["TSS"] for fold in best_C_folds]))

        print("\n=== CV Summary ===")
        for C in all_Cs:
            print(f"C={C:>8}: "
                f"mean val TSS={mean_val_TSS[C]:.4f} | test: "
                f"mean TSS={mean_TSS[C]:.4f}, "
                f"mean HSS={mean_HSS[C]:.4f}, "
                f"mean F1={mean_F1[C]:.4f}, "
                f"mean AUC={mean_AUC[C]:.4f}, "
                f"mean MCC={mean_MCC[C]:.4f}")

        print(f"\nBest C by mean validation TSS: {best_C} (best fold: {best_fold})")

        if save_models and path_manager:
            for fold, model in enumerate(fold_models_by_C[best_C]):
                model_path = path_manager.get_model_path(label, fold, model_type = "SK")
                print(f"Saving model (C={best_C}) to: {model_path}")
                joblib.dump(model, model_path)
            best_model_path = path_manager.get_model_path(label, best = True, model_type = "SK")
            print(f"Saving best model (C={best_C}, fold {best_fold}) to: {best_model_path}")
            joblib.dump(fold_models_by_C[best_C][best_fold], best_model_path)
        fold_models_by_C.clear()

        '''all_Cs = sorted(set(r["C"] for r in results))
        mean_TSS = {C: np.mean([r["TSS"] for r in results if r["C"] == C]) for C in all_Cs}
        mean_HSS = {C: np.mean([r["HSS"] for r in results if r["C"] == C]) for C in all_Cs}
        mean_F1 = {C: np.mean([r["F1"] for r in results if r["C"] == C]) for C in all_Cs}
        mean_AUC = {C: np.mean([r["AUC"] for r in results if r["C"] == C]) for C in all_Cs}
        mean_MCC = {C: np.mean([r["MCC"] for r in results if r["C"] == C]) for C in all_Cs}

        best_C = max(mean_TSS, key=mean_TSS.get)
        print("\n=== CV Summary ===")
        for C in all_Cs:
            print(f"C={C:>8}: mean TSS={mean_TSS[C]:.4f}, mean HSS={mean_HSS[C]:.4f}",
                  f"mean F1={mean_F1[C]:.4f}, mean AUC={mean_AUC[C]:.4f}, mean MCC={mean_MCC[C]:.4f}")
        print(f"\nBest C by mean TSS: {best_C}")
        '''
        
        #best_fold = int(np.argmax([m["val_auc"] for m in fold_metrics]))
        #summary_metrics[label] = {"folds": fold_metrics, "best_fold": best_fold}
        
        # Save metrics after each window to prevent data loss
        file_path = os.path.dirname(metrics_file)+"/"+model_name+f"/metrics/metrics_{label}.json"
        
        file_path_txt = file_path.replace(".json",".txt") 
        
        if path_manager:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)     
                
        with open(file_path_txt, "w") as f:
            f.write(f"\n=== CV Summary === window: {label} ({steps} timesteps) ===\n")
            for C in all_Cs:
                f.write(
                    f"C={C:>8}: "
                    f"mean val TSS={mean_val_TSS[C]:.4f} | test: "
                    f"mean TSS={mean_TSS[C]:.4f}, "
                    f"mean HSS={mean_HSS[C]:.4f}, "
                    f"mean F1={mean_F1[C]:.4f}, "
                    f"mean AUC={mean_AUC[C]:.4f}, "
                    f"mean MCC={mean_MCC[C]:.4f}\n"
                )
            f.write(f"\nBest C by mean validation TSS: {best_C} (best fold: {best_fold})\n")
            


        '''with open(file_path, "w") as f:
            json.dump(results, f, indent=3)        
        print(f"✅ CV summary saved to '{metrics_file}'")
        '''
    return summary_metrics

def run_cv_training_pt(window_map, base_path="", save_models=True, device="cuda",
                       patience=5, num_epochs=20, n_folds=5, model_fn=None,
                       batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                       path_manager=None, metrics_file="./metrics/summary_metrics.json",
                       data_prefix="M+_flare", lr_scheduler_config=None,
                       loss_config=None, threshold_config=None, calibration_config=None,
                       early_stopping_config=None, clearml_task=None ):
    """
    Runs k-fold cross-validation training across different input time windows.

    Args:
        window_map: Dictionary mapping window labels to number of timesteps
        base_path: Base path for data files
        save_models: Whether to save model checkpoints
        device: Device to use for training (cuda/cpu)
        patience: Number of epochs to wait for improvement before early stopping
        num_epochs: Maximum number of epochs to train
        n_folds: Number of cross-validation folds
        model_fn: Function to create model instances
        batch_size: Batch size for training and validation
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay (L2 regularization) for the optimizer
        path_manager: PathManager instance for handling paths
        metrics_file: Path to save the metrics summary
        data_prefix: Prefix for data files (e.g., "M+_flare", "X1_flare")
        lr_scheduler_config: Dictionary containing learning rate scheduler configuration
        loss_config: Dictionary containing loss function configuration
        threshold_config: Dictionary containing threshold optimization configuration
        calibration_config: Dictionary containing calibration configuration
            - enabled: bool, whether to enable calibration
            - type: str, calibrator type ("platt" or "temperature")

    Returns:
        Dictionary containing metrics for each window
    """
    import tempfile, json
    from clearml import Task
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("⚠️ CUDA not available. Switching to CPU.")

    # Helper function for computing best value from monitor history
    def best_from_history(hist, mode):
        if not hist:
            return float("nan")
        return float(min(hist) if mode == "min" else max(hist))

    # Optional: get ClearML logger
    logger = clearml_task.get_logger() if clearml_task else None
    
    summary_metrics = {}

    for label, steps in window_map.items():
        print(f"\nRunning CV for window: {label} ({steps} timesteps)")
        fold_metrics = []
        
        for fold in range(n_folds):
            print(f"\nFold {fold + 1}/{n_folds}")

            # Load training and validation datasets  
            train_dataset = FlareHDF5Dataset(os.path.join(base_path, f"{data_prefix}_cv_{fold}_train.h5"), timesteps=steps)
            val_dataset = FlareHDF5Dataset(os.path.join(base_path, f"{data_prefix}_cv_{fold}_val.h5"), timesteps=steps)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            #   
            #model = model_fn()
            model = model_fn(steps)
            model.to(device)
            
            # Setup training with loss function
            if loss_config:
                criterion = create_loss_function(loss_config, train_dataset, device)
                print(f"Using {loss_config.get('type', 'unknown')} loss function")
            else:
                from .metrics import compute_pos_weight
                pos_weight = compute_pos_weight(train_dataset).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print("Using standard BCE loss function")
                
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Train model
            model, metrics_last, train_history, val_history, val_metric_history, best_epoch, lr_history, threshold_history, best_threshold, monitor_history, monitor_name_history, metrics_best_epoch = train_model(
                model, train_loader, val_loader, optimizer, criterion,
                num_epochs=num_epochs, device=device, patience=patience,
                lr_scheduler_config=lr_scheduler_config,
                threshold_config=threshold_config,
                early_stopping_config=early_stopping_config
            )
            
            # --- NEW: Report per-epoch metrics to ClearML --- 
            if logger is not None:
                for epoch in range(len(train_history)):
                    # Loss curves
                    logger.report_scalar(title="Loss train", series=f"train_{label}_fold{fold}", value=train_history[epoch], iteration=epoch)
                    logger.report_scalar(title="Loss validation", series=f"val_{label}_fold{fold}", value=val_history[epoch], iteration=epoch)
                    
                    # Validation metrics
                    for metric_name, metric_vals in val_metric_history.items():
                        val = float(metric_vals[epoch])  # ensure numeric
                        logger.report_scalar(title=f"Metrics_{metric_name}_{label}", series=f"{metric_name}_fold{fold}", value=val, iteration=epoch)

                        # Optionally report summary metrics per fold
                        if logger is not None:
                            for metric_name, metric_val in metrics_last.items():
                                try:
                                    val = float(metric_val)
                                except Exception:
                                    val = None
                                if val is not None:
                                    logger.report_scalar(title=f"{label} Fold Summary", series=metric_name, value=val, iteration=fold)
            
            # ===========================================
            # POST-HOC CALIBRATION (per fold)
            # ===========================================
            print(f"🔧 Applying post-hoc calibration for fold {fold}")
            
            # 1) Collect validation logits
            z_val, y_val, harp_val, t_val = collect_logits(model, val_loader, device=device)
            
            # 2) Auto-choose calibrator type if needed
            cal_kind = calibration_config.get("type", "platt") if calibration_config else "platt"
            few_pos_threshold = calibration_config.get("few_pos_threshold", 25) if calibration_config else 25
            if cal_kind == "auto":
                pos_val = int(y_val.sum())
                cal_kind = "temperature" if pos_val < few_pos_threshold else "platt"
                print(f"🤖 Auto-chose {cal_kind} calibrator (n_pos_val={pos_val}, threshold={few_pos_threshold})")
            
            # 3) Fit calibrator on validation data only
            if calibration_config and calibration_config.get("enabled", True):
                calibrator = create_calibrator(cal_kind)
                calibrator.fit(z_val, y_val)
                p_val = calibrator.transform(z_val)
                cal_state = calibrator.state_dict()
                cal_state["n_pos_val"] = int(y_val.sum())
                cal_state["resolved_type"] = cal_kind
                print(f"✅ Fitted {cal_kind} calibrator on validation data")
            else:
                calibrator, cal_state = None, {"type": "none", "n_pos_val": int(y_val.sum())}
                p_val = sigmoid(z_val)
                print("ℹ️ Calibration disabled, using raw sigmoid probabilities")
            
            # 4) SANITY CHECK: AUC invariance (only when calibration is enabled)
            # Compute AUC for diagnostics (always needed)
            from sklearn.metrics import roc_auc_score
            
            def safe_auc(y, s):
                """Safe AUC calculation that handles single-class splits."""
                y = np.asarray(y)
                if len(np.unique(y)) < 2:
                    return None
                return roc_auc_score(y, s)
            
            # Always compute AUC for both raw and calibrated (for summary metrics)
            auc_raw_val = safe_auc(y_val, z_val)
            auc_cal_val = safe_auc(y_val, p_val)
            
            if calibrator is not None:
                # AUC invariance check (robust tolerance)
                TOL = 5e-6
                if auc_raw_val is not None:
                    auc_diff_val = abs(auc_raw_val - auc_cal_val)
                    assert auc_diff_val < TOL, f"AUC changed by {auc_diff_val:.2e} on validation! (tolerance: {TOL:.2e})"
                    print(f"✅ AUC invariance check passed (diff: {auc_diff_val:.2e})")
                else:
                    print("⚠️ Skipping AUC check on validation (single-class split)")
            else:
                print("ℹ️ Skipping AUC invariance check (calibration disabled)")
                # When no calibration, raw and cal are the same
                auc_cal_val = auc_raw_val
            
            # 5) Optimize threshold on calibrated validation probabilities
            if threshold_config and threshold_config.get("enabled", False):
                thr_opt = create_threshold_optimizer(
                    n_thresholds=threshold_config.get("n_thresholds", 200),
                    threshold_range=tuple(threshold_config.get("threshold_range", [0.0, 1.0]))
                )
                thr_metric = threshold_config.get("metric", "TSS")
                thr_kwargs = {}
                if thr_metric == "precision_at_recall":
                    thr_kwargs["min_recall"] = threshold_config.get("min_recall", 0.8)
                elif thr_metric == "f1_at_precision":
                    thr_kwargs["min_precision"] = threshold_config.get("min_precision", 0.8)
                
                thr_res = thr_opt.optimize_threshold(y_val, p_val, metric=thr_metric, **thr_kwargs)
                tau_star = float(thr_res["optimal_threshold"])
                print(f"🎯 Optimal threshold on calibrated val: {tau_star:.4f} (metric: {thr_res['optimal_metric_value']:.4f})")
            else:
                tau_star = 0.5
                print(f"ℹ️ Using default threshold: {tau_star}")
            
            # 6) Compute calibration diagnostics on validation
            n_bins = calibration_config.get("n_bins", 15) if calibration_config else 15
            val_diag = reliability_ece(y_val, p_val, n_bins=n_bins)
            val_diag_raw = reliability_ece(y_val, sigmoid(z_val), n_bins=n_bins)
            
            # SANITY CHECK: ECE improvement
            ece_improvement = val_diag_raw["ECE"] - val_diag["ECE"]
            print(f"📊 Val ECE: {val_diag['ECE']:.4f} (raw: {val_diag_raw['ECE']:.4f}, improvement: {ece_improvement:+.4f})")
            print(f"📊 Val Brier: {val_diag['Brier']:.4f} (raw: {val_diag_raw['Brier']:.4f})")
            
            # 7) Evaluate on test set with frozen calibrator and threshold
            test_dataset = FlareHDF5Dataset(os.path.join(base_path, f"{data_prefix}_test.h5"), timesteps=steps)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            z_test, y_test, harp_test, t_test = collect_logits(model, test_loader, device=device)
            p_test = calibrator.transform(z_test) if calibrator else sigmoid(z_test)
            y_pred = (p_test >= tau_star).astype(int)
            
            # SANITY CHECK: AUC invariance on test (only when calibration is enabled)
            # Always compute AUC for both raw and calibrated (for summary metrics)
            auc_raw_test = safe_auc(y_test, z_test)
            auc_cal_test = safe_auc(y_test, p_test)
            
            if calibrator is not None:
                if auc_raw_test is not None:
                    auc_diff_test = abs(auc_raw_test - auc_cal_test)
                    assert auc_diff_test < TOL, f"AUC changed by {auc_diff_test:.2e} on test! (tolerance: {TOL:.2e})"
                    print(f"Test AUC invariance check passed (diff: {auc_diff_test:.2e})")
                else:
                    print("Skipping AUC check on test (single-class split)")
            else:
                print("Skipping test AUC invariance check (calibration disabled)")
                # When no calibration, raw and cal are the same
                auc_cal_test = auc_raw_test
            
            # 8) Compute validation point metrics at tau* (for ablation tables)
            y_val_pred = (p_val >= tau_star).astype(int)
            val_point_metrics = compute_metrics(y_val, y_val_pred)
            
            # 9) Compute test metrics and expected confusion
            test_metrics = compute_metrics(np.array(y_test), y_pred)
            test_metrics["optimal_threshold"] = tau_star
            
            # Add expected confusion matrix
            exp_TP, exp_FP, exp_FN, exp_TN = expected_confusion(p_test, tau_star)
            test_metrics.update({
                "exp_TP": exp_TP,
                "exp_FP": exp_FP, 
                "exp_FN": exp_FN,
                "exp_TN": exp_TN,
                "alerts_at_tau": int((p_test >= tau_star).sum())
            })
            
            # 10) Compute test calibration diagnostics
            test_diag = reliability_ece(y_test, p_test, n_bins=n_bins)
            test_diag_raw = reliability_ece(y_test, sigmoid(z_test), n_bins=n_bins)
            
            # 11) Compute prior-adjusted test diagnostics (separates calibration drift from base-rate shift)
            pi_val = float(y_val.mean())
            pi_test = float(y_test.mean())
            p_test_adj = prior_adjust(p_test, pi_val, pi_test)
            test_diag_adj = reliability_ece(y_test, p_test_adj, n_bins=n_bins)
            
            # 12) Save per-sample outputs and comprehensive artifacts
            if path_manager:
                os.makedirs("./preds_val", exist_ok=True)
                os.makedirs("./preds_test", exist_ok=True)
                os.makedirs("./calibration", exist_ok=True)
                
                # Save validation predictions
                save_predictions_csv(
                    z_val, y_val, (p_val >= tau_star).astype(int),
                    f"./preds_val/{data_prefix}_{label}_fold{fold}.csv",
                    harp_val, t_val, prob_cal=p_val, prob_raw=sigmoid(z_val)
                )
                
                # Save test predictions
                save_predictions_csv(
                    z_test, y_test, y_pred,
                    f"./preds_test/{data_prefix}_{label}_fold{fold}.csv",
                    harp_test, t_test, prob_cal=p_test, prob_raw=sigmoid(z_test)
                )
                
                # Comprehensive calibration report
                import sklearn
                import scipy
                import subprocess
                import datetime
                
                try:
                    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                    stderr=subprocess.DEVNULL).decode().strip()[:8]
                except:
                    git_sha = "unknown"
                
                # Get comprehensive seed information for reproducibility
                try:
                    import random
                    python_seed = random.getstate()[1][0] if hasattr(random, 'getstate') else "unknown"
                except:
                    python_seed = "unknown"
                
                try:
                    numpy_seed = int(np.random.get_state()[1][0]) if hasattr(np.random, "get_state") else "unknown"
                except:
                    numpy_seed = "unknown"
                
                try:
                    torch_seed = int(torch.initial_seed()) if hasattr(torch, "initial_seed") else "unknown"
                except:
                    torch_seed = "unknown"
                
                try:
                    cuda_deterministic = bool(getattr(torch.backends.cudnn, "deterministic", False))
                except:
                    cuda_deterministic = "unknown"
                
                calibration_artifacts = {
                    "calibrator": cal_state,
                    "tau_star": tau_star,
                    "thr_metric": threshold_config.get("metric", "TSS") if threshold_config else "default",
                    "few_pos_threshold": few_pos_threshold,
                    "seeds": {
                        "python": python_seed,
                        "numpy": numpy_seed,
                        "torch": torch_seed,
                        "cuda_deterministic": cuda_deterministic
                    },
                    
                    # Validation diagnostics
                    "val_metrics": {
                        "AUC_raw": float(auc_raw_val),
                        "AUC_cal": float(auc_cal_val),
                        "ECE_raw": float(val_diag_raw["ECE"]),
                        "ECE_cal": float(val_diag["ECE"]),
                        "Brier_raw": float(val_diag_raw["Brier"]),
                        "Brier_cal": float(val_diag["Brier"]),
                        "n_val": len(y_val),
                        "n_pos_val": int(y_val.sum())
                    },
                    
                    # Validation point metrics at tau* (for ablation tables)
                    "val_point_metrics_at_tau": convert_to_serializable(val_point_metrics),
                    
                    # Test diagnostics
                    "test_metrics": {
                        "AUC_raw": float(auc_raw_test),
                        "AUC_cal": float(auc_cal_test),
                        "ECE_raw": float(test_diag_raw["ECE"]),
                        "ECE_cal": float(test_diag["ECE"]),
                        "Brier_raw": float(test_diag_raw["Brier"]),
                        "Brier_cal": float(test_diag["Brier"]),
                        "n_test": len(y_test),
                        "n_pos_test": int(y_test.sum())
                    },
                    
                    # Prior-adjusted test diagnostics (separates calibration drift from base-rate shift)
                    "test_metrics_prior_adjusted": {
                        "ECE_adj": float(test_diag_adj["ECE"]),
                        "Brier_adj": float(test_diag_adj["Brier"]),
                        "pi_val": pi_val,
                        "pi_test": pi_test
                    },
                    
                    # ECE binning details
                    "ece_config": {
                        "n_bins": n_bins,
                        "bin_edges": val_diag["bin_edges"].tolist() if hasattr(val_diag["bin_edges"], "tolist") else val_diag["bin_edges"]
                    },
                    
                    # Versioning
                    "lib_versions": {
                        "sklearn": sklearn.__version__,
                        "scipy": scipy.__version__
                    },
                    "git_sha": git_sha,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "note": "prior_adjust available for analysis of prevalence shift"
                }
                
                # Use path manager to save calibration artifacts in run-specific directory
                calibration_filename = f"{data_prefix}_{label}_fold{fold}.json"
                calibration_path = path_manager.get_calibration_path(calibration_filename)
                
                with open(calibration_path, "w") as f:
                    json.dump(calibration_artifacts, f, indent=2)
                
                print(f"💾 Saved comprehensive calibration report to {calibration_path}")
            
            # Define early stopping config for this fold
            monitor_for_fold = (early_stopping_config or {}).get("monitor", "val_loss").lower()
            mode_for_fold = (early_stopping_config or {}).get("mode", "min").lower()
            min_delta_for_fold = float((early_stopping_config or {}).get("min_delta", 0.0))
            patience_for_fold = int((early_stopping_config or {}).get("patience", patience))
            
            # Use best-epoch metrics from train_model (now returned directly)
            
            fold_metrics.append({
                "fold": fold,
                "train_history": train_history,
                "val_history": val_history,
                "val_metric_history": val_metric_history,
                "val_metrics_last_epoch": metrics_last,  # Renamed for clarity
                "val_metrics_best_epoch": metrics_best_epoch,
                "val_point_metrics_at_tau": convert_to_serializable(val_point_metrics),  # For easier aggregation
                "best_epoch": best_epoch,
                "lr_history": lr_history,
                "threshold_history": threshold_history,
                "best_threshold": best_threshold,
                "monitor_history": monitor_history,
                "monitor_name_history": monitor_name_history,
                "early_stopping": {
                    "monitor": monitor_for_fold,
                    "mode": mode_for_fold,
                    "min_delta": min_delta_for_fold,
                    "patience": patience_for_fold,
                    "best_epoch": best_epoch,
                    "best_value": best_from_history(monitor_history, mode_for_fold),
                    "best_loss": float(val_history[best_epoch]) if best_epoch < len(val_history) else float("nan"),
                },
                "calibration": {
                    "enabled": calibration_config.get("enabled", False) if calibration_config else False,
                    "type": calibration_config.get("type", "none") if calibration_config else "none",
                    "val_ECE": val_diag["ECE"],
                    "val_Brier": val_diag["Brier"],
                    "tau_star": tau_star
                },
                "test_metrics": test_metrics
            })

            if save_models and path_manager:
                model_path = path_manager.get_model_path(label, fold)
                print(f"Saving model to: {model_path}")
                torch.save(model.state_dict(), model_path)

        # Store metrics for this window
        # Pick best_fold consistently with the monitor used for early stopping
        mon = (early_stopping_config or {}).get("monitor", "val_loss").lower()
        md = (early_stopping_config or {}).get("mode", "min").lower()
        
        # Robust best-fold selection that handles fallbacks correctly
        key_map = {"tss": "TSS", "f1": "F1", "mcc": "MCC", "hss": "HSS"}

        def fold_best_value_for_selection(m, mon, md):
            # Return a scalar "best" value for this fold consistent with the monitor
            if mon == "val_loss":
                series = np.asarray(m["val_history"], dtype=float)
                return float(np.min(series) if md == "min" else np.max(series))
            elif mon == "auc":
                series = np.asarray(m["val_metric_history"]["AUC"], dtype=float)
                finite = np.isfinite(series)
                if finite.any():
                    # AUC comparable across folds; don't mix with other metrics
                    return float(np.max(series[finite]) if md == "max" else np.min(series[finite]))
                else:
                    # No finite AUC: ensure this fold can't win
                    return float("-inf") if md == "max" else float("+inf")
            else:
                k = key_map.get(mon, None)
                if k is None:
                    # unknown monitor -> default to val_loss (min)
                    series = np.asarray(m["val_history"], dtype=float)
                    return float(np.min(series))
                series = np.asarray(m["val_metric_history"][k], dtype=float)
                return float(np.max(series) if md == "max" else np.min(series))

        vals = [fold_best_value_for_selection(m, mon, md) for m in fold_metrics]
        # Choose the index consistent with the mode (sentinel values ensure proper selection)
        best_fold = int(np.argmax(vals) if md == "max" else np.argmin(vals))
        
        summary_metrics[label] = {
            "folds": fold_metrics,
            "best_fold": best_fold
        }

        # Note: Test evaluation is now done per-fold with calibration
        # No need for separate test evaluation here

        # Save metrics after each window to prevent data loss
        if path_manager:
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            # Convert all numpy types to Python native types
            serializable_metrics = convert_to_serializable(summary_metrics)
            with open(metrics_file, "w") as f:
                json.dump(serializable_metrics, f, indent=2)

    return summary_metrics

# Convert all numpy types in summary_metrics to native Python types
def convert_metrics(obj):
    if isinstance(obj, dict):
        return {k: convert_metrics(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_metrics(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


def run_test(
    model_name ,
    window_map,
    base_path,
    data_prefix,  
    model_checkpoint_path=None, 
    clearml_task=None,
    device = "cuda",
    batch_size = 64,
    path_manager = None, 
    model_fn = None,
    tau_star = 0.5,
    test_dataset_path = None
): 
    from clearml import Task
     
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        

    # Optional: get ClearML logger
    logger = clearml_task.get_logger() if clearml_task else None
    
    test_metrics = {}
    
    for label, steps in window_map.items():
        print(f"\Testing for window: {label} ({steps} timesteps)")
        
        if  path_manager and  model_name in SK_MODELS : 
            model_checkpoint_path = path_manager.get_model_path(label, best = True, model_type = "SK")
        elif path_manager and  model_name in PT_MODELS : #model_checkpoint_path is None and path_manager and  model_name in PT_MODELS : 
            model_checkpoint_path = path_manager.get_model_path(label, best = True,  model_type = "PT")        
        else:
            raise ValueError("❌ model_checkpoint_path must be provided")
            
        # ----------------------------------------------------------
        # 1. Load test set
        # ----------------------------------------------------------
        
        if  test_dataset_path:
            test_dataset = FlareHDF5Dataset(
                test_dataset_path,
                timesteps=steps
            )
        else:
            test_dataset = FlareHDF5Dataset(
                os.path.join(base_path, f"{data_prefix}_test.h5"),
                timesteps=steps
            )
        
                
        if model_checkpoint_path and model_name in PT_MODELS :
            
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
              
            # ----------------------------------------------------------
            # 2. Load model
            # ----------------------------------------------------------
            model = model_fn(steps)
            model.to(device)            
            model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
            print(f"✅ Loaded model checkpoint: {model_checkpoint_path}")
            model.eval()
  
            # ----------------------------------------------------------
            # 4. Collect logits
            # ----------------------------------------------------------
            z_test, y_test, harp_test, t_test = collect_logits(
                model, test_loader, device=device
            )

            # ----------------------------------------------------------
            # 5. Apply calibration
            # ----------------------------------------------------------
            from .calibration import sigmoid
 
            p_test = sigmoid(z_test)

            # ----------------------------------------------------------
            # 6. Apply threshold
            # ----------------------------------------------------------
            y_pred = (p_test >= tau_star).astype(int)

            # ----------------------------------------------------------
            # 7. Compute metrics
            # ----------------------------------------------------------
            from .metrics import compute_metrics
            test_metrics = compute_metrics(y_test, y_pred)

            test_metrics["tau_star"] = tau_star
            test_metrics["alerts_at_tau"] = int(y_pred.sum())

            print("📊 Test metrics:")
            for k, v in test_metrics.items():
                print(f"  - {k}: {v}")

            metrics = {
                "z_test": z_test,
                "p_test": p_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "harp": harp_test,
                "timestamps": t_test,
                "test_metrics": test_metrics
            }
            if logger is not None:
                for metric_name, metric_value in test_metrics.items():
                    logger.report_scalar(
                        title="TEST METRICS",
                        series=metric_name,
                        value=float(metric_value),
                        iteration=steps
                    )

            test_metrics[label ] = metrics

        
        elif model_checkpoint_path and model_name in SK_MODELS :
            import joblib
            from sklearn.metrics import f1_score
            from sklearn.metrics import matthews_corrcoef
            
            print(f"✅ Loaded model checkpoint: {model_checkpoint_path}")
            model = joblib.load(model_checkpoint_path)
             
            
            X_test = test_dataset.X.numpy()
            y_test = test_dataset.Y.numpy().astype(int) 
            mask_test = test_dataset.mask.numpy()
            y_pred = model.predict(X_test, mask = mask_test)
            
            
            from .metrics import compute_metrics
            test_metrics = compute_metrics(y_test, y_pred)
            

            print("📊 Test metrics:")
            for k, v in test_metrics.items():
                print(f"  - {k}: {v}")
            
            y_test_proba = model.predict_proba(X_test, mask = mask_test)
            p_test = y_test_proba[:, 1]
    #        test_loss = -np.mean(y_test * np.log(p_test) + (1 - y_test) * np.log(1 - p_test))
            
            metrics =  { 
                "p_test": p_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "test_metrics": test_metrics  
            } 
            
            if logger is not None:
                for metric_name, metric_value in test_metrics.items():
                    logger.report_scalar(
                        title="TEST METRICS",
                        series=metric_name,
                        value=float(metric_value),
                        iteration=steps
                    )
                    
            test_metrics[label ] = metrics
        
    

    return test_metrics