from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                           confusion_matrix, balanced_accuracy_score, jaccard_score, 
                           matthews_corrcoef, cohen_kappa_score, roc_auc_score, 
                           average_precision_score, log_loss, brier_score_loss,
                           mean_squared_error, mean_absolute_error)
import numpy as np

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * (prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    f05 = 1.25 * (prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
    
    jacc = jaccard_score(y_true, y_pred, average='binary')
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    ford = fn / (fn + tn) if (fn + tn) > 0 else 0.0
    youden_j = rec + specificity - 1
    
    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        ll = log_loss(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)
        mse = mean_squared_error(y_true, y_proba)
        mae = mean_absolute_error(y_true, y_proba)
    else:
        roc_auc = pr_auc = ll = brier = mse = mae = 0.0
    
    return {
        'cm': cm,
        'acc': acc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'specificity': specificity,
        'f1': f1,
        'f2': f2,
        'f05': f05,
        'jacc': jacc,
        'mcc': mcc,
        'kappa': kappa,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'fdr': fdr,
        'ford': ford,
        'youden_j': youden_j,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'll': ll,
        'brier': brier,
        'mse': mse,
        'mae': mae
    }

def print_evaluation_results(metrics, dataset_name):
    print(f"\n=== EVALUATION RESULTS FOR {dataset_name} ===")
    print(f"Confusion Matrix:\n{metrics['cm']}")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Balanced Accuracy: {metrics['bal_acc']:.4f}")
    print(f"Precision (PPV): {metrics['prec']:.4f}")
    print(f"Recall (Sensitivity/TPR): {metrics['rec']:.4f}")
    print(f"Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"F2 Score: {metrics['f2']:.4f}")
    print(f"F0.5 Score: {metrics['f05']:.4f}")
    print(f"Jaccard Index: {metrics['jacc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"NPV: {metrics['npv']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}")
    print(f"FNR: {metrics['fnr']:.4f}")
    print(f"FDR: {metrics['fdr']:.4f}")
    print(f"FOR: {metrics['ford']:.4f}")
    print(f"Youden's J: {metrics['youden_j']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC (Average Precision): {metrics['pr_auc']:.4f}")
    print(f"Log Loss: {metrics['ll']:.4f}")
    print(f"Brier Score: {metrics['brier']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
