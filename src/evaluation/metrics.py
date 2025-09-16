import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, balanced_accuracy_score, jaccard_score,
    matthews_corrcoef, cohen_kappa_score, average_precision_score,
    log_loss, brier_score_loss, fbeta_score
)

def calculate_comprehensive_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {}
    metrics['cm'] = cm
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['bal_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['prec'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['rec'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1'] = 2 * (metrics['prec'] * metrics['rec']) / (metrics['prec'] + metrics['rec']) if (metrics['prec'] + metrics['rec']) > 0 else 0.0
    metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, average='binary')
    metrics['jacc'] = jaccard_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics['youden_j'] = metrics['rec'] + metrics['specificity'] - 1
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        metrics['ll'] = log_loss(y_true, y_proba)
        metrics['brier'] = brier_score_loss(y_true, y_proba)
    except:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
        metrics['ll'] = 0.0
        metrics['brier'] = 0.0
    
    return metrics

def print_metrics_summary(metrics):
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Balanced Accuracy: {metrics['bal_acc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['prec']:.4f}")
    print(f"Recall: {metrics['rec']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Matthews Correlation: {metrics['mcc']:.4f}")
    if 'space_dit_score' in metrics:
        print(f"Space DIT Score: {metrics['space_dit_score']:.4f}")
    if 'ksp_orbital_efficiency' in metrics:
        print(f"KSP Orbital Efficiency: {metrics['ksp_orbital_efficiency']:.4f}")