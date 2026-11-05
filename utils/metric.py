
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_acc_at_skip_rate(samples, skip_rate):

    metric_values = np.array([s['metric'] for s in samples])
    
    sorted_indices = np.argsort(metric_values)[::-1]
    
    n_samples = len(samples)
    k = int(round(skip_rate * n_samples))
    k = max(0, min(n_samples, k))

    skip_mask = np.zeros(n_samples, dtype=bool)
    if k > 0:
        skip_mask[sorted_indices[:k]] = True
    
    final_predictions = []
    final_labels = []
    
    correct_count = 0
    for i, s in enumerate(samples):
        if skip_mask[i]:
            final_predictions.append(s['zs_pred'])
            final_labels.append(s['target'])
            if s['zs_pred'] == s['target']:
                correct_count += 1
        else:
            final_predictions.append(s['tta_pred'])
            final_labels.append(s['target'])
            if s['tta_pred'] == s['target']:
                correct_count += 1
    
    acc = correct_count / n_samples * 100

    return acc


def compute_roc_auc(samples):
    metric_values = []
    labels = []
    
    for s in samples:
        metric_values.append(s['metric'])
        zs_correct = (s['zs_pred'] == s['target'])
        tta_correct = (s['tta_pred'] == s['target'])
        should_skip = not ((not zs_correct) and tta_correct)
        labels.append(1 if should_skip else 0)
    
    metric_values = np.array(metric_values)
    labels = np.array(labels)
    
    fpr, tpr, _ = roc_curve(labels, metric_values)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc


def compute_aep(samples, num_steps=21):
    skip_rates = np.linspace(0.0, 1.0, num_steps)
    accs = np.array([compute_acc_at_skip_rate(samples, p) for p in skip_rates])
    weighted_acc = accs * 2 * (1 - skip_rates)
    aep = np.trapz(weighted_acc, skip_rates)
    return aep




def compute_skip_metrics(metrics_list):
    samples = metrics_list
    
    zs_correct = sum(1 for s in samples if s['zs_pred'] == s['target'])
    tta_correct = sum(1 for s in samples if s['tta_pred'] == s['target'])
    n_samples = len(samples)
    
    zs_acc = zs_correct / n_samples * 100
    tta_acc = tta_correct / n_samples * 100
    
    auc = compute_roc_auc(samples)
    skip_rates, skip_accs = compute_acc_at_skip_rates(samples)
    skip_rates_array = np.asarray(skip_rates)
    skip_accs_array = np.asarray(skip_accs)
    weighted_acc = skip_accs_array * 2 * (1 - skip_rates_array)
    aep = np.trapz(weighted_acc, skip_rates_array)
  
    
    return {
        'auc': auc,
        'aep': aep,
        'zs_acc': zs_acc,
        'tta_acc': tta_acc,
        'skip_rates': skip_rates,
        'skip_accs': skip_accs,
    }


def compute_acc_at_skip_rates(metrics_list, num_steps=21):
    skip_rates = np.linspace(0.0, 1.0, num_steps)
    accs = []
    
    for p in skip_rates:
        acc = compute_acc_at_skip_rate(metrics_list, p)
        accs.append(acc)
    
    return skip_rates.tolist(), accs


