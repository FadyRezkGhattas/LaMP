
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from data.datasets import get_all_labels

def get_metrics(task, tokenizer):
    greater_is_better = True
    labels = get_all_labels(task)
    if task == "LaMP-2":
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
        best_metric = "accuracy"
    elif task == "LaMP-3":
        compute_metrics = create_metric_mae_rmse(tokenizer = tokenizer, all_labels = labels)
        best_metric = "mae"
        greater_is_better = False
    elif task == "LaMP-4":
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP-5":
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    else:
        raise ValueError(f"Task {task} not supported")
    return compute_metrics, best_metric, labels, greater_is_better