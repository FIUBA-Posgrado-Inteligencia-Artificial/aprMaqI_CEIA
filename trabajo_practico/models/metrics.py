import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score


class MetricsCollector:
    def __init__(self):
        self.records = {}

    def add(self, model_name, metrics: dict):
        self.records[model_name] = metrics

    def to_dataframe(self):
        return pd.DataFrame(self.records).T


def eval_metrics(y_true, y_pred, pos_label=1):
    return {
        "fbeta_0.5": fbeta_score(y_true, y_pred, beta=0.5, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
    }
