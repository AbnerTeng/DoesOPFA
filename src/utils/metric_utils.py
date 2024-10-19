"""
evaluation metrics from scratch
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


class Metrics:
    """
    Evaluation metrics for classification
    """
    def __init__(self, pred_data: pd.Series, true_data: pd.Series) -> None:
        self.pred_data = pred_data
        self.true_data = true_data


    def accuracy(self) -> float:
        """
        Accuracy
        
        count(pred = true) / total 
        """
        return (self.pred_data == self.true_data).sum() / self.true_data.shape[0]


    def precision(self, target: int) -> float:
        """
        Precision
        
        count(pred = true = 1) / count(pred = 1)
        
        target: 0 or 1
        """
        average = "binary" if len(self.pred_data.unique()) == 2 else "macro"

        return precision_score(
            self.true_data, self.pred_data, pos_label=target, average=average
        )


    def recall(self, target: int) -> float:
        """
        Recall
        
        count(pred = true = 1) / count(true = 1)
        
        target: 0 or 1
        """
        average = "binary" if len(self.pred_data.unique()) == 2 else "macro"

        return recall_score(
            self.true_data, self.pred_data, pos_label=target, average=average
        )


    def f_score(self, target: int) -> float:
        """
        F score
        
        (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall
        """
        average = "binary" if len(self.pred_data.unique()) == 2 else "macro"

        return f1_score(
            self.true_data, self.pred_data, pos_label=target, average=average
        )


    def entropy(self, num_classes: int) -> float:
        """
        Entropy
        """
        self.pred_data = self.pred_data.astype(int)

        if not np.issubdtype(self.true_data.values.dtype, np.integer):
            raise ValueError("true_data must be integer")

        if not np.issubdtype(self.pred_data.values.dtype, np.integer):
            raise ValueError("pred_data must be integer")

        epsilon = 1e-15
        true_label_one_hot = np.eye(num_classes)[self.true_data.values]
        predicted_label_one_hot = np.eye(num_classes)[self.pred_data.values]
        loss = -np.sum(
            true_label_one_hot * np.log(
                predicted_label_one_hot + epsilon
            )
        ) / self.true_data.shape[0]

        return loss
