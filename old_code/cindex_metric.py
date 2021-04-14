# Name: cindex_metric.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: March 4, 2021

# Code adapted from https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/

import numpy as np
import tensorflow as tf
from sksurv.metrics import concordance_index_censored

class CindexMetric:
    """Computes concordance index across one epoch."""

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def update_state(self, y_true: Dict[str, tf.Tensor], y_pred: tf.Tensor) -> None:
        """Collect observed time, event indicator and predictions for a batch.

        Args:
            y_true - dictionary containing: 
                    1) label_time = tensor containing observed time for one batch
                    2) label_event = tensor containing event indicator for one batch
            y_pred - tf.Tensor containing predicted risk scores for one batch
        """
        self._data["label_time"].append(y_true["label_time"].numpy())
        self._data["label_event"].append(y_true["label_event"].numpy())
        self._data["prediction"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        # Combine labels and predictions into one dictionary
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        # Using c-index function sksurv
        results = concordance_index_censored(
            data["label_event"] == 1,
            data["label_time"],
            data["prediction"])

        result_data = {}
        names = ("cindex", "concordant", "discordant", "tied_risk")
        for k, v in zip(names, results):
            result_data[k] = v

        return result_data
