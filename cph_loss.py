# Name: cph_loss.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: March 4, 2021

# Code adapted from https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/

import numpy as np
import tensorflow as tf

def safe_normalize(x):
    """
    Normalize risk scores to avoid exp underflowing
    
    Note that only risk scores relative to each other matter. 
    If min. risk score is negative, shift scores so minimum is at zero.
    
    Args:
        x - tf.Tensor, risk scores
        
    Returns:
        normalized x as tf.Tensor
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm


def logsumexp_masked(risk_scores, mask, axis=0, keepdims=None):
    """
    Compute logsumexp across `axis` for entries where `mask` is true
    
    Args:
        risk_scores - tf.Tensor of predicted outputs of CoxPH, must be 2D
        mask - numpy array of boolean values with risk sets in rows, shape = (n_samples, n_samples)
        axis - int indicating which axis to perform sum across, should be axis samples is on (?)
        keepdims - bool, wheter to retain reduced dimensions in calculations
    
    Return:
        output - tf.Tensor logsumexp for risk scores
    """
    risk_scores.shape.assert_same_rank(mask.shape)
    
    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)
        
        # Subtract max value before taking exponential for numerical stability
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax
        
        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis-axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
            
    return output


class CoxPHLoss(tf.keras.losses.Loss):
    """
    Negative partial log-likelihood of Cox's proportional hazards model
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        """
        Compute loss
        
        Args:
            y_true: list/tuple of rank 2 tf.Tensors
                   - first element holds binary vector where 1 indicates event, 0 indicates censoring
                   - second element holds riskset = boolean matrix where ith row denots risk set of ith instance
            y_pred: rank 2 tf.Tensor of predicted outputs
        
        Returns:
            loss: tf.Tensor containing loss for each instance in a batch
        """
        event, riskset = y_true
        predictions = y_pred
        
        # INPUT CHECKING
        pred_shape = predictions.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)
        
        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal rank of event (received %s)" % (
                pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)
            
        
        event = tf.cast(event, predictions.dtype)
        # Normalize risk scores
        predictions = safe_normalize(predictions)
        
        # More input checking
        with tf.name_scope("assertions"):
            assertions = (
                            tf.debugging.assert_less_equal(event, 1.),
                            tf.debugging.assert_greater_equal(event, 0.),
                            tf.debugging.assert_type(riskset, tf.bool)
                         )
        
        # move batch dimension to the end so predictions get broadcast row-wise when multiplying by riskset
        # row-wise when multiplying by riskset
        pred_t = tf.transpose(predictions)
        
        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        
        assert rr.shape.as_list() == predictions.shape.as_list(), "Hello"
        
        losses = tf.math.multiply(event, rr - predictions)
        
        return losses