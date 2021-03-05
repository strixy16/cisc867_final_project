# Name: train_and_evaluate.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: March 5, 2021

# Code adapted from https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/

import numpy as np
import tensorflow as tf
import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean

from cph_loss import CoxPHLoss
from cindex_metric import CindexMetric


class TrainAndEvaluateModel:
    """
    Callable function to set up a CPH neural net model to be trained and evaluated
    
    Args:
        model - a tf.Keras model
        
    """
    
    def __init__(self, model, model_dir, train_dataset, eval_dataset, learning_rate, num_epochs):
        
        self.num_epochs = num_epochs
        self.model_dir = model_dir
        self.model = model
        
        self.train_ds = train_dataset
        self.val_ds = eval_dataset
        
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()
        
        self.train_loss_metric = Mean(name="train_loss")
        self.val_loss_metric = Mean(name="val_loss")
        self.val_cindex_metric = CindexMetric()
        
    @tf.function
    def train_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            
            train_loss = self.loss_fn(y_true=)