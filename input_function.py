# Name: input_function.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: March 4, 2021

# Contains Class definition for InputFunction to be used in CNN
# Code adapted from https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/

import numpy as np
import tensorflow as tf

def _make_riskset(time):
    """
    Compute mask that represents each sample's risk set

    Args:
        time - numpy array of observed event times, shape = (n_samples,)


    Returns:
        risk_set - numpy array of boolean values with risk sets in rows, shape = (n_samples, n_samples)
    """

    assert time.ndim == 1

    # Sort in descending order
    o = np.argsort(-time, kind="mergesort")

    # Initialize risk set
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)

    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True

    return risk_set


class InputFunction:
    """
    Callable input function that computes the risk set for each batch

    Args:
        images - numpy array of images, shape = (n_samples, height, width)
        time - numpy array of observed time labels, shape = (n_samples,)
        event - numpy array of event indicator, shape = (n_samples,)
        batch_size - int, number of samples per batch
        drop_last - bool, whether to drop the last incomplete batch
        shuffle - bool, whether to shuffle data
        seed - int, random number seed
    """

    def __init__(self, images, time, event, batch_size=64, drop_last=False, shuffle=False, seed = 89):
        # If image is 3D, reduce dimension to 2D
        if images.ndim == 3:
            images = images[..., np.newaxis]

        self.images = images
        self.time = time
        self.event = event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def size(self):
        """
        Total number of samples
        """
        return self.images.shape[0]

    def steps_per_epoch(self):
        """
        Number of batches for one epoch
        """
        return int(np.floor(self.size() / self.batch_size))

    def _get_data_batch(self, index):
        """
        Compute risk set for samples in a batch

        Args:
            index - indices for the batch

        Returns:
            images - numpy array of images in the batch
            labels - dictionary of tuples (str, numpy array) with event, time, and riskset labels
        """
        time = self.time[index]
        event = self.event[index]
        images = self.images[index]

        # Create dictionary of the labels for the batch samples
        labels = {
                    "label_event": event.astype(np.int32),
                    "label_time": time.astype(np.int32),
                    "label_riskset": _make_riskset(time)
                 }

        return images, labels

    def _iter_data(self):
        """
        Generator that yields one batch at a time

        Returns:
            Iterable object over get_data_batch output
        """
        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)

        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch)]
            yield self._get_data_batch(idx)

        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            yield self._get_data_batch(idx)

    def _get_shapes(self):
        """
        Return shapes of data returned by self._iter_data

        Returns:
            images - tf.TensorShape, shape specification for images
            labels - dictionary of (str, tf.TensorShape), shape specification for labels
        """
        batch_size = self.batch_size if self.drop_last else None
        h, w, c = self.images.shape[1:]
        images = tf.TensorShape([batch_size, h, w, c])

        labels = {
                    k: tf.TensorShape((batch_size,))
                    for k in ("label_event", "label_time")
                 }
        labels["label_riskset"] = tf.TensorShape((batch_size, batch_size))
        return images, labels

    def _get_dtypes(self):
        """
        Return dtypes of data returned by self._iter_data

        Returns:
            tf.float32 - tf.Dtype
            labels - dtype of labels
        """
        labels = {
                    "label_event": tf.int32,
                    "label_time": tf.float32,
                    "label_riskset": tf.bool
                 }
        return tf.float32, labels

    def _make_dataset(self):
        """
        Create dataset from generator

        Returns:
            ds - Dataset containing data, types, and shapes of the images and labels
        """
        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            self._get_dtypes(),
            self._get_shapes()
        )
        return ds

    def __call__(self):
        return self._make_dataset()
    