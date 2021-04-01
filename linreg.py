# Name: linreg.py
# Environment: Python 3.8
# Author: Katy Scott
# Created: April 1, 2021

from imp import reload
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPool2D
from tensorflow.keras.optimizers import Adam
import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2
from tqdm import tqdm
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union

# Imports from my own code
from patient_data_split import pat_train_test_split
from input_function import InputFunction, _make_riskset
import cph_loss
import cindex_metric

