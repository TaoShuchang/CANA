

import json
import os
import pickle
import random
import time
from urllib import request

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.csr import csr_matrix
import torch
import torch_sparse
from torch_sparse import SparseTensor

pd.set_option('display.width', 1000)

class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop attack process.

    """
    def __init__(self, patience=100, epsilon=1e-4):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        r"""

        Parameters
        ----------
        score : float
            Value of attack acore.

        """
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True

    def reset(self):
        self.min_score = None
        self.stop = False
        self.count = 0