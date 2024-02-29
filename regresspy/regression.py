from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from loss import mae, sse, mse, rmse
from gradient_descent import forward, backward
from regresspy import loss


class Regression(object):
    def __init__(self, epochs=50, learning_rate=0.001):
        self._epochs = epochs
        self._lr = learning_rate
        self._weights = {}
        self._X = None
        self._Y = None
    
    def fit(self, X: ndarray, Y:ndarray) -> None:
        assert X.shape[0] == Y.shape[0]

        self._initialize_weights(X.shape)
        assert self._weights['W'].shape == (X.shape[1], 1)
        assert self._weights['B'].shape == (1, 1)

        self._train(X, Y)

    def predict(self, X: ndarray) -> ndarray:

        predictions = X @ self._weights['W'] + self._weights['B']
        return predictions

    def score(self, X: ndarray, Y: ndarray, metric='rmse') -> float:

        metrics = {
            'mae': mae,
            'sse': sse,
            'mse': mse,
            'rmse': rmse
        }

        predictions = X @ self._weights['W'] + self._weights['B']
        if metric == 'mae':
            score = mae(Y, predictions)
        elif metric == 'sse':
            score = sse(Y, predictions)
        elif metric == 'mse':
            score = mse(Y, predictions)
        else:
            score = rmse(Y, predictions)
        return score

    def _initialize_weights(self, shape: Tuple[int, int]) -> None:

        self._weights = {
            'W': np.random.rand(shape[1], 1),
            'B': np.random.rand(1, 1)
        }
    
    def _train(self, X: ndarray, Y: ndarray) -> None:

        for i in range(self._epochs):
            print('Epoch: ', i+1)
            loss, info = forward(X, Y, self._weights) #TODO Compute forward propagation
            print('Loss: ', loss)
            grads = backward(info, self._weights) #TODO Compute backward propagation
            self._weights['W'] = self._weights['W'] - grads['W']*self._lr #TODO
            self._weights['B'] = self._weights['B'] - grads['B']*self._lr #TODO
