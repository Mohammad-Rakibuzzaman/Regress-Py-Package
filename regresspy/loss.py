import numpy as np
from numpy import ndarray


def mae(pred: ndarray, label: ndarray) -> ndarray:
    return np.mean(np.abs(label - pred))


def sse(pred: ndarray, label: ndarray) -> ndarray:
    return np.sum(np.power((label - pred), 2))


def mse(pred: ndarray, label: ndarray) -> ndarray:
    return np.mean(np.power((label - pred), 2))


def rmse(pred: ndarray, label: ndarray) -> ndarray:
    return np.sqrt(np.mean(np.power((label - pred), 2)))