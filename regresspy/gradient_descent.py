from typing import Dict, Tuple

import numpy as np
from numpy import ndarray


def forward(X: ndarray, Y: ndarray, weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:
    W = weights['W']
    B = weights['B']

    N = np.dot(X, W)

    P = N + B

    loss = np.mean(np.power((Y - P), 2))

    info = {}
    info['X'] = X
    info['Y'] = Y
    info['N'] = N
    info['P'] = P

    return loss, info


def backward(info: Dict[str, ndarray], weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    dLdP = -2 * (info['Y'] - info['P'])

    dPdN = np.ones_like(info['N'])
    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(info['X'], (1, 0))

    dLdW = np.dot(dNdW, dLdN)

    dLdB = (dLdP * dPdB).sum(axis=0)

    gradients = {}
    gradients['W'] = dLdW
    gradients['B'] = dLdB

    return gradients