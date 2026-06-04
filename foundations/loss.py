import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1-eps)

        tmp = 0
        for i in range(len(y_true)):
            if y_true[i] != 1:
                tmp += -(np.log(1-y_pred[i]))
            else:
                tmp += -(np.log(y_pred[i]))
            
        tmp = tmp / len(y_true)

        return round(tmp, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1-eps)
        tmp = 0
        
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if y_true[i][j] != 1:
                    continue
                else:
                    tmp += -(np.log(y_pred[i][j]))

        tmp = tmp / len(y_true)

        return round(tmp, 4)
