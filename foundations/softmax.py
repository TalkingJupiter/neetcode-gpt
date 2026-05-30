import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        stable_z = []
        result = []
        max_z = max(z)
        for i in range(len(z)):
            stable_z.append(z[i] - max_z)
        
        sum_exp = np.sum(np.exp(stable_z))

        for i in range(len(z)):
            rst = round(np.exp(stable_z[i])/ sum_exp, 4)
            result.append(rst)
        
        return result
        
        




