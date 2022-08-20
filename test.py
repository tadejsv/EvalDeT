import numpy as np

a = np.array([np.arange(i, i + 10) for i in range(10)])


print(a[[[0], [2], [3]], [5, 7, 9]])
