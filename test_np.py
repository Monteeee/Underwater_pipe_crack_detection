import numpy as np

a = np.array([[3, 1], [4, 2], [1, 3]])
print(a)
b = np.argmax(a, axis=1) 
print(b)
