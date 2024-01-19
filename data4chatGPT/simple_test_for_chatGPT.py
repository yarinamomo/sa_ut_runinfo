import numpy as np

# Create two matrices with different shapes
matrix1 = np.random.rand(3, 4)
matrix2 = np.random.rand(2, 3)

# Attempt to perform matrix multiplication with mismatched shapes
result = np.dot(matrix1, matrix2)
