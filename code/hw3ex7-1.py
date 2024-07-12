import numpy as np

np.random.seed(4444)

n = 100

X1 = np.random.normal(3, 9, n)
X2 = 0.5 * X1 + np.random.normal(4, 4, n)

# Stack the samples into a 100x2 matrix
samples = np.column_stack((X1, X2))

sample_mean = np.mean(samples, axis=0)

print(sample_mean)
