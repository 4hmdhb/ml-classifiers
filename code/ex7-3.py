import numpy as np

np.random.seed(4444)

n = 100

X1 = np.random.normal(3, 9, n)
X2 = 0.5 * X1 + np.random.normal(4, 4, n)

# Stack the samples into a 100x2 matrix
samples = np.column_stack((X1, X2))

# Compute the eigenvectors and eigenvalues of the covariance matrix

cov = np.cov(samples, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov)

print("Eigenvalues: ", eigenvalues)
print("Eigenvectors:", eigenvectors )


