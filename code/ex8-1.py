import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Separate MNIST data into validation and training batches, also flatten image
mnist = np.load('mnist-data-hw3.npz')
data = mnist['training_data']
data = data.reshape(60000,-1)
labels = mnist['training_labels']


# Normalize
data = data / np.linalg.norm(data, axis=1, keepdims=True)


def fit_gaussian(data, labels):
    means = []
    covariances = []
    for digit in range(10): # For each digit 0-9
        digit_data = data[labels == digit]
        mean = np.mean(digit_data, axis=0)
        covariance = np.cov(digit_data, rowvar=False)

        means.append(mean)
        covariances.append(covariance)
    return means, covariances

means, covariances = fit_gaussian(data, labels)


plt.figure(figsize=(10,10))
plt.title(f"Covariance Matrix for Digit Class 4")
sns.heatmap(covariances[4], square=True, cmap='viridis')
plt.show()

# Extract the diagonal terms from the covariance matrix
diagonal_terms = np.diag(covariance_matrix)



