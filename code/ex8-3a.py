import numpy as np
from numpy.linalg import slogdet, solve
from scipy.stats import multivariate_normal

from scipy.sparse.linalg import cg


np.random.seed(444)  # For reproducibility
mnist = np.load('mnist-data-hw3.npz')
data = mnist['training_data']
data = data.reshape(60000,-1)
# Normalize
data = data / np.linalg.norm(data, axis=1, keepdims=True)
labels = mnist['training_labels']

def fit_gaussian(data, labels):
    means = []
    covariances = []
    for digit in range(10): # For each digit 0-9
        digit_data = data[labels == digit]
        mean = np.mean(digit_data, axis=0)
        covariance = np.cov(digit_data, rowvar=False)
        covariance += np.eye(covariance.shape[0]) * 1e-6
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def lda_score(x, mean, pooled_covariance):
    class_priors_log = multivariate_normal.logpdf(x, mean, pooled_covariance)
    sols, _ = cg(pooled_covariance, x)
    log_posts = np.dot(x, sols.T) - 1/2 * np.dot(mean, sols.T) + class_priors_log
    return log_posts

def heavy_computes(size):
    indices = np.random.choice(range(0, 60000), 10000, replace=False)
    validation_data = data[indices]
    validation_labes = labels[indices]
    #training batch
    mask = np.ones(60000, dtype=bool)  # Initialize mask with all True
    mask[indices] = False       # Set indices to remove to False
    training_data = data[mask]
    training_labels = labels[mask]

    means, covariances = fit_gaussian(training_data, training_labels)
    n_features = training_data.shape[1]
    pooled_covariance = np.zeros((n_features, n_features))
    for i in range(10):
        pooled_covariance += covariances[i] * (validation_labes == i).sum()
    pooled_covariance /= training_data.shape[0]
    predictions = []
    for sample in validation_data:
        scores = [lda_score(sample, m, pooled_covariance) for m in means]
        predictions.append(np.argmax(scores))
    counter = 0
    for i in range(len(validation_data)):
        if (predictions[i] != validation_labes[i]):
            counter += 1
    return 1 - (counter/len(validation_data))

print(heavy_computes(100))