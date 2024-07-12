import numpy as np
import matplotlib.pyplot as plt


np.random.seed(4444)
n = 100
X1 = np.random.normal(3, 9, n)
X2 = 0.5 * X1 + np.random.normal(4, 4, n)
samples = np.column_stack((X1, X2))
mean = np.mean(samples, axis=0)

cov = np.cov(samples, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov)


eigenvector1 = eigenvectors[:,0] * np.sqrt(eigenvalues[0])
eigenvector2 = eigenvectors[:,1] * np.sqrt(eigenvalues[1])

plt.figure(figsize=(8, 8))

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)


plt.quiver(mean[0], mean[1], eigenvector1[0], eigenvector1[1], angles='xy', scale_units='xy', scale=1, color='b')
plt.quiver(mean[0], mean[1], eigenvector2[0], eigenvector2[1], angles='xy', scale_units='xy', scale=1, color='b')


plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Sample Data and Eigenvectors')

plt.gca().set_aspect('equal', 'box')

plt.grid(True)
plt.show()
