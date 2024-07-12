import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

E = np.array([-1, 2])
var = np.array([[2, 1], [1, 4]])

x, y = np.mgrid[-5:5:.01, -3:7:.01]
pos = np.dstack((x, y))

rv = multivariate_normal(E, var)
pdf = rv.pdf(pos)

# Plot the isocontours of the PDF
plt.figure(figsize=(8, 8))
plt.contour(x, y, pdf, levels=np.linspace(np.min(pdf), np.max(pdf), 10), cmap='plasma')
plt.colorbar(label='Probability Density')
plt.title('Isocontours of a Bivariate Normal Distribution')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
