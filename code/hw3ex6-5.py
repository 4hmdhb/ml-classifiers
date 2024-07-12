import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

E = np.array([1, 1])
var = np.array([[2, 0], [0, 1]])

x, y = np.mgrid[-4:6:.01, -5:5:.01]
pos = np.dstack((x, y))

rv = multivariate_normal(E, var)

pdf1 = rv.pdf(pos)

E = np.array([-1, -1])
var = np.array([[2, 1], [1, 2]])

rv = multivariate_normal(E, var)

pdf2 = rv.pdf(pos)


# Plot the isocontours of the PDF
plt.figure(figsize=(8, 8))
plt.contour(x, y, pdf1-pdf2, levels=np.linspace(np.min(pdf1-pdf2), np.max(pdf1-pdf2), 11), cmap='jet')
plt.colorbar(label='Probability Density')
plt.title('Isocontours of a Difference of Bivariate Normal Distributions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

