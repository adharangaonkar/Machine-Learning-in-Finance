import numpy as np
import matplotlib.pyplot as plt

mu = 10
sigma = 3
s = np.random.lognormal(mu, sigma, 1000)
x = np.linspace(0, 100, 1000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2*sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()