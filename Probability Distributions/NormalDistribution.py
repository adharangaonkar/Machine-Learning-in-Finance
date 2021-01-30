import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

'''Normal Distribution '''

u = np.random.normal(0,1,1);
mu1 = 100
Sigma1 = 225
r1 = mu1 + math.sqrt(Sigma1)*u #ask why have we used r1 here
sigma1 = math.sqrt(Sigma1)

x1 = np.linspace(-100, 200, 5000)
y1 = ss.norm.pdf(x1, mu1, sigma1)
plt.plot(x1, y1)

mu2 = 150
Sigma2 = 225
r2 = mu2 + math.sqrt(Sigma2)*u
sigma2 = math.sqrt(Sigma2)
x2 = np.linspace(100, 500, 5000)
u = 1*(np.random.uniform(0,1,5000)<=0.2)
y2 = u*y1+(1-u)*ss.norm.pdf(x2, mu2, sigma2)
plt.plot(x2,y2)

plt.show()