# Student's t Distribution

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
"""
Lesser the degree of freedom, fatter the tails are for a Students T Distribution
"""
df = 4
x = np.linspace(-5, 5, 1000)
plt.plot(x, ss.t.pdf(x, df),'r-', lw=3, alpha=0.6, label='t pdf')


df = 50
x = np.linspace(-5, 5, 1000)
plt.plot(x, ss.t.pdf(x, df),'b-.', lw=3, alpha=0.6, label='t pdf')

df = 30
plt.plot(x, ss.t.pdf(x, df),'r:', lw=3, alpha=0.6, label='t pdf')

plt.plot(x, ss.norm.pdf(x),'k:', lw=3, alpha=0.6, label='normal pdf')

"""
Degree of Freedom when 30 for a Students T Distribution resembles a Normal Distribution
"""

df = 100
x = np.linspace(-5, 5, 1000)
plt.plot(x, ss.t.pdf(x, df),'b-', lw=3, alpha=0.6, label='t pdf')

plt.show()