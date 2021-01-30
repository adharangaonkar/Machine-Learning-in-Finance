import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

''' Uniform Distribution '''
u = np.random.uniform(0,1,1);
b = 270;
a = 230;
r0 = a+(b-a)*u;
x0 = np.linspace(a,b,100);
y0 = 1/(b-a)*np.ones((100,1));
plt.plot(x0, y0, 'r-', lw=5, alpha=0.6, label='uniform pdf');
plt.show()
