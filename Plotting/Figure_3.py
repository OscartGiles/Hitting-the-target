# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:03:16 2018

@author: oscar
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context = "paper", style = "white", 
        rc= {'axes.labelsize': 10, 
             'axes.titlesize': 12,
             'xtick.labelsize': 10,
             'ytick.labelsize':10,
             'savefig.dpi' : 500}, 
            font = 'sans-serif')

mu = 1
x = np.linspace(-3.5, 3.5, 1000)
y = sts.norm.pdf(x, mu, 1)

resid_x = -0.04

plt.figure(figsize = (4,3))

#Plot the residual 
#plt.plot([-1.43, -1.43], [0.04, sts.norm.pdf(-1.43)], color = '.6', linestyle = '-')

plt.arrow(resid_x, sts.norm.pdf(resid_x, mu, 1),  0, -sts.norm.pdf(resid_x, mu, 1), color = '.25', 
          length_includes_head = True, width = 0.01, head_width = 0.25, head_length = 0.01*3)

plt.text(resid_x-0.205, -0.05, r'$y_{i}^{*}$')

c = np.linspace(-2.4, 2.4, 4)
[plt.axvline(i, ymin = 0.07, ymax = 0.85, color = '0.7', linestyle = '--') for i in c]
plt.plot(x, y, color = 'k')


line_ypos = 0.6

#Add a lower line
plt.plot([-4, 4], [line_ypos,line_ypos], color = 'k')

#Plot mean
plt.plot(mu, line_ypos, 'o', color = 'k')
plt.text(mu -0.1, line_ypos+0.035, r'$\mu_i = X_{i}^{T} \beta$') 

#Add numerical labels
c_plus = np.append(c, 2.4+1.6)
[plt.text(c_plus[i] - 0.9, sts.norm.pdf(mu, mu, 1) + 0.05, l) for i, l in enumerate(['1', '2', '3', '4', '5'])]




#Plot sd

#plt.plot([-1,1], [sts.norm.pdf(0) + 0.035, sts.norm.pdf(0) + 0.035], '-', color = '0.35')
#plt.text(-0.13, sts.norm.pdf(0)/2 +0.025, r'$\sigma$') 

plt.ylim(ymin = -0.1, ymax = 0.65)
plt.xlim([-3.8, 3.8])
plt.axis('off')