# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:38:55 2018

@author: pscog
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 1, 1)

plt.figure(figsize = (3, 2.26))

y1 = 1
plt.plot(range(2), [y1, y1], color = 'k', alpha = 0.2)
plt.plot([.3, .6], [y1, y1], color = 'k', linewidth = 1)
plt.plot(.3, y1, marker = 'o', color = '0.8', alpha = 1)
plt.plot(.6, y1, marker = 'o', color = 'k')

y2 = 0.5
plt.plot(range(2), [y2, y2], color = 'k', alpha = 0.2)
plt.plot([.2, .4], [y2, y2], color = '#1651af', linewidth = 1)
plt.plot(.2, y2, marker = 'o', color = '#b3caef', alpha = 1)
plt.plot(.4, y2, marker = 'o', color = '#1651af')


y3 = 0.0
plt.plot(range(2), [y3, y3], color = 'k', alpha = 0.2)
plt.plot([.5, .7], [y3, y3], color = '#af1b15', linewidth = 1)
plt.plot(.5, y3, marker = 'o', color = '#f9ccca', alpha = 1)
plt.plot(.7, y3, marker = 'o', color = '#af1b15')
         
         
y4 = -0.5
plt.plot(range(2), [y4, y4], color = 'k', alpha = 0.2)
#plt.plot([.2, .4], [y4, y4], color = '#1651af', linewidth = 1)
plt.plot(.2, y4, marker = 'o', color = '#1651af', alpha = 1)
#plt.plot(.4, y4, marker = 'o', color = '#1651af')


y5 = -1
plt.plot(range(2), [y5, y5], color = 'k', alpha = 0.2)
#plt.plot([.5, .7], [y5, y5], color = '#af1b15', linewidth = 1)
plt.plot(.5, y5, marker = 'o', color = '#af1b15', alpha = 1)
#plt.plot(.7, y5, marker = 'o', color = '#af1b15')
         
         
         
plt.axis('off')