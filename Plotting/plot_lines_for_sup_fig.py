# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:38:55 2018

@author: pscog
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


beta_colors = [
 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (1.0, 0.4980392156862745, 0.0),
 (1.0, 0.4980392156862745, 0.0)]



x = np.linspace(0, 1, 1)

plt.figure(figsize = (3, 2.26))

y1 = 1
plt.plot(range(2), [y1, y1], color = '.8')
plt.plot([.3, .6], [y1, y1], color = 'k', linewidth = 1)
plt.plot(.3, y1, marker = 'o', markeredgecolor = 'k', markerfacecolor="w")
plt.plot(.6, y1, marker = 'o', color = 'k')

y2 = 0.5
plt.plot(range(2), [y2, y2], color = 'k', alpha = 0.2)
plt.plot([.2, .4], [y2, y2], color = beta_colors[2], linewidth = 1)
plt.plot(.2, y2, marker = 'o', markeredgecolor = beta_colors[2], markerfacecolor = 'w')
plt.plot(.4, y2, marker = 'o', color = beta_colors[2])


y3 = 0.0
plt.plot(range(2), [y3, y3], color = 'k', alpha = 0.2)
plt.plot([.5, .7], [y3, y3], color = beta_colors[0], linewidth = 1)
plt.plot(.5, y3, marker = 'o', markeredgecolor = beta_colors[0], markerfacecolor = 'w')
plt.plot(.7, y3, marker = 'o', color = beta_colors[0])
         
         
y4 = -0.5
plt.plot(range(2), [y4, y4], color = 'k', alpha = 0.2)
#plt.plot([.2, .4], [y4, y4], color = '#1651af', linewidth = 1)
plt.plot(.2, y4, marker = 'o', markeredgecolor = beta_colors[2], markerfacecolor = 'w')
#plt.plot(.4, y4, marker = 'o', color = '#1651af')


y5 = -1
plt.plot(range(2), [y5, y5], color = 'k', alpha = 0.2)
#plt.plot([.5, .7], [y5, y5], color = '#af1b15', linewidth = 1)
plt.plot(.5, y5, marker = 'o', markeredgecolor = beta_colors[0], markerfacecolor = 'w')
#plt.plot(.7, y5, marker = 'o', color = '#af1b15')
         
         
         
plt.axis('off')