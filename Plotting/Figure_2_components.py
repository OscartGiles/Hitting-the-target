# -*- coding: utf-8 -*-
"""
Author:
    Oscar Giles
"""


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import pandas as pd


#sns.set(context = "paper", style = "white", 
#        rc= {'axes.labelsize': 10, 
#             'axes.titlesize': 12,
#             'xtick.labelsize': 10,
#             'ytick.labelsize':10,
#             'savefig.dpi' : 500}, 
#            font = 'sans-serif')

#Plot steering A
steeringA = pd.read_csv("ref_paths/random_a.ref", sep = '\t', names = ['time', 'x', 'y']).values
y_max = 163
x_max = 260

plt.figure(figsize = (6,4))
plt.axis('equal')
plt.plot(steeringA[:,2], steeringA[:,1], color = 'k')

plt.plot(steeringA[-1,2], steeringA[-1,1], marker = 'o', markeredgecolor = 'k', markerfacecolor = '0.95', markersize = 15)
plt.plot(steeringA[0,2], steeringA[0,1], marker = 'o', markeredgecolor = 'k', markerfacecolor = 'k', markersize = 15)

#Draw bored
plt.gca().add_patch(
    patches.Rectangle(
        (0,0),   # (x,y)
        x_max,          # width
        y_max,          # height
        edgecolor = 'k',
        facecolor = 'none'
    )
)
    
plt.xlim([-5,  x_max+5])
plt.ylim([-5, y_max+5])


plt.axis('off')


##Plot Steering B
steeringB = pd.read_csv("ref_paths/random_b.ref", sep = '\t', names = ['time', 'x', 'y']).values

plt.figure(figsize = (6,4))
plt.axis('equal')
plt.plot(steeringB[:,2], steeringB[:,1], color = 'k')

plt.plot(steeringB[-1,2], steeringB[-1,1], marker = 'o', markeredgecolor = 'k', markerfacecolor = '0.95', markersize = 15)
plt.plot(steeringB[0,2], steeringB[0,1], marker = 'o', markeredgecolor = 'k', markerfacecolor = 'k', markersize = 15)

#Draw bored
plt.gca().add_patch(
    patches.Rectangle(
        (0,0),   # (x,y)
        x_max,          # width
        y_max,          # height
        edgecolor = 'k',
        facecolor = 'none'
    )
)
    
plt.xlim([-5,  x_max+5])
plt.ylim([-5, y_max+5])

plt.axis('off')


##Plot figure 8
fig8 = pd.read_csv("ref_paths/reference_path_figure8.ref", sep = '\t', names = ['time', 'x', 'y']).values

plt.figure(figsize = (6,4))
plt.axis('equal')
plt.plot(fig8[:,2], fig8[:,1], color = 'k')
plt.plot(fig8[15,2], fig8[15,1], marker = 'o', markeredgecolor = 'k', markerfacecolor = '0.95', markersize = 15)

#Draw bored
plt.gca().add_patch(
    patches.Rectangle(
        (0,0),   # (x,y)
        x_max,          # width
        y_max,          # height
        edgecolor = 'k',
        facecolor = 'none'
    )
)
    
plt.xlim([-5,  x_max+5])
plt.ylim([-5, y_max+5])

plt.axis('off')

plt.figure(figsize = (6,4))
plt.axis('equal')
plt.plot(fig8[:,2], fig8[:,1], '-', dashes=(5, 12), color = 'k')
plt.plot(fig8[15,2], fig8[15,1], marker = 'o', markeredgecolor = 'k', markerfacecolor = '0.95', markersize = 15)

#Draw bored
plt.gca().add_patch(
    patches.Rectangle(
        (0,0),   # (x,y)
        x_max,          # width
        y_max,          # height
        edgecolor = 'k',
        facecolor = 'none'
    )
)
    
plt.xlim([-5,  x_max+5])
plt.ylim([-5, y_max+5])
plt.axis('off')



#Plot Aiming task
theta = -np.pi/2
R = np.array([[np.cos(theta), np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])

start_pos = np.array([81, 220])
end_pos = np.array([81, 30]) 
dots_pos = np.array([[45.77, 176.91], 
                     [80.75, 69.275],
                     [115.724, 176.91], 
                     [24.195, 110.389],
                     [137.34, 110.389]]) 
    

dots_pos[:,1] = dots_pos[:,1] * - 1 + x_max
start_pos[1] = start_pos[1] * - 1 + x_max
end_pos [1] = end_pos [1] * - 1 + x_max


plt.figure(figsize = (4,6))
plt.axis('equal')
plt.plot(dots_pos[:,0], dots_pos[:,1], 'o', markerfacecolor = 'w', markeredgecolor = 'k', ms = 15)

plt.plot(start_pos[0], start_pos[1], 'o', color = 'r', ms = 1)
plt.plot(end_pos[0], end_pos[1], 'o', color = 'b', ms = 1)

plt.plot(dots_pos[3,0], dots_pos[3,1], 'o', markerfacecolor = 'k', markeredgecolor = 'k', ms = 15)
plt.plot(dots_pos[4,0], dots_pos[4,1], 'o',  markerfacecolor = '0.8', markeredgecolor = 'k', ms = 15)

#Draw bored
plt.gca().add_patch(
    patches.Rectangle(
        (0,0),   # (x,y)
        y_max,          # width
        x_max,          # height
        edgecolor = 'k',
        facecolor = 'none'
    )
)
    
plt.xlim([-5,  y_max])
plt.ylim([-5, x_max])

plt.axis('off')