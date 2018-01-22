# -*- coding: utf-8 -*-
"""
@author: Oscar T Giles
@Email: o.t.giles@leeds.ac.uk

Plot the beta coefficients from the ordered probit models.

Notes:
The easiest way to run is to download the anaconda python distribution (https://www.anaconda.com/) which will install all the packages you need (except pymc2)
"""

from __future__ import division, print_function, unicode_literals #For python 2 (Best to run with python 3)

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker 
import scipy.stats as sts
from pymc.utils import hpd #Requires pymc2 to calculate the highest posterior density

x_names = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
              'Tracking', 'Balance:\nOpen', 'Balance:\nClosed']
x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
              'Tracking', 'Balance: Open', 'Balance: Closed']

invert = ['Steering', 'Aiming', 
              'Tracking']

beta_colors = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (1.0, 0.4980392156862745, 0.0),
 (1.0, 0.4980392156862745, 0.0)]

sns.set_context("paper", rc= {'axes.labelsize': 10})

fig, ax = plt.subplots(3, 8, figsize = (8.6, 3.8))


samples = pd.read_csv("..//MCMC_samples//maths_samples.csv", index_col = 0) #Get the data
beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]
sigma = samples['sigma']


for i, var in enumerate(x_names):

    beta_val = beta.values[:,i]
    sns.kdeplot(beta_val, ax = ax[0, i], color = beta_colors[i], shade = True)
    hdi = hpd(beta_val, 0.1)
    
    #Translate from axes coordiantes to data coorindates to get the y position of the errorbar line
    axis_to_data = ax[0, i].transAxes + ax[0, i].transData.inverted()
    y_pos = axis_to_data.transform((0, 0.08))[1]
    
    ax[0, i].errorbar(hdi.mean(), y_pos, xerr = hdi.mean() - hdi[0], color = 'k', elinewidth = 2)
    ax[0, i].plot(beta.values[:,i].mean(), y_pos, 'ok')
    if i > 1:
        ax[0, i].axvline(0, linestyle = '--', color = "0")

    ax[0, i].get_yaxis().set_ticks([])
    ax[0, i].set_xlabel(var)
    sns.despine()    
    


samples = pd.read_csv("..//MCMC_samples//readings_samples.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]
sigma = samples['sigma']

#x_names = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
#              'Tracking', 'Balance:\nOpen', 'Balance:\nClosed']
#x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
#              'Tracking', 'Balance: Open', 'Balance: Closed']


for i, var in enumerate(x_names):
    
    sns.kdeplot(beta.values[:,i], ax = ax[1, i], color = beta_colors[i], shade = True)
    hdi = hpd(beta.values[:,i], 0.1)
    
    #Translate from axes coordiantes to data coorindates to get the y position of the errorbar line
    axis_to_data = ax[1, i].transAxes + ax[1, i].transData.inverted()
    y_pos = axis_to_data.transform((0, 0.08))[1]
    
    ax[1, i].errorbar(hdi.mean(), y_pos, xerr = hdi.mean() - hdi[0], color = 'k', elinewidth = 2)
    ax[1, i].plot(beta.values[:,i].mean(), y_pos, 'ok')
    if i > 1:
        ax[1, i].axvline(0, linestyle = '--', color = "0")
    ax[1, i].get_yaxis().set_ticks([])

    ax[1, i].set_xlabel(var)
    sns.despine()    

 


samples = pd.read_csv("..//MCMC_Samples//writings_samples.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]
sigma = samples['sigma']

#x_names = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
#              'Tracking', 'Balance:\nOpen', 'Balance:\nClosed']
#x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
#              'Tracking', 'Balance: Open', 'Balance: Closed']


for i, var in enumerate(x_names):
    
    sns.kdeplot(beta.values[:,i], ax = ax[2, i], color = beta_colors[i], shade = True)
    hdi = hpd(beta.values[:,i], 0.1)
    
    #Translate from axes coordiantes to data coorindates to get the y position of the errorbar line
    axis_to_data = ax[2, i].transAxes + ax[2, i].transData.inverted()
    y_pos = axis_to_data.transform((0, 0.08))[1]
    
    ax[2, i].errorbar(hdi.mean(), y_pos, xerr = hdi.mean() - hdi[0], color = 'k', elinewidth = 2)
    ax[2, i].plot(beta.values[:,i].mean(), y_pos, 'ok')
    if i > 1:
        ax[2, i].axvline(0, linestyle = '--', color = "0")
    ax[2, i].get_yaxis().set_ticks([])

    ax[2, i].set_xlabel(var)
    sns.despine()    
    
    i += 1


##Configure all the axis ticks
loc = ticker.MultipleLocator(base=0.4) # this locator puts ticks at regular intervals
[ax[i, 1].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.04) # this locator puts ticks at regular intervals
[ax[i, 2].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.7) # this locator puts ticks at regular intervals
[ax[i, 3].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.9) # this locator puts ticks at regular intervals
[ax[i, 4].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.05) # this locator puts ticks at regular intervals
[ax[i, 5].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.03) # this locator puts ticks at regular intervals
[ax[i, 6].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.03) # this locator puts ticks at regular intervals
[ax[i, 7].xaxis.set_major_locator(loc) for i in range(3)]


loc = ticker.MultipleLocator(base=2) # this locator puts ticks at regular intervals
ax[2, 0].xaxis.set_major_locator(loc)
loc = ticker.MultipleLocator(base=3) # this locator puts ticks at regular intervals
ax[1,0].xaxis.set_major_locator(loc)


#Set all the column xlabels to be the same
[ax[i, 1].set_xlim([0.08, 2]) for i in range(3)]
[ax[i, 2].set_xlim([-0.06, 0.065]) for i in range(3)]
[ax[i, 3].set_xlim([-2.5, 0.05]) for i in range(3)]
[ax[i, 4].set_xlim([-2.5, 1]) for i in range(3)]
[ax[i, 5].set_xlim([-0.08, 0.06]) for i in range(3)]
[ax[i, 6].set_xlim([-0.04, 0.04]) for i in range(3)]
[ax[i, 7].set_xlim([-0.05, 0.04]) for i in range(3)]


ax[0,0].set_ylabel("Mathematics")
ax[1,0].set_ylabel("Reading")
ax[2,0].set_ylabel("Writing")

#Adjust the subplots
plt.subplots_adjust(top = 0.94, bottom = 0.16, left = 0.04, right = 0.96, hspace = 1, wspace = 0.2)

plt.show()