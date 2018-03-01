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

x_names = [ 'Age', 'IntT', 'Steering', 'Aiming', 
              'Tracking', 'Balance: Open', 'Balance: Closed']
x_names2 = [ 'Age', 'IntT', 'Steering', 'Aiming', 
              'Tracking', 'Balance: Open', 'Balance: Closed']

invert = ['Steering', 'Aiming', 
              'Tracking', 'Balance:\nOpen', 'Balance:\nClosed']

beta_colors = [
 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (1.0, 0.4980392156862745, 0.0),
 (1.0, 0.4980392156862745, 0.0)]

sns.set(context = "paper", style = "white", 
        rc= {'axes.labelsize': 10, 
             'axes.titlesize': 12,
             'xtick.labelsize': 10,
             'ytick.labelsize':10,
             'savefig.dpi' : 1000,
             'xtick.major.size': 2,
             'xtick.minor.size': 0.0,}, 
            font = 'sans-serif')


fig, ax = plt.subplots(3, 7, figsize = (8.6, 3.8), sharex = 'col')


samples = pd.read_csv("..//MCMC_samples//maths_samples.csv", index_col = 0) #Get the data
beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names].values

beta = beta[:,1:] #Drop the intercept
sigma = samples['sigma']



for i, var in enumerate(x_names):

    beta_val = beta[:,i]
    sns.kdeplot(beta_val, ax = ax[0, i], color = beta_colors[i], shade = True)
    hdi = hpd(beta_val, 0.1)
    
    #Translate from axes coordiantes to data coorindates to get the y position of the errorbar line
    axis_to_data = ax[0, i].transAxes + ax[0, i].transData.inverted()
    y_pos = axis_to_data.transform((0, 0.08))[1]
    
    ax[0, i].errorbar(hdi.mean(), y_pos, xerr = hdi.mean() - hdi[0], color = 'k', elinewidth = 2)
    ax[0, i].plot(beta_val.mean(), y_pos, 'ok')
    if i > 0:
        ax[0, i].axvline(0, linestyle = '--', color = "0")

    ax[0, i].get_yaxis().set_ticks([])
    ax[0, i].set_xlabel(var)
    
    if var in invert:
        ax[0,i].invert_xaxis()
    sns.despine()    
    

samples = pd.read_csv("..//MCMC_samples//readings_samples.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names].values

beta = beta[:,1:] #Drop the intercept
sigma = samples['sigma']


for i, var in enumerate(x_names):
    
    beta_val = beta[:, i]  
    sns.kdeplot(beta_val, ax = ax[1, i], color = beta_colors[i], shade = True)
    hdi = hpd(beta_val, 0.1)
    
    #Translate from axes coordiantes to data coorindates to get the y position of the errorbar line
    axis_to_data = ax[1, i].transAxes + ax[1, i].transData.inverted()
    y_pos = axis_to_data.transform((0, 0.08))[1]
    
    ax[1, i].errorbar(hdi.mean(), y_pos, xerr = hdi.mean() - hdi[0], color = 'k', elinewidth = 2)
    ax[1, i].plot(beta_val.mean(), y_pos, 'ok')
    if i > 0:
        ax[1, i].axvline(0, linestyle = '--', color = "0")
    ax[1, i].get_yaxis().set_ticks([])

    ax[1, i].set_xlabel(var)
    sns.despine()    

 


samples = pd.read_csv("..//MCMC_Samples//writings_samples.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]
beta = samples[beta_names].values
beta = beta[:,1:] #Drop the intercept
sigma = samples['sigma']


for i, var in enumerate(x_names):
    
    beta_val = beta[:, i]
    sns.kdeplot(beta_val, ax = ax[2, i], color = beta_colors[i], shade = True)
    hdi = hpd(beta_val, 0.1)
    
    #Translate from axes coordiantes to data coorindates to get the y position of the errorbar line
    axis_to_data = ax[2, i].transAxes + ax[2, i].transData.inverted()
    y_pos = axis_to_data.transform((0, 0.08))[1]
    
    ax[2, i].errorbar(hdi.mean(), y_pos, xerr = hdi.mean() - hdi[0], color = 'k', elinewidth = 2)
    ax[2, i].plot(beta_val.mean(), y_pos, 'ok')
    if i > 0:
        ax[2, i].axvline(0, linestyle = '--', color = "0")
    ax[2, i].get_yaxis().set_ticks([])

    ax[2, i].set_xlabel(var)
    sns.despine()    
    
    i += 1


##Configure all the axis ticks
loc = ticker.MultipleLocator(base=0.3) # this locator puts ticks at regular intervals
[ax[i, 0].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.04) # this locator puts ticks at regular intervals
[ax[i, 1].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
[ax[i, 2].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
[ax[i, 3].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.05) # this locator puts ticks at regular intervals
[ax[i,4].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.04) # this locator puts ticks at regular intervals
[ax[i,5].xaxis.set_major_locator(loc) for i in range(3)]

loc = ticker.MultipleLocator(base=0.04) # this locator puts ticks at regular intervals
[ax[i,6].xaxis.set_major_locator(loc) for i in range(3)]


def format_func(value, tick_number):
    # find number of multiples of pi/2
    if value == 0.0:
        
        return '0'
    else:
        return value

ax[0,1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax[0,4].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax[0,5].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax[0,6].xaxis.set_major_formatter(plt.FuncFormatter(format_func))


ax[0,0].set_ylabel("Mathematics")
ax[1,0].set_ylabel("Reading")
ax[2,0].set_ylabel("Writing")

ax[0,0].set_xlim([0.82, 1.78])
ax[0,1].set_xlim([-0.05, 0.06])
ax[0,3].set_xlim([0.33, -2.3])
ax[0,4].set_xlim([0.023, -0.07])
ax[0,5].set_xlim([0.03, -0.06])
ax[0,6].set_xlim([0.03, -0.06])



#Add major labels
ax[2,3].text(0.5, -1.25, r'$\beta$ coefficients', transform=ax[2,3].transAxes, va='bottom', ha='center', fontsize = 18)
ax[1,0].text(-0.4, -1, 'Attainment Measure', transform=ax[1,0].transAxes, va='bottom', ha='center', rotation = 90, fontsize = 18)

#Adjust the subplots
plt.subplots_adjust(top=0.965,
                    bottom=0.2,
                    left=0.07,
                    right=0.985,
                    hspace=0.995,
                    wspace=0.190)

plt.show()