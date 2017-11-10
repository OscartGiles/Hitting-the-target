# -*- coding: utf-8 -*-
"""
@author: Oscar T Giles
@Email: o.t.giles@leeds.ac.uk


Plot the effect sizes

Notes:
The easiest way to run is to download the anaconda python distribution (https://www.anaconda.com/) which will install all the packages you need (except pymc2)
"""

from __future__ import division, print_function, unicode_literals

import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
from pymc.utils import hpd

#From the current working directory
import bayesplotting as bp


#Maths
x_names = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
              'Tracking', 'Balance:\nOpen', 'Balance:\nClosed']
x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
              'Tracking', 'Balance: Open', 'Balance: Closed']
#sp.errorplot(x_names, beta.values, ls = "None")


sns.set_context("paper", rc= {'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9})
sns.set_style("white", {'xtick.major.size': 4,
               'xtick.minor.size': 2})


effect_sd = [8.35, 0.48, 0.34, 10.91, 11.04, 9.50] #These values are obtained from the SUR Bayesian analysis

#Effect size Math
samples = pd.read_csv("..//MCMC_samples//maths_samples.csv", index_col = 0) #Get the data
beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]


all_effect_size = np.empty((beta.shape[0], len(effect_sd), 3))

for i in range(len(effect_sd)):
  
    dat = ((beta.values[:,i+2] * effect_sd[i] * 2 ) / beta.values[:,1]) 
    dat = dat*12
    
    all_effect_size[:, i, 0] = dat
        

#Effect size Reading
samples = pd.read_csv("..//MCMC_samples//readings_samples.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]

for i in range(len(effect_sd)):
  
    dat = ((beta.values[:,i+2] * effect_sd[i] * 2 ) / beta.values[:,1]) 
    dat = dat*12
    
    all_effect_size[:, i, 1] = dat


#Effect size Writing
samples = pd.read_csv("..//MCMC_samples//writings_samples.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]


for i in range(len(effect_sd)):
  
    dat = ((beta.values[:,i+2] * effect_sd[i] * 2 ) / beta.values[:,1]) 
    dat = dat*12
    
    all_effect_size[:, i, 2] = dat



#Plot the effect sizes
names = x_names[2:]
outcome = ['Mathematics', 'Reading', 'Writing']
samp_names = range(all_effect_size.shape[0])

###Create pandas data frame of MCMC samples
all_names = list(itertools.product(samp_names, names, outcome))
index = pd.MultiIndex.from_tuples(all_names, names=['sample', 'Task', 'Attainment'])
s = pd.Series(all_effect_size.flatten(), index=index).reset_index()


s = s[(s["Task"] != 'Balance:\nOpen') & (s["Task"] != 'Balance:\nClosed')] #Remove balance tasks


def make_pos(row):
    """Make the biggest values positive"""
    if row['Task'] in ['Steering', 'Aiming', 'Tracking']:        
        return row[0] * -1    
    else:    
        return row[0]

s[0] = s.apply(lambda row: make_pos(row), axis = 1)
#df['Normalized'] = df.apply(lambda row : normalise_row, axis=1) 

names = ['InT', 'Steering', 'Aiming', 'Tracking']
#color_pal = sns.color_palette("Set1", 3)
color_pal = ['k', 'w', "0.75"]
hue_order = ['']

def positive_mean(x):
    
    return np.abs(np.mean(x))

f, ax = plt.subplots(1,1)


bp.bar_plot(x = "Task", y = 0, hue = "Attainment", data = s, estimator = np.mean, 
            color = color_pal, ax = ax,  order = names, hue_order = outcome, width = 0.25, edgecolor = ['k', 'k', 'k', 'k'], linewidth = 1)


bp.error_plot(x = "Task", y = 0, hue = "Attainment", data = s, 
              linestyle = "None", color = "k",               
              hpd_alpha = np.std, ax = ax, order = names, hue_order = outcome)


plt.axhline(0, color = "0.75", linestyle = "--")
sns.despine()
plt.legend(title = "Attainment Measure", bbox_to_anchor = (0.85, 0.9))

plt.xlabel("Sensorimotor Task")
plt.ylabel("Equivalent change in age (months)")
plt.show()

