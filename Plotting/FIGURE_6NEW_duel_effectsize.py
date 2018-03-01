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
import matplotlib.ticker as ticker 
#From the current working directory
import bayesplotting as bp


#Maths
x_names = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
              'Tracking', 'Balance:\nOpen', 'Balance:\nClosed']
x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Steering', 'Aiming', 
              'Tracking', 'Balance: Open', 'Balance: Closed']
#sp.errorplot(x_names, beta.values, ls = "None")


sns.set(context = "paper", style = "white", 
        rc= {'axes.labelsize': 10, 
             'axes.titlesize': 12,
             'xtick.labelsize': 10,
             'ytick.labelsize':10,
             'savefig.dpi' : 1000,
             'xtick.major.size': 2,
             'xtick.minor.size': 0.0,}, 
            font = 'sans-serif')



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

f, ax = plt.subplots(1,2, figsize = (6.9, 3.2), sharey = True)


bp.bar_plot(x = "Task", y = 0, hue = "Attainment", data = s, estimator = np.mean, 
            color = color_pal, ax = ax[0],  order = names, hue_order = outcome, width = 0.25, edgecolor = ['k', 'k', 'k', 'k'], linewidth = 1)


bp.error_plot(x = "Task", y = 0, hue = "Attainment", data = s, 
              linestyle = "None", color = "k",               
              hpd_alpha = np.std, ax = ax[0], order = names, hue_order = outcome)


ax[0].axhline(0, color = "0.75", linestyle = "--")
sns.despine()
ax[0].legend(title = "Attainment Measure", bbox_to_anchor = (0.45, 0.65))

ax[0].set_xlabel("Sensorimotor Task")
ax[0].set_ylabel("Equivalent change in age (months)")



##PanelB

samples_a = pd.read_csv("..//MCMC_samples//maths_samples_w_attainment.csv", index_col = 0) #Get the data
samples_no_a = pd.read_csv("..//MCMC_samples//maths_samples.csv", index_col = 0) #Get the data


#Get samples for attainment model
beta_names = [x for x in samples_a.columns if 'beta' in x]
beta_a = samples_a[beta_names]
sigma_a = samples_a['sigma']

#Get samples for no attainment model
beta_names = [x for x in samples_no_a.columns if 'beta' in x]
beta_no_a = samples_no_a[beta_names[:-2]] #Remove the open and closed balance betas
sigma_no_a = samples_no_a['sigma']


x_names = [r'$\beta_0$', 'Age', 'InT', 'Ckat: Tracing', 'Ckat: Aiming',  'CKAT: Tracking', 'reading', 'Writing']

beta_a = beta_a.values[:,:6]
beta_no_a = beta_no_a.values


#The next three lines convert he beta values to the typical range metric. Uncomment for just beta values
effect_sd = [8.35, 0.48, 0.34, 10.91]
beta_a[:,2:] = (((beta_a[:, 2:] * effect_sd * 2).T / beta_no_a[:,1]).T) * 12
beta_no_a[:,2:] = (((beta_no_a[:, 2:] * effect_sd * 2).T / beta_no_a[:,1]).T) * 12

all_beta = np.stack((beta_a, beta_no_a)).swapaxes(0,1).swapaxes(1, 2)
all_beta = all_beta[:,2:]
all_beta = all_beta[:,:,::-1]


#Bit of a fudge to make all the age parameters positive. Else some will have negative age (we just want to absolute)
all_beta[:,1:] = all_beta[:,1:] * -1
all_beta[:,-1, -1] = all_beta[:,-1, -1] * -1
#Plot effect sizes
##Effect size plots
names = x_names[2:]
task_names = ['IntT', 'Steering', 'Aiming',  'Tracking']
model_names = ['Mathematics', 'Mathematics\n(Reading and Writing predictors)']
#model_names = ['One']
samp_names = range(all_beta.shape[0])

###Create pandas data frame of MCMC samples
all_names = list(itertools.product(samp_names, task_names, model_names))
index = pd.MultiIndex.from_tuples(all_names, names=['sample', 'Task', 'Model'])

s = pd.Series(all_beta.flatten(), index=index).reset_index()

def positive_mean(x):
    
    return np.abs(np.mean(x))

color_pal = ['k', '0.95']

bp.bar_plot(x = "Task", y = 0, hue = "Model", data = s,  
            color = color_pal, ax = ax[1],  order = task_names, hue_order = model_names, width = 0.25, 
            edgecolor = ['k', 'k', 'k', 'k'], linewidth = 1)

bp.error_plot(x = "Task", y = 0, hue = "Model", data = s, 
              linestyle = "None", color = "k",               
              hpd_alpha = np.std, ax = ax[1], order = task_names, hue_order = model_names)


plt.legend(title = "Model predictors", bbox_to_anchor = (0.33, 0.75), labelspacing = 1.05)
#ax.set_ylim([0, 0.9])
ax[1].set_xlabel("Sensorimotor Task")
ax[1].set_ylabel("Equivalent change in age (months)")


[ax[i].text(0.02, 1.135, label, transform=ax[i].transAxes,va='top', ha='right', fontsize = 18) for i, label in enumerate(['a', 'b'])]

sns.despine()

plt.subplots_adjust(top=0.880,
                    bottom=0.145,
                    left=0.078,
                    right=0.973,
                    hspace=0.2,
                    wspace=0.207)


bonus_figures = False
if bonus_figures:
    f, ax = plt.subplots(1,1, sharex = 'col', figsize = (8, 3.5))
    color_pal = ['k', 'w', "0.75"]
        
    
    plt.axhline(0, color = "0.0", linestyle = "--")
    
    out =bp.plot_violin(x = "Task", y = 0, hue = "Attainment", data = s, 
                 ax = ax,  order = names, hue_order = outcome, palette = color_pal, max_width = 0.25)
       
    
    bp.error_plot(x = "Task", y = 0, hue = "Attainment", data = s, 
                  linestyle = "None", color = "0.2",               
                  hpd_alpha = 0.05, ax = ax, order = names, hue_order = outcome)
    
    plt.ylim([-5, 15])
    sns.despine()
    plt.tight_layout()


    
    f, ax = plt.subplots(1,4, sharey = True, figsize = (8, 2))
    color_pal = ['k', 'w', "0.75"]
    
    for i, name in enumerate(names):
        
        ax[i].axhline(0, color = "0.0", linestyle = "--")
              
        
        out  = bp.plot_violin(x = "Attainment", y = 0, data = s[s['Task'] == name], 
                     ax = ax[i],  order = outcome, palette = color_pal, vert = True)
        
        ax[i].set_title("{}".format(name))
                
        sns.despine()
        
    [ax[i].set_xticks(range(len(outcome))) for i in range(4)]
    
    [ax[i].set_xticklabels(outcome) for i in range(4)]
    
    #loc = ticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    #[ax[i].xaxis.set_major_locator(loc) for i in range(3)]
    #[ax[i].set_xlim([-5, 15]) for i in range(3)]
    
    plt.tight_layout()
    
    
    
    
    color_pal = ['w', '0.85', "0.55"]
    plt.figure()
    plt.axhline(0, color = "0.0", linestyle = "--")
    sns.violinplot(x = "Task", y = 0, hue = "Attainment", data = s, 
                   palette = color_pal, inner = None, 
                   bw = 1.5)
    
    sns.despine()
    
    #bp.point_plot(x = "Task", y = 0, hue = "Attainment", data = s, 
    #             ax = ax,  order = names, hue_order = outcome,  marker = 'o', linestyle = '', color = 'k')
    #
    #
    
    
    #
    #plt.axhline(0, color = "0.75", linestyle = "--")
    #sns.despine()
    #plt.legend(title = "Attainment Measure", bbox_to_anchor = (0.85, 0.9))