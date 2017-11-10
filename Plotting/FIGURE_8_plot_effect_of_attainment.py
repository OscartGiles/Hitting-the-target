"""
@author: Oscar T Giles
@Email: o.t.giles@leeds.ac.uk

Notes:
The easiest way to run is to download the anaconda python distribution (https://www.anaconda.com/) which will install all the packages you need.
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import itertools
import bayesplotting as bp


sns.set_style("white")
sns.set_context("paper", rc= {'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9})

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
#all_beta = np.flip(all_beta, axis = -1)
#all_beta = np.diff(all_beta, axis = -1)

#Bit of a fudge to make all the age parameters positive. Else some will have negative age (we just want to absolute)
all_beta[:,1:] = all_beta[:,1:] * -1
all_beta[:,-1, -1] = all_beta[:,-1, -1] * -1
#Plot effect sizes
##Effect size plots
names = x_names[2:]
task_names = ['IntT', 'Steering', 'Aiming',  'Tracking']
model_names = ['Excluding Reading and Writing', 'Including Reading and Writing']
#model_names = ['One']
samp_names = range(all_beta.shape[0])

###Create pandas data frame of MCMC samples
all_names = list(itertools.product(samp_names, task_names, model_names))
index = pd.MultiIndex.from_tuples(all_names, names=['sample', 'Task', 'Model'])

s = pd.Series(all_beta.flatten(), index=index).reset_index()

def positive_mean(x):
    
    return np.abs(np.mean(x))

color_pal = ['k', '0.95']
f, ax = plt.subplots(1,1)
bp.bar_plot(x = "Task", y = 0, hue = "Model", data = s,  
            color = color_pal, ax = ax,  order = task_names, hue_order = model_names, width = 0.25, 
            edgecolor = ['k', 'k', 'k', 'k'], linewidth = 1)


bp.error_plot(x = "Task", y = 0, hue = "Model", data = s, 
              linestyle = "None", color = "k",               
              hpd_alpha = np.std, ax = ax, order = task_names, hue_order = model_names)

plt.legend(title = "Model predictors", bbox_to_anchor = (0.6, 0.9))
#ax.set_ylim([0, 0.9])
plt.xlabel("Predictor")
plt.ylabel("Mathematics: Equivilent change in age (months)")
sns.despine()