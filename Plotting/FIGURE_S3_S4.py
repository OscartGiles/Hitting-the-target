"""
Created on Wed Mar 09 16:34:30 2016

@author: ps09og
"""


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import pdb
import scipy.stats as sts
import patsy, pickle
import itertools as it
from pymc.utils import hpd
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker 
from mpl_toolkits.mplot3d import axes3d
import stanplotting as sp
import matplotlib


samples = pd.read_csv("..//MCMC_samples//maths_samples2.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]
sigma = samples['sigma']

x_names = [r'$\beta_0$', 'Age', 'InT', 'Ckat:\nTracing', 'Ckat:\nAiming', 
              'Ckat:\nTracking', 'Balance:\nOpen', 'Balance:\nClosed']
x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Ckat: Tracing', 'Ckat: Aiming', 
              'Ckat: Tracking', 'Balance: Open', 'Balance: Closed']
#sp.errorplot(x_names, beta.values, ls = "None")

beta_colors = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
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
             'savefig.dpi' : 500}, 
            font = 'sans-serif')



###PPC and data plotting
rdata = pd.read_csv("..//Raw_data//master_concat_data.csv")
rdata = rdata.dropna()
    



############################################
###################PPC######################
############################################
############################################
############################################


y_true = rdata['Attainment_Maths'].values
y_rep = samples[[i for i in samples.columns if 'y_rep' in i]].values


#PPC test statistics
fig_pp, ax_pp = plt.subplots(1, 2, figsize = (5, 2), sharey = True)
sns.kdeplot(y_rep.mean(axis = 1), shade = True, alpha = 0.4, color = beta_colors[1], ax = ax_pp[0])
sns.kdeplot(y_rep.std(axis = 1), shade = True, alpha = 0.4, color = beta_colors[1],ax = ax_pp[1])


ax_pp[0].set_xlabel(r"Mean ($y^{rep}$)")
ax_pp[1].set_xlabel(r"Std ($y^{rep}$)")
[ax_pp[i].get_yaxis().set_visible(False) for i in range(2)]
ax_pp[0].axvline(y_true.mean(), color = '.3', linestyle = '--')
ax_pp[1].axvline(y_true.std(), color = '.3', linestyle = '--')

ax_pp[0].set_ylim([0, 3.6])
ax_pp[0].set_xlabel("Mean")
ax_pp[1].set_xlabel("SD")

sns.despine()


[ax_pp[i].text(-0.1, 1.22, label, transform=ax_pp[i].transAxes,va='top', ha='right', fontsize = 18) for i, label in enumerate(['a', 'b'])]

ax_pp[0].xaxis.set_major_locator(ticker.MultipleLocator(.25))
ax_pp[1].xaxis.set_major_locator(ticker.MultipleLocator(.25))

plt.subplots_adjust(top=0.81,
                    bottom=0.225,
                    left=0.09,
                    right=0.94,
                    hspace=0.2,
                    wspace=0.2)




plt.figure(figsize = (3.5, 3.5))
plt.plot(rdata['interception'], y_rep.mean(axis = 0), 'or', color = beta_colors[0], label = r"$\mathbb{E}(y^{rep})$")
plt.plot(rdata['interception'], y_true, 'ob', color = beta_colors[1], label = r"$y$" )
plt.xlabel("IntT")
plt.ylabel("Mathematics Attainment")
plt.legend(title = "Data", frameon = True, bbox_to_anchor = (0.35, 1))
sns.despine()
plt.tight_layout()


