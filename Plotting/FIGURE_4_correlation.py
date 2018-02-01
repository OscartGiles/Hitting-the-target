"""
@author: Oscar T Giles
@Email: o.t.giles@leeds.ac.uk

Plots for the correlation analysis

Notes:
The easiest way to run is to download the anaconda python distribution (https://www.anaconda.com/) which will install all the packages you need (except pymc2)
"""

from __future__ import division, print_function, unicode_literals

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import scipy.stats as sts
from matplotlib.ticker import MaxNLocator

#x_names = [r'$\beta_0$', 'Age', 'InT', 'Ckat:\nTracing', 'Ckat:\nAiming', 
#              'Ckat:\nTracking', 'Balance:\nOpen', 'Balance:\nClosed']
#x_names2 = [r'$\beta_0$', 'Age', 'InT', 'Ckat: Tracing', 'Ckat: Aiming', 
#              'Ckat: Tracking', 'Balance: Open', 'Balance: Closed']

#Online colors
beta_colors = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (1.0, 0.4980392156862745, 0.0),
 (1.0, 0.4980392156862745, 0.0)]

#Offline colors
beta_colors = ['k' for i in range(6)]

sns.set(context = "paper", style = "white", 
        rc= {'axes.labelsize': 10, 
             'axes.titlesize': 12,
             'xtick.labelsize': 10,
             'ytick.labelsize':10,
             'savefig.dpi' : 500}, 
            font = 'sans-serif')




rdata = pd.read_csv("..//Raw_data//master_concat_data.csv")
rdata = rdata.dropna()
    
def linear_regression(x, y):   
    
    slope, intercept, r_value, p_value, std_err = sts.linregress(x, y)

    pred_val = intercept + slope * x
        
    residuals = y - pred_val
    
    return slope, intercept, r_value, p_value, std_err, residuals


###############################################################
#-------------------------------------------------------------#
#--------------------Plot linear regression-------------------#
#-------------------------------------------------------------#
#-------------------------------------------------------------#
#-------------------------------------------------------------#
###############################################################

figure_sc, ax_sc = plt.subplots(1,4, figsize = (7, 2.2))

ax_sc[1].scatter(rdata['age'], rdata['Attainment_Maths'], color = beta_colors[0], alpha = 0.35, s = 25)
slope, intercept, r_value1, p_value1, std_err = sts.linregress(rdata['age'],rdata['Attainment_Maths'])
line = slope*rdata['age']+intercept
ax_sc[1].plot(rdata['age'], line, 'k')

ax_sc[2].scatter(rdata['age'], rdata['interception'], color = beta_colors[1], alpha = 0.35, s = 25)
slope, intercept, r_value2, p_value2, std_err = sts.linregress(rdata['age'],rdata['interception'])
line = slope*rdata['age']+intercept
ax_sc[2].plot(rdata['age'], line, 'k')

ax_sc[0].scatter(rdata['interception'], rdata['Attainment_Maths'], color = beta_colors[2], alpha = 0.35, s = 25)
slope, intercept, r_value3, p_value3, std_err = sts.linregress(rdata['interception'], rdata['Attainment_Maths'])
line = slope*rdata['interception'] + intercept
ax_sc[0].plot(rdata['interception'], line, 'k')

ax_sc[1].set_xlabel("Age")
ax_sc[1].set_ylabel("Mathematics Attainment")

ax_sc[2].set_xlabel("Age")
ax_sc[2].set_ylabel("IntT")

ax_sc[0].set_xlabel("IntT")
ax_sc[0].set_ylabel("Mathematics Attainment")

ax_sc[1].xaxis.set_major_locator(MaxNLocator(integer = True))
ax_sc[2].xaxis.set_major_locator(MaxNLocator(integer = True))

sns.despine()

int_resids = linear_regression(rdata['age'], rdata['interception'])[-1]
maths_resids = linear_regression(rdata['age'], rdata['Attainment_Maths'])[-1]


slope, intercept, r_value4, p_value4, std_err, resids = linear_regression(int_resids, maths_resids)

ax_sc[3].scatter(int_resids, maths_resids, color = beta_colors[5], alpha = 0.35, s = 25)

ax_sc[3].plot(int_resids, int_resids * slope + intercept, 'k')
sns.despine()

ax_sc[3].set_xlabel("IntT (Age controlled)")
ax_sc[3].set_ylabel("Mathematics Attainment\n(Age controlled)")

ax_sc[0].text(0.1, 0.90, "r = {:3.3f}".format(r_value1),  transform=ax_sc[0].transAxes)
ax_sc[1].text(0.1, 0.90, "r = {:3.3f}".format(r_value2),  transform=ax_sc[1].transAxes)
ax_sc[2].text(0.1, 0.90, "r = {:3.3f}".format(r_value3),  transform=ax_sc[2].transAxes)
ax_sc[3].text(0.1, 0.90, "r = {:3.3f}".format(r_value4),  transform=ax_sc[3].transAxes)


[ax_sc[i].text(-0.1, 1.22, label, transform=ax_sc[i].transAxes,va='top', ha='right', fontsize = 18) for i, label in enumerate(['a', 'b', 'c', 'd'])]

#plt.tight_layout()
plt.subplots_adjust(top=0.84,
                    bottom=0.2,
                    left=0.08,
                    right=0.945,
                    hspace=0.2,
                    wspace=0.86)

plt.show()

