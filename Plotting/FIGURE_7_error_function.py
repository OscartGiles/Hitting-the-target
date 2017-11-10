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
import scipy.stats as sts
import patsy
from matplotlib.ticker import MaxNLocator


samples = pd.read_csv("..//MCMC_samples//maths_all_no_attainment.csv", index_col = 0) #Get the data

beta_names = [x for x in samples.columns if 'beta' in x]
beta = samples[beta_names]
sigma = samples['sigma']

cut_names = [x for x in samples.columns if 'cuts' in x]
cuts = samples[cut_names]

preds = ['age', 'interception', 'Ckat_Tracing', 'Ckat_aiming',
       'Ckat_Tracking']

pred_string = "".join([" {} +".format(i) for i in preds])[:-2]
pred_string = "~" + pred_string

sns.set_context("paper")
sns.set_style("white")

rdata = pd.read_csv("..//Raw_data//maths_data.csv")
rdata = rdata.dropna()    


#Check how many data points fall outside of the expected value
#For every data point predict the range of values you would expect them to fall in
def prob_ob(x, K):
    
    p_vec = np.empty(K)
    
    for k in range(K):
        
        p_vec[k] = (x == k+1).sum()
        
    return p_vec / len(x)

def get_uniques(x, K):
    
    p_vec = np.empty(K)
    
    for k in range(K):
        
        p_vec[k] = (x == k).sum()
        
    return p_vec / len(x)
        

y = rdata['Attainment_Maths'].values
X = rdata[['age', 'interception', 'Ckat_Tracing', 'Ckat_aiming',
       'Ckat_Tracking']]
X2 = patsy.dmatrix(pred_string, data = X)


beta_vals = beta.values
sigma = sigma.values
cuts = cuts.values


def calc_theta(eta, cutoffs, sigma, K):
    """args:
        eta: predicted latent value
        cutoffs: cutoff estimates
        sigma: standard dev estimate
        K: number of possible outcome values
    """
    
    
    theta = np.empty((len(eta), K))    
   
        
    theta[:, 0] = 1 - sts.norm.cdf((eta - cutoffs[:,0]) / sigma )
    
    for k in range(1, K-1):
            
        theta[:, k] = sts.norm.cdf((eta-cutoffs[:, k-1])/sigma) - sts.norm.cdf((eta-cutoffs[:,k])/sigma)
   
    theta[:, K-1] = sts.norm.cdf((eta - cutoffs[:, K-2])/sigma)
    
    return theta



#Plot the effect of eta with mean cuts and sigma
plus_eta = 8.35 * 2 * beta['beta.3'].values

p1 = calc_theta(np.repeat(5, plus_eta.shape[0]), cuts, sigma, 14)
p2 = calc_theta(5 + plus_eta, cuts, sigma, 14)
p_contrast = (p2[:,4:].sum(axis = 1) - p1[:,4:].sum(axis = 1))


width = 0.4
offset = 0.2
f1, ax1 = plt.subplots(1,2)
ax1[0].bar(np.array(range(14))-offset, p1.mean(0), width =width, alpha = 0.75)
ax1[0].bar(np.array(range(14))+offset, p2.mean(0), width = width, alpha = 0.75)

ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[0].set_xticklabels(np.arange(1,15))

ax1[0].set_xlabel("Maths Outcome")
ax1[0].set_ylabel("P(Maths Outcome)")
ax1[0].set_xlim([0, 8])
sns.despine()

sns.kdeplot(p1[:,4:].sum(axis = 1), ax = ax1[1], shade = True, label = "5")
sns.kdeplot(p2[:,4:].sum(axis = 1), ax = ax1[1], shade = True, label = "5 + IntT typical range")
ax1[1].set_xlabel(r"P(Maths Outcome $\geq$ 5)" )
ax1[1].get_yaxis().set_ticks([])
ax1[1].legend(title = r"$\mu$ value", edgecolor = "k", bbox_to_anchor = (0.65, 1), frameon = True)
sns.despine()

p_contrast = (p2[:,4:].sum(axis = 1) - p1[:,4:].sum(axis = 1))



