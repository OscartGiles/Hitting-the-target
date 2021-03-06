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
import matplotlib.ticker as ticker


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

sns.set(context = "paper", style = "white", 
        rc= {'axes.labelsize': 10, 
             'axes.titlesize': 12,
             'xtick.labelsize': 10,
             'ytick.labelsize':10,
             'savefig.dpi' : 1000,
             'xtick.major.size': 2,
             'xtick.minor.size': 0.0,}, 
            font = 'sans-serif')

rdata = pd.read_csv("..//Raw_data//master_concat_data.csv")
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


x_labels = [str(i) for i in range(1, 14+1)]
#Plot the effect of eta with mean cuts and sigma
plus_eta = 8.35 * 2 * beta['beta.3'].values

p1 = calc_theta(np.repeat(5, plus_eta.shape[0]), cuts, sigma, 14)
p2 = calc_theta(5 + plus_eta, cuts, sigma, 14)
p_contrast = (p2[:,4:].sum(axis = 1) - p1[:,4:].sum(axis = 1))


width = 0.4
offset = 0.2
f1, ax1 = plt.subplots(1,2, sharex = True, figsize = (6, 3))
ax1[0].bar(np.array(range(1, 14+1))-offset, p1.mean(0), width =width, label = r"$\mu_1$", color = '#ED9340', alpha = 0.8)
ax1[0].bar(np.array(range(1, 14+1))+offset, p2.mean(0), width = width, label = r"$\mu_2$", color = '#003C96', alpha = 0.8)


ax1[0].set_xlabel("Obs Attainment Score (y)")
ax1[0].set_ylabel(r"P(y$\vert \mu$)")
ax1[0].legend(title = "Latent\nAttainment Score",   loc = 1)

sns.despine()



#Plot 2
prob_ratio = np.log(p2/p1)

N = 100 #Number of samples to plot

idx = np.random.randint(0, N, size = N)


for samp_idx in idx:
    
    
    ax1[1].plot(range(1, 14+1), prob_ratio[samp_idx], alpha = 0.05, color = 'k')
    
ax1[1].plot(range(1, 14+1), prob_ratio.mean(0), alpha = 1, color = 'k')    
ax1[1].axhline(0, color = 'k', linestyle = '--')
ax1[1].set_xlabel("Obs Attainment Score (y)")
ax1[1].set_ylabel(r"$log[P(y \vert \mu_2)$ / $P(y \vert \mu_1)]$")


#[ax1[i].xaxis.set_major_locator(MaxNLocator(integer=True)) for i in range(2)]
#[ax1[i].set_xticklabels(np.arange(1,15)) for i in range(2)]
[ax1[i].set_xlim([1, 13]) for i in range(2)]


sns.despine()


#ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

[ax1[i].text(0.02, 1.135, label, transform=ax1[i].transAxes,va='top', ha='right', fontsize = 18) for i, label in enumerate(['a', 'b'])]


plt.subplots_adjust(top=0.877,
                    bottom=0.167,
                    left=0.111,
                    right=0.966,
                    hspace=0.2,
                    wspace=0.283)

plt.show()


