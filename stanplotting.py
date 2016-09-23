# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:09:49 2016

@author: oscar


A library of functions for plotting stan models.
Only errotplot() is a general function. The rest are specific to certain files (remove them in future versions)
"""

import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
from pymc.utils import hpd #Worth implimenting an HPD function/ Or nick theres
import scipy.stats as sts
import pandas as pd
import pdb

def errorplot(x, y, hue = None, ax = None, estimator = np.mean, alpha = 0.05, 
              color = 'k', label_rotation = 'horizontal', marker = 'o', ls = '--', **kwargs):
    """
    x: X values or X labels
    y: nXm array where the first axis is the samples and the second is the xfactor
    ax: matplotlib axis to draw to
    estimator: A numpy function (must have the axis argument)
    kwargs: Any other plt.errorbar args
    
    """    
    
    if ax == None:
        ax = plt.gca()
    
    label = None 
        
    if y.ndim == 2:
    
        x_val = np.array(range(len(x)))    
        central_tendancy = estimator(y, axis = 0 )    
        hpd_vals = hpd(y, alpha)
        
        error = np.empty(hpd_vals.shape)
        error[0] = central_tendancy - hpd_vals[0]
        error[1] = hpd_vals[1] - central_tendancy
    
        ax.errorbar(x_val, central_tendancy, yerr = error, ecolor = color, fmt = "None", **kwargs)        
        ax.plot(x_val, central_tendancy, color = color, marker = marker, ls = ls)
        
    elif y.ndim ==3:
        
        x_val = np.array(xrange(len(x)))    
        central_tendancy = estimator(y, axis = 0)
        
        n_hue_levels = y.shape[-1]
        hue_width = 0.5 / n_hue_levels
        offsets = np.linspace(0, 0.5 - hue_width, n_hue_levels)
        offsets -= offsets.mean()        

        hpd_vals = hpd(y, alpha)
        
        
        for i in range(central_tendancy.shape[1]):   
            
            error = np.empty(hpd_vals.shape[:2])
            
            error[0] = central_tendancy[:,i] - hpd_vals[0,:,i]
            error[1] = hpd_vals[1,:,i] - central_tendancy[:,i]
            
            label = hue[i]
            
            if isinstance(color, list):      

                col = color[i]
            
            elif isinstance(color, str):
                col = color     
                
            
            ax.errorbar(x_val + offsets[i], central_tendancy[:,i], yerr = error, ecolor = col, 
                        fmt = "None", **kwargs)
            ax.plot(x_val + offsets[i], central_tendancy[:,i], color = col, marker = marker, ls = ls, label = label)
            
    else:
        raise AttributeError("Input has too many dimensions")
            
    
    ax.set_xticks(x_val)
    ax.set_xticklabels(x,rotation = label_rotation)    
    
    if y.ndim ==2:
        ax.set_xlim([-1, x_val.max() +1 ])    
        
    elif y.ndim ==3:
        ax.set_xlim([x_val[0] - offsets[i] - 0.5 , x_val[-1]+ offsets[-1] + 0.5 ])  



        
def plot_HDI(sampleVec, y_value, axis = None, vert = False, marker = 'o', **kwargs):
    """Plot the HDI as a error bar graph.
    Only plots one error bar at the moment
    Args:
    
    sampleVec: A vector of mcmc samples
    credMass: The mass within the HDI    
    """
    
    if axis == None:
        axis = plt.gca()
        

    
    theta_means = np.mean(sampleVec)
    
    #theta_hpd = HDI_of_MCMC(sampleVec, 0.95)
    theta_hpd = hpd(sampleVec, 0.05)
    theta_hpd = np.array((theta_means - theta_hpd[0], theta_hpd[1] - theta_means))
   
#    pdb.set_trace()
    if vert == False:
        axis.errorbar(theta_means, y_value, xerr = theta_hpd.reshape(2,1), marker = marker, color = 'k', **kwargs) ##reshape error to be a 2XN array
    else:
        axis.errorbar(y_value, theta_means, yerr = theta_hpd.reshape(2,1), marker = marker, color = 'k', **kwargs)
   
def prob_int_point(predicted_y, cutoffs, sigma, K):
    
    """Calculate theta. A vector of probabilities that sum to 1, indicating the probability of being in 
    each outcome K. A bit of a mess. But it works"""
           
    theta = np.empty(K)    
    theta.fill(9999)   
            
    eta = predicted_y
          
    theta[0] = 1 - sts.norm.cdf((eta - cutoffs[0]) / sigma )
    #        pdb.set_trace()
            
    for k in range(1, K-1):
        
        theta[k] = sts.norm.cdf((eta-cutoffs[k-1])/sigma) - sts.norm.cdf((eta-cutoffs[k])/sigma)
   
    theta[K-1] = sts.norm.cdf((eta - cutoffs[K-2])/sigma)
            
    return theta  
    
def prob_int(predicted_y, cutoffs, sigma, K):
    """A function only for the school attainment posterior. Can be removed in future version.
    Calculate theta. A vector of probabilities that sum to 1, 
    indicating the probability of being in each outcome K"""
  
    if isinstance(predicted_y, pd.Series): #If a panda series convert to a list
        
        predicted_y = predicted_y.values
        
    if isinstance(sigma, pd.Series): #If a panda series convert to a list
        
        sigma = sigma.values        
  

    if isinstance(predicted_y, np.float):
        
        theta = np.empty((len(sigma), K))
        predicted_y = [predicted_y]
        len_y = 1
        
    else:
        
        theta = np.empty((len(predicted_y), K))
        len_y = len(predicted_y)
        
        
    theta.fill(9999)
    
       
    if len_y > 1:        
            
        eta = predicted_y
      
        theta[:, 0] = 1 - sts.norm.cdf((eta - cutoffs[:,0]) / sigma )
       
        for k in range(1, K-1):
            
            theta[:, k] = sts.norm.cdf((eta-cutoffs[:, k-1])/sigma) - sts.norm.cdf((eta-cutoffs[:,k])/sigma)
   
        theta[:, K-1] = sts.norm.cdf((eta - cutoffs[:, K-2])/sigma)
            
        return theta
        
    elif len_y == 1:
        for i in range(len(sigma)):
            
            eta = predicted_y[0]

            theta[i, 0] = 1 - sts.norm.cdf((eta - cutoffs[i,0]) / sigma[i] )
    #        pdb.set_trace()
            
            for k in range(1, K-1):
                
                theta[i, k] = sts.norm.cdf((eta-cutoffs[i, k-1])/sigma[i]) - sts.norm.cdf((eta-cutoffs[i,k])/sigma[i])
       
            theta[i, K-1] = sts.norm.cdf((eta - cutoffs[i, K-2])/sigma[i])
         
        return theta  
        
def plot_HDI(sampleVec, y_value, axis = None, vert = False, marker = 'o', **kwargs):
    """Plot the HDI as a error bar graph.
    Only plots one error bar at the moment. Think THis is only useful for the motor_attainment_posterior file
    Args:
    
    sampleVec: A vector of mcmc samples
    credMass: The mass within the HDI    
    """
    
    if axis == None:
        axis = plt.gca()
        

    
    theta_means = np.mean(sampleVec)
    
    #theta_hpd = HDI_of_MCMC(sampleVec, 0.95)
    theta_hpd = hpd(sampleVec, 0.05)
    theta_hpd = np.array((theta_means - theta_hpd[0], theta_hpd[1] - theta_means))
   
#    pdb.set_trace()
    if vert == False:
        axis.errorbar(theta_means, y_value, xerr = theta_hpd.reshape(2,1), marker = marker, color = 'k', **kwargs) ##reshape error to be a 2XN array
    else:
        axis.errorbar(y_value, theta_means, yerr = theta_hpd.reshape(2,1), marker = marker, color = 'k', **kwargs)

