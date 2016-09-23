# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 16:34:30 2016

@author: ps09og
"""

from __future__ import division, print_function
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import pdb
import scipy.stats as sts
import pystan, patsy, pickle
import itertools as it
from pymc.utils import hpd
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker 
from mpl_toolkits.mplot3d import axes3d
import stanplotting as sp


data = pd.read_csv("10_01_16_ALL_METRICS.csv") #Get the data

data['dob'] = pd.to_datetime(data['DOB']) #Calculate participant date of birth
data['age'] = pd.to_datetime("2014/11/1") - data['dob']
data['age'] = (data['age'] / np.timedelta64(1, 'D')).astype(int) / 365.25
data['year_group'] = data['Reg'].apply(lambda x: x[0]).astype(int)
data = data[['year_group', 'age', 'interception', 'Ckat_Tracing', 'Ckat_aiming', 
    'Ckat_Tracking', 'Open', 'Closed', 'Attainment_Maths', 
    'Attainment_Reading', 'Attainment_Writing']]



##LETS TAKE A LOOK AT THE DATA SET HERE
#data.dropna(how = "any", inplace = True)
#
#g = sns.PairGrid(data[['age', 'interception', 'Ckat_Tracing',
#                       'Ckat_aiming', 'Ckat_Tracking', 'Attainment_Maths', 
#    'Attainment_Reading', 'Attainment_Writing']])
#g.map_diag(sns.kdeplot)
#g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);


corr = data[['age', 'interception', 'Ckat_Tracing',
                       'Ckat_aiming', 'Ckat_Tracking', 'Attainment_Maths', 
    'Attainment_Reading', 'Attainment_Writing']].corr()


##NOW LETS DO SOME BAYESIAN ANALYSIS
            
##Load or compile Stan model
sm = pickle.load(open('Ordered_Probit.pkl', 'rb')) #Load Stan Model

#sm = pystan.StanModel(file='Ordered_Probit.Stan') #Compile Stan Model
#with open('Ordered_Probit.pkl', 'wb') as f: #Save Stan model to file
#    pickle.dump(sm, f)
#    
patsy_str = "{} ~ age + interception + Ckat_Tracing + Ckat_aiming + Ckat_Tracking + Open + Closed".format('Attainment_Maths')
print(patsy_str)
Y, X = patsy.dmatrices(patsy_str, data = data) #Create lower Design Matrices

Y = np.asarray(Y).astype('int').flatten()
X = np.asarray(X)

stan_data = dict(y = Y, X = X, N = len(Y), J = np.unique(Y).shape[0],
                         Q = X.shape[1]) #Stan data with corrections made so all integer variables start at 1 (for Stan)
                         
fit = sm.sampling(data=stan_data, iter = 1000, chains=4, refresh = 1) #Fit model   
la = fit.extract(permuted = True)

sns.distplot(la['beta'][:,2])
                        
#    
#resample = True
#attainment_metric = ['Attainment_Maths', 'Attainment_Reading', 'Attainment_Writing']
#
#if resample:
#    
#    post_la = {}
#    for atn in attainment_metric:
#        
#        patsy_str = "{} ~ 0 + age + interception + Ckat_Tracing + Ckat_aiming + Ckat_Tracking + Open + Closed".format(atn)
#        print(patsy_str)
#        Y, X = patsy.dmatrices(patsy_str, data = data) #Create lower Design Matrices
#        X = np.asarray(X)
#        Y = np.asarray(Y).flatten().astype(int)
#        
#        stan_data = dict(y = Y, x = X, N = len(Y), K = np.unique(Y).shape[0],
#                         D = X.shape[1]) #Stan data with corrections made so all integer variables start at 1 (for Stan)
#        
#        fit = sm.sampling(data=stan_data, iter = 10000, chains=4, refresh = 10, init = "0") #Fit model  
#               
#      
#        post_la[atn] = fit.extract(permuted=True) #Extract fit and save to file
#        with open('print_fit_{}.txt'.format(atn), 'wb') as f: #Save fit print output to file
#            print(fit, file=f)
#    
#    with open('IT_PREDICTS_MATHS_LA.pkl', 'wb') as f:
#        pickle.dump(post_la, f)
#else:
#    post_la = pickle.load(open("IT_PREDICTS_MATHS_LA.pkl")) #Load fit from file

#
#resample_submodels = False
#if resample_submodels:   
#    
#    post_la_no_it = {}
#    for atn in attainment_metric:
#        
#        patsy_str = "{} ~ 0 + age + interception + Ckat_Tracing + Ckat_aiming + Ckat_Tracking + Open + Closed".format(atn)
#        print(patsy_str)
#        Y, X = patsy.dmatrices(patsy_str, data = data) #Create lower Design Matrices        
#        X = np.asarray(X)
#        X = X[:,[0,2,3,4,5,6]]
#        
#        Y = np.asarray(Y).flatten().astype(int)
#        
#        stan_data = dict(y = Y, x = X, N = len(Y), K = np.unique(Y).shape[0],
#                         D = X.shape[1]) #Stan data with corrections made so all integer variables start at 1 (for Stan)
#        
#        fit = sm.sampling(data=stan_data, iter = 10000, chains=4, refresh = 10, init = "0") #Fit model  
#        
#        
#        post_la_no_it[atn] = fit.extract(permuted=True) #Extract fit and save to file
#        with open('print_fit_no_it{}.txt'.format(atn), 'wb') as f: #Save fit print output to file
#            print(fit, file=f)      
#        
#      
#        log_lik = post_la_no_it[atn]['log_lik']            
#        np.savetxt("log_lik_no_it{}.csv".format(atn), log_lik, delimiter=',')
#        
#        del log_lik
#        del post_la_no_it[atn]['log_lik']       
#
#    
#    with open('IT_PREDICTS_MATHS_LA_no_it.pkl', 'wb') as f:
#        pickle.dump(post_la_no_it, f)
#else:
#    post_la_no_it = pickle.load(open("IT_PREDICTS_MATHS_LA_no_it.pkl")) #Load fit from file
#    
#
#
#
#f,ax = plt.subplots(3,1, sharex = True, sharey = True)
#i = 0
#for atn in attainment_metric:
#    
#    sp.errorplot(['age', 'interception', 'Ckat_Tracing', 'Ckat_aiming', 
#    'Ckat_Tracking', 'Open', 'Closed'], post_la[atn]['beta'], ax = ax[i],
#    ls = "None")
#    ax[i].set_xlabel(atn)
#    i += 1
#
#
#####Plotting function######
#def my_formatter(x, pos):
#    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
#    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
#    formatted as -.4)."""
#    val_str = '{:g}'.format(x)
#    if np.abs(x) > 0 and np.abs(x) < 1:
#        return val_str.replace("0", "", 1)
#    else:
#        return val_str
#
##-------------------------------------------------#
##-------------------------------------------------#
##--------------------PPC--------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##atn = attainment_metric[0]
##
##samples = np.array([1,3,5,8], dtype = np.int)
##patsy_str = "{} ~ 0 + age + interception + Ckat_Tracing + Ckat_aiming + Ckat_Tracking + Open + Closed".format(atn)
##print(patsy_str)
##Y, X = patsy.dmatrices(patsy_str, data = data) #Create lower Design Matrices
##y_rep = np.empty((len(samples), Y.shape[0]), dtype = np.int)
##
##
##for s in samples:
##    theta = np.empty((Y.shape[0], np.unique(Y).shape[0]))
##    
##    eta = np.dot(X, post_la[atn]['beta'][s])
##    
##    theta[:, 0] = 1 - sts.norm.cdf(eta - post_la[atn]['c'][s,0])
##    
##    for k in xrange(1,np.unique(Y).shape[0]-2):
##        theta[:, k] = sts.norm.cdf(eta - post_la[atn]['c'][s,k-1]) - sts.norm.cdf(eta - post_la[atn]['c'][s,k])
##        
##    theta[:,np.unique(Y).shape[0]-1] = sts.norm.cdf(eta - post_la[atn]['c'][s,-1])
##    
##    
##    for p in xrange(Y.shape[0]):
##        y_rep[s, p] = np.dot(xrange(1,15), np.random.multinomial(1, theta[p]))
##
#
##beta ~ normal(0, K);
##
##	for (n in 1:N) {
##		real eta;
##		eta <- x[n] * beta;
##		theta[1] <- 1 - Phi(eta - c[1]);
##		for (k in 2:(K-1)){
##			theta[k] <- Phi(eta - c[k-1]) - Phi(eta - c[k]);
##		}
##		theta[K] <- Phi(eta - c[K-1]);
##		y[n] ~ categorical(theta);
##	}
##        
#####PLOT BETA VALUES
#        
###Math
#sns.set_context("paper")
#sns.set_style("white")
#plt.figure(figsize = (7.2, 5.5))
#beta_colors = sns.color_palette("Set1", 4)
#sigma_colors = sns.color_palette("hls", 1)
#
#ax_math_intercept = plt.subplot2grid((3, 4), (0, 0), colspan = 1)
#ax_math_slope1 = plt.subplot2grid((3, 4), (0, 1), colspan = 1)
#ax_math_slope2 = plt.subplot2grid((3, 4), (0, 2), colspan = 1)
#ax_math_sigma = plt.subplot2grid((3, 4), (0, 3), colspan = 1)
#
#sns.kdeplot(post_la['Attainment_Maths']['beta_0_new'], color = beta_colors[0], shade = True, ax = ax_math_intercept, legend = False)
#sns.kdeplot(post_la['Attainment_Maths']['beta_new'][:,0], color = beta_colors[1], shade = True, ax = ax_math_slope1, legend = False)
#sns.kdeplot(post_la['Attainment_Maths']['beta_new'][:,1], color = beta_colors[2], shade = True, ax = ax_math_slope2, legend = False)
#
#ax_math_slope2.axvline(0, linestyle = 'dashed', color = 'k')
#
#sp.plot_HDI(post_la['Attainment_Maths']['beta_new'][:,1], y_value = 5.1, axis = ax_math_slope2, fmt = 'o')
#sns.kdeplot(post_la['Attainment_Maths']['new_sig'], color = beta_colors[3], shade = True, ax = ax_math_sigma, legend = False)
#
#
###Reading
#
#ax_reading_intercept = plt.subplot2grid((3, 4), (1, 0), colspan = 1, sharex = ax_math_intercept, sharey = ax_math_intercept)
#ax_reading_slope1 = plt.subplot2grid((3, 4), (1, 1), colspan = 1, sharex = ax_math_slope1, sharey = ax_math_slope1)
#ax_reading_slope2 = plt.subplot2grid((3, 4), (1, 2), colspan = 1, sharex = ax_math_slope2, sharey = ax_math_slope2)
#ax_reading_sigma = plt.subplot2grid((3, 4), (1, 3), colspan = 1, sharex = ax_math_sigma, sharey = ax_math_sigma)
#
#sns.kdeplot(post_la['Attainment_Reading']['beta_0_new'], color = beta_colors[0], shade = True, ax = ax_reading_intercept, legend = False)
#sns.kdeplot(post_la['Attainment_Reading']['beta_new'][:,0], color = beta_colors[1], shade = True, ax = ax_reading_slope1, legend = False)
#sns.kdeplot(post_la['Attainment_Reading']['beta_new'][:,1], color = beta_colors[2], shade = True, ax = ax_reading_slope2, legend = False)
#
#ax_reading_slope2.axvline(0, linestyle = 'dashed', color = 'k')
#
#sp.plot_HDI(post_la['Attainment_Reading']['beta_new'][:,1], y_value = 5.1, axis = ax_reading_slope2, fmt = 'o')
#sns.kdeplot(post_la['Attainment_Reading']['new_sig'], color = beta_colors[3], shade = True, ax = ax_reading_sigma, legend = False)
#
#
#
###Writing
#
#ax_writing_intercept = plt.subplot2grid((3, 4), (2, 0), colspan = 1, sharex = ax_math_intercept, sharey = ax_math_intercept)
#ax_writing_slope1 = plt.subplot2grid((3, 4), (2, 1), colspan = 1, sharex = ax_math_slope1, sharey = ax_math_slope1)
#ax_writing_slope2 = plt.subplot2grid((3, 4), (2, 2), colspan = 1, sharex = ax_math_slope2, sharey = ax_math_slope2)
#ax_writing_sigma = plt.subplot2grid((3, 4), (2, 3), colspan = 1, sharex = ax_math_sigma, sharey = ax_math_sigma)
#
#sns.kdeplot(post_la['Attainment_Writing']['beta_0_new'], color = beta_colors[0], shade = True, ax = ax_writing_intercept, legend = False)
#sns.kdeplot(post_la['Attainment_Writing']['beta_new'][:,0], color = beta_colors[1], shade = True, ax = ax_writing_slope1, legend = False)
#sns.kdeplot(post_la['Attainment_Writing']['beta_new'][:,1], color = beta_colors[2], shade = True, ax = ax_writing_slope2, legend = False)
#
#ax_writing_slope2.axvline(0, linestyle = 'dashed', color = 'k')
#
#sp.plot_HDI(post_la['Attainment_Writing']['beta_new'][:,1], y_value = 5.1, axis = ax_writing_slope2, fmt = 'o')
#sns.kdeplot(post_la['Attainment_Writing']['new_sig'], color = beta_colors[3], shade = True, ax = ax_writing_sigma, legend = False)
#
#
#sns.despine()
#
#
#
#
#
#######Labels and axis limits#####
#
#label_pad = 15
#fs = 18
#
#ax_writing_intercept.set_xlabel(r"$\alpha$", labelpad = label_pad, fontsize = fs)
##ax_reading_intercept.set_ylabel("Density")
#
#ax_writing_slope1.set_xlabel(r"$\beta_1$ (Age)", labelpad = label_pad, fontsize = fs)
#ax_writing_slope2.set_xlabel(r"$\beta_2$ (IT Score)", labelpad = label_pad, fontsize = fs)
#
##ax_writing_slope2.set_ylim([0, 45])
##ax_writing_slope2.set_xlim([-0.01, 0.08])
#ax_writing_sigma.set_xlabel(r"$\sigma$ (SD)", labelpad = label_pad, fontsize = fs)
#
#ax_writing_intercept.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
#ax_writing_slope1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
#ax_writing_slope2.xaxis.set_major_locator(ticker.MultipleLocator(0.03))
#ax_writing_sigma.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
#formatter = ticker.FuncFormatter(my_formatter)
#ax_writing_slope2.xaxis.set_major_formatter(formatter) 
#
#
#ax_writing_intercept.axes.get_yaxis().set_ticks([])
#ax_reading_intercept.axes.get_yaxis().set_ticks([])
#ax_math_intercept.axes.get_yaxis().set_ticks([])
#
#ax_writing_slope1.axes.get_yaxis().set_visible(False)
#ax_reading_slope1.axes.get_yaxis().set_visible(False)
#ax_math_slope1.axes.get_yaxis().set_visible(False)
#
#ax_writing_slope2.axes.get_yaxis().set_visible(False)
#ax_reading_slope2.axes.get_yaxis().set_visible(False)
#ax_math_slope2.axes.get_yaxis().set_visible(False)
#
#ax_writing_sigma.axes.get_yaxis().set_visible(False)
#ax_reading_sigma.axes.get_yaxis().set_visible(False)
#ax_math_sigma.axes.get_yaxis().set_visible(False)
#
#ax_writing_intercept.set_ylabel("Writing", fontsize = fs)
#ax_reading_intercept.set_ylabel("Reading", fontsize = fs)
#ax_math_intercept.set_ylabel("Maths", fontsize = fs)
#
#
##ADD TEXT LABELS TO EACH AXES (A, B, C...)
#ax_math_intercept.text(0, 1.35, 'A',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_math_intercept.transAxes,
#        color='k', fontsize=15)
#
#ax_reading_intercept.text(0, 1.35, 'E',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_reading_intercept.transAxes,
#        color='k', fontsize=15)
#        
#ax_writing_intercept.text(0, 1.35, 'I',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_writing_intercept.transAxes,
#        color='k', fontsize=15)
#        
#ax_math_slope1.text(0, 1.35, 'B',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_math_slope1.transAxes,
#        color='k', fontsize=15)
#        
#ax_reading_slope1.text(0, 1.35, 'F',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_reading_slope1.transAxes,
#        color='k', fontsize=15)
#        
#ax_writing_slope1.text(0, 1.35, 'J',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_writing_slope1.transAxes,
#        color='k', fontsize=15)
#        
#        
#ax_math_slope2.text(0, 1.35, 'C',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_math_slope2.transAxes,
#        color='k', fontsize=15)
#
#ax_reading_slope2.text(0, 1.35, 'G',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_reading_slope2.transAxes,
#        color='k', fontsize=15)
#        
#ax_writing_slope2.text(0, 1.35, 'K',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_writing_slope2.transAxes,
#        color='k', fontsize=15)        
#        
#        
#ax_math_sigma.text(0, 1.35, 'D',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_math_sigma.transAxes,
#        color='k', fontsize=15)
#
#ax_reading_sigma.text(0, 1.35, 'H',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_reading_sigma.transAxes,
#        color='k', fontsize=15)     
#
#
#ax_writing_sigma.text(0, 1.35, 'L',
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax_writing_sigma.transAxes,
#        color='k', fontsize=15)
#        
#        
#sns.despine()
#plt.tight_layout()
#
#
#####---------------3d Plot--------------------#####
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
#
#exp_by_year = data.groupby("year_group").median()
#exp_by_school = data.median()
#
#AGE = np.linspace(4,11,100)    
#INTERCEPTIVE = np.linspace(0,54, 100)
#ATTAINMENT = np.arange(1, 15, 1)
#year_groups = [1, 2, 3, 4, 5, 6] #corresponding year_group
#age_groups = ["5-6", "6-7", "7-8", "8-9", "9-10", "10-11"]
#
#n_levels = 14
#prob_above_k = np.empty((len(AGE), len(INTERCEPTIVE), n_levels))
#
#maths_la = post_la['Attainment_Maths']
#
#beta_n = 8 #THERE ARE 7 PARAMETERS
#beta_loc = np.empty(beta_n)
#
#beta_loc[0] = np.mean(maths_la['beta_0_new'])
#for b in range(0, beta_n-1):
#    
#    beta_loc[b+1] = np.mean(maths_la['beta_new'][:,b]) #Weird indexing to get the slopes out
#
#cutoffs = maths_la['new_c'].mean(0)
#sigma = maths_la['new_sig'].mean()
#
#
#try: 
##    raise AttributeError
#    prob_above_k = pickle.load(open( "prob_above_K.pickle", "rb" ))
#    print("Loaded prob_above_k")
#except (AttributeError, IOError):    
#    print("WARNING: prob_above_k NOT FOUND. Processing posterior samples")
#    prob_above_k = np.empty((len(AGE), len(INTERCEPTIVE), n_levels))
#    
#    for a in range(len(AGE)):
#        print(a - len(AGE))
#        for h in range(len(INTERCEPTIVE)):           
#      
#            data_vector = np.array((1, AGE[a], INTERCEPTIVE[h],  exp_by_school.loc['Ckat_Tracing'], 
#                                    exp_by_school.loc['Ckat_aiming'],  exp_by_school.loc['Ckat_Tracking'], 
#                                    exp_by_school.loc['Open'], exp_by_school.loc['Closed']))
#         
##            pdb.set_trace()
#            predicted = np.dot(data_vector, beta_loc)
#            theta = sp.prob_int_point( predicted, cutoffs, sigma, n_levels) #A vector of probabilities of falling into each accademic age category
#            
#          
#            for k in range(n_levels):
#                
#                prob_above_k[a, h, k] = theta[k:].sum() #The probability of being above the mean Attainment score for the year group/ nHits      
#    
#    pickle.dump(prob_above_k, open("prob_above_K.pickle", "wb" ))
#
#
#X_3d, Y_3d = np.meshgrid(AGE, INTERCEPTIVE)
#C = X_3d * beta_loc[1] + Y_3d * beta_loc[2] #COLOUR MAP SHOWS THE LMA SCALE
#C= C/C.max()  
#
#line_colors = sns.husl_palette(6, h =.2)
#D_fig = plt.figure()
#D_ax = D_fig.add_subplot(111, projection='3d')
#
#patches = np.empty(6, dtype = np.object)
#labels = np.empty(6, dtype = np.object)
#c = 0
#
#for k in [4, 6, 8, 10, 12, 14]:
#
#    D_ax.plot_surface(X_3d, Y_3d, prob_above_k[:,:,k-1].T, rstride = X_3d.shape[0]//18, 
#                      cstride = X_3d.shape[0]//7, color=line_colors[c], alpha = 0.6)
#    patches[c] = mpatches.Patch(color=line_colors[c], alpha = 0.6, label='%i' %k)
#    labels[c] = "Grade {}".format(k)
#    c += 1
#
#D_ax.set_xlabel("Age")
#D_ax.set_ylabel("IT Score")
#D_ax.set_ylim([0, 54])
#
#D_ax.yaxis.set_label_position("top")
#    
#D_ax.set_xlim([4, 11])
#D_ax.set_zlabel(r"P(Math Score $\geq$ k)")
#
#plt.gca().zaxis.set_major_formatter(formatter) 
#
#for i in D_ax.get_yaxis().majorTicks:
#    i.set_pad(0.2)
#D_ax.elev = 26
#D_ax.azim = -72
#
#D_ax.legend(title = "k", handles = patches.tolist(), loc = 0, bbox_to_anchor=(1.04,0.8)) ##Handles must be a python list. Not a numpy array 
##plt.legend(title = "k", loc = 0, bbox_to_anchor=(1.04,0.8))
#plt.tight_layout()
#plt.show()
#
#####---------------3d Plot Surface------------#####
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
##-------------------------------------------------#
#plot_surface = True
#
#if plot_surface:
#
#    age_groups = range(5,11) #corresponding year_group
#    
#    age_groups.reverse()
#    hits = [20, 25, 30, 35, 40, 45]    
#    n_levels = 14
#    prob_above_exp = np.empty((len(age_groups), len(hits), maths_la[maths_la.keys()[0]].shape[0]))
#    #
#    ##GRAPH PLOTTING VARS
#    N = len(age_groups)
#    ind = np.arange(len(hits))  # the x locations for the groups
#    
#    width = np.linspace(0, 0.5, N)
#    
#    markers = ['o', 's', 'v', '*', 'D', 'p', ]
#    line_colors = sns.color_palette("BuGn_r", 6)
#    plt.figure()
#    ##CALCULATE METRICS AND PLOT
#    
#    for a in range(len(age_groups)):
#        for h in range(len(hits)):
##            print (a, h)
#            yr_idx = age_groups[a]
#    
#            predicted = (maths_la['beta_0_new'] + maths_la['beta_new'][:,0] * yr_idx
#                        + maths_la['beta_new'][:,1] * hits[h] 
#                        + maths_la['beta_new'][:,2] * exp_by_school.loc[ 'Ckat_Tracing'] 
#                        + maths_la['beta_new'][:,3] * exp_by_school.loc['Ckat_aiming'] 
#                        + maths_la['beta_new'][:,4] * exp_by_school.loc['Ckat_Tracking'] 
#                        + maths_la['beta_new'][:,5] * exp_by_school.loc['Open'] 
#                        + maths_la['beta_new'][:,6] * exp_by_school.loc['Closed'])
#            
#            theta = sp.prob_int( predicted, maths_la['new_c'], maths_la['new_sig'], n_levels) #A vector of probabilities of falling into each accademic age category
#            prob_above_exp_temp = theta[:, int(round((exp_by_school.loc['Attainment_Maths']))):].sum(axis = 1) #The probability of being above the mean Attainment score for the year group/ nHits
#            
#            prob_above_exp[a, h] = prob_above_exp_temp 
#    
#            sp.plot_HDI(prob_above_exp[a, h], ind[h], vert = True, fmt = 'none', ecolor = 'k', capthick = 1, capsize = 10 )
#    
#    
#    
#    #PLOT DOTS
#    #pdb.set_trace()
#    
#    for a in range(len(age_groups)):
#        plt.plot(ind, prob_above_exp[a, :].mean(axis = 1), marker = markers[a], color = 'k', mfc = line_colors[a], mec = 'k', mew = 1, ms = 8,  label = age_groups[a])
#    
#    plt.legend(title = "Age", bbox_to_anchor=(0.95, 1), loc=2, labelspacing=1.5)
#    
#    plt.ylabel(r"P(Math Score $\geq$ 6)")
#    plt.xlabel('IT Score')
#    plt.xlim([-0.2, ind[-1] + width[-1] + 0.2])
#    plt.ylim([0,1.1])
#    plt.gca().yaxis.set_major_formatter(formatter) 
#    
#    plt.xticks(ind)
#    plt.gca().set_xticklabels(hits)
#    
#    sns.despine()
#    plt.show()    