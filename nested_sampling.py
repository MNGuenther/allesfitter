#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:05:28 2018

@author:
Dr. Maximilian N. Guenther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import os
import dynesty
from scipy.special import ndtri
from scipy.stats import truncnorm
from multiprocessing import Pool
from contextlib import closing
import pickle
from time import time as timer

#::: warnings
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 

#::: allesfitter modules
from . import config
from .computer import update_params, calculate_lnlike
from .general_output import show_initial_guess, logprint




###############################################################################
#::: Nested Sampling log likelihood
###############################################################################
def ns_lnlike(theta):
#    global config.BASEMENT
    
#    then = timer() #Time before the operations start
    params = update_params(theta)
    
    # normalisation
    #TODO
    
    lnlike = 0
    
#    def lnlike_1(inst, key):
#        residuals = calculate_residuals(params, inst, key)
#        inv_sigma2_w = calculate_inv_sigma2_w(params, inst, key, residuals=residuals)
#        lnlike_1 = -0.5*(np.nansum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w)))
#        return lnlike_1
    
    try:
        for inst in config.BASEMENT.settings['inst_phot']:
            lnlike += calculate_lnlike(params, inst, 'flux') #lnlike_1(inst, 'flux')
        
        for inst in config.BASEMENT.settings['inst_rv']:
            lnlike += calculate_lnlike(params, inst, 'rv') #lnlike_1(inst, 'rv')
    except:
        lnlike = -np.inf
        
#    now = timer() #Time after it finished
#    print("lnlike took: ", now-then, " seconds")
    return lnlike



###############################################################################
#::: Nested Sampling prior transform
###############################################################################
def ns_prior_transform(utheta):
#    global config.BASEMENT
    
    theta = np.zeros_like(utheta)*np.nan
    for i in range(len(theta)):
        if config.BASEMENT.bounds[i][0]=='uniform':
            theta[i] = utheta[i]*(config.BASEMENT.bounds[i][2]-config.BASEMENT.bounds[i][1]) + config.BASEMENT.bounds[i][1]
        elif config.BASEMENT.bounds[i][0]=='normal':
            theta[i] = config.BASEMENT.bounds[i][1] + config.BASEMENT.bounds[i][2]*ndtri(utheta[i])
        elif config.BASEMENT.bounds[i][0]=='trunc_normal':
            theta[i] = my_truncnorm_isf(utheta[i],config.BASEMENT.bounds[i][1],config.BASEMENT.bounds[i][2],config.BASEMENT.bounds[i][3],config.BASEMENT.bounds[i][4]) 
        else:
            raise ValueError('Bounds have to be "uniform", "normal" and "trunc_normal". Input from "params.csv" was "'+config.BASEMENT.bounds[i][0]+'".')
    return theta
    

def my_truncnorm_isf(q,a,b,mean,std):
    a_scipy = 1.*(a - mean) / std
    b_scipy = 1.*(b - mean) / std
    return truncnorm.isf(q,a_scipy,b_scipy,loc=mean,scale=std)



###############################################################################
#::: Nested Sampling fitter class
###############################################################################
def ns_fit(datadir):
    
    #::: init
    config.init(datadir)
    
    #::: show initial guess
    show_initial_guess()
        
        
    #::: settings
    nlive  = config.BASEMENT.settings['ns_nlive']    # (default 500) number of live points
    bound  = config.BASEMENT.settings['ns_bound']    # (default 'single') use MutliNest algorithm for bounds
    ndim   = config.BASEMENT.ndim                    # number of parameters
    sample = config.BASEMENT.settings['ns_sample']   # (default 'auto') random walk sampling
    tol    = config.BASEMENT.settings['ns_tol']      # (defualt 0.01) the stopping criterion
        
     
    #::: run
    if config.BASEMENT.settings['ns_modus']=='static':
        logprint('\nRunning Static Nested Sampler...')
        logprint('--------------------------')
        t0 = timer()
        
        if config.BASEMENT.settings['multiprocess']:
             with closing(Pool(processes=(config.BASEMENT.settings['multiprocess_cores']))) as pool:
                logprint('\nRunning on', config.BASEMENT.settings['multiprocess_cores'], 'CPUs.')
                sampler = dynesty.NestedSampler(ns_lnlike, ns_prior_transform, ndim, 
                                                pool=pool, queue_size=config.BASEMENT.settings['multiprocess_cores'], 
                                                bound=bound, sample=sample, nlive=nlive)
                sampler.run_nested(dlogz=tol, print_progress=True)
            
        else:
            sampler = dynesty.NestedSampler(ns_lnlike, ns_prior_transform, ndim,
                                            bound=bound, sample=sample, nlive=nlive)
            sampler.run_nested(dlogz=tol, print_progress=True)
            
        t1 = timer()
        timedynesty = (t1-t0)
        logprint("\nTime taken to run 'dynesty' (in static mode) is {} hours".format(int(timedynesty/60./60.)))


    elif config.BASEMENT.settings['ns_modus']=='dynamic':
        logprint('\nRunning Dynamic Nested Sampler...')
        logprint('--------------------------')
        t0 = timer()
        
        if config.BASEMENT.settings['multiprocess']:
             with closing(Pool(processes=config.BASEMENT.settings['multiprocess_cores'])) as pool:
                logprint('\nRunning on', config.BASEMENT.settings['multiprocess_cores'], 'CPUs.')
                sampler = dynesty.DynamicNestedSampler(ns_lnlike, ns_prior_transform, ndim, 
                                                       pool=pool, queue_size=config.BASEMENT.settings['multiprocess_cores'], 
                                                       bound=bound, sample=sample)
                sampler.run_nested(nlive_init=nlive, dlogz_init=tol, print_progress=True)
            
        else:
            sampler = dynesty.DynamicNestedSampler(ns_lnlike, ns_prior_transform, ndim,
                                                   bound=bound, sample=sample)
            sampler.run_nested(nlive_init=nlive, print_progress=True)
            
        t1 = timer()
        timedynestydynamic = (t1-t0)
        logprint("\nTime taken to run 'dynesty' (in dynamic mode) is {} hours".format(int(timedynestydynamic/60./60.)))


    #::: pickle-save the 'results' class
    results = sampler.results
    with open( os.path.join(config.BASEMENT.outdir,'save_ns.pickle'), 'wb') as f:
        pickle.dump(results, f)
    