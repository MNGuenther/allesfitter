#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:03:21 2018

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import os
import emcee
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
#solves python>=3.8 issues, see https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
from multiprocessing import Pool
from contextlib import closing
from time import time as timer

#::: warnings
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 

#::: allesfitter modules
from . import config
from .computer import update_params, calculate_lnlike_total
from .general_output import logprint
from .mcmc_output import print_autocorr




###############################################################################
#::: MCMC log likelihood
###############################################################################
def mcmc_lnlike(theta):
    
    params = update_params(theta)
    lnlike = calculate_lnlike_total(params)
    
#    lnlike = 0
#    
#    for inst in config.BASEMENT.settings['inst_phot']:
#        lnlike += calculate_lnlike(params, inst, 'flux')
#    
#    for inst in config.BASEMENT.settings['inst_rv']:
#        lnlike += calculate_lnlike(params, inst, 'rv')
#        
#    if np.isnan(lnlike) or np.isinf(lnlike):
#        lnlike = -np.inf

    return lnlike

    
        
###############################################################################
#::: MCMC log prior
###############################################################################
def mcmc_lnprior(theta):
    '''
    bounds has to be list of len(theta), containing tuples of form
    ('none'), ('uniform', lower bound, upper bound), or ('normal', mean, std)
    '''
    lnp = 0.        
    
    for th, b in zip(theta, config.BASEMENT.bounds):
        if b[0] == 'uniform':
            if not (b[1] <= th <= b[2]): 
                return -np.inf
        elif b[0] == 'normal':
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[2]) * np.exp( - (th - b[1])**2 / (2.*b[2]**2) ) )
        elif b[0] == 'trunc_normal':
            if not (b[1] <= th <= b[2]): 
                return -np.inf
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[4]) * np.exp( - (th - b[3])**2 / (2.*b[4]**2) ) )
        else:
            raise ValueError('Bounds have to be "uniform" or "normal". Input from "params.csv" was "'+b[0]+'".')
    return lnp



###############################################################################
#::: MCMC log probability
###############################################################################    
def mcmc_lnprob(theta):
    '''
    has to be top-level for  for multiprocessing pickle
    '''
    lp = mcmc_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
#        try:
        ln = mcmc_lnlike(theta)
        return lp + ln
#        except:
#            return -np.inf
        


###########################################################################
#::: MCMC fit
###########################################################################
def mcmc_fit(datadir):
    
    #::: init
    config.init(datadir)
    
    
    continue_old_run = False
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5')):
        overwrite = str(input(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5')+\
                              ' already exists.\n'+\
                              'What do you want to do?\n'+\
                              '1 : overwrite the save file\n'+\
                              '2 : append to the save file\n'+\
                              '3 : abort\n'))
        if (overwrite == '1'):
            continue_old_run = False
        elif (overwrite == '2'):
            continue_old_run = True
        else:
            raise ValueError('User aborted operation.')
            
    
    #::: continue on the backend / reset the backend
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5')) and not continue_old_run:
        #backend.reset(config.BASEMENT.settings['mcmc_nwalkers'], config.BASEMENT.ndim)
        os.remove(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'))
        
        
    #::: set up a fresh backend
    backend = emcee.backends.HDFBackend(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'))
    
    
    #::: helper fct
    def run_mcmc(sampler):
        
        #::: set initial walker positions
        if continue_old_run:
            p0 = backend.get_chain()[-1,:,:]
            already_completed_steps = backend.get_chain().shape[0] * config.BASEMENT.settings['mcmc_thin_by']
        else:
            p0 = config.BASEMENT.theta_0 + config.BASEMENT.init_err * np.random.randn(config.BASEMENT.settings['mcmc_nwalkers'], config.BASEMENT.ndim)
            already_completed_steps = 0
        
        #::: make sure the inital positions are within the limits
        for i, b in enumerate(config.BASEMENT.bounds):
            if b[0] == 'uniform':
                p0[:,i] = np.clip(p0[:,i], b[1], b[2]) 
        
        #::: if pre-runs are wished for, and if we are not continuing an existing run
        if continue_old_run==False:
            for i in range(config.BASEMENT.settings['mcmc_pre_run_loops']):
                logprint("\nRunning pre-run loop",i+1,'/',config.BASEMENT.settings['mcmc_pre_run_loops'])
                
                #::: run the sampler        
                sampler.run_mcmc(p0,
                                 config.BASEMENT.settings['mcmc_pre_run_steps'],
                                 progress=config.BASEMENT.settings['print_progress'])
                
                #::: get maximum likelhood solution
                log_prob = sampler.get_log_prob(flat=True)
                posterior_samples = sampler.get_chain(flat=True)
                ind_max = np.argmax(log_prob)
                p0 = posterior_samples[ind_max,:] + config.BASEMENT.init_err * np.random.randn(config.BASEMENT.settings['mcmc_nwalkers'], config.BASEMENT.ndim)
    
                #::: reset sampler and backend
                #backend.reset(config.BASEMENT.settings['mcmc_nwalkers'], config.BASEMENT.ndim)
                os.remove(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'))
                sampler.reset()
        
        #::: run the sampler        
        logprint("\nRunning full MCMC")
        sampler.run_mcmc(p0,
                         int((config.BASEMENT.settings['mcmc_total_steps'] - already_completed_steps)/config.BASEMENT.settings['mcmc_thin_by']), 
                         thin_by=int(config.BASEMENT.settings['mcmc_thin_by']), 
                         progress=config.BASEMENT.settings['print_progress'])
        
        return sampler


    #::: Run
    logprint("\nRunning MCMC...")
    logprint('--------------------------')
    t0 = timer()
    if config.BASEMENT.settings['multiprocess']:
        with closing(Pool(processes=(config.BASEMENT.settings['multiprocess_cores']))) as pool: #multiprocessing
#        with closing(Pool(cpu_count()-1)) as pool: #pathos
            logprint('\nRunning on', config.BASEMENT.settings['multiprocess_cores'], 'CPUs.')
            sampler = emcee.EnsembleSampler(config.BASEMENT.settings['mcmc_nwalkers'], 
                                            config.BASEMENT.ndim, 
                                            mcmc_lnprob,
                                            moves=config.BASEMENT.settings['mcmc_moves'],
                                            pool=pool, 
                                            backend=backend)
            sampler = run_mcmc(sampler)
        t1 = timer()
        timemcmc = (t1-t0)
        logprint("\nTime taken to run 'emcee' on", config.BASEMENT.settings['multiprocess_cores'], "cores is {:.2f} hours".format(timemcmc/60./60.))
        
    else:
        sampler = emcee.EnsembleSampler(config.BASEMENT.settings['mcmc_nwalkers'],
                                        config.BASEMENT.ndim,
                                        mcmc_lnprob,
                                        moves=config.BASEMENT.settings['mcmc_moves'],
                                        backend=backend)
        sampler = run_mcmc(sampler)
        t1 = timer()
        timemcmc = (t1-t0)
        logprint("\nTime taken to run 'emcee' on a single core is {:.2f} hours".format(timemcmc/60./60.))
    

    #::: Check performance and convergence
    logprint('\nAcceptance fractions:')
    logprint('--------------------------')
    logprint(sampler.acceptance_fraction)
    
    print_autocorr(sampler)
    
    
    #::: return a German saying
    try:
        with open(os.path.join(os.path.dirname(__file__), 'utils', 'quotes2.txt')) as dataset:
            return(np.random.choice([l for l in dataset]))
    except:
        return('42')
    
    