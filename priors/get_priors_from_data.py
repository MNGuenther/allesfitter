#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:46:47 2018

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
#import numpy as np
#import matplotlib.pyplot as plt
import os
#from scipy.optimize import differential_evolution

#::: my modules
from exoplanets.lightcurves import gp_decor
from exoplanets.rvs import estimate_jitter

#::: allesfitter modules
from .. import config
from ..computer import update_params, calculate_model






def get_priors_from_data(datadir):
    global rv_err
    global data_minus_model
    
    
    #::: init
    config.init(datadir)
    
    #::: params
    params = update_params(config.BASEMENT.theta_0)
    
    #::: set up directory
    priordir = os.path.join(datadir,'priors')
    if not os.path.exists(priordir): os.makedirs(priordir)
    
      
    
    #::: set up phot summary file
    fname_summary = os.path.join(datadir,'priors','summary_phot.csv')
    with open( fname_summary, 'w+' ) as f:
        f.write('#name,gp_log_sigma_median,gp_log_sigma_ll,gp_log_sigma_ul,gp_log_rho_median,gp_log_rho_ll,gp_log_rho_ul,log_yerr_median,log_yerr_ll,log_yerr_ul\n')
                  
    #::: run               
    for inst in config.BASEMENT.settings['inst_phot']:
        
        key = 'flux'
        print('\n###############################################################################')
        print(inst + ' ' +key)
        print('###############################################################################')
        outdir = os.path.join(datadir,'priors',inst)
        fname = inst+'_'+key
        
        time = config.BASEMENT.data[inst]['time']
        model = calculate_model(params, inst, key)
        data_minus_model = config.BASEMENT.data[inst][key] - model
        gp_decor(
                     time, data_minus_model, 
                     multiprocess=config.BASEMENT.settings['multiprocess'], multiprocess_cores=config.BASEMENT.settings['multiprocess_cores'],
                     outdir=outdir, fname=fname, fname_summary=fname_summary
                     )
                
    
        
    #::: set up rv summary file
    fname_summary = os.path.join(datadir,'priors','summary_rv.csv')
    with open( fname_summary, 'w+' ) as f:
        f.write('#name,log_yerr_median,log_yerr_ll,log_yerr_ul\n')
                
    #::: run
    for inst in config.BASEMENT.settings['inst_rv']:
        
        key = 'rv'
        print('\n###############################################################################')
        print(inst + ' ' +key)
        print('###############################################################################')
        outdir = os.path.join(datadir,'priors',inst)
        fname = inst+'_'+key
                    
        time = config.BASEMENT.data[inst]['time']
        model = calculate_model(params, inst, key)
        data_minus_model = config.BASEMENT.data[inst][key] - model
        white_noise = config.BASEMENT.data[inst]['white_noise_'+key]
        
        estimate_jitter(
                        time, data_minus_model, white_noise,
                        multiprocess=config.BASEMENT.settings['multiprocess'], multiprocess_cores=config.BASEMENT.settings['multiprocess_cores'],
                        outdir=outdir, fname=fname, fname_summary=fname_summary
                        )
        
        
        #guess the total yerr = np.sqrt( rv_err**2 + jitter**2  )
#        yerr = np.nanstd(data_minus_model) * np.ones_like(data_minus_model)
        
        #re-calcualte an initital guess for log_jitter 
#        initial = np.nanmedian( np.log( np.sqrt( yerr**2 - rv_err**2 ) ) )
#        print(initial)
        
#        def neg_log_like(theta):
#            log_jitter = theta
#            jitter = np.exp(log_jitter)
#            yerr = np.sqrt( rv_err**2 + jitter**2 )
#            inv_sigma2_w = 1./yerr**2
#            return +0.5*(np.nansum((data_minus_model)**2 * inv_sigma2_w - np.log(inv_sigma2_w)))
        
        
        #::: Diff. Evol.
#        bounds = [(-23,0)]
#        soln_DE = differential_evolution(neg_log_like, bounds)      

#        print(inst, key)
##        print('\tguess:', initial, 'lnlike:', neg_log_like(initial))
#        print('\tDE:', soln_DE.x[0], 'lnlike:', neg_log_like(soln_DE.x[0]))
        
    
    
        
