#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:46:47 2018

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

#::: allesfitter modules
from .. import config
from ..computer import update_params, calculate_model
from ..exoworlds_rdx.lightcurves import gp_decor
from ..exoworlds_rdx.lightcurves.index_transits import index_transits, index_eclipses
from ..exoworlds_rdx.rvs import estimate_jitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})






###############################################################################
#::: get a good initial guess for the baselines, errors & jitters
###############################################################################
def estimate_noise(datadir):
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
        f.write('#name,gp_ln_sigma_median,gp_ln_sigma_ll,gp_ln_sigma_ul,gp_ln_rho_median,gp_ln_rho_ll,gp_ln_rho_ul,ln_yerr_median,ln_yerr_ll,ln_yerr_ul\n')
                  
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
        f.write('#name,ln_yerr_median,ln_yerr_ll,ln_yerr_ul\n')
                
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
        




###############################################################################
#::: get a good initial guess for the baselines, errors & jitters
###############################################################################
def estimate_noise_out_of_transit(datadir, inst=None, **kwargs):
    
    #::: init
    config.init(datadir)
    
    #::: set up directory
    priordir = os.path.join(datadir,'priors')
    if not os.path.exists(priordir): os.makedirs(priordir)
    
    #::: set up phot summary file
    fname_summary = os.path.join(datadir,'priors','summary_phot.csv')
    with open( fname_summary, 'w+' ) as f:
        f.write('#name,gp_ln_sigma_median,gp_ln_sigma_ll,gp_ln_sigma_ul,gp_ln_rho_median,gp_ln_rho_ll,gp_ln_rho_ul,ln_yerr_median,ln_yerr_ll,ln_yerr_ul\n')
                  
    if inst is None:
        instruments = config.BASEMENT.settings['inst_phot']
    else:
        instruments = [inst]
        
    #::: run               
    for inst in instruments:
        
        key = 'flux'
        print('\n###############################################################################')
        print(inst + ' ' +key)
        print('###############################################################################')
        outdir = os.path.join(datadir,'priors',inst)
        fname = inst+'_'+key
        
        time = config.BASEMENT.fulldata[inst]['time']
        flux = config.BASEMENT.fulldata[inst][key]
        
        ind_in = []
        for companion in config.BASEMENT.settings['companions_phot']:
            epoch  = config.BASEMENT.params[companion+'_epoch']
            period = config.BASEMENT.params[companion+'_period']
            width  = config.BASEMENT.settings['fast_fit_width']
            if config.BASEMENT.settings['secondary_eclipse']:
                ind_ecl1, ind_ecl2, _ = index_eclipses(time,epoch,period,width,width) #TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1)
                ind_in += list(ind_ecl2)
            else:
                buf = list(index_transits(time,epoch,period,width)[0])
                ind_in += buf
        ind_in = np.sort(np.unique(ind_in))
    
        ind_out = []
        for i in np.arange(0, len(time), 1):
            if i not in ind_in:
                ind_out.append(i)
            
        
#        flux_err = config.BASEMENT.fulldata[inst]['err_scales_flux']
#        X = np.column_stack((time, flux, np.ones_like(flux)))
#        np.savetxt(inst+'_outoftransit.csv', X, delimiter=',')
                
        gp_decor(
                 time, flux, 
                 ind_in = ind_in, ind_out=ind_out,
                 multiprocess=config.BASEMENT.settings['multiprocess'], multiprocess_cores=config.BASEMENT.settings['multiprocess_cores'],
                 outdir=outdir, fname=fname, fname_summary=fname_summary, 
                 **kwargs
                 )
    
        
        


###############################################################################
#::: old
###############################################################################
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
        

    
    
###############################################################################
#::: old 2
###############################################################################
#def get_initial_guess_gp(datadir):
#    import celerite
#    from celerite import terms
#    from scipy.optimize import minimize
#    from .computer import update_params
#    
#    config.init(datadir)
#    base = config.BASEMENT
#    params = update_params(base.theta_0)
#    
#    for inst in base.settings['inst_phot']:
#        key = 'flux'
#        model = calculate_model(params, inst, key, xx=None)
#        x = base.data[inst]['time']
#        y = base.data[inst][key] - model
##        yerr_weights = config.BASEMENT.data[inst]['err_scales_'+key]
#        yerr = np.nanstd(y) #overwrite yerr; works better for removing smooth global trends
#        
#        kernel = terms.Matern32Term(log_sigma=1., log_rho=1.)
#        gp = celerite.GP(kernel, mean=np.nanmean(y)) 
#        gp.compute(x, yerr=yerr) #constrain on x/y/yerr
#         
#        def neg_log_like(gp_params, y, gp):
#            gp.set_parameter_vector(gp_params)
#            return -gp.log_likelihood(y)
#        
#        def grad_neg_log_like(gp_params, y, gp):
#            gp.set_parameter_vector(gp_params)
#            return -gp.grad_log_likelihood(y)[1]
#        
#        initial_params = gp.get_parameter_vector()
#        bounds = gp.get_parameter_bounds()
#        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
#                        method="L-BFGS-B", bounds=bounds, args=(y, gp))
##        gp.set_parameter_vector(soln.x)
#        
#        inv_sigma2_w = calculate_inv_sigma2_w(params, inst, key)
#    
#        print('baseline_gp1_'+key+'_'+inst + ':', soln.x[0])
#        print('baseline_gp2_'+key+'_'+inst + ':', soln.x[1])
#        print('inv_sigma2_'+key+'_'+inst + ':', np.nanmean(inv_sigma2_w))
