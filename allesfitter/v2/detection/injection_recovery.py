#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:41:19 2019

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
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itertools
from transitleastsquares import catalog_info

#::: my modules
from .transit_search import tls_search
from ..generator import inject_lc_model
from .injection_recovery_output import irplot
try:
    from exoworlds.tess import tessio
except:
    pass

#::: settings and constants
np.random.seed(42)
eps = 1e-12




###############################################################################
#::: helper functiomns
###############################################################################

#::: set up the logfile
def setup_logfile(logfname):
    '''
    Summary:
    --------
    - sets up a logfile for the injection-recovery tests
    - starts user dialog if a logfile already exists
    
    Inputs:
    -------
    logfname : str
        file path and name for the logfile
    
    Returns:
    -------
    ex : np struct array / None
        existing logfile
    '''
    if os.path.exists(logfname):
        response = input('Log file already exists. Do you want to (1) overwrite it, (2) append missing rows, or (3) abort?\n')
        if response == '1':   
            with open(logfname,'w') as f: 
                f.write('inj_period,inj_rplanet,tls_period,tls_depth,tls_duration,tls_epoch,tls_SDE,tls_SNR\n')
            return None
        elif response == '2': 
            ex = np.genfromtxt(logfname, names=True, dtype=None, delimiter=',')
            return ex
        else:
            raise ValueError('User aborted.') 
    else:
        with open(logfname,'w') as f: 
            f.write('inj_period,inj_rplanet,tls_period,tls_depth,tls_duration,tls_SDE,tls_SNR\n')
        return None                
        


#::: check if a certain injection has already been logged, or if it needs to be done
def to_do_or_not_to_do_that_is_the_question(ex, period, rplanet):
    #if no previous logfile exists, then do it
    if ex is None:
        return True #do it
    #if a previous logfile exists and it includes those period and rplanet values (to 1e-6 precision), then skip this pair
    else:
        for i in range(len(ex)):
            if (np.abs(period - ex['inj_period'][i]) < 1e-6) and (np.abs(rplanet - ex['inj_rplanet'][i]) < 1e-6):
                return False #skip it
    #if the pair was not skipped before, then go ahead and do it
    return True #do it
    


###############################################################################
#::: Inject a grid of ellc transits and TLS search on each lightcurve
###############################################################################
def inject_and_tls_search(time, flux, flux_err,                                #lightcurve
                          periods=1., rplanets=11., logfname='logfile.csv',    #for inject_lightcurve_model()
                          # SNR_threshold=5.,                                  #for tls_search()
                          # known_transits=None, 
                          # mask_multiplier=1.5,
                          # R_host=1., R_host_min=0.13, R_host_max=3.5, 
                          # M_host=1., M_host_min=0.1, M_host_max=1.,
                          # ldc=[0.4804, 0.1867],
                          # n_transits_min=3, 
                          # show_plot=False, save_plot=False, outdir='',
                          injection_params=None,
                          injection_settings=None,
                          tls_settings=None,
                          **kwargs):
    '''
    Summary:
    --------
    - injects a planet signal via ellc, for a given period and radius (at random epoch)
    - runs TLS on these injected data and infos
    
    Inputs:
    -------
    time : array of flaot
        time stamps of observations
    flux : array of flaot
        normalized flux
    flux_err : array of flaot
        error of normalized flux
        
    Optional Inputs:
    ----------------
    **kwargs : keyword arguments
        any other keyword arguments, see inject_lightcurve_model() and tls_search()
    
    Returns:
    -------
    Nothing, but a list of all TLS results will get saved to a log file
    '''

    #::: convert input
    periods = np.atleast_1d(periods)
    rplanets = np.atleast_1d(rplanets)
    if injection_params is None: injection_params = defaults.get_default_params()
    else: injection_params = defaults.fill_params()
        
        
    #::: set up a logfile
    ex = setup_logfile(logfname)
    
    #::: cycle through all period and R_companion
    print('\n', flush=True)
    for period, rplanet in tqdm(itertools.product(periods, rplanets), total=len(periods)*len(rplanets)): #combining the two for loops
        
        #::: check if this was already done
        if to_do_or_not_to_do_that_is_the_question(ex, period, rplanet):
            print('\tP = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth --> do')
            epoch = time[0] + np.random.random()*period
            
            #::: inject
            flux2 = inject_lc_model(time, flux, flux_err, **kwargs)
            
            #::: recover
            results_all = tls_search(time, flux2, flux_err,
                                     period_min=0.5*period, period_max=2*period, #NEW NEW NEW MUCH SPEED MUCH FAST
                                     transit_depth_min=0.5*(1.*rplanet/R_host)**2, #NEW NEW NEW MUCH SPEED MUCH FAST
                                     **kwargs)
            
            #::: check if something was found
            if len(results_all)>0:
                for r in results_all:
                    with open(logfname,'a') as f:
                        f.write(format(period, '.5f')+','+
                                format(rplanet, '.5f')+','+
                                format(r.period, '.5f')+','+
                                format(r.depth, '.5f')+','+
                                format(r.duration, '.5f')+','+
                                format(r.epoch, '.5f')+','+
                                format(r.SDE, '.5f')+','+
                                format(r.snr, '.5f')+'\n')
            else:
                with open(logfname,'a') as f:
                    f.write(format(period, '.5f')+','+
                            format(rplanet, '.5f')+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+'\n')
                
        else:
            print('\tP = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth --> skipped (already exists)')
    
    #::: finish
    irplot(logfname)
    print('Done.')
            



###############################################################################
#::: Inject an ellc transit and TLS search using tessio
###############################################################################
def inject_and_tls_search_by_tic(tic_id, sigma_multiplier=3, **kwargs):
    '''
    Summary:
    -------
    - wrapper around inject_and_tls_search()
    - retrieves the SPOC PDC-SAP lightcurve
    - retrieves all TIC catalog information from MAST
    - injects a planet signal via ellc, for a given period and radius (at random epoch)
    - runs TLS on these injected data and infos
    
    Inputs:
    -------
    tic_id : str or int
        TIC ID
        
    Optional Inputs:
    ----------------
    sigma_multiplier : float 
        set TLS' uniform bounds for the host's radius and mass at sigma_multiplier times the error bars from TICv8
        default is 3
    **kwargs : keyword arguments
        any other keyword arguments, see inject_and_tls_search()
        
    Returns:
    -------
    a list of all TLS results will get saved to a log file
    '''
    
    #::: format inputs
    tic_id = str(int(tic_id))

    #::: load data
    time, flux, flux_err = tessio.get(tic_id, pipeline='spoc', PDC=True, unpack=True)
    
    #::: load TIC info
    ldc, R_host, R_host_lerr, R_host_uerr, M_host, M_host_lerr, M_host_uerr = catalog_info(TIC_ID=int(tic_id))
    print('\nTICv8 info:')
    print('Quadratic limb darkening u_0, u_1', ldc[0], ldc[1])
    print('Stellar radius', R_host, '+', R_host_lerr, '-', R_host_uerr)
    print('Stellar mass', M_host, '+', M_host_lerr, '-', M_host_uerr)
    
    #::: run
    inject_and_tls_search(time, flux, flux_err,
                          R_host=R_host, R_host_min=R_host-sigma_multiplier*R_host_lerr, R_host_max=R_host+sigma_multiplier*R_host_uerr, 
                          M_host=M_host, M_host_min=M_host-sigma_multiplier*M_host_lerr, M_host_max=M_host+sigma_multiplier*M_host_uerr,
                          ldc=ldc,
                          **kwargs)
            


###############################################################################
#::: diagnostic TLS plots
###############################################################################
def tls_plot(results):
    
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    
    ax = axes[0]
    ax.axvline(results.period, alpha=0.4, lw=3)
    ax.set_xlim(np.min(results.period), np.max(results.period))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    ax.set_ylabel(r'SDE')
    ax.set_xlabel('Period (days)')
    ax.plot(results.period, results.power, color='k', lw=0.5)
    ax.set_title('Period '+format(results.period, '.5f') + ', SNR '+format(results.snr, '.1f'))
    ax.set_xlim(0, max(results.period))
#    plt.show(fig)
#    fig.savefig('plots/'+'Periodogram P = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth'+'.pdf', bbox_inches='tight')
#    plt.close(fig)

    ax = axes[1]
    ax.plot(results.model_folded_phase, results.model_folded_model, color='red')
    ax.scatter(results.folded_phase, results.folded_y, color='b', s=10, alpha=0.5, zorder=2)
    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.99,1.01)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Relative flux');
    ax.set_title('Period '+format(results.period, '.5f'))
#    plt.show(fig)
#    fig.savefig('plots/'+'TLS transit P = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth'+'.pdf', bbox_inches='tight')
#    plt.close(fig)
    
    
    
###############################################################################
#::: main
###############################################################################
if __name__ == '__main__':
    pass
    
    ###########################################################################
    #::: Example: injection recovery test with TLS and tessio
    ###########################################################################
    # tic_id = '269701147'
    # period = np.arange(2,160+eps,2) #in days
    # R_companion = np.arange(0.8,4+eps,0.1) #in R_earth
    # SNR_threshold = 3.
    # known_transits = {'epoch':[2458715.3547, 2458726.0526, 2458743.5534],
    #                   'period':[8.8806, 28.5810, 38.3497],
    #                   'duration':[3.09/24., 4.45/24., 5.52/24.]
    #                   }
    # logfname = 'TIC_269701147.csv'
    
    # inject_and_tls_search_by_tic(tic_id, period, R_companion, logfname, 
    #                               SNR_threshold=SNR_threshold,
    #                               known_transits=known_transits)
    
    

    