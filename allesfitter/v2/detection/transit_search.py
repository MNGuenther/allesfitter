#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:55:39 2020

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
import os
import numpy as np
import matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares as tls
from transitleastsquares import transit_mask, cleaned_array, catalog_info

#::: my modules
try:
    from exoworlds.tess import tessio
except:
    pass



###############################################################################
#::: helper functiomns
###############################################################################

#::: apply a mask (if wished so)
def mask(time, flux, period, duration, T0):
    intransit = transit_mask(time, period, duration, T0)
    time = time[~intransit]
    flux = flux[~intransit]
    time, flux = cleaned_array(time, flux)
    return time, flux
    


###############################################################################
#::: TLS search on an input lightcurve
###############################################################################
def tls_search(time, flux, flux_err,                                           #lightcurve
               SNR_threshold=5.,                                               #mine
               known_transits=None,                                            #mine
               mask_multiplier=1.5,                                            #mine
               show_plot=False, save_plot=False, outdir='',                    #mine
               R_host=1., R_host_min=0.13, R_host_max=3.5,                     #tls
               M_host=1., M_host_min=0.1, M_host_max=1.,                       #tls
               ldc=[0.4804, 0.1867],                                           #tls
               n_transits_min=3,                                               #tls
               **tls_kwargs):                                                  #tls
    '''
    Summary:
    -------
    runs TLS on these data with the given infos
    
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
    SNR_threshold : float
        the SNR threshold at which to stop the TLS search
    known_transits : None or dict
        if dict and one transit is already known: 
            known_transits = {'period':[1.3], 'duration':[2.1], 'epoch':[245800.0]}
        if dict and multiple transits are already known: 
            known_transits = {'period':[1.3, 21.0], 'duration':[2.1, 4.1], 'epoch':[245800.0, 245801.0]}
        'period' is the period of the transit
        'duration' must be the total duration, i.e. from first ingress point to last egrees point, in days
        'epoch' is the epoch of the transit
    R_host : float
        radius of the star (e.g. median)
        default 1 R_sun (from TLS)
    R_host_min : float
        minimum radius of the star (e.g. 1st percentile)
        default 0.13 R_sun (from TLS)
    R_host_max : float
        maximum radius of the star (e.g. 99th percentile)
        default 3.5 R_sun (from TLS)
    M_host : float
        mass of the star (e.g. median)
        default 1. M_sun (from TLS)
    M_host_min : float
        minimum mass of the star (e.g. 1st percentile)
        default 0.1 M_sun (from TLS)
    M_host_max : float
        maximum mass of the star (e.g. 99th percentile)
        default 1. M_sun (from TLS)    
    ldc : list
        quadratic limb darkening parameters
        default [0.4804, 0.1867]
    show_plot : bool
        show a plot of each phase-folded transit candidate and TLS model in the terminal 
    show_plot : bool
        save a plot of each phase-folded transit candidate and TLS model into outdir
    outdir : string
        if None, use the current working directory
    **tls_kwargs : keyword arguments
        any other TLS keyword arguments, e.g. period_min=0.4
        
    Returns:
    -------
    list of all TLS results
    '''
    
    SNR = 1e12
    FOUND_SIGNAL = False
    results_all = []    
    if len(outdir)>0 and not os.path.exists(outdir): os.makedirs(outdir)    
    
    #::: apply a mask (if wished so)
    if known_transits is not None:
        for period, duration, T0 in zip(known_transits['period'], known_transits['duration'], known_transits['epoch']):
            time, flux = mask(time, flux, period, duration, T0)
    
    #::: search for the rest
    i = 0
    while (SNR >= SNR_threshold) and (FOUND_SIGNAL==False):
        model = tls(time, flux, flux_err)
        results = model.power(R_star=R_host, R_star_min=R_host_min, R_star_max=R_host_max, 
                              M_star=M_host, M_star_min=M_host_min, M_star_max=M_host_max,
                              u=ldc, 
                              n_transits_min=n_transits_min, 
                              show_progress_bar=False,
                              **tls_kwargs)
        
        if results.snr >= SNR_threshold:
            time, flux = mask(time, flux, results.period, mask_multiplier*results.duration, results.T0)
            results_all.append(results)
        
            if show_plot or save_plot:
                fig, ax = plt.subplots()
                ax.plot(results['folded_phase'], results['folded_y'], 'b.')
                ax.plot(results['model_folded_phase'], results['model_folded_model'], 'r-', lw=2)
                ax.text( 1.02, 0.95, 'Period = ' + np.format_float_positional(results['period'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
                ax.text( 1.02, 0.85, 'Depth = ' + np.format_float_positional(1e3*(1.-results['depth']),4) + ' ppt', ha='left', va='center', transform=ax.transAxes )
                ax.text( 1.02, 0.75, 'Duration = ' + np.format_float_positional(24*(1.-results['duration']),4) + ' h', ha='left', va='center', transform=ax.transAxes )
                ax.text( 1.02, 0.65, 'Epoch = ' + np.format_float_positional(results['T0'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
                ax.text( 1.02, 0.55, 'SNR = ' + np.format_float_positional(results['snr'],4), ha='left', va='center', transform=ax.transAxes )
                if save_plot:
                    fig.savefig(os.path.join(outdir,'tls_signal_'+str(i)), bbox_inches='tight')
                if show_plot:
                    plt.show(fig)
                else:
                    plt.close(fig)
            
        SNR = results.snr
        i+=1
                
    return results_all



###############################################################################
#::: TLS search using tessio
###############################################################################
def tls_search_by_tic(tic_id, sigma_multiplier=3, **kwargs):
    '''
    Summary:
    -------
    - wrapper around tls_search()
    - retrieves the SPOC PDC-SAP lightcurve
    - retrieves all TIC catalog information from MAST
    - runs TLS on these data and infos
        
    Inputs:
    -------
    tic_id : str
        TIC ID
        
    Optional Inputs:
    ----------------
    sigma_multiplier : float 
        set TLS' uniform bounds for the host's radius and mass at sigma_multiplier times the error bars from TICv8
        default is 3
    **kwargs : keyword arguments
        any other keyword arguments, see tls_search_by_tic()
        
    
    Returns:
    -------
        - list of all TLS results
    '''
    
    #::: format inputs
    tic_id = str(int(tic_id))
    
    #::: load data and inject transit
    time, flux, flux_err = tessio.get(tic_id, pipeline='spoc', PDC=True, unpack=True)
    
    #::: load TIC info
    ldc, R_host, R_host_lerr, R_host_uerr, M_host, M_host_lerr, M_host_uerr = catalog_info(TIC_ID=int(tic_id))
    print('\nTICv8 info:')
    print('Quadratic limb darkening u_0, u_1', ldc[0], ldc[1])
    print('Stellar radius', R_host, '+', R_host_lerr, '-', R_host_uerr)
    print('Stellar mass', M_host, '+', M_host_lerr, '-', M_host_uerr)
    
    return tls_search(time, flux, flux_err,
                      R_host=R_host, R_host_min=R_host-sigma_multiplier*R_host_lerr, R_host_max=R_host+sigma_multiplier*R_host_uerr, 
                      M_host=M_host, M_host_min=M_host-sigma_multiplier*M_host_lerr, M_host_max=M_host+sigma_multiplier*M_host_uerr,
                      ldc=ldc,
                      **kwargs)



###############################################################################
#::: main
###############################################################################
if __name__ == '__main__':
    pass
    
    ###########################################################################
    #::: Example: search for a transit with TLS and tessio
    ###########################################################################
    # tic_id = '269701147'
    # SNR_threshold=5.,
    # known_transits = {'epoch':[2458715.3547, 2458726.0526, 2458743.5534],
    #                   'period':[8.8806, 28.5810, 38.3497],
    #                   'duration':[3.09/24., 4.45/24., 5.52/24.]
    #                  }
    
    # results_all = tls_search_by_tic(tic_id,
    #                 SNR_threshold=SNR_threshold,
    #                 known_transits=known_transits)
    # print(results_all)