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
def tls_search(time, flux, flux_err, 
               SNR_threshold=5.,
               known_transits=None,
               R_star=1., R_star_min=0.13, R_star_max=3.5, 
               M_star=1., M_star_min=0.1, M_star_max=1.,
               ):
    '''
    Inputs:
    -------
    time : array of flaot
        time stamps of observations
    flux : array of flaot
        normalized flux
    flux_err : array of flaot
        error of normalized flux
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
    R_star : float
        radius of the star (e.g. median)
        default 1 R_sun (from TLS)
    R_star_min : float
        minimum radius of the star (e.g. 1st percentile)
        default 0.13 R_sun (from TLS)
    R_star_max : float
        maximum radius of the star (e.g. 99th percentile)
        default 3.5 R_sun (from TLS)
    M_star : float
        mass of the star (e.g. median)
        default 1. M_sun (from TLS)
    M_star_min : float
        minimum mass of the star (e.g. 1st percentile)
        default 0.1 M_sun (from TLS)
    M_star_max : float
        maximum mass of the star (e.g. 99th percentile)
        default 1. M_sun (from TLS)         
    Summary:
    -------
    This runs TLS on these data with the given infos
    
    Returns:
    -------
    List of all TLS results
    '''
    
    SNR = 1e12
    FOUND_SIGNAL = False
    results_all = []        
    
    #::: apply a mask (if wished so)
    if known_transits is not None:
        for period, duration, T0 in zip(known_transits['period'], known_transits['duration'], known_transits['epoch']):
            time, flux = mask(time, flux, period, duration, T0)
    
    #::: search for the rest
    while (SNR >= SNR_threshold) and (FOUND_SIGNAL==False):
        model = tls(time, flux)
        results = model.power(R_star=R_star, R_star_min=R_star_min, R_star_max=R_star_max, 
                              M_star=M_star, M_star_min=M_star_min, M_star_max=M_star_max,
#                              period_min=period_min, period_max=period_max,
                              n_transits_min=1, quiet=True, show_progress_bar=False)
        
        if results.snr >= SNR_threshold:
            time, flux = mask(time, flux, results.period, 2*results.duration, results.T0)
            results_all.append(results)
            
        SNR = results.snr
        
    return results_all



###############################################################################
#::: TLS search using tessio
###############################################################################
def tls_search_by_tic(tic_id,
                      SNR_threshold=5.,
                      known_transits=None):
    '''
    Inputs:
    -------
    tic_id : str
        TIC ID
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
        
    Summary:
    -------
        - retrieves the SPOC PDC-SAP lightcurve
        - retrieves all TIC catalog information from MAST
        - runs TLS on these data and infos
    
    Returns:
    -------
        - list of all TLS results
    '''
    
    #::: format inputs
    tic_id = str(int(tic_id))
    
    #::: load data and inject transit
    dic = tessio.get(tic_id, pipeline='spoc', PDC=True)
    time, flux, flux_err = dic['time'], dic['flux'], dic['flux_err']
    
    #::: load TIC info
    ab, R_star, R_star_lerr, R_star_uerr, M_star, M_star_lerr, M_star_uerr = catalog_info(TIC_ID=int(tic_id))
    print('TICv8 info:')
    print('Quadratic limb darkening a, b', ab[0], ab[1])
    print('Stellar radius', R_star, '+', R_star_lerr, '-', R_star_uerr)
    print('Stellar mass', M_star, '+', M_star_lerr, '-', M_star_uerr)
    
    return tls_search(time, flux, flux_err,
                      R_star=R_star, R_star_min=R_star-R_star_lerr, R_star_max=R_star+R_star_uerr, 
                      M_star=M_star, M_star_min=M_star-M_star_lerr, M_star_max=M_star+M_star_uerr,
                      known_transits=known_transits)



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