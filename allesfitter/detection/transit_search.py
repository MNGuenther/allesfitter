#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:55:39 2020

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
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from pprint import pprint
from datetime import datetime
from astropy import units as u
from astropy import constants as c
from astropy.stats import sigma_clip
from astropy.timeseries import BoxLeastSquares as bls
from ..exoworlds_rdx.lightcurves.index_transits import index_transits
# import time as timer
import contextlib

#::: specific modules
try:
    from wotan import flatten
except ImportError:
    pass

try:
    from transitleastsquares import transitleastsquares as tls
    from transitleastsquares import transit_mask, catalog_info
except ImportError:
    pass

#::: my modules
try:
    from exoworlds.tess import tessio
except:
    pass
from ..exoworlds_rdx.lightcurves.lightcurve_tools import plot_phase_folded_lightcurve, rebin_err  
from ..time_series import clean, slide_clip
from ..lightcurves import tessclean
from ..inout import write_json, write_csv
from ..plotting import fullplot, brokenplot, tessplot, monthplot


#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

    

###############################################################################
#::: print to logfile
###############################################################################
def logprint(*text, options=None):
    original = sys.stdout
    try:
        with open(os.path.join(options['outdir'],'logfile.log'), 'a' ) as f:
            sys.stdout = f
            print(*text)
    except OSError:
        pass #For unknown reasons, the combination of open() and os.path.join() does not work on some Windows versions
    sys.stdout = original
    
    
    
###############################################################################
#::: pretty-print to logfile
###############################################################################
def logpprint(*text, options=None):
    original = sys.stdout
    try:
        with open(os.path.join(options['outdir'],'logfile.log'), 'a' ) as f:
            sys.stdout = f
            pprint(*text)
    except OSError:
        pass #For unknown reasons, the combination of open() and os.path.join() does not work on some Windows versions
    sys.stdout = original


    
###############################################################################
#::: apply a mask (if wished so)
###############################################################################
def mask(time, flux, flux_err, period, duration, T0):
    intransit = transit_mask(time, period, duration, T0)
    time = time[~intransit]
    flux = flux[~intransit]
    if flux_err is not None:
        flux_err = flux_err[~intransit]
        time, flux, flux_err = clean(time, flux, flux_err)
    else:
        time, flux = clean(time, flux)
    return time, flux, flux_err
    


###############################################################################
#::: check for multiples of a value (e.g., of a period)
###############################################################################
def is_multiple_of(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b) 
    result = a % b
    return (abs(result/b) <= tolerance) or (abs((b-result)/b) <= tolerance)



###############################################################################
#::: BLS search on an input lightcurve
###############################################################################
def bls_search(time, flux, flux_err=None):
    if flux_err is None: 
        ind = np.where(~np.isnan(time*flux))
        time = np.array(time)[ind]
        flux = np.array(flux)[ind]
    else:
        ind = np.where(~np.isnan(time*flux*flux_err))
        time = np.array(time)[ind]
        flux = np.array(flux)[ind]
        flux_err = np.array(flux_err)[ind]
    print(time, flux)
    plt.figure()
    plt.plot(time, flux, 'b.')
    model = bls(time * u.day, flux, dy=flux_err)
    print(model)
    periodogram = model.autopower(0.05)
    plt.plot(periodogram.period, periodogram.power)  
    # max_power = np.argmax(periodogram.power)
    # stats = model.compute_stats(periodogram.period[max_power],
    #                             periodogram.duration[max_power],
    #                             periodogram.transit_time[max_power])
    # print(stats)
        
    
    
###############################################################################
#::: get TLS kwargs from TICv8
###############################################################################
def get_tls_kwargs_by_tic(tic_id, sigma=3, tls_kwargs=None, quiet=True):
    #mass comes first, radius comes second in the TLS source code for catalog_info()
    u, M_star, M_star_lerr, M_star_uerr, R_star, R_star_lerr, R_star_uerr = catalog_info(TIC_ID=int(tic_id))
    if not quiet:
        print('TICv8 info:')
        print('Quadratic limb darkening u_0, u_1', u[0], u[1])
        print('Stellar radius', R_star, '+', R_star_lerr, '-', R_star_uerr)
        print('Stellar mass', M_star, '+', M_star_lerr, '-', M_star_uerr)
        
    if tls_kwargs is None: tls_kwargs = {}
    tls_kwargs['R_star']=float(R_star)
    tls_kwargs['R_star_min']=R_star-sigma*R_star_lerr
    tls_kwargs['R_star_max']=R_star+sigma*R_star_uerr
    tls_kwargs['M_star']=float(M_star)
    tls_kwargs['M_star_min']=M_star-sigma*M_star_lerr
    tls_kwargs['M_star_max']=M_star+sigma*M_star_uerr
    tls_kwargs['u']=u
    
    if np.isnan(tls_kwargs['R_star']): 
        tls_kwargs['R_star'] = 1.
        warnings.warn("tls_kwargs: R_star was undefined in TICv8. Filling it with R_star=1.")
    if np.isnan(tls_kwargs['R_star_min']): 
        tls_kwargs['R_star_min'] = 0.13
        warnings.warn("tls_kwargs: R_star_min was undefined in TICv8. Filling it with R_star_min=0.13")
    if np.isnan(tls_kwargs['R_star_max']): 
        tls_kwargs['R_star_max'] = 3.5
        warnings.warn("tls_kwargs: R_star_max was undefined in TICv8. Filling it with R_star_max=3.5")
    if np.isnan(tls_kwargs['M_star']): 
        tls_kwargs['M_star'] = 1.
        warnings.warn("tls_kwargs: M_star was undefined in TICv8. Filling it with M_star=1.")
    if np.isnan(tls_kwargs['M_star_min']): 
        tls_kwargs['M_star_min'] = 0.1
        warnings.warn("tls_kwargs: M_star_min was undefined in TICv8. Filling it with M_star_min=0.1")
    if np.isnan(tls_kwargs['M_star_max']): 
        tls_kwargs['M_star_max'] = 1.
        warnings.warn("tls_kwargs: M_star_max was undefined in TICv8. Filling it with M_star_max=0.1")
    if np.isnan(tls_kwargs['u']).any(): 
        tls_kwargs['u'] = [0.4804, 0.1867]
        warnings.warn("tls_kwargs: u was undefined in TICv8. Filling it with u=[0.4804, 0.1867]")

    return tls_kwargs



###############################################################################
#::: write TLS reuslts as a dictionary to a json file
###############################################################################
def write_tls_results(fname, results):
    '''
    Parameters
    ----------
    fname : str
        Name of the output json file.
    results : transitleastsuqares.results class
        The results returned form a TLS run.

    Returns
    -------
    None.

    Outputs
    -------
    A json file that contains a dictionary of the most important tls results.
    The json file can be read into Python again via allesfitter's read_dic.
    
    Explanation
    -----------
    The TLS results object contains the following keys, where 
    'short' indicates it's a float or short list (e.g., the found period or depth per transit) and 
    'long' indicates it's a humongous array (e.g., the whole light curve).
    We only want to save the 'short' parts to save space:
        SDE short
        SDE_raw short
        chi2_min short
        chi2red_min short
        period short
        period_uncertainty short
        T0 short
        duration short
        depth short
        depth_mean short
        depth_mean_even short
        depth_mean_odd short
        transit_depths short
        transit_depths_uncertainties short
        rp_rs short
        snr short
        snr_per_transit short
        snr_pink_per_transit short
        odd_even_mismatch short
        transit_times short
        per_transit_count short
        transit_count short
        distinct_transit_count short
        empty_transit_count short
        FAP short
        in_transit_count short
        after_transit_count short
        before_transit_count short
        periods long
        power long
        power_raw long
        SR long
        chi2 long
        chi2red long
        model_lightcurve_time long
        model_lightcurve_model long
        model_folded_phase long
        folded_y long
        folded_dy long
        folded_phase long
        model_folded_model long
    Also:
        correct_duration short
        model long (our self-made model(time) curve)
    '''
    dic = {}
    for key in ['SDE', 'SDE_raw', 'chi2_min', 'chi2red_min', 'period', 'period_uncertainty',\
                'T0', 'duration', 'depth', 'depth_mean', 'depth_mean_even', 'depth_mean_odd',\
                'transit_depths', 'transit_depths_uncertainties', 'rp_rs',\
                'snr', 'snr_per_transit', 'snr_pink_per_transit', 'odd_even_mismatch',\
                'transit_times', 'per_transit_count', 'transit_count', 'distinct_transit_count',\
                'empty_transit_count', 'FAP', 'in_transit_count', 'after_transit_count',\
                'before_transit_count'] + ['correct_duration']: 
        if (type(results[key])!=np.ndarray): #if it's not an array, save it as is
            dic[key] = results[key]
        else:  #if it's a short array, save as list (for json)
            dic[key] = results[key].tolist()
    write_json(fname, dic)
    
    
    
###############################################################################
#::: function to convert the results into a dictionary
###############################################################################
def _to_dic(results):
    dic = {}
    for key in results: 
        dic[key] = results[key]
    return dic



###############################################################################
#::: TLS search on an input lightcurve
###############################################################################
def tls_search(time, flux, flux_err, plot=True, plot_type='brokenplot', **kwargs):
    '''
    Summary:
    -------
    This runs TLS on these data with the given infos
    
    Inputs:
    -------
    time : array of flaot
        time stamps of observations
    flux : array of flaot
        normalized flux
    flux_err : array of flaot
        error of normalized flux
    **kwargs : collection of keyword arguments
        All keyword arguments will be passed to TLS.
        Missing keywords will be replaced with default values:
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
            u : list
                quadratic limb darkening parameters
                default [0.4804, 0.1867]
            period_min : float
                the minimum period to be searched (in days)
            period_max : float
                the maximum period to be searched (in days)
            show_progress_bar : bool
                Show a progress bar for TLS
                default True
            SNR_threshold : float
                the SNR threshold at which to stop the TLS search
                default 5
            SDE_threshold : float
                the SDE threshold at which to stop the TLS search
                default -inf
            FAP_threshold : float
                the False Alarm Probability threshold at which to stop the TLS search
                default inf
            quiet : bool
                silence all TLS outprint
                default True
        
    Returns:
    -------
    results_all : list of dictionaries
        List of all dictionaries containing the TLS results 
        (with dictionaries made from the transitleastsqaures.results class).
    fig_all : list of matplotlib.figure object, optional
        List of all summary figures. Only returned if plot is True.
    '''
    
    #::: seeed
    np.random.seed(42)
    
    
    #::: handle inputs
    time, flux, flux_err = clean(time, flux, flux_err)
    plot_bool = plot
        
    if 'show_progress_bar' not in kwargs: kwargs['show_progress_bar'] = True
    if 'SNR_threshold' not in kwargs: kwargs['SNR_threshold'] = 5.
    if 'SDE_threshold' not in kwargs: kwargs['SDE_threshold'] = -np.inf #don't trust SDE
    if 'FAP_threshold' not in kwargs: kwargs['FAP_threshold'] = np.inf #don't trust FAP 
    if 'quiet' not in kwargs: kwargs['quiet'] = True
    if 'inj_period' not in kwargs: kwargs['inj_period'] = np.nan
    
    non_tls_keys = ['SNR_threshold','SDE_threshold','FAP_threshold','quiet','inj_period']
    tls_kwargs_original = {key: kwargs[key] for key in kwargs.keys() if key not in non_tls_keys} #for the original tls
    #the rest is filled automatically by TLS if it was not given
    print('tls_kwargs_original', tls_kwargs_original)
    
    #::: init
    SNR = 1e12
    SDE = 1e12
    FAP = 0
    FOUND_SIGNAL = False
    results_all = []     
    fig_lightcurve_all = []     
    fig_folded_all = []        
    
    
    #::: function to run it once
    def _run1(time, flux, flux_err):
        if kwargs['quiet']:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = tls(time, flux, flux_err)
                        results = model.power(**tls_kwargs_original)
        else:
            model = tls(time, flux, flux_err)
            results = model.power(**tls_kwargs_original)
        
        results = _to_dic(results)
        results['detection'] = (results['snr'] >= kwargs['SNR_threshold']) and (results['SDE'] >= kwargs['SDE_threshold']) and (results['FAP'] <= kwargs['FAP_threshold'])
        results['correct_duration'] = np.nan        
        results['R_planet_'] = np.nan

        
        if results['detection']:
            #::: calculcate the correct_duration, as TLS sometimes returns unreasonable durations
            ind_tr_phase = np.where( results['model_folded_model'] < 1. )[0]
            results['correct_duration'] = results['period'] * (results['model_folded_phase'][ind_tr_phase[-1]] - results['model_folded_phase'][ind_tr_phase[0]])
            
            if 'R_star' in kwargs:
                results['R_planet'] = results['rp_rs'] * kwargs['R_star'] * 109.07637070600963 #from Rsun to Rearth
            
        return results
            
            
    #::: function to plot it once
    # def _plot1(time, flux, flux_err, results):
    #     fig, axes = plt.subplots(1, 3, figsize=(20,5), tight_layout=True)
        
    #     ax = axes[0]
    #     ax.plot(results['folded_phase'], results['folded_y'], 'k.', color='silver', rasterized=True)
    #     bintime, binflux, binflux_err, _ = rebin_err(results['folded_phase'], results['folded_y'], dt = 0.001*results['period'], ferr_type='medsig', ferr_style='sem')
    #     ax.plot(bintime, binflux, 'b.', rasterized=True)
    #     ax.plot(results['model_folded_phase'], results['model_folded_model'], 'r-', lw=3)
        
    #     ax = axes[1]
    #     ax.plot((results['folded_phase']-0.5)*results['period']*24, results['folded_y'], 'k.', color='silver', rasterized=True)
    #     bintime, binflux, binflux_err, _ = rebin_err((results['folded_phase']-0.5)*results['period']*24, results['folded_y'], dt = 0.001*results['period']*24, ferr_type='medsig', ferr_style='sem')
    #     ax.plot(bintime, binflux, 'bo', rasterized=True)
    #     ax.plot((results['model_folded_phase']-0.5)*results['period']*24, results['model_folded_model'], 'r-', lw=3)
    #     ax.set(xlim=[ -1.5*results['correct_duration']*24, +1.5*results['correct_duration']*24 ], xlabel='Time (h)', yticks=[])
        
    #     ax = axes[2]
    #     ax.text( .02, 0.95, 'P = ' + np.format_float_positional(results['period'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
    #     ax.text( .02, 0.85, 'Depth = ' + np.format_float_positional(1e3*(1.-results['depth']),4) + ' ppt', ha='left', va='center', transform=ax.transAxes )
    #     ax.text( .02, 0.75, 'Duration = ' + np.format_float_positional(24*results['correct_duration'],4) + ' h', ha='left', va='center', transform=ax.transAxes )
    #     ax.text( .02, 0.65, 'T_0 = ' + np.format_float_positional(results['T0'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
    #     ax.text( .02, 0.55, 'SNR = ' + np.format_float_positional(results['snr'],4), ha='left', va='center', transform=ax.transAxes )
    #     ax.text( .02, 0.45, 'SDE = ' + np.format_float_positional(results['SDE'],4), ha='left', va='center', transform=ax.transAxes )
    #     ax.text( .02, 0.35, 'FAP = ' + np.format_float_scientific(results['FAP'],4), ha='left', va='center', transform=ax.transAxes )
    #     ax.set_axis_off()
        
    #     return fig
            
            
    #::: search for transits in a loop
    while (SNR >= kwargs['SNR_threshold']) and (SDE >= kwargs['SDE_threshold']) and (FAP <= kwargs['FAP_threshold']) and (FOUND_SIGNAL==False):
       
        #::: run once 
        results = _run1(time, flux, flux_err)
        
        #::: if a transit was detected, store the results, plot, and apply a mask for the next run
        if results['detection']:
            results_all.append(results)
            
            results['model'] = np.interp(time, results['model_lightcurve_time'], results['model_lightcurve_model'])
            
            if plot_bool:
                # fig = _plot1(time, flux, flux_err, results)
                fig_lightcurve = _tls_search_plot_lightcurve(time, flux, results, typ=plot_type)
                fig_folded = _tls_search_plot_folded(time, flux, results)
                fig_lightcurve_all.append(fig_lightcurve)
                fig_folded_all.append(fig_folded)
           
            time, flux, flux_err = mask(time, flux, flux_err, 
                                        results['period'], 
                                        np.max((1.5*results['correct_duration'])), 
                                        results['T0'])

        #::: update values
        SNR = results['snr']
        SDE = results['SDE']
        FAP = results['FAP']
        if is_multiple_of(results['period'],kwargs['inj_period']): SNR = -np.inf #if run as part of an injection-recovery test, then abort if it matches the injected period
        
                
    #::: return
    if plot_bool:
        return results_all, fig_lightcurve_all, fig_folded_all
    else:
        return results_all



def _cut(time, model_lightcurve_time, model_lightcurve_flux):
    return np.interp(time, model_lightcurve_time, model_lightcurve_flux) 



def _tls_search_plot_lightcurve(time, flux, results, typ='fullplot'):
    """
    ...

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.
    flux_err : TYPE
        DESCRIPTION.
    results : TYPE
        DESCRIPTION.
    typ : TYPE, optional
        'fullplot', 'brokenplot', 'tessplot', 'monthplot'. The default is 'fullplot'.

    Returns
    -------
    None.
    """
    
    if typ=='fullplot':
        axes = fullplot(time, flux)
        axes = fullplot(results['model_lightcurve_time'], results['model_lightcurve_model'], color='r', ls='-', marker='', lw=3, zorder=100, axes=axes)
    elif typ=='brokenplot':
        axes = brokenplot(time, flux)
        axes = brokenplot(results['model_lightcurve_time'], results['model_lightcurve_model'], color='r', ls='-', marker='', lw=3, zorder=100, axes=axes)
    elif typ=='tessplot':
        trend = _cut(time, results['model_lightcurve_time'], results['model_lightcurve_model'])
        axes = tessplot(time, flux, trend=trend)
        # axes = tessplot(results['model_lightcurve_time'], results['model_lightcurve_model'], color='r', ls='-', marker='', lw=3, zorder=100, axes=axes, shade=False)
    elif typ=='monthplot':
        axes = monthplot(time, flux)
        axes = monthplot(results['model_lightcurve_time'], results['model_lightcurve_model'], color='r', ls='-', marker='', lw=3, zorder=100, axes=axes)
    
    return plt.gcf()
    

    
def _tls_search_plot_folded(time, flux, results):
    """
    ...

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.
    results : TYPE
        DESCRIPTION.
    axes : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    axes : TYPE
        DESCRIPTION.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(12,3), tight_layout=True)
    
    ax = axes[0]
    bintime, binflux, binflux_err, _ = rebin_err(results['folded_phase'], results['folded_y'], dt = 0.001*results['period'], ferr_type='medsig', ferr_style='sem')
    ax.plot(bintime, binflux, 'b.', rasterized=True)
    ax.plot(results['model_folded_phase'], results['model_folded_model'], 'r-', lw=3)
    ylim = ax.get_ylim()
    ax.plot(results['folded_phase'], results['folded_y'], 'k.', color='silver', rasterized=True, zorder=-1)
    ax.set_ylim(ylim)
    
    ax = axes[1]
    bintime, binflux, binflux_err, _ = rebin_err((results['folded_phase']-0.5)*results['period']*24, results['folded_y'], dt = 0.001*results['period']*24, ferr_type='medsig', ferr_style='sem')
    ax.plot(bintime, binflux, 'bo', rasterized=True)
    ax.plot((results['model_folded_phase']-0.5)*results['period']*24, results['model_folded_model'], 'r-', lw=3)
    ax.set(xlim=[ -1.5*results['correct_duration']*24, +1.5*results['correct_duration']*24 ], xlabel='Time (h)', yticks=[])
    ylim = ax.get_ylim()
    ax.plot((results['folded_phase']-0.5)*results['period']*24, results['folded_y'], 'k.', color='silver', rasterized=True, zorder=-1)
    ax.set_ylim(ylim)
    
    ax = axes[2]
    ax.text( .02, 0.95, 'P = ' + np.format_float_positional(results['period'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.85, 'Depth = ' + np.format_float_positional(1e3*(1.-results['depth']),4) + ' ppt', ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.75, 'Duration = ' + np.format_float_positional(24*results['correct_duration'],4) + ' h', ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.65, 'T_0 = ' + np.format_float_positional(results['T0'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.55, 'SNR = ' + np.format_float_positional(results['snr'],4), ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.45, 'SDE = ' + np.format_float_positional(results['SDE'],4), ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.35, 'FAP = ' + np.format_float_scientific(results['FAP'],4), ha='left', va='center', transform=ax.transAxes )
    ax.text( .02, 0.25, 'R_planet/R_star = ' + np.format_float_positional(results['rp_rs'],4), ha='left', va='center', transform=ax.transAxes )
    if ~np.isnan(results['R_planet']): 
        ax.text( .02, 0.15, 'R_planet = ' + np.format_float_positional(results['R_planet'],4), ha='left', va='center', transform=ax.transAxes )
    ax.set_axis_off()
    
    return fig
    


def _tls_search_plot_individual(time, flux, flux_err, results):
    pass #TODO

    

###############################################################################
#::: Convenient wrapper for TESS tasks
###############################################################################
#TODO: work in progress
def tls_search_tess(time, flux, flux_err, 
                     wotan_kwargs=None,
                     tls_kwargs=None,
                     bad_regions=None,
                     options=None):

    if options is None: options = {}
    if 'outdir' not in options: options['outdir'] = ''
    if wotan_kwargs is None: wotan_kwargs = {'flatten': {'method':'biweight', 'window_length':1}}
    
    #::: logprint
    with open( os.path.join(options['outdir'], 'logfile.log'), 'w' ) as f:
        f.write('TLS search, UTC ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    logprint('\nWotan kwargs:', options=options)
    logpprint(wotan_kwargs, options=options)
    logprint('\nTLS kwargs:', options=options)
    logpprint(tls_kwargs, options=options)
    logprint('\nOptions:', options=options)
    logpprint(options, options=options)
    
    
    #::: cleaning
    flux_clean, fig1, fig2, fig3 = \
        tessclean(time, flux, plot=True,
                  method=wotan_kwargs['flatten']['method'],
                  window_length=wotan_kwargs['flatten']['window_length'],
                  bad_regions=bad_regions)
    
    write_csv(os.path.join(options['outdir'],'flux_clean.csv'), (time, flux_clean, flux_err), header='time,flux_clean,flux_err')
    
    with PdfPages( os.path.join(options['outdir'],'flux_clean.pdf') ) as pdf:
        pdf.savefig( fig1 )
        pdf.savefig( fig2 )
        pdf.savefig( fig3 )
    plt.close('all')
    
    
    #::: transit search
    results_all, fig_lightcurve_all, fig_folded_all = \
        tls_search(time, flux_clean, flux_err, 
                   plot=True, plot_type='tessplot',
                   **tls_kwargs)
    
    if len(results_all)>0:
        with open( os.path.join(options['outdir'],'tls_summary.txt'), 'w' ) as f:
            f.write('TLS found '+str(len(results_all))+' potential signal(s).')
            
        for i, results in enumerate(results_all):
            write_tls_results( os.path.join(options['outdir'],'tls_signal_'+str(i)+'.txt'), results )
            
        for i, (fig1, fig2) in enumerate(zip(fig_lightcurve_all, fig_folded_all)):
            with PdfPages( os.path.join(options['outdir'],'tls_signal_'+str(i)+'.pdf') ) as pdf:
                pdf.savefig( fig1 )
                pdf.savefig( fig2 )
        plt.close('all')
        
    else:
        with open( os.path.join(options['outdir'],'tls_summary.txt'), 'w' ) as f:
            f.write('TLS found no potential signal(s).')
        
            

###############################################################################
#::: TLS search on an input lightcurve
###############################################################################
# def tls_search_old(time, flux, flux_err,
#                known_transits=None,
#                tls_kwargs=None,
#                wotan_kwargs=None,
#                options=None):
#     '''
#     Summary:
#     -------
#     This runs TLS on these data with the given infos
    
#     Inputs:
#     -------
#     time : array of flaot
#         time stamps of observations
#     flux : array of flaot
#         normalized flux
#     flux_err : array of flaot
#         error of normalized flux
        
#     Optional Inputs:
#     ----------------
#     known_transits : None or dict
#         >> can be used to mask known transits before running TLS
#         if None
#             nothing happens
#         if dict 
#             if one transit is already known, give for example: 
#                 known_transits = {'period':[1.3], 'duration':[2.1], 'epoch':[245800.0]}
#             if multiple transits are already known, give for example: 
#                 known_transits = {'name':['b','c'], 'period':[1.3, 21.0], 'duration':[2.1, 4.1], 'epoch':[245800.0, 245801.0]}
#             'period' is the period of the known transit(s)
#             'duration' is the total duration of the known transit(s), i.e. from first ingress point to last egrees point, in days
#             'epoch' is the epoch of the known transit(s)
        
#     tls_kwargs : None, str or dict:
#         >> can be used to fine-tune the TLS algorithm
#         if None
#             the default parameters will be chosen (see below)
#         if 'default'
#             the default parameters will be chosen (see below)
#         if dict
#             a dictionary with the following keywords is expected; 
#             missing keywords will be replaced with default values
#             R_star : float
#                 radius of the star (e.g. median)
#                 default 1 R_sun (from TLS)
#             R_star_min : float
#                 minimum radius of the star (e.g. 1st percentile)
#                 default 0.13 R_sun (from TLS)
#             R_star_max : float
#                 maximum radius of the star (e.g. 99th percentile)
#                 default 3.5 R_sun (from TLS)
#             M_star : float
#                 mass of the star (e.g. median)
#                 default 1. M_sun (from TLS)
#             M_star_min : float
#                 minimum mass of the star (e.g. 1st percentile)
#                 default 0.1 M_sun (from TLS)
#             M_star_max : float
#                 maximum mass of the star (e.g. 99th percentile)
#                 default 1. M_sun (from TLS)    
#             u : list
#                 quadratic limb darkening parameters
#                 default [0.4804, 0.1867]
#             SNR_threshold : float
#                 the SNR threshold at which to stop the TLS search
#                 default 5
#             SDE_threshold : float
#                 the SDE threshold at which to stop the TLS search
#                 default -inf
#             FAP_threshold : float
#                 the False Alarm Probability threshold at which to stop the TLS search
#                 default inf
#             period_min : float
#                 the minimum period to be searched (in days)
#             period_max : float
#                 the maximum period to be searched (in days)
        
#     wotan_kwargs : None, str, or dict:
#         >> can be used to detrend the data before the TLS search
#         if None
#             the default detrending will run (see below)
#         if str is 'default'
#             the default detrending will run (see below)
#         if str is 'off'
#             no detrending will run
#         if dict
#             a dictionary with two sub-dictionaries is expected; 
#             missing keywords will be replaced with default values
#             wotan_kwargs['slide_clip'] : dict
#                 this dictionary contains all slide clipping arguments
#                 window_length : float
#                     slide clip window length
#                     default 1
#                 low : float
#                     slide clip lower sigma
#                     default 20
#                 high : float
#                     slide clip upper sigma
#                     default 3
#             wotan_kwargs['flatten'] : dict
#                 this dictionary contains contains all detrending arguments
#                 method : str
#                     detrending method
#                     default 'biweight'
#                 window_length : float
#                     detrending window length in days
#                     default 1         
            
#     options : None or dict, keywords:
#         >> can be used for any general options
#         if None
#             the default options will be used (see below)
#         if dict
#             a dictionary with the following keywords is expected;
#             missing keywords will be replaced with default values
#             show_plot : bool
#                 can show a plot of each phase-folded transit candidate and TLS model in the terminal 
#                 default is True
#             save_plot : bool or str
#                 can save a plot of each phase-folded transit candidate and TLS model into outdir
#                 if True, will be set to '123'
#                 if str, then: '1': detrended plot, '2': TLS plot, '3': all TLS plots, and any combinations thereof
#                 default is True
#             save_csv : bool
#                 can save a csv of the detrended lightcurve
#                 default is True
#             outdir : string
#                 if None
#                     a new directory called "results" will be created in the current folder
#                 default is "tls_results_[wotan_flatten_method]_[wotan_flatten_window_length]"
        
#     Returns:
#     -------
#     List of all TLS results
#     '''
    
#     #::: seeed
#     np.random.seed(42)
    
    
#     #::: handle inputs
#     def clean(time,flux,flux_err):
#         if flux_err is None:
#             ind = np.where( ~np.isnan(time*flux) )[0]
#             time = time[ind]
#             flux = flux[ind]
#         else:
#             ind = np.where( ~np.isnan(time*flux*flux_err) )[0]
#             time = time[ind]
#             flux = flux[ind]
#             flux_err = flux_err[ind]
#         return time, flux, flux_err
    
#     time, flux, flux_err = clean(time,flux,flux_err)
#     time_input = 1.*time
#     flux_input = 1.*flux #for plotting
        
    
#     if type(wotan_kwargs)==str and wotan_kwargs=='off': 
#         detrend = False
#     else:
#         detrend = True
#         if (wotan_kwargs is None) or (type(wotan_kwargs)==str and wotan_kwargs=='default'): wotan_kwargs={} 
#         if 'slide_clip' not in wotan_kwargs: wotan_kwargs['slide_clip'] = {}
#         if wotan_kwargs['slide_clip'] is not None:
#             if 'window_length' not in wotan_kwargs['slide_clip']: wotan_kwargs['slide_clip']['window_length'] = 1.
#             if 'low' not in wotan_kwargs['slide_clip']: wotan_kwargs['slide_clip']['low'] = 20.
#             if 'high' not in wotan_kwargs['slide_clip']: wotan_kwargs['slide_clip']['high'] = 3.
    
#     if 'flatten' not in wotan_kwargs: wotan_kwargs['flatten'] = {}
#     if wotan_kwargs['flatten'] is not None:
#         if 'method' not in wotan_kwargs['flatten']: wotan_kwargs['flatten']['method'] = 'biweight'
#         if 'window_length' not in wotan_kwargs['flatten']: wotan_kwargs['flatten']['window_length'] = 1.
#     #the rest is filled automatically by Wotan
        
#     if tls_kwargs is None: tls_kwargs = {}
#     if 'show_progress_bar' not in tls_kwargs: tls_kwargs['show_progress_bar'] = False
#     if 'SNR_threshold' not in tls_kwargs: tls_kwargs['SNR_threshold'] = 5.
#     if 'SDE_threshold' not in tls_kwargs: tls_kwargs['SDE_threshold'] = -np.inf #don't trust SDE
#     if 'FAP_threshold' not in tls_kwargs: tls_kwargs['FAP_threshold'] = np.inf #don't trust FAP 
#     tls_kwargs_original = {key: tls_kwargs[key] for key in tls_kwargs.keys() if key not in ['SNR_threshold','SDE_threshold','FAP_threshold']} #for the original tls
#     #the rest is filled automatically by TLS
    
#     if options is None: options = {}
#     if 'show_plot' not in options: options['show_plot'] = True
#     if type(options['show_plot'])==bool and (options['show_plot'] is True): options['show_plot']='123' #1: detrended plot, 2: TLS plot, 3: all TLS plots
#     if type(options['show_plot'])==bool and (options['show_plot'] is False): options['show_plot']='' #1: detrended plot, 2: TLS plot, 3: all TLS plots
#     if 'save_plot' not in options: options['save_plot'] = True
#     if type(options['save_plot'])==bool and (options['save_plot'] is True): options['save_plot']='123' #1: detrended plot, 2: TLS plot, 3: all TLS plots
#     if type(options['save_plot'])==bool and (options['save_plot'] is False): options['save_plot']='' #1: detrended plot, 2: TLS plot, 3: all TLS plots
#     if 'save_csv' not in options: options['save_csv'] = True
#     if 'outdir' not in options: 
#         if detrend:
#             options['outdir'] = 'tls_results_'+wotan_kwargs['flatten']['method']+'_'+str(wotan_kwargs['flatten']['window_length'])
#         else:
#             options['outdir'] = 'tls_results_undetrended'
#     if 'quiet' not in options: options['quiet'] = True
#     if 'inj_period' not in options: options['inj_period'] = np.nan
    
    
#     #::: init
#     SNR = 1e12
#     SDE = 1e12
#     FAP = 0
#     FOUND_SIGNAL = False
#     results_all = []      
#     if len(options['outdir'])>0 and not os.path.exists(options['outdir']): os.makedirs(options['outdir'])  
    
    
#     #::: logprint
#     with open( os.path.join(options['outdir'], 'logfile.log'), 'w' ) as f:
#         f.write('TLS search, UTC ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + '\n')
#     logprint('\nWotan kwargs:', options=options)
#     logpprint(wotan_kwargs, options=options)
#     logprint('\nTLS kwargs:', options=options)
#     logpprint(tls_kwargs, options=options)
#     logprint('\nOptions:', options=options)
#     logpprint(options, options=options)
    
#     # timer1 = timer.time()
#     # print('t1', timer1 - timer0)
            
#     #::: apply a mask (if wished so)
#     if known_transits is not None:
#         for period, duration, T0 in zip(known_transits['period'], known_transits['duration'], known_transits['epoch']):
#             time, flux, flux_err = mask(time, flux, flux_err, period, duration, T0)
    
    
#     #::: global sigma clipping
#     flux = sigma_clip(flux, sigma_upper=3, sigma_lower=20)
    
#     # timer2 = timer.time()
#     # print('t2', timer2 - timer0)
    
#     #::: detrend (if wished so)
#     if detrend:
        
#         #::: slide clipping (super slow)
#         # if wotan_kwargs['slide_clip'] is not None: flux = slide_clip(time, flux, **wotan_kwargs['slide_clip']) #slide_clip is super slow (10 seconds for a TESS 2 min lightcurve for a single Sector)
#         # timer3a = timer.time()
#         # print('t3a', timer3a - timer0)   
    
#         #::: fast slide clipping (super fast)
#         if wotan_kwargs['slide_clip'] is not None: 
#             flux = slide_clip(time, flux, **wotan_kwargs['slide_clip']) #slide_clip is super fast (<1 seconds for a TESS 2 min lightcurve for a single Sector)
#             flux_clip = 1*flux
#         # timer3a = timer.time()
#         # print('t3a', timer3a - timer0)   
        
#         #::: detrending (super fast)
#         if wotan_kwargs['flatten'] is not None: 
#             flux, trend = flatten(time, flux, return_trend=True, **wotan_kwargs['flatten']) #flatten is super fast, (<1 second for a TESS 2 min lightcurve for a single Sector)
#         # timer3b = timer.time()
#         # print('t3b', timer3b - timer0)   
        
#         #::: global sigma clipping on the flattened flux (super fast)
#         flux = sigma_clip(flux, sigma_upper=3, sigma_lower=20)
#         # timer3c = timer.time()
#         # print('t3c', timer3c - timer0)   
        
#         if ('1' in options['show_plot']) or ('1' in options['save_plot']):
#             gone = np.isnan(time_input*flux_input)
#             print(time_input, gone)
#             axes = tessplot(time_input[gone], flux_input[gone], color='r')
#             tessplot(time, flux_clip, trend=trend, axes=axes, shade=False)
#             for ax in axes: ax.set_ylabel('Flux\n(original)')
#             fig1 = plt.gcf()
            
#             axes = tessplot(time, flux_clip, trend=trend)
#             for ax in axes: ax.set_ylabel('Flux\n(clipped)')
#             fig2 = plt.gcf()
            
#             axes = tessplot(time, flux)
#             fig3 = plt.gcf()
#             for ax in axes: ax.set_ylabel('Flux\n(clipped & detrended)')
            
#             # fig, axes = plt.subplots(2,1, figsize=(40,8))
#             # brokenplot(time_input, flux_input, trend=trend, ax=axes[0])
#             # axes[0].set(ylabel='Flux (input)', xticklabels=[])
#             # brokenplot(time, trend, fmt='r-', ax=axes[0])
#             # axes[0].plot(time_input, flux_input, 'b.', rasterized=True)
#             # axes[0].plot(time, trend, 'r-', lw=2)
#             # brokenplot(time_input, flux_input, ax=axes[1], clip=True)
#             # brokenplot(time, trend, fmt='r-', ax=axes[1], clip=True)
#             # axes[1].set(ylabel='Flux (clipped)', xticklabels=[])
#             # brokenplot(time, flux, ax=axes[1])
#             # axes[1].plot(time, flux, 'b.', rasterized=True)
#             # axes[1].set(ylabel='Flux (detrended)', xlabel='Time (BJD)')
#             # axes[2].set(ylabel='Flux (detrended)')
#         if ('1' in options['save_plot']):
#             # try: 
#             f = os.path.join(options['outdir'],'flux_'+wotan_kwargs['flatten']['method']+'.pdf')
#             with PdfPages(f) as pdf:
#                 pdf.savefig( fig1 )
#                 pdf.savefig( fig2 )
#                 pdf.savefig( fig3 )
#             #     fig.savefig(os.path.join(options['outdir'],'flux_'+wotan_kwargs['flatten']['method']+'.pdf'), bbox_inches='tight') #some matplotlib versions crash when saving pdf...
#             # except: 
#             #     fig.savefig(os.path.join(options['outdir'],'flux_'+wotan_kwargs['flatten']['method']+'.jpg'), bbox_inches='tight') #some matplotlib versions need pillow for jpg (conda install pillow)...
        
#         if ('1' in options['show_plot']):
#             plt.show()
#         else:
#             plt.close('all')
                
#         if options['save_csv']:
#             if flux_err is None: flux_err0 = np.nan*time
#             else: flux_err0 = flux_err
#             X = np.column_stack((time, flux, flux_err0, trend))
#             np.savetxt(os.path.join(options['outdir'],'flux_'+wotan_kwargs['flatten']['method']+'.csv'), X, delimiter=',', header='time,flux_detrended,flux_err,trend')
        
#         time_detrended = 1.*time #just for plotting
#         flux_detrended = 1.*flux #just for plotting
        
#     # timer3d = timer.time()
#     # print('t3d', timer3d - timer0)    
    
    
#     #::: search for transits
#     i = 0
#     ind_trs = []
#     while (SNR >= tls_kwargs['SNR_threshold']) and (SDE >= tls_kwargs['SDE_threshold']) and (FAP <= tls_kwargs['FAP_threshold']) and (FOUND_SIGNAL==False):
        
#         if options['quiet']:
#             with open(os.devnull, 'w') as devnull:
#                 with contextlib.redirect_stdout(devnull):
#                     with warnings.catch_warnings():
#                         warnings.simplefilter("ignore")
#                         model = tls(time, flux, flux_err)
#                         results = model.power(**tls_kwargs_original)
#         else:
#             model = tls(time, flux, flux_err)
#             results = model.power(**tls_kwargs_original)
            
#         # timer4 = timer.time()
#         # print('t4', timer4 - timer0)  
        
#         # plt.figure()
#         # plt.plot(time, flux, 'b.')
#         # pprint(tls_kwargs_original)
#         # pprint(results)
#         # err
        
#         if (results['snr'] >= tls_kwargs['SNR_threshold']) and (results['SDE'] >= tls_kwargs['SDE_threshold']) and (results['FAP'] <= tls_kwargs['FAP_threshold']):
            
#             #::: calculcate the correct_duration, as TLS sometimes returns unreasonable durations
#             ind_tr_phase = np.where( results['model_folded_model'] < 1. )[0]
#             correct_duration = results['period'] * (results['model_folded_phase'][ind_tr_phase[-1]] - results['model_folded_phase'][ind_tr_phase[0]])
            
#             #::: mark transit
#             ind_tr, ind_out = index_transits(time_input, results['T0'], results['period'], correct_duration)
#             ind_trs.append(ind_tr)
            
#             #::: mask out detected transits and append results
#             time1, flux1 = 1*time, 1*flux #for plotting
#             time, flux, flux_err = mask(time, flux, flux_err, results['period'], np.max((1.5*correct_duration)), results['T0'])
#             results = _to_dic(results)
#             results['correct_duration'] = correct_duration
#             results_all.append(results)
            
#             #::: write TLS stats to file
#             write_tls_results(os.path.join(options['outdir'],'tls_signal_'+str(i)+'.txt'), results)
#             # with open(os.path.join(options['outdir'],'tls_signal_'+str(i)+'.txt'), 'wt') as out:
#             #     pprint(results, stream=out)
    
#             # timer5 = timer.time()
#             # print('t5', timer5 - timer0)  
    
#             #::: individual TLS plots
#             if ('2' in options['show_plot']) or ('2' in options['save_plot']):
#                 fig = plt.figure(figsize=(20,8), tight_layout=True)
#                 gs = fig.add_gridspec(2,3)
                
#                 ax = fig.add_subplot(gs[0,:])
#                 ax.plot(time1, flux1, 'k.', color='silver', rasterized=True)
#                 bintime, binflux, binflux_err, _ = rebin_err(time1, flux1, dt = 10./60/24, ferr_type='medsig', ferr_style='sem') #in 10 min intervals
#                 ax.plot(bintime, binflux, 'b.', rasterized=True)
#                 ax.plot(results['model_lightcurve_time'], results['model_lightcurve_model'], 'r-', lw=3)
#                 ax.set(xlabel='Time (BJD)', ylabel='Flux')
                
#                 ax = fig.add_subplot(gs[1,0])
#                 ax.plot(results['folded_phase'], results['folded_y'], 'k.', color='silver', rasterized=True)
#                 bintime, binflux, binflux_err, _ = rebin_err(results['folded_phase'], results['folded_y'], dt = 0.001*results['period'], ferr_type='medsig', ferr_style='sem')
#                 ax.plot(bintime, binflux, 'b.', rasterized=True)
#                 # plot_phase_folded_lightcurve(time1, flux1, results['period'], results['T0'], dt=0.002, ax=ax)
#                 ax.plot(results['model_folded_phase'], results['model_folded_model'], 'r-', lw=3)
#                 # ax.set(xlabel='Phase', ylabel='Flux')
                
#                 ax = fig.add_subplot(gs[1,1])
#                 ax.plot((results['folded_phase']-0.5)*results['period']*24, results['folded_y'], 'k.', color='silver', rasterized=True)
#                 bintime, binflux, binflux_err, _ = rebin_err((results['folded_phase']-0.5)*results['period']*24, results['folded_y'], dt = 0.001*results['period']*24, ferr_type='medsig', ferr_style='sem')
#                 ax.plot(bintime, binflux, 'bo', rasterized=True)
#                 # plot_phase_folded_lightcurve(time1*24, flux1, results['period']*24, results['T0'], ax=ax, dt=0.002)
#                 ax.plot((results['model_folded_phase']-0.5)*results['period']*24, results['model_folded_model'], 'r-', lw=3)
#                 ax.set(xlim=[ -1.5*correct_duration*24, +1.5*correct_duration*24 ], xlabel='Time (h)', yticks=[])
                
#                 ax = fig.add_subplot(gs[1,2])
#                 ax.text( .02, 0.95, 'P = ' + np.format_float_positional(results['period'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
#                 ax.text( .02, 0.85, 'Depth = ' + np.format_float_positional(1e3*(1.-results['depth']),4) + ' ppt', ha='left', va='center', transform=ax.transAxes )
#                 ax.text( .02, 0.75, 'Duration = ' + np.format_float_positional(24*correct_duration,4) + ' h', ha='left', va='center', transform=ax.transAxes )
#                 ax.text( .02, 0.65, 'T_0 = ' + np.format_float_positional(results['T0'],4) + ' d', ha='left', va='center', transform=ax.transAxes )
#                 ax.text( .02, 0.55, 'SNR = ' + np.format_float_positional(results['snr'],4), ha='left', va='center', transform=ax.transAxes )
#                 ax.text( .02, 0.45, 'SDE = ' + np.format_float_positional(results['SDE'],4), ha='left', va='center', transform=ax.transAxes )
#                 ax.text( .02, 0.35, 'FAP = ' + np.format_float_scientific(results['FAP'],4), ha='left', va='center', transform=ax.transAxes )
#                 ax.set_axis_off()
#                 if ('2' in options['save_plot']):
#                     try: fig.savefig(os.path.join(options['outdir'],'tls_signal_'+str(i)+'.pdf'), bbox_inches='tight') #some matplotlib versions crash when saving pdf...
#                     except: fig.savefig(os.path.join(options['outdir'],'tls_signal_'+str(i)+'.jpg'), bbox_inches='tight') #some matplotlib versions need pillow for jpg (conda install pillow)...
#                 if ('2' in options['show_plot']):
#                     plt.show(fig)
#                 else:
#                     plt.close(fig)
                    
#             # timer6 = timer.time()
#             # print('t6', timer6 - timer0)  
            
#         SNR = results['snr']
#         SDE = results['SDE']
#         FAP = results['FAP']
#         if is_multiple_of(results['period'],options['inj_period']): SNR = -np.inf #if run as part of an inejction-recovery test, then abort if it matches the injected period
#         i+=1
        
        
        
#     #::: full lightcurve plot
#     if ('3' in options['show_plot']) or ('3' in options['save_plot']):
        
#         if detrend:
#             fig, axes = plt.subplots(2,1, figsize=(40,8), tight_layout=True)
#             ax = axes[0]
#             ax.plot(time_input, flux_input, 'k.', color='grey', rasterized=True)
#             ax.plot(time_input, trend, 'r-', lw=2)
#             for number, ind_tr in enumerate(ind_trs):
#                 ax.plot(time_input[ind_tr], flux_input[ind_tr], marker='.', linestyle='none', label='signal '+str(number))
#             ax.set(ylabel='Flux (input)', xticklabels=[])
#             ax.legend()

#             ax = axes[1]
#             ax.plot(time_detrended, flux_detrended, 'k.', color='grey', rasterized=True)
#             for number, ind_tr in enumerate(ind_trs):
#                 ax.plot(time_detrended[ind_tr], flux_detrended[ind_tr], marker='.', linestyle='none', label='signal '+str(number))
#             ax.set(ylabel='Flux (detrended)', xlabel='Time (BJD)')
#             ax.legend()
            
#         else:
#             fig = plt.figure(figsize=(20,4), tight_layout=True)
#             fig, ax = plt.subplots(1,1, figsize=(40,4))
#             ax.plot(time_input, flux_input, 'k.', color='grey', rasterized=True)
#             ax.set(ylabel='Flux (input)', xlabel='Time (BJD)')
#             for number, ind_tr in enumerate(ind_trs):
#                 ax.plot(time_input[ind_tr], flux_input[ind_tr], marker='.', linestyle='none', label='signal '+str(number))
#             ax.legend()
        
#         if ('3' in options['save_plot']):
#             try: fig.savefig(os.path.join(options['outdir'],'tls_signal_all.pdf'), bbox_inches='tight') #some matplotlib versions crash when saving pdf...
#             except: fig.savefig(os.path.join(options['outdir'],'tls_signal_all.jpg'), bbox_inches='tight') #some matplotlib versions need pillow for jpg (conda install pillow)...
#         if ('3' in options['show_plot']):
#             plt.show(fig)
#         else:
#             plt.close(fig)                    
                
            
#     return results_all



###############################################################################
#::: TLS search using tessio
###############################################################################
def tls_search_by_tic(tic_id,
                      tls_kwargs=None, SNR_threshold=5., known_transits=None,
                      options=None):
    '''
    Summary:
    -------
    wrapper around tls_search()
    retrieves the SPOC PDC-SAP lightcurve
    retrieves all TIC catalog information from MAST
    calls tls_search()
    
    Inputs:
    -------
    tic_id : str
        TIC ID
        
    Optional Inputs:
    ----------------
    see tls_search()
        
    Returns:
    -------
    list of all TLS results
    '''
    
    #::: handle inputs
    if options is None: options = {}
    if 'show_plot' not in options: options['show_plot']=False
    if 'save_plot' not in options: options['save_plot']=False
    if 'outdir' not in options: options['outdir']=''
    
    #::: format inputs
    tic_id = str(int(tic_id))
    
    #::: load data and inject transit
    time, flux, flux_err = tessio.get(tic_id, pipeline='spoc', PDC=True, unpack=True)
    
    #::: load TIC info / tls kwargs
    tls_kwargs = get_tls_kwargs_by_tic(tic_id, tls_kwargs=tls_kwargs)
    
    return tls_search(time, flux, flux_err,
                      tls_kwargs=tls_kwargs,
                      SNR_threshold=SNR_threshold,
                      known_transits=known_transits,
                      options=options)



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