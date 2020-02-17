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
import ellc
from astropy.constants import G
from astropy import units as u
from tqdm import tqdm
import itertools
from transitleastsquares import catalog_info

#::: my modules
from allesfitter.exoworlds_rdx.lightcurves import lightcurve_tools as lct
from .transit_search import tls_search
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
    Set up a logfile for the injection-recovery tests;
    starts user dialog if a logfile already exists
    
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
                f.write('inj_period,inj_rplanet,tls_period,tls_depth,tls_duration,tls_SDE,tls_SNR\n')
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
                
        
    
#::: make an ellc model lightcurve
def make_model(time, flux, flux_err, epoch, period, rplanet, R_star=1., M_star=1., show_plot=False, save_plot=False, save_csv=False):
    '''
    Inputs:
    -------
    time : array of float
        in days
    flux : array of float
        relative flux of the 'underlying' lightcurve
    flux_err : array of float
        error of relative flux of the 'underlying' lightcurve
    epoch : float
        epoch in days
    period : float
        period in days
    R_planet : float
        radius of the planet, in R_earth
    R_star : float
        radius of the star, in R_sun
        default is 1.
    M_star: float
        mass of the star, in M_sun
        default is 1.
    show_plot : bool
        show the plot in the terminal, or close it
    save_plot : bool
        save the plot to a file, or not
    save_csv : bool
        save the lightcurve to a file, or not
        
    Returns:
    --------
    flux2 : array of float
        relative flux with injected signal
    '''
    
    a = (G/(4*np.pi**2) * (period*u.d)**2 * M_star*u.Msun)**(1./3.) #in AU, with astropy units
    a = a.to(u.AU) #in AU, with astropy units
    
    radius_1 = (R_star*u.Rsun / a).decompose().value
    radius_2 = (rplanet*u.Rearth / a).decompose().value
    a = a.value
    
    model_flux = ellc.lc(
           t_obs = time, 
           radius_1 = radius_1,
           radius_2 = radius_2,
           sbratio = 0, 
           incl = 90, 
           light_3 = 0,
           t_zero = epoch, 
           period = period,
           a = None,
           q = 1,
           f_c = None, f_s = None,
           ldc_1=[0.6,0.2], ldc_2 = None,
           gdc_1 = None, gdc_2 = None,
           didt = None,
           domdt = None,
           rotfac_1 = 1, rotfac_2 = 1,
           hf_1 = 1.5, hf_2 = 1.5,
           bfac_1 = None, bfac_2 = None,
           heat_1 = None, heat_2 = None, 
           lambda_1 = None, lambda_2 = None,
           vsini_1 = None, vsini_2 = None,
           t_exp=None, n_int=None, 
           grid_1='default', grid_2='default',
           ld_1='quad', ld_2=None,
           shape_1='sphere', shape_2='sphere',
           spots_1=None, spots_2=None, 
           exact_grav=False, verbose=1)
    
    flux2 = flux+model_flux-1
    
    if show_plot or save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(20,10), sharex='col', sharey='row', gridspec_kw={'width_ratios': [3,1]})
        
        axes[0,0].plot(time, flux2, 'b.', rasterized=True)
        axes[0,0].plot(time, model_flux, 'r-')
        
        phase, phaseflux, phaseflux_err, N, phi = lct.phase_fold(time, flux2, period, epoch, dt=0.02, ferr_type='medsig', ferr_style='std', sigmaclip=True)
        axes[0,1].plot(phi, flux2, 'k.', color='lightgrey', rasterized=True)
        axes[0,1].errorbar(phase, phaseflux, yerr=phaseflux_err, fmt='b.')

        axes[1,0].plot(time, flux+model_flux-1, 'b.', rasterized=True)
        axes[1,0].plot(time, model_flux, 'r-', lw=2)
        axes[1,0].set(ylim=[0.99,1.01])
        
        phase, phaseflux, phaseflux_err, N, phi = lct.phase_fold(time, flux2, period, epoch, dt=0.02, ferr_type='medsig', ferr_style='std', sigmaclip=True)
        axes[1,1].plot(phi, flux2, 'k.', color='lightgrey')
        axes[1,1].errorbar(phase, phaseflux, yerr=phaseflux_err, fmt='b.')
        axes[1,0].plot(time, model_flux, 'r-', lw=2)
        axes[1,1].set(ylim=[0.99,1.01])
        
        plt.suptitle('\nP = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth')
        plt.tight_layout()
        
        if show_plot:
            plt.show(fig)
            
        if save_plot:
            if not os.path.exists('plots'): os.makedirs('plots')
            fig.savefig('plots/'+'Lightcurve P = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth'+'.jpg', bbox_inches='tight')
            plt.close(fig)
    
    if save_csv:
        X = np.column_stack((time, flux2, flux_err))
        np.savetxt('csv/'+'Lightcurve P = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth'+'.csv', X, delimiter=',')

    return flux2



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
#::: Inject an ellc transit and TLS search on an input lightcurve
###############################################################################
def inject_and_tls_search(time, flux, flux_err, 
                          periods, rplanets, logfname, 
                          SNR_threshold=5.,
                          known_transits=None, 
                          R_star=1., R_star_min=0.13, R_star_max=3.5, 
                          M_star=1., M_star_min=0.1, M_star_max=1.,
                          show_plot=False, save_plot=False):
    '''
    Inputs:
    -------
    time : array of flaot
        time stamps of observations
    flux : array of flaot
        normalized flux
    flux_err : array of flaot
        error of normalized flux
    periods : float or array of float
        a period or list of periods for injections
    rplanets : float or array of float
        a planet radius or list of planet radii for injections
    logfname : str
        file path and name for the log file
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
    show_plot : bool
        show a plot in the terminal or not
    save_plot : bool
        save a plot or not
        
    Summary:
    -------
    - Injects a planet signal via ellc, for a given period and radius (at random epoch)
    - Runs TLS on these injected data and infos
    
    Returns:
    -------
    Nothing, but a list of all TLS results will get saved to a log file
    '''

    #::: set up a logfile
    ex = setup_logfile(logfname)
    
    #::: cycle through all periods and rplanets
    print('\n', flush=True)
    for period, rplanet in tqdm(itertools.product(periods, rplanets), total=len(periods)*len(rplanets)): #combining the two for loops
        
        if to_do_or_not_to_do_that_is_the_question(ex, period, rplanet):
            print('\tP = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth --> do')
            epoch = time[0] + np.random.random()*period
            flux2 = make_model(time, flux, flux_err, epoch, period, rplanet, 
                               R_star=R_star, M_star=M_star,
                               show_plot=show_plot, save_plot=save_plot)
            results_all = tls_search(time, flux2, flux_err,
                                     SNR_threshold=SNR_threshold,
                                     known_transits=known_transits,
                                     R_star=R_star, R_star_min=R_star_min, R_star_max=R_star_max, 
                                     M_star=M_star, M_star_min=M_star_min, M_star_max=M_star_max)
            if len(results_all)>0:
                for r in results_all:
                    with open(logfname,'a') as f:
                        f.write(format(period, '.5f')+','+
                                format(rplanet, '.5f')+','+
                                format(r.period, '.5f')+','+
                                format(r.depth, '.5f')+','+
                                format(r.duration, '.5f')+','+
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
                            'nan'+'\n')
                
        else:
            print('\tP = '+str(period)+' days, Rp = '+str(rplanet)+' Rearth --> skipped (already exists)')
    
    #::: finish
    irplot(logfname)
    print('Done.')
            



###############################################################################
#::: Inject an ellc transit and TLS search using tessio
###############################################################################
def inject_and_tls_search_by_tic(tic_id, 
                                 periods, rplanets, logfname, 
                                 SNR_threshold=5.,
                                 known_transits=None, 
                                 show_plot=False, save_plot=False):
    '''
    Inputs:
    -------
    tic_id : str or int
        TIC ID
    periods : float or array of float
        a period or list of periods for injections
    rplanets : float or array of float
        a planet radius or list of planet radii for injections
    logfname : str
        file path and name for the log file
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
    show_plot : bool
        show a plot in the terminal or not
    save_plot : bool
        save a plot
        
    Summary:
    -------
        - retrieves the SPOC PDC-SAP lightcurve
        - retrieves all TIC catalog information from MAST
        - injects a planet signal via ellc, for a given period and radius (at random epoch)
        - runs TLS on these injected data and infos
    
    Returns:
    -------
        - a list of all TLS results will get saved to a log file
    '''
    
    #::: format inputs
    tic_id = str(int(tic_id))
    periods = np.atleast_1d(periods)
    rplanets = np.atleast_1d(rplanets)

    #::: load data
    dic = tessio.get(tic_id)
    time, flux, flux_err = dic['time'], dic['flux'], dic['flux_err']
    
    #::: load TIC info
    ab, R_star, R_star_lerr, R_star_uerr, M_star, M_star_lerr, M_star_uerr = catalog_info(TIC_ID=int(tic_id))
    print('TICv8 info:')
    print('Quadratic limb darkening a, b', ab[0], ab[1])
    print('Stellar radius', R_star, '+', R_star_lerr, '-', R_star_uerr)
    print('Stellar mass', M_star, '+', M_star_lerr, '-', M_star_uerr)
    
    #::: run
    inject_and_tls_search(time, flux, flux_err, periods, rplanets, logfname, 
                          SNR_threshold=SNR_threshold,
                          known_transits=known_transits, 
                          R_star=R_star, R_star_min=R_star-R_star_lerr, R_star_max=R_star+R_star_uerr, 
                          M_star=M_star, M_star_min=M_star-M_star_lerr, M_star_max=M_star+M_star_uerr,
                          show_plot=show_plot, save_plot=save_plot)
            


###############################################################################
#::: diagnostic TLS plots
###############################################################################
def tls_plot(results):
    
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    
    ax = axes[0]
    ax.axvline(results.period, alpha=0.4, lw=3)
    ax.set_xlim(np.min(results.periods), np.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    ax.set_ylabel(r'SDE')
    ax.set_xlabel('Period (days)')
    ax.plot(results.periods, results.power, color='k', lw=0.5)
    ax.set_title('Period '+format(results.period, '.5f') + ', SNR '+format(results.snr, '.1f'))
    ax.set_xlim(0, max(results.periods))
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
    # periods = np.arange(2,160+eps,2) #in days
    # rplanets = np.arange(0.8,4+eps,0.1) #in R_earth
    # SNR_threshold = 3.
    # known_transits = {'epoch':[2458715.3547, 2458726.0526, 2458743.5534],
    #                   'period':[8.8806, 28.5810, 38.3497],
    #                   'duration':[3.09/24., 4.45/24., 5.52/24.]
    #                   }
    # logfname = 'TIC_269701147.csv'
    
    # inject_and_tls_search_by_tic(tic_id, periods, rplanets, logfname, 
    #                               SNR_threshold=SNR_threshold,
    #                               known_transits=known_transits)
    
    

    