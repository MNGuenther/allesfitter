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
from astropy import units as u

#::: my modules
from .transit_search import tls_search
from .injection_recovery_output import irplot
try:
    from exoworlds.tess import tessio
except:
    pass
from ..v2.classes import allesclass2
from ..v2.translator import translate

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
                


#::: check if a certain injection has already been logged, or if it needs to be done
def to_do_or_not_to_do_that_is_the_question(ex, period, rplanet):
    #if no previous logfile exists, then do it
    if ex is None:
        return True #do it
    #if a previous logfile exists and it includes those period and rplanet values (to 1e-6 precision), then skip this pair
    else:
        N = len(np.atleast_1d(ex['inj_period']))
        for i in range(N):
            if (np.abs(period - np.atleast_1d(ex['inj_period'])[i]) < 1e-6) and (np.abs(rplanet - np.atleast_1d(ex['inj_rplanet'])[i]) < 1e-6):
                return False #skip it
    #if the pair was not skipped before, then go ahead and do it
    return True #do it
    


def inject(time, flux, flux_err, epoch, period, r_companion_earth, r_host=1, m_host=1):
    dic = translate(period=period,
                    r_companion_earth=r_companion_earth,
                    m_companion_earth=0,
                    r_host=r_host,
                    m_host=m_host, 
                    quiet=True)
    alles = allesclass2()
    alles.settings = {'companions_phot':['b'], 'inst_phot':['buf']}
    alles.params = {'b_rr':dic['rr'], 'b_rsuma':dic['rsuma'], 'b_epoch':epoch, 'b_period':period}
    alles.params_host = {'R_host':r_host, 'M_host':m_host}
    alles.fill()   
    model_flux = alles.generate_model('buf', 'flux', time)
    return model_flux + flux - 1
    


###############################################################################
#::: Inject an ellc transit and TLS search on an input lightcurve
###############################################################################
def inject_and_tls_search(time, flux, flux_err, 
                          period_grid, r_companion_earth_grid, 
                          inj_options=None,
                          SNR_threshold=5.,
                          known_transits=None, 
                          tls_kwargs=None,
                          tls_options=None):
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
        
    options : None or dict, keywords:
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

    #::: format inputs
    period_grid = np.atleast_1d(period_grid)
    r_companion_earth_grid = np.atleast_1d(r_companion_earth_grid)
    
    if inj_options is None: inj_options={}
    inj_options['logfname'] = 'injection_recovery_test.csv'
    
    if tls_kwargs is None: tls_kwargs = {}
    if 'R_star' not in tls_kwargs: tls_kwargs['R_star'] = 1.
    if 'M_star' not in tls_kwargs: tls_kwargs['M_star'] = 1.
    
    #::: set up a logfile
    ex = setup_logfile(inj_options['logfname'])
    
    #::: cycle through all periods and rplanets
    print('\n', flush=True)
    for period, r_companion_earth in tqdm(itertools.product(period_grid, r_companion_earth_grid), total=len(period_grid)*len(r_companion_earth_grid)): #combining the two for loops
        
        if to_do_or_not_to_do_that_is_the_question(ex, period, r_companion_earth):
            print('\tP = '+str(period)+' days, Rp = '+str(r_companion_earth)+' Rearth --> do')
            epoch = time[0] + np.random.random()*period
            
            flux2 = inject(time, flux, flux_err, epoch, period, r_companion_earth, tls_kwargs['R_star'], tls_kwargs['M_star'])
            
            tls_kwargs['period_min'] = period - 0.5
            tls_kwargs['period_max'] = period + 0.5
            tls_kwargs['transit_depth_min'] = 0.5*(( (r_companion_earth*u.Rearth)/(tls_kwargs['R_star']*u.Rsun) ).decompose().value)**2
            results_all = tls_search(time, flux2, flux_err,
                                     SNR_threshold=SNR_threshold,
                                     known_transits=known_transits,
                                     tls_kwargs=tls_kwargs,
                                     options=tls_options)
            if len(results_all)>0:
                for r in results_all:
                    with open(inj_options['logfname'],'a') as f:
                        f.write(format(period, '.5f')+','+
                                format(r_companion_earth, '.5f')+','+
                                format(r.period, '.5f')+','+
                                format(r.depth, '.5f')+','+
                                format(r.duration, '.5f')+','+
                                format(r.SDE, '.5f')+','+
                                format(r.snr, '.5f')+'\n')
            else:
                with open(inj_options['logfname'],'a') as f:
                    f.write(format(period, '.5f')+','+
                            format(r_companion_earth, '.5f')+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+'\n')
                
        else:
            print('\tP = '+str(period)+' days, Rp = '+str(r_companion_earth)+' Rearth --> skipped (already exists)')
    
    #::: finish
    irplot(inj_options['logfname'])
    print('Done.')
            


def get_tls_kwargs_by_tic(tic_id, tls_kwargs=None):
    u, R_star, R_star_lerr, R_star_uerr, M_star, M_star_lerr, M_star_uerr = catalog_info(TIC_ID=int(tic_id))
    print('TICv8 info:')
    print('Quadratic limb darkening u_0, u_1', u[0], u[1])
    print('Stellar radius', R_star, '+', R_star_lerr, '-', R_star_uerr)
    print('Stellar mass', M_star, '+', M_star_lerr, '-', M_star_uerr)
    if tls_kwargs is None: tls_kwargs = {}
    tls_kwargs['R_star']=float(R_star)
    tls_kwargs['R_star_min']=R_star-3*R_star_lerr
    tls_kwargs['R_star_max']=R_star+3*R_star_uerr
    tls_kwargs['M_star']=float(M_star)
    tls_kwargs['M_star_min']=M_star-3*M_star_lerr
    tls_kwargs['M_star_max']=M_star+3*M_star_uerr
    tls_kwargs['u']=u    
    return tls_kwargs



###############################################################################
#::: Inject an ellc transit and TLS search using tessio
###############################################################################
def inject_and_tls_search_by_tic(tic_id, 
                                  period_grid, r_companion_earth_grid, 
                                  SNR_threshold=5.,
                                  known_transits=None, 
                                  tls_kwargs=None,
                                  options=None):
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
    period_grid = np.atleast_1d(period_grid)
    r_companion_earth_grid = np.atleast_1d(r_companion_earth_grid)

    #::: load data
    dic = tessio.get(tic_id)
    time, flux, flux_err = dic['time'], dic['flux'], dic['flux_err']
    
    #::: load tls kwargs
    tls_kwargs = get_tls_kwargs_by_tic(tic_id, tls_kwargs=tls_kwargs)
    
    #::: run
    inject_and_tls_search(time, flux, flux_err, 
                          period_grid, r_companion_earth_grid, 
                          SNR_threshold=SNR_threshold,
                          known_transits=known_transits, 
                          tls_kwargs=tls_kwargs,
                          options=options)
            


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
    
    

    