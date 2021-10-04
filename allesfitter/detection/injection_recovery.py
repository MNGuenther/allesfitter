#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:41:19 2019

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
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itertools
from transitleastsquares import period_grid as tls_period_grid
from astropy.constants import G
from astropy import units as u
import ellc
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
#solves python>=3.8 issues, see https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from contextlib import closing

#::: my modules
from ..lightcurves import index_eclipses_smart
from .transit_search import get_tls_kwargs_by_tic, tls_search
from .injection_recovery_output import irplot
try:
    from exoworlds.tess import tessio
except:
    pass

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: settings and constants
np.random.seed(42)
eps = 1e-12




###############################################################################
#::: helper functiomns
###############################################################################

#::: set up the logfile
def setup_logfile(options):
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
    f = os.path.join(options['outdir'],options['logfname'])
    
    if os.path.exists(f):
        if options['user_response'] is None:
            response = input('Log file already exists. Do you want to (1) overwrite it, (2) append missing rows, or (3) abort?\n')
        else:
            response = options['user_response']
        if response == '1':   
            print('Overwriting log file...')
            with open(f,'w') as f: 
                f.write('inj_period,inj_rplanet,inj_depth,inj_epoch,tls_period,tls_depth,tls_duration,tls_epoch,tls_SNR,tls_SDE,tls_FAP\n')
            return None
        elif response == '2': 
            print('Appending missing rows to log file...')
            ex = np.genfromtxt(f, names=True, dtype=None, delimiter=',')
            return ex
        else:
            raise ValueError('User aborted. Response was: "'+str(response)+'"')
    else:
        with open(f,'w') as f: 
            f.write('inj_period,inj_rplanet,inj_depth,inj_epoch,tls_period,tls_depth,tls_duration,tls_epoch,tls_SNR,tls_SDE,tls_FAP\n')
        return None
                


#round and str to avoid float issues
def mystr(x):
    return '{:.3f}'.format(x)


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
    


# def inject(time, flux, flux_err, epoch, period, r_companion_earth, r_host=1, m_host=1, ldc=[0.5,0.5], window=None):
#     dic = translate(period=period,
#                     r_companion_earth=r_companion_earth,
#                     m_companion_earth=0,
#                     r_host=r_host,
#                     m_host=m_host, 
#                     quiet=True)
#     alles = allesclass2()
#     alles.settings = {'companions_phot':['b'], 'inst_phot':['telescope'], 'host_ld_law_telescope':'quad'}
#     alles.params = {'b_rr':dic['rr'], 'b_rsuma':dic['rsuma'], 'b_epoch':epoch, 'b_period':period, 'host_ldc_telescope':ldc}
#     alles.params_host = {'R_host':r_host, 'M_host':m_host}
#     alles.fill()   
    
#     ind_ecl1, ind_ecl2, ind_out = index_eclipses_smart(time, alles.params['b_epoch'], alles.params['b_period'], alles.params['b_rr'], alles.params['b_rsuma'], alles.params['b_cosi'], alles.params['b_f_s'], alles.params['b_f_c'], 
#                                                         extra_factor=1.1)
#     model_flux = np.ones_like(time)
#     model_flux[ind_ecl1] = alles.generate_model(time[ind_ecl1], inst='telescope', key='flux')
#     inj_flux = model_flux + flux - 1
    
#     if window in ['transit', 'eclipse']:
#         ind_ecl1, ind_ecl2, ind_out = index_eclipses_smart(time, alles.params['b_epoch'], alles.params['b_period'], alles.params['b_rr'], alles.params['b_rsuma'], alles.params['b_cosi'], alles.params['b_f_s'], alles.params['b_f_c'], 
#                                                             extra_factor=10)
#         return time[ind_ecl1], inj_flux[ind_ecl1], flux_err[ind_ecl1]
    
#     elif window is None:
#         return time, inj_flux, flux_err
    


# ::: a fast injection for a simple, circular, and small planet
def inject(time, flux, flux_err, epoch, period, r_companion_earth, r_host=1, m_host=1, ldc=[0.5,0.5], dil=0, return_depth=False, window=None):
    a = ( (G/(4*np.pi**2) * (period*u.d)**2 * (m_host*u.Msun))**(1./3.) ).to(u.AU).value  #in AU 
    r_host_over_a = ((r_host*u.Rsun)/(a*u.AU)).decompose().value #unitless
    r_companion_over_a = ((r_companion_earth*u.Rearth)/(a*u.AU)).decompose().value #unitless
    rr = r_companion_over_a/r_host_over_a
    rsuma = r_companion_over_a + r_host_over_a
    
    ind_ecl1, ind_ecl2, ind_out = index_eclipses_smart(time, epoch, period, rr, rsuma, 0., 0., 0., extra_factor=1.1)
    model_flux = np.ones_like(time)
    if len(ind_ecl1) > 0:
        model_flux[ind_ecl1] = ellc.lc(
                                      t_obs = time[ind_ecl1], 
                                      radius_1 = r_host_over_a, 
                                      radius_2 = r_companion_over_a, 
                                      sbratio = 0., 
                                      incl = 90., 
                                      t_zero = epoch,
                                      period = period,
                                      ldc_1 = ldc,
                                      ld_1 = 'quad',
                                      light_3 = dil / (1. - dil),
                                      verbose = False
                                      )
    inj_flux = model_flux + flux - 1
    
    if return_depth == True:
        depth = 1. - ellc.lc(
                            t_obs = [epoch], 
                            radius_1 = r_host_over_a, 
                            radius_2 = r_companion_over_a, 
                            sbratio = 0., 
                            incl = 90., 
                            t_zero = epoch,
                            period = period,
                            ldc_1 = ldc,
                            ld_1 = 'quad',
                            light_3 = dil / (1. - dil),
                            verbose = False
                            )[0]
        return time, inj_flux, flux_err, depth
    
    else:
        return time, inj_flux, flux_err
    
    # if window in ['transit', 'eclipse']:
    #     ind_ecl1, ind_ecl2, ind_out = index_eclipses_smart(time, epoch, period, rr, rsuma, 0., 0., 0., extra_factor=10)
    #     # inj_flux[ind_out] = np.nan
    #     # return time, inj_flux, flux_err
    #     return time[ind_ecl1], inj_flux[ind_ecl1], flux_err[ind_ecl1]
    # elif window is None:
    #     return time, inj_flux, flux_err

    

###############################################################################
#::: Inject an ellc transit and TLS search on an input lightcurve
###############################################################################
def inject_and_tls_search(time, flux, flux_err, 
                          period_grid, r_companion_earth_grid,
                          dil=0.,
                          known_transits=None, 
                          tls_kwargs=None,
                          wotan_kwargs=None,
                          options=None):
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
        
    known_transits : None or dict
        see allesfitter.detection.tls_search()
        
    tls_kwargs : None or dict
        see allesfitter.detection.tls_search()
        
    wotan_kwargs : None or dict
        see allesfitter.detection.tls_search()
        
    options : None or dict, keywords:
        logfname : str
            file path and name for the log file
            default 
        for the rest see allesfitter.detection.wotan_kwargs()
        
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
    
    if tls_kwargs is None: tls_kwargs = {}
    if 'u' not in tls_kwargs: tls_kwargs['u'] = [0.4804, 0.1867] #TLS defaults
    if 'R_star' not in tls_kwargs: tls_kwargs['R_star'] = 1. #TLS defaults
    if 'M_star' not in tls_kwargs: tls_kwargs['M_star'] = 1. #TLS defaults
    if 'SNR_threshold' not in tls_kwargs: tls_kwargs['SNR_threshold'] = 3. #use the lowest SNR to make multiple histograms from SNR 3 to 7
    tls_kwargs['use_threads'] = 1 #no multiprocessing for TLS itself; all multiprocessing is done here
    
    # wotan_kwargs --> allesfitter.detection.tls_search()
    # if wotan_kwargs is None: wotan_kwargs = {}
    # if 'slide_clip' not in wotan_kwargs: wotan_kwargs['slide_clip'] = None  #don't use slide clip (too slow)
    # if 'flatten' not in wotan_kwargs: wotan_kwargs['flatten'] = {} #use a 1 day biweight as default
    # if wotan_kwargs['flatten'] is not None:
    #     if 'method' not in wotan_kwargs['flatten']: wotan_kwargs['flatten']['method'] = 'biweight'
    #     if 'window_length' not in wotan_kwargs['flatten']: wotan_kwargs['flatten']['window_length'] = 1.

    if options is None: options={}
    if 'logfname' not in options: options['logfname'] = 'injection_recovery_test.csv'
    if 'save_plot' not in options: options['save_plot'] = '2' #save TLS plots only
    if 'outdir' not in options: options['outdir'] = 'results'
    if not os.path.exists(options['outdir']): os.makedirs(options['outdir'])
    if 'quiet' not in options: options['quiet'] = True
    if 'multiprocess' not in options: options['multiprocess'] = True
    if 'multiprocess_cores' not in options: options['multiprocess_cores'] = cpu_count()-1
    if 'user_response' not in options: options['user_response'] = None
    

    #::: set up a logfile
    ex = setup_logfile( options )
    
    
    #::: def execution function (for multiprocessing)
    def executer(arg1):
        seed, period, r_companion_earth = arg1
        np.random.seed(seed) #need a fresh seed for every multiprocess
        
        #::: check if it already exists; if not, proceed
        if to_do_or_not_to_do_that_is_the_question(ex, period, r_companion_earth):
           
            #::: draw a random period
            epoch = time[0] + np.random.random()*period
            
            #::: inject transit
            time2, flux2, flux_err2, depth = inject(time, flux, flux_err, epoch, period, r_companion_earth, tls_kwargs['R_star'], tls_kwargs['M_star'], tls_kwargs['u'], dil, return_depth=True, window=None)
            
            #::: pick tight period_min and period_max for TLS, to speed things up
            tls_periods = tls_period_grid(R_star=tls_kwargs['R_star'], M_star=tls_kwargs['M_star'], time_span=time[-1]-time[0])
            ind = np.argmin(np.abs(tls_periods-period))
            tls_kwargs['period_min'] = tls_periods[min(ind+500,len(tls_periods)-1)] #tls_periods is sorted in descending order
            tls_kwargs['period_max'] = tls_periods[max(ind-500,0)] #tls_periods is sorted in descending order

            #::: copy specific options
            options2 = options.copy()
            options2['outdir'] = os.path.join(options['outdir'],'details','P_'+mystr(period)+'_Rp_'+mystr(r_companion_earth))
            options2['inj_period'] = period
            
            #::: run tls with all kwargs
            results_all = tls_search(time2, flux2, flux_err2,
                                     known_transits=known_transits,
                                     tls_kwargs=tls_kwargs,
                                     wotan_kwargs=wotan_kwargs,
                                     options=options2)
            
            #::: write results to file
            if len(results_all)>0:
                for r in results_all:
                    with open(os.path.join(options['outdir'],options['logfname']),'a') as f:
                        f.write(format(period, '.5f')+','+
                                format(r_companion_earth, '.5f')+','+
                                format(depth*1e3, '.5f')+','+ #in ppt
                                format(epoch, '.5f')+','+
                                format(r.period, '.5f')+','+
                                format(r.depth, '.5f')+','+
                                format(r.duration, '.5f')+','+
                                format(r.T0, '.5f')+','+
                                format(r.snr, '.5f')+','+
                                format(r.SDE, '.5f')+','+
                                format(r.FAP, '.5f')+'\n')
            else:
                with open(os.path.join(options['outdir'],options['logfname']),'a') as f:
                    f.write(format(period, '.5f')+','+
                            format(r_companion_earth, '.5f')+','+
                            format(depth*1e3, '.5f')+','+ #in ppt
                            format(epoch, '.5f')+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+','+
                            'nan'+'\n')
                           
        else:
            pass
        
        
    #::: store all args
    arg_all = []
    seed = 42
    for period, r_companion_earth in itertools.product(period_grid, r_companion_earth_grid): #combining the two for loops
        arg1 = (seed, period, r_companion_earth)
        arg_all.append( arg1 )
        seed += 1
            
        
    #::: cycle through all periods and rplanets
    print('\n', flush=True)
    
    if options['multiprocess'] is False:
        print('Running on a single core')
        for arg1 in tqdm(arg_all):
            executer(arg1)
        
    else:
        print('Running on '+str(options['multiprocess_cores'])+' cores.')
        with closing( Pool(processes=options['multiprocess_cores']) ) as p:
            r = list( tqdm( p.imap(executer, arg_all), total=len(arg_all) ) )
            p.close(); p.join(); p.clear() #needed for pathos, despite closing()
        del p #needed for pathos, despite closing()
                 
        
    #::: finish
    irplot(os.path.join(options['outdir'],options['logfname']), 3, options=options)
    irplot(os.path.join(options['outdir'],options['logfname']), 4, options=options)
    irplot(os.path.join(options['outdir'],options['logfname']), 5, options=options)
    irplot(os.path.join(options['outdir'],options['logfname']), 6, options=options)
    irplot(os.path.join(options['outdir'],options['logfname']), 7, options=options)
    print('Done.')
            


# def get_tls_kwargs_by_tic(tic_id, tls_kwargs=None):
#     u, R_star, R_star_lerr, R_star_uerr, M_star, M_star_lerr, M_star_uerr = catalog_info(TIC_ID=int(tic_id))
#     print('TICv8 info:')
#     print('Quadratic limb darkening u_0, u_1', u[0], u[1])
#     print('Stellar radius', R_star, '+', R_star_lerr, '-', R_star_uerr)
#     print('Stellar mass', M_star, '+', M_star_lerr, '-', M_star_uerr)
#     if tls_kwargs is None: tls_kwargs = {}
#     tls_kwargs['R_star']=float(R_star)
#     tls_kwargs['R_star_min']=R_star-3*R_star_lerr
#     tls_kwargs['R_star_max']=R_star+3*R_star_uerr
#     tls_kwargs['M_star']=float(M_star)
#     tls_kwargs['M_star_min']=M_star-3*M_star_lerr
#     tls_kwargs['M_star_max']=M_star+3*M_star_uerr
#     tls_kwargs['u']=u    
#     return tls_kwargs



###############################################################################
#::: Inject an ellc transit and TLS search using tessio
###############################################################################
def inject_and_tls_search_by_tic(tic_id, 
                                  period_grid, r_companion_earth_grid, 
                                  known_transits=None, 
                                  tls_kwargs=None,
                                  wotan_kwargs=None,
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
    time, flux, flux_err = tessio.get(tic_id, unpack=True)
    if (time is None) or (len(time)<5000): #abort if no reasonable data was found (at least 5000 exposures, aka 7 days of data)
        print('Skipped. No data was found for TIC '+tic_id+'.')
        return None
    
    #::: load tls kwargs
    tls_kwargs = get_tls_kwargs_by_tic(tic_id, tls_kwargs=tls_kwargs)
    if any(np.isnan([tls_kwargs['u'][0], tls_kwargs['u'][1], tls_kwargs['R_star'], tls_kwargs['M_star']])): #abort if TICv8 doesn't hold the stellar info
        print('Skipped. No TICv8 information was found for TIC '+tic_id+'.')
        return None
    
    #::: run
    inject_and_tls_search(time, flux, flux_err, 
                          period_grid, r_companion_earth_grid, 
                          known_transits=known_transits, 
                          tls_kwargs=tls_kwargs,
                          wotan_kwargs=wotan_kwargs,
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
    
    

    