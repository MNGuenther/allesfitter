#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:19:30 2018

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
#import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from corner import corner
from tqdm import tqdm 
from astropy.constants import M_earth, M_jup, M_sun, R_earth, R_jup, R_sun, au
import copy

#::: allesfitter modules
from . import config
from .utils import latex_printer
from .general_output import logprint
from .priors.simulate_PDF import simulate_PDF
from .computer import update_params, calculate_model
from .exoworlds_rdx.lightcurves.index_transits import index_transits

#::: constants
#M_earth = 5.9742e+24 	#kg 	Earth mass
#M_jup   = 1.8987e+27 	#kg 	Jupiter mass
#M_sun   = 1.9891e+30 	#kg 	Solar mass
#R_earth = 6378136      #m 	Earth equatorial radius
#R_jup   = 71492000 	#m 	Jupiter equatorial radius
#R_sun   = 695508000 	#m 	Solar radius




def derive(samples, mode):
    '''
    Derives parameter of the system using Winn 2010
    
    Input:
    ------
    samples : array
        samples from the mcmc or nested sampling
    mode : str
        'mcmc' or 'ns'
        
    Returns:
    --------
    derived_samples : dict 
        with keys 'i', 'R1a', 'R2a', 'k', 'depth_undiluted', 'b_tra', 'b_occ', 'Ttot', 'Tfull'
        each key contains all the samples derived from the MCMC samples 
        (not mean values, but pure samples!)
        i = inclination 
        R1a = R1/a, radius companion over semiamplitude
        R2a = R2/a, radius star over semiamplitude
        Ttot = T_{1-4}, total transit width 
        Tfull = T_{2-3}, full transit width
        
    Output:
    -------
    latex table of results
    corner plot of derived values posteriors
    '''
    
    N_samples = samples.shape[0]
    

    ###############################################################################
    #::: stellar 'posteriors'
    ###############################################################################
    buf = np.genfromtxt( os.path.join(config.BASEMENT.datadir,'params_star.csv'), delimiter=',', names=True, dtype=None, encoding='utf-8', comments='#' )
    star = {}
    star['R_star'] = simulate_PDF(buf['R_star'], buf['R_star_lerr'], buf['R_star_uerr'], size=N_samples, plot=False)
    star['M_star'] = simulate_PDF(buf['M_star'], buf['M_star_lerr'], buf['M_star_uerr'], size=N_samples, plot=False)
    star['Teff_star'] = simulate_PDF(buf['Teff_star'], buf['Teff_star_lerr'], buf['Teff_star_uerr'], size=N_samples, plot=False)
    
    
    
    ###############################################################################
    #::: derive all the params
    ###############################################################################
    companions = config.BASEMENT.settings['companions_all']
    
    def get_params(key):
        ind = np.where(config.BASEMENT.fitkeys==key)[0]
        if len(ind)==1: return samples[:,ind].flatten() #if it was fitted for
        else: 
            try:
                return config.BASEMENT.params[key] #else take the input value
            except KeyError:
                return np.nan
        
    def sin_d(alpha): return np.sin(np.deg2rad(alpha))
    def cos_d(alpha): return np.cos(np.deg2rad(alpha))
    def arcsin_d(x): return np.rad2deg(np.arcsin(x))
    def arccos_d(x): return np.rad2deg(np.arccos(x))

    derived_samples = {}
    for companion in companions:
        
        #::: radii
        derived_samples[companion+'_R_star/a'] = get_params(companion+'_rsuma') / (1. + get_params(companion+'_rr'))
        derived_samples[companion+'_R_companion/a'] = get_params(companion+'_rsuma') * get_params(companion+'_rr') / (1. + get_params(companion+'_rr'))
        derived_samples[companion+'_R_companion_(R_earth)'] = star['R_star'] * get_params(companion+'_rr') * R_sun.value / R_earth.value #in R_earth
        derived_samples[companion+'_R_companion_(R_jup)'] = star['R_star'] * get_params(companion+'_rr') * R_sun.value / R_jup.value #in R_jup
        

        #::: orbit
        derived_samples[companion+'_a_(R_sun)'] = star['R_star'] / derived_samples[companion+'_R_star/a']   
        derived_samples[companion+'_a_(AU)'] = derived_samples[companion+'_a_(R_sun)'] * R_sun.value/au.value
        derived_samples[companion+'_i'] = arccos_d(get_params(companion+'_cosi')) #in deg
        derived_samples[companion+'_e'] = get_params(companion+'_f_s')**2 + get_params(companion+'_f_c')**2
        derived_samples[companion+'_e_sinw'] = get_params(companion+'_f_s') * np.sqrt(derived_samples[companion+'_e'])
        derived_samples[companion+'_e_cosw'] = get_params(companion+'_f_c') * np.sqrt(derived_samples[companion+'_e'])
        derived_samples[companion+'_w'] = arccos_d( get_params(companion+'_f_c') / np.sqrt(derived_samples[companion+'_e']) ) #in deg, from 0 to 180
        if np.isnan(derived_samples[companion+'_w']).all():
            derived_samples[companion+'_w'] = 0.
        
        
        #::: mass
        if companion+'_K' in config.BASEMENT.params:
            a_1 = 0.019771142 * get_params(companion+'_K') * get_params(companion+'_period') * np.sqrt(1. - derived_samples[companion+'_e']**2)/sin_d(derived_samples[companion+'_i'])
    #        derived_samples[companion+'_a_rv'] = (1.+1./ellc_params[companion+'_q'])*a_1
            derived_samples[companion+'_q'] = 1./(( derived_samples[companion+'_a_(R_sun)'] / a_1 ) - 1.)
            derived_samples[companion+'_M_companion_(M_earth)'] = derived_samples[companion+'_q'] * star['M_star'] * M_sun.value / M_earth.value #in M_earth
            derived_samples[companion+'_M_companion_(M_jup)'] = derived_samples[companion+'_q'] * star['M_star'] * M_sun.value / M_jup.value #in M_jup
        else:
            derived_samples[companion+'_M_companion'] = None
            
            
        #::: time of occultation    
        if config.BASEMENT.settings['secondary_eclipse'] is True:
            derived_samples[companion+'_dt_occ'] = get_params(companion+'_period')/2. * (1. + 4./np.pi * derived_samples[companion+'_e'] * cos_d(derived_samples[companion+'_w'])  ) #approximation
        else:
            derived_samples[companion+'_dt_occ'] = None
        
        
        #::: impact params
        derived_samples[companion+'_b_tra'] = (1./derived_samples[companion+'_R_star/a']) * get_params(companion+'_cosi') * ( (1.-derived_samples[companion+'_e']**2) / ( 1.+derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
        
        if config.BASEMENT.settings['secondary_eclipse'] is True:
            derived_samples[companion+'_b_occ'] = (1./derived_samples[companion+'_R_star/a']) * get_params(companion+'_cosi') * ( (1.-derived_samples[companion+'_e']**2) / ( 1.-derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
        else:
            derived_samples[companion+'_b_occ'] = None
        
        
        #::: transit duration 
        derived_samples[companion+'_T_tra_tot'] = get_params(companion+'_period')/np.pi *24.  \
                                  * np.arcsin( derived_samples[companion+'_R_star/a'] \
                                             * np.sqrt( (1.+get_params(companion+'_rr'))**2 - derived_samples[companion+'_b_tra']**2 )\
                                             / sin_d(derived_samples[companion+'_i']) ) #in h
        derived_samples[companion+'_T_tra_full'] = get_params(companion+'_period')/np.pi *24.  \
                                  * np.arcsin( derived_samples[companion+'_R_star/a'] \
                                             * np.sqrt( (1.-get_params(companion+'_rr'))**2 - derived_samples[companion+'_b_tra']**2 )\
                                             / sin_d(derived_samples[companion+'_i']) ) #in h
                                  

        #::: primary and secondary eclipse depths (per inst) / transit and occultation depths (per inst)
        for inst in config.BASEMENT.settings['inst_phot']:
            
            N_less_samples = 1000
            
            derived_samples[companion+'_depth_tr_diluted_'+inst] = np.zeros(N_less_samples)*np.nan #1e3*get_params(companion+'_rr')**2 #in ppt
            derived_samples[companion+'_depth_occ_diluted_'+inst]  = np.zeros(N_less_samples)*np.nan
            derived_samples[companion+'_ampl_ellipsoidal_diluted_'+inst]  = np.zeros(N_less_samples)*np.nan
            derived_samples[companion+'_ampl_sbratio_diluted_'+inst] = np.zeros(N_less_samples)*np.nan
            derived_samples[companion+'_ampl_geom_albedo_diluted_'+inst] = np.zeros(N_less_samples)*np.nan
            derived_samples[companion+'_ampl_gdc_diluted_'+inst] = np.zeros(N_less_samples)*np.nan
            
            
            #::: iterate through all samples, draw different models and measure the depths
            logprint('Deriving primary and secondary eclipse depths / transit and occultation depths from model curves...')
            for i in tqdm( range(N_less_samples) ):
                
                #::: setting the model params and time axes
                ii = np.random.randint(low=0,high=samples.shape[0])
                s = samples[ii,:]
                p = update_params(s)
                xx0 = p[companion+'_epoch'] + np.arange(-0.25*p[companion+'_period'], 0.75*p[companion+'_period'], 1./24./60.) #one full orbit without the transit (in time units), in 1 min steps
                ind_tr, ind_out = index_transits(xx0, p[companion+'_epoch'], p[companion+'_period'], np.median(derived_samples[companion+'_T_tra_tot'])/24.) #find the primary eclipse / transit
            
            
                #:: calculating primary eclipse / transit depth
                xx = xx0[ind_tr]
                model = calculate_model(p, inst, 'flux', xx=xx) #evaluated on xx (!)
                derived_samples[companion+'_depth_tr_diluted_'+inst][i] = ( 1. - np.min(model) ) * 1e3 #in ppt
        
        
                #:: calculating secondary eclipse / occultation depth
                if config.BASEMENT.settings['secondary_eclipse'] is True:
                    xx = xx0[ind_out]

                    def plottle(fname, title, model):
                        if i==0:
                            fig = plt.figure()
                            plt.plot(xx,model,'b.')
                            plt.title(title)
                            fig.savefig(os.path.join(config.BASEMENT.outdir,fname), bbox_inches='tight')
                            plt.close(fig)
                    
                # logprint('Deriving occultation depths from model curves...')
                # for i in tqdm( range(N_less_samples) ):
                #     ii = np.random.randint(low=0,high=samples.shape[0])
                #     s = samples[ii,:]
                #     p = update_params(s)
                #     xx = p[companion+'_epoch'] + np.arange(-0.25*p[companion+'_period'], 0.75*p[companion+'_period'], 1./24./60.) #one full orbit without the transit (in time units), in 1 min steps
                #     ind_tr, ind_out = index_transits(xx, p[companion+'_epoch'], p[companion+'_period'], np.median(derived_samples[companion+'_T_tra_tot'])/24.) #ignore the secondary eclipse here
                #     xx = xx[ind_out]
                    
                    #::: occultation depth (very simplistic; only ok for exoplanets, not binaries)
                    model = calculate_model(p, inst, 'flux', xx=xx) #evaluated on xx (!)
                    if i==0: plottle('phase_curve_occultation_depth.pdf', 'occultation depth', model)
                    derived_samples[companion+'_depth_occ_diluted_'+inst][i] = ( np.max(model) - np.min(model) ) * 1e6 #in ppm


                    #::: amplitude of ellipsoidal modulation alone (ignoring all other effects)
                    p2 = copy.deepcopy(p)
#                    save_host_shape_inst = copy.deepcopy(config.BASEMENT.settings['host_shape_'+inst])
#                    save_companion_shape_inst = copy.deepcopy(config.BASEMENT.settings[companion+'_shape_'+inst])
                    p2['b_sbratio_'+inst] = 0
#                    config.BASEMENT.settings['host_shape_'+inst] = 'sphere'
#                    config.BASEMENT.settings['b_shape_'+inst] = 'sphere'
                    p2['b_geom_albedo_'+inst] = 0
                    p2['host_gdc_'+inst] = 0
                    p2['host_bfac_'+inst] = 0
                    model = calculate_model(p2, inst, 'flux', xx=xx) #evaluated on xx (!)
                    plottle('phase_curve_ellipsoidal.pdf', 'ellipsoidal modulation', model)
                    derived_samples[companion+'_ampl_ellipsoidal_diluted_'+inst][i] = ( np.max(model) - 1. ) * 1e6 #in ppm
#                    config.BASEMENT.settings['host_shape_'+inst] = save_host_shape_inst
#                    config.BASEMENT.settings['b_shape_'+inst] = save_companion_shape_inst
                    
                    
                     #::: amplitude of sbratio modulation alone (ignoring all other effects)
                    p2 = copy.deepcopy(p)
                    save_host_shape_inst = copy.deepcopy(config.BASEMENT.settings['host_shape_'+inst])
                    save_companion_shape_inst = copy.deepcopy(config.BASEMENT.settings[companion+'_shape_'+inst])
#                    p2['b_sbratio_'+inst] = 0
                    config.BASEMENT.settings['host_shape_'+inst] = 'sphere'
                    config.BASEMENT.settings['b_shape_'+inst] = 'sphere'
                    p2['b_geom_albedo_'+inst] = 0
                    p2['host_gdc_'+inst] = 0
                    p2['host_bfac_'+inst] = 0
                    model = calculate_model(p2, inst, 'flux', xx=xx) #evaluated on xx (!)
                    plottle('phase_curve_sbratio.pdf', 'sbratio depth', model)
                    derived_samples[companion+'_ampl_sbratio_diluted_'+inst][i] = ( np.min(model) - 1. ) * 1e6 #in ppm
                    config.BASEMENT.settings['host_shape_'+inst] = save_host_shape_inst
                    config.BASEMENT.settings['b_shape_'+inst] = save_companion_shape_inst
                    
                    
                     #::: amplitude of geom. albedo modulation alone (ignoring all other effects)
                    p2 = copy.deepcopy(p)
                    save_host_shape_inst = copy.deepcopy(config.BASEMENT.settings['host_shape_'+inst])
                    save_companion_shape_inst = copy.deepcopy(config.BASEMENT.settings[companion+'_shape_'+inst])
                    p2['b_sbratio_'+inst] = 0
                    config.BASEMENT.settings['host_shape_'+inst] = 'sphere'
                    config.BASEMENT.settings['b_shape_'+inst] = 'sphere'
#                    p2['b_geom_albedo_'+inst] = 0
                    p2['host_gdc_'+inst] = 0
                    p2['host_bfac_'+inst] = 0
                    model = calculate_model(p2, inst, 'flux', xx=xx) #evaluated on xx (!)
                    plottle('phase_curve_geom_albedo.pdf', 'geom albedo modulation', model)
                    derived_samples[companion+'_ampl_geom_albedo_diluted_'+inst][i] = ( np.max(model) - 1. ) * 1e6 #in ppm
                    config.BASEMENT.settings['host_shape_'+inst] = save_host_shape_inst
                    config.BASEMENT.settings['b_shape_'+inst] = save_companion_shape_inst
                    
                    
                     #::: amplitude of gravity darkening modulation alone (ignoring all other effects)
                    p2 = copy.deepcopy(p)
                    save_host_shape_inst = copy.deepcopy(config.BASEMENT.settings['host_shape_'+inst])
                    save_companion_shape_inst = copy.deepcopy(config.BASEMENT.settings[companion+'_shape_'+inst])
                    p2['b_sbratio_'+inst] = 0
                    config.BASEMENT.settings['host_shape_'+inst] = 'sphere'
                    config.BASEMENT.settings['b_shape_'+inst] = 'sphere'
                    p2['b_geom_albedo_'+inst] = 0
#                    p2['host_gdc_'+inst] = 0
                    p2['host_bfac_'+inst] = 0
                    model = calculate_model(p2, inst, 'flux', xx=xx) #evaluated on xx (!)
                    plottle('phase_curve_gdc.pdf', 'grav darkening modulation', model)
                    derived_samples[companion+'_ampl_gdc_diluted_'+inst][i] = ( np.min(model) - 1. ) * 1e6 #in ppm
                    config.BASEMENT.settings['host_shape_'+inst] = save_host_shape_inst
                    config.BASEMENT.settings['b_shape_'+inst] = save_companion_shape_inst
                    
                    
            #resize the arrays to match the true N_samples (by redrawing the 1000 values)
            derived_samples[companion+'_depth_tr_diluted_'+inst]          = np.resize(derived_samples[companion+'_depth_tr_diluted_'+inst], N_samples)
            derived_samples[companion+'_depth_occ_diluted_'+inst]         = np.resize(derived_samples[companion+'_depth_occ_diluted_'+inst], N_samples)
            derived_samples[companion+'_ampl_ellipsoidal_diluted_'+inst]  = np.resize(derived_samples[companion+'_ampl_ellipsoidal_diluted_'+inst], N_samples)
            derived_samples[companion+'_ampl_sbratio_diluted_'+inst]      = np.resize(derived_samples[companion+'_ampl_sbratio_diluted_'+inst], N_samples)
            derived_samples[companion+'_ampl_geom_albedo_diluted_'+inst]  = np.resize(derived_samples[companion+'_ampl_geom_albedo_diluted_'+inst], N_samples)
            derived_samples[companion+'_ampl_gdc_diluted_'+inst]          = np.resize(derived_samples[companion+'_ampl_gdc_diluted_'+inst], N_samples)

    
        #::: undiluted (per companion; per inst)
        for inst in config.BASEMENT.settings['inst_phot']:
            dil = get_params('light_3_'+inst)
            if np.isnan(dil):
                dil = 0
        #        if np.mean(dil)<0.5: dil = 1-dil
        
            derived_samples[companion+'_depth_tr_undiluted_'+inst] = derived_samples[companion+'_depth_tr_diluted_'+inst] / (1. - dil) #in ppt
            derived_samples[companion+'_depth_occ_undiluted_'+inst] = derived_samples[companion+'_depth_occ_diluted_'+inst] / (1. - dil) #in ppm
            derived_samples[companion+'_ampl_ellipsoidal_undiluted_'+inst] = derived_samples[companion+'_ampl_ellipsoidal_diluted_'+inst] / (1. - dil) #in ppm
            derived_samples[companion+'_ampl_sbratio_undiluted_'+inst] = derived_samples[companion+'_ampl_sbratio_diluted_'+inst] / (1. - dil) #in ppm
            derived_samples[companion+'_ampl_geom_albedo_undiluted_'+inst] = derived_samples[companion+'_ampl_geom_albedo_diluted_'+inst] / (1. - dil) #in ppm
            derived_samples[companion+'_ampl_gdc_undiluted_'+inst] = derived_samples[companion+'_ampl_gdc_diluted_'+inst] / (1. - dil) #in ppm

        
        #::: equilibirum temperature
        #::: currently assumes Albedo of 0.3 and Emissivity of 1
        albedo = 0.3
        emissivity = 1.
        derived_samples[companion+'_Teq'] = star['Teff_star']  * ( (1.-albedo)/emissivity )**0.25 * np.sqrt(derived_samples[companion+'_R_star/a'] / 2.)
        
        
        #::: stellar density
        derived_samples[companion+'_host_density'] = 3. * np.pi * (1./derived_samples[companion+'_R_star/a'])**3. / (get_params(companion+'_period')*86400.)**2 / 6.67408e-8 #in cgs
  
    
        #::: the companion's surface gravity (individual posterior distribution for each companion; via Southworth+ 2007)
        try:
            derived_samples[companion+'_surface_gravity'] = 2. * np.pi / (get_params(companion+'_period')*86400.) * np.sqrt((1.-derived_samples[companion+'_e']**2)) * (get_params(companion+'_K')*1e5) / (derived_samples[companion+'_R_companion/a'])**2 / sin_d(derived_samples[companion+'_i'])
        except:
            derived_samples[companion+'_surface_gravity'] = 0.
        
        
        #::: period ratios (for ressonance studies)
        if len(companions)>1:
            for other_companion in companions:
                if other_companion is not companion:
                    derived_samples[companion+'_period/'+other_companion+'_period'] = get_params(companion+'_period') / get_params(other_companion+'_period')
                        
                    
        #::: limb darkening
        for inst in config.BASEMENT.settings['inst_all']:
            
            #::: host
            if config.BASEMENT.settings['host_ld_law_'+inst] is None:
                pass
                
            elif config.BASEMENT.settings['host_ld_law_'+inst] == 'lin':
                derived_samples['host_ldc_u1_'+inst] = get_params('host_ldc_q1_'+inst)
                
            elif config.BASEMENT.settings['host_ld_law_'+inst] == 'quad':
                derived_samples['host_ldc_u1_'+inst] = 2 * np.sqrt(get_params('host_ldc_q1_'+inst)) * get_params('host_ldc_q2_'+inst)
                derived_samples['host_ldc_u2_'+inst] = np.sqrt(get_params('host_ldc_q1_'+inst)) * (1. - 2. * get_params('host_ldc_q2_'+inst))
                
            elif config.BASEMENT.settings['host_ld_law_'+inst] == 'sing':
                raise ValueError("Sorry, I have not yet implemented the Sing limb darkening law.")
                
            else:
                print(config.BASEMENT.settings['host_ld_law_'+inst] )
                raise ValueError("Currently only 'none', 'lin', 'quad' and 'sing' limb darkening are supported.")
            
        
    #::: median stellar density
    derived_samples['mean_host_density'] = []
    for companion in companions:
        derived_samples['mean_host_density'] = np.append(derived_samples['mean_host_density'], derived_samples[companion+'_host_density'])
    
    

    
    ###############################################################################
    #::: write keys for output
    ###############################################################################
    names = []
    labels = []
    for companion in companions:
            
        names.append( companion+'_R_star/a' )
        labels.append( '$R_\star/a_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_R_companion/a'  )
        labels.append( '$R_\mathrm{'+companion+'}/a_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_R_companion_(R_earth)' )
        labels.append( '$R_\mathrm{'+companion+'}$ ($\mathrm{R_{\oplus}}$)' )
        
        names.append( companion+'_R_companion_(R_jup)' )
        labels.append( '$R_\mathrm{'+companion+'}$ ($\mathrm{R_{jup}}$)' )
        
        names.append( companion+'_a_(R_sun)' )
        labels.append( '$a_\mathrm{'+companion+'}$ ($\mathrm{R_{\odot}}$)' )
        
        names.append( companion+'_a_(AU)' )
        labels.append( '$a_\mathrm{'+companion+'}$ (AU)' )
        
        names.append( companion+'_i' )
        labels.append( '$i_\mathrm{'+companion+'}$ (deg)' )
        
        names.append( companion+'_e' )
        labels.append( '$e_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_w' )
        labels.append( '$w_\mathrm{'+companion+'}$ (deg)' )
        
        names.append( companion+'_M_companion_(M_earth)' )
        labels.append( '$M_\mathrm{'+companion+'}$ ($\mathrm{M_{\oplus}}$)' )
        
        names.append( companion+'_M_companion_(M_jup)' )
        labels.append( '$M_\mathrm{'+companion+'}$ ($\mathrm{M_{jup}}$)' )
        
        names.append( companion+'_b_tra' )
        labels.append( '$b_\mathrm{tra;'+companion+'}$' )
        
        names.append( companion+'_b_occ'  )
        labels.append( '$b_\mathrm{occ;'+companion+'}$' )
        
        names.append( companion+'_T_tra_tot'  )
        labels.append( '$T_\mathrm{tot;'+companion+'}$ (h)' )
        
        names.append( companion+'_T_tra_full' )
        labels.append( '$T_\mathrm{full;'+companion+'}$ (h)' )
        
        names.append( companion+'_host_density' )
        labels.append( '$rho_\mathrm{\star;'+companion+'}$ (cgs)' )
    
        names.append( companion+'_surface_gravity')
        labels.append( '$g_\mathrm{\star;'+companion+'}$ (cgs)' )
        
        names.append( companion+'_Teq' )
        labels.append( '$T_\mathrm{eq;'+companion+'}$ (K)' )
        
        for inst in config.BASEMENT.settings['inst_phot']:
            
            names.append( companion+'_depth_tr_undiluted_'+inst )
            labels.append( '$\delta_\mathrm{tr; undil; '+companion+'; '+inst+'}$ (ppt)' )
            
            names.append( companion+'_depth_tr_diluted_'+inst )
            labels.append( '$\delta_\mathrm{tr; dil; '+companion+'; '+inst+'}$ (ppt)' )
        
            names.append( companion+'_depth_occ_undiluted_'+inst )
            labels.append( '$\delta_\mathrm{occ; undil; '+companion+'; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_depth_occ_diluted_'+inst )
            labels.append( '$\delta_\mathrm{occ; dil; '+companion+'; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_ellipsoidal_undiluted_'+inst )
            labels.append( '$A_\mathrm{ellipsoidal; undil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_ellipsoidal_diluted_'+inst )
            labels.append( '$A_\mathrm{ellipsoidal; dil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_sbratio_undiluted_'+inst )
            labels.append( '$A_\mathrm{sbratio; undil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_sbratio_diluted_'+inst )
            labels.append( '$A_\mathrm{sbratio; dil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_geom_albedo_undiluted_'+inst )
            labels.append( '$A_\mathrm{geom. albedo; undil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_geom_albedo_diluted_'+inst )
            labels.append( '$A_\mathrm{geom. albedo; dil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_gdc_undiluted_'+inst )
            labels.append( '$A_\mathrm{grav. dark.; undil; '+inst+'}$ (ppm)' )
            
            names.append( companion+'_ampl_gdc_diluted_'+inst )
            labels.append( '$A_\mathrm{grav. dark.; dil; '+inst+'}$ (ppm)' )
            
            
            
        #::: period ratios (for ressonance studies)
        if len(companions)>1:
            for other_companion in companions:
                if other_companion is not companion:
                    names.append( companion+'_period/'+other_companion+'_period' )
                    labels.append( '$P_\mathrm{'+companion+'} / P_\mathrm{'+other_companion+'}$' )
           
            
    #::: host
    for inst in config.BASEMENT.settings['inst_all']:    
        if config.BASEMENT.settings['host_ld_law_'+inst] is None:
            pass
            
        elif config.BASEMENT.settings['host_ld_law_'+inst] == 'lin':
            names.append( 'host_ldc_u1_'+inst )
            labels.append( 'Limb darkening $u_\mathrm{1; '+inst+'}$' )
            
        elif config.BASEMENT.settings['host_ld_law_'+inst] == 'quad':
            names.append( 'host_ldc_u1_'+inst )
            labels.append( 'Limb darkening $u_\mathrm{1; '+inst+'}$' )
            names.append( 'host_ldc_u2_'+inst )
            labels.append( 'Limb darkening $u_\mathrm{2; '+inst+'}$' )
            
        elif config.BASEMENT.settings['host_ld_law_'+inst] == 'sing':
            raise ValueError("Sorry, I have not yet implemented the Sing limb darkening law.")
            
        else:
            print(config.BASEMENT.settings['host_ld_law_'+inst] )
            raise ValueError("Currently only 'none', 'lin', 'quad' and 'sing' limb darkening are supported.")
                
        names.append( companion+'_ampl_gdc_diluted_'+inst )
        labels.append( '$A_\mathrm{grav. dark.; dil; '+inst+'}$ (ppm)' )
        
        
    names.append( 'mean_host_density' )
    labels.append( '$rho_\mathrm{\star; mean}$ (cgs)' )
        
            
    ###############################################################################
    #::: delete pointless values
    ###############################################################################
    ind_good = []
    for i,name in enumerate(names):
        if (name in derived_samples) and isinstance(derived_samples[name], np.ndarray) and not any(np.isnan(derived_samples[name])) and not all(np.array(derived_samples[name])==0):
            ind_good.append(i)
            
    names = [ names[i] for i in ind_good ]
    labels = [ labels[i] for i in ind_good ]
    
    
            
    ###############################################################################
    #::: save all in pickle
    ###############################################################################
    pickle.dump(derived_samples, open(os.path.join(config.BASEMENT.outdir,mode+'_derived_samples.pickle'),'wb'))
    
    
    
    ###############################################################################
    #::: save txt & latex table & latex commands
    ###############################################################################
    with open(os.path.join(config.BASEMENT.outdir,mode+'_derived_table.csv'),'w') as outfile,\
         open(os.path.join(config.BASEMENT.outdir,mode+'_derived_latex_table.txt'),'w') as f,\
         open(os.path.join(config.BASEMENT.outdir,mode+'_derived_latex_cmd.txt'),'w') as f_cmd:
             
        outfile.write('#property,value,lower_error,upper_error,source\n')
        
        f.write('Property & Value & Source \\\\ \n')
        f.write('\\hline \n')
        f.write('\\multicolumn{4}{c}{\\textit{Derived parameters}} \\\\ \n')
        f.write('\\hline \n')
        
        for name,label in zip(names, labels):
            ll, median, ul = np.percentile(derived_samples[name], [15.865, 50., 84.135])
            outfile.write( str(label)+','+str(median)+','+str(median-ll)+','+str(ul-median)+',derived\n' )
            
            value = latex_printer.round_tex(median, median-ll, ul-median)
            f.write( label + ' & $' + value + '$ & derived \\\\ \n' )
            
            simplename = name.replace("_", "").replace("/", "over").replace("(", "").replace(")", "").replace("1", "one").replace("2", "two")
            f_cmd.write('\\newcommand{\\'+simplename+'}{$'+value+'$} %'+label+' = '+value+'\n')
            
    logprint('\nSaved '+mode+'_derived_results.csv, '+mode+'_derived_latex_table.txt, and '+mode+'_derived_latex_cmd.txt')
    
        
        
    ###############################################################################
    #::: plot corner
    ###############################################################################
    if 'mean_host_density' in names: names.remove('mean_host_density') #has (N_companions x N_dims) dimensions, thus does not match the rest
    x = np.column_stack([ derived_samples[name] for name in names ])
    fig = corner(x,
                 range = [0.999]*len(names),
                 labels = names,
                 quantiles=[0.15865, 0.5, 0.84135],
                 show_titles=True, title_kwargs={"fontsize": 14})
    fig.savefig( os.path.join(config.BASEMENT.outdir,mode+'_derived_corner.png'), dpi=100, bbox_inches='tight' )
    plt.close(fig)
    
    logprint('\nSaved '+mode+'_derived_corner.jpg')
    