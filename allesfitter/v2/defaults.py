#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:34:28 2020

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
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#::: my modules
# from .translator import translate




###############################################################################
#::: get the defaults
###############################################################################
def get_hot_jupiter_params():
    default_params = {}
    default_params['b_rr'] = 0.1
    default_params['b_rsuma'] = 0.1
    default_params['b_cosi'] = 0.
    default_params['b_epoch'] = 0.
    default_params['b_period'] = 1.
    default_params['b_K'] = 0.
    default_params['b_f_c'] = 0.
    default_params['b_f_s'] = 0.
    default_params['dil_Leonardo'] = 0.
    default_params['b_sbratio_Leonardo'] = 0.
    default_params['host_ldc_q1_Leonardo'] = 0.44502241000000003 #in q-space; quadratic limb darkening for a solar-type star
    default_params['host_ldc_q2_Leonardo'] = 0.3600659571278669 #in q-space; quadratic limb darkening for a solar-type star
    default_params['host_gdc_Leonardo'] = None
    default_params['b_gdc_Leonardo'] = None
    return default_params



def get_hot_jupiter_params_host():
    default_params_host = {'R_host':1,
                           'M_host':1,
                           'Teff_host':5700}
    return default_params_host
                      


def get_hot_jupiter_settings():
    default_settings = {'companions_phot':['b'],
                        'companions_rv':[],
                        'inst_phot':['Leonardo'],
                        'inst_rv':[],
                        'fit_ttvs':False,
                        'host_ld_law_Leonardo':'quad',
                        'b_ld_law_Leonardo':None}
    return default_settings



###############################################################################
#::: fill missing params with defaults
###############################################################################
def fill_params(params=None, settings=None):
    
    if params is None: 
        params = {}
    
    if settings is None:
        raise ValueError('Must input settings.')
    else:
        companions_all = list(np.unique(settings['companions_phot'] + settings['companions_rv']))
        inst_all = list(np.unique(settings['inst_phot'] + settings['inst_rv']))
    
    for companion in companions_all:
        for inst in inst_all:
    
            if 'dil_'+inst not in params:
                params['dil_'+inst] = 0.
            
            if companion+'_rr' not in params:
                params[companion+'_rr'] = None
                
            if companion+'_rsuma' not in params:
                params[companion+'_rsuma'] = None
                
            if companion+'_cosi' not in params:
                params[companion+'_cosi'] = 0.
                
            if companion+'_epoch' not in params:
                params[companion+'_epoch'] = None
                
            if companion+'_period' not in params:
                params[companion+'_period'] = None
                
            if companion+'_sbratio_'+inst not in params:
                params[companion+'_sbratio_'+inst] = 0.               
                
            if companion+'_a' not in params:
                params[companion+'_a'] = None
                
            if companion+'_q' not in params:
                params[companion+'_q'] = 1.
                
            if companion+'_K' not in params:
                params[companion+'_K'] = 0.
            
            if companion+'_f_c' not in params:
                params[companion+'_f_c'] = 0.
                
            if companion+'_f_s' not in params:
                params[companion+'_f_s'] = 0.
                
            if 'host_ldc_'+inst not in params:
                params['host_ldc_'+inst] = None
                
            if companion+'_ldc_'+inst not in params:
                params[companion+'_ldc_'+inst] = None
                
            if 'host_gdc_'+inst not in params:
                params['host_gdc_'+inst] = None
                
            if companion+'_gdc_'+inst not in params:
                params[companion+'_gdc_'+inst] = None
                
            if 'didt_'+inst not in params:
                params['didt_'+inst] = None
                
            if 'domdt_'+inst not in params:
                params['domdt_'+inst] = None
                
            if 'host_rotfac_'+inst not in params:
                params['host_rotfac_'+inst] = 1.
                
            if companion+'_rotfac_'+inst not in params:
                params[companion+'_rotfac_'+inst] = 1.
                
            if 'host_hf_'+inst not in params:
                params['host_hf_'+inst] = 1.5
                
            if companion+'_hf_'+inst not in params:
                params[companion+'_hf_'+inst] = 1.5
                
            if 'host_bfac_'+inst not in params:
                params['host_bfac_'+inst] = None
                
            if companion+'_bfac_'+inst not in params:
                params[companion+'_bfac_'+inst] = None
                
            if 'host_heat_'+inst not in params:
                params['host_heat_'+inst] = None
                
            if companion+'_heat_'+inst not in params:
                params[companion+'_heat_'+inst] = None
                
            if 'host_lambda_'+inst not in params:
                params['host_lambda_'+inst] = None
                
            if companion+'_lambda_'+inst not in params:
                params[companion+'_lambda_'+inst] = None
                
            if 'host_vsini' not in params:
                params['host_vsini'] = None
                
            if companion+'_vsini' not in params:
                params[companion+'_vsini'] = None
                
            if 'host_spots_'+inst not in params:
                params['host_spots_'+inst] = None
                
            if companion+'_spots_'+inst not in params:
                params[companion+'_spots_'+inst] = None
                
            if companion+'_phase_curve_beaming_'+inst not in params: #in ppt
                params[companion+'_phase_curve_beaming_'+inst] = None
                
            if companion+'_phase_curve_atmospheric_'+inst not in params: #in ppt
                params[companion+'_phase_curve_atmospheric_'+inst] = None
                
            if companion+'_phase_curve_ellipsoidal_'+inst not in params: #in ppt
                params[companion+'_phase_curve_ellipsoidal_'+inst] = None
            
    return params



def fill_params_host(params_host=None):
    
    if params_host is None: 
        params_host = {}

    if 'R_host' not in params_host:
        params_host['R_host'] = None
        
    if 'M_host' not in params_host:
        params_host['M_host'] = None
        
    if 'Teff_host' not in params_host:
        params_host['Teff_host'] = None
        
    return params_host
        


def fill_settings(settings=None):
    
    if settings is None: 
        settings = {}

    if 'companions_phot' not in settings:
        settings['companions_phot'] = []
        
    if 'companions_rv' not in settings:
        settings['companions_rv'] = []
        
    if 'inst_phot' not in settings:
        settings['inst_phot'] = []
        
    if 'inst_rv' not in settings:
        settings['inst_rv'] = []
        
    if 'phase_curve' not in settings:
        settings['phase_curve'] = False
        
    if 'phase_curve_style' not in settings:
        settings['phase_curve_style'] = None
        
    if 'fit_ttvs' not in settings:
        settings['fit_ttvs'] = False
        
    if 'exact_grav' not in settings:
        settings['exact_grav'] = False
        
    if 'N_flares' not in settings:
        settings['N_flares'] = 0.
        
    if 'N_spots' not in settings:
        settings['N_spots'] = 0.
        
    companions_all = list(np.unique(settings['companions_phot'] + settings['companions_rv']))
    inst_all = list(np.unique(settings['inst_phot'] + settings['inst_rv']))
    
    for companion in companions_all:
        for inst in inst_all:
            
            if 'host_ld_law_'+inst not in settings:
                settings['host_ld_law_'+inst] = None
                
            if companion+'_ld_law_'+inst not in settings:
                settings[companion+'_ld_law_'+inst] = None
            
            if 'host_grid_'+inst not in settings:
                settings['host_grid_'+inst] = 'default'
                
            if companion+'_grid_'+inst not in settings:
                settings[companion+'_grid_'+inst] = 'default'
            
            if 'host_shape_'+inst not in settings:
                settings['host_shape_'+inst] = 'sphere'
                
            if companion+'_shape_'+inst not in settings:
                settings[companion+'_shape_'+inst] = 'sphere'
        
    return settings



        
# def get_empty_params_host():
#     default_params_host = {'R_host':None,
#                            'M_host':None,
#                            'Teff_host':None}
#     return default_params_host
                      


# def get_empty_settings():
#     default_settings = {'companions_phot':[''],
#                         'companions_rv':[''],
#                         'inst_phot':[''],
#                         'inst_rv':[''],
#                         'fit_ttvs':False,
#                         'ld':'quad'}
#     return default_settings



###############################################################################
#::: fill gaps in an exisiting dictionary with defaults
###############################################################################
# def fill_params(params, **kwargs):
#     if params is None: params = {}
#     default_params = get_default_params()
#     for key in default_params.keys():
#         if key not in params: 
#             params[key] = default_params[key]
#     for key in kwargs:
#         if key in params:
#             params[key] = kwargs[key]
#     return params



# def fill_params_host(params_host, **kwargs):
#     if params_host is None: params_host = {}
#     default_params_host = get_default_params_host()
#     for key in default_params_host.keys():
#         if key not in params_host: 
#             params_host[key] = default_params_host[key]
#     for key in kwargs:
#         if key in params_host:
#             params_host[key] = kwargs[key]
#     return params_host



# def fill_settings(settings, **kwargs):
#     if settings is None: settings = {}
#     default_settings = get_default_settings()
#     for key in default_settings.keys():
#         if key not in settings: 
#             settings[key] = default_settings[key]
#     for key in kwargs:
#         if key in settings:
#             settings[key] = kwargs[key]
#     return settings


