#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:02:18 2020

@author: 
Dr. Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
GitHub: https://github.com/MNGuenther
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
import os, sys
from astropy.constants import G
from astropy import units as u
from pprint import pprint

from . import defaults 




def is_equal(a,b):
    return np.abs(a-b) < 1e-12



def translate_alles_to_ellc(params, settings):
    params2 = params.copy()
    companions_all = list(np.unique(settings['companions_phot'] + settings['companions_rv']))
    inst_all = list(np.unique(settings['inst_phot'] + settings['inst_rv']))
    
    
    #=========================================================================
    #::: inclination, per companion
    #=========================================================================
    for companion in companions_all:
        try:
            params2[companion+'_incl'] = np.arccos( params2[companion+'_cosi'] )/np.pi*180.
        except:
            params2[companion+'_incl'] = None
        
       
    #=========================================================================
    #::: radii, per companion
    #::: R_1/a and R_2/a --> dependent on each companion's orbit
    #=========================================================================
    for companion in companions_all:
        try:
            params2[companion+'_radius_1'] = params2[companion+'_rsuma'] / (1. + params2[companion+'_rr'])
            params2[companion+'_radius_2'] = params2[companion+'_radius_1'] * params2[companion+'_rr']
        except:
            params2[companion+'_radius_1'] = None
            params2[companion+'_radius_2'] = None
            
            
    #=========================================================================
    #::: limb darkening, per instrument
    #=========================================================================
    for inst in inst_all:
        
        #---------------------------------------------------------------------
        #::: host
        #---------------------------------------------------------------------
        if settings['host_ld_law_'+inst] is None:
            params2['host_ldc_'+inst] = None
            
        elif settings['host_ld_law_'+inst] == 'lin':
            params2['host_ldc_'+inst] = params2['host_ldc_q1_'+inst]
            
        elif settings['host_ld_law_'+inst] == 'quad':
            ldc_u1 = 2.*np.sqrt(params2['host_ldc_q1_'+inst]) * params2['host_ldc_q2_'+inst]
            ldc_u2 = np.sqrt(params2['host_ldc_q1_'+inst]) * (1. - 2.*params2['host_ldc_q2_'+inst])
            params2['host_ldc_'+inst] = [ ldc_u1, ldc_u2 ]
            
        elif settings['host_ld_law_'+inst] == 'sing':
            raise ValueError("Sorry, I have not yet implemented the Sing limb darkening law.")
            
        else:
            print(settings['host_ld_law_'+inst] )
            raise ValueError("Currently only 'none', 'lin', 'quad' and 'sing' limb darkening are supported.")
     
    
        #---------------------------------------------------------------------
        #::: companion
        #---------------------------------------------------------------------
        for companion in companions_all:
            
            if settings[companion+'_ld_law_'+inst] is None:
                params2[companion+'_ldc_'+inst] = None
                
            elif settings[companion+'_ld_law_'+inst] == 'lin':
                params2[companion+'_ldc_'+inst] = params2[companion+'_ldc_q1_'+inst]
                
            elif settings[companion+'_ld_law_'+inst] == 'quad':
                ldc_u1 = 2.*np.sqrt(params2[companion+'_ldc_q1_'+inst]) * params2[companion+'_ldc_q2_'+inst]
                ldc_u2 = np.sqrt(params2[companion+'_ldc_q1_'+inst]) * (1. - 2.*params2[companion+'_ldc_q2_'+inst])
                params2[companion+'_ldc_'+inst] = [ ldc_u1, ldc_u2 ]
                
            elif settings[companion+'_ld_law_'+inst] == 'sing':
                raise ValueError("Sorry, I have not yet implemented the Sing limb darkening law.")
                
            else:
                print(settings[companion+'_ld_law_'+inst] )
                raise ValueError("Currently only 'none', 'lin', 'quad' and 'sing' limb darkening are supported.")
            
            
    return params2
    
    
    
def translate(params=None, settings=None, quiet=False, **params_kwargs):
    '''
    A lazy-input translator that calculates whatever is possible from whatever you give it.
        
    epoch : float
        epoch in days
    period : float
        period in days
    R_companion: float
        radius of the companion
        default is 1 Rearth
    M_companion: float
        mass of the companion
        default is 1 Mearth
    R_companion_unit: str
        radius unit of the companion
        default is 'Rearth'
    M_companion_unit: float
        mass unit of the companion
        default is 'Mearth'
    R_host : float
        radius of the star, in Rsun
        default is 1
    M_host: float
        mass of the star, in Msun
        default is 1
    incl : float
        inclination in degrees
    ecc : float
        eccentricity
    omega : float
        argument of periastron in degrees
    ldc : float or list
        limb darkening coefficients
        default is [0.4804, 0.1867]
    dil : float
        dilution, D_0 = 1 - (Ftarget / (Ftarget + Fblend))
        default is 0
    sbratio : float
        surface brightness ratio
        default is 0
        
    '''
    
    
    #==========================================================================
    #::: allow only to either give a params2 dict, or a series of kwargs - not both.
    #==========================================================================
    if params is None:
        params2 = params_kwargs
    elif len(params_kwargs)>0:
        raise ValueError('Give either a params2 dict, or a series of kwargs. Not both.')
    else:
        params2 = params.copy()
        
        
    #==========================================================================
    #::: if no settings are given, use standard settings
    #==========================================================================
    if settings is None:
        settings = defaults.fill_settings() #settings won't get altered, so no copy() needed
        
    
    #==========================================================================
    #::: check R_companion units
    #==========================================================================
    try:
        if 'r_companion_earth' in params2:
            R_companion_with_unit = params2['r_companion_earth']*u.Rearth
        elif 'r_companion_jup' in params2:
            R_companion_with_unit = params2['r_companion_jup']*u.Rjup
        elif 'r_companion_sun' in params2:
            R_companion_with_unit = params2['r_companion_sun']*u.Rsun
    except:
        pass
        
        
    #==========================================================================
    #::: check M_companion units
    #==========================================================================
    try:
        if 'm_companion_earth' in params2:
            M_companion_with_unit = params2['m_companion_earth']*u.Mearth
        elif 'm_companion_jup' in params2:
            M_companion_with_unit = params2['m_companion_jup']*u.Mjup
        elif 'm_companion_sun' in params2:
            M_companion_with_unit = params2['m_companion_sun']*u.Msun
    except:
        pass
    
        
    #==========================================================================
    #::: check for disagreeing inputs
    #==========================================================================
    if 'incl' in params2 and 'cosi' in params2 and not is_equal(params2['incl'], np.rad2deg(np.arccos(params2['cosi']))):
        raise ValueError('Both incl and cosi are given, but are not consistent.')
    
    if 'a' in params2 and 'period' in params2 and 'm_host' in params2:
        if not is_equal( params2['a'], ( (G/(4*np.pi**2) * (params2['period']*u.d)**2 * (params2['m_host']*u.Msun + M_companion_with_unit))**(1./3.) ).to(u.AU).value ):
            raise ValueError('All of a, period and M_host are given, but are not consistent.')
        
    #more TBD
            
        
    #==========================================================================
    #::: display input
    #==========================================================================
    if not quiet: print('\nInput:')
    if not quiet: pprint(params2)
    
    
    #==========================================================================
    #::: check the allowed keys
    #==========================================================================
    # if not quiet: print('\nWarnings:')
    allowed_keys = ['rr',               #allesfitter params2
                    'rsuma',
                    'cosi',
                    'epoch', 
                    'period',
                    'K',
                    'f_c',
                    'f_s',
                    'sbratio',
                    'dil',
                    'ldc',              #in u-space (e.g. Claret 2017)
                    'ldc_transformed',  #in Kipping q-space (eg Kipping 2013)
                    'r_host',           #allesfitter stellar params2
                    'm_host',
                    'incl',             #allesfitter derived params2
                    'a',
                    'ecc',
                    'omega',
                    'r_host_over_a', 
                    'a_over_R_host',
                    'r_companion_over_a',
                    'r_companion_earth',
                    'm_companion_earth',
                    'r_companion_jup',
                    'm_companion_jup',
                    'r_companion_sun',
                    'm_companion_sun']
    for key in list(params2):
        if key not in allowed_keys:
            if not quiet: print('Ignoring the keyword argument "'+key+'", because it was not recognized.')
            del params2[key]
    for key in allowed_keys:
        if key not in list(params2):
            params2[key] = None
            
    
    #==========================================================================
    #::: helper
    #==========================================================================
    def set_(key, value):
        if params2[key] is None:
            params2[key] = np.around( value, decimals=15 ) #round to at most 15 digits to avoid float issues
            
            
    #==========================================================================
    #::: translate
    #==========================================================================
    try: set_('r_companion_earth', params2['rr'] * (params2['r_host']*u.Rsun).to(u.Rearth).value)
    except: pass
    try: set_('r_companion_earth', (params2['r_companion_jup']*u.Rjup).to(u.Rearth).value)
    except: pass
    try: set_('r_companion_earth', (params2['r_companion_sun']*u.Rsun).to(u.Rearth).value)
    except: pass
    
    try: set_('r_companion_jup', params2['rr'] * (params2['r_host']*u.Rsun).to(u.Rjup).value)
    except: pass
    try: set_('r_companion_jup', (params2['r_companion_earth']*u.Rearth).to(u.Rjup).value)
    except: pass
    try: set_('r_companion_jup', (params2['r_companion_sun']*u.Rsun).to(u.Rjup).value)
    except: pass
    
    try: set_('r_companion_sun', params2['rr'] * (params2['r_host']*u.Rsun).to(u.Rsun).value)
    except: pass
    try: set_('r_companion_sun', (params2['r_companion_earth']*u.Rearth).to(u.Rsun).value)
    except: pass
    try: set_('r_companion_sun', (params2['r_companion_jup']*u.Rjup).to(u.Rsun).value)
    except: pass

    try: set_('m_companion_earth', (params2['m_companion_jup']*u.Mjup).to(u.Mearth).value)
    except: pass
    try: set_('m_companion_earth', (params2['m_companion_sun']*u.Msun).to(u.Mearth).value)
    except: pass
    
    try: set_('m_companion_jup', (params2['m_companion_earth']*u.Mearth).to(u.Mjup).value)
    except: pass
    try: set_('m_companion_jup', (params2['m_companion_sun']*u.Msun).to(u.Mjup).value)
    except: pass
    
    try: set_('m_companion_sun', (params2['m_companion_earth']*u.Mearth).to(u.Msun).value)
    except: pass
    try: set_('m_companion_sun', (params2['m_companion_jup']*u.Mjup).to(u.Msun).value)
    except: pass
        
    try: set_('a', ( (G/(4*np.pi**2) * (params2['period']*u.d)**2 * (params2['m_host']*u.Msun + M_companion_with_unit))**(1./3.) ).to(u.AU).value)  #in AU 
    except: pass

    try: set_('incl', np.rad2deg(np.arccos(params2['cosi']))) #in deg
    except: pass
    try: set_('cosi', np.cos(np.deg2rad(params2['incl'])))
    except: pass
    
    try: set_('ecc', params2['f_c']**2 + params2['f_s']**2) #in deg
    except: pass
    try: set_('omega', np.rad2deg(np.mod( np.arctan2(params2['f_s'], params2['f_c']), 2*np.pi))) #in deg from 0 to 360
    except: pass
    try: set_('f_c', np.sqrt(params2['ecc']) * np.cos(np.deg2rad(params2['omega'])))
    except: pass
    try: set_('f_s', np.sqrt(params2['ecc']) * np.sin(np.deg2rad(params2['omega'])))
    except: pass
    
    try: set_('r_host_over_a', ((params2['r_host']*u.Rsun) / (params2['a']*u.AU)).decompose().value)
    except: pass
    try: set_('r_host_over_a', params['rsuma'] / (1. + params['rr']))
    except: pass
    try: set_('r_host_over_a', 1./params2['a_over_R_host'])
    except: pass


    if params2['a_over_R_host'] is None:
        try: params2['a_over_R_host'] = ((params2['a']*u.AU) / (params2['r_host']*u.Rsun)).decompose().value
        except: pass

    if params2['a_over_R_host'] is None:
        try: params2['a_over_R_host'] = 1./params2['r_host_over_a']
        except: pass

    if params2['r_companion_over_a'] is None:
        try: params2['r_companion_over_a'] = (R_companion_with_unit / (params2['a']*u.AU)).decompose().value
        except: pass

    if params2['r_companion_over_a'] is None:
        try: params2['r_companion_over_a'] = params2['rr'] / params2['a_over_R_host']
        except: pass
    
    if params2['rr'] is None:
        try: params2['rr'] = (R_companion_with_unit / (params2['r_host']*u.Rsun)).decompose().value
        except: pass
    
    if params2['rsuma'] is None:
        try: params2['rsuma'] = ((params2['r_host']*u.Rsun + R_companion_with_unit) / (params2['a']*u.AU)).decompose().value
        except: pass
    
    if params2['rsuma'] is None:
        try: params2['rsuma'] = params2['r_host_over_a'] + params2['r_companion_over_a']
        except: pass
    
    if params2['ldc_transformed'] is None:
        try:
            if settings['ld'] == 'quad':
                q1 = (params2['ldc'][0] + params2['ldc'][1])**2
                q2 = 0.5 * params2['ldc'][0] / (params2['ldc'][0] + params2['ldc'][1])
                params2['ldc_transformed'] = [q1, q2]
        except:
            pass
        
    #more TBD
        
    
    #==========================================================================
    #::: return
    #==========================================================================
    if not quiet: print('\nResults:')
    if not quiet: pprint(params2)
    return params2
    
    

# if __name__ == '__main__':
#     translate(R_companion=1, M_companion=0., cosi=0.1, M_host=0.6, R_host=0.5, period=13.)
#     translate({'rr':0.177, 'a_over_R_host':8.2, 'incl':86, 'omega':346, 'ecc':0.113, 'period':8.360613})
