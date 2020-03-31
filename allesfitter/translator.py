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




def is_equal(a,b):
    return np.abs(a-b) < 1e-12



def translate(params=None, quiet=False, **params_kwargs):
    '''
    A lazy-input translator that calculates whatever is possible from whatever you give it.
    '''
        
    #==========================================================================
    #::: allow only to either give a params dict, or a series of kwargs. 
    #::: not both.
    #==========================================================================
    if params is None:
        params = params_kwargs
    elif len(params_kwargs)>0:
        raise ValueError('Give either a params dict, or a series of kwargs. Not both.')
    
    
    #==========================================================================
    #::: check R_companion units
    #==========================================================================
    try:
        if params['R_companion_unit']=='Rearth':
            R_companion_with_unit = params['R_companion']*u.Rearth
        elif params['R_companion_unit']=='Rjup':
            R_companion_with_unit = params['R_companion']*u.Rjup
        elif params['R_companion_unit']=='Rsun':
            R_companion_with_unit = params['R_companion']*u.Rsun
        else:
            R_companion_with_unit = params['R_companion']*u.Rearth
            params['R_companion_unit'] = 'Rearth'
            if not quiet: print('Assuming R_companion_unit = R_earth.')
    except:
        pass
        
        
    #==========================================================================
    #::: check M_companion units
    #==========================================================================
    try:
        if params['M_companion_unit']=='Mearth':
            M_companion_with_unit = params['M_companion']*u.Mearth
        elif params['M_companion_unit']=='Mjup':
            M_companion_with_unit = params['M_companion']*u.Mjup
        elif params['M_companion_unit']=='Msun':
            R_companion_with_unit = params['M_companion']*u.Msun
        else:
            M_companion_with_unit = params['M_companion']*u.Mearth
            params['M_companion_unit'] = 'Mearth'
            if not quiet: print('Assuming M_companion_unit = M_earth.')
    except:
        pass
        
        
    #==========================================================================
    #::: check for disagreeing inputs
    #==========================================================================
    if 'incl' in params and 'cosi' in params and not is_equal(params['incl'], np.rad2deg(np.arccos(params['cosi']))):
        raise ValueError('Both incl and cosi are given, but are not consistent.')
    
    if 'a' in params and 'period' in params and 'M_host' in params:
        if not is_equal( params['a'], ( (G/(4*np.pi**2) * (params['period']*u.d)**2 * (params['M_host']*u.Msun + M_companion_with_unit))**(1./3.) ).to(u.AU).value ):
            raise ValueError('All of a, period and M_host are given, but are not consistent.')
        
    #more TBD
            
        
    #==========================================================================
    #::: check the allowed keys
    #==========================================================================
    if not quiet: print('\nWarnings:')
    allowed_keys = ['R_companion', #bodies
                    'M_companion',
                    'R_companion_unit',
                    'M_companion_unit',
                    'R_host',
                    'M_host',
                    'sbratio',
                    'epoch', #orbital
                    'period',
                    'incl',
                    'cosi',
                    'a',
                    'ecc',
                    'omega',
                    'K', #observables and parametrizations
                    'f_c',
                    'f_s',
                    'ld',
                    'ldc', #in u-space (e.g. Claret 2017)
                    'ldc_transformed', #in Kipping q-space (eg Kipping 2013)
                    'R_host/a',
                    'a/R_host',
                    'R_companion/a',
                    'R_companion/R_host',
                    '(R_host+R_companion)/a',
                    'dil']
    for key in list(params):
        if key not in allowed_keys:
            if not quiet: print('Ignoring the keyword argument "'+key+'", because it was not recognized.')
            del params[key]
    for key in allowed_keys:
        if key not in list(params):
            params[key] = None
    
        
    #==========================================================================
    #::: display input
    #==========================================================================
    if not quiet: pprint(params)
    
    
    #==========================================================================
    #::: translate
    #==========================================================================
    if params['a'] is None:
        try: params['a'] = ( (G/(4*np.pi**2) * (params['period']*u.d)**2 * (params['M_host']*u.Msun + M_companion_with_unit))**(1./3.) ).to(u.AU).value #in AU    
        except: pass
    
    if params['incl'] is None:
        try: params['incl'] = np.rad2deg(np.arccos(params['cosi'])) #in deg
        except: pass
    
    if params['cosi'] is None:
        try: params['cosi'] = np.cos(np.deg2rad(params['incl']))
        except: pass
    
    if params['f_c'] is None:
        try: params['f_c'] = np.sqrt(params['ecc']) * np.cos(np.deg2rad(params['omega']))
        except: pass
        
    if params['f_s'] is None:
        try: params['f_s'] = np.sqrt(params['ecc']) * np.sin(np.deg2rad(params['omega']))
        except: pass
    
    if params['a/R_host'] is None:
        try: params['a/R_host'] = 1./params['R_host/a']
        except: pass
    
    if params['a/R_host'] is None:
        try: params['R_host/a'] = ((params['R_host']*u.Rsun) / (params['a']*u.AU)).decompose().value
        except: pass

    if params['R_host/a'] is None:
        try: params['R_host/a'] = 1./params['a/R_host']
        except: pass

    if params['R_companion/a'] is None:
        try: params['R_companion/a'] = (R_companion_with_unit / (params['a']*u.AU)).decompose().value
        except: pass

    if params['R_companion/a'] is None:
        try: params['R_companion/a'] = params['R_companion/R_host'] / params['a/R_host']
        except: pass
    
    if params['R_companion/R_host'] is None:
        try: params['R_companion/R_host'] = (R_companion_with_unit / (params['R_host']*u.Rsun)).decompose().value
        except: pass
    
    if params['(R_host+R_companion)/a'] is None:
        try: params['(R_host+R_companion)/a'] = ((params['R_host']*u.Rsun + R_companion_with_unit) / (params['a']*u.AU)).decompose().value
        except: pass
    
    if params['(R_host+R_companion)/a'] is None:
        try: params['(R_host+R_companion)/a'] = params['R_host/a'] + params['R_companion/a']
        except: pass
    
    if params['ldc'] is None:
        try:
            if params['ld'] == 'quad':
                q1 = (params['ldc'][0] + params['ldc'][1])**2
                q2 = 0.5 * params['ldc'][0] / (params['ldc'][0] + params['ldc'][1])
                params['ldc_transformed'] = [q1, q2]
        except:
            pass
        
    #more TBD
        
    
    #==========================================================================
    #::: return
    #==========================================================================
    if not quiet: print('\nResults:')
    if not quiet: pprint(params)
    return params
    
    

if __name__ == '__main__':
    translate(R_companion=1, M_companion=0., cosi=0.1, M_host=0.6, R_host=0.5, period=13.)
    translate({'R_companion/R_host':0.177, 'a/R_host':8.2, 'incl':86, 'omega':346, 'ecc':0.113, 'period':8.360613})
