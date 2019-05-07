#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:49:54 2018

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




def get_normalized_flux_from_normalized_mag( normalized_mag, normalized_mag_err=None ):
    '''
    Inputs:
    -------
    
    normalized_mag : float or array of float
        the normalized magnitude (i.e. centered around 0)
        
    normalized_mag_err : float or array of float (optional; default is None)
        the error on the normalized magnitude
        if not given, only the normalized_flux is returned
        if given, both the normalized_flux and the normalized_flux_err are returned
        
    
    Returns:
    --------
    
    normalized_flux : float or array of float
        the normalized_flux
        
    normalized_flux_err: float or array of float
        the error on the normalized_flux
    '''
    if normalized_mag_err is None:
        normalized_flux = 10.**(- normalized_mag/2.5 )
        return normalized_flux
    
    else:
        normalized_flux = 10.**(- normalized_mag/2.5 )
        conv = 2.5/np.log(10)
        normalized_flux_err = (normalized_mag_err/conv)*normalized_flux
        return normalized_flux, normalized_flux_err
