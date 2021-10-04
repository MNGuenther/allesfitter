#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:49:54 2018

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
