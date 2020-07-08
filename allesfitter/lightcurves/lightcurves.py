#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:14:53 2020

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

#::: modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint

#::: my modules
import allesfitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})





def mask_ranges(x, x_min, x_max):
    """"
    Crop out values and indices out of an array x for multiple given ranges x_min to x_max.
    
    Input:
    x: array, 
    x_min: lower limits of the ranges
    x_max: upper limits of the ranges
    
    Output:
    
    
    Example:
    x = np.arange(200)    
    x_min = [5, 25, 90]
    x_max = [10, 35, 110]
    """

    mask = np.zeros(len(x), dtype=bool)
    for i in range(len(x_min)): 
        mask = mask | ((x >= x_min[i]) & (x <= x_max[i]))
    ind_mask = np.arange(len(mask))[mask]
    
    return x[mask], ind_mask, mask 



def get_first_epoch(time, epoch, period, width=0):    
    '''
    width : float
        set >0 to include transit egress to mark the first transit
    place the first_epoch at the start of the data to avoid luser mistakes
    '''
    time = np.sort(np.atleast_1d(time))
    start = np.nanmin( time )
    first_epoch = 1.*epoch + width/2. #add width/2 to catch egress
    if start<=first_epoch: first_epoch -= np.floor((first_epoch-start)/period) * period
    else: first_epoch += np.ceil((start-first_epoch)/period) * period
    return first_epoch - width/2.  #subtract width/2 to get midpoint again
    
    

def get_epoch_occ(epoch, period, f_s, f_c):
    ecc = f_s**2 + f_c**2
    ecosw = f_c * np.sqrt(ecc)
    epoch_occ = epoch + period/2. * (1. + 4./np.pi * ecosw)
    return epoch_occ



def get_Rhost_over_a(rr, rsuma):
    Rhost_over_a = rsuma / (1. + rr)
    return Rhost_over_a
    


def get_ecc_esinw_ecosw(f_s, f_c):
    ecc = f_s**2 + f_c**2
    esinw = f_s * np.sqrt(ecc)
    ecosw = f_c * np.sqrt(ecc)
    return ecc, esinw, ecosw
    
    

def impact_parameters_smart(rr, rsuma, cosi, f_s, f_c):
    #--------------------------------------------------------------------------
    #::: inputs
    #--------------------------------------------------------------------------
    Rhost_over_a = get_Rhost_over_a(rr, rsuma)
    ecc, esinw, ecosw = get_ecc_esinw_ecosw(f_s, f_c)
    
    
    #--------------------------------------------------------------------------
    #::: impact parameters
    #--------------------------------------------------------------------------
    eccentricity_correction_b_1 = ( (1. - ecc**2) / ( 1. + esinw ) )
    b_1 = (1./Rhost_over_a) * cosi * eccentricity_correction_b_1
    
    eccentricity_correction_b_2 = ( (1. - ecc**2) / ( 1. - esinw ) )
    b_2 = (1./Rhost_over_a) * cosi * eccentricity_correction_b_2
    
    
    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return b_1, b_2

    

def eclipse_width_smart(period, rr, rsuma, cosi, f_s, f_c):
    #--------------------------------------------------------------------------
    #::: inputs
    #--------------------------------------------------------------------------
    Rhost_over_a = get_Rhost_over_a(rr, rsuma)
    ecc, esinw, ecosw = get_ecc_esinw_ecosw(f_s, f_c)
    
    
    #--------------------------------------------------------------------------
    #::: impact parameters 
    #--------------------------------------------------------------------------
    b_1, b_2 = impact_parameters_smart(rr, rsuma, cosi, f_s, f_c)
    
    
    #--------------------------------------------------------------------------
    #::: widths
    #--------------------------------------------------------------------------
    eccentricity_correction_width_1 = ( np.sqrt(1. - ecc**2) / ( 1. + esinw ) )
    width_1 = period/np.pi  \
            * np.arcsin( Rhost_over_a \
                         * np.sqrt( (1. + rr)**2 - b_1**2  )\
                         / np.sin(np.arccos(cosi)) ) \
            * eccentricity_correction_width_1
    
    width_2 = width_1 * (1. + esinw) / (1. - esinw)
    
    
    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return width_1, width_2
    
    

def index_eclipses_smart(time, epoch, period, rr, rsuma, cosi, f_s, f_c, extra_factor=1.):
    '''
    
    Parameters
    ----------
    time : array
        must be sorted
    epoch : float
        must be first_epoch in the data set
    period : float
        DESCRIPTION.
    rr : float
        DESCRIPTION.
    rsuma : float
        DESCRIPTION.
    cosi : float
        DESCRIPTION.
    f_s : float
        DESCRIPTION.
    f_c : float
        DESCRIPTION.
    extra_factor : float, optional
        DESCRIPTION. The default is 1..

    Returns
    -------
    ind_ecl1 : array
        DESCRIPTION.
    ind_ecl2 : array
        DESCRIPTION.
    ind_out : array
        DESCRIPTION.

    '''
    #--------------------------------------------------------------------------
    #::: inputs
    #--------------------------------------------------------------------------
    ecc, esinw, ecosw = get_ecc_esinw_ecosw(f_s, f_c)
    
    
    #--------------------------------------------------------------------------
    #::: widths
    #--------------------------------------------------------------------------
    width_1, width_2 = eclipse_width_smart(period, rr, rsuma, cosi, f_s, f_c)
    
    
    #--------------------------------------------------------------------------
    #::: timing
    #--------------------------------------------------------------------------
    # time = np.sort(time)
    # epoch = get_first_epoch(time, epoch, period, width=2*width_1)
    epoch_occ = epoch + period/2. * (1. + 4./np.pi * ecosw)
    
    N = int( 1. * ( time[-1] - epoch ) / period ) + 1
        
    tmid_ecl1 = np.array( [ epoch + i * period for i in range(N) ] )
    tmid_ecl2 = np.array( [ epoch_occ + (i-1) * period for i in range(N+1) ] ) #(o-1) in case an occ happens before the first epoch
    
    _, ind_ecl1, mask_ecl1 = mask_ranges( time, tmid_ecl1 - width_1/2.*extra_factor, tmid_ecl1 + width_1/2.*extra_factor )           
    _, ind_ecl2, mask_ecl2 = mask_ranges( time, tmid_ecl2 - width_2/2.*extra_factor, tmid_ecl2 + width_2/2.*extra_factor )
        
    ind_out = np.arange( len(time) )[ ~(mask_ecl1 | mask_ecl2) ]
    
    
    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return ind_ecl1, ind_ecl2, ind_out
