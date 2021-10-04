#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:14:53 2020

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

#::: specific modules
try:
    from wotan import flatten
except ImportError:
    pass

#::: my modules
from .limb_darkening import LDC3
from .time_series import sigma_clip, slide_clip, mask_regions
from .plotting import tessplot




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



def translate_limb_darkening_from_u_to_q(u, law=None):
    '''
    Translate limb darkening values from the physical space (u1,u2,u3)
    into the Kipping-space (q1,q2,q3) for uniform sampling.

    Parameters
    ----------
    u : float or list of float
        Limb darkening values in the physical space.
        Either None, u, (u1,u2), or (u1,u2,u3).
    law : None or str, optional
        The limb darkening law.
        Either None, 'lin', 'quad', or 'sing'. The default is None.

    Returns
    -------
    float or list of float
        Limb darkening values in the Kipping-space.
        Either None, q, (q1,q2), or (q1,q2,q3)
    '''
    if law is None:
        return None
        
    elif law == 'lin':
        return u
    
    elif law == 'quad':
        q1 = (u[0] + u[1])**2
        q2 = 0.5*u[0] / (u[0] + u[1])
        return [ q1, q2 ]
        
    elif law == 'sing':
        return LDC3.inverse(u)

    

def translate_limb_darkening_from_q_to_u(q, law=None):
    '''
    Translate limb darkening values from the Kipping-space (q1,q2,q3) 
    into the (u1,u2,u3) space for physical interpretation.

    Parameters
    ----------
    q : float or list of float
        Limb darkening values in the Kipping-space.
        Either None, q, (q1,q2), or (q1,q2,q3)
    law : None or str, optional
        The limb darkening law.
        Either None, 'lin', 'quad', or 'sing'. The default is None.

    Returns
    -------
    float or list of float
        Limb darkening values in the physical space.
        Either None, u, (u1,u2), or (u1,u2,u3)
    '''
    if law is None:
        return None
        
    elif law == 'lin':
        return q
        
    elif law == 'quad':
        u1 = 2.*np.sqrt(q[0]) * q[1]
        u2 = np.sqrt(q[0]) * (1. - 2.*q[1])
        return [ u1, u2 ]
        
    elif law == 'sing':
        return LDC3.forward(q)



def tessclean(time, flux, plot=False, 
              method='biweight', window_length=1, bad_regions=None):
    """
    Clean a TESS light curve.

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    bad_regions : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    flux_flat : TYPE
        DESCRIPTION.
    """
    
    flux_clip = mask_regions(time, flux, bad_regions)
    flux_clip = sigma_clip(time, flux_clip, low=20, high=5) #fixed values to keep things simple
    flux_clip = slide_clip(time, flux_clip, window_length=window_length, low=20, high=3) #fixed values to keep things simple
    mask = np.isnan(flux_clip)
    
    flux_flat, trend = flatten(time, flux_clip, method=method, window_length=window_length, return_trend=True)
    
    if not plot:
        return flux_flat
    
    else:
        size = np.array([4**2 if m else 2**2 for m in mask]) #scatter uses sqrt of ms, so take ^2
        color = np.array(['r' if m else 'b' for m in mask])
        axes = tessplot(time, flux, trend=trend, size=size, color=color)
        for ax in np.atleast_1d(axes): ax.set_ylabel('Flux\n(original)')
        fig1 = plt.gcf()
        
        axes = tessplot(time, flux_clip, trend=trend)
        for ax in np.atleast_1d(axes): ax.set_ylabel('Flux\n(clipped)')
        fig2 = plt.gcf()
        
        axes = tessplot(time, flux_flat)
        fig3 = plt.gcf()
        for ax in np.atleast_1d(axes): ax.set_ylabel('Flux\n(clipped & detrended)')
        
        return flux_flat, fig1, fig2, fig3