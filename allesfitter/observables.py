#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:48:58 2020

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
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint
from astropy.constants import G
from astropy import units as u

#::: my modules
import allesfitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




###############################################################################
#::: Convert Period P (in days) to semi-major axis a (in AU)
###############################################################################
def p_to_a(P, Mp, Ms):
    '''
    Parameters
    ----------
    P : float or array
        Planet orbital period, in days.
    Mp : float or array
        Planet mass, in Mearth.
    Ms : float or array
        Star mass, in Msun.

    Returns
    -------
    a : float or array
        Planet orbital semi-major axis, in AU.
    '''
    return ( (G/(4*np.pi**2) * (P*u.d)**2 * (Ms*u.Msun + Mp*u.Mearth))**(1./3.) ).to(u.AU).value  #in AU 



###############################################################################
#::: Convert AU units to Rsun units
###############################################################################
def au_to_rsun(x): 
    '''
    Parameters
    ----------
    x : float or array
        Distance meassurement in AU.

    Returns
    -------
    x : float or array
        Distance meassurement in Rsun.
    '''
    return (np.array(x)*u.AU).to(u.Rsun).value



###############################################################################
#::: Estimate Kempton TSM from an exoplanet archive data frame
###############################################################################
def estimate_tsm_exoarchive(data):
    '''
    Parameters
    ----------
    data : dictionary
        must contain the following fields: 
        pl_rade, st_rad, sy_jmag, st_teff, pl_orbsmax, pl_masse

    Returns
    -------
    See estimate_tsm()
    '''
    
    Rp = data['pl_rade']
    Mp = data['pl_masse']
    a = data['pl_orbsmax']
    Rstar = data['st_rad']
    Teff = data['st_teff']
    mJ = data['sy_jmag']
    
    return estimate_tsm(Rp, Mp, a, Rstar, Teff, mJ)
    
    
    
###############################################################################
#::: Estimate Kempton TSM from individual variables
###############################################################################
def estimate_tsm(Rp, Mp, a, Rstar, Teff, mJ):
    '''
    Parameters
    ----------
    Rp : float or array
        Planet radius, in Rearth.
    Mp : float or array
        Planet mass, in Mearth.
    a : float or array
        Planet orbital semi-major axis, in AU.
    Rstar : float or array
        Star radius, in Rsun.
    Teff : float or array
        Star effective temperature, in Kelvin.
    mJ : float or array
        Star J-band magnitude.

    Returns
    -------
    tsm : float or array
        Transmission Spectroscopy Metric (TSM; via Kempton+ 2018).
    jwst_threshold : float or array
        Threshold for the TSM (via Kempton+ 2018).
        If the TSM value is below this threshold, it is deemed not worth observing with JWST.
    jwst_recommendation : bool or array
        True if tsm >= jwst_threshold.
        False if tsm < jwst_threshold.
    '''
    
    #::: Cast to 1d arrays and copy
    Rp = np.atleast_1d(Rp)
    Mp = np.atleast_1d(Mp)
    a = np.atleast_1d(a)
    Rstar = np.atleast_1d(Rstar)
    Teff = np.atleast_1d(Teff)
    mJ = np.atleast_1d(mJ)
    
    #::: Estimate Planet Equilibirium Temperature following Kempton et al., 2018
    Teq_mid = estimate_teq(a, Rstar, Teff)[0]
    
    #::: Estimate Planet Mass, where missing, following Chen & Kipping (2017) as implemented by Louie et al. (2018)
    #::: (as suggested by Kempton et al., 2018)
    Mp_estimate = 1*Mp
    ind = np.isnan(Mp) & (Rp<=1.23)
    Mp_estimate[ind] = 0.9718 * Rp[ind]**3.58
    ind = np.isnan(Mp) & (Rp>1.23) & (Rp<14.26)
    Mp_estimate[ind] = 1.436 * Rp[ind]**1.70
    
    #::: Estimate scale factors following Kempton et al. (2018)
    tsm_scale_factor = np.nan * Rp
    tsm_scale_factor[Rp < 1.5] = 0.190
    tsm_scale_factor[(Rp>1.5) & (Rp<=2.75)] = 1.26
    tsm_scale_factor[(Rp>2.75) & (Rp<=4)] = 1.28
    tsm_scale_factor[(Rp>4) & (Rp<=10)] = 1.15

    tsm = tsm_scale_factor * Rp**3 * Teq_mid / Mp_estimate / Rstar**2 * 10**(-mJ/5.)
    
    return tsm, tsm_scale_factor, Teq_mid, Mp_estimate



###############################################################################
#::: Estimate planet equilibrium temperature
###############################################################################
def estimate_teq(a, Rstar, Teff):
    '''
    Parameters
    ----------
    a : float or array
        Planet orbital semi-major axis, in AU.
    Teff : float or array
        Star effective temperature, in Kelvin.
    Rstar : float or array
        Star radius, in Rsun.

    Returns
    -------
    Teq_mid : float or array
        Middle-value of the planet equilibrium temperature,
        estimated for zero albedo (A=0) and full day-night heat redistribution (E=1).
    Teq_low : float or array
        Low value of the planet equilibrium temperature,
        estimated for albedo A=0.3 and day-night heat redistribution E=1.
    Teq_high : float or array
        High value of the planet equilibrium temperature,
        estimated for albedo A=0 and day-night heat redistribution E=1,
        if the atmosphere instantaneously reradiates the absorbed radiation (with no advection), f = 2/3 (e.g., Seager 2010).
    '''
    
    #a) the middle Teq
    #Estimate Teq for zero albedo (A=0) and full day-night heat redistribution (E=1)
    #[suggested by Kempton et al. (2018)]
    Teq_mid = (Teff * np.sqrt(Rstar / au_to_rsun(a)) * ( 1/4. )**(0.25))
     
    #b) the minimum Teq
    #Estimate Teq for albedo A=0.3 and day-night heat redistribution E=1
    #like for TOI-270, like for Earth
    Teq_low = (Teff * np.sqrt(Rstar / au_to_rsun(a)) * ( (1/4.) * (1-0.3) )**(0.25))
     
    #b) the maximum Teq
    #Estimate Teq for albedo A=0 and day-night heat redistribution E=1
    #If the atmosphere instantaneously reradiates the absorbed radiation (with no advection), f = 2/3 (e.g., Seager 2010).
    Teq_high = (Teff * np.sqrt(Rstar / au_to_rsun(a)) * ( 2/3. )**(0.25))
    
    return Teq_mid, Teq_low, Teq_high



###############################################################################
#::: Estimate TTV super-period of 1st order mean-motion-resonance (MMR)
###############################################################################
def estimate_ttv_super_period_of_first_order_mmr(P1, P2, MMR='2:1'):
    '''
    Estimates the TTV super-period.
    Only works for first order MMRs, e.g., 2:1, 3:2, 4:3, etc.
    Following Eq. 7 of Lithwick+ 2017, https://iopscience.iop.org/article/10.1088/0004-637X/761/2/122/pdf
    
    Parameters
    ----------
    P1 : float
        Orbital period of the inner planet.
    P2 : float
        Orbital period of the outer planet.
    MMR : str, optional
        Mean motion resonance. 
        The larger number must come first.
        The default is '2:1'.

    Returns
    -------
    TTV super-period : float
        The TTV super-period.
    '''
    
    j = int(MMR.split(':')[0])
    return 1. / np.abs( (1.*j/P2) - (1.*(j-1.)/P1) )




###############################################################################
#::: Estimate tidal locking time scale
###############################################################################
#::: THIS ONE DIDN'T WORK
def estimate_tidal_locking_time_scale_Wikipedia(P, R_companion, M_companion, M_host, typ='rock'):
    if typ == 'rock': mu = 3e10
    elif typ == 'ice': mu = 4e9
    elif typ == 'gas': mu = 4e9
    
    a = p_to_a(P, M_companion, M_host) # in AU
    a = (a*u.AU).to(u.m).value # in m
    R_companion = (R_companion*u.Rearth).to(u.m).value # in m
    M_companion = (M_companion*u.Mearth).to(u.kg).value # in kg
    M_host = (M_host*u.Msun).to(u.kg).value # in kg
    
    return 6 * (a**6 * R_companion * mu) / (M_companion + M_host**2) * 1e10



def estimate_tidal_locking_time_scale_Kastings1993(P, M_host):
    '''
    This assumes an Earth-like planet (1 Rearth, 1 Mearth, rocky).

    Parameters
    ----------
    P : float or array
        Planet orbital period, in days.
    M_host : float or array
        Mass of the host, in Msun.

    Returns
    -------
    tlock : float
        tidal locking time scale, in years.
    '''
    
    a = p_to_a(P, 1., M_host) # in AU
    a = (a*u.AU).to(u.cm).value # in cm
    M_host = (M_host*u.Msun).to(u.g).value # in grams
    
    return (1./486. * ( a / (0.027 * M_host**(1./3.)) )**6 *u.s).to(u.yr).value # in years




###############################################################################
#::: Estimate spectral types
###############################################################################
def estimate_spectral_type(Teff):
    """
    Estimate the spectral type of a main-sequence dwarf star from the Teff only.
    No check for dwarfity, giantness, or any other stuff.
    Contains: 03-9.5, B0-9.5, A0-9.5, F0-9.5, G0-9.5, K0-9.5, M0-9.5, L0-9.5, T0-9.5, Y0-2
    
    Parameters
    ----------
    Teff : float or list of float
        Stellar effective temperature
        For example [3300, 4400, np.nan]
        
    Returns
    -------
    SpT : str or list of str
        For example: ['M3.5V', 'K5V', None] 
    """
    
    here = pathlib.Path(__file__).parent.absolute()
    f = os.path.join(here,'_static','_stars','Peacut_Mamajek.csv')
    df = pd.read_csv(f)[::-1].reset_index(drop=True) # sort by ascending Teff
    Teff = np.atleast_1d(Teff) # ensure it is iterable

    if len(Teff)==1 and np.isnan(Teff[0]):
        return None
    
    elif len(Teff)==1: 
        return df['#SpT'][ np.argmin( np.abs(df['Teff'] - Teff[0]) ) ].value
        
    else:
        Teff[ (Teff<=200) | (Teff>=50000) | (np.isnan(Teff)) ] = 1e6 # replace trash with a large number
        bins = [200] + list(df['Teff'][:-1] + np.diff(df['Teff'])/2.) + [50000] # set bins, length of df+1
        ind = np.digitize(Teff, bins) - 1 # -1 because digitize counts as bins[i-1] <= x < bins[i]
        SpT_all = np.append( df['#SpT'], None ) # length of df+1, last entry is for trash
        return np.array( SpT_all[ ind ] )
    
    
    
def estimate_spectral_class(Teff):
    """
    Estimates the spectral class. 
    Similar to estimate_spectral_type(Teff), but with larger bins.
    Contains: 0, B, A, F, G, K, early M, mid M, late M, L, T, Y
    early M : M0-M3.5
    mid M : M4-M6.5
    late M : M7-M9.5
    
    Parameters
    ----------
    Teff : float or list of float
        Stellar effective temperature
        For example [3300, 4400, np.nan]

    Returns
    -------
    SpC : str or list of str
        For example: ['early M', 'K', None] 
    """
    
    SpT = estimate_spectral_type(Teff)
    SpC = []
    for s in np.atleast_1d(SpT):
        if s is None:
            SpC.append(None)
        elif s[0]=='M':
            x = float(s[1:-1])
            if (x>=0) and (x<=3.5):
                SpC.append('early M')
            elif (x>=4) and (x<=6.5):
                SpC.append('mid M')
            elif (x>=7) and (x<=9.5):
                SpC.append('late M')
        else:
            SpC.append(s[0])
            
    if len(SpC) == 1:
        return SpC[0]
    else:
        return np.array( SpC )