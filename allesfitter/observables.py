#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:48:58 2020

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
import os
import pathlib
import numpy as np
import pandas as pd
from astropy.constants import G, M_earth, M_jup, M_sun, R_earth, R_jup, R_sun, au
from astropy import units as u
from time import time as timer

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




"""
Readme:
Astropy units are nice and fancy, but can cause a 17x slow-down for parts of this code.
Use constants' values instead of units.
"""




###############################################################################
#::: Convert Period P (in days) to semi-major axis a (in AU)
###############################################################################
def P_to_a(P, Mp, Ms):
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
    #apply units (convert to kg m s)
    P = P*86400. #in s
    Ms = Ms*M_sun.value #in kg
    Mp = Mp*M_earth.value #in kg
    
    return np.cbrt(G.value/(4*np.pi**2) * P**2 * (Ms + Mp)) / au.value  #in AU 



###############################################################################
#::: Same as above, and actual not much slower with astropy units
###############################################################################
def P_to_a_astropy(P, Mp, Ms):
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
    return np.cbrt(G/(4*np.pi**2) * (P*u.d)**2 * (Ms*u.Msun + Mp*u.Mearth)).to(u.AU).value  #in AU 



###############################################################################
#::: Convert semi-amplitude K to a companion mass
###############################################################################
def calc_M_comp_from_RV(K, P, incl, ecc, M_host, 
                        return_unit='M_earth',
                        approx=False):
    """
    This code can use the full binary mass function:
    M_comp**3 * (sini)**3 / (M_host + M_comp)**2 
     = (P * K**3) / (2. * pi * G) * (1. - e**2)**(3./2.)
    (no assumption that Mp << Ms needed).

    Parameters
    ----------
    K : float or array
        Host's RV semi-major amplitude.
    P : float or array
        Companion's orbital period, in days.
    incl : float or array
        Inclination, in degrees.
    ecc : float or array
        Eccentricity.
    M_host : float or array
        Host's mass, in Msun.

    Returns
    -------
    M_comp : float or array
        Companion's mass, in units of "return_unit".
    """
    #apply units
    P = P*86400. #in s
    K = K*1e3 #from km/s to m/s
    M_host = M_host*M_sun.value #in kg
    
    #return_unit
    if return_unit=='M_earth': return_unit = M_earth.value
    if return_unit=='M_jup': return_unit = M_jup.value
    if return_unit=='M_sun': return_unit = M_sun.value
    
    #chose a rough but sensible conversion unit for least awkward scaling
    if K < 1: 
        conv_unit = M_earth.value #in kg. #if it's less than m/s, chose Earth mass
    elif K < 1e3:
        conv_unit = M_jup.value #in kg. #if it's less than km/s, chose Jupiter mass
    else:
        conv_unit = M_sun.value #in kg. #if it's more, chose Sun mass
    
    #use the full binary mass function
    if not approx:
        
        #We rearange the binary mass function as a 3rd order polynomial.
        #We start with these definitions of a and b:
        a = np.sin(np.deg2rad(incl))**3 #dimensionless
        b = ( P * K**3 / (2.*np.pi*G.value) * (1. - ecc**2)**(3./2.) ) #in kg 
        #We use Mjup for the least awkward scaling, 
        #as this equation can be used for Earth-like exoplanets and binaries alike
        
        #Rearanging the binary mass function then gives us:
        #a * M_comp**3 - b * M_comp**2 - 2*b*M_host * M_comp - b*M_host**2 = 0
            
        #To use M_comp as a scalar for numpy, 
        #we also divide the whole equation by u.Mjup**3.
        #The coefficients have to absorb that.
        #(We again use Mjup for the least awkward scaling.)
        #This gives us:
        #a * M_comp**3 - b/u.Mjup * M_comp**2 - 2*b*M_host/u.Mjup**2 * M_comp - b*M_host**2/u.Mjup**3 = 0
        #Now divide by a to give the highest order term some independence
        #M_comp**3 - b/a/u.Mjup * M_comp**2 - 2*b/a*M_host/u.Mjup**2 * M_comp - b/a*M_host**2/u.Mjup**3 = 0
        
        
        #Now we define p3,...,p0 to make it even simpler:
        #p3 * M_comp**3 + p2 * M_comp**2 + p1 * M_comp + p0 = 0,
        #with the following coefficients:
        
        # p3 = 1.
        p2 = -b/a / conv_unit
        p1 = -2*b/a*(M_host) / (conv_unit**2)
        p0 = -b/a*(M_host)**2 / (conv_unit**3)
        # print(p2, p1, p0)
        
        #::: numpy's poly.roots() is too slow for us
        # poly = Polynomial([p0,p1,p2,p3])
        # s = poly.roots() #get all three roots; at least one is real
        # s = np.real_if_close(s[np.isreal(s)][0]) #in Mjup but unitless
        # s = (s*conv_unit).to(return_unit) #in return_unit
        # print(s)
        
        #::: instead, use some good old-fashioned high school math
        #::: see, e.g., http://teacherlink.ed.usu.edu/tlnasa/reference/ImagineDVD/Files/imagine/YBA/cyg-X1-mass/binary-formula.html#return
        #::: and http://teacherlink.ed.usu.edu/tlnasa/reference/ImagineDVD/Files/imagine/YBA/cyg-X1-mass/cubic.html
        Q = (3*p1 - p2**2)/9.
        R = (9.*p2*p1 - 27.*p0 - 2.*p2**3)/54. 
        
        S = np.cbrt(R + np.sqrt(Q**3 + R**2))
        T = np.cbrt(R - np.sqrt(Q**3 + R**2))
    
        s = S + T - p2/3. 
        s = (s*conv_unit) / return_unit
        
        return s
    
        
    #use the exoplanet approximation (just for testing)
    else:
        return ( K / np.sin(incl) * np.sqrt(1 - ecc**2) \
                 * np.cbrt(P / (2*np.pi*G) * (M_host)**2) ) / return_unit 
               #TODO, something is not right here
            


###############################################################################
#::: Same as above, but 17x slower with astropy units
###############################################################################            
def calc_M_comp_from_RV_astropy(K, P, incl, ecc, M_host, 
                        return_unit=u.Mearth,
                        approx=False):
    """
    This code can use the full binary mass function:
    M_comp**3 * (sini)**3 / (M_host + M_comp)**2 
     = (P * K**3) / (2. * pi * G) * (1. - e**2)**(3./2.)
    (no assumption that Mp << Ms needed).

    Parameters
    ----------
    K : float or array
        Host's RV semi-major amplitude.
    P : float or array
        Companion's orbital period, in days.
    incl : float or array
        Inclination, in degrees.
    ecc : float or array
        Eccentricity.
    M_host : float or array
        Host's mass, in Msun.

    Returns
    -------
    M_comp : float or array
        Companion's mass, in units of "return_unit".
    """
    #apply units
    P = P*u.d
    incl = incl*u.deg
    K = K*u.km/u.s
    
    #chose a rough but sensible conversion unit for least awkward scaling
    if K.value < 1e-3: 
        conv_unit = u.Mearth #if it's less than m/s, chose Earth mass
    elif K.value < 1:
        conv_unit = u.Mjup #if it's less than k/s, chose Jupiter mass
    else:
        conv_unit = u.Msun #if it's more, chose Sun mass
    
    #use the full binary mass function
    if not approx:
        
        #We rearange the binary mass function as a 3rd order polynomial.
        #We start with these definitions of a and b:
        a = np.sin(incl)**3 #dimensionless
        b = ( P * K**3 / (2.*np.pi*G) * (1. - ecc**2)**(3./2.) ).to(conv_unit) #in Mjup 
        #We use Mjup for the least awkward scaling, 
        #as this equation can be used for Earth-like exoplanets and binaries alike
        
        #Rearanging the binary mass function then gives us:
        #a * M_comp**3 - b * M_comp**2 - 2*b*M_host * M_comp - b*M_host**2 = 0
            
        #To use M_comp as a scalar for numpy, 
        #we also divide the whole equation by u.Mjup**3.
        #The coefficients have to absorb that.
        #(We again use Mjup for the least awkward scaling.)
        #This gives us:
        #a * M_comp**3 - b/u.Mjup * M_comp**2 - 2*b*M_host/u.Mjup**2 * M_comp - b*M_host**2/u.Mjup**3 = 0
        #Now divide by a to give the highest order term some independence
        #M_comp**3 - b/a/u.Mjup * M_comp**2 - 2*b/a*M_host/u.Mjup**2 * M_comp - b/a*M_host**2/u.Mjup**3 = 0
        
        
        #Now we define p3,...,p0 to make it even simpler:
        #p3 * M_comp**3 + p2 * M_comp**2 + p1 * M_comp + p0 = 0,
        #with the following coefficients:
        
        # p3 = 1.
        p2 = -b/a / conv_unit
        p1 = -2*b/a*(M_host*u.Msun) / (conv_unit**2)
        p0 = -b/a*(M_host*u.M_sun)**2 / (conv_unit**3)
        # print(p3, p2, p1, p0)
        
        #now the whole equation and all coefficients are dimensionless
        # p3 = p3.decompose()
        # p2 = p2.decompose()
        # p1 = p1.decompose()
        # p0 = p0.decompose()
        # print(p3, p2, p1, p0)
        
        #now the whole equation and all coefficients are dimensionless
        # p3 = p3.decompose().value
        p2 = p2.decompose().value 
        p1 = p1.decompose().value 
        p0 = p0.decompose().value
        # print(p3, p2, p1, p0)
        
        #::: numpy's poly.roots() is too slow for us
        # poly = Polynomial([p0,p1,p2,p3])
        # s = poly.roots() #get all three roots; at least one is real
        # s = np.real_if_close(s[np.isreal(s)][0]) #in Mjup but unitless
        # s = (s*conv_unit).to(return_unit) #in return_unit
        # print(s)
        
        #::: instead, use some good old-fashioned high school math
        Q = (3*p1 - p2**2)/9.
        R = (9.*p2*p1 - 27.*p0 - 2.*p2**3)/54. 
        
        S = np.cbrt(R + np.sqrt(Q**3 + R**2))
        T = np.cbrt(R - np.sqrt(Q**3 + R**2))
    
        s = S + T - p2/3. 
        s = (s*conv_unit).to(return_unit)
        
        return s.value
    
        
    #use the exoplanet approximation (just for testing)
    else:
        return ( K / np.sin(incl) \
                 * np.sqrt(1 - ecc**2) \
                 * np.cbrt(P / (2*np.pi*G) * (M_host)**2) \
               ).decompose().to(return_unit).value



###############################################################################
#::: Calculate a body's density
###############################################################################
def calc_rho(R, M, 
            R_unit = u.Rsun,
            M_unit = u.Msun,
            return_unit='cgs'):
    """
    Assumes a spherical body.
    
    Parameters
    ----------
    R : array or float
        Body's radius.
    M : array or float
        Body's mass.
    R_unit : astropy unit, optional
        Radius unit. The default is u.Rsun.
    M_unit : array or float, optional
        Mass unit. The default is u.Msun.
    return_unit : str, optional
        Return unit. The default is 'cgs'.

    Returns
    -------
    None.
    """
    #apply units
    R *= R_unit
    M *= M_unit
    
    #calculate
    V = 4./3. * np.pi * R**3
    rho = M / V
    
    #return
    if return_unit == 'cgs':
        return rho.cgs.value
    else:
        return None #TODO
    
    
    
###############################################################################
#::: Derive the host's density from transit and RV observations
###############################################################################
def calc_rho_host(P, radius_1, rr, rho_comp,
                           return_unit='cgs'):
    """
    Assumes a spherical body.
    
    Parameters
    ----------
    P : array or float
        Period, in days.
    radius_1 : array or float
        R_host / a.
    rr : astropy unit, optional
        R_comp / R_host.
    rho_comp : array or float, optional
        Density of the companion, in cgs units.
    return_unit : str, optional
        Return unit. The default is 'cgs'.

    Returns
    -------
    None.
    """
    #apply units
    P *= u.d #in days
    rho_comp = rho_comp * u.g / (u.cm)**3 #in cgs
    
    #calculate
    rho_host = ( (3*np.pi) / (G*P**2) * (1./radius_1)**3 - rr**3 * rho_comp ).decompose()
    
    #return
    if return_unit == 'cgs':
        return rho_host.cgs.value
    else:
        return None #TODO



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
    
    a = P_to_a(P, M_companion, M_host) # in AU
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
    
    a = P_to_a(P, 1., M_host) # in AU
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
    
    
    
##########################################################################
#::: MAIN
##########################################################################
if __name__ == '__main__':
    
    ##########################################################################
    #::: speed test
    ##########################################################################
    N = 1000
    
    #::: 
    t0 = timer()
    for i in range(N):
        calc_M_comp_from_RV(K = 13e-3,P = 12*365, incl = 90 ,ecc = 0, M_host = 1, return_unit = u.Mjup, approx=False)
    t1 = timer()
    dt12 = (t1-t0)/N
    print('Runtime with floats:', dt12*1e3, 'ms')
    
    t3 = timer()
    for i in range(N):
        calc_M_comp_from_RV_astropy(K = 13e-3,P = 12*365, incl = 90 ,ecc = 0, M_host = 1, return_unit = u.Mjup, approx=False)
    t4 = timer()
    dt34 = (t4-t3)/N
    print('Runtime with astropy units:',  dt34*1e3, 'ms')

    print('Runntime speed up by using floats instead of astropy units:',  dt34/dt12)
    
    
    ##########################################################################
    #::: accuracy test
    ##########################################################################
    jup_exact = calc_M_comp_from_RV(K = 13e-3, 
                                    P = 12*365, 
                                    incl = 90, 
                                    ecc = 0, 
                                    M_host = 1, 
                                    return_unit = u.Mjup,
                                    approx=False)
    print('Jupiter Mass, exact:', jup_exact, 'Mjup')
    
    
    # jup_approx = calc_M_comp_from_RV(K = 13e-3, 
    #                                 P = 12*365, 
    #                                 incl = 90, 
    #                                 ecc = 0, 
    #                                 M_host = 1, 
    #                                 return_unit = u.Mjup,
    #                                 approx=True)
    # print('Jupiter Mass, approx:', jup_approx)
    
    
    ear_exact = calc_M_comp_from_RV(K = 9e-5, 
                                    P = 365, 
                                    incl = 90, 
                                    ecc = 0, 
                                    M_host = 1, 
                                    return_unit = u.Mearth,
                                    approx=False)
    print('Earth Mass, exact:', ear_exact, 'Mearth')
    
    
    # ear_approx = calc_M_comp_from_RV(K = 13e-3, 
    #                                 P = 12*365, 
    #                                 incl = 90, 
    #                                 ecc = 0, 
    #                                 M_host = 1, 
    #                                 return_unit = u.Mjup,
    #                                 approx=False)
    # print('Earth Mass, exact:', ear_approx)
    
    
    ##########################################################################
    #::: speed test
    ##########################################################################
    N = 10000
    
    #::: 
    t0 = timer()
    for i in range(N):
        P_to_a(1,1,1)
    t1 = timer()
    dt12 = (t1-t0)/N
    print('P_to_a: Runtime with floats:', dt12*1e3, 'ms')
    
    t3 = timer()
    for i in range(N):
        P_to_a(1,1,1)
    t4 = timer()
    dt34 = (t4-t3)/N
    print('P_to_a: Runtime with astropy units:',  dt34*1e3, 'ms')

    print('P_to_a: Runntime speed up by using floats instead of astropy units:',  dt34/dt12)
    