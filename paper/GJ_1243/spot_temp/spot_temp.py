#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:48:58 2020

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
from  astropy.modeling.blackbody import blackbody_lambda
from astropy import units as u




def calc_brightness_ratio_from_Teff(Teff_1, Teff_2, plot=False):

    bandpass = np.genfromtxt('tess-response-function-v1.0.csv', delimiter=',', names=['wavelength','transmission'])
    wavelength_grid = np.arange(500,2000,1)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(list(bandpass['wavelength'])+[2000], list(bandpass['transmission'])+[0], lw=2)
        ax.set(ylabel='TESS Transmission')
        ax2 = ax.twinx()
        ax2.plot(wavelength_grid, blackbody_lambda(wavelength_grid*u.nm, Teff_1*u.K), 'r-', lw=2, color='darkorange')
        ax2.plot(wavelength_grid, blackbody_lambda(wavelength_grid*u.nm, Teff_2*u.K), 'r-', lw=2, color='brown')
        ax2.set(ylabel='Blackbody Flux\n'+r'($erg \, cm^{-2} \, s^{-1} \, A^{-1} \, sr^{-1}$)')
    
    int1 = np.trapz(bandpass['transmission']*u.nm*blackbody_lambda(bandpass['wavelength']*u.nm, Teff_1*u.K), x=bandpass['wavelength']*u.nm, dx=np.diff(bandpass['wavelength']*u.nm))
    int2 = np.trapz(bandpass['transmission']*u.nm*blackbody_lambda(bandpass['wavelength']*u.nm, Teff_2*u.K), x=bandpass['wavelength']*u.nm, dx=np.diff(bandpass['wavelength']*u.nm))
    sbratio = int2/int1
    
    return sbratio



'''
TIC v8
------
GJ 1243 Teff = 3261 +- 157

allesfitter
-----------
spot 1 sbratio: 0.561 +0.060 −0.077
spot 2 sbratio: 0.22 +0.14 −0.13
'''

Teff_2 = np.arange(2200,3200,1)
sbratio = np.array([calc_brightness_ratio_from_Teff(3261, t) for t in Teff_2])

fig, ax = plt.subplots()
ax.plot(Teff_2, sbratio)
ax.axhspan(0.561-0.077, 0.561+0.060, xmin=0, xmax=1, alpha=0.3, color='brown')
ax.axhspan(0.22-0.13, 0.22+0.14, xmin=0, xmax=1, alpha=0.3, color='brown')
ax.set(xlabel=r'$T_\mathrm{eff;2}$', ylabel='Surface brighntess ratio in TESS band')

ind1 = np.where( ((0.561-0.077)<=sbratio) & (sbratio<=(0.561+0.060)) )[0]
print( 'Teff spot 1', np.median(Teff_2[ind1]), '-', np.median(Teff_2[ind1])-np.percentile(Teff_2[ind1],16), '+', np.percentile(Teff_2[ind1],84)-np.median(Teff_2[ind1]) )

ind2 = np.where( ((0.22-0.13)<=sbratio) & (sbratio<=(0.22+0.14)) )[0]
print( 'Teff spot 2', np.median(Teff_2[ind2]), '-', np.median(Teff_2[ind2])-np.percentile(Teff_2[ind2],16), '+', np.percentile(Teff_2[ind2],84)-np.median(Teff_2[ind2]) )