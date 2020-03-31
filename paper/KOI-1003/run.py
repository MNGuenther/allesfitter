#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:13:40 2018

@author:
Dr. Maximilian N. Guenther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import
 
import numpy as np
import matplotlib.pyplot as plt
import allesfitter
from allesfitter.exoworlds_rdx.lightcurves.index_transits import index_eclipses
from allesfitter.detection.transit_search import tls_search

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




###############################################################################
#::: 1) mask out the transits
###############################################################################
# time, flux, flux_err = np.genfromtxt('Kepler_Q2.csv', delimiter=',', unpack=True) #load the full lightcurve
# inc_ecl1, ind_ecl2, ind_out = index_eclipses(time, 2455005.27, 8.36, 12./24., 12./24) #place masks
# plt.figure()
# plt.plot(time, flux, 'k.', color='orange')
# plt.plot(time[ind_out], flux[ind_out], 'b.')
# X = np.column_stack((time[ind_out],flux[ind_out],flux_err[ind_out]))
# np.savetxt('allesfit_trend/Kepler.csv', X, delimiter=',') #save the masked lightcurve



###############################################################################
#::: 2) fit the trend to get a feeling for the SHO
###############################################################################
# allesfitter.show_initial_guess('allesfit_trend')
# allesfitter.mcmc_fit('allesfit_trend')
# allesfitter.mcmc_output('allesfit_trend')



###############################################################################
#::: 3) use allesclass to remove the trend, and then search for eclipses
###############################################################################
# time, flux, flux_err = np.genfromtxt('Kepler_Q2.csv', delimiter=',', unpack=True) #load the full lightcurve again
# alles = allesfitter.allesclass('allesfit_trend') #load the GP SHO trend
# trend = 1 + alles.get_posterior_median_baseline('Kepler', 'flux', xx=time) #load the GP SHO trend
# flux_detrended = flux - trend + 1

# ind_good = np.where(flux_detrended<1.005) #crude cut to remove flares etc.
# time = time[ind_good]
# flux = flux[ind_good]
# flux_err = flux_err[ind_good]
# flux_detrended = flux_detrended[ind_good]
# trend = trend[ind_good]

# fig, axes = plt.subplots(2,1,figsize=(8,8), sharex=True)
# axes[0].plot(time, flux, 'b.')
# axes[0].plot(time, trend, color='orange', lw=2)
# axes[1].plot(time, flux_detrended, 'b.')
# fig.savefig('allesfit_trend/results/plot.pdf')

# tls_search(time, flux_detrended, flux_err, SNR_threshold=5, save_plot=True, outdir='tls_search')

# X = np.column_stack((time,flux,flux_err))
# np.savetxt('allesfit/Kepler.csv', X, delimiter=',') #save the lightcurve (without flares)



###############################################################################
#::: 4) translate Roettenbacher et al., 2016, into allesfitter params for initial guess
###############################################################################
# translate({'R_companion/R_host':0.177, 'a/R_host':8.2, 'incl':86, 'omega':346, 'ecc':0.113, 'period':8.360613})



###############################################################################
#::: 5) fit everything
###############################################################################
# allesfitter.show_initial_guess('allesfit')
# allesfitter.mcmc_fit('allesfit')
allesfitter.mcmc_output('allesfit')