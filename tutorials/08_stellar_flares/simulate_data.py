#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:52:43 2018

@author:
Maximilian N. Guenther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from allesfitter.flares.aflare import aflare1




###############################################################################
#::: params
###############################################################################
N_flares = 2 
params = {}

params['flare_tpeak_1'] = 0.32
params['flare_fwhm_1'] = 57./60./24.
params['flare_ampl_1'] = 0.012

params['flare_tpeak_2'] = 0.73
params['flare_fwhm_2'] = 87./60./24.
params['flare_ampl_2'] = 0.007



###############################################################################
#::: "truth" signal
###############################################################################
time = np.arange(0, 1, 5./24./60.)

flux_flares = np.zeros_like(time)
for i in range(1,N_flares+1):
    flux_flares += aflare1(time, params['flare_tpeak_'+str(i)], params['flare_fwhm_'+str(i)], params['flare_ampl_'+str(i)], upsample=False, uptime=10)

flux = 1. + flux_flares + 1e-3*np.random.randn(len(time)) #add white noise
flux_err = 1e-3*np.ones_like(time) #record white noise errors

flux += 3e-3*np.sin(time) #add red noise

flux /= np.median(flux) #normalize



###############################################################################
#::: plots
###############################################################################
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(time, flux, 'b.', label='Flaronardo', rasterized=True)
ax.set(xlabel='BJD', ylabel='Flux')
ax.legend()
fig.savefig('allesfit/Flaronardo.pdf')



###############################################################################
#::: csv
###############################################################################
X = np.column_stack((time, flux, flux_err))
np.savetxt('allesfit/Flaronardo.csv', X, delimiter=',')