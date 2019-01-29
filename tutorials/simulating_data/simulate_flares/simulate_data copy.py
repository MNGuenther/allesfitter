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
import ellc
from allesfitter.appaloosa.aflare import aflare1




###############################################################################
#::: params
###############################################################################
planet = 'b'
params = {'b_radius_1':0.09,
          'b_radius_2':0.01,
          'b_incl':87.,
          'b_epoch':0.1,
          'b_period':1.337
        }

N_flares = 3

params['flare_tpeak_1'] = 0.3
params['flare_fwhm_1'] = 0.12
params['flare_ampl_1'] = 0.012

params['flare_tpeak_2'] = 0.5
params['flare_fwhm_2'] = 0.05
params['flare_ampl_2'] = 0.004

params['flare_tpeak_3'] = 1.4
params['flare_fwhm_3'] = 0.2
params['flare_ampl_3'] = 0.007



###############################################################################
#::: "truth" signal
###############################################################################
time = np.linspace(0, 3, 1000)
flux_ellc = ellc.lc(
                      t_obs =       time, 
                      radius_1 =    params[planet+'_radius_1'], 
                      radius_2 =    params[planet+'_radius_2'], 
                      sbratio = 0,
                      incl =        params[planet+'_incl'], 
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period']
                      )


flux_flares = np.zeros_like(time)
for i in range(1,N_flares+1):
    flux_flares += aflare1(time, params['flare_tpeak_'+str(i)], params['flare_fwhm_'+str(i)], params['flare_ampl_'+str(i)], upsample=False, uptime=10)


flux = flux_ellc + flux_flares
flux += 1e-3*np.random.randn(len(time))

flux_flares += 1e-3*np.random.randn(len(time))

flux_err = 1e-3*np.ones_like(time)



###############################################################################
#::: plots
###############################################################################
fig, axes = plt.subplots(2,1,figsize=(10,8))
axes[0].plot(time, flux, 'b.', label='FLARONARDO_transit')
axes[0].legend()
axes[1].plot(time, flux_flares, 'b.', label='FLARONARDO')
axes[1].legend()
fig.savefig('FLARONARDO.jpg')



###############################################################################
#::: csv
###############################################################################
X = np.column_stack((time, flux, flux_err))
np.savetxt('FLARONARDO_transit.csv', X, delimiter=',')

X = np.column_stack((time, flux_flares, flux_err))
np.savetxt('FLARONARDO.csv', X, delimiter=',')