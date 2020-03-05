#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:57:46 2018

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

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import ellc
from pprint import pprint

#::: my modules
from allesfitter.generative_models import inject_lc_model

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: seed
np.random.seed(42)




###############################################################################
#::: simulate data
###############################################################################
planet = 'b'

inst = 'Leonardo'
time_Leonardo = np.arange(0,27,5./60./24.)[::3]
time_Leonardo = time_Leonardo[ (time_Leonardo<2) | (time_Leonardo>4) ] #add data gaps
time_Leonardo = time_Leonardo[ (time_Leonardo<13) | (time_Leonardo>14) ] #add data gaps

flux_Leonardo = np.ones_like(time_Leonardo)
flux_err_Leonardo = np.zeros_like(time_Leonardo)

ind1 = np.where((time_Leonardo<20) | (time_Leonardo>23))[0]
flux_Leonardo[ind1] += np.random.normal(0,2e-3,size=len(ind1)) #add white noise
flux_err_Leonardo[ind1] = 2e-3*np.ones(len(ind1)) #white nosie error bars

ind2 = np.where((time_Leonardo>=20) & (time_Leonardo<=23))[0]
flux_Leonardo[ind2] += np.random.normal(0,4e-3,size=len(ind2)) #add extra white noise
flux_err_Leonardo[ind2] = 4e-3*np.ones(len(ind2)) #white nosie error bars

flux_Leonardo += 3e-4*np.sin(time_Leonardo/2.7) #add red noise

flux_Leonardo = inject_lc_model(time_Leonardo, flux_Leonardo, flux_err_Leonardo, 
                               epoch = 1.1, period = 3.4, 
                               R_companion = 1., M_companion = 1.,
                               R_companion_unit = 'Rjup', M_companion_unit = 'Mjup',
                               R_host = 1., M_host = 1., 
                               sbratio = 0.,
                               incl = 89., 
                               ecc = 0.,
                               omega = 0.,
                               dil = 0.,
                               ldc = [0.6,0.2],
                               ld = 'quad',
                               show_plot = True, save_plot = True, fname_plot = 'Leonardo.pdf')

header = 'time,flux,flux_err'
X = np.column_stack(( time_Leonardo, flux_Leonardo, flux_err_Leonardo ))
np.savetxt('allesfit/Leonardo.csv', X, delimiter=',', header=header)



###############################################################################
#::: plot
###############################################################################
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.errorbar(time_Leonardo, flux_Leonardo, yerr=flux_err_Leonardo, fmt='b.', ecolor='silver', label='Leonardo')
ax.legend(loc=1)
ax.set(xlabel='BJD', ylabel='Flux')
plt.tight_layout()
fig.savefig('allesfit/data.png', bbox_inches='tight')