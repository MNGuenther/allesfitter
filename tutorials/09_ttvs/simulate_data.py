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

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import ellc
from pprint import pprint


np.random.seed(42)



###############################################################################
#::: params
###############################################################################
params = {
          'b_radius_1':0.1,
          'b_radius_2':0.01,
          'b_sbratio':0.,
          'b_incl':89.,
          'b_epoch':1.1,
          'b_period':3.4,
          'b_K':0.1,
          'b_q':1,
          'c_radius_1':0.1,
          'c_radius_2':0.012,
          'c_sbratio':0.,
          'c_incl':89.,
          'c_epoch':0.3,
          'c_period':5.1,
          'c_K':0.1,
          'c_q':1,
          'ld_1_Leonardo':'quad',
          'ldc_1_Leonardo':[0.6,0.2],
         }

a_1 = 0.019771142 * params['b_K'] * params['b_period']
params['b_a'] = (1.+1./params['b_q'])*a_1

a_1 = 0.019771142 * params['c_K'] * params['c_period']
params['c_a'] = (1.+1./params['c_q'])*a_1



###############################################################################
#::: "truth" signals
###############################################################################
inst = 'Leonardo'
time_Leonardo = np.arange(0,27,2./60./24.)[::3]
flux_Leonardo = np.random.normal(1,2e-3,size=len(time_Leonardo)) #add white noise
flux_err_Leonardo = 2e-3*np.ones(len(time_Leonardo)) #white noise error bars

planet = 'b'
flux_Leonardo += ellc.lc(
                      t_obs =       time_Leonardo+0.02*np.sin(time_Leonardo/2.7),
                      radius_1 =    params[planet+'_radius_1'], 
                      radius_2 =    params[planet+'_radius_2'], 
                      sbratio =     params[planet+'_sbratio'],
                      incl =        params[planet+'_incl'],
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      ld_1 =        params['ld_1_'+inst],
                      ldc_1 =       params['ldc_1_'+inst]
                      ) - 1.

planet = 'c'
flux_Leonardo += ellc.lc(
                      t_obs =       time_Leonardo-0.01*np.sin(time_Leonardo/2.7),
                      radius_1 =    params[planet+'_radius_1'], 
                      radius_2 =    params[planet+'_radius_2'], 
                      sbratio =     params[planet+'_sbratio'],
                      incl =        params[planet+'_incl'],
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      ld_1 =        params['ld_1_'+inst],
                      ldc_1 =       params['ldc_1_'+inst]
                      ) - 1.

header = 'time,flux,flux_err'
X = np.column_stack(( time_Leonardo, flux_Leonardo, flux_err_Leonardo ))
np.savetxt('allesfit_linear_ephemerides/Leonardo.csv', X, delimiter=',', header=header)
np.savetxt('allesfit_with_ttvs/Leonardo.csv', X, delimiter=',', header=header)



###############################################################################
#::: plot
###############################################################################
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.errorbar(time_Leonardo, flux_Leonardo, yerr=flux_err_Leonardo, fmt='b.', ecolor='silver', label='Leonardo')
ax.legend(loc=1)
ax.set(xlabel='BJD', ylabel='Flux')
plt.tight_layout()
fig.savefig('allesfit_linear_ephemerides/data.png', bbox_inches='tight')
fig.savefig('allesfit_with_ttvs/data.png', bbox_inches='tight')