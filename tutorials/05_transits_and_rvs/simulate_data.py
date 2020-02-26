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
          'ld_1_Leonardo':'quad',
          'ldc_1_Leonardo':[0.3,0.1],
          'ld_1_Michelangelo':'quad',
          'ldc_1_Michelangelo':[0.5,0.4]
         }
a_1 = 0.019771142 * params['b_K'] * params['b_period']
params['b_a'] = (1.+1./params['b_q'])*a_1
pprint(params)

q1 = (0.3 + 0.1)**2
q2 = 0.5*0.3*(0.3 + 0.1)**(-1)
print('Leonardo q1 = '+str(q1))
print('Leonardo q1 = '+str(q2))

q1 = (0.5 + 0.4)**2
q2 = 0.5*0.5*(0.5 + 0.4)**(-1)
print('Michelangelo q1 = '+str(q1))
print('Michelangelo q1 = '+str(q2))



###############################################################################
#::: "truth" signals
###############################################################################
planet = 'b'

inst = 'Leonardo'
time_Leonardo = np.arange(0,10,5./60./24.)[::3]
time_Leonardo = time_Leonardo[ (time_Leonardo<2) | (time_Leonardo>4) ]
flux_Leonardo = ellc.lc(
                      t_obs =       time_Leonardo, 
                      radius_1 =    params[planet+'_radius_1'], 
                      radius_2 =    params[planet+'_radius_2'], 
                      sbratio =     params[planet+'_sbratio'],
                      incl =        params[planet+'_incl'],
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      ld_1 =        params['ld_1_'+inst],
                      ldc_1 =       params['ldc_1_'+inst]
                      )
flux_Leonardo += 3e-4*np.sin(time_Leonardo/2.7) #red noise
flux_Leonardo += np.random.normal(0,2e-3,size=len(flux_Leonardo)) #white noise
flux_err_Leonardo = 2e-3*np.ones_like(flux_Leonardo) #white noise
header = 'time,flux,flux_err'
X = np.column_stack(( time_Leonardo, flux_Leonardo, flux_err_Leonardo ))
np.savetxt('allesfit/Leonardo.csv', X, delimiter=',', header=header)



inst = 'Michelangelo'
time_Michelangelo = np.arange(52,52.25,2./60./24.)[::2]
flux_Michelangelo = ellc.lc(
                      t_obs =       time_Michelangelo, 
                      radius_1 =    params[planet+'_radius_1'], 
                      radius_2 =    params[planet+'_radius_2'], 
                      sbratio =     params[planet+'_sbratio'],
                      incl =        params[planet+'_incl'],
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      ld_1 =        params['ld_1_'+inst],
                      ldc_1 =       params['ldc_1_'+inst]
                      )
flux_Michelangelo += 2e-3*np.sin(time_Michelangelo*8) #red noise
flux_Michelangelo += np.random.normal(0,5e-4,size=len(flux_Michelangelo)) #white noise
flux_err_Michelangelo = 5e-4*np.ones_like(flux_Michelangelo) #white noise
header = 'time,flux,flux_err'
X = np.column_stack(( time_Michelangelo, flux_Michelangelo, flux_err_Michelangelo ))
np.savetxt('allesfit/Michelangelo.csv', X, delimiter=',', header=header)



inst = 'Donatello'
time_Donatello = [37.1, 38, 42, 55, 56, 58]
rv_Donatello = ellc.rv(
                      t_obs =       time_Donatello, 
                      a =           params[planet+'_a'],
                      incl =        params[planet+'_incl'], 
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      q =           params[planet+'_q'],
                      flux_weighted = False,
                      )[0]
rv_Donatello += np.random.normal(0,6e-3,size=len(rv_Donatello)) #white noise
rv_err_Donatello = 6e-3*np.ones_like(rv_Donatello) #white noise
header = 'time,flux,flux_err'
X = np.column_stack(( time_Donatello, rv_Donatello, rv_err_Donatello ))
np.savetxt('allesfit/Donatello.csv', X, delimiter=',', header=header)



inst = 'Raphael'
time_Raphael = [60, 60.5, 61, 61.5, 62, 62.5, 63]
#time_Raphael = np.linspace(0,5,1000)
rv_Raphael = ellc.rv(
                      t_obs =       time_Raphael, 
                      a =           params[planet+'_a'],
                      incl =        params[planet+'_incl'], 
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      q =           params[planet+'_q'],
                      flux_weighted = False,
                      )[0]
rv_Raphael += np.random.normal(0,1e-3,size=len(rv_Raphael)) #white noise
rv_err_Raphael = 1e-3*np.ones_like(rv_Raphael) #white noise
header = 'time,flux,flux_err'
X = np.column_stack(( time_Raphael, rv_Raphael, rv_err_Raphael ))
np.savetxt('allesfit/Raphael.csv', X, delimiter=',', header=header)




###############################################################################
#::: plot
###############################################################################
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes[0,0].plot(time_Leonardo, flux_Leonardo, 'b.', label='Leonardo')
axes[0,0].legend()
axes[0,0].set(xlabel='BJD', ylabel='Flux')
axes[0,1].errorbar(time_Michelangelo, flux_Michelangelo, yerr=flux_err_Michelangelo, fmt='b.', label='Michelangelo')
axes[0,1].legend()
axes[0,1].set(xlabel='BJD', ylabel='Flux')
axes[1,0].errorbar(time_Donatello, rv_Donatello, yerr=rv_err_Donatello, fmt='b.', label='Donatello')
axes[1,0].legend()
axes[1,0].set(xlabel='BJD', ylabel='RV (km/s)')
axes[1,1].errorbar(time_Raphael, rv_Raphael, yerr=rv_err_Raphael, fmt='b.', label='Raphael')
axes[1,1].legend()
axes[1,1].set(xlabel='BJD', ylabel='RV (km/s)')
plt.tight_layout()
fig.savefig('allesfit/data.png', bbox_inches='tight')