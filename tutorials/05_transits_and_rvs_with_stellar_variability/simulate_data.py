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
import os
import ellc
from pprint import pprint


np.random.seed(42)




###############################################################################
#::: params
###############################################################################
workdir = 'allesfit'
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



###############################################################################
#::: special features
###############################################################################
stellar_rotation_amp = 0.1
stellar_rotation_period = 30.

def get_stellar_var(time):    
    red_noise = 2e-1*( 2e-4*(time-60)**2 - 2e-4*(time) + 1 )
    return stellar_rotation_amp * np.sin(2.*np.pi*time/stellar_rotation_period) + red_noise
    
def get_rv(time):
    return ellc.rv(
                  t_obs =       time, 
                  a =           params[planet+'_a'],
                  incl =        params[planet+'_incl'], 
                  t_zero =      params[planet+'_epoch'],
                  period =      params[planet+'_period'],
                  q =           params[planet+'_q'],
                  flux_weighted = False,
                  )[0]
    

###############################################################################
#::: "truth" signals
###############################################################################

#==============================================================================
#::: Leonardo
#==============================================================================
planet = 'b'
inst = 'Leonardo'
time_Leonardo = np.arange(0,16,5./60./24.)[::5]
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
flux_Leonardo += np.random.normal(0,2e-3,size=len(flux_Leonardo))
flux_Leonardo += 1e-3*np.exp(time_Leonardo/4.7)*np.sin(time_Leonardo/2.7)

flux_err_Leonardo = 2e-3*np.ones_like(flux_Leonardo)
header = 'time,flux,flux_err'
X = np.column_stack(( time_Leonardo, flux_Leonardo, flux_err_Leonardo ))
np.savetxt(os.path.join(workdir,'Leonardo.csv'), X, delimiter=',', header=header)



#==============================================================================
#::: Michelangelo
#==============================================================================
planet = 'b'
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
flux_Michelangelo += np.random.normal(0,5e-4,size=len(flux_Michelangelo))
flux_Michelangelo += 2e-3*np.sin(time_Michelangelo*8)

flux_err_Michelangelo = 5e-4*np.ones_like(flux_Michelangelo)
header = 'time,flux,flux_err'
X = np.column_stack(( time_Michelangelo, flux_Michelangelo, flux_err_Michelangelo ))
np.savetxt(os.path.join(workdir,'Michelangelo.csv'), X, delimiter=',', header=header)



#==============================================================================
#::: Donatello
#==============================================================================
planet = 'b'
inst = 'Donatello'
time_Donatello = np.sort(17. + np.random.rand(40)*70.)
rv_Donatello = ellc.rv(
                      t_obs =       time_Donatello, 
                      a =           params[planet+'_a'],
                      incl =        params[planet+'_incl'], 
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      q =           params[planet+'_q'],
                      flux_weighted = False,
                      )[0]
rv_Donatello += get_stellar_var(time_Donatello)
rv_Donatello += np.random.normal(0,1e-2,size=len(rv_Donatello))
rv_err_Donatello = 6e-3*np.ones_like(rv_Donatello)
header = 'time,flux,flux_err'
X = np.column_stack(( time_Donatello, rv_Donatello, rv_err_Donatello ))
np.savetxt(os.path.join(workdir,'Donatello.csv'), X, delimiter=',', header=header)



#==============================================================================
#::: Raphael
#==============================================================================
planet = 'b'
inst = 'Raphael'
time_Raphael = np.sort(63. + np.random.rand(20)*30.)
rv_Raphael = ellc.rv(
                      t_obs =       time_Raphael, 
                      a =           params[planet+'_a'],
                      incl =        params[planet+'_incl'], 
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      q =           params[planet+'_q'],
                      flux_weighted = False,
                      )[0]
rv_Raphael += get_stellar_var(time_Raphael)
rv_Raphael += np.random.normal(0,3e-3,size=len(rv_Raphael))
rv_Raphael += 10.7
rv_err_Raphael = 1e-3*np.ones_like(rv_Raphael)
header = 'time,flux,flux_err'
X = np.column_stack(( time_Raphael, rv_Raphael, rv_err_Raphael ))
np.savetxt(os.path.join(workdir,'Raphael.csv'), X, delimiter=',', header=header)




###############################################################################
#::: plot
###############################################################################
t = np.linspace(10,100,1000)

fig, axes = plt.subplots(2,2,figsize=(10,10))
axes[0,0].plot(time_Leonardo, flux_Leonardo, 'b.', label='Leonardo')
axes[0,0].legend()
axes[0,0].set(xlabel='BJD', ylabel='Flux')
axes[0,1].errorbar(time_Michelangelo, flux_Michelangelo, yerr=flux_err_Michelangelo, fmt='b.', label='Michelangelo')
axes[0,1].legend()
axes[0,1].set(xlabel='BJD', ylabel='Flux')
axes[1,0].errorbar(time_Donatello, rv_Donatello, yerr=rv_err_Donatello, fmt='bo', label='Donatello')
axes[1,0].plot(t, get_stellar_var(t), 'g-', label='Stellar var.')
#axes[1,0].plot(t, get_rv(t), color='orange', label='RV')
axes[1,0].plot(t, get_stellar_var(t)+get_rv(t), 'r-', label='SV + planet',lw=0.5)
axes[1,0].legend()
axes[1,0].set(xlabel='BJD', ylabel='RV (km/s)')
axes[1,1].errorbar(time_Raphael, rv_Raphael, yerr=rv_err_Raphael, fmt='bo', label='Raphael')
axes[1,1].plot(t, get_stellar_var(t)+10.7, 'g-', label='Stellar var.')
axes[1,1].plot(t, get_stellar_var(t)+get_rv(t)+10.7, 'r-', label='SV + planet',lw=0.5)
axes[1,1].legend()
axes[1,1].set(xlabel='BJD', ylabel='RV (km/s)')
plt.tight_layout()

fig.savefig(os.path.join(workdir,'data.pdf'), bbox_inches='tight')