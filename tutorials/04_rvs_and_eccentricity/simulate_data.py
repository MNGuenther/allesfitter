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
import os
import numpy as np
import matplotlib.pyplot as plt
import ellc
from pprint import pprint


np.random.seed(42)




###############################################################################
#::: params
###############################################################################
e = 0.2
w = 77 #deg
f_c = np.sqrt(e) * np.cos(np.deg2rad(w))
f_s = np.sqrt(e) * np.sin(np.deg2rad(w))
params = {
          'b_incl':90.,
          'b_epoch':1.1,
          'b_period':3.4,
          'b_K':0.1,
          'b_q':1,
          'b_f_c':f_c,
          'b_f_s':f_s
         }
a_1 = 0.019771142 * params['b_K'] * params['b_period'] * np.sqrt(1. - e**2)/np.sin(params['b_incl']*np.pi/180.)
params['b_a'] = (1.+1./params['b_q'])*a_1
pprint(params)

print('e', e)
print('w', w)

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
companion = 'b'


inst = 'Donatello'
time_Donatello = [37.1, 38, 38.7, 38.9, 41, 42, 53, 53.4, 54.1, 54.7, 55, 56, 58]
rv_Donatello = ellc.rv(
                      t_obs =       time_Donatello, 
                      a =           params[companion+'_a'],
                      incl =        params[companion+'_incl'], 
                      t_zero =      params[companion+'_epoch'],
                      period =      params[companion+'_period'],
                      q =           params[companion+'_q'],
                      f_c =         params[companion+'_f_c'],
                      f_s =         params[companion+'_f_s'],
                      flux_weighted = False,
                      )[0]
white_noise_known = 1e-2
jitter = 1e-2
print(inst, 'jitter=', jitter)
print(inst, 'ln(jitter)', np.log(jitter))
white_noise_total = np.sqrt(white_noise_known**2 + jitter**2)
rv_Donatello += np.random.normal(0,white_noise_total,size=len(rv_Donatello)) #white noise (total)
rv_err_Donatello = white_noise_known*np.ones_like(rv_Donatello) #white noise (known part; jitter is the unknown part added in quadrature)
header = 'time,flux,flux_err'
X = np.column_stack(( time_Donatello, rv_Donatello, rv_err_Donatello ))
if not os.path.exists('allesfit'): os.makedirs('allesfit')
np.savetxt('allesfit/Donatello.csv', X, delimiter=',', header=header)


inst = 'Raphael'
time_Raphael = [60, 60.5, 61, 61.5, 62, 62.5, 63]
#time_Raphael = np.linspace(0,5,1000)
rv_Raphael = ellc.rv(
                      t_obs =       time_Raphael, 
                      a =           params[companion+'_a'],
                      incl =        params[companion+'_incl'], 
                      t_zero =      params[companion+'_epoch'],
                      period =      params[companion+'_period'],
                      q =           params[companion+'_q'],
                      f_c =         params[companion+'_f_c'],
                      f_s =         params[companion+'_f_s'],
                      flux_weighted = False,
                      )[0]
white_noise_known = 4e-3
jitter = 2e-3
print(inst, 'jitter=', jitter)
print(inst, 'ln(jitter)', np.log(jitter))
white_noise_total = np.sqrt(white_noise_known**2 + jitter**2)
rv_Raphael += np.random.normal(0,white_noise_total,size=len(rv_Raphael)) #white noise
rv_err_Raphael = white_noise_known*np.ones_like(rv_Raphael) #white noise
header = 'time,flux,flux_err'
X = np.column_stack(( time_Raphael, rv_Raphael, rv_err_Raphael ))
np.savetxt('allesfit/Raphael.csv', X, delimiter=',', header=header)




###############################################################################
#::: plot
###############################################################################
fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].errorbar(time_Donatello, rv_Donatello, yerr=rv_err_Donatello, fmt='b.', label='Donatello')
axes[0].legend()
axes[0].set(xlabel='BJD', ylabel='RV (km/s)')
axes[1].errorbar(time_Raphael, rv_Raphael, yerr=rv_err_Raphael, fmt='b.', label='Raphael')
axes[1].legend()
axes[1].set(xlabel='BJD', ylabel='RV (km/s)')
plt.tight_layout()
fig.savefig('allesfit/data.png', bbox_inches='tight')