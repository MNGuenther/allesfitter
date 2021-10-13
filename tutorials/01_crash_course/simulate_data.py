#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:56:02 2021

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

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import ellc

#::: random seed
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
          'ldc_1_Leonardo':[0.6,0.2],
         }
a_1 = 0.019771142 * params['b_K'] * params['b_period']
params['b_a'] = (1.+1./params['b_q'])*a_1
pprint(params)



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
flux_Leonardo += np.random.normal(0,2e-3,size=len(flux_Leonardo))
flux_Leonardo += 3e-4*np.sin(time_Leonardo/2.7)
flux_err_Leonardo = 2e-3*np.ones_like(flux_Leonardo)
header = 'time,flux,flux_err'
X = np.column_stack(( time_Leonardo, flux_Leonardo, flux_err_Leonardo ))
np.savetxt('allesfit/Leonardo.csv', X, delimiter=',', header=header)



###############################################################################
#::: plot
###############################################################################
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(time_Leonardo, flux_Leonardo, 'b.', label='Leonardo')
ax.legend()
ax.set(xlabel='BJD', ylabel='Flux')
plt.tight_layout()
fig.savefig('allesfit/data.png', bbox_inches='tight')