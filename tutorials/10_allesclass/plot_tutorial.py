#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:45:11 2019

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import allesfitter




#==============================================================================
#allesclass
#==============================================================================
#
#Tired of allesfitter's standard plots? Want to match your color scheme, or add some fancy twists? Create your own plots (and much more) with the allesclass module! Examples below, simply replace 'allesfit' with whatever name you gave your directory; also change the instrument and flux/rv accordingly.


#::: load allesclass
alles = allesfitter.allesclass('/Users/mx/Dropbox (MIT)/Science/Code/allesfitter/tutorials/02_transits/allesfit')

#::: iterate over all plot styles
for style in ['full', 'phase', 'phasezoom', 'phasezoom_occ', 'phase_variations']:
    
    #::: set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(6,6), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    #::: alles.plot(...) data and model
    alles.plot('Leonardo','b',style,ax=axes[0])
    axes[0].set_title(style)
    
    #::: alles.plot(...) residuals
    alles.plot('Leonardo','b',style+'_residuals',ax=axes[1])
    axes[1].set_title('')
    
    fig.savefig(style+'.pdf', bbox_inches='tight')



#------------------------------------------------------------------------------
#2) Full control
#------------------------------------------------------------------------------
#Want even more control, or access the data directly? Go ahead::



#------------------------------------------------------------------------------
#2.1) Full time series
#------------------------------------------------------------------------------

##::: settings
#inst = 'Leonardo'
#key = 'flux'
#
##::: load the time, flux, and flux_err
#time = alles.data[inst]['time']
#flux = alles.data[inst][key]
#flux_err = alles.data[inst]['err_scales_'+key] * alles.posterior_params_median['err_'+key+'_'+inst]
#
##::: note that the error for RV instruments is calculated differently
##rv_err = np.sqrt( alles.data[inst]['white_noise_'+key]**2 + alles.posterior_params_median['jitter_'+key+'_'+inst]**2 )
#
##::: set up the figure
#fig, axes = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
#fig.subplots_adjust(hspace=0)
#
##::: top panel: plot the data and 20 curves from random posterior samples (evaluated on a fine time grid)
#ax = axes[0]
#ax.errorbar(time, flux, flux_err, fmt='b.')
#for i in range(20):
#    time_fine = np.arange(time[0], time[-1], 0.05)
#    model_fine, baseline_fine, _ = alles.get_one_posterior_curve_set(inst, key, xx=time_fine)
#    ax.plot(time_fine, 1.+baseline_fine, 'g-', lw=2, zorder=11)
#    ax.plot(time_fine, model_fine+baseline_fine, 'r-', lw=2, zorder=12)
#
##::: bottom panel: plot the residuals; 
##::: for that, subtract the "posterior median model" and "posterior median baseline" from the data (evaluated on the time stamps of the data)
#ax = axes[1]
#baseline = alles.get_posterior_median_baseline(inst, key)
#model = alles.get_posterior_median_model(inst, key)
#ax.errorbar(time, flux-(model+baseline), flux_err, fmt='b.')
#ax.axhline(0, color='grey', linestyle='--')


