#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:09:56 2019

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

#::: modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import ellc

#::: my modules
import allesfitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})
sns.set_palette('colorblind')




###############################################################################
#::: plot a lightcurve from the posteriors
###############################################################################
companion = 'b'
inst = 'TESS'
key = 'flux'
datadir = 'allesfit_orbit'

alles = allesfitter.allesclass(datadir)
time = alles.data['TESS']['time']
flux = alles.data['TESS']['flux']
model_time = 1.*time

baseline = alles.get_posterior_median_baseline(inst, key, xx=model_time)
model_flux = alles.get_posterior_median_model(inst, key, xx=model_time)

model_fluxes = []
for i in range(20):
    model_fluxes.append( alles.get_one_posterior_model(inst, key, xx=model_time) )



#==============================================================================
#::: set up figure
#==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12,4.5), gridspec_kw={'height_ratios': [3,1], 'width_ratios':[1,2], 'hspace':0}, sharex='col')
zoom = [1e3,1e6]
for ax, z in zip(axes[0,:],zoom):
    ax.plot(time, (flux-baseline-1)*z, 'k.', color='silver', rasterized=True, zorder=1)
for ax in axes[1,:]:
    ax.plot(time, (flux-baseline-model_flux)*1e6, 'k.', color='silver', rasterized=True, zorder=1)
    ax.set(xlabel=r'$\Delta$'+'Time ('+r'$\mathrm{BJD_{TDB}}$'+')')
axes[0,0].set(xlim=[-0.1,0.1], ylabel='Flux - Offset (ppt)')
axes[0,1].set(xlim=[model_time[0],model_time[-1]], ylim=[-599,499], ylabel='Flux - Offset (ppm)')   
axes[1,0].set(ylabel='Res. (ppm)')    
axes[1,1].set(ylabel='Res. (ppm)')   



#==============================================================================
#::: ellipsoidal modulation
#==============================================================================
alles = allesfitter.allesclass(datadir)
alles.posterior_params_median['b_phase_curve_beaming_TESS'] = None             
alles.posterior_params_median['b_phase_curve_atmospheric_TESS'] = None          
# alles.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = None             
alles.posterior_params_median['b_sbratio_TESS'] = 0
model_flux_ellipsoidal = alles.get_posterior_median_model(inst, key, xx=model_time, settings=alles.settings)
axes[0,1].plot(model_time, (model_flux_ellipsoidal-1)*1e6, lw=2, ls=(0, (3, 1, 1, 1, 1, 1)), label='Ellipsoidal')



#==============================================================================
#::: atmospheric modulation
#==============================================================================
alles = allesfitter.allesclass(datadir)
alles.posterior_params_median['b_phase_curve_beaming_TESS'] = None             
# alles.posterior_params_median['b_phase_curve_atmospheric_TESS'] = None          
alles.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = None           
alles.posterior_params_median['b_sbratio_TESS'] = 0
model_flux_atmospheric = alles.get_posterior_median_model(inst, key, xx=model_time, settings=alles.settings)
axes[0,1].plot(model_time, (model_flux_atmospheric-1)*1e6, lw=2, ls='--', label='Atmospheric')



#==============================================================================
#::: Doppler boosting (beaming) modulation
#==============================================================================
alles = allesfitter.allesclass(datadir)
# alles.posterior_params_median['b_phase_curve_beaming_TESS'] = None             
alles.posterior_params_median['b_phase_curve_atmospheric_TESS'] = None          
alles.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = None           
alles.posterior_params_median['b_sbratio_TESS'] = 0
model_flux_Doppler = alles.get_posterior_median_model(inst, key, xx=model_time, settings=alles.settings)
axes[0,1].plot(model_time, (model_flux_Doppler-1)*1e6, lw=2, ls=':', label='Doppler Boosting')



#==============================================================================
#::: helper model to compute secondary eclipse depth
#==============================================================================
alles = allesfitter.allesclass(datadir)
alles.posterior_params_median['b_phase_curve_beaming_TESS'] = None             
alles.posterior_params_median['b_phase_curve_atmospheric_TESS'] = None          
alles.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = None           
# alles.posterior_params_median['b_sbratio_TESS'] = 0
model_flux_helper = alles.get_posterior_median_model(inst, key, xx=model_time, settings=alles.settings)
# axes[0,1].plot(model_time, (model_flux_helper-1)*1e6, lw=2, color='r', ls=':')

# ind_occ = np.where((time>0.4) & (time<0.6))[0]
# null_level = np.min(model_flux_helper[ind_occ])
# axes[0,1].plot(model_time, np.ones_like(model_time)*null_level, )    
    

#==============================================================================
#::: full model
#==============================================================================
for ax, z in zip(axes[0,:],zoom):
    ax.plot(time, (flux-baseline-1)*z, 'k.', color='silver', rasterized=True)
    ax.plot(model_time, (model_flux-1)*z, 'r-', lw=2, label='Full Model')
    for i in range(len(model_fluxes)):
        ax.plot(model_time, (model_fluxes[i]-1)*z, 'r-', lw=1.5, alpha=0.1)
        
# axes[0,1].plot(model_time, np.zeros_like(model_time), 'k--', lw=1)
    
    

#==============================================================================
#::: finish
#==============================================================================
for ax in axes.flatten(): 
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.tight_layout()
fig.subplots_adjust(hspace=0)
axes[0,1].legend(loc='best', ncol=5, bbox_to_anchor=[0.75,1.2])
fig.savefig('WASP-18_fit.pdf', bbox_inches='tight')



#==============================================================================
#::: print
#==============================================================================
# ind_occ = np.where((time>0.4) & (time<0.6))[0]
# print('\nNightside flux:', (np.min(model_flux_helper[ind_occ])-np.min(model_flux[ind_occ]))*1e6, 'ppm')

