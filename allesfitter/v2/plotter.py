#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:22:59 2020

@author: 
Dr. Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
GitHub: https://github.com/MNGuenther
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import os, sys

#::: my modules
from .translator import translate

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




'''
A collection of plug-and-play plotting functions, to add to axes in a gridspec or subplot layout
'''
    
    

    
###############################################################################
#::: full lightcurve
###############################################################################
def plot_lc_full(ax, time, flux, flux_err, time_grid, model_flux_grid):
    ax.errorbar(time, flux, yerr=flux_err, fmt='b.', rasterized=True, zorder=11)
    ax.plot(time_grid, model_flux_grid, 'r-', lw=2, zorder=12)
    ax.set(xlabel='Time'+r' $\mathrm{(BJD_{TDB})}$', ylabel='Flux')
    return ax


###############################################################################
#::: phase-folded lightcurve
###############################################################################
def plot_lc_phase(ax, phi, flux, phase, phaseflux, phaseflux_err, phase_grid, model_phaseflux_grid):
    ax.plot(phi, flux, 'k.', color='lightgrey', rasterized=True, zorder=10)
    ax.errorbar(phase, phaseflux, yerr=phaseflux_err, fmt='b.', zorder=11)
    ax.plot(phase_grid, model_phaseflux_grid, 'r-', lw=2, zorder=12)
    ax.set(xlabel='Phase', ylabel='Flux')
    return ax


###############################################################################
#::: phase-folded and zoomed lightcurve; same as for plot_phase(), but zoomed in x and in minutes
###############################################################################
def plot_lc_phasezoom(ax, phi, flux, phase, phaseflux, phaseflux_err, phase_grid, model_phaseflux_grid, zoomfactor):
    '''
    zoomfactor : period * 24 * 60
    '''
    ax.plot(phi*zoomfactor, flux, 'k.', color='lightgrey', rasterized=True, zorder=10)
    ax.errorbar(phase*zoomfactor, phaseflux, yerr=phaseflux_err, fmt='b.', zorder=11)
    ax.plot(phase_grid*zoomfactor, model_phaseflux_grid, 'r-', lw=2, zorder=12)
    ax.set(xlim=[-240,240])
    ax.set(xlabel='Time (min.)', ylabel='Flux')
    return ax
    
    
###############################################################################
#::: full RV series
###############################################################################
def plot_rv_full(ax, time, rv, rv_err, time_grid, model_flux_grid):
    ax.errorbar(time, rv, yerr=rv_err, fmt='bo', zorder=11)
    ax.plot(time_grid, model_flux_grid, 'r-', lw=2, zorder=12)
    ax.set(xlabel='Time'+r' $\mathrm{(BJD_{TDB})}$', ylabel='RV (km/s)')
    return ax


###############################################################################
#::: phase-folded RV series
###############################################################################
def plot_rv_phase(ax, phi, rv, rv_err, phase_grid, model_phaserv_grid):
    ax.errorbar(phi, rv, yerr=rv_err, fmt='bo', zorder=11)
    ax.plot(phase_grid, model_phaserv_grid, 'r-', lw=2, zorder=12)
    ax.set(xlabel='Phase', ylabel='RV (km/s)')
    return ax


###############################################################################
#::: info text 
###############################################################################
def plot_info(ax, text=0, params=None, settings=None, **kwargs):
    params = translate(params=params, settings=settings, quiet=True, **kwargs)
    
    ax.set_axis_off()
    
    if text==0:
        ax.text(0,0.95,'R_comp = '+np.format_float_positional(params['R_companion_earth'],2)+r' $R_\odot$ = ' + np.format_float_positional(params['R_companion_jup'],2)+r' $R_J$ = ' + np.format_float_positional(params['R_companion_sun'],2)+r' $R_\odot$', transform=ax.transAxes) #bodies
        ax.text(0,0.85,'M_comp = '+np.format_float_positional(params['M_companion_earth'],2)+r' $M_\odot$ = ' + np.format_float_positional(params['M_companion_jup'],2)+r' $M_J$ = ' + np.format_float_positional(params['M_companion_sun'],2)+r' $M_\odot$', transform=ax.transAxes)
        ax.text(0,0.75,'R_host = '+str(params['R_host'])+r' $R_\odot$', transform=ax.transAxes)
        ax.text(0,0.65,'M_host = '+str(params['M_host'])+r' $M_\odot$', transform=ax.transAxes)
        ax.text(0,0.55,'sbratio = '+str(params['sbratio']), transform=ax.transAxes)
        ax.text(0,0.45,'epoch' + ' = '+str(params['epoch'])+r' $\mathrm{BJD_{TDB}}$', transform=ax.transAxes) #orbits
        ax.text(0,0.35,'period = '+str(params['period'])+' days', transform=ax.transAxes)
        ax.text(0,0.25,'incl = '+str(params['incl'])+' deg', transform=ax.transAxes)
        ax.text(0,0.15,'ecc = '+str(params['ecc']), transform=ax.transAxes)
        ax.text(0,0.05,'omega = '+str(params['omega'])+' deg', transform=ax.transAxes)
    
    if text==1:
        ax.text(0,0.95,'dil = '+str(params['dil']), transform=ax.transAxes)
        ax.text(0,0.85,'R_comp/R_host = '+np.format_float_positional(params['rr'],5,False), transform=ax.transAxes)
        ax.text(0,0.75,'(R_comp+R_host)/a = '+np.format_float_positional(params['rsuma'],5,False), transform=ax.transAxes)
        ax.text(0,0.65,'R_comp/a = '+np.format_float_positional(params['R_companion_over_a'],5,False), transform=ax.transAxes)
        ax.text(0,0.55,'R_host/a = '+np.format_float_positional(params['R_host_over_a'],5,False), transform=ax.transAxes)
        ax.text(0,0.45,'cosi = '+np.format_float_positional(params['cosi'],5,False), transform=ax.transAxes)
        ax.text(0,0.35,r'$\sqrt{e} \cos{\omega}$ = '+np.format_float_positional(params['f_c'],5,False), transform=ax.transAxes)
        ax.text(0,0.25,r'$\sqrt{e} \sin{\omega}$ = '+np.format_float_positional(params['f_s'],5,False), transform=ax.transAxes)
        ax.text(0,0.15,'LD = '+str(params['ldc']), transform=ax.transAxes)
        try: ax.text(0,0.05,'LD transf = ['+", ".join([np.format_float_positional(item,5,False) for item in params['ldc_transformed']]) + ']', transform=ax.transAxes)
        except: pass
    return ax