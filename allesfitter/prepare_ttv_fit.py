#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:50:31 2020

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

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import os

#::: my modules
import allesfitter
from allesfitter.exoworlds_rdx.lightcurves.index_transits import get_tmid_observed_transits




###############################################################################
#::: prepare TTV fit (if chosen)
###############################################################################
def prepare_ttv_fit(datadir, ax=None):
    '''
    this must be run *after* reduce_phot_data()
    '''
    ax0 = ax
    
    alles = allesfitter.allesclass(datadir)
    
    
    def plot_all_transits_color_coded():
        width = alles.settings['fast_fit_width']
        for inst in alles.settings['inst_phot']:
            time = alles.data[inst]['time']
            for companion in alles.settings['companions_phot']:
                ind = []
                for i, t in enumerate(alles.data[companion+'_tmid_observed_transits']):
                    ind += list( np.where((time >= (t - width/2.)) & (time <= (t + width/2.)))[0] )
                ax.plot( alles.data[inst]['time'][ind], alles.data[inst]['flux'][ind], ls='none', marker='.', label=companion )
    
    
    for companion in alles.settings['companions_phot']:
        print('#TTV companion '+companion)
        all_times = []
        all_flux = []
        for inst in alles.settings['inst_phot']:
            all_times += list(alles.data[inst]['time'])
            all_flux += list(alles.data[inst]['flux'])
        
        alles.data[companion+'_tmid_observed_transits'] = get_tmid_observed_transits(all_times,alles.initial_guess_params_median[companion+'_epoch'],alles.initial_guess_params_median[companion+'_period'],alles.settings['fast_fit_width'])
        for i, t in enumerate(alles.data[companion+'_tmid_observed_transits']):
            print(companion+'_ttv_transit_'+str(i+1)+',0,1,uniform -0.05 0.05,TTV$_\mathrm{'+companion+';'+str(i+1)+'}$,d')
                    
        #::: plot
        flux_min = np.nanmin(all_flux)
        flux_max = np.nanmax(all_flux)
        if ax0 is None:
            days = np.max(all_times) - np.min(all_times)
            figsizex = np.max( [5, 5*(days/10.)] )
            fig, ax = plt.subplots(figsize=(figsizex, 4)) #figsize * 5 for every 20 days
        for inst in alles.settings['inst_phot']:
            ax.plot(alles.fulldata[inst]['time'], alles.fulldata[inst]['flux'],ls='none',marker='.',color='silver')
            # ax.plot(alles.data[inst]['time'], alles.data[inst]['flux'],ls='none',marker='.',label=inst) #color code by instrument
        plot_all_transits_color_coded() #color code by companion
                
        ax.plot( alles.data[companion+'_tmid_observed_transits'], np.ones_like(alles.data[companion+'_tmid_observed_transits'])*0.997*flux_min, 'k^', zorder=12 )
        for i, tmid in enumerate(alles.data[companion+'_tmid_observed_transits']):
            ax.text( tmid, 0.992*flux_min, str(i+1), ha='center', zorder=12 )  
            ax.axvline( tmid, color='grey', zorder=11 )
        ax.set(ylim=[0.99*flux_min, 1.002*flux_max], xlabel='Time (BJD)', ylabel='Realtive Flux', title='Companion '+companion) 
        ax.legend(loc='best')
    
        if not os.path.exists( os.path.join(datadir,'results') ): os.makedirs(os.path.join(datadir,'results'))
        fname = os.path.join(datadir,'results','preparation_for_TTV_fit_'+companion+'.jpg')
        fig = plt.gcf()
        fig.savefig(fname, bbox_inches='tight' )  
        plt.close(fig)
                