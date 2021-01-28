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
import warnings

#::: specific modules
try:
    from wotan import flatten
except ImportError:
    pass

#::: my modules
import allesfitter
from allesfitter.lightcurves import eclipse_width_smart
from allesfitter.exoworlds_rdx.lightcurves.index_transits import get_tmid_observed_transits
# from .general_output import logprint


    

###############################################################################
#::: prepare TTV fit (if chosen)
###############################################################################
def prepare_ttv_fit(datadir, ax=None):
    '''
    this must be run *after* reduce_phot_data()
    '''
    ax0 = ax
    
    alles = allesfitter.allesclass(datadir)
    window = alles.settings['fast_fit_width']
       
    if not os.path.exists( os.path.join(datadir,'ttv_preparation') ): os.makedirs(os.path.join(datadir,'ttv_preparation'))
    with open(os.path.join(datadir,'ttv_preparation','ttv_initial_guess_params.csv'),'w') as f:
        f.write('')
    
    def plot_all_transits_color_coded():
        for inst in alles.settings['inst_phot']:
            time = alles.data[inst]['time']
            for companion in alles.settings['companions_phot']:
                ind = []
                for i, t in enumerate(alles.data[companion+'_tmid_observed_transits']):
                    ind += list( np.where((time >= (t - window/2.)) & (time <= (t + window/2.)))[0] )
                ax.plot( alles.data[inst]['time'][ind], alles.data[inst]['flux'][ind], ls='none', marker='.', label=companion )
    
    
    for companion in alles.settings['companions_phot']:
        with open(os.path.join(datadir,'ttv_preparation','ttv_initial_guess_params.csv'),'a') as f:
            f.write('#TTV companion '+companion+',,,,,\n')
        
        #----------------------------------------------------------------------
        #::: get combined data from all instruments
        #----------------------------------------------------------------------
        all_times = []
        all_flux = []
        for inst in alles.settings['inst_phot']:
            all_times += list(alles.data[inst]['time'])
            all_flux += list(alles.data[inst]['flux'])
        ind_sort = np.argsort(all_times)
        all_times = np.array(all_times)[ind_sort]
        all_flux = np.array(all_flux)[ind_sort]
        
        #----------------------------------------------------------------------
        #::: get eclipse window
        #----------------------------------------------------------------------
        alles.initial_guess_params_median[companion+'_epoch']
        eclipse_width = eclipse_width_smart(alles.initial_guess_params_median[companion+'_period'], 
                                    alles.initial_guess_params_median[companion+'_rr'], 
                                    alles.initial_guess_params_median[companion+'_rsuma'], 
                                    alles.initial_guess_params_median[companion+'_cosi'], 
                                    alles.initial_guess_params_median[companion+'_f_s'], 
                                    alles.initial_guess_params_median[companion+'_f_c'], 
                                    )[0]
        
        #----------------------------------------------------------------------
        #::: compute tmid, ttv_guess and make per-transit-plots
        #----------------------------------------------------------------------
        tmids = []
        
        alles.data[companion+'_tmid_observed_transits'] = get_tmid_observed_transits(all_times,alles.initial_guess_params_median[companion+'_epoch'],alles.initial_guess_params_median[companion+'_period'],alles.settings['fast_fit_width'])
        N = len(alles.data[companion+'_tmid_observed_transits'])
        fig, axes = plt.subplots(N, 1, figsize=(6,4*N), sharey=True, tight_layout=True)
        
        for i, t in enumerate(alles.data[companion+'_tmid_observed_transits']):
            ind_tr1 = np.where((all_times >= (t - window/2.)) & (all_times <= (t + window/2.)))[0]
            tr_times = all_times[ind_tr1]
            tr_flux = all_flux[ind_tr1]
            t_exp = np.median(np.diff(tr_times))
            N_points_in_eclipse = int(eclipse_width/t_exp)
            try:
                trend = flatten(tr_times, tr_flux, window_length=eclipse_width/2., method='biweight', return_trend=True)[1]
                tmid = np.median( tr_times[ np.argsort(trend)[0:int(N_points_in_eclipse/2.)] ] )
            except:
                warnings.warn('Install wotan for improved performance of prepare_ttv_fit().')
                trend = None
                tmid = np.median( tr_times[ np.argsort(tr_times)[0:int(N_points_in_eclipse/2.)] ] )
            ttv_guess = tmid - t
            tmids.append(tmid)
            
            ax = axes[i]
            ax.plot(tr_times, tr_flux, 'b.', rasterized=True)
            if trend is not None: ax.plot(tr_times, trend, 'r-')
            ax.axvline(t,c='grey',ls='--',label='linear prediction')
            ax.axvline(tmid,c='r',ls='--',label='flux minimum')
            ax.set(xlabel='Time', ylabel='Flux', xlim=[t-window/2., t+window/2.])
            ax.text(0.95,0.95,'Transit '+str(i+1), va='top', ha='right', transform=ax.transAxes)
            with open(os.path.join(datadir,'ttv_preparation','ttv_initial_guess_params.csv'),'a') as f:
               f.write(companion+'_ttv_transit_'+str(i+1)+','+np.format_float_positional(ttv_guess,4)+',1,uniform '+np.format_float_positional(ttv_guess-0.01,4)+' '+np.format_float_positional(ttv_guess+0.01,4)+',TTV$_\mathrm{'+companion+';'+str(i+1)+'}$,d\n')
        axes[0].legend()
        fig.savefig(os.path.join(datadir,'ttv_preparation','ttv_preparation_'+companion+'_per_transit.pdf'), bbox_inches='tight')
        plt.close(fig)
         
        tmids = np.array(tmids)
        
        
        #----------------------------------------------------------------------
        #::: ttv guess 0-C plot
        #----------------------------------------------------------------------
        nr = np.array([ int(np.round( (t-tmids[0]) / alles.initial_guess_params_median[companion+'_period'] )) for t in tmids ]) #get corresponding transit number
        nr -= int(nr[-1]/2.) #shift into the middle of the data set
        period_mean, epoch_mean = np.polyfit(nr, tmids, 1)             
        
        fig, axes = plt.subplots(2,1,figsize=(6,8),tight_layout=True,sharex=True)
        axes[0].plot(nr, tmids, 'bo', label='Companion '+companion)
        axes[0].plot(nr, epoch_mean + nr * period_mean, 'b-')
        axes[0].set(xlabel='Nr.', ylabel='Transit mid-time')
        axes[0].legend()
        axes[1].plot(nr, tmids-(epoch_mean + nr * period_mean), 'bo')
        axes[1].axhline(0,c='grey',ls='--')
        fig.savefig(os.path.join(datadir,'ttv_preparation','ttv_preparation_'+companion+'_oc.pdf'), bbox_inches='tight')
        
        period_dev = np.abs( (period_mean-alles.initial_guess_params_median[companion+'_period'])/alles.initial_guess_params_median[companion+'_period'] )
        epoch_dev = np.abs( (epoch_mean-alles.initial_guess_params_median[companion+'_epoch'])/alles.initial_guess_params_median[companion+'_epoch'] )
        
        print('\nCompanion', companion)
        print('Initial guess for mean period and epoch:')
        print(np.format_float_positional(alles.initial_guess_params_median[companion+'_period']), 
              np.format_float_positional(alles.initial_guess_params_median[companion+'_epoch']))
        print('New estimate for mean period and epoch:')
        print(np.format_float_positional(period_mean,4), 
              np.format_float_positional(epoch_mean,4))
        # print('Deviation from another:')
        # print(np.format_float_positional(period_dev,4), 
        #       np.format_float_positional(epoch_dev,4))
        if (period_dev > 0.01) or (epoch_dev > 0.01):
            print('\n! Consider updating your initial guess to these new estimated mean values.')
            print('\n! If you do, then you must rerun this code.')
        else:
            print('\n! Looks great! You are ready to fit.')

        
        #----------------------------------------------------------------------
        #::: full lightcurve plot
        #----------------------------------------------------------------------
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
    
        fname = os.path.join(datadir,'ttv_preparation','ttv_preparation_'+companion+'.jpg')
        fig = plt.gcf()
        fig.savefig(fname, bbox_inches='tight' )  
        plt.close(fig)
                