#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:50:31 2020

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
from allesfitter.exoworlds_rdx.lightcurves.index_transits import index_transits, get_tmid_observed_transits
from allesfitter.plotting import fullplot, brokenplot, chunkplot, monthplot, tessplot


    

###############################################################################
#::: prepare TTV fit (if chosen)
###############################################################################
def prepare_ttv_fit(datadir, style='fullplot', max_transits=20):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter; must contain all the data files; output directories and files will also be created inside datadir
    style : str
        chose between 'fullplot', 'monthplot', and 'tessplot'; this defines how the plot looks
        default is 'fullplot'
    max_transits : int
        the maximum number of transits to be plotted into the same pdf. If there were more transits than `max_transits`
        additional plots will be created.
    Outputs:
    --------
    None
    
    Notes:
    ------
    This function must be run *after* reduce_phot_data()
    Throughout, we use fast_fit_width as the approximation for the transit window
    '''
    
    #----------------------------------------------------------------------
    #::: setup
    #----------------------------------------------------------------------
    alles = allesfitter.allesclass(datadir)
    window = alles.settings['fast_fit_width']
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] #the default sequence of plot colors (for the companions)
    if not os.path.exists( os.path.join(datadir,'ttv_preparation') ): os.makedirs(os.path.join(datadir,'ttv_preparation'))
    with open(os.path.join(datadir,'ttv_preparation','ttv_initial_guess_params.csv'),'w') as f:
        f.write('')
                
        
    #----------------------------------------------------------------------
    #::: get the combined, full data from all instruments
    #----------------------------------------------------------------------
    time_combined = []
    flux_combined = []
    for inst in alles.settings['inst_phot']:
        time_combined += list(alles.fulldata[inst]['time'])
        flux_combined += list(alles.fulldata[inst]['flux'])
    ind_sort = np.argsort(time_combined)
    time_combined = np.array(time_combined)[ind_sort]
    flux_combined = np.array(flux_combined)[ind_sort]
    
    
    #----------------------------------------------------------------------
    #::: get eclipse widths per companion
    #----------------------------------------------------------------------
    eclipse_width = {}
    for companion in alles.settings['companions_phot']:
        alles.initial_guess_params_median[companion+'_epoch']
        eclipse_width[companion] = eclipse_width_smart(alles.initial_guess_params_median[companion+'_period'], 
                                                        alles.initial_guess_params_median[companion+'_rr'], 
                                                        alles.initial_guess_params_median[companion+'_rsuma'], 
                                                        alles.initial_guess_params_median[companion+'_cosi'], 
                                                        alles.initial_guess_params_median[companion+'_f_s'], 
                                                        alles.initial_guess_params_median[companion+'_f_c'])[0]
    
    
    #----------------------------------------------------------------------
    #::: loop over all photometric companions
    #----------------------------------------------------------------------
    for companion in alles.settings['companions_phot']:
        with open(os.path.join(datadir,'ttv_preparation','ttv_initial_guess_params.csv'),'a') as f:
            f.write('#TTV companion '+companion+',,,,,\n')
        
        
        #----------------------------------------------------------------------
        #::: compute tmid and a ttv_guess, and make per-transit-plots per companion
        #----------------------------------------------------------------------
        tmid_estimates = []
        tmid_linear_predictions = get_tmid_observed_transits(time_combined, 
                                                             alles.initial_guess_params_median[companion+'_epoch'], 
                                                             alles.initial_guess_params_median[companion+'_period'],
                                                             window)
        N = len(tmid_linear_predictions)
        end_transit_index = max_transits
        for i, tmid1 in enumerate(tmid_linear_predictions):
            plot_index = i % max_transits
            if plot_index == 0:
                end_transit_index = i + max_transits if i + max_transits < N else N
                fig, axes = plt.subplots((end_transit_index - i), 1, figsize=(6, 4 * (end_transit_index - i)), sharey=True, tight_layout=True)
            #::: estimate the observed tranist midtime by computing the minimum of the data around the linearly predicted transit midtime
            ind_tr1 = np.where((time_combined >= (tmid1 - window/2.)) & (time_combined <= (tmid1 + window/2.)))[0]
            tr_times = time_combined[ind_tr1]
            tr_flux = flux_combined[ind_tr1]
            t_exp = np.median(np.diff(tr_times))
            N_points_in_eclipse = int(eclipse_width[companion]/t_exp)
            try:
                trend = flatten(tr_times, tr_flux, window_length=eclipse_width[companion]/2., method='biweight', return_trend=True)[1]
                tmid2 = np.median( tr_times[ np.argsort(trend)[0:int(N_points_in_eclipse/2.)] ] ) # the tmid estimated from the observations as the minimum of the data
            except:
                warnings.warn('Install wotan for improved performance of prepare_ttv_fit().')
                trend = None
                tmid2 = np.median( tr_times[ np.argsort(tr_times)[0:int(N_points_in_eclipse/2.)] ] ) # the tmid estimated from the observations as the minimum of the data
            ttv_guess = tmid2 - tmid1 # a guess for the TTV
            tmid_estimates.append(tmid2)
            
            #::: plot this per transit
            ax = axes[plot_index] if isinstance(axes, (list, np.ndarray)) else axes
            ax.plot(tr_times, tr_flux, marker='.', ls='none', color=alles.settings[companion+'_color'], rasterized=True)
            if trend is not None: ax.plot(tr_times, trend, 'r-')
            ax.axvline(tmid1,c='grey',ls='--',label='linear prediction')
            ax.axvline(tmid2,c='r',ls='--',label='flux minimum')
            ax.set(xlabel='Time', ylabel='Flux', xlim=[tmid1-window/2., tmid1+window/2.])
            ax.text(0.95,0.95,'Transit '+str(i+1), va='top', ha='right', transform=ax.transAxes)
            with open(os.path.join(datadir,'ttv_preparation','ttv_initial_guess_params.csv'),'a') as f:
               f.write(companion+'_ttv_transit_'+str(i+1)+','+np.format_float_positional(ttv_guess,4)+',1,uniform '+np.format_float_positional(ttv_guess-0.01,4)+' '+np.format_float_positional(ttv_guess+0.01,4)+',TTV$_\mathrm{'+companion+';'+str(i+1)+'}$,d\n')
            if i == end_transit_index - 1:
                ax_for_legend = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
                ax_for_legend.legend()
                fig.savefig(os.path.join(datadir,'ttv_preparation','ttv_preparation_'+companion+'_per_transit_' + str(end_transit_index) + 'th.pdf'), bbox_inches='tight')
                plt.close(fig)
         
        tmid_estimates = np.array(tmid_estimates)
        
        
        #----------------------------------------------------------------------
        #::: ttv guess 0-C plot
        #----------------------------------------------------------------------
        nr = np.array([ int(np.round( (t-tmid_estimates[0]) / alles.initial_guess_params_median[companion+'_period'] )) for t in tmid_estimates ]) #get corresponding transit number
        nr -= int(nr[-1]/2.) #shift into the middle of the data set
        period_mean, epoch_mean = np.polyfit(nr, tmid_estimates, 1)             
        
        fig, axes = plt.subplots(2,1,figsize=(6,8),tight_layout=True,sharex=True)
        axes[0].plot(nr, tmid_estimates, marker='o', ls='none', color=alles.settings[companion+'_color'], label=companion)
        axes[0].plot(nr, epoch_mean + nr * period_mean, marker='', ls='--', color='grey')
        axes[0].set(ylabel='Transit mid-time')
        axes[0].legend()
        axes[1].plot(nr, tmid_estimates-tmid_linear_predictions, marker='o', ls='none', color=alles.settings[companion+'_color'])
        axes[1].axhline(0, ls='--', color='grey')
        axes[1].set(xlabel='Nr.', ylabel='TTV (min.)')
        fig.savefig(os.path.join(datadir,'ttv_preparation','ttv_preparation_'+companion+'_oc.pdf'), bbox_inches='tight')
        plt.close(fig)
        
        
        #----------------------------------------------------------------------
        #::: compute and output the deviation
        #----------------------------------------------------------------------
        period_dev = np.abs( (period_mean-alles.initial_guess_params_median[companion+'_period'])/alles.initial_guess_params_median[companion+'_period'] )
        epoch_dev = np.abs( (epoch_mean-alles.initial_guess_params_median[companion+'_epoch'])/alles.initial_guess_params_median[companion+'_epoch'] )
        
        print('\nCompanion', companion)
        print('Initial guess for mean period and epoch:')
        print(np.format_float_positional(alles.initial_guess_params_median[companion+'_period']), 
              np.format_float_positional(alles.initial_guess_params_median[companion+'_epoch']))
        print('New estimate for mean period and epoch:')
        print(np.format_float_positional(period_mean,4), 
              np.format_float_positional(epoch_mean,4))
        if (period_dev > 0.01) or (epoch_dev > 0.01):
            print('\n! Consider updating your initial guess to these new estimated mean values.')
            print('\n! If you do, then you must rerun this code.')
        else:
            print('\n! Looks great! You are ready to fit.')


        #----------------------------------------------------------------------
        #::: full lightcurve plot
        #----------------------------------------------------------------------
        #::: plot all data points
        if style=='fullplot':
            axes = fullplot(time_combined, flux_combined, color='silver')
        elif style=='brokenplot':
            axes = brokenplot(time_combined, flux_combined, color='silver')
        elif style=='chunkplot':
            axes = chunkplot(time_combined, flux_combined, color='silver')
        elif style=='monthplot':
            axes = monthplot(time_combined, flux_combined, color='silver')
        elif style=='tessplot':
            axes = tessplot(time_combined, flux_combined, color='silver')
        else:
            raise ValueError("The keyword argument 'style' must be 'fullplot', 'monthplot', or 'tessplot'.")
        
        #::: mark the tranists/eclipses of each photometric companion
        for i, c in enumerate(alles.settings['companions_phot']):
            ind_tr, ind_out = index_transits(time_combined, alles.initial_guess_params_median[c+'_epoch'], alles.initial_guess_params_median[c+'_period'], window)
            if style=='fullplot':
                axes = fullplot(time_combined[ind_tr], flux_combined[ind_tr], color=alles.settings[c+'_color'], ax=axes, label=c)
            elif style=='brokenplot':
                axes = brokenplot(time_combined[ind_tr], flux_combined[ind_tr], color=alles.settings[c+'_color'], bax=axes, label=c)
            elif style=='chunkplot':
                axes = chunkplot(time_combined[ind_tr], flux_combined[ind_tr], color=alles.settings[c+'_color'], axes=axes, label=c)
            elif style=='monthplot':
                axes = monthplot(time_combined[ind_tr], flux_combined[ind_tr], color=alles.settings[c+'_color'], axes=axes, label=c)
            elif style=='tessplot':
                axes = tessplot(time_combined[ind_tr], flux_combined[ind_tr], color=alles.settings[c+'_color'], axes=axes, label=c)
                
        #::: add legend
        axes = np.atleast_1d(axes)
        for i, c in enumerate(alles.settings['companions_phot']):
            axes[0].text(0.02+i*0.02, 0.95, c, color=alles.settings[c+'_color'], ha='left', va='top', transform=axes[0].transAxes, zorder=15)
            
        #::: add vertical lines and numbers
        flux_min = np.nanmin(flux_combined)
        flux_max = np.nanmax(flux_combined)
        for ax in axes:
            for i, tmid in enumerate(alles.data[companion+'_tmid_observed_transits']):
                if (tmid>ax.get_xlim()[0]) & (tmid<ax.get_xlim()[1]):
                    ax.text( tmid, 0.992*flux_min, str(i+1), color=alles.settings[companion+'_color'], ha='center', zorder=12 )  
                    ax.axvline( tmid, color='lightgrey', zorder=11 )
                    ax.set(ylim=[0.99*flux_min, 1.002*flux_max], title='Companion '+companion) 
            
        #::: wrap up
        fname = os.path.join(datadir,'ttv_preparation','ttv_preparation_'+companion+'.jpg')
        fig = plt.gcf()
        fig.savefig(fname, bbox_inches='tight' )  
        plt.close(fig)
                