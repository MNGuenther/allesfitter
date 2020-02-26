#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:10:51 2018

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
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import warnings
from astropy.time import Time
#import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 

#::: allesfitter modules
from . import config
from .utils import latex_printer
from .computer import update_params,\
                     calculate_model, rv_fct, flux_fct,\
                     calculate_baseline,calculate_stellar_var,\
                     calculate_yerr_w
from .exoworlds_rdx.lightcurves import lightcurve_tools as lct
from .exoworlds_rdx.lightcurves.index_transits import get_tmid_observed_transits
                    
                     
                     
###############################################################################
#::: print function that prints into console and logfile at the same time
############################################################################### 
def logprint(*text):
    if config.BASEMENT.settings['print_progress']:
        print(*text)
    original = sys.stdout
    try:
        with open( os.path.join(config.BASEMENT.outdir,'logfile_'+config.BASEMENT.now+'.log'), 'a' ) as f:
            sys.stdout = f
            print(*text)
    except OSError:
        pass #For unknown reasons, the combination of open() and os.path.join() does not work on some Windows versions
    sys.stdout = original
                     
    
    
###############################################################################
#::: draw samples from the initial guess
###############################################################################
def draw_initial_guess_samples(Nsamples=1):
    if Nsamples==1:
        samples = np.array([config.BASEMENT.theta_0])
    else:
        samples = config.BASEMENT.theta_0 + config.BASEMENT.init_err * np.random.randn(Nsamples, len(config.BASEMENT.theta_0))    
    return samples
        
    
    
###############################################################################
#::: plot all data in one panel
###############################################################################
def plot_panel(datadir):
    
    config.init(datadir)
    
    if len(config.BASEMENT.settings['inst_phot'])>0 and len(config.BASEMENT.settings['inst_rv'])>0:
        fig, axes = plt.subplots(2,1,figsize=(20,10))
    elif len(config.BASEMENT.settings['inst_phot'])>0:
        fig, axes = plt.subplots(1,1,figsize=(20,5))
        axes = [axes]
    elif len(config.BASEMENT.settings['inst_rv'])>0:
        fig, axes = plt.subplots(1,1,figsize=(20,5))
        axes = [None,axes]
    
    for inst in config.BASEMENT.settings['inst_phot']:
        ax = axes[0]
        ax.plot(config.BASEMENT.fulldata[inst]['time'], config.BASEMENT.fulldata[inst]['flux'], marker='.', ls='none', color='lightgrey', rasterized=True)
        ax.plot(config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['flux'], marker='.', ls='none', label=inst, rasterized=True)
        ax.legend()
        ax.set(ylabel='Relative Flux', xlabel='Time (BJD)')
        
    for inst in config.BASEMENT.settings['inst_rv']:
        ax = axes[1]
        ax.plot(config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['rv'], marker='.', ls='none', label=inst)
        ax.legend()
        ax.set(ylabel='RV (km/s)', xlabel='Time (BJD)')
    
    fig.savefig( os.path.join(config.BASEMENT.outdir,'data_panel.pdf'), bbox_inches='tight' )
    return fig, axes
        



###############################################################################
#::: plot all transits in one panel
###############################################################################
def plot_panel_transits(datadir, ax=None, insts=None, companions=None, colors=None, title=None, ppm=False, ylim=None, yticks=None, fontscale=2):

    config.init(datadir)
    
    #::: more plotting settings
    SMALL_SIZE = 8*fontscale
    MEDIUM_SIZE = 10*fontscale
    BIGGER_SIZE = 12*fontscale
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    samples = draw_initial_guess_samples()
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    
    if companions is None:
        companions = config.BASEMENT.settings['companions_phot']
    if colors is None:
        colors = [sns.color_palette('deep')[i] for i in [0,1,3]]
    if insts is None:
        insts = config.BASEMENT.settings['inst_phot']

    ally = []
    
    if ax is None:
        ax_init = None
        fig, axes = plt.subplots(len(insts),len(companions),figsize=(6*len(companions),4*len(insts)), sharey=True, sharex=True)
        axes = np.atleast_2d(axes).T
    else:
        ax_init = ax
        axes = np.atleast_2d(ax).T
        
    for i,(companion, color) in enumerate(zip(companions, colors)):
        
        for j,inst in enumerate(insts):
            ax = axes[i,j]
            
            key='flux'
            if title is None:
                if i==0:
                    title=inst
                else:
                    title=''
            if j==len(insts)-1:
                xlabel=r'$\mathrm{ T - T_0 \ (h) }$'
            else:
                xlabel=''
            if i==0:
                if ppm:
                    ylabel=r'$\Delta$ Flux (ppm)'
                else:
                    ylabel='Relative Flux'
            else:
                ylabel=''
            alpha = 1.
                    
            x = config.BASEMENT.data[inst]['time']
            baseline_median = calculate_baseline(params_median, inst, key) #evaluated on x (!)
            y = config.BASEMENT.data[inst][key] - baseline_median
            
            zoomfactor = params_median[companion+'_period']*24.
            
            for other_companion in config.BASEMENT.settings['companions_phot']:
                if companion!=other_companion:
                    model = flux_fct(params_median, inst, other_companion)
                    y -= model
                    y += 1.
            
            if ppm:
                y = (y-1)*1e6
                    
            dt = 20./60./24. / params_median[companion+'_period']
                
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)    
            ax.plot( phi*zoomfactor, y, 'b.', color='silver', rasterized=True )
            ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, linestyle='none', marker='o', ms=8, color=color, capsize=0, zorder=11 )
            ax.set_xlabel(xlabel, fontsize=BIGGER_SIZE)
            ax.set_ylabel(ylabel, fontsize=BIGGER_SIZE)

            ax.text(0.97,0.87,companion,ha='right',va='bottom',transform=ax.transAxes,fontsize=BIGGER_SIZE)
            ax.text(0.03,0.87,title,ha='left',va='bottom',transform=ax.transAxes,fontsize=MEDIUM_SIZE)
        
            ally += list(phase_y)
            
            #model, phased
            xx = np.linspace( -4./zoomfactor, 4./zoomfactor, 1000)
            xx2 = params_median[companion+'_epoch'] + np.linspace( -4./zoomfactor, 4./zoomfactor, 1000)*params_median[companion+'_period']
            for ii in range(samples.shape[0]):
                s = samples[ii,:]
                p = update_params(s)
#                p = update_params(s, phased=True)
                model = flux_fct(p, inst, companion, xx=xx2) #evaluated on xx2 (!)
                if ppm:
                    model = (model-1)*1e6
                ax.plot( xx*zoomfactor, model, 'r-', alpha=alpha, zorder=12, lw=2 )
                 
    if ppm:
        ylim0 = np.nanmin(ally) - 500
        ylim1 = np.nanmax(ally) + 500
    else:
        ylim0 = np.nanmin(ally) - 500/1e6
        ylim1 = np.nanmax(ally) + 500/1e6
   
    if ylim is None:
        ylim = [ylim0, ylim1]
    
    for i in range(len(companions)):
        for j in range(len(insts)):
            ax = axes[i,j]
            ax.set( xlim=[-4,4], ylim=ylim )
            if yticks is not None:
                ax.set(yticks=yticks)
            ax.set_xticklabels(ax.get_xticks(), {'fontsize': MEDIUM_SIZE})
            ax.set_yticklabels(ax.get_yticks(), {'fontsize': MEDIUM_SIZE})
            
    
    plt.tight_layout()

    if ax_init is None:
        fig.savefig( os.path.join(config.BASEMENT.outdir,'data_panel_transits.pdf'), bbox_inches='tight' )
        return fig, axes
    else:
        return ax




###############################################################################
#::: plot
###############################################################################
def afplot(samples, companion):
    '''
    Inputs:
    -------
    samples : array
        samples from the initial guess, or from the MCMC / Nested Sampling posteriors
    '''
#    global config.BASEMENT
      
    N_inst = len(config.BASEMENT.settings['inst_all'])
    
    
    if 'do_not_phase_fold' in config.BASEMENT.settings and config.BASEMENT.settings['do_not_phase_fold']:
        fig, axes = plt.subplots(N_inst,1,figsize=(6*1,4*N_inst))
        styles = ['full']
    elif config.BASEMENT.settings['phase_variations']:
        fig, axes = plt.subplots(N_inst,5,figsize=(6*5,4*N_inst))
        styles = ['full','phase','phase_variation','phasezoom','phasezoom_occ']
    elif config.BASEMENT.settings['secondary_eclipse']:
        fig, axes = plt.subplots(N_inst,4,figsize=(6*4,4*N_inst))
        styles = ['full','phase','phasezoom','phasezoom_occ']
    else:
        fig, axes = plt.subplots(N_inst,3,figsize=(6*3,4*N_inst))
        styles = ['full','phase','phasezoom']
    axes = np.atleast_2d(axes)
    
    for i,inst in enumerate(config.BASEMENT.settings['inst_all']):
        for j,style in enumerate(styles):
#            print(i,j,inst,style)
            #::: don't phase-fold single day photometric follow-up
#            if (style=='phase') & (inst in config.BASEMENT.settings['inst_phot']) & ((config.BASEMENT.data[inst]['time'][-1] - config.BASEMENT.data[inst]['time'][0]) < 1.):
#                axes[i,j].axis('off')
            #::: don't zoom onto RV data
            if ('zoom' in style) & (inst in config.BASEMENT.settings['inst_rv']):
                axes[i,j].axis('off')
            #::: don't plot if the companion is not covered by an instrument
            elif (inst in config.BASEMENT.settings['inst_phot']) & (companion not in config.BASEMENT.settings['companions_phot']):
                axes[i,j].axis('off')
            #::: don't plot if the companion is not covered by an instrument
            elif (inst in config.BASEMENT.settings['inst_rv']) & (companion not in config.BASEMENT.settings['companions_rv']):
                axes[i,j].axis('off')
            else:
                plot_1(axes[i,j], samples, inst, companion, style)

    plt.tight_layout()
    return fig, axes



###############################################################################
#::: plot_1 (helper function)
###############################################################################
def plot_1(ax, samples, inst, companion, style, timelabel='Time', base=None, rasterized=True, marker='.', linestyle='none', color='b', markersize=8):
    '''
    Inputs:
    -------
    ax : matplotlib axis
    
    samples : array
        Prior or posterior samples to plot the fit from
    
    inst: str
        Name of the instrument (e.g. 'TESS')
        
    companion : None or str
        None or 'b'/'c'/etc.
        
    style:
        'full' / 'per_transit' / 'phase' / 'phasezoom' / 'phasezoom_occ' /'phase_variation'
        'full_residuals' / 'phase_residuals' / 'phasezoom_residuals' / 'phasezoom_occ_residuals' / 'phase_variation_residuals'
        
    timelabel:
        'Time' / 'Time_since'
        
            
    Notes:
    ------
    yerr / epoch / period: 
        come from the initial_guess value or the MCMC median (not from individual samples)

    '''
#    global config.BASEMENT
    
    if base==None:
        base = config.BASEMENT
    
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    
    if inst in base.settings['inst_phot']:
        key='flux'
        baseline_plus = 1.
        if style in ['full']:
            ylabel='Relative Flux'
        elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_variations']:
            ylabel='Relative Flux - Baseline'
        elif style in ['full_residuals', 'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_variations_residuals']:
            ylabel='Residuals'
    elif inst in base.settings['inst_rv']:
        key='rv'
        baseline_plus = 0.
        if style in ['full']:
            ylabel='RV (km/s)'
        elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_variations']:
            ylabel='RV (km/s) - Baseline'
        elif style in ['full_residuals', 'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_variations_residuals']:
            ylabel='Residuals'
        
    else:
        raise ValueError('inst should be listed in inst_phot or inst_rv...')
    
    
    if samples.shape[0]==1:
        alpha = 1.
    else:
        alpha = 0.1
        
    
    #==========================================================================
    # full time series, not phased
    # plot the 'undetrended' data
    # plot each sampled model + its baseline 
    #==========================================================================
    if style in ['full', 'full_residuals']:
        
        #::: set it up
        x = base.data[inst]['time']
        
        if timelabel=='Time_since':
            x = np.copy(x)
            objttime = Time(x, format='jd', scale='utc')
            xsave = np.copy(x)
            x -= x[0]

        y = 1.*base.data[inst][key]
        yerr_w = calculate_yerr_w(params_median, inst, key)
        
        
        #::: calculate residuals (if wished)
        if style in ['full_residuals']:
            model = calculate_model(params_median, inst, key)
            baseline = calculate_baseline(params_median, inst, key)
            stellar_var = calculate_stellar_var(params_median, 'all', key, xx=x)
            y -= model+baseline+stellar_var
            
            
        #::: plot data, not phase        
#        ax.errorbar(base.fulldata[inst]['time'], base.fulldata[inst][key], yerr=np.nanmedian(yerr_w), marker='.', linestyle='none', color='lightgrey', zorder=-1, rasterized=True ) 
        ax.errorbar(x, y, yerr=yerr_w, marker=marker, linestyle=linestyle, color=color, markersize=markersize, capsize=0, rasterized=rasterized )  
        if base.settings['color_plot']:
            ax.scatter(x, y, c=x, marker='o', rasterized=rasterized, cmap='inferno', zorder=11 ) 
            
        if timelabel=='Time_since':
            ax.set(xlabel='Time since %s [days]' % objttime[0].isot[:10], ylabel=ylabel, title=inst)
        elif timelabel=='Time':
            ax.set(xlabel='Time (BJD)', ylabel=ylabel, title=inst)
            
            
        #::: plot model + baseline, not phased
        if style in ['full']:
            
            #if <1 day of photometric data: plot with 2 min resolution
            if ((x[-1] - x[0]) < 1): 
                dt = 2./24./60. 
            #else: plot with 30 min resolution
            else: 
                dt = 30./24./60. 
                    
            if key == 'flux':
                xx_full = np.arange( x[0], x[-1]+dt, dt)
                Npoints_chunk = 48
                for i_chunk in tqdm(range(int(1.*len(xx_full)/Npoints_chunk)+2)):
                    xx = xx_full[i_chunk*Npoints_chunk:(i_chunk+1)*Npoints_chunk] #plot in chunks of 48 points (1 day)
                    if len(xx)>0 and any( (x>xx[0]) & (x<xx[-1]) ): #plot only where there is data
                        for i in range(samples.shape[0]):
                            s = samples[i,:]
                            p = update_params(s)
                            model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
                            baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
                            stellar_var = calculate_stellar_var(p, 'all', key, xx=xx) #evaluated on xx (!)
                            ax.plot( xx, baseline+stellar_var+baseline_plus, 'k-', color='orange', alpha=alpha, zorder=12 )
                            ax.plot( xx, model+baseline+stellar_var, 'r-', alpha=alpha, zorder=12 )
            elif key=='rv':
                xx = np.arange( x[0], x[-1]+dt, dt)
                for i in range(samples.shape[0]):
                    s = samples[i,:]
                    p = update_params(s)
                    model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
                    baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
                    stellar_var = calculate_stellar_var(p, 'all', key, xx=xx) #evaluated on xx (!)
                    ax.plot( xx, baseline+stellar_var+baseline_plus, 'k-', color='orange', alpha=alpha, zorder=12 )
                    ax.plot( xx, model+baseline+stellar_var, 'r-', alpha=alpha, zorder=12 )
        
        #::: other stuff
        if timelabel=='Time_since':
            x = np.copy(xsave)
            
            
            
            
    #==========================================================================
    # phase-folded time series
    # get a 'median' baseline from intial guess value / MCMC median result
    # detrend the data with this 'median' baseline
    # then phase-fold the 'detrended' data
    # plot each phase-folded model (without baseline)
    # Note: this is not ideal, as we overplot models with different epochs/periods/baselines onto a phase-folded plot
    #==========================================================================
    elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_variations',
                   'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_variations_residuals']:
        
        #::: data - baseline_median
        x = 1.*base.data[inst]['time']
        baseline_median = calculate_baseline(params_median, inst, key) #evaluated on x (!)
        stellar_var_median = calculate_stellar_var(params_median, 'all', key, xx=x) #evaluated on x (!)
        
        y = base.data[inst][key] - baseline_median - stellar_var_median
        yerr_w = calculate_yerr_w(params_median, inst, key)
        
        #::: zoom?
        if style in ['phasezoom', 'phasezoom_occ', 
                     'phasezoom_residuals', 'phasezoom_occ_residuals']: 
            zoomfactor = params_median[companion+'_period']*24.
        else: 
            zoomfactor = 1.
        
        
        #----------------------------------------------------------------------
        #::: Radial velocity
        #::: need to take care of multiple companions
        #----------------------------------------------------------------------
        if (inst in base.settings['inst_rv']):
            
            #::: remove other companions
            for other_companion in base.settings['companions_rv']:
                if companion!=other_companion:
                    model = rv_fct(params_median, inst, other_companion)[0]
                    y -= model
            
            
            #::: calculate residuals (if wished)
            if style in ['phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_variations_residuals']:
                model = rv_fct(params_median, inst, companion)[0]
                y -= model
                
                
            #::: plot data, phased        
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = 0.002, ferr_type='meansig', ferr_style='sem', sigmaclip=False)    
            if len(x) > 500:
                ax.plot( phi*zoomfactor, y, 'k.', color='lightgrey', rasterized=rasterized )
                ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, marker=marker, linestyle=linestyle, color=color, markersize=markersize,  capsize=0, rasterized=rasterized, zorder=11 )
            else:
                ax.errorbar( phi*zoomfactor, y, yerr=yerr_w, marker=marker, linestyle=linestyle, color=color, markersize=markersize,  capsize=0, rasterized=rasterized, zorder=11 )            
            ax.set(xlabel='Phase', ylabel=ylabel, title=inst+', companion '+companion+' only')
    
    
            #::: plot model, phased (if wished)
            if style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_variations']:
                xx = np.linspace( -0.25, 0.75, 1000)
                xx2 = params_median[companion+'_epoch']+np.linspace( -0.25, 0.75, 1000)*params_median[companion+'_period']
                for i in range(samples.shape[0]):
                    s = samples[i,:]
                    p = update_params(s)
#                    p = update_params(s, phased=True)
                    model = rv_fct(p, inst, companion, xx=xx2)[0]
                    ax.plot( xx*zoomfactor, model, 'r-', alpha=alpha, zorder=12 )
            
        
        #----------------------------------------------------------------------
        #::: Photometry
        #----------------------------------------------------------------------
        elif (inst in base.settings['inst_phot']):
            
            #::: remove other companions
            for other_companion in base.settings['companions_phot']:
                if companion!=other_companion:
                    model = flux_fct(params_median, inst, other_companion)
                    y -= (model-1.)
                    
                    
            #::: calculate residuals (if wished)
            if style in ['phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_variations_residuals']:
                model = flux_fct(params_median, inst, companion)
                y -= model
                    
                
            #::: plot data, phased  
            if style in ['phase', 
                         'phase_residuals']:
                dt = 0.002
            elif style in ['phase_variations', 
                           'phase_variations_residuals']:
                dt = 0.01            
            elif style in ['phasezoom', 'phasezoom_occ', 
                           'phasezoom_residuals', 'phasezoom_occ_residuals']: 
                dt = 15./60./24. / params_median[companion+'_period']
                
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=False)    
            if len(x) > 500:
                if style in ['phase_variations', 
                             'phase_variations_residuals']:
                    ax.plot( phase_time*zoomfactor, phase_y, 'b.', color=color, rasterized=rasterized, zorder=11 )                    
                else: 
                    ax.plot( phi*zoomfactor, y, 'b.', color='lightgrey', rasterized=rasterized )
                    ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, marker=marker, linestyle=linestyle, color=color, markersize=markersize,  capsize=0, rasterized=rasterized, zorder=11 )
            else:
                ax.errorbar( phi*zoomfactor, y, yerr=yerr_w, marker=marker, linestyle=linestyle, color=color, markersize=markersize,  capsize=0, rasterized=rasterized, zorder=11 )  
                if base.settings['color_plot']:
                    ax.scatter( phi*zoomfactor, y, c=x, marker='o', rasterized=rasterized, cmap='inferno', zorder=11 )          
            ax.set(xlabel='Phase', ylabel=ylabel, title=inst+', companion '+companion)
    
    
            #::: plot model, phased (if wished)
            if style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_variations']:
                
                if style in ['phase', 'phase_variations']:
                    xx = np.linspace( -0.25, 0.75, 1000)
                    xx2 = params_median[companion+'_epoch'] + np.linspace( -0.25, 0.75, 1000)*params_median[companion+'_period']
                elif style in ['phasezoom']:
                    xx = np.linspace( -10./zoomfactor, 10./zoomfactor, 1000)
                    xx2 = params_median[companion+'_epoch'] + np.linspace( -10./zoomfactor, 10./zoomfactor, 1000)*params_median[companion+'_period']
                elif style in ['phasezoom_occ']:
                    
                    xx = np.linspace( (-10.+zoomfactor/2.)/zoomfactor, (10.+zoomfactor/2.)/zoomfactor, 1000)
                    xx2 = params_median[companion+'_epoch'] + np.linspace( (-10.+zoomfactor/2.)/zoomfactor, (10.+zoomfactor/2.)/zoomfactor, 1000)*params_median[companion+'_period']
    
                for i in range(samples.shape[0]):
                    s = samples[i,:]
                    p = update_params(s)
#                    p = update_params(s, phased=True)
                    model = flux_fct(p, inst, companion, xx=xx2) #evaluated on xx (!)
                    ax.plot( xx*zoomfactor, model, 'r-', alpha=alpha, zorder=12 )
             
        
        #::: x-zoom?        
        if style in ['phasezoom',
                     'phasezoom_residuals']:
                ax.set( xlim=[-4,4], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )
        elif style in ['phasezoom_occ',
                       'phasezoom_occ_residuals']:
                ax.set( xlim=[-4+zoomfactor/2.,4+zoomfactor/2.], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )
        
        
        #::: y-zoom onto phase variations
        elif style in ['phasezoom_occ']:
                ax.set( ylim=[0.999,1.0005] )
       
        if style in ['phase_variation', 
                     'phase_variation_residuals']:
                ax.set( ylim=[0.9999,1.0001] )




    
###############################################################################
#::: plot individual transits
###############################################################################
def afplot_per_transit(samples, inst, companion, base=None, rasterized=True, marker='.', linestyle='none', color='b', markersize=8):
        
    if base==None:
        base = config.BASEMENT
        
    if inst in base.settings['inst_phot']:
        key = 'flux'
        ylabel = 'Realtive Flux'
        baseline_plus = 1.
    elif inst in base.settings['inst_rv']:
        key = 'rv'   
        ylabel = 'RV (km/s)'
        
    if samples.shape[0]==1:
        alpha = 1.
    else:
        alpha = 0.1
        
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    width = 8./24.
    x = base.data[inst]['time']
    y = 1.*base.data[inst][key]
    yerr_w = calculate_yerr_w(params_median, inst, key)
    
    tmid_observed_transits = get_tmid_observed_transits(x, params_median[companion+'_epoch'], params_median[companion+'_period'], width)
    N_transits = len(tmid_observed_transits)
    
    fig, axes = plt.subplots(N_transits,1,figsize=(6,4*N_transits),sharey=True)
    
    if N_transits>0:
        axes = np.atleast_1d(axes)
        axes[0].set(title=inst)
        
        for i, t in enumerate(tmid_observed_transits):
            ax = axes[i]
            
            #::: mark data
            ind = np.where((x >= (t - width/2.)) & (x <= (t + width/2.)))[0]
            
            #::: plot model
            ax.errorbar(x[ind], y[ind], yerr=yerr_w[ind], marker=marker, linestyle=linestyle, color=color, markersize=markersize,  capsize=0, rasterized=rasterized )  
            ax.set(xlabel='Time (BJD)', ylabel=ylabel)
            
            #::: plot model + baseline, not phased
            dt = 2./24./60. 
            xx = np.arange(x[ind][0], x[ind][-1]+dt, dt)
            for i in range(samples.shape[0]):
                s = samples[i,:]
                p = update_params(s)
                model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
                baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
                stellar_var = calculate_stellar_var(p, 'all', key, xx=xx) #evaluated on xx (!)
                ax.plot( xx, baseline+stellar_var+baseline_plus, 'k-', color='orange', alpha=alpha, zorder=12 )
                ax.plot( xx, model+baseline+stellar_var, 'r-', alpha=alpha, zorder=12 )
            ax.set(xlim=[t-4./24., t+4./24.])
            ax.axvline(t,color='g',lw=2,ls='--')
            
    else:
        warnings.warn('No transit of companion '+companion+' for '+inst+'.')
    
    return fig, axes
            
                


    
###############################################################################
#::: update params with MCMC/NS results
###############################################################################
def get_params_from_samples(samples):
    '''
    read MCMC results and update params
    '''
    theta_median = np.percentile(samples, 50, axis=0)
    theta_ul = np.percentile(samples, 84, axis=0) - theta_median
    theta_ll = theta_median - np.percentile(samples, 16, axis=0)
    params_median = update_params(theta_median)
    params_ll = update_params(theta_ll)
    params_ul = update_params(theta_ul)
    
    return params_median, params_ll, params_ul



###############################################################################
#::: save table
###############################################################################
def save_table(samples, mode):
    '''
    Inputs:
    -------
    samples : array
        posterior samples
    mode : string
        'mcmc' or 'ns'
    '''
    
    params, params_ll, params_ul = get_params_from_samples(samples)
    
    with open( os.path.join(config.BASEMENT.outdir,mode+'_table.csv'), 'w' ) as f:
        f.write('#name,median,lower_error,upper_error,label,unit\n')
        f.write('#Fitted parameters,,,\n')
        for i, key in enumerate(config.BASEMENT.allkeys):
            if key not in config.BASEMENT.fitkeys:
                f.write(key + ',' + str(params[key]) + ',' + '(fixed),(fixed),'+config.BASEMENT.labels[i]+','+config.BASEMENT.units[i]+'\n')
            else:
                f.write(key + ',' + str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]) + ',' + config.BASEMENT.labels[i] + ',' + config.BASEMENT.units[i] + '\n' )
   
        
        
###############################################################################
#::: save Latex table
###############################################################################
def save_latex_table(samples, mode):
    '''
    Inputs:
    -------
    samples : array
        posterior samples
    mode : string
        'mcmc' or 'ns'
    '''
    
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    label = 'none'
    
#    derived_samples['a_AU'] = derived_samples['a']*0.00465047 #from Rsun to AU
        
    with open(os.path.join(config.BASEMENT.outdir,mode+'_latex_table.txt'),'w') as f,\
         open(os.path.join(config.BASEMENT.outdir,mode+'_latex_cmd.txt'),'w') as f_cmd:
            
        f.write('parameter & value & unit & fit/fixed \\\\ \n')
        f.write('\\hline \n')
        f.write('\\multicolumn{4}{c}{\\textit{Fitted parameters}} \\\\ \n')
        f.write('\\hline \n')
        
        for i, key in enumerate(config.BASEMENT.allkeys):
            if key not in config.BASEMENT.fitkeys:                
                value = str(params_median[key])
                f.write(config.BASEMENT.labels[i] + ' & $' + value + '$ & '  + config.BASEMENT.units[i] + '& fixed \\\\ \n')            
                f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{$'+value+'$} %'+label+' = '+value+'\n')

            else:            
                value = latex_printer.round_tex(params_median[key], params_ll[key], params_ul[key])
                f.write(config.BASEMENT.labels[i] + ' & $' + value + '$ & ' + config.BASEMENT.units[i] + '& fit \\\\ \n' )
                f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{$='+value+'$} %'+label+' = '+value+'\n')


    
###############################################################################
#::: show initial guess
###############################################################################
def show_initial_guess(datadir, do_logprint=True, do_plot=True, return_figs=False):
    #::: init
    config.init(datadir)    
    
    #::: show initial guess
    if do_logprint: 
        logprint_initial_guess()
    if do_plot: 
        return plot_initial_guess(return_figs=return_figs)
    
    

###############################################################################
#::: logprint initial guess
###############################################################################
def logprint_initial_guess():
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
            
    Outputs:
    --------
    This will output information into the console, 
    and create a file called datadir/results/initial_guess.pdf
    '''
        
    logprint('\nSettings:')
    logprint('--------------------------')
    for key in config.BASEMENT.settings:
        if config.BASEMENT.settings[key]!='':
            logprint('{0: <30}'.format(key), '{0: <15}'.format(str(config.BASEMENT.settings[key]))) #I hate Python 3, and I hope someone finds this comment...
        else:
            logprint('\n{0: <30}'.format(key))

    logprint('\nParameters:')
    logprint('--------------------------')    
    for i, key in enumerate(config.BASEMENT.params):
        if key in config.BASEMENT.fitkeys: 
            ind = np.where( config.BASEMENT.fitkeys == key )[0][0]
            logprint('{0: <30}'.format(key), '{0: <15}'.format(str(config.BASEMENT.params[key])), '{0: <5}'.format('free'), '{0: <30}'.format(str(config.BASEMENT.bounds[ind])) )
        else: 
            if config.BASEMENT.params[key]!='':
                logprint('{0: <30}'.format(key), '{0: <15}'.format(str(config.BASEMENT.params[key])), '{0: <5}'.format('set'))
            else:
                logprint('\n{0: <30}'.format(key))
    
    logprint('\nExternal priors:')
    logprint('--------------------------')  
    if 'host_density' in config.BASEMENT.external_priors:
        logprint('\nStellar density prior (automatically set):', config.BASEMENT.external_priors['host_density'], '(g cm^-3)')
    else:
        logprint('No external priors defined.')
    
    logprint('\nndim:', config.BASEMENT.ndim)



###############################################################################
#::: plot initial guess
###############################################################################
def plot_initial_guess(return_figs=False):
    
    samples = draw_initial_guess_samples()
    
    if return_figs==False:
        for companion in config.BASEMENT.settings['companions_all']:
            fig, axes = afplot(samples, companion)
            fig.savefig( os.path.join(config.BASEMENT.outdir,'initial_guess_'+companion+'.pdf'), bbox_inches='tight' )
            plt.close(fig)
            
        for companion in config.BASEMENT.settings['companions_phot']:
            for inst in config.BASEMENT.settings['inst_phot']:
                fig, axes = afplot_per_transit(samples, inst, companion)
                fig.savefig( os.path.join(config.BASEMENT.outdir,'initial_guess_per_transit_'+inst+'_'+companion+'.pdf'), bbox_inches='tight' )
                plt.close(fig)
        return None
    
    else:
        fig_list = []
        for companion in config.BASEMENT.settings['companions_all']:
            fig, axes = afplot(samples, companion)
            fig_list.append(fig)
        return fig_list
            
    
    
    
###############################################################################
#::: plot initial guess
###############################################################################
def plot_ttv_results(params_median, params_ll, params_ul):
    for companion in config.BASEMENT.settings['companions_all']:
        fig, axes = plt.subplots()
        axes.axhline(0, color='grey', linestyle='--')
        for i in range(len(config.BASEMENT.data[companion+'_tmid_observed_transits'])):
            axes.errorbar( i+1, params_median[companion+'_ttv_transit_'+str(i+1)]*24*60, 
                           yerr=np.array([[ params_ll[companion+'_ttv_transit_'+str(i+1)]*24*60, params_ul[companion+'_ttv_transit_'+str(i+1)]*24*60 ]]).T, 
                           color=config.BASEMENT.settings[companion+'_color'], fmt='.')
        axes.set(xlabel='Tranist Nr.', ylabel='TTV (mins)')
        fig.savefig( os.path.join(config.BASEMENT.outdir,'ttv_results_'+companion+'.pdf'), bbox_inches='tight' )
        plt.close(fig)
    
    
    
    
###############################################################################
#::: get latex labels
###############################################################################
def get_labels(datadir, as_type='dic'):
    config.init(datadir)
    
    if as_type=='2d_array':
        return config.BASEMENT.labels
        
    if as_type=='dic':
        labels_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.allkeys==key)[0]
            labels_dic[key] = config.BASEMENT.labels[ind][0]
        return labels_dic
    


###############################################################################
#::: top-level user interface
###############################################################################
def get_data(datadir):
    config.init(datadir)
    return config.BASEMENT.data



def get_settings(datadir):
    config.init(datadir)
    return config.BASEMENT.settings



#def get_params(datadir):
#    config.init(datadir)
#    return config.BASEMENT.data