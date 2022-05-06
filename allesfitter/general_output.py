#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:10:51 2018

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
                     calculate_yerr_w,\
                     flux_subfct_sinusoidal_phase_curves
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
        
    for inst in config.BASEMENT.settings['inst_rv2']:
        ax = axes[1]
        ax.plot(config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['rv2'], marker='.', ls='none', label=inst)
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
            ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, ls='none', marker='o', ms=8, color=color, capsize=0, zorder=11 )
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
    
    print('Plotting collage for companion', companion+'...')
    
    if config.BASEMENT.settings['fit_ttvs'] is False:
      
        N_inst = len(config.BASEMENT.settings['inst_all'])
        
        if 'do_not_phase_fold' in config.BASEMENT.settings and config.BASEMENT.settings['do_not_phase_fold']:
            fig, axes = plt.subplots(N_inst,1,figsize=(6*1,4*N_inst))
            styles = ['full']
        elif config.BASEMENT.settings['phase_curve']:
            fig, axes = plt.subplots(N_inst,5,figsize=(6*5,4*N_inst))
            styles = ['full','phase','phase_curve','phasezoom','phasezoom_occ']
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
                #::: don't zoom onto RV data (actually, let's do it, for the RM effects' sake #yolo #2fast2rm)
                # if ('zoom' in style) & (inst in config.BASEMENT.settings['inst_rv']):
                #     axes[i,j].axis('off')
                #::: don't plot if the companion is not covered by an instrument
                if (inst in config.BASEMENT.settings['inst_phot']) & (companion not in config.BASEMENT.settings['companions_phot']):
                    axes[i,j].axis('off')
                #::: don't plot if the companion is not covered by an instrument
                elif (inst in config.BASEMENT.settings['inst_rv']) & (companion not in config.BASEMENT.settings['companions_rv']):
                    axes[i,j].axis('off')
                else:
                    plot_1(axes[i,j], samples, inst, companion, style)
    
        plt.tight_layout()
        return fig, axes

    else:
        return None, None
    
    
    
###############################################################################
#::: guesstimate median values for plotting stuff
###############################################################################
def guesstimator(params_median, companion, base=None):
    
    if base==None:
        base = config.BASEMENT
        
    try:
        #==========================================================================
        # guesstimate the median e, omega, R_star_over_a, and b_tra for below
        #==========================================================================
        e = params_median[companion+'_f_s']**2 + params_median[companion+'_f_c']**2
        w = np.mod( np.arctan2(params_median[companion+'_f_s'], params_median[companion+'_f_c']), 2*np.pi) #in rad, from 0 to 2*pi
        R_star_over_a = params_median[companion+'_rsuma'] / (1. + params_median[companion+'_rr'])
        eccentricity_correction_b_tra = ( (1. - e**2) / ( 1. + e*np.sin(w) ) )
        b_tra = (1./R_star_over_a) * params_median[companion+'_cosi'] * eccentricity_correction_b_tra
            
                
        #==========================================================================
        # guesstimate the primary eclipse / transit duration (T14; total duration)
        #==========================================================================
        eccentricity_correction_T_tra = ( np.sqrt(1. - e**2) / ( 1. + e*np.sin(w) ) )
        T_tra_tot = params_median[companion+'_period'] / np.pi * 24. \
                    * np.arcsin( R_star_over_a \
                                 * np.sqrt( (1. + params_median[companion+'_rr'])**2 - b_tra**2 ) \
                                 / np.sin( np.arccos(params_median[companion+'_cosi'])) ) \
                    * eccentricity_correction_T_tra #in h
        
        
        #==========================================================================
        # dynamically set the x-axis zoom window to 3 * T_tra_tot
        #==========================================================================
        if not np.isnan(T_tra_tot):
            zoomwindow = 3 * T_tra_tot #in h
        else:
            zoomwindow = base.settings['zoom_window'] * 24. #user input is in days, convert here to hours


        #==========================================================================
        # dynamically set the y-axis zoom window to [1.-2.*depth, 1.+depth]   
        #==========================================================================
        depth = (params_median[companion+'_rr'])**2
        y_zoomwindow = [1.-2.*depth, 1.+depth]        
        
        
        #==========================================================================
        # guesstimate where the secondary eclipse / occultation is
        #==========================================================================
        phase_shift = 0.5 * (1. + 4./np.pi * e * np.cos(w)) #in phase units; approximation from Winn2010
    
        return zoomwindow, y_zoomwindow, phase_shift #in h; in phase units

    except:
        return 8., [0.98,1.02], 0. #in h; in rel. flux; in phase units



###############################################################################
#::: plot_1 (helper function)
###############################################################################
def plot_1(ax, samples, inst, companion, style, 
           base=None, dt=None,
           zoomwindow=None, force_binning=False,
           kwargs_data=None,
           kwargs_model=None,
           kwargs_ax=None):
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
        
    style: str
        'full' / 'per_transit' / 'phase' / 'phasezoom' / 'phasezoom_occ' /'phase_curve'
        'full_residuals' / 'phase_residuals' / 'phasezoom_residuals' / 'phasezoom_occ_residuals' / 'phase_curve_residuals'
    
    zoomwindow: int or float
        the full width of the window to zoom into (in hours)
        default: 8 hours
    
    base: a BASEMENT class object
        (for internal use only)
        
    dt : float
        time steps on which the model should be evaluated for plots
        in days
        default for style='full': 2 min for <1 day of data; 30 min for >1 day of data.
        
    Notes:
    ------
    yerr / epoch / period: 
        come either from
        a) the initial_guess value or 
        b) the MCMC median,
        depending on what is plotted (i.e. not from individual samples)

    '''

    #==========================================================================
    #::: interpret input
    #==========================================================================
    if base==None:
        base = config.BASEMENT
    
    if samples is not None:
        params_median, params_ll, params_ul = get_params_from_samples(samples)
    
    if kwargs_data is None: kwargs_data = {}
    if 'label' not in kwargs_data: kwargs_data['label'] = inst
    if 'marker' not in kwargs_data: kwargs_data['marker'] = '.'
    if 'markersize' not in kwargs_data: kwargs_data['markersize'] = 8.
    if 'ls' not in kwargs_data: kwargs_data['ls'] = 'none'
    if 'color' not in kwargs_data: kwargs_data['color'] = 'b'
    if 'alpha' not in kwargs_data: kwargs_data['alpha'] = 1.
    if 'rasterized' not in kwargs_data: kwargs_data['rasterized'] = True
    
    if kwargs_model is None: kwargs_model = {}
    if 'marker' not in kwargs_model: kwargs_model['marker'] = 'none'
    if 'markersize' not in kwargs_model: kwargs_model['markersize'] = 0.
    if 'ls' not in kwargs_model: kwargs_model['ls'] = '-'
    if 'color' not in kwargs_model: kwargs_model['color'] = 'r'
    if 'alpha' not in kwargs_model: kwargs_model['alpha'] = 1.
    
    if kwargs_ax is None: kwargs_ax = {}
    if 'title' not in kwargs_ax: kwargs_ax['title'] = None
    if 'xlabel' not in kwargs_ax: kwargs_ax['xlabel'] = None
    if 'ylabel' not in kwargs_ax: kwargs_ax['ylabel'] = None
    
    timelabel = 'Time' #removed feature
    
    
    #==========================================================================
    #::: helper fct
    #==========================================================================
    def set_title(title1):
        if kwargs_ax['title'] is None: return title1
        else: return kwargs_ax['title']
    
    
    #==========================================================================
    #::: do stuff
    #==========================================================================
    if inst in base.settings['inst_phot']:
        key='flux'
        baseline_plus = 1.
        if style in ['full']:
            ylabel = 'Relative Flux'
        elif style in ['full_minus_offset']:
            ylabel = 'Relative Flux - Offset'
        elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_curve']:
            ylabel = 'Relative Flux - Baseline'
        elif style in ['full_residuals', 'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_curve_residuals']:
            ylabel = 'Residuals'
            
    elif inst in base.settings['inst_rv']:
        key='rv'
        baseline_plus = 0.
        if style in ['full']:
            ylabel = 'RV (km/s)'
        elif style in ['full_minus_offset']:
            ylabel = 'RV (km/s) - Offset'
        elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_curve']:
            ylabel = 'RV (km/s) - Baseline'
        elif style in ['full_residuals', 'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_curve_residuals']:
            ylabel = 'Residuals'
            
    elif inst in base.settings['inst_rv2']:
        key='rv2'
        baseline_plus = 0.
        if style in ['full']:
            ylabel = 'RV (km/s)'
        elif style in ['full_minus_offset']:
            ylabel = 'RV (km/s) - Offset'
        elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_curve']:
            ylabel = 'RV (km/s) - Baseline'
        elif style in ['full_residuals', 'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_curve_residuals']:
            ylabel = 'Residuals'
        
    else:
        raise ValueError('inst should be: inst_phot, inst_rv, or inst_rv2...')
    
    
    if samples is not None:
        if samples.shape[0]==1:
            alpha = 1.
        else:
            alpha = 0.1
        

    zoomwindow, y_zoomwindow, phase_shift = guesstimator(params_median, companion, base=base)
        
 
    #==========================================================================
    # full time series, not phased
    # plot the 'undetrended' data
    # plot each sampled model + its baseline 
    #==========================================================================
    if style in ['full', 'full_minus_offset', 'full_residuals']:
        
        #::: set it up
        x = base.data[inst]['time']
        
        if timelabel=='Time_since':
            x = np.copy(x)
            objttime = Time(x, format='jd', scale='utc')
            xsave = np.copy(x)
            x -= x[0]

        y = 1.*base.data[inst][key]
        yerr_w = calculate_yerr_w(params_median, inst, key)
        
        
        #::: remove offset only (if wished)
        if style in ['full_minus_offset']:
            baseline = calculate_baseline(params_median, inst, key)
            y -= np.median(baseline)
            
            
        #::: calculate residuals (if wished)
        if style in ['full_residuals']:
            model = calculate_model(params_median, inst, key)
            baseline = calculate_baseline(params_median, inst, key)
            stellar_var = calculate_stellar_var(params_median, 'all', key, xx=x)
            y -= model+baseline+stellar_var
            
            
        #::: plot data, not phase        
#        ax.errorbar(base.fulldata[inst]['time'], base.fulldata[inst][key], yerr=np.nanmedian(yerr_w), marker='.', ls='none', color='lightgrey', zorder=-1, rasterized=True ) 
        # ax.errorbar(x, y, yerr=yerr_w, marker=kwargs_data['marker'], markersize=kwargs_data['markersize'], ls=kwargs_data['ls'], color=kwargs_data['color'], alpha=kwargs_data['alpha'], capsize=0, rasterized=kwargs_data['rasterized'] )  
        ax.errorbar(x, y, yerr=yerr_w, capsize=0, **kwargs_data)  
        if base.settings['color_plot']:
            ax.scatter(x, y, c=x, marker='o', rasterized=kwargs_data['rasterized'], cmap='inferno', zorder=11 ) 
            
        if timelabel=='Time_since':
            ax.set(xlabel='Time since %s [days]' % objttime[0].isot[:10], ylabel=ylabel, title=set_title(inst))
        elif timelabel=='Time':
            ax.set(xlabel='Time (BJD)', ylabel=ylabel, title=set_title(inst))
            
            
        #::: plot model + baseline, not phased
        if (style in ['full','full_minus_offset']) and (samples is not None):
            
            #if <1 day of photometric data: plot with 2 min resolution
            if dt is None:
                if ((x[-1] - x[0]) < 1): 
                    dt = 2./24./60. 
                #else: plot with 30 min resolution
                else: 
                    dt = 30./24./60. 
                    
            if key == 'flux':
                xx_full = np.arange( x[0], x[-1]+dt, dt)
                N_points_per_chunk = 48
                N_chunks = int(1.*len(xx_full)/N_points_per_chunk)+2
                if N_chunks < 60:
                    for i_chunk in tqdm(range(N_chunks)):
                        xx = xx_full[i_chunk*N_points_per_chunk:(i_chunk+1)*N_points_per_chunk] #plot in chunks of 48 points (1 day)
                        if len(xx)>0 and any( (x>xx[0]) & (x<xx[-1]) ): #plot only where there is data
                            for i in range(samples.shape[0]):
                                s = samples[i,:]
                                p = update_params(s)
                                model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
                                baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
                                if style in ['full_minus_offset']:
                                    baseline -= np.median(baseline)
                                stellar_var = calculate_stellar_var(p, 'all', key, xx=xx) #evaluated on xx (!)
                                ax.plot( xx, baseline+stellar_var+baseline_plus, marker=None, ls='-', color='orange', alpha=alpha, zorder=12 )
                                ax.plot( xx, model+baseline+stellar_var, 'r-', alpha=alpha, zorder=12 )
                else:
                    ax.text(0.05, 0.95, '(The model is not plotted here because the\nphotometric data spans more than 60 days)', fontsize=10, va='top', ha='left', transform=ax.transAxes)  
            elif key in ['rv', 'rv2']: 
                xx = np.arange( x[0], x[-1]+dt, dt)
                for i in range(samples.shape[0]):
                    s = samples[i,:]
                    p = update_params(s)
                    model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
                    baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
                    if style in ['full_minus_offset']:
                        baseline -= np.median(baseline)                    
                    stellar_var = calculate_stellar_var(p, 'all', key, xx=xx) #evaluated on xx (!)
                    ax.plot( xx, baseline+stellar_var+baseline_plus, marker=None, ls='-', color='orange', alpha=alpha, zorder=12 )
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
    elif style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_curve',
                   'phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_curve_residuals']:
        
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
        if (inst in base.settings['inst_rv']) or (inst in base.settings['inst_rv2']):
            
            #::: get key
            if (inst in base.settings['inst_rv']): i_return = 0
            elif (inst in base.settings['inst_rv2']): i_return = 1
              
                
            #::: remove other companions
            for other_companion in base.settings['companions_rv']:
                if companion!=other_companion:
                    model = rv_fct(params_median, inst, other_companion)[i_return]
                    y -= model
            
            
            #::: calculate residuals (if wished)
            if style in ['phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_curve_residuals']:
                model = rv_fct(params_median, inst, companion)[i_return]
                y -= model
                
                
            #::: plot data, phased        
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = 0.002, ferr_type='meansig', ferr_style='sem', sigmaclip=False)    
            if (len(x) > 500) or force_binning:
                ax.plot( phi*zoomfactor, y, marker='.', ls=None, color='lightgrey', rasterized=kwargs_data['rasterized'] ) #don't allow any other kwargs_data here
                ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, capsize=0, zorder=11, **kwargs_data )
            else:
                ax.errorbar( phi*zoomfactor, y, yerr=yerr_w, capsize=0, zorder=11, **kwargs_data )      
            ax.set(xlabel='Phase', ylabel=ylabel, title=set_title(inst+', companion '+companion+' only'))
    
    
            #::: plot model, phased (if wished)
            if (style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_curve']) and (samples is not None):
                xx = np.linspace( -0.25, 0.75, 1000)
                xx2 = params_median[companion+'_epoch']+np.linspace( -0.25, 0.75, 1000)*params_median[companion+'_period']
                for i in range(samples.shape[0]):
                    s = samples[i,:]
                    p = update_params(s)
#                    p = update_params(s, phased=True)
                    model = rv_fct(p, inst, companion, xx=xx2)[i_return]
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
            if style in ['phase_residuals', 'phasezoom_residuals', 'phasezoom_occ_residuals', 'phase_curve_residuals']:
                model = flux_fct(params_median, inst, companion)
                y -= model
                    
                
            #::: plot data, phased  
            if style in ['phase', 
                         'phase_residuals']:
                dt = 0.002
            elif style in ['phase_curve', 
                           'phase_curve_residuals']:
                dt = 0.01            
            elif style in ['phasezoom', 'phasezoom_occ', 
                           'phasezoom_residuals', 'phasezoom_occ_residuals']: 
                dt1 = 15./60./24. / params_median[companion+'_period'] #draw a point every 15 minutes per 1 day orbital period
                dt2 = (zoomwindow/3/24.) / 50. #use 100 points per transit duration
                dt = np.nanmin([dt1,dt2]) #pick the smaller one
                
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=False)    
            buf = phi*zoomfactor
            buf = buf[(buf>-4) & (buf<4)] #just counting the points in the transit window
            if (len(buf) > 80) or force_binning:
                if style in ['phase_curve', 
                             'phase_curve_residuals']:
                    ax.plot( phase_time*zoomfactor, phase_y, 'b.', color=kwargs_data['color'], rasterized=kwargs_data['rasterized'], zorder=11 )                    
                else: 
                    ax.plot( phi*zoomfactor, y, marker='.', ls='none', color='lightgrey', rasterized=kwargs_data['rasterized'], )
                    ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, capsize=0, zorder=11, **kwargs_data )
            else:
                ax.errorbar( phi*zoomfactor, y, yerr=yerr_w, capsize=0, zorder=11, **kwargs_data )
                if base.settings['color_plot']:
                    ax.scatter( phi*zoomfactor, y, c=x, marker='o', rasterized=kwargs_data['rasterized'], cmap='inferno', zorder=11 )          
            ax.set(xlabel='Phase', ylabel=ylabel, title=set_title(inst+', companion '+companion))
    
    
            #::: plot model, phased (if wished)
            if style in ['phase', 'phasezoom', 'phasezoom_occ', 'phase_curve']:
                
                if style in ['phase', 'phase_curve']:
                    xx = np.linspace(-0.25, 0.75, 1000)
                    xx2 = params_median[companion+'_epoch'] + xx * params_median[companion+'_period']
                elif style in ['phasezoom']:
                    xx = np.linspace( -10./zoomfactor, 10./zoomfactor, 1000)
                    xx2 = params_median[companion+'_epoch'] + xx * params_median[companion+'_period']
                elif style in ['phasezoom_occ']:
                    xx = np.linspace( -10./zoomfactor + phase_shift, 10./zoomfactor + phase_shift, 1000 )
                    xx2 = params_median[companion+'_epoch'] + xx * params_median[companion+'_period']
    
                if samples is not None:
                    for i in range(samples.shape[0]):
                        s = samples[i,:]
                        p = update_params(s)
    #                    p = update_params(s, phased=True)
                        model = flux_fct(p, inst, companion, xx=xx2) #evaluated on xx (!)
                        ax.plot( xx*zoomfactor, model, 'r-', alpha=alpha, zorder=12 )
             
        
        #----------------------------------------------------------------------
        #::: Set axes limits
        #----------------------------------------------------------------------
        #::: x-zoom?
        if style in ['phasezoom',
                     'phasezoom_residuals']:
                ax.set( xlim=[-zoomwindow/2.,zoomwindow/2.], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )
        elif style in ['phasezoom_occ',
                       'phasezoom_occ_residuals']:
                xlower = -zoomwindow/2. + phase_shift*params_median[companion+'_period']*24.
                xupper = zoomwindow/2. + phase_shift*params_median[companion+'_period']*24.
                ax.set( xlim=[xlower, xupper], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )
        
        
        #::: y-zoom onto transit and phase variations
        if style in ['phasezoom']:
            try:
                # buf = phase_y[(phase_time>-zoomwindow/24./2.) & (phase_time<zoomwindow/24./2.)] #TODO: replace with proper eclipse indexing
                # def nanptp(arr): return np.nanmax(arr)-np.nanmin(arr)
                # y0 = np.nanmin(buf)-0.1*nanptp(buf)
                # y1 = np.nanmax(buf)+0.1*nanptp(buf)
                # if y1>y0: ax.set(ylim=[y0,y1])
                ax.set(ylim=y_zoomwindow)
            except:
                pass
            
        if style in ['phasezoom_occ']:
            try:
                buf = phase_y[phase_time>0.25] #TODO: replace with proper eclipse indexing
                def nanptp(arr): return np.nanmax(arr)-np.nanmin(arr)
                y0 = np.nanmin(buf)-0.1*nanptp(buf)
                y1 = np.nanmax(buf)+0.1*nanptp(buf)
                if y1>y0: ax.set(ylim=[y0,y1])
            except:
                pass
                # ax.axis('off')
                # ax.set( ylim=[0.999,1.0005] )
       
        if style in ['phase_curve', 
                     'phase_curve_residuals']:
            try:
                phase_curve_no_dips = flux_subfct_sinusoidal_phase_curves(params_median, inst, companion, np.ones_like(xx), xx=xx)
                y0 = np.min(phase_curve_no_dips)-0.1*np.ptp(phase_curve_no_dips)
                y1 = np.max(phase_curve_no_dips)+0.1*np.ptp(phase_curve_no_dips)
                if y1>y0: ax.set(ylim=[y0,y1])
            except:
                pass
                # ax.axis('off')
                # ax.set( ylim=[0.999,1.001] )




    
###############################################################################
#::: plot individual transits
###############################################################################
def afplot_per_transit(samples, inst, companion, base=None, kwargs_dict=None):
        
    print('Plotting individual transits for companion', companion, 'and instrument', inst+'...')
    
    #==========================================================================
    #::: input
    #==========================================================================
    if base==None: base = config.BASEMENT
    if kwargs_dict is None: kwargs_dict = {}
    # if 'window' not in kwargs_dict: kwargs_dict['window'] = 8./24. # in days
    if 'rasterized' not in kwargs_dict: kwargs_dict['rasterized'] = True
    if 'marker' not in kwargs_dict: kwargs_dict['marker'] = '.'
    if 'ls' not in kwargs_dict: kwargs_dict['ls'] = 'none'
    if 'color' not in kwargs_dict: kwargs_dict['color'] = 'b'
    if 'markersize' not in kwargs_dict: kwargs_dict['markersize'] = 8
    if 'max_transits' not in kwargs_dict: kwargs_dict['max_transits'] = 20
    if 'first_transit' not in kwargs_dict: kwargs_dict['first_transit'] = 0

    
    #==========================================================================
    #::: translate input
    #==========================================================================
    # window = kwargs_dict['window']
    rasterized = kwargs_dict['rasterized']
    marker = kwargs_dict['marker']
    ls = kwargs_dict['ls']
    color = kwargs_dict['color']
    markersize = kwargs_dict['markersize']
    max_transits = kwargs_dict['max_transits']
    first_transit = kwargs_dict['first_transit']

        
    #==========================================================================
    #::: configurations
    #==========================================================================
    if inst in base.settings['inst_phot']:
        key = 'flux'
        ylabel = 'Realtive Flux'
        baseline_plus = 1.
    elif inst in base.settings['inst_rv']:
        key = 'rv'   
        ylabel = 'RV (km/s)'
    elif inst in base.settings['inst_rv2']:
        key = 'rv2'   
        ylabel = 'RV (km/s)'
        
    if samples.shape[0]==1:
        alpha = 1.
    else:
        alpha = 0.1
        
        
    #==========================================================================
    #::: load data and models
    #==========================================================================
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    
    zoomwindow, y_zoomwindow, phase_shift = guesstimator(params_median, companion, base=base)
    zoomwindow /= 24. #in days
    T_tra_tot = zoomwindow/3. #in days
    
    x = base.data[inst]['time']
    y = 1.*base.data[inst][key]
    yerr_w = calculate_yerr_w(params_median, inst, key)
    
    tmid_observed_transits = get_tmid_observed_transits(x, params_median[companion+'_epoch'], params_median[companion+'_period'], T_tra_tot)
    total_transits = len(tmid_observed_transits)
    last_transit = first_transit + max_transits if first_transit + max_transits < len(tmid_observed_transits) else len(tmid_observed_transits)
    tmid_observed_transits = tmid_observed_transits[first_transit:last_transit]
    N_transits = len(tmid_observed_transits)
    
    if N_transits>0:
        fig, axes = plt.subplots(N_transits, 1, figsize=(6,4*N_transits), sharey=True, tight_layout=True)
        axes = np.atleast_1d(axes)
        axes[0].set(title=inst)
        
        for i, t in tqdm(enumerate(tmid_observed_transits),total=N_transits):
            transit_label = first_transit + i
            ax = axes[i]
            
            #::: mark data
            ind = np.where((x >= (t - zoomwindow/2.)) & (x <= (t + zoomwindow/2.)))[0]
            
            #::: plot model
            ax.errorbar(x[ind], y[ind], yerr=yerr_w[ind], marker=marker, ls=ls, color=color, markersize=markersize, alpha=1, capsize=0, rasterized=rasterized)  

            #::: plot model + baseline, not phased
            dt = 2./24./60. #2 min steps; in days
            xx = np.arange(x[ind][0], x[ind][-1]+dt, dt)
            for j in range(samples.shape[0]):
                s = samples[j,:]
                p = update_params(s)
                model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
                baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
                stellar_var = calculate_stellar_var(p, 'all', key, xx=xx) #evaluated on xx (!)
                ax.plot( xx, baseline+stellar_var+baseline_plus, marker=None, ls='-', color='orange', alpha=alpha, zorder=12 )
                ax.plot( xx, model+baseline+stellar_var, 'r-', alpha=alpha, zorder=12 )
            ax.set(xlim=[t-zoomwindow/2., t+zoomwindow/2.])
            ax.axvline(t,color='grey',lw=2,ls='--',label='linear prediction')
            if base.settings['fit_ttvs']==True:
                ax.axvline(t+params_median[companion+'_ttv_transit_'+str(transit_label+1)],color='r',lw=2,ls='--',label='TTV midtime')
            
            #::: axes decoration
            ax.set(xlabel='Time', ylabel=ylabel)
            ax.text(0.95, 0.95, 'Transit '+str(transit_label+1), va='top', ha='right', transform=ax.transAxes)
            
        if base.settings['fit_ttvs']==True:
            axes[0].legend(loc='upper left')
            
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6,4), tight_layout=True)
        axes = np.atleast_1d(axes)
        axes[0].axis('off')
        axes[0].text(0.5, 0.5, 'No transit of companion '+companion+' for '+inst+'.', fontsize=10, va='center', ha='center', transform=axes[0].transAxes)
        # warnings.warn('No transit of companion '+companion+' for '+inst+'.')
    
    return fig, axes, last_transit, total_transits
            
                


    
###############################################################################
#::: update params with MCMC/NS results
###############################################################################
def get_params_from_samples(samples):
    '''
    read MCMC or NS results and update params
    '''
    theta_median = np.nanpercentile(samples, 50, axis=0)
    theta_ul = np.nanpercentile(samples, 84, axis=0) - theta_median
    theta_ll = theta_median - np.nanpercentile(samples, 16, axis=0)
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
                simplename = key.replace("_", "").replace("/", "over").replace("(", "").replace(")", "").replace("1", "one").replace("2", "two").replace("3", "three")
                comment = config.BASEMENT.labels[i] + '$=' + value + '$ ' + config.BASEMENT.units[i]
                comment = comment.replace("$$","")
                f_cmd.write('\\newcommand{\\'+simplename+'}{$'+value+'$} %'+comment+'\n')
                # f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{$'+value+'$} %'+label+' = '+value+'\n')

            else:            
                value = latex_printer.round_tex(params_median[key], params_ll[key], params_ul[key])
                f.write(config.BASEMENT.labels[i] + ' & $' + value + '$ & ' + config.BASEMENT.units[i] + '& fit \\\\ \n' )
                simplename = key.replace("_", "").replace("/", "over").replace("(", "").replace(")", "").replace("1", "one").replace("2", "two").replace("3", "three")
                comment = config.BASEMENT.labels[i] + '$=' + value + '$ ' + config.BASEMENT.units[i]
                comment = comment.replace("$$","")
                f_cmd.write('\\newcommand{\\'+simplename+'}{$'+value+'$} %'+comment+'\n')
                 # f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{$='+value+'$} %'+label+' = '+value+'\n')


    
###############################################################################
#::: show initial guess
###############################################################################
def show_initial_guess(datadir, quiet=False, do_logprint=True, do_plot=True, return_figs=False, kwargs_dict=None):
    #::: init
    config.init(datadir, quiet=quiet)    
    
    #::: show initial guess
    if do_logprint: 
        logprint_initial_guess()
    if do_plot: 
        return plot_initial_guess(return_figs=return_figs, kwargs_dict=kwargs_dict)
    
    

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
def plot_initial_guess(return_figs=False, kwargs_dict=None):
    
    samples = draw_initial_guess_samples()
    
    if return_figs==False:
        for companion in config.BASEMENT.settings['companions_all']:
            fig, axes = afplot(samples, companion)
            if fig is not None:
                fig.savefig( os.path.join(config.BASEMENT.outdir,'initial_guess_'+companion+'.pdf'), bbox_inches='tight' )
                plt.close(fig)
        if kwargs_dict is None:
            kwargs_dict = {}
        for companion in config.BASEMENT.settings['companions_phot']:
            for inst in config.BASEMENT.settings['inst_phot']:
                first_transit = 0
                while(first_transit >= 0):
                    try:
                        kwargs_dict['first_transit'] = first_transit
                        fig, axes, last_transit, total_transits = afplot_per_transit(samples, inst, companion, kwargs_dict=kwargs_dict)
                        fig.savefig( os.path.join(config.BASEMENT.outdir,'initial_guess_per_transit_'+inst+'_'+companion+'_' + str(last_transit) + 'th.pdf'), bbox_inches='tight' )
                        plt.close(fig)
                        if total_transits > 0 and last_transit < total_transits - 1:
                            first_transit = last_transit
                        else:
                            first_transit = -1
                    except Exception as e:
                        first_transit = -1
                        pass
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
        axes.axhline(0, color='grey', ls='--')
        for i in range(len(config.BASEMENT.data[companion+'_tmid_observed_transits'])):
            axes.errorbar( i+1, params_median[companion+'_ttv_transit_'+str(i+1)]*24*60, 
                           yerr=np.array([[ params_ll[companion+'_ttv_transit_'+str(i+1)]*24*60, params_ul[companion+'_ttv_transit_'+str(i+1)]*24*60 ]]).T, 
                           color=config.BASEMENT.settings[companion+'_color'], fmt='.')
        axes.set(xlabel='Transit Nr.', ylabel='TTV (mins)')
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
