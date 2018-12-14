#!/usr/bin/env python2
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
#import pickle
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 

#::: my modules
from exoworlds.lightcurves import lightcurve_tools as lct

#::: allesfitter modules
from . import config
from .utils import latex_printer
from .computer import update_params,\
                     calculate_model, rv_fct, flux_fct,\
                     calculate_baseline, calculate_yerr_w
                     
                     
                     
                     
###############################################################################
#::: print function that prints into console and logfile at the same time
############################################################################### 
def logprint(*text):
    print(*text)
    original = sys.stdout
    with open( os.path.join(config.BASEMENT.outdir,'logfile_'+config.BASEMENT.now+'.log'), 'a' ) as f:
        sys.stdout = f
        print(*text)
    sys.stdout = original
                     
    
    
###############################################################################
#::: draw samples from the initial guess
###############################################################################
def draw_initial_guess_samples(Nsamples=1):
#    global config.BASEMENT
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
    
    fig, axes = plt.subplots(2,1,figsize=(20,10))
    
    for inst in config.BASEMENT.settings['inst_phot']:
        axes[0].plot(config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['flux'], marker='.', ls='none', label=inst)

    for inst in config.BASEMENT.settings['inst_rv']:
        axes[1].plot(config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['rv'], marker='.', ls='none', label=inst)
        
    axes[0].legend()
    axes[1].legend()
    
    return fig, axes
        


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
def plot_1(ax, samples, inst, companion, style):
    '''
    Inputs:
    -------
        companion : str (optional)
            only needed if style=='_phase' or '_phasezoom'
            None, 'b', 'c', etc.
            
    Notes:
    ------
    yerr / epoch / period: 
        come from the initial_guess value or the MCMC median (not from individual samples)

    '''
#    global config.BASEMENT
    
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    
    if inst in config.BASEMENT.settings['inst_phot']:
        key='flux'
        ylabel='Flux'
    elif inst in config.BASEMENT.settings['inst_rv']:
        key='rv'
        ylabel='RV (km/s)'
    else:
        raise ValueError('inst should be listed in inst_phot or inst_rv...')
    
    
    if samples.shape[0]==1:
        alpha = 1.
    else:
        alpha = 0.1
        

    ###############################################################################
    # full time series, not phased
    # plot the 'undetrended' data
    # plot each sampled model + its baseline 
    ###############################################################################
    if style=='full':
        
        #::: set it up
        x = config.BASEMENT.data[inst]['time']
        y = config.BASEMENT.data[inst][key]
        yerr_w = calculate_yerr_w(params_median, inst, key)
        
        #::: plot data, not phase
        ax.errorbar( x, y, yerr=yerr_w, fmt='b.', capsize=0, rasterized=True )  
        if config.BASEMENT.settings['color_plot']:
            ax.scatter( x, y, c=x, marker='o', rasterized=True, cmap='inferno', zorder=11 ) 
        ax.set(xlabel='Time (d)', ylabel=ylabel, title=inst)
        
        #::: plot model + baseline, not phased
        if ((x[-1] - x[0]) < 1): dt = 2./24./60. #if <1 day of data: plot with 2 min resolution
        else: dt = 30./24./60. #else: plot with 30 min resolution
        xx = np.arange( x[0], x[-1]+dt, dt) 
        for i in range(samples.shape[0]):
            s = samples[i,:]
            p = update_params(s)
            model = calculate_model(p, inst, key, xx=xx) #evaluated on xx (!)
            baseline = calculate_baseline(p, inst, key, xx=xx) #evaluated on xx (!)
            ax.plot( xx, model+baseline, 'r-', alpha=alpha, rasterized=True, zorder=12 )
            
            
    ###############################################################################
    # phased - and optionally zoomed
    # get a 'median' baseline from intial guess value / MCMC median result
    # detrend the data with this 'median' baseline
    # then phase-fold the 'detrended' data
    # plot each phase-folded model (without baseline)
    # TODO: This is not ideal, as we overplot models with different 
    #       epochs/periods/baselines onto a phase-folded plot
    ###############################################################################
    else:
        
        #::: data - baseline_median
        x = config.BASEMENT.data[inst]['time']
        baseline_median = calculate_baseline(params_median, inst, key) #evaluated on x (!)
        y = config.BASEMENT.data[inst][key] - baseline_median
        yerr_w = calculate_yerr_w(params_median, inst, key)
        
        #::: zoom?
        if 'phasezoom' in style: 
            zoomfactor = params_median[companion+'_period']*24.
        else: 
            zoomfactor = 1.
        
        
        #::: if RV
        #::: need to take care of multiple companions
        if (inst in config.BASEMENT.settings['inst_rv']):
            
            for other_companion in config.BASEMENT.settings['companions_rv']:
                if companion!=other_companion:
                    model = rv_fct(params_median, inst, other_companion)[0]
                    y -= model
            
            #data, phased        
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = 0.002, ferr_type='meansig', ferr_style='sem', sigmaclip=False)    
            if len(x) > 500:
                ax.plot( phi*zoomfactor, y, 'b.', color='lightgrey', rasterized=True )
                ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, fmt='b.', capsize=0, rasterized=True, zorder=11 )
            else:
                ax.errorbar( phi*zoomfactor, y, yerr=yerr_w, fmt='b.', capsize=0, rasterized=True, zorder=11 )            
            ax.set(xlabel='Phase', ylabel=r'$\Delta$ '+ylabel, title=inst+', companion '+companion+' only')
    
            #model, phased
            xx = np.linspace( -0.25, 0.75, 1000)
            for i in range(samples.shape[0]):
                s = samples[i,:]
                p = update_params(s, phased=True)
                model = rv_fct(p, inst, companion, xx=xx)[0]
                ax.plot( xx*zoomfactor, model, 'r-', alpha=alpha, rasterized=True, zorder=12 )
            
        
        #::: if photometry
        elif (inst in config.BASEMENT.settings['inst_phot']):
            #data, phased  
            if 'phase_variation' in style:
                dt = 0.01                
#            elif (x[-1] - x[0]) < 1.:
#                dt = 0.00001 #i.e. do not bin
#                print('here')
            else:
                dt = 0.002
                
            phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(x, y, params_median[companion+'_period'], params_median[companion+'_epoch'], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=False)    
            if len(x) > 500:
                if 'phase_variation' not in style: 
                    ax.plot( phi*zoomfactor, y, 'b.', color='lightgrey', rasterized=True )
                    ax.errorbar( phase_time*zoomfactor, phase_y, yerr=phase_y_err, fmt='b.', capsize=0, rasterized=True, zorder=11 )
                else:
                    ax.plot( phase_time*zoomfactor, phase_y, 'b.', rasterized=True, zorder=11 )                    
            else:
                ax.errorbar( phi*zoomfactor, y, yerr=yerr_w, fmt='b.', capsize=0, rasterized=True, zorder=11 )  
                if config.BASEMENT.settings['color_plot']:
                    ax.scatter( phi*zoomfactor, y, c=x, marker='o', rasterized=True, cmap='inferno', zorder=11 )          
            ax.set(xlabel='Phase', ylabel=r'$\Delta$ '+ylabel, title=inst+', companion '+companion)
    
            #model, phased
            if style=='phasezoom':
                xx = np.linspace( -4./zoomfactor, 4./zoomfactor, 1000)
            elif style=='phasezoom_occ':
                xx = np.linspace( (-4.+zoomfactor/2.)/zoomfactor, (4.+zoomfactor/2.)/zoomfactor, 1000)
            else:
                xx = np.linspace( -0.25, 0.75, 1000)
            for i in range(samples.shape[0]):
                s = samples[i,:]
                p = update_params(s, phased=True)
                model = flux_fct(p, inst, companion, xx=xx) #evaluated on xx (!)
                ax.plot( xx*zoomfactor, model, 'r-', alpha=alpha, rasterized=True, zorder=12 )
             
        
        #::: zoom?        
        if 'phasezoom' in style:     ax.set( xlim=[-4,4], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )
        if 'phasezoom_occ' in style: ax.set( xlim=[-4+zoomfactor/2.,4+zoomfactor/2.], ylim=[0.999,1.0005], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )
        
        #::: zoom onto phase variations?
        if 'phase_variation' in style: ax.set( ylim=[0.9999,1.0001] )

    
###############################################################################
#::: update params with MCMC/NS results
###############################################################################
def get_params_from_samples(samples):
    '''
    read MCMC results and update params
    '''

    buf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))                                         
    theta_median = [ item[0] for item in buf ]
    theta_ul = [ item[1] for item in buf ]
    theta_ll = [ item[2] for item in buf ]
    params_median = update_params(theta_median)
    params_ul = update_params(theta_ul)
    params_ll = update_params(theta_ll)
    
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
    
    with open( os.path.join(config.BASEMENT.outdir,mode+'_table.csv'), 'wb' ) as f:
        f.write('############ Fitted parameters ############\n')
        for i, key in enumerate(config.BASEMENT.allkeys):
            if key not in config.BASEMENT.fitkeys:
                f.write(key + ',' + str(params[key]) + ',' + '(fixed),\n')
            else:
                f.write(key + ',' + str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]) + '\n' )
   
        
        
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
        
    with open(os.path.join(config.BASEMENT.outdir,mode+'_latex_table.txt'),'wb') as f,\
         open(os.path.join(config.BASEMENT.outdir,mode+'_latex_cmd.txt'),'wb') as f_cmd:
            
        f.write('parameter & value & unit & fit/fixed \\\\ \n')
        f.write('\\hline \n')
        f.write('\\multicolumn{4}{c}{\\textit{Fitted parameters}} \\\\ \n')
        f.write('\\hline \n')
        
        for i, key in enumerate(config.BASEMENT.allkeys):
            if key not in config.BASEMENT.fitkeys:                
                value = str(params_median[key])
                f.write(config.BASEMENT.labels[i] + ' & $' + value + '$ & '  + config.BASEMENT.units[i] + '& (fixed) \\\\ \n')            
                f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{'+label+'$='+value+'$} \n')

            else:            
                value = latex_printer.round_tex(params_median[key], params_ll[key], params_ul[key])
                f.write(config.BASEMENT.labels[i] + ' & $' + value + '$ & ' + config.BASEMENT.units[i] + '& \\\\ \n' )
                f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{'+label+'$='+value+'$} \n')

    

###############################################################################
#::: show initial guess
###############################################################################
def show_initial_guess(do_logprint=True, initial_guess_plot=True, return_figs=False):
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
#    global config.BASEMENT
            
    if do_logprint:
        
        logprint('\nSettings:')
        logprint('--------------------------')
        for key in config.BASEMENT.settings:
            if config.BASEMENT.settings[key]!='':
                logprint('{0: <30}'.format(key), '{0: <15}'.format(config.BASEMENT.settings[key]))
            else:
                logprint('\n{0: <30}'.format(key))
    
        logprint('\nParameters:')
        logprint('--------------------------')    
        for i, key in enumerate(config.BASEMENT.params):
            if key in config.BASEMENT.fitkeys: 
                ind = np.where( config.BASEMENT.fitkeys == key )[0][0]
                logprint('{0: <30}'.format(key), '{0: <15}'.format(config.BASEMENT.params[key]), '{0: <5}'.format('free'), '{0: <30}'.format(config.BASEMENT.bounds[ind]) )
            else: 
                if config.BASEMENT.params[key]!='':
                    logprint('{0: <30}'.format(key), '{0: <15}'.format(config.BASEMENT.params[key]), '{0: <5}'.format('set'))
                else:
                    logprint('\n{0: <30}'.format(key))
        
        logprint('\nndim:', config.BASEMENT.ndim)
                
        
#    print '\nLikelihoods:'
#    print '--------------------------'
#    print 'lnprior:\t', lnprior(theta_0, bounds)
#    print 'lnlike: \t', lnlike(theta_0, params, fitkeys, settings)
#    print 'lnprob: \t', lnprob(theta_0, bounds, params, fitkeys, settings)  
        

    if initial_guess_plot==True:
        samples = draw_initial_guess_samples()
        
        if return_figs==False:
            for companion in config.BASEMENT.settings['companions_all']:
                    fig, axes = afplot(samples, companion)
                    fig.savefig( os.path.join(config.BASEMENT.outdir,'initial_guess_'+companion+'.jpg'), dpi=100, bbox_inches='tight' )
                    plt.close(fig)
        
        else:
            fig_list = []
            for companion in config.BASEMENT.settings['companions_all']:
                fig, axes = afplot(samples, companion)
                fig_list.append(fig)
            return fig_list
            
    
    
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
    