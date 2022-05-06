#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:28:55 2018

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
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import gzip
try:
   import cPickle as pickle
except:
   import pickle
from copy import deepcopy
from dynesty import utils as dyutils
from dynesty import plotting as dyplot
import warnings

#::: allesfitter modules
from . import config
from . import deriver
from .computer import calculate_model, calculate_baseline, calculate_stellar_var
from .general_output import afplot, afplot_per_transit, save_table, save_latex_table, logprint, get_params_from_samples, plot_ttv_results
from .plot_top_down_view import plot_top_down_view
from .utils.colormaputil import truncate_colormap
from .utils.latex_printer import round_tex
from .statistics import residual_stats
                     

    

###############################################################################
#::: draw samples from the ns results (internally in the code)
###############################################################################
def draw_ns_posterior_samples(results, Nsamples=None, as_type='2d_array'):
    '''
    ! posterior samples are drawn as resampled weighted samples !
    ! do not confuse posterior_samples (weighted, resampled) with results['samples'] (unweighted) !
    '''
    weights = np.exp(results['logwt'] - results['logz'][-1])
    np.random.seed(42)
    posterior_samples = dyutils.resample_equal(results['samples'], weights)    
    if Nsamples:
        posterior_samples = posterior_samples[np.random.randint(len(posterior_samples), size=Nsamples)]

    if as_type=='2d_array':
        return posterior_samples
    
    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            posterior_samples_dic[key] = posterior_samples[:,ind].flatten()
        return posterior_samples_dic



###############################################################################
#::: analyse the output from save_ns.pickle file
###############################################################################
def ns_output(datadir):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
            
    Outputs:
    --------
    This will output information into the console, and create a output files 
    into datadir/results/ (or datadir/QL/ if QL==True)    
    '''
    config.init(datadir)
    
    
    #::: security check
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'ns_table.csv')):
        try:
            overwrite = str(input('Nested Sampling output files already exists in '+config.BASEMENT.outdir+'\n'+\
                                  '-----------------------------------------------'+''.join(['-']*len(config.BASEMENT.outdir))+'\n'\
                                  'What do you want to do?\n'+\
                                  '1 : overwrite the output files\n'+\
                                  '2 : abort\n'))
            if (overwrite == '1'):
                print('\n')
                pass
            else:
                raise ValueError('User aborted operation.')
        except EOFError:
            warnings.warn("Nested Sampling output files already existed from a previous run, and were automatically overwritten.")
            pass
    
    
    #::: load the save_ns.pickle
    f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir,'save_ns.pickle.gz'), 'rb')
    results = pickle.load(f)
    f.close()
           
    
    #::: plot the fit
    posterior_samples_for_plot = draw_ns_posterior_samples(results, Nsamples=20) #only 20 samples for plotting
    
    for companion in config.BASEMENT.settings['companions_all']:
        fig, axes = afplot(posterior_samples_for_plot, companion)
        if fig is not None:
            fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_'+companion+'.pdf'), bbox_inches='tight' )       
            plt.close(fig)

    if kwargs_dict is None:
        kwargs_dict = {}
    for companion in config.BASEMENT.settings['companions_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            first_transit = 0
            while (first_transit >= 0):
                kwargs_dict['first_transit'] = first_transit
                fig, axes, last_transit, total_transits = afplot_per_transit(posterior_samples_for_plot, inst, companion,
                                                             kwargs_dict=kwargs_dict)
                fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_per_transit_'+inst+'_'+companion+'_' + str(last_transit) + 'th.pdf'), bbox_inches='tight' )
                plt.close(fig)
                if total_transits > 0 and last_transit < total_transits - 1:
                    first_transit = last_transit
                else:
                    first_transit = -1
            
    
    
    #::: retrieve the results
    posterior_samples = draw_ns_posterior_samples(results)                               # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)     # params drawn form these posterior_samples
    
    
    #::: output the results
    logprint('\nResults:')
    logprint('----------')
#    print(results.summary())
    logZdynesty = results.logz[-1]                                                       # value of logZ
    logZerrdynesty = results.logzerr[-1]                                                 # estimate of the statistcal uncertainty on logZ
    logprint('log(Z) = {} +- {}'.format(logZdynesty, logZerrdynesty))
    logprint('Nr. of posterior samples: {}'.format(len(posterior_samples)))
    
    
    #::: make pretty titles for the plots  
    labels, units = [], []
    for i,l in enumerate(config.BASEMENT.fitlabels):
        labels.append( str(config.BASEMENT.fitlabels[i]) )
        units.append( str(config.BASEMENT.fitunits[i]) )
        
    results2 = deepcopy(results) # results.copy() does not work anymore since dynesty 1.2                 
    params_median2, params_ll2, params_ul2 = params_median.copy(), params_ll.copy(), params_ul.copy()     # params drawn form these posterior_samples; only needed for plots (subtract epoch offset)  
    fittruths2 = config.BASEMENT.fittruths.copy()
    for companion in config.BASEMENT.settings['companions_all']:
        
        if companion+'_epoch' in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==companion+'_epoch')[0][0]
            results2['samples'][:,ind] -= int(params_median[companion+'_epoch'])                #np.round(params_median[companion+'_epoch'],decimals=0)
            units[ind] = str(units[ind]+'-'+str(int(params_median[companion+'_epoch']))+'d')    #np.format_float_positional(params_median[companion+'_epoch'],0)+'d')
            fittruths2[ind] -= int(params_median[companion+'_epoch'])
            params_median2[companion+'_epoch'] -= int(params_median[companion+'_epoch'])
                

    for i,l in enumerate(labels):
        if len( units[i].strip(' ') ) > 0:
            labels[i] = str(labels[i]+' ('+units[i]+')')
        
        
    #::: traceplot    
    cmap = truncate_colormap( 'Greys', minval=0.2, maxval=0.8, n=256 )
    tfig, taxes = dyplot.traceplot(results2, labels=labels, quantiles=[0.16, 0.5, 0.84], truths=fittruths2, post_color='grey', trace_cmap=[cmap]*config.BASEMENT.ndim, trace_kwargs={'rasterized':True})
    plt.tight_layout()
    
    
    #::: cornerplot
    # ndim = results2['samples'].shape[1]
    fontsize = np.min(( 24. + 0.5*config.BASEMENT.ndim, 40 ))
    try:
        cfig, caxes = dyplot.cornerplot(results2, labels=labels, span=[0.997 for i in range(config.BASEMENT.ndim)], quantiles=[0.16, 0.5, 0.84], truths=fittruths2, hist_kwargs={'alpha':0.25,'linewidth':0,'histtype':'stepfilled'}, 
                                        label_kwargs={"fontsize":fontsize, "rotation":45, "horizontalalignment":'right'})
    except:
        logprint('! WARNING')
        logprint('Dynesty corner plot could not be created. Please contact maxgue@mit.edu.')
        cfig, caxes = plt.subplots(config.BASEMENT.ndim,config.BASEMENT.ndim,figsize=(2*config.BASEMENT.ndim,2*config.BASEMENT.ndim))
        
        
    #::: runplot
#    rfig, raxes = dyplot.runplot(results)
#    rfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_run.jpg'), dpi=100, bbox_inches='tight' )
#    plt.close(rfig)
    
    
    #::: set allesfitter titles and labels
    for i, key in enumerate(config.BASEMENT.fitkeys): 
        
        value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
        ctitle = r'' + labels[i] + '\n' + r'$=' + value + '$'
        ttitle = r'' + labels[i] + r'$=' + value + '$'
        if len(config.BASEMENT.fitkeys)>1:
            # caxes[i,i].set_title(ctitle)
            caxes[i,i].set_title(ctitle, fontsize=fontsize, rotation=45, horizontalalignment='left')
            taxes[i,1].set_title(ttitle)
            for i in range(caxes.shape[0]):
                for j in range(caxes.shape[1]):
                    caxes[i,j].xaxis.set_label_coords(0.5, -0.5)
                    caxes[i,j].yaxis.set_label_coords(-0.5, 0.5)
        
                    if i==(caxes.shape[0]-1): 
                        fmt = ScalarFormatter(useOffset=False)
                        caxes[i,j].xaxis.set_major_locator(MaxNLocator(nbins=3))
                        caxes[i,j].xaxis.set_major_formatter(fmt)
                    if (i>0) and (j==0):
                        fmt = ScalarFormatter(useOffset=False)
                        caxes[i,j].yaxis.set_major_locator(MaxNLocator(nbins=3))
                        caxes[i,j].yaxis.set_major_formatter(fmt)
                        
                    for tick in caxes[i,j].xaxis.get_major_ticks(): tick.label.set_fontsize(24) 
                    for tick in caxes[i,j].yaxis.get_major_ticks(): tick.label.set_fontsize(24)    
        else:
            caxes[i,i].set_title(ctitle)
            taxes[1].set_title(ttitle)
            caxes[i,i].xaxis.set_label_coords(0.5, -0.5)
            caxes[i,i].yaxis.set_label_coords(-0.5, 0.5)
               
            
    #::: save and close the trace- and cornerplot
    tfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_trace.pdf'), bbox_inches='tight' )
    plt.close(tfig)
    cfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_corner.pdf'), bbox_inches='tight' )
    plt.close(cfig)


    #::: save the tables
    save_table(posterior_samples, 'ns')
    save_latex_table(posterior_samples, 'ns')
    

    #::: derive values (using stellar parameters from params_star.csv)
    deriver.derive(posterior_samples, 'ns')
    
    
    #::: check the residuals
    for inst in config.BASEMENT.settings['inst_all']:
        if inst in config.BASEMENT.settings['inst_phot']: key='flux'
        elif inst in config.BASEMENT.settings['inst_rv']: key='rv'
        model = calculate_model(params_median, inst, key)
        baseline = calculate_baseline(params_median, inst, key)
        stellar_var = calculate_stellar_var(params_median, inst, key)
        residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
        residual_stats(residuals)
    
    
    #::: make top-down orbit plot (using stellar parameters from params_star.csv)
    try:
        params_star = np.genfromtxt( os.path.join(config.BASEMENT.datadir,'params_star.csv'), delimiter=',', names=True, dtype=None, encoding='utf-8', comments='#' )
        fig, ax = plot_top_down_view(params_median, params_star)
        fig.savefig( os.path.join(config.BASEMENT.outdir,'top_down_view.pdf'), bbox_inches='tight' )
        plt.close(fig)        
    except:
        logprint('\nOrbital plots could not be produced.')
    
    
    #::: plot TTV results (if wished for)
    if config.BASEMENT.settings['fit_ttvs'] == True:
        plot_ttv_results(params_median, params_ll, params_ul)
    
    
    #::: clean up
    logprint('\nDone. For all outputs, see', config.BASEMENT.outdir)
    
    
    #::: return a nerdy quote
    try:
        with open(os.path.join(os.path.dirname(__file__), 'utils', 'quotes.txt')) as dataset:
            return(np.random.choice([l for l in dataset]))
    except:
        return('42')
    
    

def ns_derive(datadir): #emergency function if matplotlib and Mac OSX crash
    posterior_samples = get_ns_posterior_samples(datadir, as_type='2d_array')
    deriver.derive(posterior_samples, 'ns')
    
    
    
###############################################################################
#::: get NS samples (for top-level user)
###############################################################################
def get_ns_posterior_samples(datadir, Nsamples=None, as_type='dic'):
    config.init(datadir)
    
    try:
        f = gzip.GzipFile(os.path.join(datadir,'results','save_ns.pickle.gz'), 'rb')
        results = pickle.load(f)
        f.close()
        
    except:
        with open(os.path.join(datadir,'results','save_ns.pickle'),'rb') as f:
            results = pickle.load(f)    

    return draw_ns_posterior_samples(results, Nsamples=Nsamples, as_type=as_type)
    
    
 
###############################################################################
#::: get NS params (for top-level user)
###############################################################################   
def get_ns_params(datadir):
    posterior_samples = get_ns_posterior_samples(datadir, Nsamples=None, as_type='2d_array')  # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)     # params drawn form these posterior_samples
    return params_median, params_ll, params_ul
