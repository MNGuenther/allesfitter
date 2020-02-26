#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:28:55 2018

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
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
#import bzip2
import gzip
try:
   import cPickle as pickle
except:
   import pickle
from dynesty import utils as dyutils
from dynesty import plotting as dyplot
import warnings

#::: allesfitter modules
from . import config
from . import deriver
from .general_output import afplot, afplot_per_transit, save_table, save_latex_table, logprint, get_params_from_samples, plot_ttv_results
from .plot_top_down_view import plot_top_down_view
from .utils.colormaputil import truncate_colormap
from .utils.latex_printer import round_tex
                     

    

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
            overwrite = str(input('Nested Sampling output files already exists in '+config.BASEMENT.outdir+'.\n'+\
                                  'What do you want to do?\n'+\
                                  '1 : overwrite the output files\n'+\
                                  '2 : abort\n'))
            if (overwrite == '1'):
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
        fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_'+companion+'.pdf'), bbox_inches='tight' )
#        f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir,'ns_fit.pickle.gz'), 'wb')
#        pickle.dump((fig,axes), f)
#        f.close()        
        plt.close(fig)

    for companion in config.BASEMENT.settings['companions_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            fig, axes = afplot_per_transit(posterior_samples_for_plot, inst, companion)
            fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_per_transit_'+inst+'_'+companion+'.pdf'), bbox_inches='tight' )
            plt.close(fig)
            
    
    #::: retrieve the results
    posterior_samples = draw_ns_posterior_samples(results)                               # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)     # params drawn form these posterior_samples
    
    #::: output the results
    logprint('\nResults:')
    logprint('--------------------------')
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
        
    results2 = results.copy()                    
    params_median2, params_ll2, params_ul2 = params_median.copy(), params_ll.copy(), params_ul.copy()     # params drawn form these posterior_samples; only needed for plots (subtract epoch offset)  
    fittruths2 = config.BASEMENT.fittruths.copy()
    for companion in config.BASEMENT.settings['companions_all']:
        
        if companion+'_epoch' in config.BASEMENT.fitkeys:
            ind    = np.where(config.BASEMENT.fitkeys==companion+'_epoch')[0][0]
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
    ndim = results2['samples'].shape[1]
    cfig, caxes = dyplot.cornerplot(results2, labels=labels, span=[0.997 for i in range(ndim)], quantiles=[0.16, 0.5, 0.84], truths=fittruths2, hist_kwargs={'alpha':0.25,'linewidth':0,'histtype':'stepfilled'}, 
                                    label_kwargs={"fontsize":32, "rotation":45, "horizontalalignment":'right'})


    #::: runplot
#    rfig, raxes = dyplot.runplot(results)
#    rfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_run.jpg'), dpi=100, bbox_inches='tight' )
#    plt.close(rfig)
    

    #::: set allesfitter titles (OLD)
    # for i, key in enumerate(config.BASEMENT.fitkeys): 
        
    #     value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
    #     ttitle = r'' + labels[i] + r'$=' + value + '$'
    #     ctitle = r'' + labels[i] + '\n' + r'$=' + value + '$'
    #     if len(config.BASEMENT.fitkeys)>1:
    #         caxes[i,i].set_title(ctitle)
    #         taxes[i,1].set_title(ttitle)
    #         for i in range(caxes.shape[0]):
    #             for j in range(caxes.shape[1]):
    #                 caxes[i,j].xaxis.set_label_coords(0.5, -0.5)
    #                 caxes[i,j].yaxis.set_label_coords(-0.5, 0.5)
    #     else:
    #         caxes.set_title(ctitle)
    #         taxes[1].set_title(ttitle)
    #         caxes.xaxis.set_label_coords(0.5, -0.5)
    #         caxes.yaxis.set_label_coords(-0.5, 0.5)
    
    
    #::: set allesfitter titles and labels
    for i, key in enumerate(config.BASEMENT.fitkeys): 
        
        value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
        ctitle = r'' + labels[i] + '\n' + r'$=' + value + '$'
        ttitle = r'' + labels[i] + r'$=' + value + '$'
        if len(config.BASEMENT.fitkeys)>1:
            # caxes[i,i].set_title(ctitle)
            caxes[i,i].set_title(ctitle, fontsize=32, rotation=45, horizontalalignment='left')
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
            caxes.set_title(ctitle)
            taxes[1].set_title(ttitle)
            caxes.xaxis.set_label_coords(0.5, -0.5)
            caxes.yaxis.set_label_coords(-0.5, 0.5)
               
            
    #::: save and close the trace- and cornerplot
    tfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_trace.pdf'), bbox_inches='tight' )
    plt.close(tfig)
    cfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_corner.pdf'), bbox_inches='tight' )
    plt.close(cfig)


    #::: save the tables
    save_table(posterior_samples, 'ns')
    save_latex_table(posterior_samples, 'ns')
    

    #::: derive values (using stellar parameters from params_star.csv)
    if os.path.exists(os.path.join(config.BASEMENT.datadir,'params_star.csv')):
        deriver.derive(posterior_samples, 'ns')
    else:
        print('File "params_star.csv" not found. Cannot derive final parameters.')
   
    
    #::: make top-down orbit plot (using stellar parameters from params_star.csv)
    if os.path.exists( os.path.join(config.BASEMENT.datadir,'params_star.csv') ):
        params_star = np.genfromtxt( os.path.join(config.BASEMENT.datadir,'params_star.csv'), delimiter=',', names=True, dtype=None, encoding='utf-8', comments='#' )
        try:
            fig, ax = plot_top_down_view(params_median, params_star)
            fig.savefig( os.path.join(config.BASEMENT.outdir,'top_down_view.pdf'), bbox_inches='tight' )
            plt.close(fig)        
        except:
            warnings.warn('Orbital plots could not be produced.')
    else:
        print('File "params_star.csv" not found. Cannot derive final parameters.')
    
    
    #::: plot TTV results (if wished for)
    if config.BASEMENT.settings['fit_ttvs'] == True:
        plot_ttv_results(params_median, params_ll, params_ul)
    
    
    #::: clean up
    logprint('Done. For all outputs, see', config.BASEMENT.outdir)
    
    
    #::: return a nerdy quote
    try:
        with open(os.path.join(os.path.dirname(__file__), 'utils', 'quotes.txt')) as dataset:
            return(np.random.choice([l for l in dataset]))
    except:
        return('42')
    
    

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