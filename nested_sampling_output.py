#!/usr/bin/env python2
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
import os
import pickle
from dynesty import utils as dyutils
from dynesty import plotting as dyplot

#::: allesfitter modules
from . import config
from . import deriver
from .general_output import afplot, save_table, save_latex_table, logprint, get_params_from_samples
from .utils.colormaputil import truncate_colormap
from .utils.latex_printer import round_tex
                     

    

###############################################################################
#::: draw samples from the ns results (internally in the code)
###############################################################################
def draw_ns_samples(results, Nsamples=None):
    weights = np.exp(results['logwt'] - results['logz'][-1])
    samples = dyutils.resample_equal(results.samples, weights)    
    if Nsamples:
        samples = samples[np.random.randint(len(samples), size=20)]
    return samples



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
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'ns_fit.jpg')):
        overwrite = raw_input('Nested Sampling output files already exists in '+config.BASEMENT.outdir+'.\n'+\
                              'What do you want to do?\n'+\
                              '1 : overwrite the output files\n'+\
                              '2 : abort\n')
        if (overwrite == '1'):
            pass
        else:
            raise ValueError('User aborted operation.')
    
    #::: load the save_ns.pickle
    with open( os.path.join(config.BASEMENT.outdir,'save_ns.pickle'),'rb' ) as f:
        results = pickle.load(f)
           
        
    #::: plot the fit        
    samples = draw_ns_samples(results, Nsamples=20) #only 20 samples for plotting
    for planet in config.BASEMENT.settings['planets_all']:
        fig, axes = afplot(samples, planet)
        fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_'+planet+'.jpg'), dpi=100, bbox_inches='tight' )
        plt.close(fig)
    
    
    #::: output the results
    logprint('\nResults:')
    logprint('--------------------------')
#    print(results.summary())
    samples = draw_ns_samples(results)    #all samples
#    plt.figure()
#    plt.plot(np.arange(len(samples[:,0])), samples[:,0])
#    plt.show()
#    plt.figure()
#    plt.plot(np.arange(len(samples[:,1])), samples[:,1])
#    plt.show()
    logZdynesty = results.logz[-1]        # value of logZ
    logZerrdynesty = results.logzerr[-1]  # estimate of the statistcal uncertainty on logZ
    logprint('log(Z) = {} +- {}'.format(logZdynesty, logZerrdynesty))
    logprint('Nr. of posterior samples: {}'.format(len(samples)))
    
    #::: plot all the diagnositc plots
#    rfig, raxes = dyplot.runplot(results)
#    rfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_run.jpg'), dpi=100, bbox_inches='tight' )
#    plt.close(rfig)
    
    
    #::: make pretty titles for the plots  
    labels, units = [], []
    for i,l in enumerate(config.BASEMENT.fitlabels):
        labels.append( str(config.BASEMENT.fitlabels[i]) )
        units.append( str(config.BASEMENT.fitunits[i]) )
        
    results2 = results.copy()    
    params_median, params_ll, params_ul = get_params_from_samples(results['samples'])

    for planet in config.BASEMENT.settings['planets_all']:
        
#        if planet+'_period' in config.BASEMENT.fitkeys:
#            ind    = np.where(config.BASEMENT.fitkeys==planet+'_period')[0][0]
#            results2['samples'][:,ind] -= np.round(params_median[planet+'_period'],decimals=3)
#            units[ind] = str(units[ind]+'-'+np.format_float_positional(params_median[planet+'_period'],3)+'d')
            
        if planet+'_epoch' in config.BASEMENT.fitkeys:
            ind    = np.where(config.BASEMENT.fitkeys==planet+'_epoch')[0][0]
            results2['samples'][:,ind] -= int(params_median[planet+'_epoch']) #np.round(params_median[planet+'_epoch'],decimals=0)
            units[ind] = str(units[ind]+'-'+str(int(params_median[planet+'_epoch']))+'d') #np.format_float_positional(params_median[planet+'_epoch'],0)+'d')
            config.BASEMENT.fittruths[ind] -= int(params_median[planet+'_epoch'])
            
    for i,l in enumerate(labels):
        if units[i]!='':
            labels[i] = str(labels[i]+' ('+units[i]+')')
        
    #::: traceplot    
    cmap = truncate_colormap( 'Greys', minval=0.2, maxval=0.8, n=256 )
    tfig, taxes = dyplot.traceplot(results2, labels=labels, truths=config.BASEMENT.fittruths, post_color='grey', trace_cmap=[cmap]*config.BASEMENT.ndim)
    plt.tight_layout()
    
    #::: cornerplot
    cfig, caxes = dyplot.cornerplot(results2, labels=labels, truths=config.BASEMENT.fittruths, hist_kwargs={'alpha':0.25,'linewidth':0,'histtype':'stepfilled'})

    #::: set allesfitter titles
    for i, key in enumerate(config.BASEMENT.fitkeys):    
        params_median, params_ll, params_ul = get_params_from_samples(results2['samples'])
        value = round_tex(params_median[key], params_ll[key], params_ul[key])
        ttitle = r'' + labels[i] + r'$=' + value + '$'
        taxes[i,1].set_title(ttitle)
        ctitle = r'' + labels[i] + '\n' + r'$=' + value + '$'
        caxes[i,i].set_title(ctitle)
    for i in range(caxes.shape[0]):
        for j in range(caxes.shape[1]):
            caxes[i,j].xaxis.set_label_coords(0.5, -0.5)
            caxes[i,j].yaxis.set_label_coords(-0.5, 0.5)
            
    #::: save and close the trace- and cornerplot
    tfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_trace.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(tfig)
    cfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_corner.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(cfig)


    #::: save the tables
    save_table(samples, 'ns')
    save_latex_table(samples, 'ns')
    

    #::: derive values (using stellar parameters from params_star.csv)
    if os.path.exists(os.path.join(config.BASEMENT.datadir,'params_star.csv')):
        deriver.derive(samples, 'ns')
    else:
        print('File "params_star.csv" not found. Cannot derive final parameters.')
    
    
    logprint('Done. For all outputs, see', config.BASEMENT.outdir)
    
    

###############################################################################
#::: get NS samples (for top-level user)
###############################################################################
def get_ns_samples(datadir, Nsamples=None, as_type='dic'):
    config.init(datadir)
    with open(os.path.join(datadir,'results','save_ns.pickle'),'rb') as f:
        results = pickle.load(f)
    samples = draw_ns_samples(results, Nsamples=Nsamples)
    
    if as_type=='2d_array':
        return samples
    
    elif as_type=='dic':
        samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            samples_dic[key] = samples[:,ind].flatten()
        return samples_dic