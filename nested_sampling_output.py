#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:28:55 2018

@author:
Dr. Maximilian N. Guenther
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
from .general_output import afplot, save_table, save_latex_table, logprint
                     

    

###############################################################################
#::: draw samples from the MCMC save.5 (internally in the code)
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
        fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit'+planet+'.jpg'), dpi=100, bbox_inches='tight' )
        plt.close(fig)
    
    
    #::: output the results
    logprint('\nResults:')
    logprint('--------------------------')
#    print(results.summary())
    samples = draw_ns_samples(results)    #all samples
    plt.figure()
    plt.plot(np.arange(len(samples[:,0])), samples[:,0])
    plt.show()
    plt.figure()
    plt.plot(np.arange(len(samples[:,1])), samples[:,1])
    plt.show()
    logZdynesty = results.logz[-1]        # value of logZ
    logZerrdynesty = results.logzerr[-1]  # estimate of the statistcal uncertainty on logZ
    logprint('Static: log(Z) = {} +- {}'.format(logZdynesty, logZerrdynesty))
    logprint('Nr. of posterior samples: {}'.format(len(samples)))
    
    #::: plot all the diagnositc plots
#    rfig, raxes = dyplot.runplot(results)
#    rfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_run.jpg'), dpi=100, bbox_inches='tight' )
#    plt.close(rfig)
    
    tfig, taxes = dyplot.traceplot(results)
    tfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_trace.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(tfig)
    
    cfig, caxes = dyplot.cornerplot(results)
    cfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_corner.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(cfig)


    #::: save the tables
    save_table(samples, 'ns')
    save_latex_table(samples, 'ns')
    

    #::: derive values (using stellar parameters from params_star.csv)
    try:
        deriver.derive(samples, 'mcmc')
    except:
        print('File "params_star.csv" not found. Cannot derive final parameters.')
    
    
    logprint('Done. For all outputs, see', config.BASEMENT.outdir)
    
    

###############################################################################
#::: get NS samples (for top-level user)
###############################################################################
def get_ns_samples(datadir, Nsamples=None, QL=False, as_type='dic'):
    config.init(datadir, QL=QL)
    with open(os.path.join(datadir,'save_ns.pickle'),'rb') as f:
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