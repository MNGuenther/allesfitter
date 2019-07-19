#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:18:20 2018

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
import os
import gzip
try:
   import cPickle as pickle
except:
   import pickle
from shutil import copyfile
try:
    import emcee
except:
    pass



#:::: allesfitter modules
from . import config

from .mcmc import mcmc_fit
from .nested_sampling import ns_fit

from . import general_output
from . import nested_sampling_output

from .general_output import show_initial_guess, draw_initial_guess_samples, get_labels, get_data, get_settings
from .nested_sampling_output import get_ns_posterior_samples, get_ns_params, ns_output
from .mcmc_output import get_mcmc_posterior_samples, mcmc_output, draw_mcmc_posterior_samples

from .computer import calculate_model, calculate_baseline, calculate_stellar_var

from .priors import transform_priors
from .priors.estimate_noise import estimate_noise

from .postprocessing.nested_sampling_compare_logZ import ns_plot_bayes_factors
from .postprocessing.plot_violins import ns_plot_violins, mcmc_plot_violins

def GUI():
    allesfitter_path = os.path.dirname( os.path.realpath(__file__) )
    os.system( 'jupyter notebook "' + os.path.join(allesfitter_path,'GUI.ipynb') + '"')


class allesclass():
    def __init__(self,datadir):
        config.init(datadir)
        self.fulldata = config.BASEMENT.fulldata
        self.data = config.BASEMENT.data
        self.settings = config.BASEMENT.settings
        
        self.initial_guess_samples = draw_initial_guess_samples()
        self.initial_guess_params_median = general_output.get_params_from_samples(self.initial_guess_samples)[0]
        
        try:
            self.params_star = config.BASEMENT.params_star
        except:
            pass
        
        try:
            self.external_priors = config.BASEMENT.external_priors
        except:
            pass
        
        try:
            if os.path.exists( os.path.join(config.BASEMENT.outdir,'save_ns.pickle.gz') ):
                f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir,'save_ns.pickle.gz'), 'rb')
                results = pickle.load(f)
                f.close()
                self.posterior_samples = nested_sampling_output.draw_ns_posterior_samples(results) # all weighted posterior_samples
                self.posterior_params = nested_sampling_output.draw_ns_posterior_samples(results, as_type='dic') # all weighted posterior_samples
                self.posterior_params_median, self.posterior_params_ll, self.posterior_params_ul = general_output.get_params_from_samples(self.posterior_samples)
            elif os.path.exists( os.path.join(config.BASEMENT.outdir,'mcmc_save.h5') ):
                copyfile(os.path.join(datadir,'results','mcmc_save.h5'), os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
                reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'), read_only=True )
                self.posterior_samples = draw_mcmc_posterior_samples(reader) # all weighted posterior_samples           
                self.posterior_params = draw_mcmc_posterior_samples(reader, as_type='dic') # all weighted posterior_samples
                self.posterior_params_median, self.posterior_params_ll, self.posterior_params_ul = general_output.get_params_from_samples(self.posterior_samples)
        except:
            pass
        
        
        
    #::: posterior median
    
    def get_posterior_median_model(self, inst, key, xx=None):
        print(self.posterior_params_median)
        return calculate_model(self.posterior_params_median, inst, key, xx=xx)
        
    def get_posterior_median_baseline(self, inst, key, xx=None, model=None):
        return calculate_baseline(self.posterior_params_median, inst, key, xx=xx, model=model)
    
    def get_posterior_median_stellar_var(self, inst, key, xx=None):
        return calculate_stellar_var(self.posterior_params_median, key, xx=xx)
    
    
    
    #::: initial guess
    
    def get_initial_guess_model(self, inst, key, xx=None):
        return calculate_model(self.initial_guess_params_median, inst, key, xx=xx)
    
    def get_initial_guess_baseline(self, inst, key, xx=None, model=None):
        return calculate_baseline(self.initial_guess_params_median, inst, key, xx=xx, model=model)
    
    def get_initial_guess_stellar_var(self, inst, key, xx=None):
        return calculate_stellar_var(self.initial_guess_params_median, key, xx=xx)
    
    
    
    #::: one posterior sample
    
    def get_one_posterior_curve_set(self, inst, key, xx=None, sample_id=None):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            buf[k] = self.posterior_params[k][sample_id]
        return calculate_model(buf, inst, key, xx=xx), calculate_baseline(buf, inst, key, xx=xx), calculate_stellar_var(buf, key, xx=xx)
    
    def get_one_posterior_model(self, inst, key, xx=None, sample_id=None):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            buf[k] = self.posterior_params[k][sample_id]
        return calculate_model(buf, inst, key, xx=xx)

    def get_one_posterior_baseline(self, inst, key, xx=None, sample_id=None):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            buf[k] = self.posterior_params[k][sample_id]
        return calculate_baseline(buf, inst, key, xx=xx)
    
    def get_one_posterior_stellar_var(self, inst, key, xx=None, sample_id=None):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            buf[k] = self.posterior_params[k][sample_id]
        return calculate_stellar_var(buf, key, xx=xx)
    
    
    
#::: version
__version__ = '0.9.1'