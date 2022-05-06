#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:18:20 2018

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

#::: modules
import numpy as np
import matplotlib.pyplot as plt
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

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




#:::: allesfitter modules
from . import config

from .mcmc import mcmc_fit
from .nested_sampling import ns_fit

from . import general_output
from . import nested_sampling_output

from .general_output import show_initial_guess, draw_initial_guess_samples, get_labels, get_data, get_settings
from .nested_sampling_output import get_ns_posterior_samples, get_ns_params, ns_output, ns_derive
from .mcmc_output import get_mcmc_posterior_samples, mcmc_output, draw_mcmc_posterior_samples, draw_mcmc_posterior_samples_at_maximum_likelihood

from .computer import calculate_model, calculate_baseline, calculate_stellar_var, calculate_yerr_w, update_params

from .priors import transform_priors
from .priors.estimate_noise import estimate_noise, estimate_noise_out_of_transit

from .prepare_ttv_fit import prepare_ttv_fit

from .postprocessing.nested_sampling_compare_logZ import get_logZ, ns_plot_bayes_factors
from .postprocessing.plot_violins import ns_plot_violins, mcmc_plot_violins
from .postprocessing.plot_histograms import plot_histograms

from .plotting import fullplot, fullplot_csv, brokenplot, brokenplot_csv, tessplot, tessplot_csv

from .lightcurves import translate_limb_darkening_from_q_to_u as q_to_u
from .lightcurves import translate_limb_darkening_from_u_to_q as u_to_q



def GUI():
    allesfitter_path = os.path.dirname( os.path.realpath(__file__) )
    os.system( 'jupyter notebook "' + os.path.join(allesfitter_path,'GUI.ipynb') + '"' )

        


class allesclass():
    def __init__(self,datadir,quiet=True):
        config.init(datadir,quiet=quiet)
        self.BASEMENT = config.BASEMENT
        self.fulldata = config.BASEMENT.fulldata
        self.data = config.BASEMENT.data
        self.settings = config.BASEMENT.settings
        self.labels = get_labels(datadir, as_type='dic')
        
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
        
        #::: nested sampling?
        if os.path.exists( os.path.join(config.BASEMENT.outdir,'save_ns.pickle.gz') ):
            f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir,'save_ns.pickle.gz'), 'rb')
            results = pickle.load(f)
            f.close()
            self.posterior_samples = nested_sampling_output.draw_ns_posterior_samples(results) # all weighted posterior_samples
            self.posterior_params = nested_sampling_output.draw_ns_posterior_samples(results, as_type='dic') # all weighted posterior_samples
            self.posterior_params_median, self.posterior_params_ll, self.posterior_params_ul = general_output.get_params_from_samples(self.posterior_samples)
        
        #::: mcmc?
        elif os.path.exists( os.path.join(config.BASEMENT.outdir,'mcmc_save.h5') ):
            copyfile(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'), os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
            reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'), read_only=True )
            self.posterior_samples = draw_mcmc_posterior_samples(reader) # all weighted posterior_samples           
            self.posterior_params = draw_mcmc_posterior_samples(reader, as_type='dic') # all weighted posterior_samples
            self.posterior_samples_at_maximum_likelihood = draw_mcmc_posterior_samples_at_maximum_likelihood(reader)
            self.posterior_params_at_maximum_likelihood = draw_mcmc_posterior_samples_at_maximum_likelihood(reader, as_type='dic')
            self.posterior_params_median, self.posterior_params_ll, self.posterior_params_ul = general_output.get_params_from_samples(self.posterior_samples)
            os.remove(os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
            
        elif os.path.exists( os.path.join(config.BASEMENT.outdir,'ns_derived_samples.pickle') ):
            self.posterior_derived_params = pickle.load(open(os.path.join(datadir,'ns_derived_samples.pickle'),'rb'))
            
        elif os.path.exists( os.path.join(config.BASEMENT.outdir,'mcmc_derived_samples.pickle') ):
            self.posterior_derived_params = pickle.load(open(os.path.join(datadir,'mcmc_derived_samples.pickle'),'rb'))
            
            
        
    
    #::: plot
    
    def plot(self, inst, companion, style, 
             fig=None, ax=None, mode='posterior', Nsamples=20, samples=None, dt=None, 
             zoomwindow=8., force_binning=False,
             kwargs_data=None, kwargs_model=None, kwargs_ax=None):
        '''
        Required input:
        ---------------
        inst: str
            Name of the instrument (e.g. 'TESS')
            
        companion : None or str
            'b' / 'c' / ...
            
        style:
            'full' / 'phase' / 'phasezoom'
            
            
        Optional input:
        ---------------
        fig : matplotlib figure object
        
        ax : matplotlib axis object
        
        mode : str
            'posterior' / 'initial guess'
            
        Nsamples : int
            If mode=='posterior', this is the number of posterior curves to be plotted
            
        samples : array
            Prior or posterior samples to plot the fit from
        
        timelabel:
            'Time' / 'Time_since'
        '''
        if ax is None: fig, ax = plt.subplots(1,1)
        if (samples is None) and (Nsamples>0) and (mode != 'data'):
            if mode=='posterior':
                samples = self.posterior_samples[np.random.randint(len(self.posterior_samples), size=Nsamples)]
            elif mode=='initial_guess':
                samples = self.initial_guess_samples
            else:
                raise ValueError('Variable "mode" has to be "posterior" or "initial_guess".')
        general_output.plot_1(ax, samples, inst, companion, style, base=self, dt=dt, zoomwindow=zoomwindow, force_binning=force_binning, kwargs_data=kwargs_data, kwargs_ax=kwargs_ax)
        return fig, ax
        
        
    #::: posterior median
    
    def get_posterior_median_model(self, inst, key, xx=None, phased=False, settings=None):
        if phased==False:
            return calculate_model(self.posterior_params_median, inst, key, xx=xx, settings=settings)
        elif phased==True:
            p = update_params(self.posterior_params_median, phased=True)
            return calculate_model(p, inst, key, xx=xx)
        
    def get_posterior_median_baseline(self, inst, key, xx=None, model=None, phased=False):
        if phased==False:
            return calculate_baseline(self.posterior_params_median, inst, key, xx=xx, model=model)
        elif phased==True:
            raise ValueError('Not yet implemented.')
            
    def get_posterior_median_stellar_var(self, inst, key, xx=None, phased=False):
        if phased==False:
            return calculate_stellar_var(self.posterior_params_median, inst, key, xx=xx)
        elif phased==True:
            raise ValueError('Not yet implemented.')    
    
    def get_posterior_median_residuals(self, inst, key):
        model = self.get_posterior_median_model(inst, key)
        baseline = self.get_posterior_median_baseline(inst, key, model=model)
        stellar_var = self.get_posterior_median_stellar_var(inst, key)
        return self.data[inst][key] - model - baseline - stellar_var
    
    def get_posterior_median_yerr(self, inst, key):
        return calculate_yerr_w(self.posterior_params_median, inst, key)



    #::: posterior at maximum likelihood
    
    def get_posterior_at_maximum_likelihood_model(self, inst, key, xx=None, phased=False, settings=None):
        if phased==False:
            return calculate_model(self.posterior_params_at_maximum_likelihood, inst, key, xx=xx, settings=settings)
        elif phased==True:
            p = update_params(self.posterior_params_at_maximum_likelihood, phased=True)
            return calculate_model(p, inst, key, xx=xx)
        
    def get_posterior_at_maximum_likelihood_baseline(self, inst, key, xx=None, model=None, phased=False):
        if phased==False:
            return calculate_baseline(self.posterior_params_at_maximum_likelihood, inst, key, xx=xx, model=model)
        elif phased==True:
            raise ValueError('Not yet implemented.')
            
    def get_posterior_at_maximum_likelihood_stellar_var(self, inst, key, xx=None, phased=False):
        if phased==False:
            return calculate_stellar_var(self.posterior_params_at_maximum_likelihood, inst, key, xx=xx)
        elif phased==True:
            raise ValueError('Not yet implemented.')    
    
    def get_posterior_at_maximum_likelihood_residuals(self, inst, key):
        model = self.get_posterior_at_maximum_likelihood_model(inst, key)
        baseline = self.get_posterior_at_maximum_likelihood_baseline(inst, key, model=model)
        stellar_var = self.get_posterior_at_maximum_likelihood_stellar_var(inst, key)
        return self.data[inst][key] - model - baseline - stellar_var
    
    def get_posterior_at_maximum_likelihood_yerr(self, inst, key):
        return calculate_yerr_w(self.posterior_params_median, inst, key)


    
    #::: initial guess
    
    def get_initial_guess_model(self, inst, key, xx=None, phased=False):
        if phased==False:
            return calculate_model(self.initial_guess_params_median, inst, key, xx=xx)
        elif phased==True:
            raise ValueError('Not yet implemented.')
    
    def get_initial_guess_baseline(self, inst, key, xx=None, model=None, phased=False):
        if phased==False:
            return calculate_baseline(self.initial_guess_params_median, inst, key, xx=xx, model=model)
        elif phased==True:
            raise ValueError('Not yet implemented.')
    
    def get_initial_guess_stellar_var(self, inst, key, xx=None, phased=False):
        if phased==False:
            return calculate_stellar_var(self.initial_guess_params_median, inst, key, xx=xx)
        elif phased==True:
            raise ValueError('Not yet implemented.')
    
    
    
    #::: one posterior sample
    
    def get_one_posterior_curve_set(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if phased==False:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased==True:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_model(buf, inst, key, xx=xx), calculate_baseline(buf, inst, key, xx=xx), calculate_stellar_var(buf, inst, key, xx=xx)
    
    def get_one_posterior_model(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if phased==False:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased==True:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_model(buf, inst, key, xx=xx)

    def get_one_posterior_baseline(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if phased==False:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased==True:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_baseline(buf, inst, key, xx=xx)
    
    def get_one_posterior_stellar_var(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if phased==False:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased==True:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_stellar_var(buf, key, xx=xx)
    
    
    
#::: version
__version__ = '1.2.9'
