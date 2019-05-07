#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:44:29 2018

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
from shutil import copyfile
import emcee
from corner import corner
import warnings

#::: allesfitter modules
from . import config
from . import deriver
from .general_output import afplot, save_table, save_latex_table, logprint




###############################################################################
#::: draw samples from the MCMC save.5 (internally in the code)
###############################################################################
def draw_mcmc_posterior_samples(sampler, Nsamples=None, as_type='2d_array'):
    '''
    Default: return all possible sampels
    Set e.g. Nsamples=20 for plotting
    '''
#    global config.BASEMENT
    posterior_samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    if Nsamples:
        posterior_samples = posterior_samples[np.random.randint(len(posterior_samples), size=20)]

    if as_type=='2d_array':
        return posterior_samples
    
    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            posterior_samples_dic[key] = posterior_samples[:,ind].flatten()
        return posterior_samples_dic



###############################################################################
#::: plot the MCMC chains
###############################################################################
def plot_MCMC_chains(sampler):
    
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    #plot chains; emcee_3.0.0 format = (nsteps, nwalkers, nparameters)
    fig, axes = plt.subplots(config.BASEMENT.ndim+1, 1, figsize=(6,3*config.BASEMENT.ndim) )
    
    #::: plot the lnprob_values; emcee_3.0.0 format = (nsteps, nwalkers)
    axes[0].plot(log_prob, '-', rasterized=True)
    axes[0].axvline( 1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by'], color='k', linestyle='--' )
    mini = np.min(log_prob[int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']):,:])
    maxi = np.max(log_prob[int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']):,:])
    axes[0].set( title='lnprob', xlabel='steps', rasterized=True,
                 ylim=[mini, maxi] )
    axes[0].set_xticklabels( [int(label) for label in axes[0].get_xticks()*config.BASEMENT.settings['mcmc_thin_by']] )
    
    #:::plot all chains of parameters
    for i in range(config.BASEMENT.ndim):
        ax = axes[i+1]
        ax.set(title=config.BASEMENT.fitkeys[i], xlabel='steps')
        ax.plot(chain[:,:,i], '-', rasterized=True)
        ax.axvline( 1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by'], color='k', linestyle='--' )
#        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_xticklabels( [int(label) for label in ax.get_xticks()*config.BASEMENT.settings['mcmc_thin_by']] )

    plt.tight_layout()
    return fig, axes
    
    
    
###############################################################################
#::: plot the MCMC corner plot
###############################################################################
def plot_MCMC_corner(sampler):
    samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    
    fig = corner(samples, 
                 labels = config.BASEMENT.fitkeys,
                 range = [0.999]*config.BASEMENT.ndim,
                 quantiles=[0.15865, 0.5, 0.84135],
                 show_titles=True, title_kwargs={"fontsize": 14},
                 truths=config.BASEMENT.fittruths)
            
    return fig



###############################################################################
#::: print autocorr
###############################################################################
def print_autocorr(sampler):
    logprint('\nConvergence check:')
    logprint('--------------------------')
    
    logprint('{0: <20}'.format('Total steps:'),        '{0: <10}'.format(config.BASEMENT.settings['mcmc_total_steps']))
    logprint('{0: <20}'.format('Burn steps:'),         '{0: <10}'.format(config.BASEMENT.settings['mcmc_burn_steps']))
    logprint('{0: <20}'.format('Evaluation steps:'),   '{0: <20}'.format(config.BASEMENT.settings['mcmc_total_steps'] - config.BASEMENT.settings['mcmc_burn_steps']))
    
    discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by'])
    tau = sampler.get_autocorr_time(discard=discard, c=5, tol=10, quiet=True)*config.BASEMENT.settings['mcmc_thin_by']
    logprint('Autocorrelation times:')
    logprint('\t', '{0: <30}'.format('parameter'), '{0: <20}'.format('tau (in steps)'), '{0: <20}'.format('Chain length (in multiples of tau)'))
    for i, key in enumerate(config.BASEMENT.fitkeys):
        logprint('\t', '{0: <30}'.format(key), '{0: <20}'.format(tau[i]), '{0: <20}'.format((config.BASEMENT.settings['mcmc_total_steps'] - config.BASEMENT.settings['mcmc_burn_steps']) / tau[i]))
        
        

###############################################################################
#::: analyse the output from save_mcmc.h5 file
###############################################################################
def mcmc_output(datadir):
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
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'mcmc_fit.pdf')):
        try:
            overwrite = str(input('MCMC output files already exists in '+config.BASEMENT.outdir+'.\n'+\
                                  'What do you want to do?\n'+\
                                  '1 : overwrite the output files\n'+\
                                  '2 : abort\n'))
            if (overwrite == '1'):
                pass
            else:
                raise ValueError('User aborted operation.')
        except EOFError:
            warnings.warn("MCMC output files already existed from a previous run, and were automatically overwritten.")
            pass
    
    #::: load the mcmc_save.h5
    #::: copy over into tmp file (in case chain is still running and you want a quick look already)     
    copyfile(os.path.join(datadir,'results','mcmc_save.h5'), os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
    reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'), read_only=True )
    completed_steps = reader.get_chain().shape[0]*config.BASEMENT.settings['mcmc_thin_by']
    if completed_steps < config.BASEMENT.settings['mcmc_total_steps']: 
        #go into quick look mode
        #set burn_steps automatically to 75% of the chain length
        config.BASEMENT.settings['mcmc_total_steps'] = config.BASEMENT.settings['mcmc_thin_by']*reader.get_chain().shape[0]
        config.BASEMENT.settings['mcmc_burn_steps'] = int(0.75*config.BASEMENT.settings['mcmc_total_steps'])
    
    #::: print autocorr
    print_autocorr(reader)

    #::: plot the fit
    posterior_samples = draw_mcmc_posterior_samples(reader, Nsamples=20) #only 20 samples for plotting
    for companion in config.BASEMENT.settings['companions_all']:
        fig, axes = afplot(posterior_samples, companion)
        fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_fit_'+companion+'.pdf'), bbox_inches='tight' )
        plt.close(fig)
    
    #::: plot the chains
    fig, axes = plot_MCMC_chains(reader)
    fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_chains.pdf'), bbox_inches='tight' )
    plt.close(fig)

    #::: plot the corner
    fig = plot_MCMC_corner(reader)
    fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_corner.pdf'), bbox_inches='tight' )
    plt.close(fig)

    #::: save the tables
    posterior_samples = draw_mcmc_posterior_samples(reader) #all samples
    save_table(posterior_samples, 'mcmc')
    save_latex_table(posterior_samples, 'mcmc')
    
    #::: derive values (using stellar parameters from params_star.csv)
    if os.path.exists( os.path.join(config.BASEMENT.datadir,'params_star.csv') ):
        deriver.derive(posterior_samples, 'mcmc')
    else:
        print('File "params_star.csv" not found. Cannot derive final parameters.')
    
    #::: clean up and delete the tmp file
    os.remove(os.path.join(datadir,'results','mcmc_save_tmp.h5'))
    
    logprint('Done. For all outputs, see', config.BASEMENT.outdir)
    
    

###############################################################################
#::: get MCMC samples (for top-level user)
###############################################################################
def get_mcmc_posterior_samples(datadir, Nsamples=None, QL=False, as_type='dic'):
    config.init(datadir, QL=QL)
    reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'save.h5'), read_only=True )
    return draw_mcmc_posterior_samples(reader, Nsamples=Nsamples, as_type=as_type) #only 20 samples for plotting
    