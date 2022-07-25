#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:44:29 2018

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
from matplotlib.ticker import ScalarFormatter, FixedLocator
import os
from shutil import copyfile
import emcee
from corner import corner
import warnings

#::: allesfitter modules
from . import config
from . import deriver
from .computer import calculate_model, calculate_baseline, calculate_stellar_var
from .general_output import afplot, afplot_per_transit, save_table, save_latex_table, logprint, get_params_from_samples, plot_ttv_results
from .plot_top_down_view import plot_top_down_view
from .utils.latex_printer import round_tex
from .statistics import residual_stats




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
#::: draw the maximum likelihood samples from the MCMC save.5 (internally in the code)
###############################################################################
def draw_mcmc_posterior_samples_at_maximum_likelihood(sampler, as_type='1d_array'):
    log_prob = sampler.get_log_prob(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    posterior_samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    ind_max = np.argmax(log_prob)
    posterior_samples = posterior_samples[ind_max,:]
    
    if as_type=='1d_array':
        return posterior_samples

    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            posterior_samples_dic[key] = posterior_samples[ind].flatten()
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
    axes[0].xaxis.set_major_locator(FixedLocator(axes[0].get_xticks())) #useless line to bypass useless matplotlib warnings
    axes[0].set_xticklabels( [int(label) for label in axes[0].get_xticks()*config.BASEMENT.settings['mcmc_thin_by']] )
    
    #:::plot all chains of parameters
    for i in range(config.BASEMENT.ndim):
        ax = axes[i+1]
        ax.set(title=config.BASEMENT.fitkeys[i], xlabel='steps')
        ax.plot(chain[:,:,i], '-', rasterized=True)
        ax.axvline( 1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by'], color='k', linestyle='--' )
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks())) #useless line to bypass useless matplotlib warnings
        ax.set_xticklabels( [int(label) for label in ax.get_xticks()*config.BASEMENT.settings['mcmc_thin_by']] )

    plt.tight_layout()
    return fig, axes
    
    
    
###############################################################################
#::: plot the MCMC corner plot
###############################################################################
#def plot_MCMC_corner(sampler):
#    samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
#    
#    fig = corner(samples, 
#                 labels = config.BASEMENT.fitkeys,
#                 range = [0.999]*config.BASEMENT.ndim,
#                 quantiles=[0.15865, 0.5, 0.84135],
#                 show_titles=True, title_kwargs={"fontsize": 14},
#                 truths=config.BASEMENT.fittruths)
#            
#    return fig

def plot_MCMC_corner(sampler):
    samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    
    params_median, params_ll, params_ul = get_params_from_samples(samples)
    params_median2, params_ll2, params_ul2 = params_median.copy(), params_ll.copy(), params_ul.copy()
    fittruths2 = config.BASEMENT.fittruths.copy()

    #::: make pretty titles for the corner plot  
    labels, units = [], []
    for i,l in enumerate(config.BASEMENT.fitlabels):
        labels.append( str(config.BASEMENT.fitlabels[i]) )
        units.append( str(config.BASEMENT.fitunits[i]) )
    
    for companion in config.BASEMENT.settings['companions_all']:
        if companion+'_epoch' in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==companion+'_epoch')[0][0]
            samples[:,ind] -= int(params_median[companion+'_epoch'])                #np.round(params_median[companion+'_epoch'],decimals=0)
            units[ind] = str(units[ind]+'-'+str(int(params_median[companion+'_epoch']))+'d')    #np.format_float_positional(params_median[companion+'_epoch'],0)+'d')
            fittruths2[ind] -= int(params_median[companion+'_epoch'])
            params_median2[companion+'_epoch'] -= int(params_median[companion+'_epoch'])
                
    for i,l in enumerate(labels):
        if len( units[i].strip(' ') ) > 0:
            labels[i] = str(labels[i]+' ('+units[i]+')')
        
    #::: corner plot
    fontsize = np.min(( 24. + 0.5*config.BASEMENT.ndim, 40 ))
    fig = corner(samples, 
                 labels = labels,
                 range = [0.999]*config.BASEMENT.ndim,
                 quantiles=[0.15865, 0.5, 0.84135],
                 show_titles=False, 
                 #title_kwargs={"fontsize": 14},
                 label_kwargs={"fontsize":fontsize, "rotation":45, "horizontalalignment":'right'},
                 max_n_ticks=3,
                 truths=fittruths2, truth_color="r")
    caxes = np.reshape(np.array(fig.axes), (config.BASEMENT.ndim,config.BASEMENT.ndim))

    #::: set allesfitter titles
    for i, key in enumerate(config.BASEMENT.fitkeys): 
        
        value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
        ctitle = r'' + labels[i] + '\n' + r'$=' + value + '$'
        if len(config.BASEMENT.fitkeys)>1:
            # caxes[i,i].set_title(ctitle)
            caxes[i,i].set_title(ctitle, fontsize=fontsize, rotation=45, horizontalalignment='left')
            for i in range(caxes.shape[0]):
                for j in range(caxes.shape[1]):
                    caxes[i,j].xaxis.set_label_coords(0.5, -0.5)
                    caxes[i,j].yaxis.set_label_coords(-0.5, 0.5)
        
                    if i==(caxes.shape[0]-1): 
                        fmt = ScalarFormatter(useOffset=False)
                        caxes[i,j].xaxis.set_major_formatter(fmt)
                    if (i>0) and (j==0):
                        fmt = ScalarFormatter(useOffset=False)
                        caxes[i,j].yaxis.set_major_formatter(fmt)
                        
                    for tick in caxes[i,j].xaxis.get_major_ticks(): tick.label.set_fontsize(24) 
                    for tick in caxes[i,j].yaxis.get_major_ticks(): tick.label.set_fontsize(24)    
        else:
            caxes[i,i].set_title(ctitle)
            caxes[i,i].xaxis.set_label_coords(0.5, -0.5)
            caxes[i,i].yaxis.set_label_coords(-0.5, 0.5)
            
            
    return fig



###############################################################################
#::: print autocorr
###############################################################################
def print_autocorr(sampler):
    logprint('\nConvergence check')
    logprint('-------------------')
    
    logprint('{0: <20}'.format('Total steps:'),        '{0: <10}'.format(config.BASEMENT.settings['mcmc_total_steps']))
    logprint('{0: <20}'.format('Burn steps:'),         '{0: <10}'.format(config.BASEMENT.settings['mcmc_burn_steps']))
    logprint('{0: <20}'.format('Evaluation steps:'),   '{0: <20}'.format(config.BASEMENT.settings['mcmc_total_steps'] - config.BASEMENT.settings['mcmc_burn_steps']))
    
    N_evaluation_samples = int( 1. * config.BASEMENT.settings['mcmc_nwalkers'] * (config.BASEMENT.settings['mcmc_total_steps']-config.BASEMENT.settings['mcmc_burn_steps']) / config.BASEMENT.settings['mcmc_thin_by'] )
    logprint('{0: <20}'.format('Evaluation samples:'),   '{0: <20}'.format(N_evaluation_samples))
     
    # if N_evaluation_samples>200000:
    #     answer = input('It seems like you are asking for ' + str(N_evaluation_samples) + 'MCMC evaluation samples (calculated as mcmc_nwalkers * (mcmc_total_steps-mcmc_burn_steps) / mcmc_thin_by).'+\
    #                     'That is an aweful lot of samples.'+\
    #                     'What do you want to do?\n'+\
    #                     '1 : continue at any sacrifice\n'+\
    #                     '2 : abort and increase the mcmc_thin_by parameter in settings.csv (do not do this if you continued an old run!)\n')
    #     if answer==1: 
    #         pass
    #     else:
    #         raise ValueError('User aborted the run.')


    discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by'])
    tau = sampler.get_autocorr_time(discard=discard, c=5, tol=10, quiet=True)*config.BASEMENT.settings['mcmc_thin_by']
    logprint('Autocorrelation times:')
    logprint('\t', '{0: <30}'.format('parameter'), '{0: <20}'.format('tau (in steps)'), '{0: <20}'.format('Chain length (in multiples of tau)'))
    converged = True
    for i, key in enumerate(config.BASEMENT.fitkeys):
        chain_length = (1.*(config.BASEMENT.settings['mcmc_total_steps'] - config.BASEMENT.settings['mcmc_burn_steps']) / tau[i])
        logprint('\t', '{0: <30}'.format(key), '{0: <20}'.format(tau[i]), '{0: <20}'.format(chain_length))
        if (chain_length < 30) or np.isinf(chain_length) or np.isnan(chain_length):
            converged = False
            
    if converged:
        logprint('\nSuccesfully converged! All chains are at least 30x the autocorrelation length.\n')
    else:
        logprint('\nNot yet converged! Some chains are less than 30x the autocorrelation length. Please continue to run with longer chains, or start again with more walkers.\n')
        

###############################################################################
#::: analyse the output from save_mcmc.h5 file
###############################################################################
def mcmc_output(datadir, quiet=False):
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
    config.init(datadir, quiet=quiet)
    
    
    #::: security check
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'mcmc_table.csv')):
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
    copyfile(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'), os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
    reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'), read_only=True )
    completed_steps = reader.get_chain().shape[0]*config.BASEMENT.settings['mcmc_thin_by']
    if completed_steps < config.BASEMENT.settings['mcmc_total_steps']: 
        #go into quick look mode
        #check how many total steps are actually done so far:
        config.BASEMENT.settings['mcmc_total_steps'] = config.BASEMENT.settings['mcmc_thin_by']*reader.get_chain().shape[0]
        #if this is at least twice the wished-for burn_steps, then let's keep those
        #otherwise, set burn_steps automatically to 75% of how many total steps are actually done so far
        if config.BASEMENT.settings['mcmc_total_steps'] > 2*config.BASEMENT.settings['mcmc_burn_steps']:
            pass
        else:
            config.BASEMENT.settings['mcmc_burn_steps'] = int(0.75*config.BASEMENT.settings['mcmc_total_steps'])
    
    
    #::: print autocorr
    print_autocorr(reader)


    #::: plot the fit
    posterior_samples = draw_mcmc_posterior_samples(reader, Nsamples=20) #only 20 samples for plotting
    
    for companion in config.BASEMENT.settings['companions_all']:
        fig, axes = afplot(posterior_samples, companion)
        if fig is not None:
            fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_fit_'+companion+'.pdf'), bbox_inches='tight' )
            plt.close(fig)
            
    for companion in config.BASEMENT.settings['companions_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            first_transit = 0
            while (first_transit >= 0):
                try:
                    fig, axes, last_transit, total_transits = afplot_per_transit(posterior_samples, inst, companion, kwargs_dict={'first_transit':first_transit})
                    fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_fit_per_transit_'+inst+'_'+companion+'_' + str(last_transit) + 'th.pdf'), bbox_inches='tight' )
                    plt.close(fig)
                    if total_transits > 0 and last_transit < total_transits - 1:
                        first_transit = last_transit
                    else:
                        first_transit = -1
                except:
                    first_transit = -1
                    pass
    
    #::: plot the chains
    fig, axes = plot_MCMC_chains(reader)
    try: #some matplotlib versions cannot handle jpg
        fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_chains.jpg'), bbox_inches='tight' )
    except:
        fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_chains.png'), bbox_inches='tight' )
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
        logprint('File "params_star.csv" not found. Cannot derive final parameters.')
    
    
    #::: retrieve the median parameters and curves
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)
    
    
    #::: check the residuals
    for inst in config.BASEMENT.settings['inst_all']:
        if inst in config.BASEMENT.settings['inst_phot']: key='flux'
        elif inst in config.BASEMENT.settings['inst_rv']: key='rv'
        elif inst in config.BASEMENT.settings['inst_rv2']: key='rv2'
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
        
        
    #::: clean up and delete the tmp file
    os.remove(os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
    
    logprint('\nDone. For all outputs, see', config.BASEMENT.outdir, '\n')
    
    
    #::: return a nerdy quote
    try:
        with open(os.path.join(os.path.dirname(__file__), 'utils', 'quotes.txt')) as dataset:
            return(np.random.choice([l for l in dataset]))
    except:
        return('42')
    
    

###############################################################################
#::: get MCMC samples (for top-level user)
###############################################################################
def get_mcmc_posterior_samples(datadir, Nsamples=None, as_type='dic'): #QL=False, 
    # config.init(datadir, QL=QL)
    config.init(datadir)
    reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'), read_only=True )
    return draw_mcmc_posterior_samples(reader, Nsamples=Nsamples, as_type=as_type) #only 20 samples for plotting
    
