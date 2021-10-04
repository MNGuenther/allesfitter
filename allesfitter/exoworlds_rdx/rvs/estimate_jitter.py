#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:32:25 2018

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
import os, sys
from datetime import datetime
import emcee
import corner
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
#solves python>=3.8 issues, see https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
from multiprocessing import Pool, cpu_count
from contextlib import closing

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

np.random.seed(21)




###############################################################################
#::: set up MCMC log probability function
#::: (has to be top-level for pickle)
###############################################################################
def log_probability(params):
    '''
    works on X, Y
    '''
    
    try:
        ll = log_likelihood(params)
        lp = external_log_prior(params)
    except:
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf
    return ll + lp



###############################################################################
#::: priors
###############################################################################  
def external_log_prior(params):
    log_jitter = params
    
    lp = 0
    if not (-23 < log_jitter < 0):
        lp = -np.inf
    
    return lp

    

###############################################################################
#::: likelihood
###############################################################################          
def log_likelihood(theta):
    log_jitter = theta
    jitter = np.exp(log_jitter)
    yerr = np.sqrt( WHITE_NOISE**2 + jitter**2 )
    inv_sigma2_w = 1./yerr**2
    return -0.5*(np.nansum((Y)**2 * inv_sigma2_w - np.log(inv_sigma2_w)))
        


###############################################################################
#::: run
###############################################################################
def estimate_jitter(x,y,white_noise,
        jitter_guess=None,
        mean=0.,
        nwalkers=50, thin_by=50, burn_steps=2500, total_steps=5000,
        bin_width=None,
        xlabel='x', ylabel='y', ydetr_label='ydetr',
        outdir='jitter_fit', fname=None, fname_summary=None,
        multiprocess=False, multiprocess_cores=None):
    
    '''
    Required Input:
    ---------------
    x : array of float
        x-values of the data set
    y : array of float
        y-values of the data set
    white_noise : array of float / float
        white_noise on y-values of the data set
        
    Optional Input:
    ---------------
    mean : float (default 0.)
        mean of the data set
        the default is 1., assuming usually y will be normalized flux
    nwalkers : int
        number of MCMC walkers
    thin_by : int
        thinning the MCMC chain by how much
    burn_steps : int
        how many steps to burn in the MCMC
    total_steps : int
        total MCMC steps (including burn_steps)
    xlabel : str
        x axis label (for plots)
    ylabel : str
        y axis label (for plots)       
    ydetr_label : str
        y_detr axis label (for plots)    
    outdir : str
        name of the output directory
    fname : str
        prefix of the output files (e.g. a planet name)
    multiprocess : bool (default False)
        run MCMC on multiprocess_cores cores        
    multiprocess_cores : bool (default None)
        run MCMC on many cores        
    '''
    
    #::: this is ugly, I know;
    #::: blame the multiprocessing and pickling issues, 
    #::: which demand global variables for efficiency    
    global X
    global Y
    global WHITE_NOISE
    global MEAN
    X = x
    Y = y
    WHITE_NOISE = white_noise
    MEAN = mean
    

    #::: outdir
    if not os.path.exists(outdir): os.makedirs(outdir)
    
    
    #::: print function that prints into console and logfile at the same time 
    now = datetime.now().isoformat()
    def logprint(*text):
        print(*text)
        original = sys.stdout
        with open( os.path.join(outdir,fname+'logfile_'+now+'.log'), 'a' ) as f:
            sys.stdout = f
            print(*text)
        sys.stdout = original

    
    #::: fname
    if fname is not None:
        fname += '_jitter_fit_'
    else:
        fname = 'jitter_fit_'

    
    #::: MCMC plot settings
    names = [r'$\log{(y_\mathrm{err})}$']
    discard = int(1.*burn_steps/thin_by)


    logprint('\nStarting...')


    #guess the total yerr = np.sqrt( rv_err**2 + jitter**2  )
    yerr = np.nanstd(Y) * np.ones_like(Y)


    #re-calcualte an initital guess for log_jitter 
    if jitter_guess is None:
        jitter_guess = np.nanmedian( np.log( np.sqrt( yerr**2 - white_noise**2 ) ) )


    #::: plot the data
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=white_noise, fmt=".b", capsize=0)
    ax.set( xlabel=xlabel, ylabel=ylabel, title='Original data' )
    fig.savefig( os.path.join(outdir,fname+'data.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    

    
    
    ###########################################################################
    #::: MCMC fit
    ###########################################################################
    if multiprocess and not multiprocess_cores:
        multiprocess_cores = cpu_count()-1
        
    logprint('\nRunning MCMC fit...')
    if multiprocess: logprint('\tRunning on', multiprocess_cores, 'CPUs.')   
    
    
    #::: all
    initial = np.array([jitter_guess])

    
    #::: set up MCMC
    ndim = len(initial)
    backend = emcee.backends.HDFBackend(os.path.join(outdir,fname+'mcmc_save.h5')) # Set up a new backend
    backend.reset(nwalkers, ndim)


    #::: run MCMC
    def run_mcmc(sampler):
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        sampler.run_mcmc(p0, total_steps/thin_by, thin_by=thin_by, progress=True);
    
    if multiprocess:    
        with closing(Pool(processes=(multiprocess_cores))) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
            run_mcmc(sampler)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
        run_mcmc(sampler)
    
    logprint('\nAcceptance fractions:')
    logprint(sampler.acceptance_fraction)

    tau = sampler.get_autocorr_time(discard=discard, c=5, tol=10, quiet=True)*thin_by
    logprint('\nAutocorrelation times:')
    logprint('\t', '{0: <30}'.format('parameter'), '{0: <20}'.format('tau (in steps)'), '{0: <20}'.format('Chain length (in multiples of tau)'))
    for i, key in enumerate(names):
        logprint('\t', '{0: <30}'.format(key), '{0: <20}'.format(tau[i]), '{0: <20}'.format((total_steps-burn_steps) / tau[i]))
    
    

        
    ###########################################################################
    #::: Output
    ###########################################################################
    def get_params_from_samples(samples, names):
        '''
        read MCMC results and update params
        '''
    
        buf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))                                         
        theta_median = [ item[0] for item in buf ]
        theta_ul = [ item[1] for item in buf ]
        theta_ll = [ item[2] for item in buf ]
        params_median = { n:t for n,t in zip(names,theta_median) }
        params_ul = { n:t for n,t in zip(names,theta_ul) }
        params_ll = { n:t for n,t in zip(names,theta_ll) }
        
        return params_median, params_ll, params_ul
            
        
    #::: get the samples, 
    samples = sampler.get_chain(flat=True, discard=discard)
    
    
    #::: get the resulting params dictionaries
    params, params_ll, params_ul = get_params_from_samples(samples, names)

    
    #::: Save the resulting parameters in a table
    with open( os.path.join(outdir,fname+'table.csv'), 'wb' ) as f:
        f.write('name,median,ll,ul\n')
        for i, key in enumerate(names):
            f.write(key + ',' + str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]) + '\n' )
    
    
    #::: if requested, append a row into the summary file, too
    if fname_summary is not None:
        with open( fname_summary, 'ab' ) as f:
            f.write(fname[0:-1] + ',')
            for i, key in enumerate(names):
                f.write(str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]))
                if i<len(names)-1:
                    f.write(',')
                else:
                    f.write('\n')


    #::: plot chains; format of chain = (nwalkers, nsteps, nparameters)
    fig, axes = plt.subplots(ndim+1, 1, figsize=(6,4*(ndim+1)) )
    steps = np.arange(0,total_steps,thin_by)
    
    
    #::: plot the lnprob_values (nwalkers, nsteps)
    for j in range(nwalkers):
        axes[0].plot(steps, sampler.get_log_prob()[:,j], '-')
    axes[0].set( ylabel='lnprob', xlabel='steps' )
    
    
    #:::plot all chains of parameters
    for i in range(ndim):
        ax = axes[i+1]
        ax.set( ylabel=names[i], xlabel='steps')
        for j in range(nwalkers):
            ax.plot(steps, sampler.chain[j,:,i], '-')
        ax.axvline( burn_steps, color='k', linestyle='--' )
    
    plt.tight_layout()
    fig.savefig( os.path.join(outdir,fname+'mcmc_chains.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
 
        
    #::: plot corner
    fig = corner.corner(samples,
                        labels=names,
                        show_titles=True, title_kwargs={"fontsize": 12});
    fig.savefig( os.path.join(outdir,fname+'mcmc_corner.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)


    logprint('\nDone. All output files are in '+outdir)
    


if __name__ == '__main__':
    pass
