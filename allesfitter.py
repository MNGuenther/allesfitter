#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:10:35 2018

@author:
Dr. Maximilian N. Guenther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
"""

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import os
import ellc
import emcee
from corner import corner
from scipy.optimize import minimize
import numpy.polynomial.polynomial as poly
from mytools import lightcurve_tools as lct
from mytools import index_transits
from multiprocessing import Pool, cpu_count
from contextlib import closing
from pprint import pprint
from shutil import copyfile
import latex_printer
import warnings
warnings.simplefilter('ignore', np.RankWarning)
try:
    import celerite
    from celerite import terms
except ImportError:
    warnings.warn("Cannot import package 'celerite', thus 'hybrid_GP' baseline models will not be supported.")



###############################################################################
#::: set data outside and use as a global variable to enhance emcee
###############################################################################
data = {}



###############################################################################
#::: load settings
###############################################################################
def load_settings(datadir):
    buf = np.genfromtxt( os.path.join(datadir,'settings.csv'), dtype=None, delimiter=',', names=True )

    settings = {}
    for key in ['planets_phot', 'planets_rv', 'inst_phot', 'inst_rv']:         
        if buf[key]: settings[key] = str(buf[key]).split(' ')
        else:        settings[key] = []
    settings['nwalkers'] = int(buf['nwalkers'])
    settings['total_steps'] = int(buf['total_steps'])
    settings['burn_steps'] = int(buf['burn_steps'])
    settings['thin_by'] = int(buf['thin_by'])
    settings['multiprocess'] = bool(buf['multiprocess'])
    settings['baseline_fit'] = str(buf['baseline_fit'])
    return settings

    

###############################################################################
#::: load params
###############################################################################
def load_params(datadir):

    buf = np.genfromtxt(os.path.join(datadir,'params.csv'), delimiter=',',comments='#',dtype=None,names=True)
    
    allkeys = buf['name']
    labels = buf['label']
    units = buf['unit']
    
    params = {}
    for i,key in enumerate(allkeys):
        params[key] = buf['value'][i]
    
    ind_fit = (buf['fit']==1)
    fitkeys = buf['name'][ ind_fit ]
    theta_0 = buf['value'][ ind_fit ]
    init_err = buf['init_err'][ ind_fit ]
    bounds = [ str(item).split(' ') for item in buf['bounds'][ ind_fit ] ]
    for i, item in enumerate(bounds):
        bounds[i] = [ item[0], float(item[1]), float(item[2]) ]
        
    return theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units
    


###############################################################################
#::: load data
###############################################################################
def load_data(datadir, settings, params=None, fast_fit=False):
    '''
    Example: 
    -------
        A lightcurve is stored as
            data['TESS']['time'], data['TESS']['flux']
        A RV curve is stored as
            data['HARPS']['time'], data['HARPS']['flux']
    
    Inputs:
    -------
    datadir : str
        ...
    settings : dict
        ...
    params : dict (optional)
        only needed if fast_fit==True
    fast_fit : bool (optional)
        if True, fit only 12h around each transit, neglect the rest of the photometric data
    '''
    data = {}
    for inst in settings['inst_phot']:
        time, flux, flux_err = np.genfromtxt(os.path.join(datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)         
        if fast_fit: time, flux, flux_err = reduce_phot_data(time, flux, flux_err, params, settings)
        data[inst] = {
                      'time':time,
                      'flux':flux,
                      'err_scales_flux':flux_err/np.nanmean(flux_err)
                     }
        
    for inst in settings['inst_rv']:
        time, rv, rv_err = np.genfromtxt(os.path.join(datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)         
        data[inst] = {
                      'time':time,
                      'rv':rv,
                      'err_scales_rv':rv_err/np.nanmean(rv_err)
                     }      
    return data




###############################################################################
#::: cut away the out-of-transit regions to speed up the fit
###############################################################################   
def reduce_phot_data(time, flux, flux_err, params, settings):
    ind_in = []
          
    for planet in settings['planets_phot']:
        t0 = time[0]
        dt = params[planet+'_epoch'] - t0
        n = np.max( int( dt/params[planet+'_period'] )+1, 0 )
        epoch = params[planet+'_epoch'] - n*params[planet+'_period']    
        dic = {'TIME':time, 'EPOCH':epoch, 'PERIOD':params[planet+'_period'], 'WIDTH':8./24.}
        ind_in += list(index_transits.index_transits(dic)[0])
    time = time[ind_in]
    flux = flux[ind_in]
    flux_err = flux_err[ind_in]
    return time, flux, flux_err



###############################################################################
#::: update params
###############################################################################                
def update_params(theta, params, fitkeys):
    params2 = params.copy()
    for i, key in enumerate(fitkeys):
        params2[key] = theta[i]    
#    print params2
    return params2    



###############################################################################
#::: convert input params into ellc params
###############################################################################  
def get_ellc_params(params, inst, planet, key, phased):
    global data
    
    ellc_params = params.copy()
    
    
    ###########################################################################
    #::: phase-folded?
    #::: (it's important to have this before calculating the semi-major axis!)
    ###########################################################################
    if phased:
        ellc_params[planet+'_epoch'] = 0.
        ellc_params[planet+'_period'] = 1.
    
    
    ###########################################################################
    #::: photometry
    ###########################################################################
    if key=='flux':
        #::: R_1/a and R_2/a
        ellc_params[planet+'_radius_1'] = params[planet+'_rsuma'] / (1. + params[planet+'_rr'])
        ellc_params[planet+'_radius_2'] = ellc_params[planet+'_radius_1'] * params[planet+'_rr']
        
        #::: incl
        ellc_params[planet+'_incl'] = np.arccos( params[planet+'_cosi'] )/np.pi*180.
        
        #::: limb darkening
        ldcode_to_ldstr = [
            "none",#   :  0,
            "lin",#    :  1,
            "quad",#   :  2,
            "sing",#   :  3,
            "claret",# :  4,
            "log",#  :  5,
            "sqrt",#  :  6,
            "exp",#    :  7,
            "power-2",#:  8,
            "mugrid"# : -1
          ]
        ellc_params['ld_1_'+inst] = ldcode_to_ldstr[ int(params['ld_law_'+inst]) ]
        
        if ellc_params['ld_1_'+inst] == 'lin':
            ellc_params['ldc_1_'+inst] = params['ldc_q1_'+inst]
            
        elif ellc_params['ld_1_'+inst] == 'quad':
            ldc_u1 = 2.*np.sqrt(params['ldc_q1_'+inst]) * params['ldc_q2_'+inst]
            ldc_u2 = np.sqrt(params['ldc_q1_'+inst]) * (1. - 2.*params['ldc_q2_'+inst])
            ellc_params['ldc_1_'+inst] = [ ldc_u1, ldc_u2 ]
            
        else:
            raise ValueError("Currently only 'lin' and 'quad' limb darkening implemented.")
    
    
    ###########################################################################
    #::: RV
    ###########################################################################
    elif key=='rv':        
        ellc_params[planet+'_incl'] = np.arccos( ellc_params[planet+'_cosi'] )/np.pi*180.

        ecc = ellc_params[planet+'_f_s']**2 + ellc_params[planet+'_f_c']**2
        a_1 = 0.019771142 * ellc_params[planet+'_K'] * ellc_params[planet+'_period'] * np.sqrt(1. - ecc**2)/np.sin(ellc_params[planet+'_incl']*np.pi/180.)
        ellc_params[planet+'_a'] = (1.+1./ellc_params[planet+'_q'])*a_1
        
    
    return ellc_params



    
###############################################################################
#::: flux fct
###############################################################################
def flux_fct(time, params, inst, planet, phased=False):
    ellc_params = get_ellc_params(params, inst, planet, 'flux', phased)
    
    model_flux = ellc.lc(
                      t_obs =       time, 
                      radius_1 =    ellc_params[planet+'_radius_1'], 
                      radius_2 =    ellc_params[planet+'_radius_2'], 
                      sbratio =     ellc_params[planet+'_sbratio_'+inst], 
                      incl =        ellc_params[planet+'_incl'], 
                      light_3 =     ellc_params['light_3_'+inst],
                      t_zero =      ellc_params[planet+'_epoch'],
                      period =      ellc_params[planet+'_period'],
                      f_c =         ellc_params[planet+'_f_c'],
                      f_s =         ellc_params[planet+'_f_s'],
                      ldc_1 =       ellc_params['ldc_1_'+inst],
#                      ldc_2 = ldc_2,
                      ld_1 =        ellc_params['ld_1_'+inst],
#                      ld_2 = 'quad' 
                      )
                 
    return model_flux
        
    

###############################################################################
#::: rv fct
###############################################################################
def rv_fct(time, params, inst, planet, phased=False):
    ellc_params = get_ellc_params(params, inst, planet, 'rv', phased)
        
    model_rv1, model_rv2 = ellc.rv(
                      t_obs =   time, 
                      incl =    ellc_params[planet+'_incl'], 
                      t_zero =  ellc_params[planet+'_epoch'],
                      period =  ellc_params[planet+'_period'],
                      a =       ellc_params[planet+'_a'],
                      f_c =     ellc_params[planet+'_f_c'],
                      f_s =     ellc_params[planet+'_f_s'],
                      q =       ellc_params[planet+'_q'],
                      flux_weighted = False
                      )
                 
    return model_rv1, model_rv2



###############################################################################
#::: log likelihood 
###############################################################################
def lnlike(theta, params, fitkeys, settings):
    params = update_params(theta, params, fitkeys)
    
    lnlike = 0
    
    def lnlike_1(key):
        model_y = calculate_model_y(data[inst]['time'], params, settings, inst, key)
        y_res = calculate_y_res(params, settings, inst, key, model_y)
        inv_sigma2 = data[inst]['err_scales_'+key]**(-2) * params['inv_sigma2_'+key+'_'+inst]
        return -0.5*(np.nansum((y_res)**2 * inv_sigma2 - np.log(inv_sigma2)))
    
    for inst in settings['inst_phot']:
        lnlike += lnlike_1('flux')
        
    for inst in settings['inst_rv']:
        lnlike += lnlike_1('rv')

    return lnlike



###############################################################################
#::: calculate model curve
###############################################################################      
def calculate_model_y(x, params, settings, inst, key, phased=False):
    
    if key=='flux':
        depth = 0.
        for planet in settings['planets_phot']:
            depth += ( 1. - flux_fct(x, params, inst, planet, phased) )
        model_flux = 1. - depth
        return model_flux
    
    elif key=='rv':
        model_rv = 0.
        for planet in settings['planets_rv']:
            model_rv += rv_fct(x, params, inst, planet, phased)[0]
        return model_rv
    
    elif (key=='centdx') | (key=='centdy'):
        raise ValueError("Fitting for 'centdx' and 'centdy' not yet implemented.")
        #TODO
        
    else:
        raise ValueError("Variable 'key' has to be 'flux', 'rv', 'centdx', or 'centdy'.")


        
###############################################################################
#::: calculate residuals
###############################################################################  
def calculate_y_res(params, settings, inst, key, model_y):
    '''
    Returns:
    --------
    y : array
        if settings['baseline_fit']=='traditional':
            data - model - offset
        if settings['baseline_fit']=='hybrid_poly':
            data - model - offset - polynomial baseline MLE fit
        if settings['baseline_fit']=='hybrid_GP':
            data - model - offset - GP baseline baseline MLE fit
    '''
    global data
    
    
    model_res = data[inst][key] - params['offset_'+key+'_'+inst] - model_y
    
    
    ###########################################################################
    #::: traditional (constant offset)
    ###########################################################################
    if settings['baseline_fit'] == 'traditional':
        y_res = model_res
        return y_res
    
    
    ###########################################################################
    #::: hybrid_poly (like Gillon+2012)
    ###########################################################################
    elif settings['baseline_fit'] == 'hybrid_poly':
        #polyfit needs the x-axis reset to [0,1], otherwise it goes nuts
        xx = (data[inst]['time'] - data[inst]['time'][0])
        xx /= xx[-1]
        polyorder = int(params['baseline_polyorder_'+key+'_'+inst])
        if polyorder>=0:
            inv_sigma = data[inst]['err_scales_'+key]**(-1) * np.sqrt(params['inv_sigma2_'+key+'_'+inst])
            weights = inv_sigma/np.nanmax(inv_sigma)
            ind = np.isfinite(model_res) #polyfit can't handle NaN
            params_poly = poly.polyfit(xx[ind],model_res[ind],polyorder,w=weights[ind]) #WARNING: returns params in reverse order than np.polyfit!!!
            baseline = poly.polyval(xx, params_poly)
        else:
            raise ValueError("Parameters 'baseline polyorder' have to be >= 0.")
        y_res = model_res - baseline
        
#        plt.figure()
#        plt.errorbar(data[inst]['time'], model_res,yerr=1./inv_sigma,fmt='k.')
#        plt.plot(data[inst]['time'], baseline, 'r-', lw=2)
#        plt.show()
        
        return y_res
    
    
    ###########################################################################
    #::: hybrid_GP (like Gillon+2012, but with a GP instead of polynomials)
    ###########################################################################    
    elif settings['baseline_fit'] == 'hybrid_GP':
        xx = data[inst]['time']
        yy = model_res
#        yerr = data[inst]['err_scales_'+key] * (1./np.sqrt(params['inv_sigma2_'+key+'_'+inst]))
        yerr = np.nanstd(yy)
        kernel = terms.Matern32Term(log_sigma=1., log_rho=1.)
        gp = celerite.GP(kernel, mean=0.) 
        gp.compute(xx, yerr=yerr)
         
        def neg_log_like(params, yy, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(yy)
        
        def grad_neg_log_like(params, yy, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(yy)[1]
        
        
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(yy, gp))
        gp.set_parameter_vector(soln.x)
        
        baseline = gp.predict(yy, xx)[0]
        y_res = model_res - baseline
        
#        plt.figure()
#        plt.plot(xx,yy,'k.')
#        plt.plot(xx, baseline, 'r-')
#        plt.show()
        
        return y_res
        
    else:
        raise ValueError("Setting 'baseline_fit' has to be 'traditional','hybrid_poly', or 'hybrid_GP', but is:"+settings['baseline_fit'])
        
    
    

###############################################################################
#::: log prior
###############################################################################
def lnprior(theta, bounds):
    '''
    bounds has to be list of len(theta), containing tuples of form
    ('none'), ('uniform', lower bound, upper bound), or ('normal', mean, std)
    '''
    lnp = 0.        
    
    for th, b in zip(theta, bounds):
        if (b[0] == 'uniform') and not (b[1] <= th <= b[2]):
            return -np.inf
        elif b[0] == 'normal':
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[2]) * np.exp( - (th - b[1])**2 / (2.*b[2]**2) ) )
       
    return lnp



###############################################################################
#::: log probability
###############################################################################    
def lnprob(theta, bounds, params, fitkeys, settings):
    lp = lnprior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    else:
        try:
            ln = lnlike(theta, params, fitkeys, settings)
            return lp + ln
        except:
            return -np.inf 
    


################################################################################
##::: negative log likelihood = MLE fit cost function
################################################################################
#def neg_lnlike(theta, params, fitkeys, settings):
#    return -lnlike(theta, params, fitkeys, settings)
    


################################################################################
##::: translate MCMC bounds into MLE bounds
################################################################################
#def translate_bounds(MCMC_bounds):
#    '''
#    Example: 
#        for 10 fit parameters, the bounds need to look like
#            MCMC bounds: [ array(10xll), array(10xul) ]
#            MLE bounds: array( [ll,ul], [ll,ul], ..., [ll,ul] )
#    '''
#    MLE_bounds = [ [MCMC_bounds[0][i]+1e-12, MCMC_bounds[1][i]-1e-12] for i in range(len(MCMC_bounds[0])) ]
#    return MLE_bounds
    

    
################################################################################
##::: MLE fit
################################################################################
#def MLE_fit(theta_0, MCMC_bounds, params, fitkeys, settings):
#    '''
#    Example: 
#        for 10 fit parameters, the bounds need to look like
#            MCMC bounds: [ array(10xll), array(10xul) ]
#            MLE bounds: array( [ll,ul], [ll,ul], ..., [ll,ul] )
#    '''
#    MLE_bounds = translate_bounds(MCMC_bounds)
#    solver = minimize(
#                    neg_lnlike, 
#                    theta_0,
##                    method="L-BFGS-B", 
#                    bounds=MLE_bounds, 
#                    args=(data, params, fitkeys, settings)
#                    )
#    results = solver.x
#    return results
    


###########################################################################
#::: MCMC fit
###########################################################################
def MCMC_fit(outdir, theta_0, init_err, bounds, params, fitkeys, settings, continue_old_run=False):
    ndim = len(theta_0)
    
    #::: set up a backend
    backend = emcee.backends.HDFBackend(os.path.join(outdir,'save.h5'))
    
    #::: continue on the backend / reset the backend
    if not continue_old_run:
        backend.reset(settings['nwalkers'], ndim)
    
    #::: helper fct
    def run_mcmc(sampler):
        
        #::: set initial walker positions
        if continue_old_run:
            p0 = backend.get_chain()[-1,:,:]
            already_completed_steps = backend.get_chain().shape[0] * settings['thin_by']
        else:
            p0 = theta_0 + init_err * np.random.randn(settings['nwalkers'], ndim)
            already_completed_steps = 0
        
        #::: make sure the inital positions are within the limits
        for i, b in enumerate(bounds):
            if b[0] == 'uniform':
                p0[:,i] = np.clip(p0[:,i], b[1], b[2]) 
        
        #::: run the sampler        
        print("Running burn-in and production...")
        sampler.run_mcmc(p0,
                         (settings['total_steps'] - already_completed_steps)/settings['thin_by'], 
                         thin_by=settings['thin_by'], 
                         progress=True)
        
        print 'Acceptance fraction:', sampler.acceptance_fraction
        try:
            print 'Autocorrelation times:', sampler.get_autocorr_time(discard=settings['burn_steps'])
        except:
            print 'Autocorrelation times:', 'chains not converged yet'

    #::: run
    if settings['multiprocess']:    
        with closing(Pool(processes=(cpu_count()-1))) as pool:
            print 'Running on', cpu_count()-1, 'CPUs.'    
            sampler = emcee.EnsembleSampler(settings['nwalkers'], 
                                            ndim, 
                                            lnprob, 
                                            args=(bounds, params, fitkeys, settings), 
                                            pool=pool, 
                                            backend=backend)
            run_mcmc(sampler)
    else:
        sampler = emcee.EnsembleSampler(settings['nwalkers'],
                                        ndim,
                                        lnprob,
                                        args=(bounds, params, fitkeys, settings),
                                        backend=backend)
        run_mcmc(sampler)
        
    return sampler



###############################################################################
#::: update params with MCMC results
###############################################################################
def update_params_with_MCMC_results(settings, params, fitkeys, sampler):
    '''
    read MCMC results and update params
    '''
    samples = sampler.get_chain(flat=True, discard=settings['burn_steps']/settings['thin_by'])

    buf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))                                         
    theta_mcmc = [ item[0] for item in buf ]
    theta_mcmc_ul = [ item[1] for item in buf ]
    theta_mcmc_ll = [ item[2] for item in buf ]
    params = update_params(theta_mcmc, params, fitkeys)
    params_ul = update_params(theta_mcmc_ul, params, fitkeys)
    params_ll = update_params(theta_mcmc_ll, params, fitkeys)
    
    return params, params_ul, params_ll

    
###############################################################################
#::: plot
###############################################################################
def plot(settings, params, styles, theta_0=None, init_err=None, sampler=None, fitkeys=None, planet='b', show_all_data=True):
    '''
    Inputs:
    -------
    style : str
        'data_d', 'initial_guess_d', 'MCMC_results_d'
        'data_phase', 'initial_guess_phase', 'MCMC_results_phase'
        'data_phasezoom', 'initial_guess_phasezoom', 'MCMC_results_phasezoom'
        
    theta_0, init_err, sampler, fitkeys : (optional)
        only needed if you want to plot 'initial_guess_*' or 'MCMC_results_*'
    '''
    
    instruments = settings['inst_phot']+settings['inst_rv']    
    N_inst = len(instruments)
    keys = ['flux']*len(settings['inst_phot']) + ['rv']*len(settings['inst_rv'])
    
    fig, axes = plt.subplots(N_inst,len(styles),figsize=(6*3,4*N_inst))
    axes = np.atleast_2d(axes)
                
    #TODO: not clean, just a hack to have the same samples for all inst & phase-foldings
    #TODO: rewrite the plot function to be more efficient, and handle such things
    samples = draw_samples_for_plot(styles[0], theta_0, init_err, sampler, settings, fitkeys)

    for i,inst in enumerate(instruments):
        for j,style in enumerate(styles):
            plot_1(axes[i,j], settings, params, style, inst, keys[i], samples, fitkeys, planet)

    plt.tight_layout()
    return fig, axes



def draw_samples_for_plot(style, theta_0, init_err, sampler, settings, fitkeys):
    if ('initial_guess' in style) & (theta_0 is not None) & (init_err is not None):
        return draw_init_samples_for_plot(theta_0, init_err)
    elif ('MCMC_results' in style) & (sampler is not None) & (fitkeys is not None):
        return draw_MCMC_samples_for_plot(sampler, settings, fitkeys)
    else:
        raise ValueError('Need to pass the right variables to "plot" in order to plot samples.')


def draw_init_samples_for_plot(theta_0, init_err, Nsamples=20):
    samples = theta_0 + init_err * np.random.randn(Nsamples, len(theta_0))    
    return samples



def draw_MCMC_samples_for_plot(sampler, settings, fitkeys, Nsamples=20):
    samples = sampler.get_chain(flat=True, discard=settings['burn_steps']/settings['thin_by'])
    samples = samples[np.random.randint(len(samples), size=20)]
    return samples



###############################################################################
#::: plot_1 (helper function)
###############################################################################
def plot_1(ax, settings, params, style, inst, key, samples, fitkeys, planet):
    '''
    Inputs:
    -------
        planet : str (optional)
            only needed if style=='_phase' or '_phasezoom'
            None, 'b', 'c', etc.
    '''
    global data
    
    if key=='flux': ylabel='Flux'
    elif key=='rv': ylabel='RV (km/s)'
    
    model_y = calculate_model_y(data[inst]['time'], params, settings, inst, key)
    y_res = calculate_y_res(params, settings, inst, key, model_y)
    y = y_res + model_y
    yerr = data[inst]['err_scales_'+key]*(1./np.sqrt(params['inv_sigma2_'+key+'_'+inst]))


    #::: not phased
    if 'phase' not in style:
        
        #data, not phase
        ax.errorbar( data[inst]['time'], y, yerr=yerr, fmt='b.', capsize=0, rasterized=True )
        ax.set(xlabel='Time (d)', ylabel=ylabel, title=inst)
    
        #model, not phased
        if ( ('initial_guess' in style) | ('MCMC_results' in style) ) & (samples is not None) & (fitkeys is not None):
            if (data[inst]['time'][-1] - data[inst]['time'][-1] < 10): #less than 10 days of data
                dt = 5./24./60. #5 min resolution
            else:
                dt = 30./24./60. #30 min resolution
            model_time = np.arange( data[inst]['time'][0], data[inst]['time'][-1], dt) 
            for i in range(samples.shape[0]):
                s = samples[i,:]
                p = update_params(s, params, fitkeys)
                model_y = calculate_model_y(model_time, p, settings, inst, key)
                ax.plot( model_time, model_y, 'r-', alpha=0.1, zorder=10, rasterized=True )
            
            
    
    #::: phased - and optionally zoomed
    else:

        if 'zoom' not in style: zoomfactor = 1.
        else: zoomfactor = params[planet+'_period']*24.
                
        #data, phased
        phase_time, phase_y, phase_y_err, _, phi = lct.phase_fold(data[inst]['time'], y, params[planet+'_period'], params[planet+'_epoch'], dt = 0.002, ferr_type='meansig', ferr_style='sem', sigmaclip=True)    
        #::: plot the binned points only if there are at least 2 points per bin on average
        if len(phase_time) <= 0.5*len(phi):
            ax.plot( phi*zoomfactor, y, 'b.', color='lightgrey', rasterized=True )
            ax.errorbar( phase_time*zoomfactor, phase_y, phase_y_err, fmt='b.', capsize=0, rasterized=True )
        else:
            ax.errorbar( phi*zoomfactor, y, yerr=yerr, fmt='b.', capsize=0, rasterized=True)    
        ax.set(xlabel='Phase', ylabel=ylabel, title=inst+', planet '+planet)
        

        #model, phased
        if ( ('initial_guess' in style) | ('MCMC_results' in style) ) & (samples is not None) & (fitkeys is not None):
            model_phase = np.linspace( -0.25, 0.75, 1000)
            for i in range(samples.shape[0]):
                s = samples[i,:]
                p = update_params(s, params, fitkeys)
                model_y = calculate_model_y(model_phase, p, settings, inst, key, phased=True)
                ax.plot( model_phase*zoomfactor, model_y, 'r-', alpha=0.1, zorder=10, rasterized=True )
         
        if 'zoom' in style: ax.set( xlim=[-4,4], xlabel=r'$\mathrm{ T - T_0 \ (h) }$' )



###############################################################################
#::: plot the MCMC chains
###############################################################################
def plot_MCMC_chains(data, params, fitkeys, settings, sampler):
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    ndim = chain.shape[2]
    labels = fitkeys
    
    #plot chains; emcee_3.0.0 format = (nsteps, nwalkers, nparameters)
    fig, axes = plt.subplots(ndim+1, 1, figsize=(6,3*ndim) )
    
    #::: plot the lnprob_values; emcee_3.0.0 format = (nsteps, nwalkers)
    axes[0].plot(log_prob, '-', rasterized=True)
    mini = np.min(log_prob[settings['burn_steps']/settings['thin_by']:,:])
    maxi = np.max(log_prob[settings['burn_steps']/settings['thin_by']:,:])
    axes[0].set( ylabel='lnprob', xlabel='steps', rasterized=True,
                 ylim=[mini, maxi] )
    
    #:::plot all chains of parameters
    for i in range(ndim):
        ax = axes[i+1]
        ax.set(ylabel=labels[i], xlabel='steps')
        ax.plot(chain[:,:,i], '-', rasterized=True)
        ax.axvline( settings['burn_steps']/settings['thin_by'], color='k', linestyle='--' )
    
    plt.tight_layout()
    return fig, axes
    
    
    
###############################################################################
#::: plot the MCMC corner plot
###############################################################################
def plot_MCMC_corner(fitkeys, settings, sampler):
    samples = sampler.get_chain(flat=True, discard=settings['burn_steps']/settings['thin_by'])
    ndim = len(fitkeys)
    
    fig = corner(samples, 
            labels = fitkeys,
             range = [0.999]*ndim,
             quantiles=[0.15865, 0.5, 0.84135],
             show_titles=True, title_kwargs={"fontsize": 14})
            
    return fig




###############################################################################
#::: save table
###############################################################################
def save_table(outdir, settings, params, fitkeys, allkeys, sampler):
    params, params_ll, params_ul = update_params_with_MCMC_results(settings, params, fitkeys, sampler)
    
    with open( os.path.join(outdir,'results.csv'), 'wb' ) as f:
        f.write('############ Fitted parameters ############\n')
        for i, key in enumerate(allkeys):
            if key not in fitkeys:
                f.write(key + ',' + str(params[key]) + ',' + '(fixed),\n')
            else:
                f.write(key + ',' + str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]) + '\n' )
   
        


###############################################################################
#::: save Latex table
###############################################################################
def save_latex_table(outdir, settings, params, fitkeys, allkeys, labels, units, sampler):
    params, params_ll, params_ul = update_params_with_MCMC_results(settings, params, fitkeys, sampler)
    label = 'none'
    
#    derived_samples['a_AU'] = derived_samples['a']*0.00465047 #from Rsun to AU
        
    with open(os.path.join(outdir,'latex_table.txt'),'wb') as f,\
         open(os.path.join(outdir,'latex_cmd.txt'),'wb') as f_cmd:
            
             
        f.write('parameter & value & unit & fit/fixed \\\\ \n')
        f.write('\\hline \n')
        f.write('\\multicolumn{4}{c}{\\textit{Fitted parameters}} \\\\ \n')
        f.write('\\hline \n')
        
        for i, key in enumerate(allkeys):
            if key not in fitkeys:                
                value = str(params[key])
                f.write(labels[i] + ' & $' + value + '$ & '  + units[i] + '& (fixed) \\\\ \n')            
                f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{'+label+'$='+value+'$} \n')

            else:            
                value = latex_printer.round_tex(params[key], params_ll[key], params_ul[key])
                f.write(labels[i] + ' & $' + value + '$ & ' + units[i] + '& \\\\ \n' )
                f_cmd.write('\\newcommand{\\'+key.replace("_", "")+'}{'+label+'$='+value+'$} \n')

    
    
    
###############################################################################
#::: run
###############################################################################
def run(datadir, fast_fit=False, continue_old_run=False):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
    fast_fit : bool (optional; default is False)
        if False: 
            use all photometric data for the fit
        if True: 
            only use photometric data in an 8h window around the transit 
            requires a good initial guess of the epoch and period
    continue_olf_run : bool (optional; default is False)
        if False:
            overwrite any previously created files
        if True:
            continue writing into the pre-existing chain (datadir/results/save.h5)
            once done, it will still overwrite the results files
            
    Outputs:
    --------
    This will output some information into the console, 
    and create output files into datadir/results/
    '''
    global data

    print 'Starting...'    
    
    ###############################################################################
    #::: load settings, data and input params
    #::: set output directory
    #::: plot & show the initial guess & settings
    ###############################################################################
    settings, theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units, outdir = init(datadir, fast_fit)
    

    ###############################################################################
    #::: plot & show the initial guess & settings
    ###############################################################################
    show_initial_guess(datadir, fast_fit=fast_fit)
    

    ###############################################################################
    #::: run MCMC fit
    ###############################################################################
    print 'Running MCMC fit...'
    sampler = MCMC_fit(outdir, theta_0, init_err, bounds, params, fitkeys, settings, continue_old_run)
    
    
    ###############################################################################
    #::: create all the output
    ###############################################################################
    analyse_output(datadir, fast_fit=fast_fit) #fast_fit=False here, in order to show the full data    
    
    print 'Done.'
    
    
    
###############################################################################
#::: init
###############################################################################
def init(datadir, fast_fit):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
    fast_fit : bool (optional; default is False)
        if False: 
            use all photometric data for the plot
        if True: 
            only use photometric data in an 8h window around the transit 
            requires a good initial guess of the epoch and period
            
    Returns:
    --------
    All the variables needed for allesfitter.MCMC_fit
    '''
    global data
    
    settings = load_settings(datadir)
    theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units = load_params(datadir)
    data = load_data(datadir, settings, params=params, fast_fit=fast_fit)  

    outdir = os.path.join(datadir,'results')
    if not os.path.exists(outdir): os.makedirs(outdir)

    return settings, theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units, outdir
    


###############################################################################
#::: show initial guess
###############################################################################
def show_initial_guess(datadir, fast_fit=False):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
    fast_fit : bool (optional; default is False)
        if False: 
            use all photometric data for the plot
        if True: 
            only use photometric data in an 8h window around the transit 
            requires a good initial guess of the epoch and period
            
    Outputs:
    --------
    This will output information into the console, 
    and create a file called datadir/results/initial_guess.pdf
    '''
    global data
    
    settings, theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units, outdir = init(datadir, fast_fit)    
    
    print '\nSettings & intitial guess:'
    print '--------------------------'
    for i, key in enumerate(allkeys):
        if key in fitkeys: 
            print '{0: <20}'.format(key), '{0: <15}'.format(params[key]), '{0: <5}'.format('free')
        else: 
            print '{0: <20}'.format(key), '{0: <15}'.format(params[key]), '{0: <5}'.format('set')
            
            
    print '\nFit parameters:'
    print '--------------------------'
    print 'ndim =', len(theta_0)
    for i, key in enumerate(fitkeys):
        print '{0: <20}'.format(key), '{0: <15}'.format(theta_0[i]), '{0: <30}'.format(bounds[i])
        
    print '\nLikelihoods:'
    print '--------------------------'
    print 'lnprior:\t', lnprior(theta_0, bounds)
    print 'lnlike: \t', lnlike(theta_0, params, fitkeys, settings)
    print 'lnprob: \t', lnprob(theta_0, bounds, params, fitkeys, settings)        

    
    fig, axes = plot(settings, params, ['initial_guess_d', 'initial_guess_phase', 'initial_guess_phasezoom'], theta_0=theta_0, init_err=init_err, fitkeys=fitkeys)
    fig.savefig( os.path.join(outdir,'initial_guess.pdf'), bbox_inches='tight' )
    plt.close(fig)
    
    

###############################################################################
#::: analyse the output from .h5 file
###############################################################################
def analyse_output(datadir, fast_fit=False, QL=False):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
    fast_fit : bool (optional; default is False)
        if False: 
            use all photometric data for the plot
        if True: 
            only use photometric data in an 8h window around the transit 
            requires a good initial guess of the epoch and period
    QL : bool (optional; default is False)
        if False: 
            read out the chains from datadir/results/save.h5
            WARNING: this breaks any running MCMC that tries writing into this file!
        if True: 
           allows a quick look (QL) at the MCMC results while MCMC is still running
           copies the chains from results/save.h5 file over to QL/save.h5 and opens that file
           set burn_steps automatically to half the chain length
            
    Outputs:
    --------
    This will output information into the console, and create a output files 
    into datadir/results/ (or datadir/QL/ if QL==True)    
    '''
    global data
    
    settings, theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units, outdir = init(datadir, fast_fit)
    
    if QL:
        outdir = os.path.join( datadir,'QL' )
        if not os.path.exists( outdir ): os.makedirs( outdir )
        copyfile(os.path.join(datadir,'results','save.h5'), 
                 os.path.join(outdir,'save.h5'))
    
    reader = emcee.backends.HDFBackend( os.path.join(outdir,'save.h5'), read_only=True )

    if QL:
        settings['total_steps'] = reader.get_chain().shape[0]
        settings['burn_steps'] = int(0.75*settings['thin_by']*reader.get_chain().shape[0])

    try:
        print 'Autocorrelation times:', reader.get_autocorr_time(discard=settings['burn_steps'])
    except:
        print 'Autocorrelation times:', 'chains not converged yet'

    fig, axes = plot(settings, params, ['MCMC_results_d', 'MCMC_results_phase', 'MCMC_results_phasezoom'], fitkeys=fitkeys, sampler=reader)
    fig.savefig( os.path.join(outdir,'fit.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(fig)
    
    fig, axes = plot_MCMC_chains(data, params, fitkeys, settings, reader)
    fig.savefig( os.path.join(outdir,'chains.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(fig)

    fig = plot_MCMC_corner(fitkeys, settings, reader)
    fig.savefig( os.path.join(outdir,'corner.jpg'), dpi=50, bbox_inches='tight' )
    plt.close(fig)

    save_table(outdir, settings, params, fitkeys, allkeys, reader)

    save_latex_table(outdir, settings, params, fitkeys, allkeys, labels, units, reader)
    
    print 'Done. For all outputs, see', outdir