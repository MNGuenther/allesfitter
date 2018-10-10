#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:41:29 2018

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
#import matplotlib.pyplot as plt
import ellc
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import numpy.polynomial.polynomial as poly
#from pprint import pprint
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 
warnings.filterwarnings('ignore', category=RuntimeWarning) 
try:
    import celerite
    from celerite import terms
except ImportError:
    warnings.warn("Cannot import package 'celerite', thus 'hybrid_GP' baseline models will not be supported.")

#allesfitter modules
from . import config
#from .switchlang.switchlang import switch




###############################################################################
#::: convert input params into ellc params
###############################################################################  
def update_params(theta, phased=False):
#    global config.BASEMENT
    params = config.BASEMENT.params.copy()
    
    #::: first, sync over from theta
    for i, key in enumerate(config.BASEMENT.fitkeys):
        params[key] = theta[i]   
    
    
    #::: phase-folded? (it's important to have this before calculating the semi-major axis!)
    if phased:
        for planet in config.BASEMENT.settings['planets_all']:
            params[planet+'_epoch'] = 0.
            params[planet+'_period'] = 1.
    
    
    #::: general params (used for both photometry and RV)
    for planet in config.BASEMENT.settings['planets_all']:
        #::: incl
        params[planet+'_incl'] = np.arccos( params[planet+'_cosi'] )/np.pi*180.
        
        
    #::: photometry
    for planet in config.BASEMENT.settings['planets_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            #::: errors
            key='flux'
            params['inv_sigma2_'+key+'_'+inst] = 1. / ( np.exp( params['log_err_'+key+'_'+inst] ) )**2
            
            #::: R_1/a and R_2/a
            params[planet+'_radius_1'] = params[planet+'_rsuma'] / (1. + params[planet+'_rr'])
            params[planet+'_radius_2'] = params[planet+'_radius_1'] * params[planet+'_rr']
            
            if config.BASEMENT.settings['ld_law_'+inst] == 'lin':
                params['ldc_1_'+inst] = params['ldc_q1_'+inst]
                
            elif config.BASEMENT.settings['ld_law_'+inst] == 'quad':
                ldc_u1 = 2.*np.sqrt(params['ldc_q1_'+inst]) * params['ldc_q2_'+inst]
                ldc_u2 = np.sqrt(params['ldc_q1_'+inst]) * (1. - 2.*params['ldc_q2_'+inst])
                params['ldc_1_'+inst] = [ ldc_u1, ldc_u2 ]
                
            elif config.BASEMENT.settings['ld_law_'+inst] == 'sing':
                raise ValueError("Sorry, I have not yet implemented the Sing limb darkening law.")
                
            else:
                raise ValueError("Currently only 'lin' and 'quad' limb darkening implemented.")
    
    
    #::: RV
    for planet in config.BASEMENT.settings['planets_rv']:
        for inst in config.BASEMENT.settings['inst_rv']:
            #::: errors
            key='rv'
            params['inv_sigma2_'+key+'_'+inst] = 1. / ( np.exp( params['log_err_'+key+'_'+inst] ) )**2
        
        #::: semi-major axis
        ecc = params[planet+'_f_s']**2 + params[planet+'_f_c']**2
        a_1 = 0.019771142 * params[planet+'_K'] * params[planet+'_period'] * np.sqrt(1. - ecc**2)/np.sin(params[planet+'_incl']*np.pi/180.)
        params[planet+'_a'] = (1.+1./params[planet+'_q'])*a_1
        
#        print('\n------')
#        print(planet, inst)
#        print('fsc:', params[planet+'_f_s'], params[planet+'_f_c'])
#        print('ecc', ecc)
#        print(params[planet+'_K'], params[planet+'_period'])
#        print('a1', a_1)
#        print('a', params[planet+'_a'])

    return params



###############################################################################
#::: flux fct
###############################################################################
def flux_fct(params, inst, planet, xx=None):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
#    global config.BASEMENT
    
    if xx is None:
        xx = config.BASEMENT.data[inst]['time']
        
    model_flux = ellc.lc(
                      t_obs =       xx, 
                      radius_1 =    params[planet+'_radius_1'], 
                      radius_2 =    params[planet+'_radius_2'], 
                      sbratio =     params[planet+'_sbratio_'+inst], 
                      incl =        params[planet+'_incl'], 
                      light_3 =     params['light_3_'+inst],
                      t_zero =      params[planet+'_epoch'],
                      period =      params[planet+'_period'],
                      f_c =         params[planet+'_f_c'],
                      f_s =         params[planet+'_f_s'],
                      ldc_1 =       params['ldc_1_'+inst],
#                      ldc_2 = ldc_2,
                      ld_1 =        config.BASEMENT.settings['ld_law_'+inst],
#                      ld_2 = 'quad' 
                      )
             
    return model_flux
    


###############################################################################
#::: rv fct
###############################################################################
def rv_fct(params, inst, planet, xx=None):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
#    global config.BASEMENT
    
    if xx is None:
        xx = config.BASEMENT.data[inst]['time']
    
    model_rv1, model_rv2 = ellc.rv(
                      t_obs =   xx, 
                      incl =    params[planet+'_incl'], 
                      t_zero =  params[planet+'_epoch'],
                      period =  params[planet+'_period'],
                      a =       params[planet+'_a'],
                      f_c =     params[planet+'_f_c'],
                      f_s =     params[planet+'_f_s'],
                      q =       params[planet+'_q'],
                      flux_weighted = False
                      )
    
    return model_rv1, model_rv2



###############################################################################
#::: calculate residuals
###############################################################################  
def calculate_lnlike(params, inst, key):
    
    if config.BASEMENT.settings['baseline_'+key+'_'+inst] != 'sample_GP':
        residuals = calculate_residuals(params, inst, key)
        inv_sigma2_w = calculate_inv_sigma2_w(params, inst, key, residuals=residuals)
        return -0.5*(np.nansum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w)))
    
    else:
        model = calculate_model(params, inst, key)
        
        x = config.BASEMENT.data[inst]['time']
        y = config.BASEMENT.data[inst][key] - model
        
        #TODO: not very elegant... 
        #TODO: restructure stuff to avoid this 2x computation and spagetthi code
        inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * params['inv_sigma2_'+key+'_'+inst]
        yerr = 1./np.sqrt(inv_sigma2_w)
        
#        X = np.column_stack((x, y, yerr))
#        np.savetxt('obama.txt', X)
        
        kernel = terms.Matern32Term(log_sigma=params['baseline_gp1_'+key+'_'+inst], 
                                    log_rho=params['baseline_gp2_'+key+'_'+inst])
        gp = celerite.GP(kernel)
        gp.compute(x, yerr=yerr)
        lnlike = gp.log_likelihood(y)
        
    #    baseline2 = gp.predict(y, x)[0]
    #    plt.figure()
    #    plt.plot(x,y,'k.', color='grey')
    #    plt.plot(xx,baseline,'r-', lw=2)
    #    plt.plot(x,baseline2,'ro', lw=2)
    #    plt.title(inst+' '+key+' '+str(gp.get_parameter_vector()))
    #    plt.show()
    #    raw_input('press enter to continue')
    
        return lnlike
    
    

        
        

###############################################################################
#::: calculate residuals
###############################################################################  
def calculate_residuals(params, inst, key):
    '''
    Note:
    -----
    No 'xx' here, because residuals can only be calculated on given data
    (not on an arbitrary xx grid)
    '''       
#    global config.BASEMENT
    
#        then = timer()
    model = calculate_model(params, inst, key)
    baseline = calculate_baseline(params, inst, key, model=model)
    residuals = config.BASEMENT.data[inst][key] - model - baseline
#        now = timer() #Time after it finished
#        print("calculate_residuals took: ", now-then, " seconds for "+inst+" "+key)
    return residuals


    
###############################################################################
#::: calculate model
###############################################################################      
def calculate_model(params, inst, key, xx=None):
#    global config.BASEMENT
#        then = timer()
        
    if key=='flux':
        depth = 0.
        for planet in config.BASEMENT.settings['planets_phot']:
            depth += ( 1. - flux_fct(params, inst, planet, xx=xx) )
        model_flux = 1. - depth
#            now = timer() #Time after it finished
#            print("calculate_model flux took: ", now-then, " seconds for "+inst)
        return model_flux
    
    elif key=='rv':
        model_rv = 0.
        for planet in config.BASEMENT.settings['planets_rv']:
            model_rv += rv_fct(params, inst, planet, xx=xx)[0]
#            now = timer() #Time after it finished
#            print("calculate_model rv took: ", now-then, " seconds for "+inst)
        return model_rv
    
    elif (key=='centdx') | (key=='centdy'):
        raise ValueError("Fitting for 'centdx' and 'centdy' not yet implemented.")
        #TODO
        
    else:
        raise ValueError("Variable 'key' has to be 'flux', 'rv', 'centdx', or 'centdy'.")



###############################################################################
#::: calculate baseline
###############################################################################   
def calculate_baseline(params, inst, key, model=None, xx=None):
    '''
    Inputs:
    -------
    xx : array of float (optional; default=None)
        if given, evaluate the baseline fit on the xx values 
        (e.g. a finer time grid for plotting)
        else, it's the same as data[inst]['time']
        
    Returns: 
    --------
    baseline : array of float
        the baseline evaluate on the grid x (or xx, if xx!=None)
    '''
#    global config.BASEMENT
    
    if model is None: model = calculate_model(params, inst, key, xx=None)
    x = config.BASEMENT.data[inst]['time']
    y = config.BASEMENT.data[inst][key] - model
    yerr_weights = config.BASEMENT.data[inst]['err_scales_'+key]
    if xx is None:  xx = 1.*x
    
    '''
    x : array of float
        time stamps of the data
    y : array of float
        y = data_y - model_y (!!!)
        i.e., the values that you want to constrain the baseline on
    yerr_weights : array of float
        normalized error weights on y
    '''
    
    baseline_method = config.BASEMENT.settings['baseline_'+key+'_'+inst]
    return baseline_switch[baseline_method](x, y, yerr_weights, xx, params, inst, key)


#    if config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'hybrid_offset':
#        return baseline_hybrid_offset(y, yerr_weights)
#    
#    elif 'hybrid_poly_' in config.BASEMENT.settings['baseline_'+key+'_'+inst]:
#        return baseline_hybrid_poly(x, y, yerr_weights, xx, key, inst)
#        
#    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'hybrid_spline':
#        return baseline_hybrid_spline(x, y, yerr_weights, xx)
#
#    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'hybrid_GP':
#        return baseline_hybrid_GP(x, y, xx)
#
#    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_offset':
#        return baseline_sample_offset(xx, key, inst, params)
#    
#    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_linear':
#        return baseline_sample_linear()
#        
#    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_GP':
#        return baseline_sample_GP()
#        
#    else:
#        baseline_raise_error(inst, key)
    
    
    
    
###########################################################################
#::: hybrid_offset (like Gillon+2012, but only remove mean offset)
###########################################################################
def baseline_hybrid_offset(*args):
#    y, yerr_weights = args
    x, y, yerr_weights, xx, params, inst, key = args
    inv_sigma = 1./yerr_weights
    weights = inv_sigma/np.nanmean(inv_sigma) #weights should be normalized inv_sigma
    ind = np.isfinite(y) #np.average can't handle NaN
    return np.average(y[ind], weights=weights[ind])
 

    
###########################################################################
#::: hybrid_poly (like Gillon+2012)
###########################################################################   
def baseline_hybrid_poly(*args):
#    x, y, yerr_weights, xx, key, inst = args
    x, y, yerr_weights, xx, params, inst, key = args
#        then = timer()
    polyorder = config.BASEMENT.settings['baseline_'+key+'_'+inst][-1]
    xx = (xx - x[0])/x[-1] #polyfit needs the xx-axis scaled to [0,1], otherwise it goes nuts
    x = (x - x[0])/x[-1] #polyfit needs the x-axis scaled to [0,1], otherwise it goes nuts
    if polyorder>=0:
        inv_sigma = 1./yerr_weights
        weights = inv_sigma/np.nanmean(inv_sigma) #weights should be normalized inv_sigma
        ind = np.isfinite(y) #polyfit can't handle NaN
        params_poly = poly.polyfit(x[ind],y[ind],polyorder,w=weights[ind]) #WARNING: returns params in reverse order than np.polyfit!!!
        baseline = poly.polyval(xx, params_poly) #evaluate on xx (!)
    else:
        raise ValueError("'polyorder' has to be > 0.")
#        now = timer() #Time after it finished
#        print("hybrid_poly took: ", now-then, " seconds.")
    return baseline    



###########################################################################
#::: hybrid_spline (like Gillon+2012, but with a cubic spline)
###########################################################################
def baseline_hybrid_spline(*args):
#    x, y, yerr_weights, xx = args
    x, y, yerr_weights, xx, params, inst, key = args
#        then = timer()
    inv_sigma = 1./yerr_weights
    weights = inv_sigma/np.nanmean(inv_sigma) #weights should be normalized inv_sigma
    ind = np.isfinite(y) #mask NaN
    spl = UnivariateSpline(x[ind],y[ind],w=weights[ind],s=np.sum(weights[ind]))
    baseline = spl(xx)
    
#        print 'fitting splines'
#        plt.figure()
#        plt.plot(x,y,'k.', color='grey')
#        plt.plot(xx,baseline,'r-', lw=2)
#        plt.show()
#        raw_input('press enter to continue')
#        now = timer() #Time after it finished
#        print("hybrid_spline took: ", now-then, " seconds.")
    return baseline   

     
    
###########################################################################
#::: hybrid_GP (like Gillon+2012, but with a GP)
###########################################################################           
def baseline_hybrid_GP(*args):
#    x, y, xx = args
    x, y, yerr_weights, xx, params, inst, key = args
    yerr = np.nanstd(y) #overwrite yerr; works better for removing smooth global trends
    kernel = terms.Matern32Term(log_sigma=1., log_rho=1.)
    gp = celerite.GP(kernel, mean=np.nanmean(y)) 
    gp.compute(x, yerr=yerr) #constrain on x/y/yerr
     
    def neg_log_like(gp_params, y, gp):
        gp.set_parameter_vector(gp_params)
        return -gp.log_likelihood(y)
    
    def grad_neg_log_like(gp_params, y, gp):
        gp.set_parameter_vector(gp_params)
        return -gp.grad_log_likelihood(y)[1]
    
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(y, gp))
    gp.set_parameter_vector(soln.x)
    
    baseline = gp.predict(y, xx)[0] #constrain on x/y/yerr, evaluate on xx (!)
    return baseline 



###########################################################################
#::: sample_offset
########################################################################### 
def baseline_sample_offset(*args):
#    xx, key, inst, params = args
    x, y, yerr_weights, xx, params, inst, key = args
    return params['baseline_offset_'+key+'_'+inst] * np.ones_like(xx)
        


###########################################################################
#::: sample_linear
########################################################################### 
def baseline_sample_linear(*args):
    raise ValueError('Not yet implemented.')
        
    
    
###########################################################################
#::: sample_GP
########################################################################### 
def baseline_sample_GP(*args):
    x, y, yerr_weights, xx, params, inst, key = args
    #TODO: not very elegant... restructure stuff to avoid this 2x computation
    inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * params['inv_sigma2_'+key+'_'+inst]
    yerr = 1./np.sqrt(inv_sigma2_w)
    
    kernel = terms.Matern32Term(log_sigma=params['baseline_gp1_'+key+'_'+inst], 
                                log_rho=params['baseline_gp2_'+key+'_'+inst])
    gp = celerite.GP(kernel)
    gp.compute(x, yerr=yerr)
    baseline = gp.predict(y, xx)[0]
    
#    baseline2 = gp.predict(y, x)[0]
#    plt.figure()
#    plt.plot(x,y,'k.', color='grey')
#    plt.plot(xx,baseline,'r-', lw=2)
#    plt.plot(x,baseline2,'ro', lw=2)
#    plt.title(inst+' '+key+' '+str(gp.get_parameter_vector()))
#    plt.show()
#    raw_input('press enter to continue')
    
    return baseline



###########################################################################
#::: raise error
###########################################################################   
def baseline_raise_error(*args):
#    inst, key = args
    x, y, yerr_weights, xx, params, inst, key = args
    raise ValueError('Setting '+'baseline_'+key+'_'+inst+' has to be sample_offset / sample_linear / hybrid_offset / hybrid_poly_1 / hybrid_poly_2 / hybrid_poly_3 / hybrid_pol_4 / hybrid_spline / hybrid_GP, '+\
                     "\nbut is:"+config.BASEMENT.settings['baseline_'+key+'_'+inst])



###########################################################################
#::: baselien_switch
###########################################################################    
baseline_switch = \
    {
    'hybrid_offset' : baseline_hybrid_offset,
    'hybrid_poly_0' : baseline_hybrid_poly,
    'hybrid_poly_1' : baseline_hybrid_poly,
    'hybrid_poly_2' : baseline_hybrid_poly,
    'hybrid_poly_3' : baseline_hybrid_poly,
    'hybrid_poly_4' : baseline_hybrid_poly,
    'hybrid_poly_5' : baseline_hybrid_poly,
    'hybrid_poly_6' : baseline_hybrid_poly,
    'hybrid_spline' : baseline_hybrid_spline,
    'hybrid_GP' :     baseline_hybrid_GP,
    'sample_offset' : baseline_sample_offset,
    'sample_linear' : baseline_sample_linear,
    'sample_GP' :     baseline_sample_GP #only for plotting    
    }
    
    
    
###############################################################################
#::: def calculate inv_sigma2
###############################################################################  
def calculate_inv_sigma2_w(params, inst, key, residuals=None):
    '''
    _w means "weighted", a.k.a. multiplied by data[inst]['err_scales_'+key]**(-2)
    '''
#    global config.BASEMENT
    
    if residuals is None:
        residuals = calculate_residuals(params, inst, key)
        
    
    if config.BASEMENT.settings['error_'+key+'_'+inst].lower() == 'sample': #traditional (sampling in MCMC)
        inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * params['inv_sigma2_'+key+'_'+inst]
        return inv_sigma2_w
    
    
    elif config.BASEMENT.settings['error_'+key+'_'+inst].lower() == 'hybrid': #'hybrid_inv_sigma2'
        
        def neg_log_like(inv_sigma2, inst, key, residuals):
            inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * inv_sigma2
            return + 0.5*(np.nansum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w)))
            
        def grad_neg_log_like(inv_sigma2, inst, key, residuals):
            inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * inv_sigma2
            return np.array( + 0.5*(np.nansum((residuals)**2 - 1./inv_sigma2_w)) )
        
#        guess = params['inv_sigma2_'+key+'_'+inst]
        guess = 1./np.std(residuals)**2 #Warning: this relies on a good initial guess for the model, otherwise std(residuals) will be nuts

        #::: MLE (gradient based)
        soln_MLE = minimize(neg_log_like, guess,
                        method = 'L-BFGS-B', jac=grad_neg_log_like,
                        bounds=[(1e-16,1e+16)], args=(inst, key, residuals))
        
        #::: Diff. Evol.
#        bounds = [(0.001*guess,1000.*guess)]
#        soln_DE = differential_evolution(neg_log_like, bounds, args=(inst, key, residuals))

        inv_sigma2 = soln_MLE.x[0]      
        inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * inv_sigma2

#        print inst, key
#        print '\tguess:', int(guess), 'lnlike:', neg_log_like(guess, inst, key, residuals)
#        print '\tMLE:', int(soln_MLE.x[0]), 'lnlike:', neg_log_like(soln_MLE.x[0], inst, key, residuals) 
#        print '\tDE:', int(soln_DE.x[0]), 'lnlike:', neg_log_like(soln_DE.x[0], inst, key, residuals)
        
        return inv_sigma2_w
    

    else:
        raise ValueError('Setting '+'error_'+key+'_'+inst+' has to be sample / hybrid, '+\
                         "\nbut is:"+params['error_'+key+'_'+inst])