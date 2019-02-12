#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:54:58 2018

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

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skewnorm




np.random.seed(42)




def simulate_PDF(median, lower_err, upper_err, size=1, plot=True):
    '''
    Simulates a draw of posterior samples from a value and asymmetric errorbars
    by assuming the underlying distribution is a skewed normal distribution.
    
    Developed to estimate PDFs from literature exoplanet parameters that did not report their MCMC chains.
    
    Inputs:
    -------
    median : float
        the median value that was reported
    lower_err : float
        the lower errorbar that was reported
    upper_err : float
        the upper errorbar that was reported
    size : int
        the number of samples to be drawn
        
    Returns:
    --------
    samples : array of float
        the samples drawn from the simulated skrewed normal distribution
    '''
    
    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
    samples = skewnorm.rvs(alpha, loc=sigma, scale=omega, size=size)
    
    if plot==False:
        return samples

    else:
        lower_err = np.abs(lower_err)
        upper_err = np.abs(upper_err)
        x = np.arange(median-4*lower_err, median+4*upper_err, 0.01)
        fig = plt.figure()
        for i in range(3): plt.axvline([median-lower_err, median, median+upper_err][i], color='k', lw=2)
        plt.plot( x, skewnorm.pdf(x, alpha, loc=sigma, scale=omega), 'r-', lw=2 )
        fit_percentiles = skewnorm.ppf([0.16, 0.5, 0.84], alpha, loc=sigma, scale=omega)
        for i in range(3): plt.axvline(fit_percentiles[i], color='r', ls='--', lw=2)
        plt.hist(samples, density=True, color='red', alpha=0.5)
        return samples, fig



def calculate_skewed_normal_params(median, lower_err, upper_err):
    '''
    Fits a screwed normal distribution via its CDF to the [16,50,84]-percentiles
    
    Inputs:
    -------
    median : float
        the median value that was reported
    lower_err : float
        the lower errorbar that was reported
    upper_err : float
        the upper errorbar that was reported
    size : int
        the number of samples to be drawn
        
    Returns:
    --------
    sigma : float
        the mean of the fitted skewed normal distribution
    omega : float
        the std of the fitted skewed normal distribution
    alpha : float
        the skewness parameter
    '''
    
    lower_err = np.abs(lower_err)
    upper_err = np.abs(upper_err)
    
    def fake_lnlike(p):
        sigma, omega, alpha = p
        eq1 = skewnorm.ppf(0.5, alpha, loc=sigma, scale=omega) - median
        eq2 = skewnorm.ppf(0.16, alpha, loc=sigma, scale=omega) - (median-lower_err)
        eq3 = skewnorm.ppf(0.84, alpha, loc=sigma, scale=omega) - (median+upper_err)
        fake_lnlike = np.log( eq1**2 + eq2**2 + eq3**2 )
#        print fake_lnlike
        return fake_lnlike

    std = np.mean([lower_err, upper_err])
    initial_guess = (median, std, 0) #sigma, omega, alpha 
#    print 'initial_guess:', initial_guess
    sol = minimize(fake_lnlike, initial_guess, bounds=[(None,None), (0,None), (None,None)])
    sigma, omega, alpha = sol.x
#    print 'fake_lnlike:', fake_lnlike((sigma, omega, alpha))
#    print 'Parameters:'
#    print 'sigma', sigma
#    print 'omega', omega
#    print 'alpha', alpha
    
    return sigma, omega, alpha

    
    
if __name__ == '__main__':
    '''
    For testing, simulate a skewed normal distribution with parameters
    alpha_0, loc=sigma_0, scale=omega_0,
    perform the fit and compare the resulting PDF
    '''
    
    ###########################################################################
    # ::: simulate a skewed normal PDF, measure median and errors, and fit
    ###########################################################################
#    sigma_0 = 0.
#    omega_0 = 5.
#    alpha_0 = 2
#    
#    x = np.arange(-20+sigma_0,20+sigma_0,0.01)
#    y = skewnorm.pdf(x, alpha_0, loc=sigma_0, scale=omega_0)
#    plt.figure()
#    plt.plot( x, y, 'b-', lw=4 )
#    
#    median = skewnorm.ppf(0.5, alpha_0, loc=sigma_0, scale=omega_0)
#    lower_err = median - skewnorm.ppf(0.16, alpha_0, loc=sigma_0, scale=omega_0)
#    upper_err = skewnorm.ppf(0.84, alpha_0, loc=sigma_0, scale=omega_0) - median
#    plt.axvline(median)
#    plt.axvline(median-lower_err)
#    plt.axvline(median+upper_err)
#    
#    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
#    plt.plot( x, skewnorm.pdf(x, alpha, loc=sigma, scale=omega), 'r--', lw=2 )



    ###########################################################################
    # ::: example inclination
    ###########################################################################
#    (median, lower_err, upper_err) = (84.3, -2.0, 1.3)
#    samples, fig = simulate_posterior_samples(median, lower_err, upper_err, size=1, plot=True)
#    print(np.percentile(samples, [16,50,84]))


#    lower_err = np.abs(lower_err)
#    upper_err = np.abs(upper_err)
#    
#    x = np.arange(median-4*lower_err, median+4*upper_err, 0.01)
#    plt.figure()
#    plt.axvline(median, color='k', lw=2)
#    plt.axvline(median-lower_err, color='k', lw=2)
#    plt.axvline(median+upper_err, color='k', lw=2)
#    
#    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
#    plt.plot( x, skewnorm.pdf(x, alpha, loc=sigma, scale=omega), 'r-', lw=2 )
#    
#    fit_percentiles = skewnorm.ppf([0.16, 0.5, 0.84], alpha, loc=sigma, scale=omega)
#    for i in range(3): plt.axvline(fit_percentiles[i], color='r', ls='--', lw=2)
#    
#    rvs = simulate_posterior_samples(median, lower_err, upper_err, size=1000)
#    plt.hist(rvs, density=True, color='lightblue')
