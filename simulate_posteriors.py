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
from scipy.optimize import fsolve
from scipy.stats import norm, skewnorm
import math



def simulate_posterior_samples(median, lower_err, upper_err):
    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
    
    return 0

    
    
def calculate_skewed_normal_params(median, lower_err, upper_err):
    
    def equations(p):
        sigma, omega, alpha = p
        print '\n-----'
        print p
        eq1 = skewnorm.ppf(0.5, alpha, loc=sigma, scale=omega) - median
        eq2 = skewnorm.ppf(0.16, alpha, loc=sigma, scale=omega) - (median-lower_err)
        eq3 = skewnorm.ppf(0.84, alpha, loc=sigma, scale=omega) - (median+upper_err)
        print (eq1, eq2, eq3)
        print '-----'
        return (eq1, eq2, eq3)

    err = np.mean([lower_err, upper_err])
    initial_guess = (median, err**2, 0) #sigma, omega, alpha 
    sigma, omega, alpha = fsolve(equations, initial_guess )
    print 'Equations:', equations((sigma, omega, alpha))
    print 'Parameters:'
    print 'sigma', sigma
    print 'omega', omega
    print 'alpha', alpha
    
    return sigma, omega, alpha
    
    
    
if __name__ == '__main__':
    '''
    For testing, simulate a skewed normal distribution with parameters
    alpha_0, loc=sigma_0, scale=omega_0,
    perform the fit and compare the resulting PDF
    '''
    sigma_0 = 7.
    omega_0 = 2.
    alpha_0 = 9.
    
    x = np.arange(-6+sigma_0,6+sigma_0,0.01)
    y = skewnorm.pdf(x, alpha_0, loc=sigma_0, scale=omega_0)
    plt.figure()
    plt.plot( x, y, 'b-', lw=4 )
    
    median = skewnorm.ppf(0.5, alpha_0, loc=sigma_0, scale=omega_0)
    lower_err = median - skewnorm.ppf(0.16, alpha_0, loc=sigma_0, scale=omega_0)
    upper_err = skewnorm.ppf(0.84, alpha_0, loc=sigma_0, scale=omega_0) - median
    plt.axvline(median)
    plt.axvline(median-lower_err)
    plt.axvline(median+upper_err)
    
    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
    plt.plot( x, skewnorm.pdf(x, alpha, loc=sigma, scale=omega), 'r--', lw=2 )

