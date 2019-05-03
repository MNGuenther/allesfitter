#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:10:28 2018

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
#import matplotlib.pyplot as plt

from .simulate_PDF import simulate_PDF as spdf
#from ..latex_printer import round_tex


    
    
def get_cosi_from_i(i, Nsamples=10000):
    i = spdf(i[0], i[1], i[2], size=Nsamples, plot=False)
    ind_good = np.where( (i>=0) & (i<=90) )[0]
    i = i[ind_good]
    cosi = np.cos(np.deg2rad(i))
    ll, median, ul = np.percentile(cosi, [16,50,84])
    return median, median-ll, ul-median



def get_cosi_from_b(b, a_over_Rstar, Nsamples=10000):
    b = spdf(b[0], b[1], b[2], size=Nsamples, plot=False)    
    a_over_Rstar = spdf(a_over_Rstar[0], a_over_Rstar[1], a_over_Rstar[2], size=Nsamples, plot=False)
    ind_good = np.where( (b>=0) & (a_over_Rstar>0) )[0]
    b = b[ind_good]
    a_over_Rstar = a_over_Rstar[ind_good]
    cosi = b/a_over_Rstar
    ll, median, ul = np.percentile(cosi, [16,50,84])
    return median, median-ll, ul-median



def get_Rsuma_from_a_over_Rstar(a_over_Rstar, Rp_over_Rstar, Nsamples=10000):
    a_over_Rstar = spdf(a_over_Rstar[0], a_over_Rstar[1], a_over_Rstar[2], size=Nsamples, plot=False)
    Rp_over_Rstar = spdf(Rp_over_Rstar[0], Rp_over_Rstar[1], Rp_over_Rstar[2], size=Nsamples, plot=False)
    ind_good = np.where( (Rp_over_Rstar>=0) & (a_over_Rstar>0) )[0]
    Rp_over_Rstar = Rp_over_Rstar[ind_good]
    a_over_Rstar = a_over_Rstar[ind_good]
    Rsuma = 1./a_over_Rstar * (1. + Rp_over_Rstar)
    ll, median, ul = np.percentile(Rsuma, [16,50,84])
    return median, median-ll, ul-median
    


def get_Rsuma_from_Rstar_over_a(Rstar_over_a, Rp_over_Rstar, Nsamples=10000):
    Rstar_over_a = spdf(Rstar_over_a[0], Rstar_over_a[1], Rstar_over_a[2], size=Nsamples, plot=False)
    Rp_over_Rstar = spdf(Rp_over_Rstar[0], Rp_over_Rstar[1], Rp_over_Rstar[2], size=Nsamples, plot=False)
    ind_good = np.where( (Rp_over_Rstar>=0) & (Rstar_over_a>0) )[0]
    Rp_over_Rstar = Rp_over_Rstar[ind_good]
    Rstar_over_a = Rstar_over_a[ind_good]
    Rsuma = Rstar_over_a * (1. + Rp_over_Rstar)
    ll, median, ul = np.percentile(Rsuma, [16,50,84])
    return median, median-ll, ul-median

    
    
def get_sqrtesinw(e, w, Nsamples=10000):
    e = spdf(e[0], e[1], e[2], size=Nsamples, plot=False)
    w = spdf(w[0], w[1], w[2], size=Nsamples, plot=False)
    ind_good = np.where( (e>=0) & (w>=0) & (w<=360) )[0]
    e = e[ind_good]
    w = w[ind_good]
    sqrtesinw = np.sqrt(e) * np.sin(np.deg2rad(w))
    ll, median, ul = np.percentile(sqrtesinw, [16,50,84])
    return median, median-ll, ul-median



def get_sqrtecosw(e, w, Nsamples=10000):
    e = spdf(e[0], e[1], e[2], size=Nsamples, plot=False)
    w = spdf(w[0], w[1], w[2], size=Nsamples, plot=False)
    ind_good = np.where( (e>=0) & (e<=1) & (w>=0) & (w<=360) )[0]
    e = e[ind_good]
    w = w[ind_good]
    sqrtecosw = np.sqrt(e) * np.cos(np.deg2rad(w))
    ll, median, ul = np.percentile(sqrtecosw, [16,50,84])
    return median, median-ll, ul-median



#def get_priors_from_literature(i, a_over_Rstar, Rp_over_Rstar, Nsamples=10000, quiet=False):
#    '''
#    Inputs:
#    -------
#    i : tuple or list
#        the median, lower error (median - 16th percentile), and upper error (84th percentile - median) 
#        of the inclination in the form (median, lower_error, upper_error)
#        
#    a_over_Rstar : tuple or list
#        the median, lower error (median - 16th percentile), and upper error (84th percentile - median) 
#        of a_over_Rstar in the form (median, lower_error, upper_error)
#        
#    Rp_over_Rstar : tuple or list
#        the median, lower error (median - 16th percentile), and upper error (84th percentile - median) 
#        of Rp_over_Rstar in the form (median, lower_error, upper_error)
#        
#    Returns:
#    --------
#    prints the values and error bars for cosi, Rstar_over_a, Rsuma, and Rp_over_a
#    
#    Example:
#    --------
#    i = (82.80, 0.17, 0.17)
#    a_over_Rstar = (5.851, 0.038, 0.037)
#    Rp_over_Rstar = (0.14075, 0.00035, 0.00035)
#    get_priors_from_literature(i, a_over_Rstar, Rp_over_Rstar)
#    '''
#    
#    #::: calculate cosi
#    if i is not None:
#        i, fig = spdf(i[0], i[1], i[2], size=Nsamples, plot=True)
#        plt.title('i')
#        cosi = np.cos(np.deg2rad(i))
#        ll, median, ul = np.percentile(cosi, [16,50,84])
#        if not quiet: print('cosi =', round_tex(median, median-ll, ul-median))
#    
#    
#    #::: calculate (R_star+R_p)/a, R_star/a, and R_p/a
#    if (a_over_Rstar is not None) & (Rp_over_Rstar is not None):
#        a_over_Rstar, fig = spdf(a_over_Rstar[0], a_over_Rstar[1], a_over_Rstar[2], size=Nsamples, plot=True)
#        plt.title('a / R_star')
#        
#        Rp_over_Rstar, fig = spdf(Rp_over_Rstar[0], Rp_over_Rstar[1], Rp_over_Rstar[2], size=Nsamples, plot=True)
#        plt.title('R_p / R_star')
#        
#        Rstar_over_a = 1./a_over_Rstar 
#        ll, median, ul = np.percentile(Rstar_over_a, [16,50,84])
#        if not quiet: print('Rstar_over_a =', round_tex(median, median-ll, ul-median))
#        
#        Rsuma = Rstar_over_a * (1. + Rp_over_Rstar)
#        ll, median, ul = np.percentile(Rsuma, [16,50,84])
#        if not quiet: print('Rsuma =', round_tex(median, median-ll, ul-median))
#        
#        Rp_over_a = Rp_over_Rstar / a_over_Rstar
#        ll, median, ul = np.percentile(Rp_over_a, [16,50,84])
#        if not quiet: print('Rp_over_a =', round_tex(median, median-ll, ul-median))

    