#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:10:28 2018

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

from .simulate_PDF import simulate_PDF as spdf
from .latex_printer import round_tex




###############################################################################
#::: calculations and output
###############################################################################
def get_priors_from_literature(i, a_over_Rstar, Rp_over_Rstar, Nsamples=10000):
    
    #::: calculate cosi
    if i is not None:
        i, fig = spdf(i[0], i[1], i[2], size=Nsamples, plot=True)
        plt.title('i')
        cosi = np.cos(np.deg2rad(i))
        ll, median, ul = np.percentile(cosi, [16,50,84])
        print 'cosi =', round_tex(median, median-ll, ul-median)
    
    
    #::: calculate (R_star+R_p)/a, R_star/a, and R_p/a
    if (a_over_Rstar is not None) & (Rp_over_Rstar is not None):
        a_over_Rstar, fig = spdf(a_over_Rstar[0], a_over_Rstar[1], a_over_Rstar[2], size=Nsamples, plot=True)
        plt.title('a / R_star')
        
        Rp_over_Rstar, fig = spdf(Rp_over_Rstar[0], Rp_over_Rstar[1], Rp_over_Rstar[2], size=Nsamples, plot=True)
        plt.title('R_p / R_star')
        
        Rstar_over_a = 1./a_over_Rstar 
        ll, median, ul = np.percentile(Rstar_over_a, [16,50,84])
        print 'Rstar_over_a =', round_tex(median, median-ll, ul-median)
        
        Rsuma = Rstar_over_a * (1. + Rp_over_Rstar)
        ll, median, ul = np.percentile(Rsuma, [16,50,84])
        print 'Rsuma =', round_tex(median, median-ll, ul-median)
        
        Rp_over_a = Rp_over_Rstar / a_over_Rstar
        ll, median, ul = np.percentile(Rp_over_a, [16,50,84])
        print 'Rp_over_a =', round_tex(median, median-ll, ul-median)




###############################################################################
#::: user input
###############################################################################
if __name__ == '__main__':
    pass
    #::: example
#    i = (82.80, 0.17, 0.17)
#    a_over_Rstar = (5.851, 0.038, 0.037)
#    Rp_over_Rstar = (0.14075, 0.00035, 0.00035)
#    get_priors_from_literature(i, a_over_Rstar, Rp_over_Rstar)
    