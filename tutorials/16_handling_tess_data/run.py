#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:01:44 2021

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

#::: my modules
import allesfitter
from allesfitter.io import read_csv
from allesfitter.plotting import tessplot
from allesfitter.lightcurves import tessclean
from allesfitter.detection.transit_search import tls_search

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})


    
    
if __name__ == "__main__":
    
    #::: Let's load a TESS example light curve
    time, flux, flux_err = read_csv('tess.csv')
    
    #::: Let's plot it and have a look!
    # tessplot(time, flux);
    
    #::: Uff, quite some noise! Let's clean it and look at the plots!
    # tessclean(time, flux, plot=True);
    
    #::: Better, but let's fine-tune things a bit more!
    # flux_clean, fig1, fig2, fig3 = \
    #     tessclean(time, flux, plot=True, 
    #               method='biweight', window_length=0.5, 
    #               bad_regions=[(2458420,2458424), (2458427.9,2458428.2)])
    
    #::: Nice, much cleaner! Let's use the clean light curve and look for a transit.
    # results_all, fig_all = \
    #     tls_search(time, flux_clean, flux_err, plot=True,
    #                 period_min=0.5, period_max=20.)