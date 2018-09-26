#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:40:13 2018

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
from allesfitter import allesfitter

np.random.seed(42)



datadir = 'allesfit_example_1/'


###############################################################################
# Plot the data and an initial guess.
###############################################################################
'''
Uncomment to use the functions.

Function:
---------
allesfitter.initial_guess(datadir, fast_fit=False)

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
#allesfitter.show_initial_guess(datadir)
#allesfitter.show_initial_guess(datadir, fast_fit=True)


###############################################################################
# Run the MCMC fit.
###############################################################################
'''
Uncomment to use the functions.


Function:
---------
allesfitter.run(datadir, fast_fit=False, continue_old_run=False)

(automatically runs allesfitter.show_initial_guess and allesfitter.analyse_output, too.)

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
#allesfitter.run(datadir)
#allesfitter.run(datadir, fast_fit=True)
#allesfitter.run(datadir, fast_fit=True, continue_old_run=True)


###############################################################################
# Create outputs
###############################################################################
'''
Uncomment to use the functions.

Function:
---------
allesfitter.analyse_output(datadir, fast_fit=False, QL=False)

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
#allesfitter.analyse_output(datadir, QL=True)
#allesfitter.analyse_output(datadir) 