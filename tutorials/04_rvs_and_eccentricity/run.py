#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:57:05 2018

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




import allesfitter

'''
uncomment whatever you want to run
'''



##############################################################################
#::: fit a cicular model
##############################################################################

# allesfitter.show_initial_guess('allesfit_circular_model')
# allesfitter.mcmc_fit('allesfit_circular_model')
# allesfitter.mcmc_output('allesfit_circular_model')

# allesfitter.show_initial_guess('allesfit_circular_model')
# allesfitter.ns_fit('allesfit_circular_model')
# allesfitter.ns_output('allesfit_circular_model')



##############################################################################
#::: fit an eccentric model
##############################################################################

# allesfitter.show_initial_guess('allesfit_eccentric_model')
# allesfitter.mcmc_fit('allesfit_eccentric_model')
# allesfitter.mcmc_output('allesfit_eccentric_model')

# allesfitter.show_initial_guess('allesfit_eccentric_model')
# allesfitter.ns_fit('allesfit_eccentric_model')
# allesfitter.ns_output('allesfit_eccentric_model')



##############################################################################
#::: compare Bayesian evidence
##############################################################################

# fig, ax = allesfitter.ns_plot_bayes_factors(['allesfit_circular_model','allesfit_eccentric_model'])
# fig.savefig('Bayesian_evidence.pdf', bbox_inches='tight')