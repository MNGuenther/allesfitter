#!/usr/bin/env python2
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
"""



import allesfitter
from allesfitter import ns_plot_bayes_factors, ns_plot_violins



'''
uncomment whatever you want to run
'''


###############################################################################
#::: run the fit
###############################################################################
#allesfitter.ns_fit('allesfit_TMNT_TTVs')
#allesfitter.ns_output('allesfit_TMNT_TTVs')



################################################################################
##::: plot bayes factors
################################################################################
#datadirs = [ 'allesfit_TMNT', 'allesfit_TMNT_TTVs']
#labels = [ 'No TTVs', 'TTVs' ]
#
#collection_of_run_names = (datadirs)
#collection_of_labels = (labels)
#
#fig, ax = ns_plot_bayes_factors(collection_of_run_names, collection_of_labels)
#fig.savefig('pub/bayes_factors.pdf', bbox_inches='tight')
#
#
#
################################################################################
##::: plot posterior violins
################################################################################
#keys = ['b_period', 'b_epoch', 'b_rr', 'b_rsuma', 'b_cosi', 'b_K']     
#  
#for key in keys:
#    fig, ax = ns_plot_violins(datadirs, labels, key)
#    fig.savefig('pub/violins_'+key+'.pdf', bbox_inches='tight')
