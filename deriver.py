#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:19:30 2018

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
import os
#import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from corner import corner

#::: allesfitter modules
from . import config
from .utils import latex_printer
from .general_output import logprint
from .priors.simulate_PDF import simulate_PDF

#::: constants
M_earth = 5.9742e+24 	#kg 	Earth mass
M_jup   = 1.8987e+27 	#kg 	Jupiter mass
M_sun   = 1.9891e+30 	#kg 	Solar mass
R_earth = 6378136      #m 	Earth equatorial radius
R_jup   = 71492000 	#m 	Jupiter equatorial radius
R_sun   = 695508000 	#m 	Solar radius



def derive(samples, mode, output_units='jup'):
    '''
    Derives parameter of the system using Winn 2010
    
    Input:
    ------
    samples : array
        samples from the mcmc or nested sampling
    mode : str
        'mcmc' or 'ns'
        
    Returns:
    --------
    derived_samples : dict 
        with keys 'i', 'R1a', 'R2a', 'k', 'depth_undiluted', 'b_tra', 'b_occ', 'Ttot', 'Tfull'
        each key contains all the samples derived from the MCMC samples 
        (not mean values, but pure samples!)
        i = inclination 
        R1a = R1/a, radius planet over semiamplitude
        R2a = R2/a, radius star over semiamplitude
        Ttot = T_{1-4}, total transit width 
        Tfull = T_{2-3}, full transit width
        
    Output:
    -------
    latex table of results
    corner plot of derived values posteriors
    '''
    
    N_samples = samples.shape[0]
    

    ###############################################################################
    #::: stellar 'posteriors'
    ###############################################################################
    buf = np.genfromtxt( os.path.join(config.BASEMENT.datadir,'params_star.csv'), delimiter=',', names=True, comments='#' )
    star = {}
    star['R_star'] = simulate_PDF(buf['R_star'], buf['R_star_lerr'], buf['R_star_uerr'], size=N_samples, plot=False)
    star['M_star'] = simulate_PDF(buf['M_star'], buf['M_star_lerr'], buf['M_star_uerr'], size=N_samples, plot=False)
    
    
    
    ###############################################################################
    #::: derive all the params
    ###############################################################################
    planets = config.BASEMENT.settings['planets_all']
    
    def get_params(key):
        ind = np.where(config.BASEMENT.fitkeys==key)[0]
        if len(ind)==1: return samples[:,ind].flatten() #if it was fitted for
        else: return config.BASEMENT.params[key] #else take the input value
        
    def sin_d(alpha): return np.sin(np.deg2rad(alpha))
    def cos_d(alpha): return np.cos(np.deg2rad(alpha))
    def arcsin_d(x): return np.rad2deg(np.arcsin(x))
    def arccos_d(x): return np.rad2deg(np.arccos(x))

    derived_samples = {}
    for planet in planets:
        #::: radius related
        derived_samples[planet+'_R_star/a'] = get_params(planet+'_rsuma') / (1. + get_params(planet+'_rr'))
        derived_samples[planet+'_R_planet/a'] = get_params(planet+'_rsuma') * get_params(planet+'_rr') / (1. + get_params(planet+'_rr'))
        derived_samples[planet+'_R_planet'] = star['R_star'] * get_params(planet+'_rr') * R_sun / R_earth #in R_earth
        suffix='earth'
        if np.mean(derived_samples[planet+'_R_planet']) > 8.: #if R_planet > 8 R_earth, convert to R_jup
            derived_samples[planet+'_R_planet'] = derived_samples[planet+'_R_planet'] * R_earth / R_jup #in R_jup
            suffix='jup'
#        derived_samples[planet+'_depth_undiluted'] = 100.*get_params(planet+'_rr')**2 #in %

        #::: orbit related
        derived_samples[planet+'_a'] = star['R_star'] / derived_samples[planet+'_R_star/a']                
        derived_samples[planet+'_i'] = arccos_d(get_params(planet+'_cosi')) #in deg
        derived_samples[planet+'_e'] = get_params(planet+'_f_s')**2 + get_params(planet+'_f_c')**2
        derived_samples[planet+'_w'] = arcsin_d( get_params(planet+'_f_s') / np.sqrt(derived_samples[planet+'_e']) ) #in deg
        
        #::: mass related
        if planet+'_K' in config.BASEMENT.params:
            a_1 = 0.019771142 * get_params(planet+'_K') * get_params(planet+'_period') * np.sqrt(1. - derived_samples[planet+'_e']**2)/sin_d(derived_samples[planet+'_i'])
    #        derived_samples[planet+'_a_rv'] = (1.+1./ellc_params[planet+'_q'])*a_1
            derived_samples[planet+'_q'] = 1./(( derived_samples[planet+'_a'] / a_1 ) - 1.)
            if suffix=='earth':
                derived_samples[planet+'_M_planet'] = derived_samples[planet+'_q'] * star['M_star'] * M_sun / M_earth #in M_earth
            elif suffix=='jup':
                derived_samples[planet+'_M_planet'] = derived_samples[planet+'_q'] * star['M_star'] * M_sun / M_jup #in M_jup
        else:
            derived_samples[planet+'_M_planet'] = None
            
        #transit timing related
        derived_samples[planet+'_dt_occ'] = get_params(planet+'_period')/2. * (1. + 4./np.pi * derived_samples[planet+'_e'] * cos_d(derived_samples[planet+'_w'])  ) #approximation
        derived_samples[planet+'_b_tra'] = (1./derived_samples[planet+'_R_star/a']) * get_params(planet+'_cosi') * ( (1.-derived_samples[planet+'_e']**2) / ( 1.+derived_samples[planet+'_e']*sin_d(derived_samples[planet+'_w']) ) )
        derived_samples[planet+'_b_occ'] = (1./derived_samples[planet+'_R_star/a']) * get_params(planet+'_cosi') * ( (1.-derived_samples[planet+'_e']**2) / ( 1.-derived_samples[planet+'_e']*sin_d(derived_samples[planet+'_w']) ) )
        derived_samples[planet+'_T_tra_tot'] = get_params(planet+'_period')/np.pi *24.  \
                                  * np.arcsin( derived_samples[planet+'_R_star/a'] \
                                             * np.sqrt( (1.+get_params(planet+'_rr'))**2 - derived_samples[planet+'_b_tra']**2 )\
                                             / sin_d(derived_samples[planet+'_i']) ) #in h
        derived_samples[planet+'_T_tra_full'] = get_params(planet+'_period')/np.pi *24.  \
                                  * np.arcsin( derived_samples[planet+'_R_star/a'] \
                                             * np.sqrt( (1.-get_params(planet+'_rr'))**2 - derived_samples[planet+'_b_tra']**2 )\
                                             / sin_d(derived_samples[planet+'_i']) ) #in h

#        derived_samples['loc_x_sky'] = buf('locx')*pixel_size
#        derived_samples['loc_y_sky'] = buf('locy')*pixel_size

#    for inst in settings['inst_phot']:
#        dil = get_params('light_3_'+inst)
#        if np.mean(dil)<0.5: dil = 1-dil
#        derived_samples[planet+'_depth_diluted_'+inst] = derived_samples[planet+'_depth_undiluted'] * (1. - dil) #in %


    
    
    ###############################################################################
    #::: write keys for output
    ###############################################################################
    names = []
    labels = []
    units = []
    for planet in planets:
        for name,label,unit in zip( [planet+'_R_star/a'                 , planet+'_R_planet/a'                            , planet+'_R_planet'             , planet+'_a'              , planet+'_i'                , planet+'_e'                , planet+'_w'                   , planet+'_M_planet'         , planet+'_b_tra'               , planet+'_b_occ'               , planet+'_T_tra_tot'          , planet+'_T_tra_full'              ],\
                                    ['$R_\star/a_\mathrm{'+planet+'}$' , '$R_\mathrm{'+planet+'}/a_\mathrm{'+planet+'}$', '$R_\mathrm{'+planet+'}$'       , '$a_\mathrm{'+planet+'}$' , '$i_\mathrm{'+planet+'}$'  , '$e_\mathrm{'+planet+'}$'   , '$\omega_\mathrm{'+planet+'}$', '$M_\mathrm{'+planet+'}$'   , '$b_\mathrm{tra;'+planet+'}$', '$b_\mathrm{occ;'+planet+'}$', '$T_\mathrm{tot;'+planet+'}$', '$T_\mathrm{full;'+planet+'}$'    ],\
                                    ['-'                                , '-'                                             , '$\mathrm{R_{'+suffix+'}}$'    , '$\mathrm{R_{\odot}}$'   , 'deg'                      , '-'                        , 'deg'                         , '$\mathrm{M_{'+suffix+'}}$'  , '-'                           , '-'                           , 'h'                          , 'h'                               ]):
            names.append(name) 
            labels.append(label)
            units.append(unit)
            
#        for inst in settings['inst_phot']:
#            names.append( planet+'_depth_diluted_'+inst )
#            units.append( '%' )
            
            
            
    ###############################################################################
    #::: delete pointless values
    ###############################################################################
    ind_good = []
    for i,name in enumerate(names):
        if isinstance(derived_samples[name], np.ndarray) and not any(np.isnan(derived_samples[name])) and not all(np.array(derived_samples[name])==0):
            ind_good.append(i)
            
    names = [ names[i] for i in ind_good ]
    labels = [ labels[i] for i in ind_good ]
    units = [ units[i] for i in ind_good ]
    
    
            
    ###############################################################################
    #::: save all in pickle
    ###############################################################################
    pickle.dump(derived_samples, open(os.path.join(config.BASEMENT.outdir,mode+'_derived_samples.pickle'),'wb'))
    
    
    
    ###############################################################################
    #::: save txt & latex table & latex commands
    ###############################################################################
    with open(os.path.join(config.BASEMENT.outdir,mode+'_derived_table.csv'),'wb') as outfile,\
         open(os.path.join(config.BASEMENT.outdir,mode+'_derived_latex_table.txt'),'wb') as f,\
         open(os.path.join(config.BASEMENT.outdir,mode+'_derived_latex_cmd.txt'),'wb') as f_cmd:
             
        outfile.write('name,unit,value,lower_error,upper_error\n')
        
        f.write('parameter & value & unit & - \\\\ \n')
        f.write('\\hline \n')
        f.write('\\multicolumn{4}{c}{\\textit{Derived parameters}} \\\\ \n')
        f.write('\\hline \n')
        
        for name,label,unit in zip(names, labels, units):
            ll, median, ul = np.percentile(derived_samples[name], [15.865, 50., 84.135])
            outfile.write( str(label)+','+str(unit)+','+str(median)+','+str(median-ll)+','+str(ul-median)+'\n' )
            
            value = latex_printer.round_tex(median, median-ll, ul-median)
            f.write( label + ' & $' + value + '$ & ' + unit +' \\\\ \n' )
            
            f_cmd.write('\\newcommand{\\'+name.replace("_", "")+'}{'+name+'$='+value+'$} \n')
            
    logprint('\nSaved '+mode+'_derived_results.csv, '+mode+'_derived_latex_table.txt, and '+mode+'_derived_latex_cmd.txt')
    
        
        
    ###############################################################################
    #::: plot corner
    ###############################################################################
#    for name,unit in zip(names, units):
#        fig = plt.figure()
#        plt.title(name+str(len(derived_samples[name])))
#        plt.hist(derived_samples[name])

    x = np.column_stack([ derived_samples[name] for name in names ])
    fig = corner(x,
                 range = [0.999]*len(names),
                 labels = names,
                 quantiles=[0.15865, 0.5, 0.84135],
                 show_titles=True, title_kwargs={"fontsize": 14})
    fig.savefig( os.path.join(config.BASEMENT.outdir,mode+'_derived_corner.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(fig)
    
    logprint('\nSaved '+mode+'_derived_corner.jpg')
    