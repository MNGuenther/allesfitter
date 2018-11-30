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
from tqdm import tqdm 

#::: allesfitter modules
from . import config
from .utils import latex_printer
from .general_output import logprint
from .priors.simulate_PDF import simulate_PDF
from .computer import update_params, calculate_model

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
        R1a = R1/a, radius companion over semiamplitude
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
    companions = config.BASEMENT.settings['companions_all']
    
    def get_params(key):
        ind = np.where(config.BASEMENT.fitkeys==key)[0]
        if len(ind)==1: return samples[:,ind].flatten() #if it was fitted for
        else: 
            try:
                return config.BASEMENT.params[key] #else take the input value
            except KeyError:
                return np.nan
        
    def sin_d(alpha): return np.sin(np.deg2rad(alpha))
    def cos_d(alpha): return np.cos(np.deg2rad(alpha))
    def arcsin_d(x): return np.rad2deg(np.arcsin(x))
    def arccos_d(x): return np.rad2deg(np.arccos(x))

    derived_samples = {}
    for companion in companions:
        
        #::: radii
        derived_samples[companion+'_R_star/a'] = get_params(companion+'_rsuma') / (1. + get_params(companion+'_rr'))
        derived_samples[companion+'_R_companion/a'] = get_params(companion+'_rsuma') * get_params(companion+'_rr') / (1. + get_params(companion+'_rr'))
        derived_samples[companion+'_R_companion'] = star['R_star'] * get_params(companion+'_rr') * R_sun / R_earth #in R_earth
        suffix='earth'
        if np.mean(derived_samples[companion+'_R_companion']) > 8.: #if R_companion > 8 R_earth, convert to R_jup
            derived_samples[companion+'_R_companion'] = derived_samples[companion+'_R_companion'] * R_earth / R_jup #in R_jup
            suffix='jup'
        derived_samples[companion+'_depth_undiluted'] = 1e3*get_params(companion+'_rr')**2 #in mmag


        #::: orbit
        derived_samples[companion+'_a'] = star['R_star'] / derived_samples[companion+'_R_star/a']                
        derived_samples[companion+'_i'] = arccos_d(get_params(companion+'_cosi')) #in deg
        derived_samples[companion+'_e'] = get_params(companion+'_f_s')**2 + get_params(companion+'_f_c')**2
        derived_samples[companion+'_e_sinw'] = get_params(companion+'_f_s') * np.sqrt(derived_samples[companion+'_e'])
        derived_samples[companion+'_e_cosw'] = get_params(companion+'_f_c') * np.sqrt(derived_samples[companion+'_e'])
        derived_samples[companion+'_w'] = arccos_d( get_params(companion+'_f_c') / np.sqrt(derived_samples[companion+'_e']) ) #in deg, from 0 to 180
        
        
        #::: mass
        if companion+'_K' in config.BASEMENT.params:
            a_1 = 0.019771142 * get_params(companion+'_K') * get_params(companion+'_period') * np.sqrt(1. - derived_samples[companion+'_e']**2)/sin_d(derived_samples[companion+'_i'])
    #        derived_samples[companion+'_a_rv'] = (1.+1./ellc_params[companion+'_q'])*a_1
            derived_samples[companion+'_q'] = 1./(( derived_samples[companion+'_a'] / a_1 ) - 1.)
            if suffix=='earth':
                derived_samples[companion+'_M_companion'] = derived_samples[companion+'_q'] * star['M_star'] * M_sun / M_earth #in M_earth
            elif suffix=='jup':
                derived_samples[companion+'_M_companion'] = derived_samples[companion+'_q'] * star['M_star'] * M_sun / M_jup #in M_jup
        else:
            derived_samples[companion+'_M_companion'] = None
            
            
        #::: time of occultation    
        if config.BASEMENT.settings['secondary_eclipse'] is True:
            derived_samples[companion+'_dt_occ'] = get_params(companion+'_period')/2. * (1. + 4./np.pi * derived_samples[companion+'_e'] * cos_d(derived_samples[companion+'_w'])  ) #approximation
        else:
            derived_samples[companion+'_dt_occ'] = None
        
        #::: impact params
        derived_samples[companion+'_b_tra'] = (1./derived_samples[companion+'_R_star/a']) * get_params(companion+'_cosi') * ( (1.-derived_samples[companion+'_e']**2) / ( 1.+derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
         
        if config.BASEMENT.settings['secondary_eclipse'] is True:
            derived_samples[companion+'_b_occ'] = (1./derived_samples[companion+'_R_star/a']) * get_params(companion+'_cosi') * ( (1.-derived_samples[companion+'_e']**2) / ( 1.-derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
        else:
            derived_samples[companion+'_b_occ'] = None
        
        #::: transit duration 
        derived_samples[companion+'_T_tra_tot'] = get_params(companion+'_period')/np.pi *24.  \
                                  * np.arcsin( derived_samples[companion+'_R_star/a'] \
                                             * np.sqrt( (1.+get_params(companion+'_rr'))**2 - derived_samples[companion+'_b_tra']**2 )\
                                             / sin_d(derived_samples[companion+'_i']) ) #in h
        derived_samples[companion+'_T_tra_full'] = get_params(companion+'_period')/np.pi *24.  \
                                  * np.arcsin( derived_samples[companion+'_R_star/a'] \
                                             * np.sqrt( (1.-get_params(companion+'_rr'))**2 - derived_samples[companion+'_b_tra']**2 )\
                                             / sin_d(derived_samples[companion+'_i']) ) #in h
                                  

        #::: secondary eclipse / occultation depth (per inst)
        for inst in config.BASEMENT.settings['inst_phot']:
            derived_samples[companion+'_depth_occ_max_diluted_'+inst]  = np.zeros(N_samples)*np.nan
            derived_samples[companion+'_depth_occ_norm_diluted_'+inst] = np.zeros(N_samples)*np.nan
            
            if config.BASEMENT.settings['secondary_eclipse'] is True:
                    logprint('Deriving occultation depths from model curves...')
                    xx = np.linspace( 0.25, 0.75, 500)
                    for i in tqdm( range(N_samples) ):
                        s = samples[i,:]
                        p = update_params(s, phased=True)
                        model = calculate_model(p, inst, 'flux', xx=xx) #evaluated on xx (!)
                        derived_samples[companion+'_depth_occ_max_diluted_'+inst][i] = ( np.max(model) - np.min(model) ) * 1e6 #in ppm
                        derived_samples[companion+'_depth_occ_norm_diluted_'+inst][i] = ( 1. - np.min(model) ) * 1e6 #in ppm
                    
    
        #::: undiluted (not per companions; per inst)
        for inst in config.BASEMENT.settings['inst_phot']:
            dil = get_params('light_3_'+inst)
        #        if np.mean(dil)<0.5: dil = 1-dil
            derived_samples[companion+'_depth_diluted_'+inst] = derived_samples[companion+'_depth_undiluted'] * (1. - dil) #in mmag
            derived_samples[companion+'_depth_occ_max_undiluted_'+inst] = derived_samples[companion+'_depth_occ_max_diluted_'+inst] / (1. - dil) #in ppm
            derived_samples[companion+'_depth_occ_norm_undiluted_'+inst] = derived_samples[companion+'_depth_occ_norm_diluted_'+inst] / (1. - dil) #in ppm
        
    
    ###############################################################################
    #::: write keys for output
    ###############################################################################
    names = []
    labels = []
    units = []
    for companion in companions:
        for name,label,unit in zip( [companion+'_R_star/a'                 , companion+'_R_companion/a'                            , companion+'_R_companion'             , companion+'_a'              , companion+'_i'                , companion+'_e'                , companion+'_w'                   , companion+'_M_companion'         , companion+'_b_tra'               , companion+'_b_occ'               , companion+'_T_tra_tot'          , companion+'_T_tra_full'              ],\
                                    ['$R_\star/a_\mathrm{'+companion+'}$' , '$R_\mathrm{'+companion+'}/a_\mathrm{'+companion+'}$', '$R_\mathrm{'+companion+'}$'       , '$a_\mathrm{'+companion+'}$' , '$i_\mathrm{'+companion+'}$'  , '$e_\mathrm{'+companion+'}$'   , '$\omega_\mathrm{'+companion+'}$', '$M_\mathrm{'+companion+'}$'   , '$b_\mathrm{tra;'+companion+'}$', '$b_\mathrm{occ;'+companion+'}$', '$T_\mathrm{tot;'+companion+'}$', '$T_\mathrm{full;'+companion+'}$'    ],\
                                    ['-'                                , '-'                                             , '$\mathrm{R_{'+suffix+'}}$'    , '$\mathrm{R_{\odot}}$'   , 'deg'                      , '-'                        , 'deg'                         , '$\mathrm{M_{'+suffix+'}}$'  , '-'                           , '-'                           , 'h'                          , 'h'                               ]):
            names.append(name) 
            labels.append(label)
            units.append(unit)
            
        names.append( companion+'_depth_undiluted' )
        labels.append( '$\delta_\mathrm{undil}$' )
        units.append( '$\mathrm{mmag}$' )
            
        for inst in config.BASEMENT.settings['inst_phot']:
            
            names.append( companion+'_depth_diluted_'+inst )
            labels.append( '$\delta_\mathrm{dil; '+inst+'}$' )
            units.append( '$\mathrm{mmag}$' )
            
            names.append( companion+'_depth_occ_max_undiluted_'+inst )
            labels.append( '$\delta_\mathrm{occ; max; undil; '+inst+'}$' )
            units.append( '$\mathrm{mmag}$' )
            
            names.append( companion+'_depth_occ_norm_undiluted_'+inst )
            labels.append( '$\delta_\mathrm{occ; norm; undil; '+inst+'}$' )
            units.append( '$\mathrm{mmag}$' )
            
            names.append( companion+'_depth_occ_max_diluted_'+inst )
            labels.append( '$\delta_\mathrm{occ; max; dil; '+inst+'}$' )
            units.append( '$\mathrm{mmag}$' )
            
            names.append( companion+'_depth_occ_norm_diluted_'+inst )
            labels.append( '$\delta_\mathrm{occ; norm; dil; '+inst+'}$' )
            units.append( '$\mathrm{mmag}$' )
            
            
            
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
    