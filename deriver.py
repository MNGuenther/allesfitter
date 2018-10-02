#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:19:30 2018

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
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import allesfitter, latex_printer
import emcee
from corner import corner
from shutil import copyfile

#::: constants
M_earth = 5.9742e+24 	#kg 	Earth mass
M_jup   = 1.8987e+27 	#kg 	Jupiter mass
M_sun   = 1.9891e+30 	#kg 	Solar mass
R_earth = 6378136      #m 	Earth equatorial radius
R_jup   = 71492000 	#m 	Jupiter equatorial radius
R_sun   = 695508000 	#m 	Solar radius



def derive(datadir, QL=False, output_units='jup'):
    '''
    Derives parameter of the system using Winn 2010
    
    Input:
    ------
    datadir : str
        ...
        ! This needs to contain a file called 'params_star.csv' which contains all the infos of the host star !
    ...
    
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


    settings, theta_0, init_err, bounds, params, fitkeys, allkeys, labels, units, outdir = allesfitter.init(datadir, False)
    
    if QL:
        outdir = os.path.join( datadir,'QL' )
        if not os.path.exists( outdir ): os.makedirs( outdir )
    
    
    ###############################################################################
    #::: safety check: ask user for permission
    ###############################################################################
    f = os.path.join(outdir,'fit.jpg')
    if os.path.exists( f ):
        overwrite = raw_input('Output already exists in '+outdir+'. Overwrite output files? Y = yes, N = no\n')
        if not (overwrite.lower() == 'y'):
            raise ValueError('User aborted operation.')
            
    if QL:
        copyfile(os.path.join(datadir,'results','save.h5'), 
                 os.path.join(outdir,'save.h5'))
            
    reader = emcee.backends.HDFBackend( os.path.join(outdir,'save.h5'), read_only=True )

    if QL:
        settings['total_steps'] = reader.get_chain().shape[0]
        settings['burn_steps'] = int(0.75*settings['thin_by']*reader.get_chain().shape[0])



    ###############################################################################
    #::: autocorr
    ###############################################################################
    allesfitter.print_autocorr(reader, settings, fitkeys)
 
 
 
    ###############################################################################
    #::: draw samples
    ###############################################################################
    samples = reader.get_chain(flat=True, discard=settings['burn_steps']/settings['thin_by'])
#    samples = samples[np.random.randint(len(samples), size=5000)]
    N_samples = samples.shape[0]
    
    

    ###############################################################################
    #::: stellar 'posteriors'
    ###############################################################################
    buf = np.genfromtxt( os.path.join(datadir,'params_star.csv'), delimiter=',', names=True, comments='#' )
    star = {}
    star['R_star'] = buf['R_star'] + buf['R_star_err']*np.random.randn(N_samples)
    star['M_star'] = buf['M_star'] + buf['M_star_err']*np.random.randn(N_samples)
    
    
    
    ###############################################################################
    #::: derive all the params
    ###############################################################################
    planets = np.unique( settings['planets_phot']+settings['planets_rv'] )
    
    def get_params(key):
        ind = np.where(fitkeys==key)[0]
        if len(ind)==1: return samples[:,ind].flatten() #if it was fitted for
        else: return params[key] #else take the input value
        
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
        a_1 = 0.019771142 * get_params(planet+'_K') * get_params(planet+'_period') * np.sqrt(1. - derived_samples[planet+'_e']**2)/sin_d(derived_samples[planet+'_i'])
#        derived_samples[planet+'_a_rv'] = (1.+1./ellc_params[planet+'_q'])*a_1
        derived_samples[planet+'_q'] = 1./(( derived_samples[planet+'_a'] / a_1 ) - 1.)
        if suffix=='earth':
            derived_samples[planet+'_M_planet'] = derived_samples[planet+'_q'] * star['M_star'] * M_sun / M_earth #in M_earth
        elif suffix=='jup':
            derived_samples[planet+'_M_planet'] = derived_samples[planet+'_q'] * star['M_star'] * M_sun / M_jup #in M_jup
            
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
        for name,label,unit in zip([planet+'_R_star/a'                 , planet+'_R_planet/a'                  , planet+'_R_planet'             , planet+'_a'                , planet+'_i'                , planet+'_e'                , planet+'_w'                     , planet+'_M_planet'             , planet+'_b_tra'                  , planet+'_b_occ'                  , planet+'_T_tra_tot'              , planet+'_T_tra_full'              ],\
                                   ['$\mathrm{R_\star/a ('+planet+')}$', '$\mathrm{R_{planet}/a ('+planet+')}$', '$\mathrm{R_{p} ('+planet+')}$', '$\mathrm{a ('+planet+')}$', '$\mathrm{i ('+planet+')}$', '$\mathrm{e ('+planet+')}$', '$\mathrm{\omega ('+planet+')}$', '$\mathrm{M_{p} ('+planet+')}$', '$\mathrm{b_{tra} ('+planet+')}$', '$\mathrm{b_{occ} ('+planet+')}$', '$\mathrm{T_{tot} ('+planet+')}$', '$\mathrm{T_{full} ('+planet+')}$'],\
                                   ['-'                                , '-'                                   , 'R_{'+suffix+'}'               , 'R_{\odot}'                , 'deg'                      , '-'                        , 'deg'                           , 'M_{'+suffix  +'}'             , '-'                              , '-'                              , 'h'                              , 'h'                               ]):
            names.append(name) 
            labels.append(label)
            units.append(unit)
            
#        for inst in settings['inst_phot']:
#            names.append( planet+'_depth_diluted_'+inst )
#            units.append( '%' )
            
            
            
    ###############################################################################
    #::: save all in pickle
    ###############################################################################
    pickle.dump(derived_samples, open(os.path.join(outdir,'derived_samples.pickle'),'wb'))
    
    
    
    ###############################################################################
    #::: save txt & latex table & latex commands
    ###############################################################################
    with open(os.path.join(outdir,'derived_results.csv'),'wb') as outfile,\
         open(os.path.join(outdir,'derived_latex_table.txt'),'wb') as f,\
         open(os.path.join(outdir,'derived_latex_cmd.txt'),'wb') as f_cmd:
             
        outfile.write('name,unit,value,lower_error,upper_error\n')
        
        f.write('parameter & value & unit & - \\\\ \n')
        f.write('\\hline \n')
        f.write('\\multicolumn{4}{c}{\\textit{Fitted parameters}} \\\\ \n')
        f.write('\\hline \n')
        
        for name,label,unit in zip(names, labels, units):
            ll, median, ul = np.percentile(derived_samples[name], [15.865, 50., 84.135])
            outfile.write( str(label)+','+str(unit)+','+str(median)+','+str(median-ll)+','+str(ul-median)+'\n' )
            
            value = latex_printer.round_tex(median, ll, ul)
            f.write( label + ' & $' + value + '$ & $\mathrm{' + unit +'}$ \\\\ \n' )
            
            f_cmd.write('\\newcommand{\\'+name.replace("_", "")+'}{'+name+'$='+value+'$} \n')
            
    print '\nSaved derived_results.csv, derived_latex_table.txt, and derived_latex_cmd.txt'
    
        
        
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
    fig.savefig( os.path.join(outdir,'derived_corner.jpg'), dpi=100, bbox_inches='tight' )
    plt.close(fig)
    
    print '\nSaved derived_corner.jpg.'
    