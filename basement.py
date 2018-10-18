#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:17:06 2018

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

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import os
from datetime import datetime
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 

#::: my modules
from lichtkurven import index_transits, index_eclipses



###############################################################################
#::: 'Basement' class, which contains all the data, settings, etc.
###############################################################################
class Basement():
    '''
    The 'Basement' class contains all the data, settings, etc.
    '''
    
    ###############################################################################
    #::: init
    ###############################################################################
    def __init__(self, datadir):
        '''
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
                
        Returns:
        --------
        All the variables needed for allesfitter.MCMC_fit
        '''
        
        self.now = datetime.now().isoformat()
        
        self.datadir = datadir
        
        self.load_params()
        self.load_settings()
        self.load_data()
        self.change_epoch()
        
        #::: if baseline model == sample_GP, set up a GP object for photometric data
#        self.setup_GPs()
        
        #::: translate limb darkening codes from params.csv (int) into str for ellc
        self.ldcode_to_ldstr = ["none",#   :  0,
                                "lin",#    :  1,
                                "quad",#   :  2,
                                "sing",#   :  3,
                                "claret",# :  4,
                                "log",#  :  5,
                                "sqrt",#  :  6,
                                "exp",#    :  7,
                                "power-2",#:  8,
                                "mugrid"]# : -1
        
        
        #::: set up the outdir
        self.outdir = os.path.join(datadir,'results') 
        if not os.path.exists( self.outdir ): os.makedirs( self.outdir )

        #::: check if the input is consistent
        for inst in self.settings['inst_phot']:
            key='flux'
            if (self.settings['baseline_'+key+'_'+inst] == 'sample_GP') &\
               (self.settings['error_'+key+'_'+inst] != 'sample'):
                   raise ValueError('If you want to use sample_GP, you will want to sample the jitters, too!')
            
            

    ###############################################################################
    #::: load settings
    ###############################################################################
    def load_settings(self):
        
        def set_bool(text):
            if text.lower() in ['true', '1']:
                return True
            else:
                return False
            
        def unique(array):
            uniq, index = np.unique(array, return_index=True)
            return uniq[index.argsort()]
            
        rows = np.genfromtxt( os.path.join(self.datadir,'settings.csv'), dtype=None, delimiter=',' )

        self.settings = {r[0]:r[1] for r in rows}
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: General settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['planets_phot', 'planets_rv', 'inst_phot', 'inst_rv']:
            if len(self.settings[key]): 
                self.settings[key] = str(self.settings[key]).split(' ')
            else:                       
                self.settings[key] = []
        
        self.settings['planets_all']  = list(np.unique(self.settings['planets_phot']+self.settings['planets_rv'])) #sorted by b, c, d...
        self.settings['inst_all'] = list(unique( self.settings['inst_phot']+self.settings['inst_rv'] )) #sorted like user input
    
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Epoch settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for planet in self.settings['planets_all']:
            if self.settings['inst_for_'+planet+'_epoch'] in ['all','none']:
                self.settings['inst_for_'+planet+'_epoch'] = self.settings['inst_all']
            else:
                if len(self.settings['inst_for_'+planet+'_epoch']): 
                    self.settings['inst_for_'+planet+'_epoch'] = str(self.settings['inst_for_'+planet+'_epoch']).split(' ')
                else:                       
                    self.settings['inst_for_'+planet+'_epoch'] = []
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Multiprocess settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['multiprocess'] = set_bool(self.settings['multiprocess'])
        
        if 'multiprocess_cores' not in self.settings.keys():
            self.settings['multiprocess_cores'] = cpu_count()-1
        elif self.settings['multiprocess_cores'] == 'all':
            self.settings['multiprocess_cores'] = cpu_count()-1
        else:
            self.settings['multiprocess_cores'] = int(self.settings['multiprocess_cores'])
            if self.settings['multiprocess_cores'] == cpu_count():
                string = 'You are pushing your luck: you want to run on '+str(self.settings['multiprocess_cores'])+' cores, but your computer has only '+str(cpu_count())+'. I will let you go through with it this time...'
                warnings.warn(string)
            if self.settings['multiprocess_cores'] > cpu_count():
                string = 'Oops, you want to run on '+str(self.settings['multiprocess_cores'])+' cores, but your computer has only '+str(cpu_count())+'. Maybe try running on '+str(cpu_count()-1)+'?'
                raise ValueError(string)


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Fast fit
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['fast_fit'] = set_bool(self.settings['fast_fit'])
        
        if 'fast_fit_width' not in self.settings.keys():
            self.settings['fast_fit_width'] = 8./24.
        else:
            self.settings['fast_fit_width'] = np.float(self.settings['fast_fit_width'])
                
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Secondary eclipse
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'secondary_eclipse' not in self.settings.keys():
            self.settings['secondary_eclipse'] = False
        else:
            self.settings['secondary_eclipse'] = set_bool(self.settings['secondary_eclipse'])
                        
            
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: MCMC settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['mcmc_nwalkers','mcmc_total_steps','mcmc_burn_steps','mcmc_thin_by']:
            self.settings[key] = int(self.settings[key])
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Nested Sampling settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['ns_nlive'] = int(self.settings['ns_nlive'])
        self.settings['ns_tol'] = float(self.settings['ns_tol'])
#        if self.settings['ns_sample'] == 'auto':
#            if self.ndim < 10:
#                self.settings['ns_sample'] = 'unif'
#                print('Using ns_sample=="unif".')
#            elif 10 <= self.ndim <= 20:
#                self.settings['ns_sample'] = 'rwalk'
#                print('Using ns_sample=="rwalk".')
#            else:
#                self.settings['ns_sample'] = 'slice'
#                print('Using ns_sample=="slice".')
                
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Color plot
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'color_plot' not in self.settings.keys():
            self.settings['color_plot'] = False
            
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Exposure time interpolation
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            #::: if t_exp is given
            if 't_exp_'+inst in self.settings.keys() and len(self.settings['t_exp_'+inst]):
                t_exp = self.settings['t_exp_'+inst].split(' ')
                #if float
                if len(t_exp)==1:
                    self.settings['t_exp_'+inst] = np.float(t_exp[0])
                #if array
                else:
                    self.settings['t_exp_'+inst] = np.array([ np.float(t) for t in t_exp ])
            #::: if not given / given as an empty field
            else:
                self.settings['t_exp_'+inst] = None
                
            #::: if t_exp_n_int is given
            if 't_exp_'+inst in self.settings.keys() and len(self.settings['t_exp_n_int_'+inst]):
                self.settings['t_exp_n_int_'+inst] = int(self.settings['t_exp_n_int_'+inst])
                if self.settings['t_exp_n_int_'+inst] < 1:
                    raise ValueError('"t_exp_n_int_'+inst+'" must be >= 1, but is given as '+str(self.settings['t_exp_n_int_'+inst])+' in params.csv')
            else:
                self.settings['t_exp_n_int_'+inst] = None
                
                
                
    ###############################################################################
    #::: load params
    ###############################################################################
    def load_params(self):
    
        buf = np.genfromtxt(os.path.join(self.datadir,'params.csv'), delimiter=',',comments='#',dtype=None,names=True)
        
        self.allkeys = buf['name'] #len(all rows in params.csv)
        self.labels = buf['label'] #len(all rows in params.csv)
        self.units = buf['unit']   #len(all rows in params.csv)
        
        self.params = {}           #len(all rows in params.csv)
        for i,key in enumerate(self.allkeys):
            self.params[key] = np.float(buf['value'][i])
            
        self.ind_fit = (buf['fit']==1)                  #len(all rows in params.csv)
        
        self.fitkeys = buf['name'][ self.ind_fit ]      #len(ndim)
        self.fitlabels = self.labels[ self.ind_fit ]    #len(ndim)
        self.fitunits = self.units[ self.ind_fit ]      #len(ndim)
        
        self.theta_0 = buf['value'][ self.ind_fit ]     #len(ndim)
        
        if 'init_err' in buf.dtype.names:
            self.init_err = buf['init_err'][ self.ind_fit ] #len(ndim)
        else:
            self.init_err = 1e-8
            
        self.bounds = [ str(item).split(' ') for item in buf['bounds'][ self.ind_fit ] ] #len(ndim)
        for i, item in enumerate(self.bounds):
            if item[0] in ['uniform', 'normal']:
                self.bounds[i] = [ item[0], np.float(item[1]), np.float(item[2]) ]
            elif item[0] in ['trunc_normal']:
                self.bounds[i] = [ item[0], np.float(item[1]), np.float(item[2]), np.float(item[3]), np.float(item[4]) ]
            else:
                raise ValueError('Bounds have to be "uniform" or "normal". Input from "params.csv" was "'+self.bounds[i][0]+'".')
    
        self.ndim = len(self.theta_0)                   #len(ndim)

    

    ###############################################################################
    #::: load data
    ###############################################################################
    def load_data(self):
        '''
        Example: 
        -------
            A lightcurve is stored as
                data['TESS']['time'], data['TESS']['flux']
            A RV curve is stored as
                data['HARPS']['time'], data['HARPS']['flux']
        '''
        self.data = {}
        for inst in self.settings['inst_phot']:
            time, flux, flux_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)         
            if not all(np.diff(time)>=0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            if self.settings['fast_fit']: time, flux, flux_err = self.reduce_phot_data(time, flux, flux_err)
            self.data[inst] = {
                          'time':time,
                          'flux':flux,
                          'err_scales_flux':flux_err/np.nanmean(flux_err)
                         }
            
        for inst in self.settings['inst_rv']:
            time, rv, rv_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)         
            if not all(np.diff(time)>=0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            self.data[inst] = {
                          'time':time,
                          'rv':rv,
                          'white_noise_rv':rv_err
                         }
            
            
            
    ###############################################################################
    #::: change epoch
    ###############################################################################
    def change_epoch(self):
        
        #::: change epoch entry from params.csv to set epoch into the middle of the range
        for planet in self.settings['planets_all']:
            #::: get data time range
            all_data = []
            for inst in self.settings['inst_for_'+planet+'_epoch']:
                all_data += list(self.data[inst]['time'])
            start = np.nanmin( all_data )
            end = np.nanmax( all_data )
            
#            import matplotlib.pyplot as plt
#            plt.figure()
#            plt.plot(all_data, np.ones_like(all_data), 'bo')
            
            first_epoch = 1.*self.params[planet+'_epoch']
            period      = 1.*self.params[planet+'_period']
            
#            plt.axvline(first_epoch, color='r', lw=2)
            
            #::: place the first_epoch at the start of the data to avoid luser mistakes
            if start<=first_epoch:
                first_epoch -= int(np.round((first_epoch-start)/period)) * period
            else:
                first_epoch += int(np.round((start-first_epoch)/period)) * period
                
#            plt.axvline(first_epoch, color='b', lw=2)
                
            #::: place epoch_for_fit into the middle of all data
            epoch_for_fit = first_epoch + int(np.round((end-start)/2./period)) * period 
            
            #::: update params
            self.params[planet+'_epoch'] = 1.*epoch_for_fit
            
            #::: update theta and bounds
            ind_epoch_fitkeys = np.where(self.fitkeys==planet+'_epoch')[0]
            if len(ind_epoch_fitkeys):
                ind_epoch_fitkeys = ind_epoch_fitkeys[0]
                buf = 1.*self.theta_0[ind_epoch_fitkeys]
                self.theta_0[ind_epoch_fitkeys]    = 1.*epoch_for_fit                #initial guess
                
                #:::change bounds if uniform bounds
                if self.bounds[ind_epoch_fitkeys][0] == 'uniform':
                    lower = buf - self.bounds[ind_epoch_fitkeys][1]
                    upper = self.bounds[ind_epoch_fitkeys][2] - buf
                    self.bounds[ind_epoch_fitkeys][1]  = epoch_for_fit - lower           #lower bound
                    self.bounds[ind_epoch_fitkeys][2]  = epoch_for_fit + upper           #upper bound
                
                #:::change bounds if normal bounds
                elif self.bounds[ind_epoch_fitkeys][0] == 'normal':
                    mean = 1.*self.theta_0[ind_epoch_fitkeys]
                    std = 1.*self.bounds[ind_epoch_fitkeys][2]
                    self.bounds[ind_epoch_fitkeys][1]  = mean         
                    self.bounds[ind_epoch_fitkeys][2]  = std        
                    
                else:
                    raise ValueError('Parameters "bounds" have to be "uniform" or "normal".')
           
            #::: print output (for testing only)
#            print('\nSetting epoch for planet '+planet)
#            print('\tfirst epoch, from params.csv file:', first_epoch)
#            
#            print('\nOrbital cycles since then:', int( (end-start) / period))
#            
#            print('\tepoch for fit, placed in the middle of the data range:', epoch_for_fit)
#            print('Theta for fit:', self.theta_0[ind_epoch_fitkeys])
#            print('Bounds[1] for fit:', self.bounds[ind_epoch_fitkeys][1])
#            print('Bounds[2] for fit:', self.bounds[ind_epoch_fitkeys][2])
##            
#            plt.axvline(epoch_for_fit, color='g', lw=2)
#            plt.axvspan(self.bounds[ind_epoch_fitkeys][1], self.bounds[ind_epoch_fitkeys][2], alpha=0.8, color='g')
#            plt.xlim([start-10,end+10])
#            plt.show()


    ###############################################################################
    #::: reduce_phot_data
    ###############################################################################
    def reduce_phot_data(self, time, flux, flux_err):
        ind_in = []
              
        for planet in self.settings['planets_phot']:
            epoch  = self.params[planet+'_epoch']
            period = self.params[planet+'_period']
            width  = self.settings['fast_fit_width']
            if self.settings['secondary_eclipse']:
                ind_ecl1, ind_ecl2, _ = index_eclipses(time,epoch,period,width,width) #TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1)
                ind_in += list(ind_ecl2)
            else:
                ind_in += list(index_transits(time,epoch,period,width)[0])
        time = time[ind_in]
        flux = flux[ind_in]
        flux_err = flux_err[ind_in]
        return time, flux, flux_err
            