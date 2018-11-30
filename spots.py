#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:41:12 2018

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm

#::: my modules
import allesfitter
from allesfitter import config




def plot_spots_from_posteriors(datadir, Nsamples=10, command='return'):
    
    if command=='show':
        Nsamples = 1 #overwrite user input and only show 1 sample if command=='show'
        
        
    config.init(datadir)
    posterior_samples_dic = allesfitter.get_ns_posterior_samples(datadir, Nsamples=Nsamples)
    
    for sample in tqdm(range(Nsamples)):
    
        params = {}
        for key in posterior_samples_dic:
            params[key] = posterior_samples_dic[key][sample]
        
        
        for inst in config.BASEMENT.settings['inst_all']:
            
            if config.BASEMENT.settings['host_N_spots_'+inst] > 0:
                spots = [
                                     [params['host_spot_'+str(i)+'_long_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                                     [params['host_spot_'+str(i)+'_lat_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                                     [params['host_spot_'+str(i)+'_size_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                                     [params['host_spot_'+str(i)+'_brightness_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ]
                                    ]
        
                if command=='return':
                    fig = plot_spots(spots, command='return')
                    plt.suptitle('sample '+str(sample))
                    spotsdir = os.path.join(config.BASEMENT.outdir, 'spotmaps')
                    if not os.path.exists(spotsdir): os.makedirs(spotsdir)
                    fig.savefig( os.path.join(spotsdir,'host_spots_'+inst+'_posterior_sample_'+str(sample)) )
                    plt.close(fig)
                
                elif command=='show':
                    plot_spots(spots, command='show')
        
        for companion in config.BASEMENT.settings['companions_all']:
            for inst in config.BASEMENT.settings['inst_all']:
                if config.BASEMENT.settings[companion+'_N_spots_'+inst] > 0:
                    spots = [
                                         [params[companion+'_spot_'+str(i)+'_long_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ],
                                         [params[companion+'_spot_'+str(i)+'_lat_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ],
                                         [params[companion+'_spot_'+str(i)+'_size_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ],
                                         [params[companion+'_spot_'+str(i)+'_brightness_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ]
                                        ]
                    
                    if command=='return':
                        fig = plot_spots(spots, command='return')
                        plt.suptitle('sample '+str(sample))
                        spotsdir = os.path.join(config.BASEMENT.outdir, 'spotmaps')
                        if not os.path.exists(spotsdir): os.makedirs(spotsdir)
                        fig.savefig( os.path.join(spotsdir,companion+'_spots_'+inst+'_posterior_sample_'+str(sample)) )
                        plt.close(fig)
                    
                    elif command=='show':
                        plot_spots(spots, command='show')
                
                


def plot_spots(spots, command='show'):
    '''
    Inputs:
    -------
    spots : ...
    
    command : str
        'show' : show the figure (do not automatically save)
        'return' : return the figure object (do not display)
    
    e.g. for two spots:
    spots = [ [lon1, lon2],
              [lat1, lat2],
              [size1, size2],
              [brightness1, brightness2] ] 
    '''

    np.random.seed(42)

    spots = np.array(spots)
    N_spots = len(spots[0])
    radius = 1.
    N_rand = 3000
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='orange', linewidth=0, rstride=4, cstride=4, alpha=0.2, antialiased=False, shade=False)
    
    
    lon = np.linspace( 0./180.*np.pi, 360./180.*np.pi, 100 )
    lat = np.zeros_like(lon)
    x = radius * np.cos(lon) * np.cos(lat)
    y = radius * np.sin(lon) * np.cos(lat)
    z = radius * np.sin(lat)
    ax.plot(x,y,z,ls='-',c='grey',alpha=0.3)
    
    
    lat = np.linspace( 0./180.*np.pi, 360./180.*np.pi, 100 )
    lon = np.zeros_like(lat)
    x = radius * np.cos(lon) * np.cos(lat)
    y = radius * np.sin(lon) * np.cos(lat)
    z = radius * np.sin(lat)
    ax.plot(x,y,z,ls='-',c='grey',alpha=0.3)
    
    
    for i in range(N_spots):
        lon, lat, size, brightness = spots[:,i]
        
        r = size * np.sqrt(np.random.rand(N_rand))
        theta = np.random.rand(N_rand) * 2. * np.pi
        lonv = lon + r * np.cos(theta)
        latv = lat + r * np.sin(theta)
        
        lon = lonv/180.*np.pi
        lat = latv/180.*np.pi
        x = radius * np.cos(lon) * np.cos(lat)
        y = radius * np.sin(lon) * np.cos(lat)
        z = radius * np.sin(lat)
        c = brightness * np.ones_like(lon)
        sc = ax.scatter(x,y,z,c=c,marker='.', cmap='seismic', vmin=0, vmax=2, alpha=1, rasterized=True)
        ax2.scatter(lonv, latv, c=c, marker='.', cmap='seismic', vmin=0, vmax=2, alpha=1, rasterized=True)
        
        
        lon, lat, size, brightness = spots[:,i]
        
        r = size
        theta = np.linspace(0, 2*np.pi, 100)
        lonv = lon + r * np.cos(theta)
        latv = lat + r * np.sin(theta)
        
        lon = lonv/180.*np.pi
        lat = latv/180.*np.pi
        x = radius * np.cos(lon) * np.cos(lat)
        y = radius * np.sin(lon) * np.cos(lat)
        z = radius * np.sin(lat)
        ax.plot(x,y,z,'k-',zorder=20)
        ax2.plot(lonv, latv,'k-',zorder=20)
        
    plt.colorbar(sc)
        
    ax2.set(xlim=[0,360], ylim=[-90,90])
    
        
    
    if command=='return':
        return fig

    elif command=='show':
        plt.show()
    
    