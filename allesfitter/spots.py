#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:41:12 2018

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
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
# from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product
from astropy import units as u
from astropy.coordinates import SkyCoord

#::: my modules
import allesfitter
from allesfitter import config




###############################################################################
#::: new spot map plots until I find something that makes me happy
###############################################################################
def convert_one_point_to_an_area(lon, lat, size, brightness):
    '''
    convert float values to arrays with length 1000, sampling an area
    '''
    
    N = 100
    length = np.sqrt(size * np.random.uniform(0, 1, size=N))
    angle = np.pi * np.random.uniform(0, 2, size=N)
    lon += length * np.cos(angle)
    lat += length * np.sin(angle)
    brightness = np.ones(N)*brightness
    return lon, lat, brightness



def convert_many_points_to_an_area(lons, lats, sizes, brightnesses):
    '''
    convert float values to arrays with length 1000, sampling an area
    '''
    
    lon_list = []
    lat_list = []
    brightness_list = []
    for lon,lat,size,brightness in zip(lons,lats,sizes,brightnesses):
        a, b, c = convert_one_point_to_an_area(lon, lat, size, brightness)
        lon_list += list(a)
        lat_list += list(b)
        brightness_list += list(c)
        
    return lon_list, lat_list, brightness_list



def plot_spots_new(datadir):
    
    alles = allesfitter.allesclass(datadir)
    
    for inst in alles.BASEMENT.settings['inst_phot']:
        fig = plt.figure()
        plt.subplot(111, projection="aitoff")
        plt.grid(True)
        for i in [1,2]:
            lons = alles.posterior_params['host_spot_'+str(i)+'_long_'+inst][0:20]
            lats = alles.posterior_params['host_spot_'+str(i)+'_lat_'+inst][0:20]
            sizes = alles.posterior_params['host_spot_'+str(i)+'_size_'+inst][0:20]
            brightnesses = alles.posterior_params['host_spot_'+str(i)+'_brightness_'+inst][0:20]
            lon_list, lat_list, brightness_list = convert_many_points_to_an_area(lons, lats, sizes, brightnesses)
            c = SkyCoord(ra=lon_list* u.deg, dec=lat_list* u.deg)
            lon_list_aitoff = c.ra.wrap_at(180 * u.deg).radian
            lat_list_aitoff= c.dec.radian
            plt.scatter( lon_list_aitoff, lat_list_aitoff, c=brightness_list, vmin=0, vmax=1 ) 
        plt.colorbar(label='Relative spot brightness')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.tight_layout()
        plt.xticks(ticks=np.deg2rad([-150,-120,-90,-60,-30,0,30,60,90,120,150,180]), labels=['','',r'$270^\circ$','','',r'$0^\circ$','','',r'$90^\circ$','','',r'$180^\circ$'])
        plt.yticks(ticks=np.deg2rad([-90,-60,-30,0,30,60,90]), labels=['',r'$-60^\circ$',r'$-30^\circ$','0',r'$30^\circ$',r'$60^\circ$',''])
        fig.savefig( os.path.join(alles.BASEMENT.outdir,'spots_aitoff_'+inst+'.pdf'), bbox_inches='tight' )
        
        
        fig = plt.figure()
        plt.xlim([0,360])
        plt.ylim([-90,90])
        for i in [1,2]:
            lons = alles.posterior_params['host_spot_'+str(i)+'_long_'+inst][0:20]
            lats = alles.posterior_params['host_spot_'+str(i)+'_lat_'+inst][0:20]
            sizes = alles.posterior_params['host_spot_'+str(i)+'_size_'+inst][0:20]
            brightnesses = alles.posterior_params['host_spot_'+str(i)+'_brightness_'+inst][0:20]
            lon_list, lat_list, brightness_list = convert_many_points_to_an_area(lons, lats, sizes, brightnesses)
            plt.scatter( lon_list, lat_list, c=brightness_list, vmin=0, vmax=1 ) 
        plt.colorbar(label='Relative spot brightness')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.tight_layout()
        fig.savefig( os.path.join(alles.BASEMENT.outdir,'spots_cartesian_'+inst+'.pdf'), bbox_inches='tight' )





###############################################################################
#::: plot publication ready spot plot
#::: flux
#::: model for 20 samples
#::: residuals for 1 sample
#::: 2D-spot map for 1 sample
###############################################################################

def plot_publication_spots_from_posteriors(datadir, Nsamples=20, command='save', mode='default'):
    '''
    command : str
        'show', 'save', 'return', 'show and return', 'save and return'
    mode: str
        default : 5000 points, phase (-0.25,0.75), errorbars
        zhan2019 : 100 points, phase (0,2), no errorbars
    '''
    
    fig, ax1, ax2, ax3 = setup_grid()
        
    config.init(datadir)
    posterior_samples = allesfitter.get_ns_posterior_samples(datadir, Nsamples=Nsamples, as_type='2d_array')
    
    for inst in config.BASEMENT.settings['inst_all']:
        if config.BASEMENT.settings['host_N_spots_'+inst] > 0:
            
            if mode=='default':
                xx = np.linspace(config.BASEMENT.data[inst]['time'][0], config.BASEMENT.data[inst]['time'][-1], 5000)
            elif mode=='zhan2019':
                xx = np.linspace(0, 2, 10000)

            for i_sample, sample in tqdm(enumerate(posterior_samples)):
            
                params = allesfitter.computer.update_params(sample)
                    
                spots = [ [params['host_spot_'+str(i)+'_long_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                          [params['host_spot_'+str(i)+'_lat_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                          [params['host_spot_'+str(i)+'_size_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                          [params['host_spot_'+str(i)+'_brightness_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ] ]  
    
    
                model = allesfitter.computer.calculate_model(params, inst, 'flux')
                baseline = allesfitter.computer.calculate_baseline(params, inst, 'flux')

                model_xx = allesfitter.computer.calculate_model(params, inst, 'flux', xx=xx) #evaluated on xx (!)
                baseline_xx = allesfitter.computer.calculate_baseline(params, inst, 'flux', xx=xx) #evaluated on xx (!)
    
                if i_sample==0:
                    if mode=='default':
                        ax1 = axplot_data(ax1, config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['flux'], flux_err=np.exp(params['log_err_flux_'+inst]))
                        ax2 = axplot_residuals(ax2, config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['flux']-model-baseline, res_err=np.exp(params['log_err_flux_'+inst]))
                        ax3 = axplot_spots_2d(ax3, spots)  
                    elif mode=='zhan2019':
                        ax1 = axplot_data(ax1, 
                                          np.concatenate(( config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['time']+1, config.BASEMENT.data[inst]['time']+2 )), 
                                          np.concatenate(( config.BASEMENT.data[inst]['flux'], config.BASEMENT.data[inst]['flux'], config.BASEMENT.data[inst]['flux'] )), 
                                          flux_err=None)
                        ax2 = axplot_residuals(ax2, 
                                               np.concatenate(( config.BASEMENT.data[inst]['time'], config.BASEMENT.data[inst]['time']+1, config.BASEMENT.data[inst]['time']+2 )), 
                                               np.concatenate(( config.BASEMENT.data[inst]['flux']-model-baseline, config.BASEMENT.data[inst]['flux']-model-baseline, config.BASEMENT.data[inst]['flux']-model-baseline )), 
                                               res_err=None)
                        ax3 = axplot_spots_2d(ax3, spots)      
                    
                ax1 = axplot_model(ax1, xx, model_xx+baseline_xx)  
                
            ax1.locator_params(axis='y', nbins=5)
            
            if mode=='zhan2019':
                ax1.set(xlim=[0,2])
                ax2.set(xlim=[0,2])
        
            if 'save' in command:
                pubdir = os.path.join(config.BASEMENT.outdir, 'pub')
                if not os.path.exists(pubdir): os.makedirs(pubdir)
                if mode=='default':
                    fig.savefig( os.path.join(pubdir,'host_spots_'+inst+'.pdf'), bbox_inches='tight' )
                elif mode=='zhan2019':
                    fig.savefig( os.path.join(pubdir,'host_spots_'+inst+'_zz.pdf'), bbox_inches='tight' )
                plt.close(fig)
    
            if 'show' in command:
                plt.show()
                
            if 'return' in command:
                return fig, ax1, ax2, ax3
                    
                    

def setup_grid():
    fig = plt.figure(figsize=(8,3.8))
    
    gs0 = gridspec.GridSpec(1, 2)
    
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[0], hspace=0)
    ax1 = plt.Subplot(fig, gs00[:-1, :])
    ax1.set(xlabel='', xticks=[], ylabel='Flux')
    fig.add_subplot(ax1)
    ax2 = plt.Subplot(fig, gs00[-1, :])
    ax2.set(xlabel='Phase', ylabel='Res.')
    fig.add_subplot(ax2)
    
    
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])
    ax3 = plt.Subplot(fig, gs01[:, :])
    ax3.set(xlabel='Long. (deg)', ylabel='Lat. (deg.)')
    fig.add_subplot(ax3)
    
    plt.tight_layout()
    return fig, ax1, ax2, ax3
        
        

def axplot_data(ax, time, flux, flux_err=None):
    if flux_err is not None:
        ax.errorbar(time, flux,  yerr=flux_err, marker='.', linestyle='none', color='lightblue', zorder=9)
    ax.plot(time, flux, 'b.', zorder=10)
    return ax


def axplot_residuals(ax, time, res, res_err=None):
    if res_err is not None:
        ax.errorbar(time, res,  yerr=res_err, marker='.', linestyle='none', color='lightblue', zorder=9)
    ax.plot(time, res, 'b.', zorder=10)
    ax.axhline(0, color='r', linewidth=2, zorder=11)
    return ax
    

def axplot_model(ax, time, model):
    ax.plot(time, model, 'r-', zorder=11)
    return ax



        
def axplot_spots_2d(ax, spots):
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
    
    for i in range(N_spots):
        lon, lat, size, brightness = spots[:,i]
        
        r = size
        theta = np.linspace(0, 2*np.pi, 100)
        lonv = lon + r * np.cos(theta)
        latv = lat + r * np.sin(theta)

        cm = plt.get_cmap('coolwarm')
        color = cm(brightness/2.)
        sc = ax.scatter(lon, lat, c=brightness, cmap='coolwarm', vmin=0, vmax=2)
        
        a = [-360.,0.,360.]
        b = [-180.,0.,180.]
        for r in product(a, b): 
            ax.fill(lonv+r[0], latv+r[1], color=color)
            ax.plot(lonv+r[0], latv+r[1], 'k-')
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(sc, cax=cax, ticks=[0,0.5,1,1.5,2])
    cbar.set_label('Brightness')
    ax.set(xlim=[0,360], ylim=[-90,90], xticks=[0,90,180,270,360], yticks=[-90,-45,0,45,90])
    
    return ax

                        
  

###############################################################################
#::: plot the spotmaps plots
#::: 3D-spot-map and 2D-spot-map, individually for 10 samples
###############################################################################                      

def plot_spots_from_posteriors(datadir, Nsamples=10, command='save'):
    
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
        
                if command=='save':
                    fig, ax, ax2 = plot_spots(spots, command='return')
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
                    
                    if command=='save':
                        fig, ax, ax2 = plot_spots(spots, command='return')
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
        return fig, ax, ax2

    elif command=='show':
        plt.show()
    
