#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:27:01 2020

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

#::: modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint
try: 
    from brokenaxes import brokenaxes
except: 
    pass #don't throw an error for now, otherwise people will panic
import pathlib
from astropy.time import Time

#::: local imports
from allesfitter.io import read_csv
from allesfitter.time_series import sigma_clip, slide_clip

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth':1})

#::: globals (nice)
TJD_offset = 2457000

    
    

def clean_up(time, y, yerr=None, time_format='BJD_TDB'):
    time = np.array(time)    
    y = np.array(y)    
    if yerr is not None: 
        yerr = np.array(yerr)    
    
    if time_format == 'BJD_TDB': 
        time = time - TJD_offset
    
    #TODO: the below messes things up in case a "trend" is passed, too
    #TODO: it is probably not even needed, so just remove it?
    # if yerr is None:
    #     ind = np.where( ~np.isnan(time*y) )
    #     time = time[ind]
    #     y = y[ind]    
    # else:
    #     ind = np.where( ~np.isnan(time*y*yerr) )
    #     time = time[ind]
    #     y = y[ind]    
    #     yerr = yerr[ind]
        
    return time, y, yerr




def guess_labels(ax, time, y):
    
    if (np.nanpercentile(y,5) > 0.7) & (np.nanpercentile(y,95) < 1.3):
        ax.set_ylabel('Flux')
    elif (np.nanpercentile(y,5) > -0.02) & (np.nanpercentile(y,95) < 0.02):
        ax.set_ylabel('Residuals')
    else:
        ax.set_ylabel('RV (km/s)')
       
    ax.set_xlabel(r'Time (BJD$_\mathrm{TDB}$ - '+str(TJD_offset)+')')




def fullplot(time, y, yerr=None, ax=None, time_format='BJD_TDB', clip=False, **kwargs):
    '''
    Parameters
    ----------
    time : array of float
        e.g. time array (usually in days)
    y : array of float
        e.g. flux or RV array (usually as normalized flux or RV in km/s)
    yerr : array of float
        e.g. flux or RV error array (usually as normalized flux or RV in km/s)
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    time_format : str
        The format of your time array. 
        Must be either 'BJD_TDB' or 'TJD' (TESS Julian Date). 
        The default is 'BJD_TDB'.
    clip : bool, optional
        Automatically clip the input data with sigma_clip(low=4, high=4)
        and slide_clip(window_length=1, low=4, high=4). The default is True.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.
    '''
    
    if ax is None: fig, ax = plt.subplots(figsize=(12,3), tight_layout=True)
    time, y, yerr = clean_up(time, y, yerr, time_format)
    if clip:
        flux, mask_lower, mask_upper = _clip_helper(time, y)
    
    ax.errorbar(time, y, yerr=yerr, fmt='b.', ms=2, rasterized=True, **kwargs)
    if clip:
        ax.plot(time*mask_upper, ax.get_ylim()[1]*mask_upper, 'r^', color='orange', ms=10, zorder=11)
        ax.plot(time*mask_lower, ax.get_ylim()[0]*mask_lower, 'rv', color='orange', ms=10, zorder=11)
    guess_labels(ax, time, y)

    return ax



def fullplot_csv(fname, ax=None, time_format='BJD_TDB'):
    '''
    Wrapper around fullplot to plot straight from a csv file.
    See fullplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return fullplot(time, y, yerr, ax=ax, time_format=time_format)



def brokenplot(time, y, yerr=None, trend=None, dt=10, ax=None, time_format='BJD_TDB', fmt='b.', clip=False):
    '''
    Parameters
    ----------
    time : array of float
        e.g. time array (usually in days)
    y : array of float
        e.g. flux or RV array (usually as normalized flux or RV in km/s)
    yerr : array of float
        e.g. flux or RV error array (usually as normalized flux or RV in km/s)
    dt : float, optional
        The gap size after which axes will be broken. The default is 10 (usually in days).
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    time_format : str
        The format of your time array. 
        Must be either 'BJD_TDB' or 'TJD' (TESS Julian Date). 
        The default is 'BJD_TDB'.
    clip : bool, optional
        Automatically clip the input data with sigma_clip(low=4, high=4)
        and slide_clip(window_length=1, low=4, high=4). The default is True.

    Returns
    -------
    bax : brokenaxes instance
        Just like an pyplot.Axes instance
    '''
    
    if 'brokenaxes' not in sys.modules:
        raise ImportError('You must install the brokenaxes package before using allesfitter.brokenplot()\n(pip install brokenaxes)')
    
    if ax is None: fig, ax = plt.subplots(figsize=(12,3))
    time, y, yerr = clean_up(time, y, yerr, time_format)
    if clip:
        flux, mask_lower, mask_upper = _clip_helper(time, y)
        
    ind0 = [0] + list(np.where(np.diff(time)>dt)[0]+1) #start indices of data chunks
    ind1 = list(np.where(np.diff(time)>dt)[0]) + [len(time)-1] #end indices of data chunks
    xlims = [ (time[i]-(time[j]-time[i])/100.,time[j]+(time[j]-time[i])/100.) for i,j in zip(ind0,ind1) ]
    
    ax.set_axis_off() #empty the axis before brokenaxes does its magic
    bax = brokenaxes(xlims=xlims, subplot_spec=ax.get_subplotspec())
    bax.errorbar(time, y, yerr=yerr, fmt=fmt, ms=2, rasterized=True)
    
    if trend is not None:
        bax.plot(time, trend, 'r-', lw=2, zorder=100)
    
    if clip:
        bax.plot(time*mask_upper, bax.axs[0].get_ylim()[1]*mask_upper, 'r^', color='orange', ms=10, zorder=11)
        bax.plot(time*mask_lower, bax.axs[0].get_ylim()[0]*mask_lower, 'rv', color='orange', ms=10, zorder=11)
    # bax.ticklabel_format(axis='y', style='sci', useOffset=True)
    # plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    guess_labels(bax, time, y)
    bax.set_ylabel('Flux\n')
    
    return bax



def brokenplot_csv(fname, dt=10, ax=None, time_format='BJD_TDB'):
    '''
    Wrapper around brokenplot to plot straight from a csv file.
    See brokenplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return brokenplot(time, y, yerr, dt=dt, ax=ax, time_format=time_format)



def tessplot(time, flux, flux_err=None, trend=None, time_format='BJD_TDB', clip=False, sharey=False, axes=None, shade=True, **kwargs):
    '''
    Creates a new line for every new TESS Sector
    
    Parameters
    ----------
    time : array of float
        Time stamps (in days).
    flux : array of float
        Flux.
    flux_err : array of float, optional
        Flux error bars. The default is None.
    time_format : str
        The format of your time array.  Must be either 'BJD_TDB' or 
        'TJD' (TESS Julian Date). The default is 'BJD_TDB'.
    clip : bool, optional
        Automatically clip the input data with sigma_clip(low=4, high=4)
        and slide_clip(window_length=1, low=4, high=4). The default is True.
    sharey : bool
        Share the y-axis between all rows. The default is True.
    kwargs : ...
        Any keyword arguments you wish to pass to the plot function (e.g., color='red')

    Returns
    -------
    None.
    '''
    
    time, flux, flux_err = clean_up(time, flux, flux_err, time_format)
    if clip:
        flux, mask_lower, mask_upper = _clip_helper(time, flux)
        
    here = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv( os.path.join(here,'_static','_tess','tess_orbit_times_by_sector.csv'), skiprows=5)
    
    inds, sectors = [], []
    for s in range(1,max(df['Sector'])):
        line = 2*(s-1)
        t0 = float(df['Start TJD'].loc[line])
        t1 = float(df['End TJD'].loc[line+1])
        ind = np.where( (time > t0) & (time <= t1) )[0]
        if len(ind)>0:
            inds += [list(ind)]
            sectors += [s]
    
    N = len(inds)
        
    if axes is None:
        fig, axes = plt.subplots(N, figsize=(12,3*N), tight_layout=True, sharey=sharey)
        
    for i in range(N):
        # ax1 = fig.add_subplot(gssub[i,0])
        ax1 = np.atleast_1d(axes)[i]
        ind, s = inds[i], sectors[i]
        line = 2*(s-1)
        if flux_err is None:
            try:
                kwargs2 = kwargs.copy()
                del kwargs2['size']
                del kwargs2['color']
                ax1.scatter(time[ind], flux[ind], marker='.', s=kwargs['size'][ind], color=kwargs['color'][ind], rasterized=True, **kwargs2)
            except:
                ax1.plot(time[ind], flux[ind], 'b.', ms=2, rasterized=True, **kwargs)
        else:
            ax1.errorbar(time[ind], flux[ind], yerr=flux_err[ind], fmt='b.', ms=2, rasterized=True, **kwargs)
        if clip:
            ax1.scatter(time[ind]*mask_upper[ind], ax1.get_ylim()[1]*mask_upper[ind], marker='^', color='orange', s=36, zorder=11)
            ax1.scatter(time[ind]*mask_lower[ind], ax1.get_ylim()[0]*mask_lower[ind], marker='v', color='orange', s=36, zorder=11)
        if trend is not None:
            ax1.plot(time[ind], trend[ind], 'r-', lw=2, rasterized=True)
        ax1.text(0.98,0.95,'Sector '+str(s),ha='right',va='top',transform=ax1.transAxes)
        orb0 = float(df['Start TJD'].loc[line])
        orb1 = float(df['End TJD'].loc[line])
        orb2 = float(df['Start TJD'].loc[line+1])
        orb3 = float(df['End TJD'].loc[line+1])
        if shade:
            ax1.axvspan(orb0,orb1,color='b',alpha=0.1,zorder=-11)
            ax1.axvspan(orb2,orb3,color='b',alpha=0.1,zorder=-11)
        ax1.set(xlim=[orb0-0.5, orb3+0.5], ylabel='Flux')
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        if i==N-1: ax1.set(xlabel=r'Time (BJD$_\mathrm{TDB}$ - '+str(TJD_offset)+')')
    
    return axes
    


def tessplot_csv(fname, time_format='BJD_TDB', **kwargs):
    '''
    Wrapper around tessplot to plot straight from a csv file.
    See tessplot() for details.
    '''
    time, flux, flux_err = read_csv(fname)[0:3]
    return tessplot(time, flux, flux_err, time_format=time_format, **kwargs)    



def monthplot(time, flux, flux_err=None, trend=None, clip=False, sharey=False, axes=None, **kwargs):
    '''
    Creates a new line for every calendar month
    
    Parameters
    ----------
    time : array of float
        Time stamps (in days).
    flux : array of float
        Flux.
    flux_err : array of float, optional
        Flux error bars. The default is None.
    time_format : str
        The format of your time array.  Must be either 'BJD_TDB' or 
        'TJD' (TESS Julian Date). The default is 'BJD_TDB'.
    clip : bool, optional
        Automatically clip the input data with sigma_clip(low=4, high=4)
        and slide_clip(window_length=1, low=4, high=4). The default is True.
    sharey : bool
        Share the y-axis between all rows. The default is True.
    kwargs : ...
        Any keyword arguments you wish to pass to the plot function (e.g., color='red')

    Returns
    -------
    None.
    '''
    time, flux, flux_err = clean_up(time, flux, flux_err, time_format=None)
    if clip:
        flux, mask_lower, mask_upper = _clip_helper(time, flux)
    
    #::: get the months
    start_of_month = [str(year)+'-'+str(month).zfill(2)+'-01T00:00:00' 
                  for year in range(1950,2050) for month in range(1,12)]
    start_of_month = Time(start_of_month, format='isot', scale='utc')
    start_of_month_str = [t.strftime('%Y %b') for t in start_of_month]

    inds, start_jds, end_jds, months = [], [], [], []
    for i in range(len(start_of_month)-1):
        t0 = start_of_month.jd[i]
        t1 = start_of_month.jd[i+1]-1e-12
        ind = np.where( (time > t0) & (time <= t1) )[0]
        if len(ind)>0:
            inds.append( list(ind) )
            start_jds.append( start_of_month.jd[i] )
            end_jds.append( start_of_month.jd[i+1]-1e-12 )
            months.append( start_of_month_str[i] )
    
    N = len(inds)
    
    if axes is None:
        fig, axes = plt.subplots(N, figsize=(12,3*N), tight_layout=True, sharey=sharey)
    
    for i in range(N):
        # ax1 = fig.add_subplot(gssub[i,0])
        ax1 = np.atleast_1d(axes)[i]
        ind, start, end, month = inds[i], start_jds[i], end_jds[i], months[i]
        # line = 2*(s-1)
        if flux_err is None:
            ax1.plot(time[ind], flux[ind], 'b.', ms=2, rasterized=True, **kwargs)
        else:
            ax1.errorbar(time[ind], flux[ind], yerr=flux_err[ind], fmt='b.', ms=2, rasterized=True, **kwargs)
        if clip:
            ax1.plot(time[ind]*mask_upper[ind], ax1.get_ylim()[1]*mask_upper[ind], 'r^', color='orange', ms=10, zorder=11)
            ax1.plot(time[ind]*mask_lower[ind], ax1.get_ylim()[0]*mask_lower[ind], 'rv', color='orange', ms=10, zorder=11)
        if trend is not None:
            ax1.plot(time[ind], trend[ind], 'r-', lw=2, rasterized=True)
        ax1.text(0.98,0.95,month,ha='right',va='top',transform=ax1.transAxes)
        # orb0 = float(df['Start TJD'].loc[line])
        # orb1 = float(df['End TJD'].loc[line])
        # orb2 = float(df['Start TJD'].loc[line+1])
        # orb3 = float(df['End TJD'].loc[line+1])
        # ax1.plot([orb0,orb1],[1,1],'k-',c='silver',lw=2)
        # ax1.plot([orb2,orb3],[1,1],'k-',c='silver',lw=2)
        # if shade:
        #     ax1.axvspan(orb0,orb1,color='b',alpha=0.1,zorder=-11)
        #     ax1.axvspan(orb2,orb3,color='b',alpha=0.1,zorder=-11)
        ax1.set(xlim=[start-0.5, end+0.5], ylabel='Flux')
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        if i==N-1: ax1.set(xlabel=r'Time (BJD$_\mathrm{TDB}$ - '+str(TJD_offset)+')')
    
    return fig
    


def monthplot_csv(fname, **kwargs):
    '''
    Wrapper around tessplot to plot straight from a csv file.
    See tessplot() for details.
    '''
    time, flux, flux_err = read_csv(fname)[0:3]
    return monthplot(time, flux, flux_err, **kwargs)    



def _clip_helper(time, y):
    y, _, mask_upper0, mask_lower0 = sigma_clip(time, y, low=4, high=4, return_mask=True)
    y, _, mask_upper, mask_lower = slide_clip(time, y, window_length=1, low=4, high=4, return_mask=True)
    mask_upper = np.array(mask_upper0 | mask_upper, dtype=float)
    mask_upper[mask_upper==0] = np.nan
    mask_lower = np.array(mask_lower0 | mask_lower, dtype=float)
    mask_lower[mask_lower==0] = np.nan
    
    return y, mask_lower, mask_upper