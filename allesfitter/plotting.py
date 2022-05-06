#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:27:01 2020

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

#::: modules
import os, sys
import numpy as np
from matplotlib.colors import ColorConverter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorsys
import warnings
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
from datetime import date

#::: local imports
from allesfitter.inout import read_csv
from allesfitter.time_series import sigma_clip, slide_clip

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth':1})



'''
This script offers several convenience functions for plotting diverse time series data, 
and particularly lightcurves. These are, in order of generality:
    - fullplot
    - brokenplot
    - chunkplot
    - monthplot
    - tessplot
'''
    


def fullplot(time, y, yerr=None, clip=False, ax=None, **kwargs):
    '''
    Parameters
    ----------
    time : array of float
        Time stamps, can be in any format of days.
    y : array of float
        e.g. y or RV array (usually as normalized y or RV in km/s).
    yerr : array of float
        e.g. y or RV error array (usually as normalized y or RV in km/s).
    ax : matplotlib axis, optional
        An existing figure axis. The default is None.
    clip : bool, optional
        Automatically clip the input data with sigma_clip(low=4, high=4)
        and slide_clip(window_length=1, low=4, high=4). The default is True.

    Returns
    -------
    ax : matplotlib axis
        A single figure axis.
    '''
    #::: initialize
    time, y, yerr, ax, kwargs, mask_lower, mask_upper = _initialize(time, y, yerr, ax, clip, kwargs)
    
    #::: plot
    _plot1(ax, slice(None), time, y, yerr, mask_upper, mask_lower, clip, **kwargs)
        
    #::: make the axes pretty
    _set_axes(ax, time, y)
    
    #::: return
    return ax



def fullplot_csv(fname, clip=False, ax=None):
    '''
    Wrapper around fullplot to plot straight from a csv file.
    See fullplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return fullplot(time, y, yerr, clip=clip, ax=ax)



def brokenplot(time, y, yerr=None, dt=10, clip=False, bax=None, **kwargs):
    '''
    Parameters
    ----------
    time : array of float
        e.g. time array (usually in days).
    y : array of float
        e.g. y or RV array (usually as normalized y or RV in km/s).
    yerr : array of float
        e.g. y or RV error array (usually as normalized y or RV in km/s).
    dt : float, optional
        The gap size after which axes will be broken. The default is 10 (usually in days).
    bax : brokenaxes instance, optional
        An existing brokenaxis axis. Note that this cannot be a normal matplotlib axis. 
        The default is None.
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
        Just like an pyplot.Axes instance.
    '''
    #::: check if brokenaxes is installed
    if 'brokenaxes' not in sys.modules:
        raise ImportError('You must install the brokenaxes package before using allesfitter.brokenplot()\n(pip install brokenaxes)')
    
    #::: initialize (but skip the figure/axes creation here)
    time, y, yerr, _, kwargs, mask_lower, mask_upper = _initialize(time, y, yerr, 1, clip, kwargs)
    
    #::: special setup for brokenaxes
    if bax is None:
        plt.figure(figsize=(12,3))    
        ind0 = [0] + list(np.where(np.diff(time)>dt)[0]+1) #start indices of data chunks
        ind1 = list(np.where(np.diff(time)>dt)[0]) + [len(time)-1] #end indices of data chunks
        xlims = [ (time[i]-(time[j]-time[i])/100.,time[j]+(time[j]-time[i])/100.) for i,j in zip(ind0,ind1) ]
        bax = brokenaxes(xlims=xlims, wspace=0.05)
        
    #::: plot    
    _plot1(bax, slice(None), time, y, yerr, mask_upper, mask_lower, clip, is_bax=True, **kwargs)

    #::: make the axes pretty (special one necessary here)
    _set_brokenaxes(bax, time, y)  
    
    #::: return
    return bax



def brokenplot_csv(fname, dt=10, clip=False, bax=None, **kwargs):
    '''
    Wrapper around brokenplot to plot straight from a csv file.
    See brokenplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return brokenplot(time, y, yerr, dt=dt, clip=clip, bax=bax, **kwargs)



def chunkplot(time, y, yerr=None, chunk_size=30., clip=False, sharey=False, axes=None, **kwargs):
    '''
    Creates a new line for every chunk of days determined by chunk_size, with
    the count starting at the first time stamp. The time format can be arbitrary here.
    
    Parameters
    ----------
    time : array of float
        Time stamps, can be in any format of days.
    y : array of float
        y.
    y_err : array of float, optional
        y error bars. The default is None.
    chunk_size : float
        The length of each chunk of days to be plotted in a single axis. 
        The default is 30., i.e. roughly a month.
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
    #::: initialize (but skip the figure/axes creation here)
    time, y, yerr, _, kwargs, mask_lower, mask_upper = _initialize(time, y, yerr, 1, clip, kwargs)
    
    #::: get chunks of chunk_size days (in the same units of time)
    start_of_chunk = [ np.nanmin(time)+i*chunk_size for i in range( 1+int(1.*np.ptp(time)/chunk_size) ) ]
    end_of_chunk = [ x + chunk_size - 1e-12 for x in start_of_chunk ]
    start_of_chunk_str = [ 'Chunk '+str(i+1) for i in range(len(start_of_chunk)) ]
    
    #::: if this is a brand new figure, match the init_time to TESS orbits and sectors and then set up all the axes
    if axes is None:
            
        #::: select which months are covered by the data
        chunks, xlims0, xlims1  = [], [], [],
        for i in range(len(start_of_chunk)):
            t0 = start_of_chunk[i]
            t1 = end_of_chunk[i]
            ind = np.where( (time > t0) & (time <= t1) )[0]
            if len(ind)>0:
                chunks.append( start_of_chunk_str[i] )
                xlims0.append( start_of_chunk[i] )
                xlims1.append( end_of_chunk[i] )
    
        #::: set up the figure
        N = len(chunks)
        fig, axes = plt.subplots(N, figsize=(12,3*N), tight_layout=True, sharey=sharey)
            
        #::: loop over all months/axes to set up each axis
        for i, c in enumerate(chunks):
            ax1 = np.atleast_1d(axes)[i]
            
            #::: add text
            ax1.text(0.98, 0.95, c, ha='right', va='top', transform=ax1.transAxes, zorder=15)
            
            #::: set the axis xlim and labels properly
            ax1.set(xlim=[xlims0[i], xlims1[i]])    
            _set_axes(ax1, time, y, labels={'show_x': (i==N-1)})    
        
    #::: iterate over all existing, set-up axes and plot
    for i, ax1 in enumerate(np.atleast_1d(axes)):
        
        #::: check if there is data to be plotted on this axis
        ind = np.where( (time > ax1.get_xlim()[0]) & (time <= ax1.get_xlim()[1]) )[0]
        
        #::: only proceed if there is actually data in this range
        if len(ind)>0:
            
            #::: plot (check if it should be a scatter or errorbar plot)
            _plot1(ax1, ind, time, y, yerr, mask_upper, mask_lower, clip, **kwargs)
            
    #::: return
    return axes



def chunkplot_csv(fname, **kwargs):
    '''
    Wrapper around tessplot to plot straight from a csv file.
    See tessplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return chunkplot(time, y, yerr, **kwargs)  



def monthplot(time, y, yerr=None, clip=False, sharey=False, axes=None, **kwargs):
    '''
    Creates a new line for every calendar month.
    The time format must be BJD_TDB here.
    
    Parameters
    ----------
    time : array of float
        Time stamps, must be in BJD_TDB (do not use arbitrary days nor TJD).
    y : array of float
        Relative flux, RV, or residuals.
    yerr : array of float, optional
        Error bars of the y values. The default is None.
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
    #::: sanity check the data
    jd_moon_landing = 2440422.5 #(fake)
    if np.nanmin(time) < jd_moon_landing:
        raise ValueError('The time array does not seem to be in units of BJD_TDB.')
        
    #::: initialize (but skip the figure/axes creation here)
    time, y, yerr, _, kwargs, mask_lower, mask_upper = _initialize(time, y, yerr, 1, clip, kwargs)
    
    #::: get the months since the moon landing (fake)
    start_of_month = [str(year)+'-'+str(month).zfill(2)+'-01T00:00:00' 
                      for year in range(1969,date.today().year+1) for month in range(1,13)]
    start_of_month = Time(start_of_month, format='isot', scale='utc')
    start_of_month_str = [t.strftime('%Y %b') for t in start_of_month]
    
    #::: if this is a brand new figure, match the init_time to TESS orbits and sectors and then set up all the axes
    if axes is None:
            
        #::: select which months are covered by the data
        months, xlims0, xlims1  = [], [], [],
        for i in range(len(start_of_month)-1):
            t0 = start_of_month.jd[i]
            t1 = start_of_month.jd[i+1]-1e-12
            ind = np.where( (time > t0) & (time <= t1) )[0]
            if len(ind)>0:
                months.append( start_of_month_str[i] )
                xlims0.append( start_of_month.jd[i] )
                xlims1.append( start_of_month.jd[i+1]-1e-12 )
    
        #::: set up the figure
        N = len(months)
        fig, axes = plt.subplots(N, figsize=(12,3*N), tight_layout=True, sharey=sharey)
            
        #::: loop over all months/axes to set up each axis
        for i, m in enumerate(months):
            ax1 = np.atleast_1d(axes)[i]
            
            #::: add text
            ax1.text(0.98, 0.95, m, ha='right', va='top', transform=ax1.transAxes, zorder=15)
            
            #::: set the axis xlim and labels properly
            ax1.set(xlim=[xlims0[i], xlims1[i]])    
            _set_axes(ax1, time, y, labels={'show_x': (i==N-1)})    
        
    #::: iterate over all existing, set-up axes and plot
    for i, ax1 in enumerate(np.atleast_1d(axes)):
        
        #::: check if there is data to be plotted on this axis
        ind = np.where( (time > ax1.get_xlim()[0]) & (time <= ax1.get_xlim()[1]) )[0]
        
        #::: only proceed if there is actually data in this range
        if len(ind)>0:
            
            #::: plot (check if it should be a scatter or errorbar plot)
            _plot1(ax1, ind, time, y, yerr, mask_upper, mask_lower, clip, **kwargs)
    
    #::: return
    return axes
    


def monthplot_csv(fname, **kwargs):
    '''
    Wrapper around tessplot to plot straight from a csv file.
    See tessplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return monthplot(time, y, yerr, **kwargs)    
    


def tessplot(time, y, yerr=None, clip=False, sharey=False, axes=None, **kwargs):
    '''
    A collage plot for long TESS lightcurves. Creates a new axes row for every TESS Sector.
    The time input must be in BJD_TDB. The first call to this function must fully initialize it,
    i.e. it must cover all time stamps used to select the right sectors and set the axes properly.
    If your first call of tessplot only uses a fraction of the full data and you plan to add more later,
    you must use init_time to pass the correct, full time stamps to correctly initialize the figure and all axes.
    
    Parameters
    ----------
    time : array of float
        Time stamps, must be in BJD_TDB (do not use arbitrary days nor TJD).
    y : array of float
        Relative flux, RV, or residuals.
    yerr : array of float, optional
        Error bars of the y values. The default is None.
    clip : bool, optional
        Automatically clip the input data with sigma_clip(low=4, high=4)
        and slide_clip(window_length=1, low=4, high=4). The default is False.
    sharey : bool
        Share the y-axis between all rows. The default is False.
    axes : matplotlib axes, optional
        Uses existing figure axes instead of making a new figure. The default is None.
    kwargs : ...
        Any keyword arguments you wish to pass to the plot function (e.g., color='r')

    Returns
    -------
    ax : matplotlib axes
        The figure axes.
    '''
    #::: sanity check the data
    jd_moon_landing = 2440422.5 #(fake)
    if np.nanmin(time) < jd_moon_landing:
        raise ValueError('The time array does not seem to be in units of BJD_TDB.')
    
    #::: initialize (but skip the figure/axes creation here)
    time, y, yerr, _, kwargs, mask_lower, mask_upper = _initialize(time, y, yerr, 1, clip, kwargs)
    
    #::: load the TESS orbit times
    here = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv( os.path.join(here,'_static','_tess','tess_orbit_times.csv'), comment="#" )
    
    #::: if this is a brand new figure, match the init_time to TESS orbits and sectors and then set up all the axes
    if axes is None:
            
        #::: count which sectors are covered by init_time
        sectors, xlims0, xlims1 = [], [], []
        for s in range(1,max(df['Sector'])):
            line = 2*(s-1)
            t0 = Time(df['Start of Orbit'].loc[line]).jd
            t1 = Time(df['End of Orbit'].loc[line+1]).jd
            ind = np.where( (time > t0) & (time <= t1) )[0]
            if len(ind)>0:
                sectors.append(s)
                xlims0.append(t0-0.5)
                xlims1.append(t1+0.5)
    
        #::: set up the figure
        N = len(sectors)
        fig, axes = plt.subplots(N, figsize=(12,3*N), tight_layout=True, sharey=sharey)
            
        #::: loop over all sectors/axes to set up each axis
        for i, s in enumerate(sectors):
            line = 2*(s-1)
            ax1 = np.atleast_1d(axes)[i]
            
            #::: mark the TESS sectors      
            orb0 = Time(df['Start of Orbit'].loc[line]).jd
            orb1 = Time(df['End of Orbit'].loc[line]).jd
            orb2 = Time(df['Start of Orbit'].loc[line+1]).jd
            orb3 = Time(df['End of Orbit'].loc[line+1]).jd
            
            #::: add text and vspans
            ax1.text(0.98, 0.95, 'Sector '+str(s), ha='right', va='top', transform=ax1.transAxes, zorder=15)
            ax1.axvspan(orb0,orb1,color='b',alpha=0.1,zorder=-11)
            ax1.axvspan(orb2,orb3,color='b',alpha=0.1,zorder=-11)
            
            #::: set the axis xlim and labels properly
            ax1.set(xlim=[orb0-0.5, orb3+0.5])    
            _set_axes(ax1, time, y, labels={'show_x': (i==N-1)})    
        
    #::: iterate over all existing, set-up axes and plot
    for i, ax1 in enumerate(np.atleast_1d(axes)):
        
        #::: check if there is data to be plotted on this axis
        ind = np.where( (time > ax1.get_xlim()[0]) & (time <= ax1.get_xlim()[1]) )[0]
        
        #::: only proceed if there is actually data in this range
        if len(ind)>0:
            
            #::: plot (check if it should be a scatter or errorbar plot)
            _plot1(ax1, ind, time, y, yerr, mask_upper, mask_lower, clip, **kwargs)
                
    #::: return
    return axes
    


def tessplot_csv(fname, **kwargs):
    '''
    Wrapper around tessplot to plot straight from a csv file.
    See tessplot() for details.
    '''
    time, y, yerr = read_csv(fname)[0:3]
    return tessplot(time, y, yerr, **kwargs)   



def _initialize(time, y, yerr, ax, clip, kwargs):
    '''
    Prepares the data and figure. 
    '''
    #::: set default kwargs (via the pythonic spread operator)
    kwargs = {'color': 'b', 
              'marker': '.', 
              'ls': 'none', 
              'ms': 2,
              'rasterized': True,
              **kwargs}
    kwargs = {'ecolor': _get_errorbar_color(kwargs['color']),
              **kwargs}
        
    #::: clean up the data
    time, y, yerr = _clean_up(time, y, yerr)
    
    #::: clip the data (optional)
    if clip:
        y, mask_lower, mask_upper = _clip_helper(time, y)
    else:
        mask_lower, mask_upper = None, None
        
    #::: prepare the figure (optional)
    if ax is None: 
        fig, ax = plt.subplots(figsize=(12,3), tight_layout=True)
        
    #::: return
    return time, y, yerr, ax, kwargs, mask_lower, mask_upper



def _clean_up(time, y, yerr=None):    
    '''
    Just makes sure the input is in a format that can be handled.
    '''
    #::: convert to numpy arrays
    time = np.array(time)    
    y = np.array(y)    
    if yerr is not None: 
        yerr = np.array(yerr)    
    
    #::: return
    return time, y, yerr
        
        
        
def _clip_helper(time, y):
    '''
    Helps with clipping the data.
    '''
    y, _, mask_upper0, mask_lower0 = sigma_clip(time, y, low=4, high=4, return_mask=True)
    y, _, mask_upper, mask_lower = slide_clip(time, y, window_length=1, low=4, high=4, return_mask=True)
    mask_upper = np.array(mask_upper0 | mask_upper, dtype=float)
    mask_upper[mask_upper==0] = np.nan
    mask_lower = np.array(mask_lower0 | mask_lower, dtype=float)
    mask_lower[mask_lower==0] = np.nan
    
    return y, mask_lower, mask_upper



def _set_yticklabels(ax):
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))



def _guess_ylabels(y):
    '''
    Sets the y and xlabels automatically by guessing whether it is a lightcurve, 
    RV, or a residuals time series.
    '''
    if (np.nanpercentile(y,5) > 0.7) & (np.nanpercentile(y,95) < 1.3):
        return 'Relative flux', None
    elif (np.nanpercentile(y,5) > -0.02) & (np.nanpercentile(y,95) < 0.02):
        return 'Residuals', None
    else:
        return 'RV', 'km/s'
        
        

def _set_xticklabels(ax, time_offset):
    '''
    Amends axes' xticklabels (rather than changing the value of "time" that we plot.)
    This is so that we do not cause complications if we want to reuse the same axis and plot something else on here.
    All subsequent plots can thus use the original time stamps and their data will be drawn correctly.
    '''
    def mynotation(time, pos):
        return '{:g}'.format(time - time_offset)
    MyFormatter = FuncFormatter(mynotation)
    ax.xaxis.set_major_formatter(MyFormatter)
    return ax



def _guess_xlabels(time):
    '''
    Guess the xlabels automatically between 'days' and 'BJD_TDB'.
    '''
    jd_moon_landing = 2440422.5
    if np.min(time) < jd_moon_landing:
        return 'Time', 'days'
    else:
        return 'Time', 'BJD$_\mathrm{TDB}$'
    


def _set_axes(ax, time, y, labels={}):
    '''
    Sets the axes ticks and labels automatically.
    '''
    #::: default labels and automatic guesses, unless overwritten by user-input
    labels = {'x': _guess_xlabels(time)[0],
              'x_unit': _guess_xlabels(time)[1],
              'y': _guess_ylabels(y)[0],
              'y_unit': _guess_ylabels(y)[1],
              'show_x': True,
              'show_y': True,
              **labels}
    
    #::: compute time offset (in steps of 1,000 days)
    time_offset = 1000*int(1.*np.min(time)/1e3)
    
    #::: handle the ylabels
    _set_yticklabels(ax)
    if labels['show_y']:
        if labels['y_unit'] is None:
            ax.set_ylabel( labels['y'] )
        else:
            ax.set_ylabel( labels['y'] + ' (' + labels['y_unit'] +')' )
        
    #::: handle the xlabels
    _set_xticklabels(ax, time_offset)
    if labels['show_x']:
        if labels['x_unit'] is None:
            ax.set_xlabel( labels['x'] + ' - ' + '{:,.0f}'.format(time_offset) )
        else:
            ax.set_xlabel( labels['x'] + ' (' + labels['x_unit'] + ' - ' + '{:,.0f}'.format(time_offset)+')' )
  
        
  
def _set_brokenaxes(bax, time, y, labels={}):
    '''
    Sets the axes ticks and labels automatically.
    '''
    #::: default labels and automatic guesses, unless overwritten by user-input
    labels = {'x': _guess_xlabels(time)[0],
              'x_unit': _guess_xlabels(time)[1],
              'y': _guess_ylabels(y)[0],
              'y_unit': _guess_ylabels(y)[1],
              'show_x': True,
              'show_y': True,
              **labels}
    
    #::: compute time offset (in steps of 1,000 days)
    time_offset = 1000*int(1.*np.min(time)/1e3)
    
    #::: handle the ylabels
    _set_yticklabels(bax.axs[0])
    if labels['show_y']:
        if labels['y_unit'] is None:
            bax.axs[0].set_ylabel( labels['y'] )
        else:
            bax.axs[0].set_ylabel( labels['y'] + ' (' + labels['y_unit'] +')' )
        
    #::: handle the xlabels
    for ax in bax.axs:
        _set_xticklabels(ax, time_offset)
    if labels['show_x']:
        if labels['x_unit'] is None:
            bax.set_xlabel( labels['x'] + ' - ' + '{:,.0f}'.format(time_offset) )
        else:
            bax.set_xlabel( labels['x'] + '(' + labels['x_unit'] + ' - ' + '{:,.0f}'.format(time_offset)+')' )
        
        

def _scale_lightness(rgb, scale_l):
    '''
    Returns a lightened version of any rgb color 
    '''
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)



def _get_errorbar_color(color):
    '''
    Converts a color string to rgb values and then returns a lighter version
    '''
    try:
        rgb = ColorConverter.to_rgb(color)
        return _scale_lightness(rgb, 1.75)
    except:
        return None



def _none_slice(x, ind):
    '''
    Tries to slice an array and skips None instances
    '''
    if x is None:
        return None
    else:
        return x[ind]

    

def _plot1(ax1, ind, time, y, yerr, mask_upper, mask_lower, clip, is_bax=False, **kwargs):
    '''
    Plots all the stuff onto one given axis
    '''
    #::: select the right data
    time1 = time[ind]
    y1 = y[ind]
    yerr1 = _none_slice(yerr, ind)
    mask_upper1 = _none_slice(mask_upper, ind)
    mask_lower1 = _none_slice(mask_lower, ind)
    
    #::: exception for brokenaxes
    if is_bax:
        ax1b = ax1.axs[-1]
    else: 
        ax1b = ax1
    
    #::: decide if it is an errorbar or scatter plot
    if len(kwargs['color']) < len(time):
        ax1.errorbar(time1, y1, yerr=yerr1, **kwargs)
    else:
        kwargs1 = kwargs.copy()
        color1 = kwargs['color'][ind]
        kwargs1.pop('color')
        kwargs1.pop('ecolor')
        kwargs1.pop('ls')
        kwargs1.pop('ms')
        kwargs1 = {'vmin': np.nanmin(kwargs['color']),
                   'vmax': np.nanmax(kwargs['color']),
                   **kwargs1} #this is important when breaking up the data in different axes
        sc = ax1.scatter(time1, y1, c=color1, **kwargs1)
        divider = make_axes_locatable(ax1b)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        if is_bax: sc = sc[0]
        plt.colorbar(sc, cax=cax, orientation='vertical', label='Prob.') 
        #for now, the colorbar label is frozen to "Prob."; you can always overwrite it later
            
    #::: indicate clipped data (optional)
    if clip:
        ax1.plot(time1*mask_upper1, ax1b.get_ylim()[1]*mask_upper1, marker='^', ls='none', color='orange', ms=10, zorder=11)
        ax1.plot(time1*mask_lower1, ax1b.get_ylim()[0]*mask_lower1, marker='v', ls='none', color='orange', ms=10, zorder=11)


