#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:05:28 2022

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

from allesfitter.inout import read_csv
from allesfitter.plotting import *




'''
Try out all plot styles, directly loading from a csv file
'''
ax = tessplot_csv('TESS.csv')
brokenplot_csv('TESS.csv', dt=3)
chunkplot_csv('TESS.csv')
monthplot_csv('TESS.csv')
tessplot_csv('TESS.csv')


'''
Now on to trickier stuff; first, let us load the data and simulate a trend
'''
time, flux, flux_err = read_csv('TESS.csv')
color = np.sin(2*np.pi*time/10.)
trend_time = np.linspace(2458390, 2458450, 10000)
trend_flux = 1.+0.001*np.sin(2*np.pi*trend_time/10.)


'''
Now let us make all kinds of plots with weird features; clip outliers,
mark data ranges (e.g. bad data or transits), use scatter colors, and overplot the simulated trend
'''
ax = fullplot(time, flux, yerr=flux_err, clip=True, color=color, cmap='Blues_r')
fullplot(time[18000:21000], flux[18000:21000], yerr=flux_err[18000:21000], color='orange', ax=ax)
fullplot(trend_time, trend_flux, marker='', ls='-', color='r', ax=ax)


bax = brokenplot(time, flux, yerr=flux_err, dt=3, clip=True, color=color, cmap='Blues_r')
brokenplot(time[18000:21000], flux[18000:21000], yerr=flux_err[18000:21000], color='orange', bax=bax)
brokenplot(trend_time, trend_flux, marker='', ls='-', color='r', bax=bax)


axes = chunkplot(time, flux, yerr=flux_err, clip=True)
chunkplot(time[18000:21000], flux[18000:21000], yerr=flux_err[18000:21000], color='orange', axes=axes)
chunkplot(trend_time, trend_flux, marker='', ls='-', color='r', axes=axes)


axes = monthplot(time, flux, yerr=flux_err, clip=True, color=color, cmap='Blues_r') 
monthplot(time[18000:21000], flux[18000:21000], yerr=flux_err[18000:21000], color='orange', axes=axes)
monthplot(trend_time, trend_flux, marker='', ls='-', color='r', axes=axes)


axes = tessplot(time, flux, yerr=flux_err, clip=True, color=color, cmap='Blues_r', vmin=-0.5, vmax=0.5)
tessplot(time[18000:21000], flux[18000:21000], yerr=flux_err[18000:21000], clip=True, color='orange', axes=axes)
tessplot(trend_time, trend_flux, marker='', ls='-', color='r', axes=axes)