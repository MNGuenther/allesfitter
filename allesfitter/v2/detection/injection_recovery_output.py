#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:01:30 2019

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
import warnings
#import os, sys
#import scipy.ndimage
#from scipy.interpolate import griddata
#import matplotlib.ticker as mtick

    
    
    
###############################################################################
#::: fct to check if the right signal was found
###############################################################################
def is_multiple_of(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b) 
    result = a % b
    return (abs(result/b) <= tolerance) or (abs((b-result)/b) <= tolerance)



def is_detected(inj_period, tls_period):
    right_period = is_multiple_of(tls_period, inj_period/2.) #check if it is a multiple of half the period to within 5%
    
#    right_epoch = False
#    for tt in results.transit_times:
#        for i in range(-5,5):
#            right_epoch = right_epoch or (np.abs(tt-epoch+i*period) < (1./24.)) #check if any epochs matches to within 1 hour
             
#    right_depth   = (np.abs(np.sqrt(1.-results.depth)*rstar - rplanet)/rplanet < 0.05) #check if the depth matches
                
    if right_period:
        return True
    else:
        return False
                    
    
    
def is_detected_list(inj_periods, tls_periods):
    detected = []
    for i in range(len(inj_periods)):
        detected.append(is_detected(inj_periods[i], tls_periods[i]))
    return np.array(detected)
    
    

###############################################################################
#::: plot
###############################################################################
def irplot(fname, period_bins=None, rplanet_bins=None, outdir=None):
    
    #::: load the files and check which TLS detection matches an injection; 
    #::: note that one injection will have multiple TLS detections (due not false positives)
    results = np.genfromtxt(fname, delimiter=',', dtype=None, names=True)
    inj_periods = results['inj_period']
    inj_rplanets = results['inj_rplanet']
    tls_periods = results['tls_period']
    detected = is_detected_list(inj_periods, tls_periods)
    # print(detected)
    
    #::: now boil it down to unique injections and see whether any TLS detection matched it
    # period = np.unique(inj_periods)
    # rplanet = np.unique(inj_rplanets)
    period = []
    rplanet = []
    found = []
    for p in np.unique(inj_periods):
        for r in np.unique(inj_rplanets):
            period.append(p)
            rplanet.append(r)
            ind = np.where( (inj_periods==p) & (inj_rplanets==r) )[0]
            f = any( detected[ind] )
            found.append(f)
            # print(p,r,ind,f)
    period = np.array(period)
    rplanet = np.array(rplanet)
    found = np.array(found)
    
    
    
    ###############################################################################
    #::: scatter plot
    ###############################################################################
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(period, rplanet, c=found, s=100, cmap='Blues_r', edgecolors='b')
    ax.set(xlabel='Period (days)', ylabel='Radius '+r'$(R_\oplus)$')
    ax.text(0.5,1.05,'filled: not recovered | unfilled: recovered',ha='center',va='center',transform=ax.transAxes)
    fig.savefig('injection_recovery_test_scatter.pdf', bbox_inches='tight')    
    # err
    
    
    
    ###############################################################################
    #::: histogram (normed)
    ###############################################################################
    if ( len(np.unique(inj_periods)) * len(np.unique(inj_periods)) ) < 100:
        print('\n!-- WARNING: not enough samples to create a 2D histogram plot. --!\n')
    else:
        if (period_bins is not None) and (rplanet_bins is not None):
            bins = [period_bins, rplanet_bins]
        else:
            bins = [np.histogram_bin_edges(period, bins='auto'), np.histogram_bin_edges(rplanet, bins='auto')]
        h1,x,y = np.histogram2d(period[found==1], rplanet[found==1], bins=bins)
        h2,x,y = np.histogram2d(period[found==0], rplanet[found==0], bins=bins)
        normed_hist = (100.*h1/(h1+h2))
        
        fig, ax = plt.subplots(figsize=(6.5,5))
        im = plt.imshow(normed_hist.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), interpolation='none', aspect='auto', cmap='Blues_r', vmin=0, vmax=100, rasterized=True)
        plt.colorbar(im, label='Recovery rate (%)')
        plt.xlabel('Injected period (days)')
        plt.ylabel(r'Injected radius (R$_\oplus$)')
        # change_font(ax)
        
        fig.savefig('injection_recovery_test_hist2d.pdf', bbox_inches='tight')       
    
    
    
    ###############################################################################
    #::: pyplot histograms (total counts)
    ###############################################################################
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #h1,x,y,im = plt.hist2d(period[found==1], rplanet[found==1], bins=bins, cmap='Blues_r')
    #plt.colorbar(im, label='Recovery rate (%)')
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    
    
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #h2,x,y,im = plt.hist2d(period[found==0], rplanet[found==0], bins=bins, cmap='Blues_r')
    #plt.colorbar(im, label='Recovery rate (%)')
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    
    
    
    
    ###############################################################################
    #::: kdeplots
    ###############################################################################
    # fig, ax = plt.subplots(figsize=(6.5,5))
    # ax = sns.kdeplot(period[found==0], rplanet[found==0], shade=True, cmap='Blues', cbar=True, alpha=0.5)
    # ax = sns.kdeplot(period[found==1], rplanet[found==1], shade=True, cmap='Blues_r', cbar=True)
    # ax.set(xlim=[0,120], ylim=[0.8,4])
    # plt.xlabel('Injected period (days)')
    # plt.ylabel(r'Injected radius (R$_\oplus$)')
    # change_font(ax)
    
    
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #ax = sns.kdeplot(period[found==0], rplanet[found==0], shade=True, cmap='Blues', cbar=True)
    #ax.set(xlim=[15,85], ylim=[0.8,2.0])
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    
    
    
    
    ###############################################################################
    #::: others
    ###############################################################################
    # plt.figure(figsize=(5,5))
    # z = found.reshape(len(np.unique(period)), len(np.unique(rplanet)))
    # plt.imshow(z.T, origin='lower', extent=(np.amin(period), np.amax(period), np.amin(rplanet), np.amax(rplanet)), aspect='auto', interpolation='gaussian', filterrad=5, cmap='Blues_r')
    # plt.xlabel('Period (days)')
    # plt.ylabel(r'Radius (R$_\oplus$)')
    
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #plt.tricontourf(period, rplanet, found, cmap='Blues_r')
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    
    
    # grid_x, grid_y = np.mgrid[np.amin(period):np.amax(period):100j, np.amin(rplanet):np.amax(rplanet):100j]
    # grid_z = griddata((period, rplanet), found*100, (grid_x, grid_y), method='linear')
    # fig, ax = plt.subplots(figsize=(6.5,5))
    # im = plt.imshow(grid_z.T, origin='lower', extent=(np.amin(period), np.amax(period), np.amin(rplanet), np.amax(rplanet)), interpolation='none', aspect='auto', cmap='Blues_r', rasterized=True, vmin=0, vmax=100)
    # plt.colorbar(im, label='Recovery rate (%)')
    # plt.xlabel('Injected period (days)')
    # plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    #    
    #plt.savefig('injected_transit_search.pdf', bbox_inches='tight')       




###############################################################################
#::: run
###############################################################################
# plot('TIC_269701147.csv')




