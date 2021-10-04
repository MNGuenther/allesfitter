#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:58:19 2020

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
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint

#::: my modules
from .. import get_mcmc_posterior_samples, get_ns_posterior_samples, get_labels

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




def plot_histograms(datadirs, titles, keys, options=None):
        
    if options is None: options = {}
    if 'type' not in options: options['type'] = 'kde' #or 'hist'
    if 'layout' not in options: options['layout'] = '1x1' #or '5x2' etc.
    if 'simplify' not in options: options['simplify'] = False
    if 'colors' not in options: options['colors'] = None
    if 'epoch_median' not in options: options['epoch_median'] = False
    if 'show_plot' not in options: options['show_plot'] = False
    if 'save_plot' not in options: options['save_plot'] = False
    if 'outdir' not in options: options['outdir'] = '.'
    if 'font_scale' not in options: options['font_scale'] = 1.5 + 0.2*(len(keys)-1)

    
    #::: plot settings (need to renew everything 'cause seaborn likes to go cray cray)
    sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=options['font_scale'], color_codes=True)
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context(rc={'lines.markeredgewidth': 1})
    if options['colors'] is not None: 
        pal = sns.color_palette()
        for i,c in enumerate(options['colors']): pal[i] = c
        sns.set_palette(pal)
            
        
    # alles_list = [ allesclass(x) for x in datadirs ] #only load allesclass objects once to save time
    all_params = {}
    all_paramslabels = {}
    for datadir, title in zip(datadirs, titles):
        try: all_params[title] = get_mcmc_posterior_samples(datadir, as_type='dic')
        except: all_params[title] = get_ns_posterior_samples(datadir, as_type='dic')
        all_paramslabels[title] = get_labels(datadir, as_type='dic')
        
    if options['epoch_median']:
        for title in titles:
            for key in keys:
                if 'epoch' in key:
                    all_params[title][key] -= np.median(all_params[title][key])
                    all_paramslabels[title][key] += 'median'
        
    def plot1(y, ax, typ, label=None):
        if typ=='hist': 
            sns.distplot(y, ax=ax, hist_kws={'linewidth':0, 'alpha':0.5, 'density':True}, label=label, legend=False)
        if typ=='kde': 
            sns.kdeplot(y, ax=ax, shade=True, alpha=1, label=label, legend=False)
            sns.kdeplot(y, ax=ax, shade=False, alpha=1, color='k')
        
    
    if options['layout'] == '1x1':
        fig_list, ax_list = [], []
        for i, key in enumerate(keys):
            fig, ax = plt.subplots()
            # for alles, title in zip(alles_list, titles):
                # plot1(alles.posterior_params[key], ax, options['type'], label=title)
            # ax.set(xlabel=alles.labels[key], ylabel='Posterior density')
            for title in titles:
                plot1(all_params[title][key], ax, options['type'], label=title)
            ax.set(xlabel=all_paramslabels[title][key], ylabel='Posterior density')
            if options['simplify']: setup_simplify(ax)
            fig_list.append(fig)
            ax_list.append(ax)
            if options['show_plot']: plt.show(fig)
            else: plt.close(fig)
            if options['save_plot']: fig.savefig(os.path.join(options['outdir'],'histograms.pdf'), bbox_inches='tight')
        return fig_list, ax_list
        
        
    else:
        fig, axes, rows, cols = setup_layout(options['layout'])
        fig.text(0, 0.5, 'Posterior density', ha='center', va='center', rotation='vertical')
        if rows*cols < len(keys): raise ValueError('Given layout must allow at least as many axes as len(keys).')
        for i, key in enumerate(keys):
            # for alles, title in zip(alles_list, titles):
            #     if rows==1 or cols==1: ax = axes[i]
            #     else: ax = axes[ np.unravel_index(i,(rows,cols)) ]
            #     if key==keys[-1]: plot1(alles.posterior_params[key], ax, options['type'], label=title) #only show one label for the last key
            #     else: plot1(alles.posterior_params[key], ax, options['type'])
            #     ax.set(xlabel=alles.labels[key])
            for title in titles:
                if rows==1 or cols==1: ax = axes[i]
                else: ax = axes[ np.unravel_index(i,(rows,cols)) ]
                plot1(all_params[title][key], ax, options['type'], label=title)
                lgd_handles, lgd_labels = ax.get_legend_handles_labels()
                ax.set(xlabel=all_paramslabels[title][key])
                if options['simplify']: setup_simplify(ax)
        for j in range(i+1,len(keys)+1):
            if rows==1 or cols==1: ax = axes[j]
            else: ax = axes[ np.unravel_index(j,(rows,cols)) ]
            ax.axis('off')
        # lgd = fig.legend(lgd_handles, lgd_labels, loc='upper center', ncol=len(titles), bbox_to_anchor=(0.5,1.02))
        # for legobj in lgd_handles: legobj.set_linewidth(4)
        lgd = ax.legend(lgd_handles, lgd_labels, loc='upper right') #last axis gets the legends
        for line in lgd.get_lines(): line.set_linewidth(10)
        if options['show_plot']: plt.show(fig)
        else: plt.close(fig)
        if options['save_plot']: fig.savefig(os.path.join(options['outdir'],'histograms.pdf'), bbox_inches='tight')
        return fig, axes
    


def setup_simplify(ax):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set(yticks=[])



def setup_layout(layout):
    '''
    layout : str
        e.g. '1x1' or '2x3' (rows x cols)
    '''
    rows = int(layout.split('x')[0])
    cols = int(layout.split('x')[1])
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols,4*rows), tight_layout=True)
    return fig, axes, rows, cols