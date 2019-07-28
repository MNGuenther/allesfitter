"""
Pipeline to analyze allesfitter output for planet transit timings

argument 1: allesfitter path
argument 2: p-value threshold
argument 3: Boolean to select to plot wout/with TESS or wout/with/only TESS

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import scipy
import os, datetime, sys

import matplotlib.pyplot as plt

import allesfitter
import allesfitter.postprocessing.plot_viol
from allesfitter import config

import astropy

def plot_viol(pathdataoutp, pvalthrs=1e-3, boolonlytess=False):

    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('allesfitter postprocessing violin plot started at %s...' % strgtimestmp)
    
    liststrgruns = ['woutTESS', 'alldata']
    if boolonlytess:
        liststrgruns.append(['TESS'])
    
    numbruns = len(liststrgruns)
    indxruns = np.arange(numbruns)
    
    liststrgpara = [[] for i in indxruns]
    listlablpara = [[] for i in indxruns]
    listobjtalle = [[] for i in indxruns]
    for i, strgruns in enumerate(liststrgruns):
        pathdata = pathdataoutp + 'allesfits/allesfit_%s' % strgruns
        print('Reading from %s...' % pathdata)
        config.init(pathdata)
        liststrgpara[i] = np.array(config.BASEMENT.fitkeys)
        listlablpara[i] = np.array(config.BASEMENT.fitlabels)
        # read the chain
        listobjtalle[i] = allesfitter.allesclass(pathdata)
        
    liststrgparaconc = np.concatenate(liststrgpara)
    liststrgparaconc = np.unique(liststrgparaconc)
    listlablparaconc = np.copy(liststrgparaconc)
    for k, strgparaconc in enumerate(liststrgparaconc):
        for i, strgruns in enumerate(liststrgruns):
            if strgparaconc in liststrgpara[i]:
                listlablparaconc[k] = listlablpara[i][np.where(liststrgpara[i] == strgparaconc)[0][0]]
    
    ticklabl = ['w/o TESS', 'w/ TESS']
    if boolonlytess:
        ticklabl.append(['only TESS'])

    xpos = 0.6 * (np.arange(numbruns) + 1.)
    for k, strgpara in enumerate(liststrgparaconc):
        booltemp = True
        for i in indxruns:
            if not strgpara in liststrgpara[i]:
                booltemp = False
        if not booltemp:
            continue
        
        figr, axis = plt.subplots(figsize=(5, 4))
        chanlist = []
        for i in indxruns:
            chanlist.append((listobjtalle[i].posterior_params[strgpara] - np.mean(listobjtalle[i].posterior_params[strgpara])) * 24. * 60.)
        axis.violinplot(chanlist, xpos, showmedians=True, showextrema=False)
        axis.set_xticks(xpos)
        axis.set_xticklabels(ticklabl)
        if strgpara == 'b_period':
            axis.set_ylabel('P [min]')
        else:
            axis.set_ylabel(listlablparaconc[k])
        plt.tight_layout()
        
        path = pathdataoutp + 'viol_%04d.svg' % (k)
        print('Writing to %s...' % path)
        figr.savefig(path)
        plt.close()
    
    listyear = [2021, 2023, 2025]
    numbyear = len(listyear)
    indxyear = np.arange(numbyear)
    timejwst = [[[] for i in indxruns] for k in indxyear]
    for k, year in enumerate(listyear):
        epocjwst = astropy.time.Time('%d-01-01 00:00:00' % year, format='iso').jd
        for i in indxruns:
            epoc = listobjtalle[i].posterior_params['b_epoch']
            peri = listobjtalle[i].posterior_params['b_period']
            indxtran = (epocjwst - epoc) / peri
            indxtran = np.mean(np.rint(indxtran))
            if indxtran.size != np.unique(indxtran).size:
                raise Exception('')

            timejwst[k][i] = epoc + peri * indxtran

            timejwst[k][i] -= np.mean(timejwst[k][i])
            timejwst[k][i] *= 24. * 60.
    
    listfigr = []
    listaxis = []

    ## temporal evolution
    figr, axis = plt.subplots(figsize=(5, 4))
    listfigr.append(figr)
    listaxis.append(axis)
    axis.violinplot([timejwst[k][1] for k in indxyear], listyear)
    axis.set_xlabel('Year')
    axis.set_ylabel('Transit time residual [min]')
    plt.tight_layout()
    path = pathdataoutp + 'jwsttime.svg'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    ## without/with/only TESS prediction comparison
    figr, axis = plt.subplots(figsize=(5, 4))
    listfigr.append(figr)
    listaxis.append(axis)
    axis.violinplot(timejwst[1], xpos, points=2000)
    axis.set_xticks(xpos)
    axis.set_xticklabels(ticklabl)
    axis.set_ylabel('Transit time residual in 2023 [min]')
    #axis.set_ylim([-300, 300])
    plt.tight_layout()
    path = pathdataoutp + 'jwstcomp.svg'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    return listfigr, listaxis

    # all parameter summary
    figr, axis = plt.subplots(figsize=(4, 3))
    chanlist = []
    axis.violinplot(chanlist, xpos, showmedians=True, showextrema=False)
    axis.set_xticks(valutick)
    axis.set_xticklabels(labltick)
    axis.set_ylabel(lablparatemp)
    plt.tight_layout()
    path = pathdataoutp + 'para_%s.pdf'
    print('Writing to %s...' % path)
    figr.savefig(path)
    plt.close()

    # plot p values
    ## threshold p value to conclude significant difference between posteriors with and without TESS
    if pvalthrs is None:
        pvalthrs = 1e-6
    
    lablparacomp = [[] for u in indxruns]
    for u in indxruns:
    
        lablparacomp[u] = list(set(lablpara[indxrunsfrst[u]]).intersection(lablpara[indxrunsseco[u]]))
    
        # post-processing
        ## calculate the KS test statistic between the posteriors
        numbparacomp = len(lablparacomp[u])
        pval = np.empty(numbparacomp)
        for j in range(numbparacomp):
            kosm, pval[j] = scipy.stats.ks_2samp([indxrunsfrst[u]][:, j], chan[indxrunsseco[u]][:, j])
            kosm, pval[j] = scipy.stats.ks_2samp(chan[indxrunsfrst[u]][:, j], chan[indxrunsseco[u]][:, j])
    
        ## find the list of parameters whose posterior with and without TESS are unlikely to be drawn from the same distribution
        indxparagood = np.where(pval < pvalthrs)[0]
        if indxparagood.size > 0:
    
            figr, axis = plt.subplots(figsize=(12, 5))
            indxparacomp = np.arange(numbparacomp)
            axis.plot(indxparacomp, pval, ls='', marker='o')
            axis.plot(indxparacomp[indxparagood], pval[indxparagood], ls='', marker='o', color='r')
            axis.set_yscale('log')
            axis.set_xticks(indxparacomp)
            axis.set_xticklabels(lablparacomp[u])
            if u == 0:
                axis.set_title('Posteriors with TESS vs. without TESS')
            if u == 1:
                axis.set_title('Posteriors without TESS vs. only TESS')
            if u == 2:
                axis.set_title('Posteriors with TESS vs. only TESS')
    
            axis.axhline(pvalthrs, ls='--', color='black', alpha=0.3)
            plt.tight_layout()
            path = pathdataoutp + 'kosm_com%d.pdf' % u
            print('Writing to %s...' % path)
            figr.savefig(path)
            plt.close()
    

    


