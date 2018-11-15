"""
Pipeline to analyze previously known planets detected by TESS

argument 1: name of the planet
argument 2: kind of the analysis (global, occultation, etc.)
argument 3: type of the analysis (wouttess, withtess, onlytess)

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import os, datetime, sys

import matplotlib.pyplot as plt

from allesfitter import config
import pickle

# base data path for the TESS known planet project
pathdata = os.environ['KNWN_DATA_PATH'] + '/'

os.system('mkdir -p %s' % pathdata)
os.system('mkdir -p %s' % (pathdata + '/postproc/'))

strgplan = sys.argv[1]
strgkind = sys.argv[2]

strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print('Post-processing comparison pipeline for the previously TESS exoplanets started at %s...' % strgtimestmp)
    

print('Will analyze:')
print(strgplan)

# threshold p value to conclude significant difference between posteriors with and without TESS
pvalthrs = 1e-6

indxcomp = np.arange(3)

# list of rtyp
# 0 : wout
# 1 : with
# 2 : only
# list of comp
# wout with
# wout only
# with only
indxrtypfrst = [0, 0, 1]
indxrtypseco = [1, 2, 2]

liststrgrtyp = ['wout', 'with', 'only'] 
numbrtyp = len(liststrgrtyp)
indxrtyp = np.arange(numbrtyp)

lpos = [[] for i in indxrtyp]
chan = [[] for i in indxrtyp]
lablpara = [[] for i in indxrtyp]
numbwalk = [[] for i in indxrtyp]
numbswep = [[] for i in indxrtyp]
numbsamp = [[] for i in indxrtyp]
numbburn = [[] for i in indxrtyp]
factthin = [[] for i in indxrtyp]
numbpara = [[] for i in indxrtyp]

pathdataplan = pathdata + strgplan+ '/'

for i, strgrtyp in enumerate(liststrgrtyp):
    pathdata = pathdataplan + 'allesfit_%s/allesfit_%stess_ns/' % (strgkind, strgrtyp)
    print('Reading from %s...' % pathdata)
    config.init(pathdata)
    lablpara[i] = config.BASEMENT.fitlabels

    pathsave = pathdata + 'results/mcmc_save.h5'
    if False and not os.path.exists(pathsave):
        # sample from the posterior excluding the TESS data
        print('Calling allesfitter to fit the data...')
        allesfitter.mcmc_fit(pathdata)

    # read the chain
    ## MCMC
    #emceobjt = emcee.backends.HDFBackend(pathsave, read_only=True)
    #chan[i] = emceobjt.get_chain()
    #lpos[i] = emceobjt.get_log_prob()
    
    ## Nested sampling
    fileobjt = open(pathdata + 'results/save_ns.pickle', 'rb')
    objtrest = pickle.load(fileobjt)
    weig = np.exp(objtrest['logwt'] - objtrest['logz'][-1])
    chan[i] = dyutils.resample_equal(objtrest.samples, weig)    
    lpos[i] = np.zeros(chan[i].shape[0])

    # parse configuration
    numbswep[i] = config.BASEMENT.settings['mcmc_total_steps']
    numbburn[i] = config.BASEMENT.settings['mcmc_burn_steps']
    factthin[i] = config.BASEMENT.settings['mcmc_thin_by']
    numbpara[i] = config.BASEMENT.ndim

    numbwalk[i] = chan[i].shape[1]
    numbswep[i] = chan[i].shape[0] * factthin
    numbsamp[i] = lpos[i].size
    chan[i] = chan[i].reshape((-1, numbpara[i]))
    print('Found %d samples and %d parameters...' % (chan[i].shape[0], chan[i].shape[1]))

lablparacomp = [[] for u in indxcomp]
for u in indxcomp:

    lablparacomp[u] = list(set(lablpara[indxrtypfrst[u]]).intersection(lablpara[indxrtypseco[u]]))

    # post-processing
    ## calculate the KS test statistic between the posteriors
    numbparacomp = len(lablparacomp[u])
    pval = np.empty(numbparacomp)
    for j in range(numbparacomp):
        kosm, pval[j] = stats.ks_2samp(chan[indxrtypfrst[u]][:, j], chan[indxrtypseco[u]][:, j])

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
        path = pathdata + 'postproc/%s_kosm_com%d_%s.png' % (strgtimestmp, u, strgplan)
        print('Writing to %s...' % path)
        figr.savefig(path)
        plt.close()

indxrtypmaxm = 1

lablparatotl = np.unique(np.concatenate(lablpara))

for l, lablparatemp in enumerate(lablparatotl):
    figr, axis = plt.subplots()
    numbbins = 50

    listindxrtypcomp = []
    for i in indxrtyp:
        if lablparatemp in lablpara[i]:
            listindxrtypcomp.append(i)

    figr, axis = plt.subplots()
    chanlist = []
    for i in listindxrtypcomp:
        m = np.where(lablpara[i] == lablparatemp)[0][0]

        if lablparatemp == '$T_{0;b}$':
            offs = np.mean(chan[i][:, m])
            chantemp = chan[i][:, m] - offs
            print('Subtracting %g from T_b for TESS...' % offs)
        else:
            chantemp = chan[i][:, m]

        chanlist.append(chantemp)
    axis.violinplot(chanlist, showmedians=True, showextrema=False)

    if listindxrtypcomp == [0, 1]:
        labltick = ['w/o TESS', 'w/ TESS']
    if listindxrtypcomp == [0, 2]:
        labltick = ['w/o TESS', 'only TESS']
    if listindxrtypcomp == [1, 2]:
        labltick = ['w/ TESS', 'only TESS']
    if listindxrtypcomp == [0, 1, 2]:
        labltick = ['w/o TESS', 'w/ TESS', 'only TESS']
    valutick = np.arange(len(labltick)) + 1

    axis.set_xticks(valutick)
    axis.set_xticklabels(labltick)
    axis.set_ylabel(lablparatemp)
    plt.tight_layout()

    path = pathdata + 'postproc/%s_viol_pr%02d_%s.png' % (strgtimestmp, l, strgplan)
    print('Writing to %s...' % path)
    figr.savefig(path)
    plt.close()

# summary violin plot 
figr, axis = plt.subplots()
chanlist = []
for l, lablparatemp in enumerate(['']):
    chanlist.append()
axis.violinplot(chanlist, showmedians=True, showextrema=False)

axis.set_xticks(valutick)
axis.set_xticklabels(labltick)
axis.set_ylabel(lablparatemp)
plt.tight_layout()

path = pathdata + 'postproc/%s_viol_pr%02d_%s.png' % (strgtimestmp, l, strgplan)
print('Writing to %s...' % path)
figr.savefig(path)
plt.close()


