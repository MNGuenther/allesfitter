"""
Predict JWST transits for obtained posteriors

argument 1: name of the planet

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import os, sys
import pickle
from dynesty import utils as dyutils
from allesfitter import config
from astropy.time import Time
import matplotlib.pyplot as plt


# base data path for the TESS known planet project
pathdata = os.environ['KNWN_DATA_PATH'] + '/'
pathdatapost = os.environ['KNWN_DATA_PATH'] + '/postproc/'

# name of the exoplanet
strgplan = sys.argv[1]


pathdataplan = pathdata + '%s/allesfit_global/allesfit_onlytess_full/' % strgplan
config.init(pathdataplan)
fileobjt = open(pathdataplan + 'results/save_ns.pickle', 'rb')
objtrest = pickle.load(fileobjt)
weig = np.exp(objtrest['logwt'] - objtrest['logz'][-1])
chan = dyutils.resample_equal(objtrest.samples, weig)
    
listkeys = config.BASEMENT.fitkeys
listlabl = config.BASEMENT.fitlabels

# get period and epoch posteriors
for k, labl in enumerate(listlabl):
    if listkeys[k] == 'b_epoch':
        postepoc = chan[:, k]
    if listkeys[k] == 'b_period':
        postperi = chan[:, k]

# the time at which the transit is to be predicted
timejwst = '2022-01-01T00:00:00'


objttime = Time(timejwst, format='isot', scale='utc')

indxtranjwst = 1000#np.argmax(objttime.jd - 

# posterior of the predicted transit
postepocjwst = postepoc + indxtranjwst * postperi

# plot the posterior
figr, axis = plt.subplots()
axis.hist(postepocjwst)
path = pathdatapost + 'posttranjwst_%s.png' % strgplan
print 'Writing to %s...' % path
plt.savefig(path)
plt.close()



