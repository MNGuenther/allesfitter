"""
Utility functions for post processing

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

def demo_kosm(gdat):

    '''
    Demonstrates Kolmogorov-Smirnov test statistics
    '''

    path = pathdata + 'plt.png'

    if not os.path.exists(path):
        numbsamp = 1000
        numbpara = 4
        indxpara = np.arange(numbpara)

        shft = 1.5 * np.random.randn(numbpara)
        shrk = 0.5 * np.random.randn(numbpara) + 1.

        chan = np.random.randn(numbpara * numbsamp).reshape((numbsamp, numbpara))
        for k in indxpara:
            chan[:, k] = shft[k] + np.random.randn(numbsamp) / shrk[k]

        strgtitl = ''
        figr, axis = plt.subplots()
        for k in indxpara:
            plt.hist(chan[:, k], alpha=0.4, lw=10, histtype='stepfilled')
            if k != 0:
                kosm, pval = stats.ks_2samp(chan[:, 0], chan[:, k])
                strgtitl += ' %.5g' % pval
        axis.set_title(strgtitl)
        plt.tight_layout()
        figr.savefig(path)
        plt.close()


def retr_listexop(gdat):

    pass

    # temp
    #coords = SkyCoord(['124.532 -68.313', '42.42, -42.42'], unit='deg')
    #df = gts.get_time_on_silicon(coords)
    #print 'Time on silicon'
    #print 'df'
    #print df



