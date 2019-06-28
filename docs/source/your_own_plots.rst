==============================================================================
Allesclass
==============================================================================

Tired of allesfitter's standard plots? Want to match your color scheme, or add some fancy twists? Create your own plots (and much more) with the allesclass module! Example::

    alles = allesfitter.allesclass('allesfit')
    
    inst = 'TESS'
    key = 'flux'
    time = alles.data[inst]['time']
    flux = alles.data[inst][key]
    flux_err = alles.data[inst]['err_scales_'+key] * alles.posterior_params_median['err_'+key+'_'+inst]
    baseline = alles.get_posterior_median_baseline(inst, key)
    model = alles.get_posterior_median_model(inst, key)
    
    fig, axes = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    ax = axes[0]
    ax.errorbar(time, flux, flux_err, fmt='b.')
    ax.plot(time, 1.+baseline, 'g-', lw=2)
    ax.plot(time, model+baseline, 'r-', lw=2)
    
    ax = axes[1]
    ax.errorbar(time, flux-(model+baseline), flux_err, fmt='b.')
    ax.axhline(0, color='grey', linestyle='--')

(MORE EXAMPLES TBD)