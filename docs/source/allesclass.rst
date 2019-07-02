==============================================================================
allesclass
==============================================================================

Tired of allesfitter's standard plots? Want to match your color scheme, or add some fancy twists? Create your own plots (and much more) with the allesclass module! Example (replace 'allesfit' with whatever name you gave your directory; also change the instrument and flux/rv accordingly)::

    #::: load the allesclass
    alles = allesfitter.allesclass('allesfit')
    
    #::: settings
    inst = 'TESS'
    key = 'flux'
    
    #::: load the time, flux, and flux_err
    time = alles.data[inst]['time']
    flux = alles.data[inst][key]
    flux_err = alles.data[inst]['err_scales_'+key] * alles.posterior_params_median['err_'+key+'_'+inst]
    
    #::: set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    #::: top panel: plot the data and 20 curves from random posterior samples (evaluated on a fine time grid)
    ax = axes[0]
    ax.errorbar(time, flux, flux_err, fmt='b.')
    for i in range(20):
        time_fine = np.linspace(time[0], time[-1], 1000)
        model_fine, baseline_fine, _ = alles.get_one_posterior_curve_set(inst, key, xx=time_fine)
        ax.plot(time_fine, 1.+baseline_fine, 'g-', lw=2)
        ax.plot(time_fine, model_fine+baseline_fine, 'r-', lw=2)
    
    #::: bottom panel: plot the residuals; 
    #::: for that, subtract the "posterior median model" and "posterior median baseline" from the data (evaluated on the time stamps of the data)
    ax = axes[1]
    baseline = alles.get_posterior_median_baseline(inst, key)
    model = alles.get_posterior_median_model(inst, key)
    ax.errorbar(time, flux-(model+baseline), flux_err, fmt='b.')
    ax.axhline(0, color='grey', linestyle='--')

(MORE EXAMPLES TBD)