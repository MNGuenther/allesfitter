==============================================================================
allesclass
==============================================================================

Tired of allesfitter's standard output? Want to make your own plots to match your color scheme or add fancy twists? Want to read out some posterior samples? Want to access all data and models in one place?

You can do all this (and much more) with the allesclass module! Examples below, simply replace 'allesfit' with whatever name you gave your directory; also change the instrument and flux/rv accordingly::


    #::: load the allesclass
    alles = allesfitter.allesclass('allesfit')



------------------------------------------------------------------------------
1) The allesclass plot function
------------------------------------------------------------------------------
The allesclass plot function let's you create individual plots in your desired formats. The returned figure and axes object gives you control to change the plots afterwards, e.g.::


    #------------------------------------------------------------------------------
    # Photometry
    #------------------------------------------------------------------------------
    #::: iterate over all plot styles
    for style in ['full', 'phase', 'phasezoom', 'phasezoom_occ', 'phase_variations']:
    
        #::: set up the figure
        fig, axes = plt.subplots(2, 1, figsize=(6,6), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
        fig.subplots_adjust(hspace=0)
    
        #::: alles.plot(...) data and model
        alles.plot('Leonardo','b',style,ax=axes[0])
        axes[0].set_title('Leonardo, '+style)
    
        #::: alles.plot(...) residuals
        alles.plot('Leonardo','b',style+'_residuals',ax=axes[1])
        axes[1].set_title('')
    
        fig.savefig('Leonardo_'+style+'.pdf', bbox_inches='tight')


    #------------------------------------------------------------------------------
    # RV
    #------------------------------------------------------------------------------
    #::: iterate over all plot styles
    for style in ['full', 'phase']:
    
        #::: set up the figure
        fig, axes = plt.subplots(2, 1, figsize=(6,6), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
        fig.subplots_adjust(hspace=0)
    
        #::: alles.plot(...) data and model
        alles.plot('Donatello','b',style,ax=axes[0])
        axes[0].set_title('Donatello, '+style)
    
        #::: alles.plot(...) residuals
        alles.plot('Donatello','b',style+'_residuals',ax=axes[1])
        axes[1].set_title('')
    
        fig.savefig('Donatello_'+style+'.pdf', bbox_inches='tight')

            

.. image:: _static/tutorials/10_allesclass/Leonardo_full.pdf
   :target: _static/tutorials/10_allesclass/Leonardo_full.pdf
   :align: center

.. image:: _static/tutorials/10_allesclass/Leonardo_phase.pdf
   :target: _static/tutorials/10_allesclass/Leonardo_phase.pdf
   :align: center

.. image:: _static/tutorials/10_allesclass/Leonardo_phasezoom.pdf
   :target: _static/tutorials/10_allesclass/Leonardo_phasezoom.pdf
   :align: center

.. image:: _static/tutorials/10_allesclass/Leonardo_phasezoom_occ.pdf
   :target: _static/tutorials/10_allesclass/Leonardo_phasezoom_occ.pdf
   :align: center

.. image:: _static/tutorials/10_allesclass/Leonardo_phase_variations.pdf
   :target: _static/tutorials/10_allesclass/Leonardo_phase_variations.pdf
   :align: center

.. image:: _static/tutorials/10_allesclass/Donatello_full.pdf
   :target: _static/tutorials/10_allesclass/Donatello_full.pdf
   :align: center

.. image:: _static/tutorials/10_allesclass/Donatello_phase.pdf
   :target: _static/tutorials/10_allesclass/Donatello_phase.pdf
   :align: center



------------------------------------------------------------------------------
2) Full control
------------------------------------------------------------------------------
Want even more control, or access the data directly? Go ahead, e.g.::

    #::: settings
    inst = 'TESS'
    key = 'flux'
    
    #::: load the time, flux, and flux_err
    time = alles.data[inst]['time']
    flux = alles.data[inst][key]
    flux_err = alles.data[inst]['err_scales_'+key] * alles.posterior_params_median['err_'+key+'_'+inst]
    
    #::: note that the error for RV instruments is calculated differently
    #rv_err = np.sqrt( alles.data[inst]['white_noise_'+key]**2 + alles.posterior_params_median['jitter_'+key+'_'+inst]**2 )

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