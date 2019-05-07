==============================================================================
params.csv
==============================================================================

.. todo:: This page is still under construction

All fields explained in the following; replace [key] with 'flux'/'rv' and [inst] with the name of your instrument (e.g. 'TESS'/'HARPS'):

Columns explained
------------------------------------------------------------------------------
``name``

    The name of the parameter, often decorated with [key] and [inst]. See 'rows explained' for a full list of all possible parameters.

``value``

    The initial guess value. This is needed to make an initial guess plot, initiate the MCMC walkers, to shift the epoch (if `shift_epoch` in settings.csv is `True`), or to mask out out-of-transit photometry (if `fast_fit` in settings.csv is `True`).

``fit``

    Do you want to fit or freeze this parameter? This can have two values:

    - 0: You want to freeze this value to the one given in the column `value`.
    - 1: You want to sample/fit for this value.

``bounds``

    The bounds of the fit. This can have the following values:

    - `uniform a b`: a uniform prior ranging from a to b
    - `normal mu sigma`: a normal prior with mean mu and standard deviation sigma
    - `trunc_normal a b mu sigma`: a truncated normal prior with mean mu and standard deviation sigma, truncated to a range from a to b

``label``

    The labels of each parameter (for the output plots and tables).

``unit``

    The units of each parameter (for the output plots and tables).

``coupled_with``

    Optional column. For example, if you have two files `TESS1` and `TESS2`, and you want to couple their error scaling sampling, you can set `log_err_flux_TESS1` as usual, and leave all rows of `log_err_flux_TESS2` empty besides for its column `coupled_with`, in which you can then write `log_err_flux_TESS1`. *allesfitter* will not sample for the parameter of log_err_flux_TESS2, but instead copy the value of `log_err_flux_TESS1` at every sampling step. In summary, you will only sample for one parameter, and the other will mirror it.

``truth``

    Optional column. If given, you can give all the 'true values' of the parameters (where known), and they will be marked in the output corner plots and trace plots.



Rows explained
------------------------------------------------------------------------------

Astrophysical params per planet:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[tbd]


Limb darkening coefficients per instrument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[tbd]


Host star variability:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[tbd]


Errors (white noise scaling) per instrument:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[tbd]


Baselines per instrument:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(If you use the hybrid_* versions in settings.csv, the baseline parameters will be automatically optimized at every step in the sampling. If you use the sample_* options in settings.csv, they get sampled as every other paramter.)

``baseline_offset_[key]_[inst]``: 

    This is needed if you selected ``sample_offset`` in settings.csv.

``baseline_offset_[key]_[inst]`` and
``baseline_slope_[key]_[inst]``: 

    These are needed if you selected ``sample_linear`` in settings.csv. (Note if you want to physically interpret the slope result: to calculate the slope, the center point of the line fit per instrument will be put in the middle of the time stamps, and the time array will be normalized to 0 to 1.)

``baseline_gp_real_lna_[key]_[inst]`` and 
``baseline_gp_real_lnc_[key]_[inst]``: 
 
    These are needed if you selected ``sample_GP_real`` in settings.csv.

``baseline_gp_real_lna_[key]_[inst]``, 
``baseline_gp_real_lnb_[key]_[inst]``, 
``baseline_gp_real_lnc_[key]_[inst]`` and 
``baseline_gp_real_lnd_[key]_[inst]``: 

    These are needed if you selected ``sample_GP_complex`` in settings.csv.

``baseline_gp_matern32_lnsigma_[key]_[inst]`` and 
``baseline_gp_matern32_lnrho_[key]_[inst]``: 

    These are needed if you selected ``sample_GP_Matern32`` in settings.csv. *lnsigma* is the characteristic amplitude of the GP. *lnrho* is the characteristic length scale of the GP.

``baseline_gp_sho_lnS0_[key]_[inst]``, 
``baseline_gp_sho_lnQ_[key]_[inst]`` and 
``baseline_gp_sho_lnomega0_[key]_[inst]``: 

    These are needed if you selected ``sample_SHO_complex`` in settings.csv.

.. note:: For all sample_GP_* options, you can always optionally add ``baseline_gp_offset_[key]_[inst]``, in which case the GP mean is set to this parameter (rather than assumed to be 0). Note that the GP is constrained on the residuals (data-model), so typically the GP mean is 0; but, for example, if you need to sample for the systemic velocity of your RV data, you will need this.