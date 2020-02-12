==============================================================================
settings.csv
==============================================================================

All fields explained in the following; replace [key] with `flux`/`rv` and [inst] with the  name of your instrument (e.g. `TESS`/`HARPS`):

General settings
------------------------------------------------------------------------------
``companions_phot``

The companion(s) to the host star covered in the photometric data, space separated if there are multiple. For one planet for example `b`; for multiple planets for example `b c d`. For a binary companion for example `B`. Could also be `Hans` or anything really, if you feel rebellious. (string)

``companions_rv``

The companion(s) to the host star covered in the radial velocity data, space separated if there are multiple. For one planet for example `b`; for multiple planets for example `b c d`. For a binary companion for example `B`. Could also be `Hans` or anything really, if you feel rebellious. (string)

``inst_phot``

The name of the photometric instrument(s), space separated if there are multiple. For example `TESS`. The only condition is that the data file must be named exactly the same, for example `TESS.csv`. (string)

``inst_rv``

The name of the radial velocity instrument(s), space separated if there are multiple. For example `HARPS ESPRESSO`. The only condition is that the data file must be named exactly the same, for example `HARPS.csv` and `ESPRESSO.csv`. (string)



Fit performance settings
------------------------------------------------------------------------------
``multiprocess``

Do you want to run on multiple cores (`True` or `False`)

``multiprocess_cores``

On how many cores do you want to run (integer)

``fast_fit``

Do you want to mask out the out-of-transit data, and run a fast fit on the transits alone (`True` or `False`)

``fast_fit_width``

How big of a window around the transit midpoint do you want to keep (in days). For example , the default `0.33333` keeps a 0.33333 day window around the transit midpoint. Careful: if your initial epoch and period values are off by too much, this will cut out the wrong parts of the data! (float)

``secondary_eclipse``

When using fast fit, do you also want to keep a window centred around phase 0.5? For example, to keep information of the secondary eclipse in the light curve. (`True` or `False`)  

``phase_curve``

This is just for plotting, really. Do you want to plot zooms onto the phase curve? (`True` or `False`)

``shift_epoch``

Do you want to automatically put the epoch into the middle of the data set? This is to avoid correlations between epoch and period, which would occur if the epoch is set far off one edge of the data set. (`True` or `False`)

``inst_for_b_epoch``

Which data sets do you want to regard for shifting the epoch? For example, you might have 10 year old, sparse archival radial velocity data, and a brand-new TESS light curve. In this case, it is best to just pass `TESS`, because this instrument is what really pins down the epoch and period. In other cases you might want to pass `all`. (string)


MCMC settings
------------------------------------------------------------------------------
``mcmc_nwalkers``

The number of MCMC walkers (integer)

``mcmc_total_steps``

The total steps in the MCMC chain, including burn-in steps. (integer)

``mcmc_burn_steps``

The burn-in steps in the MCMC chain; obviously must be smaller than the total steps.(integer)

``mcmc_thin_by``

Only save every n-th step in the MCMC chain (integer)


Nested Sampling settings
------------------------------------------------------------------------------
``ns_modus``

Which Nested Sampling algorithm do you want to pick? Default is `dynamic`, alternative is `static`.

``ns_nlive``

How many live points to set at the beginning of a Nested Sampling run? (integer)

``ns_bound``

Use a `single` ellipse, or `multiple` ellipses? (string)

`ns_sample`

Which sampling method to use to update the live points? `rwalk` or others (string)

``ns_tol``

The tolerance of the convergence criterion. (float)


Limb darkening law per object and instrument
------------------------------------------------------------------------------
``host_ld_law_[inst]``

If 'lin' one corresponding parameter called 'ldc_q1_inst' has to be given in params.csv.
If 'quad' two corresponding parameter called 'ldc_q1_inst' and 'ldc_q2_inst' have to be given in params.csv,
If 'sing' three corresponding parameter called 'ldc_q1_inst'; 'ldc_q2_inst' and 'ldc_q3_inst' have to be given in params.csv,


Baseline settings per instrument
------------------------------------------------------------------------------
``baseline_[key]_[inst]``

The baseline / detrending method used per instrument, choose between: 

- `sample_offset`: sample for a simple, constant offset. One corresponding parameter called 'baseline_offset_[key]_[inst]' has to be given in params.csv.
- `sample_linear`: sample for a linear slope. Two corresponding parameters called 'baseline_offset_[key]_[inst]' and 'baseline_slope_[key]_[inst]' have to be given in params.csv.
- `sample_GP_real`:
- `sample_GP_complex`:
- `sample_GP_Matern32`: sample for the parameters of a Gaussian Process with a Matern 3/2 kernel. At least two corresponding parameters called 'baseline_gp_matern32_lnsigma_[key]_[inst]' (the characteristic amplitude of the GP) and 'baseline_gp_matern32_lnrho_[key]_[inst' (the characteristic length scale of the GP) have to be given in params.csv.
- `sample_GP_SHO`:
- `hybrid_offset`:
- `hybrid_poly_1`:
- `hybrid_poly_2`: 
- `hybrid_poly_3`:
- `hybrid_poly_4`:
- `hybrid_spline`:
- `hybrid_GP`: do not use this.


Error settings per instrument
------------------------------------------------------------------------------
``error_flux_[inst]``

The white noise scaling per instrument: sample / hybrid.
If 'sample' one corresponding parameter called 'log_err_key_inst' (photometry) or 'log_jitter_key_inst' (RV) has to be given in params.csv.


Exposure times for interpolation
------------------------------------------------------------------------------
``t_exp_[inst]``

Need to be in the same units as the time series. If not given the observing times will not be interpolated leading to biased results


Number of points for exposure interpolation
------------------------------------------------------------------------------
``t_exp_n_int_[inst]``

Sample as fine as possible; generally at least with a 2 min sampling for photometry, n_int=5 was found to be a good number of interpolation points for any short photometric cadence t_exp; increase to at least n_int=10 for 30 min phot. cadence the impact on RV is not as drastic and generally n_int=5 is fine enough


Number of spots per object and instrument
------------------------------------------------------------------------------
``host_N_spots_[inst]``

(integer)


Number of flares (in total)
------------------------------------------------------------------------------
``N_flares``

(integer)


TTVs
------------------------------------------------------------------------------
``fit_ttvs``
(`True` or `False`)


Stellar grid per object and instrument
------------------------------------------------------------------------------
``host_grid_[inst]``
`default` / `sparse` / ... (string)

``b_grid_[inst]``
`default` / `sparse` / ... (string)


Stellar shape per object and instrument
------------------------------------------------------------------------------
``host_shape_[inst]``
`sphere` / `roche` / ... (string)

``b_shape_[inst]``
`sphere` / `roche` / ... (string)


Flux weighted RVs per object and instrument
------------------------------------------------------------------------------
`Yes` for Rossiter-McLaughlin effect