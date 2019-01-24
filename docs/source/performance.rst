=======================================
Performance & timing
=======================================


Test 1
---------------------------------------
Fitting the Leonardo discovery photometry using Dynamic Nested Sampling, once with uniform sampling, once with random-walk sampling::

    ns_modus,dynamic
    ns_nlive,500
    ns_bound,single
    ns_sample,unif vs. rwalk
    ns_tol,0.01
    
    ndim = 8

    allesfit_Leonardo_unif/: 
	    75 minutes, 27k samples, logZ = 545.42 +- 0.14
    allesfit_Leonardo_rwalk/: 
	    31 minutes, 24k samples, logZ = 545.36 +- 0.14


Test 2
---------------------------------------
Fitting all TMNT data together using Dynamic Nested Sampling, once with uniform sampling, once with random-walk sampling::

    ns_modus,dynamic
    ns_nlive,500
    ns_bound,single
    ns_sample,(compared below)
    ns_tol,0.01
    
    ndim = 8

    allesfit_all_TMNT_unif/: 
	    > 24h, aborted 
    allesfit_all_TMNT_rwalk/: 
	    1.3h, 35111 samples, logZ = 1234.26 +- 0.23
    allesfit_all_TMNT_slice/: 
	    > 24h, aborted 
    allesfit_all_TMNT_rslice/: 
	    1h, 28074 samples, logZ = 1233.66 +- 0.23
    allesfit_all_TMNT_hslice/: 
	    16.4h, 30925 samples, logZ = 1231.49 +- 0.23


Summary:

- `unif` does not converge within a reasonable run time for higher dimensions. It seems not applicable to our exoplanet scenarios.
- `rwalk` runs fast, and finds the true solution. It shows quite "conservative" (large) posterior errorbars. This seems to be the best choice for exoplanet modelling.
- `slice` is theoretically the most robust, but does not converge within a reasonable run time for higher dimensions.
- `rslice` runs fast, but gives somewhat funky posteriors / traceplots. It seems to be overly confident while missing the true solution.
- `hslice` is very slow and gives somewhat funky posteriors / traceplots.