=======================================
Tutorials 02: Transits
=======================================

See on `GitHub <https://github.com/MNGuenther/allesfitter/tree/master/tutorials/02_transits>`_.

The Magic Next Telescope (*TMNT*) is really taking off now! You collected more data on that transiting exoplanet candidate using *Leonardo*:

.. image:: _static/tutorials/02_transits/data.png
   :target: _static/tutorials/02_transits/data.png
   :align: center

You can download the data file here: :download:`Leonardo.csv <_static/tutorials/02_transits/Leonardo.csv>`

As an experienced allesfitter, the team asks you to model the signal. You see that the lightcurve has some red noise throughout, and some nasty scattered light on day 20-23, but that won't stop you. The discovery report gives you a first guess for the transit signal:

- Epoch: 1.09 +- 0.01 days after start of observations
- Period: 3.41 +- 0.01 days
- R_planet / R_star: 0.10 +- 0.01
- (R_star + R_planet) / semi-major axis: somewhere between 0.1 and 0.3

This time you also have received stellar parameters from Gaia DR2:

- R_star: 1.00+-0.01 R_sun
- M_star: 1.00+-0.01 M_sun
- T_eff: 5700+-100 K

(weird coincidence, right?)

Now, time to launch the GUI again and fill in those fields! Feel free to experiment with some settings this time. Maybe use a simpler, linear limb darkening model? Or use Nested Sampling instead of MCMC?

Once you successfully fitted this data set, move on to the next tutorial to learn how to use GPs to deal with red noise.



