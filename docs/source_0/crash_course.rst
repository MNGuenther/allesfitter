==============================================================================
Crash course
==============================================================================

See on `GitHub <https://github.com/MNGuenther/allesfitter/tree/master/tutorials/01_crash_course>`_.

Your team from The Magic Next Telescope (*TMNT*) found a transiting exoplanet candidate using its photometric telescope *Leonardo*:

.. image:: _static/tutorials/01_crash_course/data.png
   :target: _static/tutorials/01_crash_course/data.png
   :align: center

You can download the data file here: :download:`Leonardo.csv <_static/tutorials/01_crash_course/Leonardo.csv>`

Now it is your job to model it! The discovery report gives you a first guess for the transit signal:

- Epoch: 1.09 +- 0.01 days after start of observations
- Period: 3.41 +- 0.01 days
- R_planet / R_star: 0.10 +- 0.01
- (R_star + R_planet) / semi-major axis: somewhere between 0.1 and 0.3

(Your team is still unsure about the stellar parameters, so let us leave these undefined for now.)

Simply launch *allesfitter*'s graphical user interface (GUI) via double click on the :download:`launch_allesfitter <_static/launch_allesfitter>` app (for Mac/Windows/Linux) *or* via executing the following lines in a Python console::

    import allesfitter
    allesfitter.GUI()

Now fill out the fields step by step, hit the run button, and lean back. All of this is demonstrated in this video tutorial:

.. raw:: html

   <div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/5LIci8gZZ_8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>

.. note:: The video is a bit outdated by now, but still captures 99% of the workflow. The updated crash course on GitHub simplifies and speeds up the modeling.

Congratulations! Now that you successfully modeled the data, you can schedule follow-up observations with the rest of the TMNT network: *Michelangelo* (photometry), *Donatello* (RV) and *Raphael* (RV). 

Now move on to the more advanced tutorials, which step by step introduce GPs, RV modeling, and much more.
