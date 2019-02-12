==============================================================================
Installation
==============================================================================


Installing allesfitter
------------------------------------------------------------------------------

pip install allesfitter
 


Requirements
------------------------------------------------------------------------------

Standard packages (install via conda/pip; anaconda is recommended):

- python (>=2.7 or >=3.5)
- numpy
- scikit-learn 
- matplotlib
- seaborn
- tqdm

Special packages (install via pip):

- ellc (>=1.8.0) [#f1]_
- dynesty (>=0.9.3) [#f2]_ 
- emcee (>=3.0rc2) [#f3]_
- celerite (>=0.3.0) [#f4]_
- corner (>=2.0.1) [#f5]_
- rebound (>=3.8.0) [#f6]_



.. rubric:: Footnotes

.. [#f1] requires a Fortran compiler; on Mac, you can install one e.g. via ``brew install gcc``
.. [#f2] optional; for Nested Sampling
.. [#f3] optional; for MCMC
.. [#f4] optional; for Gaussian Processes
.. [#f5] optional; if you want corner plots (you know you do.)
.. [#f6] optional; if you want top-down-view plots of the orbits

