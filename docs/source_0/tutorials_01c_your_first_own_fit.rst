==============================================================================
Tutorials 01c: Your first own allesfit
==============================================================================


Settings things up
------------------------------------------------------------------------------

The file structure is as follows:

- datadir

  * settings.csv
  * params.csv
  * tel1.csv
  * tel2.csv
  * ...		

`datadir`
For example, datadir = "users/johnwayne/tess_1b".

`tel1.csv`, `tel2.csv`, ...
All these telescope data files must be named exactly like the telesopes listed in settings.csv. So for example "TESS.csv", "HARPS.csv", and so on. The code will automatically search for these file names in the given directory.

Photometric instruments must contain three columns: time,flux,flux_err

RV instruments must contain three columns: time,rv,rv_err
  
The time unit has to be days, preferably BJD. It does not matter if it is JD/HJD/BJD, but it *must* be homogenous between all data sets tel1.csv etc. and the parameters (epoch) in params.csv.

`settings.csv` and `params.csv`
Either create these with the GUI, or copy them from the example and adjust as you need



Running the fit
------------------------------------------------------------------------------

Either run everything with the GUI, or copy `run.py` from the examples and  adjust as you need