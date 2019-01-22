=======================================
Tutorials
=======================================


Settings things up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file structure is as follows:
- datadir
  * settings.csv
  * params.csv
  * tel1.csv
  * tel2.csv
  * ...		

#### datadir
For example, datadir = "users/johnwayne/tess_1b".

#### settings.csv
- copy settings.csv from the "examples" that come with allesfitter
- adjust as you need
(todo)

#### params.csv
- copy params.csv from the "examples" that come with allesfitter
- adjust as you need
(todo)

#### tel1.csv, tel2.csv, ...
All these telescope data files must be named exactly like the telesopes listed in settings.csv. So for example "TESS.csv", "HARPS.csv", and so on. The code will automatically search for these file names in the given directory.

Photometric instruments must contain three columns:
	time	flux	flux_err

RV instruments must contain three columns:
	time	rv	rv_err
  
The time unit has to be days, preferably BJD. It does not matter if it is JD/HJD/BJD, but it *must* be homogenous between all data sets tel1.csv etc. and the parameters (epoch) in params.csv.


Running the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- copy run.py from the "examples" that come with allesfitter
- adjust as you need
(todo)
