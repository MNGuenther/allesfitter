1. Setting things up
2. Running the fit

##################################################################
1. Settings things up
##################################################################
The file structure is as follows:
	datadir
		settings.csv
		params.csv
		tel1.csv
		tel2.csv
		...		

The telescope data set files (tel1.csv etc.) must be named exactly like the telesopes listed in settings.csv. This is because the code will automatically search for these files in the given directory.

Photometric instruments must contain three columns:
	time	flux	flux_err

RV instruments must contain three columns:
	time	rv	rv_err

The time unit has to be days. It does not matter if it is JD/HJD/BJD, but it must be homogenous between all data sets tel1.csv etc. and the parameters (epoch) in params.csv.

