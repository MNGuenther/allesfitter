Gravity darkening coefficient 
-----------------------------
from Claret 2017, Table 29, row:
0.10  2.000   4.50  3.813    0.2260 A
--> gdc = 0.2260


Limb darkening coefficients, linear
--------------------------------------
from Claret 2017, Table 24, 
http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/A%2bA/600/A30/table24
4.50 	6500 	0.1 	2.0 	0.4525 	0.3861 	0.3624
--> ldc_u = 0.4525


Limb darkening coefficients, quadratic
--------------------------------------
from Claret 2017, Table 25, row:
7734 	4.50 	6500 	0.1 	2.0 	0.2192 	0.3127 	0.2487 	0.2749 	0.1103 A
--> ldc_u = [0.2192, 0.3127] #in (u1,u2) space
--> lcd_q = [., .] #in Kipping2013 space


allesfit_orbit
----------------
phase-folded and binned to 1000 points
fixed K, f_c, f_s and q
fix limb darkening and gravity darkening to Claret 2017
used nested sampling