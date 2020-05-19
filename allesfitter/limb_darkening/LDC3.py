# ==============================================================================
# =============================     LDC3 v1.0     ==============================
# ==============================================================================
#
# AUTHOR: David Kipping
#         Columbia University, Dept. of Astronomy
#         Please report any problems to: d.kipping@columbia.edu
#
# CITATION: If using this code, please cite:
#           Kipping, D. M., 2015, 'Efficient, uninformative sampling of limb 
#           darkening coefficients for a three-parameter law', MNRAS, accepted
#
# DESCRIPTION: LDC3 is a module containing three subroutines: "forward", 
#              "inverse" and "criteriatest". These subroutines perform tasks
#              related to identifying physically plausible limb darkening 
#              coefficients (LDCs) in the case of the 3-parameter limb
#              darkening law proposed by Sing et al. (2009), A&A, 505, 891:
#              I(mu)/I(1) = 1 - c_2*[1-mu] - c_3*[1-mu^(3/2)] - c_4*[1-mu^2].
#              The expressions used in LDC3 were derived in Kipping (2015),
#              and we recommend the reader review this paper before continuing.
#              See each subroutine for a description of its purpose. Note that
#              alpha_t denotes alpha_theta throughout this code.
#
# HISTORY: v1.0 Initial version released

import math

#===============================================================================
def forward(alphas):

  # DESCRIPTION: forward takes a set of LDCs in the alpha-parameterization and
  #              converts them to the c-parameterization. A likely common use
  #              of this subroutine would be when fitting LDCs. One would treat
  #              the alpha-parameters as the free parameters, but then convert 
  #              them to the more standard c-parameters before calling whatever 
  #              astronomical model describes one's observations.
  #
  # INPUTS: alphas=[alpha_h, alpha_r, alpha_t], all of which are bound between 0 
  #         and 1 and the latter is a wrap-around parameter.
  #
  # OUTPUTS: c = [c_2, c_3, c_4]

  # Forward modeling constants
  third	= 0.333333333333333
  twopi	= 6.283185307179586
  P1	= 4.500841772313891
  P2	= 17.14213562373095
  Q1	= 7.996825477806030
  Q2	= 8.566161603278331

  # Define alphas
  alpha_h = alphas[0]
  alpha_r = alphas[1]
  alpha_t = alphas[2]

  # Compute LDCs
  c_2 = (alpha_h**third)*( P1 + 0.25*math.sqrt(alpha_r)*( \
        -6.0*math.cos(twopi*alpha_t) + P2*math.sin(twopi*alpha_t) ) )
  c_3 = (alpha_h**third)*( -Q1 - Q2*math.sqrt(alpha_r)*math.sin(twopi*alpha_t) )
  c_4 = (alpha_h**third)*( P1 + 0.25*math.sqrt(alpha_r)*( \
        6.0*math.cos(twopi*alpha_t) + P2*math.sin(twopi*alpha_t) ) )

  # Define c results
  c=[0 for i in range(3)]
  c[0] = c_2
  c[1] = c_3
  c[2] = c_4
  return c
#===============================================================================

#===============================================================================
def inverse(c):

  # DESCRIPTION: inverse performs the reverse transformation of subroutine 
  #              forward. It therefore converts c-parameters into 
  #              alpha-parameters.
  #
  # INPUTS: c = [c_2, c_3, c_4] (= LDCs)
  #
  # OUTPUTS: alphas = [alpha_h, alpha_r, alpha_t]

  # Inverse modeling constants
  third	= 0.3333333333333333
  twopi	= 6.2831853071795865
  F1	= 0.9997221357486548
  F2	= 1.0002947195445628
  G1	= 51.396969619669996
  G2	= 51.426406871192850
  G3	= 0.5098605589396862
  G4	= 67.195959492893320
  G5	= 75.639610306789280
  H1	= 0.4666386075895370
  H2	= 0.5252750715749255

  # Define c
  c_2 = c[0]
  c_3 = c[1]
  c_4 = c[2]

  # Compute inverse LDCs
  alpha_h = ( F1*c_2 + F2*c_3 + F1*c_4 )**3
  alpha_r = G3/( G1*c_2 + G2*c_3 + G1*c_4 )**2
  alpha_r = alpha_r*( 576.0*(c_2 - c_4)**2 + ( G4*c_2 + G5*c_3 + G4*c_4 )**2 )
  alpha_t = math.atan2( -H1*c_2 - H2*c_3 - H1*c_4 , third*0.5*(c_4-c_2) )
  alpha_t = ( alpha_t - twopi*math.floor( alpha_t/twopi ) )/twopi

  # Define alphas_inv results
  alphas_inv=[0 for i in range(3)]
  alphas_inv[0] = alpha_h
  alphas_inv[1] = alpha_r
  alphas_inv[2] = alpha_t
  return alphas_inv
#===============================================================================

#===============================================================================
def criteriatest(usemod,c):

  # DESCRIPTION: criteriatest tests whether a set of LDCs (with the 
  #              c-parameterization) satisfy the seven analytic criteria defined
  #              in Kipping (2015). Passing this test indicates that LDCs
  #              correspond to a physically allowed limb darkened intensity
  #              profile, as defined in Kipping (2015). One may choose to use
  #              modified or unmodified versions of the criteria with the usemod
  #              logical control. The modified versions are slightly more
  #              conservative, cropping ~5% of the allowed parameter volume, but
  #              yielding a more symmetric volume.
  #
  # INPUTS: usemod (integer; 1 = use modified criteria; 0 = use unmodified 
  #                          criteria)
  #         c = [c_2, c_3, c_4] (= LDCs)
  #
  # OUTPUTS: passed (integer; 1 = test passed, 0 = test failed)

  # Define c
  c_2 = c[0]
  c_3 = c[1]
  c_4 = c[2]
  passed = 1

  # Criteria A
  if (c_2+c_3+c_4)>1.0:
    passed = 0

  # Criteria B
  if (2.0*c_2+3.0*c_3+4.0*c_4)<0.0:
    passed = 0

  # Criteria C
  if c_2<0.0:
    passed = 0

  # Criteria D
  if (c_2+c_3+c_4)<0.0:
    passed = 0

  # Criteria E
  if c_3>0.0:
    passed = 0

  # Criteria F
  if usemod==1:
    # Modified Criterion F
    if c_4<0.0:
      passed = 0
  else:
    # Unmodified Criterion F
    if c_4<-1.0:
      passed = 0

  # Criteria G
  if usemod==1:
    # Modified Criterion G
    if (32.0*c_2*c_4)<(9.0*c_3*c_3):
      passed = 0
  else:
    # Unmodified Criterion G
    if (0.375*c_3/c_4)<=0.0 and (0.375*c_3/c_4)>=-1.0:
      if (32.0*c_2*c_4)<(9.0*c_3*c_3):
        passed = 0

  # Return the result
  return passed
#===============================================================================