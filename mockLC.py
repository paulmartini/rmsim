'''
Written in Python 2.7.14 by Natalia Eire Sommer

Code for creating mock AGN light-curves. 

This module contains two classes; 'LightCurve' and 'create'. The 
general user will find 'create' to be a useful class for mock AGN 
light-curve generation, and will not need to interact directly with
'LightCurve'. 

Detailed instructions on how to use 'LightCurve' and 'create' are 
provided in docstring of each class.
'''

import os
import numpy as np
import scipy as sp
import scipy.integrate as igrt
import astropy.units as u
import time
from astropy.io import fits
from astropy.io import ascii
from astropy.cosmology import WMAP7
import pandas as pd



class LightCurve:
	'''
	Object for creating underlying/emitted light-curves for AGN
	reverberation mapping based on a damped random walk model for the
	continuum light-curve, and a smoothing and shifting of the continuum
	light-curve using a transfer function for the emission-line
	light-curve.

	Contents
	========

	Constants
	---------
	- self.length -- number of elements in underlying continuum
	  light-curve
	- self.tMax -- maximum number of days we will be using to define a
	  transfer function, 2200. Using a large number to ensure transfer
	  function is defined for wide transfer functions and long time-lags
	- self.dt -- stepsize, number of days between each element in
	  light-curve arrays, 1 day
	- self.tfLen -- number of elements used to define transfer function

	Functions for the general user
	------------------------------
	- genContMag -- generates continuum light-curve in terms of apparent
	  magnitude, as well as a corresponding array of times, based on
	  damped random walk and MacLeod+10 [1]
	- genCont -- converts continuum light-curve from apparent magnitude
	  to flux, normalising the flux to have an average flux value of 10,
	  with no units specified
	- genLineTH -- generates emission-line light-curve using a top-hat
	  transfer function centered at provided lag, with width of 10% of
	  the lag value
	- genLineGauss -- generates emission-line light-curve using a
	  Gaussian distribution transfer function centered at provided lag,
	  with standard deviation of 10% of the lag value
	- genLineGamma -- generates emission-line light-curve using a gamma
	  distribution transfer function, and shape/alpha value provided in
	  input
	- genLineRing -- generates emission-line light-curve assuming an
	  infinitesimally thin broad-line region ring at the distance
	  corresponding to the lag value, with the angle of inclination of
	  the ring relative to the observer being specified by input
	- genLineSphere -- generates emission-line light-curve assuming an
	  infinitesimally thin broad-line region sphere at the distance
	  corresponding to the lag value
	- genLineSkewedGaussian -- generates emission-line light-curve
	  using a skewed Gaussian distribution defined by the location of
	  the lag value, a scale of 10% of the lag value, and alpha=10

	Initial setup
	=============
	Start the class by creating an object, and providing it with a
	the number of elements for the underlying continuum light-curve,
	e.g.:

	>>> LC = LightCurve(5000)

	Creating light-curves
	=====================
	It is necessary to start the creation of light-curves by defining
	them in terms of apparent magnitude, despite flux being the natural
	unit of measurement. Create a continuum light-curve modelled by a
	damped random walk based on damped random walk parameters tau and
	SFinfty, as well as the observed apparent magnitude of the object,
	and an integer (to ensure different light-curve realisations if
	running process in parallel). Follow by converting continuum
	light-curve from apparent magnitude to flux, e.g.:

	>>> apparent_magnitude = 20.
	>>> SFinfty, tau = 0.2, 500.
	>>> lc = 0
	>>> LC.genContMag(apparent_magnitude, tau, SFinfty, lc)
	>>> LC.genCont()

	To obtain emission-line light-curve, convolve continuum
	light-curve with transfer function. Different functions provide
	a different transfer function to perform the convolution with,
	however, they are all dependent on the lag of the AGN, e.g.:

	>>> AGNlag = 365
	>>> LC.genLineTH(AGNlag)

	Access continuum and emission-line light-curves as properties of the
	class, e.g.:

	>>> photometry = LC.cont
	>>> spectroscopy = LC.line

	References
	==========
	[1] MacLeod C. L. et al., 2010, ApJ, 721, 1014

	'''

	# Initialisations
	def __init__(self,length):
		self.length = length
		self.tMax  = 2200.0
		self.dt    = 1.0
		self.tfLen = int(self.tMax/self.dt)
	# Ending initialisation


	def genContMag(self, mu, tau, sf, i):
		# Based on eqn. 5 in MacLeod et al. 2010

		# Predefining variables
		exponential  = np.exp(-self.dt/tau)
		norm = mu * (1.0 - exponential)
		var  = 0.5 * sf**2 * (1.0-exponential**2)
		std  = np.sqrt(var)

		# Defining vectors
		np.random.seed(int(time.time()) + i)
		self.t = np.arange(0, self.dt*self.length, self.dt)     # Defining t-vector
		self.contMag    = np.zeros(self.length)                 # Initialising contMag-vector
		self.contMag[0] = np.random.normal(mu, std)             # Drawing a random number for the first element
		for i in range(1, self.contMag.size):
			E = exponential * self.contMag[i - 1] + norm
			self.contMag[i] = np.random.normal(E, std)            # Defining elements of contMag-vector
		# End of for-loop
	# End of getContMag-function


	def genCont(self,pivot = 4812):
		
		self.cont = (3E18/(pivot**2))*10**(-0.4*(self.contMag+48.6))/10**-17       # Defining scaled cont vector
	# End of getCont-function

	def genLineTH(self, lag, scale):
		width = 0.1*lag                                 # Defining width of TH funct

		# Defining top hat transfer function vector
		tf = np.zeros(self.tfLen)
		for i in range(0, self.tfLen):
			if (i*self.dt >= lag - width and i*self.dt <= lag + width):
				tf[i] = scale/(2*width)
			# End if-statement
		# End for-loop

		# Convolving transfer funct w/cont light curve to get emission line light curve
		self.line = np.convolve(tf, self.cont)
	# End genLine-function


	def genLineGauss(self, lag):
		std = 0.1*lag                                   # Defining std of dist

		# Defining Gaussian transfer function vector
		tf = np.zeros(self.tfLen)
		for i in range(0, self.tfLen):
			tf[i] = 1.0/(np.sqrt(2*np.pi) * std) * np.exp( -(i*self.dt-lag)**2 / (2*std**2) )
		# End for-loop

		# Convolving transfer funct w/cont light curve to get emission line light curve
		self.line = np.convolve(tf, self.cont)
	# End genLineGauss-function


	def genLineGamma(self, lag, shape):
		# Predefining variables for Gamma distribution
		alpha = shape
		beta  = alpha/lag

		# Defining Gamma distribution transfer function
		tf = np.zeros(self.tfLen)
		for i in range(0, self.tfLen):
			tf[i] = beta**alpha / sp.special.gamma(alpha) * (i*self.dt)**(alpha-1) * np.exp(-beta*(i*self.dt))
			# End for-loop

		# Convolving transfer funct w/cont light curve to get emission line light curve
		self.line = np.convolve(tf, self.cont)
	# End genLineGamma-funct


	def genLineRing(self, meanLag, phi):
		# Predefining variables for geometrical transfer function for a ring
		nTheta = 1000
		theta  = np.linspace(1.0e-10, np.pi-1.0e-10, nTheta)
		lag    = meanLag * ( 1.0 - np.cos(theta)*np.cos(phi) )

		# Defining transfer function
		tf = np.zeros(self.tfLen)
		for i in range (0, self.tfLen):
			if (i*self.dt >= lag[0] and i*self.dt <= lag[nTheta-1]):
				tf[i] = 1.0/np.sqrt( -meanLag**2*np.sin(phi)**2 + i*self.dt*(2*meanLag-i*self.dt))
			# End if-statement
		# End for-loop

		# Convolving transfer funct w/cont light curve to get emission line light curve
		self.line = np.convolve(tf, self.cont)
	# End of genlineRing-function


	def genLineSphere(self, lag):
		# Defining transfer function vector for spherical distribution
		tf = np.zeros(self.tfLen)
		for i in range(0, self.tfLen):
			if (i*self.dt >= 0 and i*self.dt <= 2*lag):
				tf[i] = 1.0/(2*lag)
			# End if-statement
		# End for-loop

		# Convolving transfer funct w/cont light curve to get emission line light curve
		self.line = np.convolve(tf, self.cont)
	# End of genLineSphere-function


	def genLineSkewedGaussian(self, lag):
		sigma = 1.*lag
		alpha = 10.
		tf = np.zeros(self.tfLen)
		for i in range(self.tfLen):
			normPDF = 1/(sigma*np.sqrt(2*np.pi)) * np.exp( -(i*self.dt-lag)**2 / (2*sigma**2))
			normCDF = 0.5 * ( 1 + sp.special.erf( (alpha*(i*self.dt-lag)/sigma) / np.sqrt(2) ) )
			tf[i] = 2*normPDF*normCDF
		# End for-loop

		# Convolving transfer funct w/cont light curve to get emission line light curve
		self.line = np.convolve(tf, self.cont)
	# End of genLineSkewedGaussian-function
# END CLASS



class create:
    '''
    Object for creating mock light-curves for AGN reverberation mapping.

    Contents
    ========

    Constants
    ---------
    - self.c -- Speed of light in Angstrom/second, 299792458*1e10
    - self.AB -- Spectral density of flux for AB magnitude zeropoint,
			     given in erg/s/Hz/cm^2, 3631e-23

    Functions for the general user
    ------------------------------
    - createLCproperties -- creates general AGN and light-curve
      properties necessary to create underlying/emitted AGN
      light-curves, based on MacLeod+10 [1]
    - createUnderlyingLC -- creates underlying/emitted AGN light-curves,
      redshifted to the observer frame
    - createUncertainties -- creates uncertainties and errors to add to
      underlying/emitted light-curves to obtain a more realistic,
      measured light-curve
    - createMeasuredLC -- based on survey specifications, creates arrays
      of 0 and 1, indicating which elements of the emitted/underlying
      light-curved and uncertainties and errors should be used to obtain
      a realistic, measured light-curve
    - LCwDatesFromObservations -- based on observed light-curve, creates
      arrays of 0 and 1, indicating which elements of the
      emitted/underlying light-curved and uncertainties and errors
      should be used to obtain a realistic, measured light-curve
    - realistic6yrs -- based on two observed light-curves, creates
      arrays of 0 and 1, indicating which elements of the
      emitted/underlying light-curved and uncertainties and errors
      should be used to obtain a realistic, measured light-curve

    Other functions
    ---------------
    - getFluxes -- returns measured flux through given photometry filter
      (as well as redshifted wavelength distribution, spectral density
      of luminosity, and measured flux of AB magnitude zeropoint for the
      same filter) based on redshift and bolometric luminosity
    - getLbol -- returns bolometric luminosity (as well as redshifted
      wavelength distribution, and spectral density) for object given
      its redshift and apparent magnitude through a photometry filter
    - getAbsMag -- returns absolute magnitude through given filter,
      based on redshift, apparent magnitude (can be through different
      filter), and bolometric luminosity
    - getLag -- returns spectral luminosities and AGN time-lags for
      Hbeta, MgII, and CIV emission-lines based on Bentz+13 [2],
      Trakhtenbrot+12 [3], and Kaspi+07 [4].

    Initial setup
    =============
    Start the class by creating an object, and providing it with the
    full paths to the SDSS composite QSO template, as well as the (DES)
    r-band and i-band filter curves, e.g.:

    >>> AGN = create('/Users/username/directory/sdssqsocomposite.txt',
				     '/Users/username/directory/DES_asahi_r.dat',
				     '/Users/username/directory/DES_asahi_i.dat')

    Creating underlying properties of AGN/light-curves
    ==================================================
    After initialising the class, start by creating .fits-file and fill
    the header of the zeroeth extension with general properties of the
    AGN and all of the three possible light-curves. Use redshift and
    apparent magnitude through r-band to create header with all
    parameters necessary to create light-curves. Provide path to
    directory (need not already exist) the .fits-file should be saved to
    if desired, as well as string to name the .fits-file. Do not include
    ".fits", e.g:

    >>> z, appmag = 1.5, 20.7
    >>> fname = 'myfitsfile'
    >>> fdir = '/Users/username/fitsdirectory/'
    >>> AGN.createLCproperties(z, appmag, fname, fdir=fdir)

    Create underlying/emitted light-curves by specifying their length.
    Can also specify as the number of *different* light-curve
    realisations (based on the same parameters) to create, e.g.:

    >>> LClen, nLC = 5000, 100
    >>> fitsfile = fname+'.fits'
    >>> AGN.createUnderlyingLC(fitsfile, LClen, nLC=nLC)

    Finally, create uncertainties and errors to add to
    underlying/emitted light-curves to obtain realistic-looking
    observed/measured light-curves. Specify measurement uncertainty as
    a fraction of the average measured flux value, e.g.:

    >>> cUnct, eUnct = 0.02, 0.10
    >>> AGN.createUncertainties(fitsfile, cUnct, eUnct)

    Creating measured light-curves
    ==============================
    Having created underlying properties and light-curves, the user
    should utilise one of the provided functions to create an array
    determining which of the elements of the underlying property arrays
    should be used to create a realistic measured light-curve.

    To create a measured light-curve based on a specific survey
    strategy, the survey specifications should be specified for
    photometry and spectroscopy individually, e.g.:

    >>> cProp = [7., 3., 183., 182., 21., 7., 1825.]
    >>> eProp = [28., 7., 183., 182., 21., 7., 1825., 365.]
    >>> AGN.createMeasuredLC(fitsfile, cProp, eProp)

    In this case we have specified that the photometry survey (defined
    in cProp) should have an average cadence of 7 days, with a maximum
    deviation of 3 days from the average. The observing season should go
    for 183 days of the year, with a seasonal gap of 182 days. The Moon
    does not interfere with observations 21 days of the month, and it
    makes observations impossible 7 days of the month. The last day of
    photometry observations should be no later than 1825 days from the
    starting date. Similarly, for the spectroscopy survey (defined in
    eProp) should have an average cadence of 28 days, with a maximum
    deviation of 7 days from the average. The observing season should go
    for 183 days of the year, with a seasonal gap of 182 days. The Moon
    does not interfere with observations 21 days of the month, and it
    makes observations impossible 7 days of the month. The last day of
    spectroscopy observations should be no later than 1825 days from the
    first date of *photometry* measurements, and the first date of
    spectroscopy measurements should be no earlier than 365 days after
    the first date of photometry measurements.

    The user may choose instead or in addition to use dates based on
    observational data. The observational data are expected to be in
    .txt-files, one for photometry, and one for spectroscopy, where the
    observational dates are provided in the first column. Given two
    .txt-files, realistic observational dates may be defined e.g.:

    >>> pfile = '/Users/username/observationdirectory/photometry.txt'
    >>> sfile = '/Users/username/observationdirectory/spectroscopy.txt'
    >>> AGN.LCwDatesFromObservations(fitsfile, pfile, sfile)

    A final way of defining dates for measurements is to use
    observational dates from two files, in order to increase the
    baseline of the survey. The observational data are again expected to
    be in separate .txt-files for photometry and spectroscopy, with the
    observational dates saved in the first column, e.g.:

    >>> pfile1 = '/Users/username/observationdirectory/phot1.txt'
    >>> pfile2 = '/Users/username/observationdirectory/phot2.txt'
    >>> sfile1 = '/Users/username/observationdirectory/spec1.txt'
    >>> sfile2 = '/Users/username/observationdirectory/spec21.txt'
    >>> AGN.realistic6yrs(fitsfile, pfile1, pfile2, sfile1, sfile2)

    See also
    ========
    Documentation of data structure.

    References
    ==========
    [1] MacLeod C. L. et al., 2010, ApJ, 721, 1014
    [2] Bentz M. C. et al., 2013, ApJ, 767, 149
    [3] Trakhtenbrot B., Netzer H., 2012, MNRAS, 427, 3081
    [4] Kaspi S., Brandt W. N., Maoz D., Netzer H., Schneider D. P.,
	    Shemmer O., 2007, ApJ, 659, 997

    '''

    def __init__(self, QSOtemplate, rFilter, gFilter, iFilter):
        # Note: According to Vanden Berk et al. (2001), the SDSS composite QSO spectrum is in f_lambda
        self.QSOtemp = ascii.read(QSOtemplate)  # SDSS composite QSO template
        self.rFilter = ascii.read(rFilter)      # (DES) r-band filter
        self.gFilter = ascii.read(gFilter)      # (DES) g-band filter
        self.iFilter = ascii.read(iFilter)      # (DES) i-band filter
        self.c  = 299792458*1e10                # Speed of light in A/s
        self.AB = 3631e-23                      # Spectral density of flux for AB magnitude zeropoint,
        # in erg/s/Hz/cm^2
        # End initialisation of class


    #%% FROM OBSERVABLES TO INTRINSIC PROPERTIES
    def getFluxes(self, z, Lbol, Filter):
	    # Redshift to object's z
	    DLcm  = WMAP7.luminosity_distance(z).to_value(u.cm)
	    waveZ = self.QSOtemp['Wave']*(1.+z)
	    filtZ = np.interp(waveZ, Filter['lambda'], Filter['trans'])

	    # Normalise spectral densities
	    L5100 = Lbol / (9.*5100.)
	    lum   = L5100 * self.QSOtemp['FluxD']/self.QSOtemp['FluxD'][4300]
	    flux  = lum / (4*np.pi*DLcm*DLcm * (1.+z))

	    # Compute flux through filter
	    tempflux = igrt.trapz(waveZ * flux * filtZ, x=waveZ)
	    zeroflux = igrt.trapz(self.c/waveZ * self.AB * filtZ, x=waveZ)

	    return waveZ, lum, tempflux, zeroflux
    # End getFluxes-function


    def getLbol(self, z, appmag, Filter):
	    # Calculate magnitude for AGN w/assumed luminosity
	    L46 = 1e46                                                      # Set luminosity to L = 1e46 erg/s
	    waveZ, lum, tempflux, zeroflux = self.getFluxes(z, L46, Filter) # Find fluxes observed through filters
	    m46 = -2.5 * np.log10(tempflux/zeroflux)                        # Apparent magnitude of assumed AGN

	    # Compute (bolometric) luminosity of provided source by comparing to assumed AGN
	    Lbol = L46 * 10**(-0.4*(appmag-m46))
	    lumC = lum * 10**(-0.4*(appmag-m46))

	    return waveZ, lumC, Lbol
    # End getLbol-function


    def getAbsMag(self, z, appmag, Lbol, Filter):
	    # Calculate value of fractions in K-correction
	    tempflux, zeroflux = self.getFluxes(z, Lbol, Filter)[2:]
	    frac1 = 10**(-0.4*appmag)
	    frac2 = zeroflux/tempflux

	    # Calculate K-correction
	    KQR = -2.5 * np.log10(frac1*frac2)

	    # Calculate distance modulus
	    DLpc = WMAP7.luminosity_distance(z).to_value(u.pc)
	    DM   = 5 * np.log10(0.1 * DLpc)

	    # Calculate apparent magnitude based on the apparent-absolute magnitude relationship
	    absMag = appmag - DM - KQR
	    return absMag
    # End getAbsMag-function


    def getLag(self, z, waveZ, splum):
	    # Interpolate luminosity onto a pretty wavelength grid
	    wave = np.arange(1300., 5200.)
	    lum  = np.interp(wave, self.QSOtemp['Wave'], splum)

	    # Compute luminositites at specific wavelengths
	    lL1350 = wave[50]*lum[50]
	    lL3000 = wave[1700]*lum[1700]
	    lL5100 = wave[3800]*lum[3800]

	    # Calculate observed Hbeta-lag based on Bentz et al. (2013)
	    B13a = 1.554
	    B13b = 0.546
	    Hbeta = 10**( B13a + B13b * np.log10(lL5100*1e-44) ) * (1.+z)

	    # Calculate observed MgII-lag based on Trakhtenbrot & Netzer (2012)
	    T12a = 1.340
	    T12b = 0.615
	    MgII = 10**( T12a + T12b * np.log10(lL3000*1e-44) ) * (1.+z)

	    # Calculate observed CIV-lag based on Kaspi et al. (2007)
	    K07a = 0.24
	    K07b = 0.55
	    CIV  = 10 * K07a * np.power(lL1350*1e-43, K07b) * (1.+z)

	    return lL5100, lL3000, lL1350, Hbeta, MgII, CIV
    # End getLag-function


    #%% CREATION OF .FITS-FILE
    def createLCproperties(self, z, appmag, sig, fname, lag_frac, fdir=None):
        '''
        Generates a few parameters necessary for AGN simulation.
        Parameters are saved to zeroeth extension header of .fits-file
        saved with name and directory specified by input parameters.
        If an identical file already exists, the file will be overwritten.
        ---
        INPUT:
        z: float
          Redshift of the AGN we wish to create.
        appmag: float
          Apparent magnitude (in r-band) of the AGN we wish to create.
        fdir: string
          Directory where we wish to save the .fits-file containing the
          light-curves representing the AGN we are about to create.
        fname: string
          Name of .fits-file containing the light-curve representing the
          AGN we are about to create. Do not include ".fits".
        ---
        '''

        # Define AGN parameters directly dependent on z and rmag
        waveZ, lum, Lbol = self.getLbol(z, appmag, self.gFilter)
        absMag = self.getAbsMag(z, appmag, Lbol, self.iFilter)
        lL5100, lL3000, lL1350, Hbeta, MgII, CIV = self.getLag(z, waveZ, lum)

        Hbeta *= lag_frac
        MgII *= lag_frac
        CIV *= lag_frac
        
        
        if z <= 0.62: # H beta
            logmBH = np.random.normal(8.23,0.37)-9.
        elif z <= 1.8 and z >0.62: # MgII
            logmBH = np.random.normal(9.65,0.39)-9.
        else: # CIV
            logmBH = np.random.normal(9.55,0.27)-9.
        '''
        if z <= 0.62: # H beta
            lag = Hbeta/(1+z)
        elif z <= 1.8 and z >0.62: # MgII
            lag = MgII/(1+z)
        else: # CIV
            lag = CIV/(1+z)
        
        
        
        G = 6.67*10**-11
        c = 2.998*10**8
        f = 4.47
        Msun = 1.989*10**30
    
        lag = lag*86400
        velocity = velocity*1000
    
        logmBH = np.log10(f*(pow(velocity, 2)*c*lag/G)/Msun)-9
        '''

        # Define SFinfty and tau parameters based on MacLeod+10
        Asf  = -0.51
        Bsf  = -0.479
        Csf  = 0.131
        Dsf  = 0.18
        Atau = 2.4
        Btau = 0.17
        Ctau = 0.03
        Dtau = 0.21

        # Define wavelengths for specific emission lines
        LamH  = 5100.
        LamMg = 3000.
        LamC  = 1350.

        # Define SFinfty and tau for each emission line
        
        # use 2*std of cont data
        sfH = 2.*sig
        sfMg = 2.*sig
        sfC = 2.*sig 
        #sfH   = 10**( Asf  + Bsf*np.log10(LamH/4000)   + Csf*(absMag+23)  + Dsf*logmBH )
        #sfMg  = 10**( Asf  + Bsf*np.log10(LamMg/4000)  + Csf*(absMag+23)  + Dsf*logmBH )
        #sfC   = 10**( Asf  + Bsf*np.log10(LamC/4000)   + Csf*(absMag+23)  + Dsf*logmBH )
        tauH  = 10**( Atau + Btau*np.log10(LamH/4000)  + Ctau*(absMag+23) + Dtau*logmBH ) * (1.+z)
        tauMg = 10**( Atau + Btau*np.log10(LamMg/4000) + Ctau*(absMag+23) + Dtau*logmBH ) * (1.+z)
        tauC  = 10**( Atau + Btau*np.log10(LamC/4000)  + Ctau*(absMag+23) + Dtau*logmBH ) * (1.+z)

        # Create .fits-header
        hdr = fits.Header()
        hdr['Z'] = z                    # Redshift
        hdr['L'] = Lbol                 # Bolometric luminosity
        hdr['MBH'] = 10**(logmBH+9)     # Black hole mass
        hdr['SFH'] = sfH                # SFinfty for Hbeta
        hdr['SFM'] = sfMg               # SFinfty for MgII
        hdr['SFC'] = sfC                # SFinfty for CIV
        hdr['TAUDH'] = tauH             # tau for Hbeta
        hdr['TAUDM'] = tauMg            # tau for MgII
        hdr['TAUDC'] = tauC             # tau for CIV
        hdr['LAMBDAH'] = LamH           # Wavelength of Hbeta
        hdr['LAMBDAM'] = LamMg          # Wavelength of MgII
        hdr['LAMBDAC'] = LamC           # Wavelength of CIV
        hdr['LAMLUMH'] = lL5100         # Wavelength * spectral luminosity for Hbeta
        hdr['LAMLUMM'] = lL3000         # Wavelength * spectral luminosity for MgII
        hdr['LAMLUMC'] = lL1350         # Wavelength * spectral luminosity for CIV
        hdr['AGNLAGH'] = Hbeta          # AGN Hbeta lag
        hdr['AGNLAGM'] = MgII           # AGN MgII lag
        hdr['AGNLAGC'] = CIV            # AGN CIV lag
        hdr['RAPPMAG'] = appmag         # Apparent magnitude through r-band filter
        hdr['IABSMAG'] = absMag         # Absolute magnitude through i-band filter
        hdu = fits.PrimaryHDU(header=hdr)

        # Check whether directory is specified, name .fits-file accordingly
        if fdir:
            if not os.path.exists(fdir): os.makedirs(fdir)
            ffile = fdir+fname+'_sim.fits'
        else:
            ffile = fname+'_sim.fits'
        # End if-statement

        # Save .fits-file
        if os.path.isfile(ffile): os.remove(ffile)
        hdu.writeto(ffile)
        # End createProperties-function


    def createUnderlyingLC(self, fitsfile, LClen, nLC=1, TF='TH', emline='CIV',ID=None):
        '''
        Generates the underlying continuum and emisson line light-curve(s)
        for an AGN. Saves light-curves to next extension of .fits-file
        specified by input.
        ---
        INPUT:
        fitsfile: string
          Full name of directory and name of fits-file we will be writing
          the light-curves to.
        LClen: int
          Number of points along underlying light-curves.
          (Note: Must be longer than the length of the "survey" for the
          measured light-curves, the measured light-curves will start
          after a time corresponding to the time lag length.)
        nLC: int (optional)
          Number of different realisations of underlying light-curves
          wanted for a set of light-curve parameters. By default, a single
          light-curve pair is created.
        TF: {'TH', 'Gauss', 'SkewedGauss'} (optional)
          Desired transfer function for smoothing the continuum
          light-curve into the emission-line light-curve. 'TH' (top-hat)
          by default.
        emline: {'Hbeta','MgII','CIV'} (optional)
          Emission line we wish to use to base the emission-line
          light-curve on. 'CIV' by default.
        ---
        '''
        
        
        
        '''
        # Check that we have gotten some reasonable input
        if (TF!='TH' and TF!='Gauss' and TF!='SkewedGauss'):
            print('Please give valid transfer function: Tophat, Gaussian, or Skewed Gaussian ("TH"/"Gauss"/"SkewedGauss")')
            return -1
        # End if-statement
        '''
        # Decide which parameters to use from the header
        if (emline=='Hbeta'):
            line = 'H'
        elif (emline=='MgII'):
            line = 'M'
        elif (emline=='CIV'):
            line = 'C'
        else:
            print('Please give a valid emission line option ("Hbeta"/"MgII"/"CIV").')
            return -1
        # End if-statement
        

        # Open fits-file
        ff = fits.open(fitsfile, mode='update')

        # Extract information from .fits-file
        rmag = ff[0].header['RAPPMAG']
        taud = ff[0].header['TAUD'+line]
        sf   = ff[0].header['SF'+line]
        lag  = ff[0].header['AGNLAG'+line]
        lam  = ff[0].header['LAMBDA'+line]

        # Initialise LightCurve
        offset = 1000                                 # Create large offset to remove rand effects from transfer function
        LClen += offset                               # Prolong length of underlying light-curves by offset
        finInt = int(np.ceil(LClen))                  # Make sure we are dealing with an integer to avoid numpy problems
        LCcube = np.zeros((2, finInt-offset, nLC))    # Initialise array to save underlying light-curves in
        LC = LightCurve(LClen)                        # Initialise class
        
        llc_path = "Processed_LC/"+str(ID)+"_line.txt"
        try:
            Real_llc = pd.read_csv(llc_path,sep='   ',names = ["MJD", "flux", "err"],engine='python')
            mean_f_real = np.mean(Real_llc['flux'])
        except:
            mean_f_real = 3.
        
        for i in range(nLC):
            # Use LightCurve-class to generate light curves
            LC.genContMag(rmag, taud, sf, i)            # Generate continuum light curve in mag
            LC.genCont()   # Convert coninuum light curve to flux
            
            mean_f_sim = np.mean(LC.cont)
            
            scale = mean_f_real/mean_f_sim  
                           

            # Generate emission-line light-curve from continuum light-curve
            if (TF == 'Gauss'):
                LC.genLineGauss(lag)
            elif (TF == 'SkewedGauss'):
                LC.genLineSkewedGauss(lag)
            else:
                LC.genLineTH(lag,scale)
            # End if-statement

            # Save light curves to datacube we'll be putting into fits file
            LCcube[0,:,i] = LC.cont[offset:finInt]
            LCcube[1,:,i] = LC.line[offset:finInt]
        # End for-loop

        # Append lightcurves to .fits-file
        ff.append(fits.ImageHDU(LCcube, name='UnderlyingLCs'))

        # Add info regarding what emission line and transfer functions were used to create the
        # emission-line light-curve into relevant extension
        ff[1].header['LAMBDA'] = lam
        ff[1].header['EMLINE'] = line
        ff[1].header['TF'] = TF

        # Save the updated .fits-file
        ff.flush()
        ff.close()
        # End of createUnderlyingLC-function


    def createUncertainties(self, fitsfile, cUnct, cStd, eUnct, eStd):
	    '''
	    Generates uncertainties and errors based on input to be added to
	    prestine, underlying light-curves in order to obtain a
	    realistically measured light-curve. Saves uncertainties and errors
	    in next extension of .fits-file specified by input.
	    ---
	    INPUT:
	    fitsfile: string
	      Full name of directory and name of fits-file we will be writing
	      the light-curves to.
	    cUnct: float
	      Measurement uncertainty for continuum light-curve (fraction).
	    eUnct: float
	      Measurement uncertainty for emission-line light-curve
	      (fraction).
	    ---
	    '''

	    # Open fits-file
	    agn = fits.open(fitsfile, mode='update')

	    # Establish size of underlying light-curve arrays
	    LCmat = agn[1].data
	    LClen = len(LCmat[0,:,0])
	    nLC   = len(LCmat[0,0,:])

	    # Check that we don't have a scary matrix we're dealing with
	    if np.any(LCmat < -111):
		    print('Uh-oh. Something seems iffy. You might want to have a look.')
		    import pdb; pdb.set_trace()
	    
        # Compute mean absolute error and std
        

	    # Create matrices w/measurement uncertainties
	    cumat = cUnct * np.mean(LCmat, axis=1)[0,:] * np.ones(LCmat[0,:,:].shape)
	    eumat = eUnct * np.mean(LCmat, axis=1)[1,:] * np.ones(LCmat[1,:,:].shape)

	    # Create matrices w/measurement errors to be added to underlying LCs
	    np.random.seed(int(time.time()))
	    cemat = cumat * np.random.normal(0, 1, size=LCmat[0,:,:].shape)
	    eemat = eumat * np.random.normal(0, 1, size=LCmat[1,:,:].shape)

	    # Fill in great big matrix to save in .fits-file
	    ucube = np.zeros((2,2,LClen,nLC))
	    ucube[0,0,:,:] = cumat
	    ucube[0,1,:,:] = eumat
	    ucube[1,0,:,:] = cemat
	    ucube[1,1,:,:] = eemat

	    # Append measured light curves to .fits-file
	    agn.append(fits.ImageHDU(ucube, name='Uncertainties'))

	    # Save uncertainty information to headers
	    nExt = len(agn)
	    agn[nExt-1].header['CMESERR'] = cUnct
	    agn[nExt-1].header['EMESERR'] = eUnct

	    # Save updated .fits-file
	    agn.flush()
	    agn.close()
    # End createUncertainties-function


    def createMeasuredLC(self, fitsfile, cProp, eProp):
	    '''
	    Generates arrays of zeroes and ones to use with underlying
	    light-curves and uncertainties to form measured light-curves.
	    Ones signify the values should be used to create a measured
	    light-curve, zeroes mean they should not be included. The arrays
	    are created based on survey specifications provided by input, and
	    saved in the next extension of the .fits-file provided as input.
	    ---
	    INPUT:
	    fitsfile: string
	      Full name of directory and name of fits-file we will be writing
	      the light curves to.
	    cProp: array-like
	      7-element array containing properties of the measured continuum
	      light-curve.
	      c0 = average observational cadence
	      c1 = deviation from observational cadence
	      c2 = length of observing season
	      c3 = length of seasonal gap
	      c4 = number of days of the month the Moon is down
	      c5 = number of days of the month the Moon is up
	      c6 = last day of observing
	    eProp: array-like
	      8-element array containing properties of the measured
	      emission-line light-curve.
	      e0 = average observational cadence
	      e1 = deviation from observational cadence
	      e2 = length of observing season
	      e3 = length of seasonal gap
	      e4 = number of days of the month the Moon is down
	      e5 = number of days of the month the Moon is up
	      e6 = last day of observing
	      e7 = starting day of observations
	    ---
	    '''

	    # Fetch underlying light cuves
	    agn   = fits.open(fitsfile, mode='update')
	    LCmat = agn[1].data
	    LClen = len(LCmat[0,:,0])
	    nLC   = len(LCmat[0,0,:])

	    # Properties for measured continuum light curves
	    cMesAvg = cProp[0]              # Average cadence for photometry measurements
	    cMesDev = cProp[1]              # Max number of days the observations can deviate from the average
	    cMesObs = cProp[2]              # Length of observing season for photometry
	    cMesGap = cProp[3]              # Length of seasonal gap for photometry
	    cLunDow = cProp[4]              # Number of days of the month the Moon is down
	    cLunUpp = cProp[5]              # Number of days of the month the Moon is up
	    cLastDa = int(cProp[6])         # Last possible date of photometry measurements

	    # Properties for measured emission line light curves
	    eMesAvg = eProp[0]              # Average cadence for spectroscopy measurements
	    eMesDev = eProp[1]              # Max number of days the observations can deviate from the average
	    eMesObs = eProp[2]              # Length of observing season for spectroscopy
	    eMesGap = eProp[3]              # Length of seasonal gap for spectroscopy
	    eLunDow = eProp[4]              # Number of days of the month the Moon is down
	    eLunUpp = eProp[5]              # Number of days of the month the Moon is up
	    eLastDa = int(eProp[6])         # Last possible date of spectroscopy measurements
	    eStartD = eProp[7]              # Start of spectroscopy measurements

	    # Double check that input makes sense
	    if LClen < cLastDa:
		    print('Underlying continuum light curve shorter than desired length of measured light curve. Aborting.')
		    return -1
	    if LClen < eLastDa:
		    print('Underlying emission line light curve shorter than desired length of measured light curve. Aborting.')
		    return -1
	    if not np.floor(cMesDev) < np.floor(cMesAvg):
		    print('Deviation in continuum observations too large compared to average cadence. Aborting.')
		    return -1
	    if not np.floor(eMesDev) < np.floor(eMesAvg):
		    print('Deviation in emission line observations too large compared to average cadence. Aborting.')
		    return -1
	    # End if-statements

	    # Create average measurement dates based on survey cadence
	    cAvg = cMesAvg * np.arange(cLastDa)
	    eAvg = eMesAvg * np.arange(eLastDa)

	    # Create an array of random numbers to to add to the average cadence
	    cRand = np.random.randint(0, cMesDev+1, size=cLastDa)
	    eRand = np.random.randint(0, eMesDev+1, size=eLastDa)

	    # Create arrays of observational dates
	    cObsLong = cAvg + cRand
	    eObsLong = eAvg + eRand + eStartD

	    # Only save observational dates up to the last days we are interested in
	    cBefGaps = cObsLong[ cObsLong<cLastDa ]
	    eBefGaps = eObsLong[ eObsLong<eLastDa ]

	    # Find fractions of the year/month we are doing observations
	    cSFrac, cLFrac = float(cMesObs)/(cMesObs+cMesGap), float(cLunDow)/(cLunUpp+cLunDow)
	    eSFrac, eLFrac = float(eMesObs)/(eMesObs+eMesGap), float(eLunDow)/(eLunUpp+eLunDow)

	    # Calculate observational days as fractions of years
	    cYrFrac = cBefGaps / (cMesObs+cMesGap)
	    eYrFrac = eBefGaps / (eMesObs+eMesGap)

	    # Select out observational days within observing seasons
	    cW1gap = cBefGaps[ np.where( cYrFrac-np.floor(cYrFrac) <= cSFrac )[0] ]
	    eW1gap = eBefGaps[ np.where( eYrFrac-np.floor(eYrFrac) <= eSFrac )[0] ]

	    # Calculate observational days as fractions of months
	    cLuFrac = cW1gap / (cLunUpp+cLunDow)
	    eLuFrac = eW1gap / (eLunUpp+eLunDow)

	    # Select out observational days when the Moon is not in the way
	    cObsD = cW1gap[ np.where( cLuFrac-np.floor(cLuFrac) <= cLFrac )[0] ]
	    eObsD = eW1gap[ np.where( eLuFrac-np.floor(eLuFrac) <= eLFrac )[0] ]

	    # Create array with dates
	    elem = np.zeros(LCmat[:,:,0,].shape)
	    elem[0,cObsD.astype(int)] = 1
	    elem[1,eObsD.astype(int)] = 1

	    # Append measured light curves to .fits-file
	    agn.append(fits.ImageHDU(elem, name='MeasuredLCs'))

	    # Save AGN/LC information to headers
	    nExt = len(agn)
	    agn[nExt-1].header['ECADAVG'], agn[nExt-1].header['CCADAVG'] = eMesAvg, cMesAvg
	    agn[nExt-1].header['ECADDEV'], agn[nExt-1].header['CCADDEV'] = eMesDev, cMesDev
	    agn[nExt-1].header['EOBSLEN'], agn[nExt-1].header['COBSLEN'] = eMesObs, cMesObs
	    agn[nExt-1].header['EGAPLEN'], agn[nExt-1].header['CGAPLEN'] = eMesGap, cMesGap
	    agn[nExt-1].header['ELUNUP'],  agn[nExt-1].header['CLUNUP']  = eLunUpp, cLunUpp
	    agn[nExt-1].header['ELUNDOW'], agn[nExt-1].header['ELUNDOW'] = eLunDow, cLunDow
	    agn[nExt-1].header['ELASTDA'], agn[nExt-1].header['CLASTDA'] = eLastDa, cLastDa
	    agn[nExt-1].header['ESTARTD'] = eStartD

	    # Save updated .fits-file
	    agn.flush()
	    agn.close()
    # End createMeasuredLC-function


    def LCwDatesFromObservations(self, fitsfile, pfile, sfile):
	    '''
	    Generates arrays of zeroes and ones to use with underlying
	    light-curves and uncertainties to form measured light-curves.
	    Ones signify the values should be used to create a measured
	    light-curve, zeroes mean they should not be included. The arrays
	    are created based on observational dates from observed
	    light-curves provided as input.
	    ---
	    INPUT:
	    fitsfile: string
	      Full name of directory and name of fits-file we will be writing
	      the light curves to.
	    pfile: string
	      Path to observed continuum light-curve. It is expected that the
	      light-curve is saved in a .txt-file containing numbers only,
	      and that the first column of the file contains the (Juliean)
	      dates of observation.
	    sfile: string
	      Path to observed emission-line light-curve. It is expected that
	      the light-curve is saved in a .txt-file containing numbers only,
	      and that the first column of the file contains the (Juliean)
	      dates of observation.
	    ---
	    '''

	    # Fetch underlying light-curves
	    agn = fits.open(fitsfile, mode='update')
	    LCmat = agn[1].data

	    # Load in underlying light-curves from file
	    pmat = np.loadtxt(pfile)
	    smat = np.loadtxt(sfile)

	    # Save dates as integers so we can treat them as elements
	    pdates = np.unique(np.round((pmat[:,0]-pmat[0,0])).astype(int))
	    sdates = np.unique(np.round((smat[:,0]-pmat[0,0])).astype(int))

	    # Create array with dates
	    clen = len(pdates)
	    elen = len(sdates)
	    elem = np.zeros(LCmat[:,:,0].shape)
	    elem[0,pdates.astype(int)] = 1
	    elem[1,sdates.astype(int)] = 1

	    # Append measured light curves to .fits-file
	    agn.append(fits.ImageHDU(elem, name='MeasuredLCs'))

	    # Save updated .fits-file
	    agn.flush()
	    agn.close()
    # End LCwDatesFromObservations-function


    def realistic6yrs(self, fitsfile, pfile1, pfile2, sfile1, sfile2):
	    '''
	    Generates arrays of zeroes and ones to use with underlying
	    light-curves and uncertainties to form measured light-curves.
	    Ones signify the values should be used to create a measured
	    light-curve, zeroes mean they should not be included. The arrays
	    are created based on observational dates from two observed
	    light-curves provided as input.
	    ---
	    INPUT:
	    fitsfile: string
	      Full name of directory and name of fits-file we will be writing
	      the light curves to.
	    pfile1: string
	      Path to first observed continuum light-curve. It is expected
	      that the light-curve is saved in a .txt-file containing numbers
	      only, and that the first column of the file contains the
	      (Juliean) dates of observation.
	    pfile2: string
	      Path to second observed continuum light-curve. It is expected
	      that the light-curve is saved in a .txt-file containing numbers
	      only, and that the first column of the file contains the
	      (Juliean) dates of observation.
	    sfile1: string
	      Path to first observed emission-line light-curve. It is expected
	      that the light-curve is saved in a .txt-file containing numbers
	      only, and that the first column of the file contains the
	      (Juliean) dates of observation.
	    sfile2: string
	      Path to second observed emission-line light-curve. It is
	      expected that the light-curve is saved in a .txt-file containing
	      numbers only, and that the first column of the file contains the
	      (Juliean) dates of observation.
	    ---
	    '''

	    # Fetch underlying light-curves
	    agn = fits.open(fitsfile, mode='update')
	    LCmat = agn[1].data

	    # Load in underlying light-curves from files
	    pmat1, pmat2 = np.loadtxt(pfile1), np.loadtxt(pfile2)
	    smat1, smat2 = np.loadtxt(sfile1), np.loadtxt(sfile2)

	    # Create single array containing observational dates
	    parr = np.append(pmat1[:,0], 3*365+pmat2[:,0]) - pmat1[0,0]
	    sarr = np.append(smat1[:,0], 3*365+smat2[:,0]) - smat1[0,0]

	    # Save dates as integers so we can treat them as elements
	    pdates = np.unique(np.round(parr)).astype(int)
	    sdates = np.unique(np.round(sarr)).astype(int)

	    pdates = pdates[pdates < 2100]
	    sdates = sdates[sdates < 2100]

	    # Create array to decide whether observations were made
	    elem = np.zeros(LCmat[:,:,0].shape)
	    elem[0,pdates.astype(int)] = 1
	    elem[1,sdates.astype(int)] = 1

	    # Append measured light-curves to .fits-file
	    agn.append(fits.ImageHDU(elem, name='MeasuredLCs'))

	    # Save updated .fits-file
	    agn.flush()
	    agn.close()

    # End realistic6yrs

    def custom(self, fitsfile):

	    agn = fits.open(fitsfile, mode='update')
	    LCmat = agn[1].data
	    LClen = len(LCmat[0, :, 0])
	    nLC = len(LCmat[0, 0, :])

	    # Properties for measured continuum light curves
	    cMesAvg = 7  # Average cadence for photometry measurements
	    cMesDev = 3  # Max number of days the observations can deviate from the average
	    cMesObs = 150  # Length of observing season for photometry
	    cMesGap = 215  # Length of seasonal gap for photometry
	    cLunDow = 30  # Number of days of the month the Moon is down
	    cLunUpp = 0  # Number of days of the month the Moon is up
	    cLastDa = int(2190)  # Last possible date of photometry measurements
	    vlast = 2920 # 8 years

	    # Properties for measured emission line light curves
	    eMesAvg = 28  # Average cadence for spectroscopy measurements
	    eMesDev = 7  # Max number of days the observations can deviate from the average
	    eMesObs = 150  # Length of observing season for spectroscopy
	    eMesGap = 215  # Length of seasonal gap for spectroscopy
	    eLunDow = 21  # Number of days of the month the Moon is down
	    eLunUpp = 9  # Number of days of the month the Moon is up
	    eLastDa = int(2190)  # Last possible date of spectroscopy measurements
	    eStartD = 0  # Start of spectroscopy measurements

	    cAvg = cMesAvg * np.arange(vlast)
	    eAvg = eMesAvg * np.arange(vlast)

	    # Create an array of random numbers to to add to the average cadence
	    cRand = np.random.randint(0, cMesDev + 1, size=vlast)
	    eRand = np.random.randint(0, eMesDev + 1, size=vlast)

	    # Create arrays of observational dates
	    cObsLong = cAvg + cRand
	    eObsLong = eAvg + eRand + eStartD

	    # Only save observational dates up to the last days we are interested in
	    cBefGaps = cObsLong[cObsLong < vlast]
	    eBefGaps = eObsLong[eObsLong < vlast]

	    # Find fractions of the year/month we are doing observations
	    cSFrac, cLFrac = float(cMesObs) / (cMesObs + cMesGap), float(cLunDow) / (cLunUpp + cLunDow)
	    eSFrac, eLFrac = float(eMesObs) / (eMesObs + eMesGap), float(eLunDow) / (eLunUpp + eLunDow)

	    # Calculate observational days as fractions of years
	    cYrFrac = cBefGaps / (cMesObs + cMesGap)
	    eYrFrac = eBefGaps / (eMesObs + eMesGap)

	    cW1gap = cBefGaps[np.where((cBefGaps < cLastDa) & (cYrFrac - np.floor(cYrFrac) <= cSFrac))[0]]
	    eW1gap = eBefGaps[np.where((eBefGaps < eLastDa) & (eYrFrac - np.floor(eYrFrac) <= eSFrac))[0]]

	    newfrac = 210 / 365 # 7 month season

	    cW2gap = cBefGaps[np.where((cBefGaps > 2190) & (cYrFrac - np.floor(cYrFrac) <= newfrac))[0]]
	    eW2gap = eBefGaps[np.where((eBefGaps > 2190) & (eYrFrac - np.floor(eYrFrac) <= newfrac))[0]]

	    cDates = np.append(cW1gap, cW2gap)
	    eDates = np.append(eW1gap, eW2gap)

	    # Calculate observational days as fractions of months
	    cLuFrac = cDates / (cLunUpp + cLunDow)
	    eLuFrac = eDates / (eLunUpp + eLunDow)

	    # Select out observational days when the Moon is not in the way
	    cObsD = cDates[np.where(cLuFrac - np.floor(cLuFrac) <= cLFrac)[0]]
	    eObsD = eDates[np.where(eLuFrac - np.floor(eLuFrac) <= eLFrac)[0]]

	    # Create array with dates
	    elem = np.zeros(LCmat[:, :, 0, ].shape)
	    elem[0, cObsD.astype(int)] = 1
	    elem[1, eObsD.astype(int)] = 1

	    # Append measured light curves to .fits-file
	    agn.append(fits.ImageHDU(elem, name='custom'))

	    agn.flush()
	    agn.close()

# END CLASS
