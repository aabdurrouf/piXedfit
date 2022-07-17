import numpy as np
from math import pi, pow, sqrt, cos, sin
import sys, os
import h5py
from random import randint
from astropy.io import fits
from .fitutils import *

try:
	global PIXEDFIT_HOME
	PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'
except:
	print ("PIXEDFIT_HOME should be included in your PATH!")

os.environ["OMP_NUM_THREADS"] = "1"

__all__ = ["singleSEDfit", "singleSEDfit_specphoto", "SEDfit_from_binmap", "SEDfit_from_binmap_specphoto", 
			"SEDfit_pixels_from_fluxmap", "maps_parameters", "maps_parameters_fit_pixels", "get_params", "priors"]


class priors:
	"""Functions for defining priors to be used in the Bayesian SED fitting process. First, one need to define ranges for the parameters using :func:`params_ranges`, 
	then define shape of the prior distribution function of each parameter. The available analytic forms for the prior are uniform, Gaussian, Student's t, and gamma functions.
	User can also choose an arbitrary one, using :func:`arbitrary`. It is also possible to define a joint prior between a given parameter and stellar mass. This joint prior 
	can be adopted from a known scaling relation, such as stellar mass vs metallicity relation. Note that it is not required to define range of all parameters to be involved in SED fitting. 
	If range is not inputted, the default one will be used. It is also not required to define prior shape of all parameters. If not inputted, a uniform prior will be used. 
	Due to the defined ranges, the analytical form is truncated at the edges defined by the range.  
	
	:param ranges: 
		The ranges of the parameters.
	"""

	def __init__(self, ranges={'z':[0.0,1.0],'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-1.0,1.14],
		'log_alpha':[-2.0,2.0],'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0], 
		'dust2':[0.0,4.0],'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],
		'log_tauagn':[0.7, 2.18],'log_mw_age':[-2.0,1.14],'log_mass':[4.0,12.0]}):

		def_ranges={'z':[0.0,1.0],'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-1.0,1.14],'log_alpha':[-2.0,2.0],
			'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0],'dust2':[0.0,4.0],
			'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],
			'log_tauagn':[0.7, 2.18],'log_mw_age':[-2.0,1.14],'log_mass':[4.0,12.0]}

		# get keys in input params_range:
		keys = list(ranges.keys())

		# merge with the default one
		ranges1 = def_ranges
		for ii in range(0,len(keys)):
			if keys[ii] in ranges1:
				ranges1[keys[ii]] = ranges[keys[ii]]
		
		self.ranges = ranges1

	def params_ranges(self):
		"""Function for defining ranges of the parameters.

		:returns params_ranges:
			Ranges of the parameters to be inputted into a SED fitting function.
		"""
		return self.ranges

	def uniform(self, param):
		"""Function for assigning uniform prior to a parameter.

		:param param:
			The parameter that will be assigned the uniform prior.

		:returns prior:
			Output prior.
		"""
		prior = [param, "uniform"]
		return prior 

	def gaussian(self, param, loc, scale):
		"""Function for assigning Gaussian prior to a parameter.

		:param param:
			The parameter that will be assigned the Gaussian prior.

		:param loc:
			Peak location of the Gaussian prior.

		:param scale:
			Width or standard deviation of the Gaussian prior.

		:returns prior:
			Output prior.
		"""
		prior = [param, "gaussian", loc, scale]
		return prior

	def studentt(self, param, df, loc, scale):
		"""Function for assigning a prior in the form of Student's t distribution.

		:param param:
			The parameter that will be assigned the Student's t prior.

		:param df:
			The degree of freedom.

		:param loc:
			Peak location.

		:param scale:
			Width of the distribution.

		:returns prior:
			Output prior.
		"""
		prior = [param, "studentt", df, loc, scale]
		return prior

	def gamma(self, param, a, loc, scale):
		"""Function for assigning a prior in the form of Gamma function to a parameter.

		:param param:
			The parameter that will be assigned the Gamma prior.

		:param a:
			A shape parameter in the gamma function.

		:param loc:
			Peak location.

		:param scale:
			Width of the distribution.

		:returns prior:
			Output prior.
		"""
		prior = [param, "gamma", a, loc, scale]
		return prior

	def arbitrary(self, param, values, prob):
		"""Function for assigning an arbitrary prior.

		:param param:
			The parameter to be assigned with the arbitrary prior.

		:param values:
			Array of values.

		:param prob:
			Array of probability associated with the values.

		:returns prior:
			Output prior.

		"""
		namepr = randname("arbtprior",".dat")
		write_arbitprior(namepr,values,prob)
		os.system('mv %s %s' % (namepr,temp_dir))
		prior = [param, "arbitrary", namepr]
		return prior

	def joint_with_mass(self, param, log_mass, param_values, scale):
		"""Function for assigning a joint prior between a given parameter and stellar mass (log_mass).

		:param param:
			The parameter that will share a joint prior with the stellar mass.

		:param log_mass:
			Array of stellar mass values.

		:param param_values:
			Array of the parameter values. In this case, the parameter that shares a joint prior with the stellar mass. 

		:param scale:
			Array of width or standard deviations of the param_value.

		:returns prior:
			Output prior. 
		"""
		namepr = randname("jprmass",".dat")
		write_joint_prior(namepr,log_mass,param_values)
		os.system('mv %s %s' % (namepr,temp_dir))
		prior = [param, "joint_with_mass", namepr, scale]
		return prior



def singleSEDfit(obs_flux,obs_flux_err,filters,models_spec,params_ranges=None,params_priors=None,fit_method='mcmc',
	gal_z=None,nrands_z=10,add_igm_absorption=0, igm_type=0,likelihood='gauss',dof=2.0,nwalkers=100,nsteps=600,nsteps_cut=50,
	nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0,cosmo=0,H0=70.0,Om0=0.3,store_full_samplers=1,name_out_fits=None):
	"""Function for performing SED fitting to a single photometric SED.

	:param obs_flux: 
		Input fluxes in multiple bands. The format is 1D array with a number of elements of the array should be the sama as that of obs_flux_err and filters.
		The fluxes should be in the unit of erg/s/cm^2/Angstrom.
	
	:param obs_flux_err:
		Input flux uncertainties in multiple bands. The flux uncertainties should be in the unit of erg/s/cm^2/Angstrom.

	:param filters:
		List of photometric filters. The list of filters recognized by piXedfit can be accesses using :func:`piXedfit.utils.filtering.list_filters`. 
		Please see `this page <https://pixedfit.readthedocs.io/en/latest/manage_filters.html>`_ for information on managing filters that include listing available filters, adding, and removing filters. 

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the SED fitting process using the function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of model spectra will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:
		Ranges of the parameter defined (i.e., outputted) using the :func:`params_ranges` in the :class:`priors` class.

	:param params_priors:
		Forms of adopted priros. The acceptable format is a list, such as params_priors=[prior1, prior2, prior3] where prior1, prior2, and prior3 are output of functions in the :class:`priors` class.
	
	:param fit_method:
		Choice of method for the SED fitting. Options are: (a)'mcmc' for Markov Chain Monte Carlo, and (b)'rdsps' for Random Dense Sampling of Parameter Space.

	:param gal_z: 
		Redshift of the galaxy. If gal_z=None, redshift will be a free parameter in the fitting.  

	:param nrands_z:
		Number of random redshifts to be generated (within the chosen range as set in the params_range) in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.
		This is only relevant if gal_z=None (i.e., photometric redshift will be activated).

	:param add_igm_absorption:
		Switch for the IGM absorption. Options are: 0 for switch off and 1 for switch on.

	:param igm_type: 
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).
	
	:param likelihood: 
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: 'gauss' for the Gaussian form and 'student_t' for the student's t form.

	:param dof: 
		Degree of freedom (nu) in the Student's t likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: 
		Number of walkers in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: 
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: 
		Number of first steps of each walkers that will be cut when collecting the final sampler chains. Only relevant if fit_method='mcmc' and store_full_samplers=1.

	:param nproc: 
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: 
		Number of models to be used in the initial fitting in the MCMC method. Only relevant if fit_method='mcmc'.  

	:param perc_chi2:
		A percentile in the set of models sorted based on the chi-square values that will be considered in the calculation of the best-fit parameters (i.e., posterior-weighted averages) in the RDSPS fitting. 
		This parameter is only relevant if fit_method='rdsps'.

	:param cosmo: 
		Choices for the cosmology. Options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: 
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: 
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param store_full_samplers:
		Flag indicating whether full sampler models will be stored into the output FITS file or not. Options are: 1 or True for storing the full samplers and 0 or False otherwise.

	:param name_out_fits:
		Name of the output FITS file. This parameter is optional. If None, a default name will be adopted.
	"""

	# get number of filters:
	nbands = len(filters)

	# file of filter list
	name_filters_list = randname("filters_list",".dat")
	write_filters_list(name_filters_list,filters)
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc

	if params_ranges is None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors is None:
		params_priors = []

	if gal_z is None or gal_z<=0.0:
		gal_z = -99.0
		free_z = 1
	else:
		free_z = 0

	# configuration file
	name_config = randname("config_file",".dat")
	flg_write = write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc)
	os.system('mv %s %s' % (name_config,temp_dir))

	# input SED text file
	name_SED_txt = randname("inputSED_file",".dat")
	write_input_singleSED(name_SED_txt,obs_flux,obs_flux_err)
	os.system('mv %s %s' % (name_SED_txt,temp_dir))

	# output files name:
	if name_out_fits is None:
		if fit_method=='mcmc' or fit_method=='MCMC':
			name_out_fits = randname("mcmc_fit",".fits")
		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_out_fits = randname("rdsps_fit",".fits")

	if fit_method=='mcmc' or fit_method=='MCMC':
		name_samplers_hdf5 = randname("samplers_",".hdf5")

		os.system("mpirun -n %d python %s./mc_p1.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																	name_SED_txt,name_samplers_hdf5))
		os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
		
		if store_full_samplers==1 or store_full_samplers==True:
			os.system("mpirun -n %d python %s./mc_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																		name_samplers_hdf5,name_out_fits))
		elif store_full_samplers==0 or store_full_samplers==False:
			os.system("mpirun -n %d python %s./mc_nsmp_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																			name_samplers_hdf5,name_out_fits))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()

	elif fit_method=='rdsps' or fit_method=='RDSPS':
		if store_full_samplers==1 or store_full_samplers==True:
			if free_z==0:
				os.system("mpirun -n %d python %s./rd_fz.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																			name_SED_txt,name_out_fits))
			elif free_z==1:
				os.system("mpirun -n %d python %s./rd_vz.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																			name_SED_txt,name_out_fits))
		elif store_full_samplers==0 or store_full_samplers==False:
			if free_z==0:
				os.system("mpirun -n %d python %s./rd_fz_nsmp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_SED_txt,name_out_fits))
			elif free_z==1:
				os.system("mpirun -n %d python %s./rd_vz_nsmp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_SED_txt,name_out_fits))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()

	else:
		print ("The input fit_method is not recognized!")
		sys.exit()

	# free disk space:
	os.system("rm %s%s" % (temp_dir,name_config))
	os.system("rm %s%s" % (temp_dir,name_filters_list))
	os.system("rm %s%s" % (temp_dir,name_SED_txt))
	if fit_method=='mcmc' or fit_method=='MCMC':
		os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))
	if len(flg_write)>0:
		for ii in range(0,len(flg_write)):
			os.system("rm %s%s" % (temp_dir,flg_write[ii]))

	return name_out_fits


def singleSEDfit_specphoto(obs_flux,obs_flux_err,filters,spec_wave,spec_flux,spec_flux_err,models_spec,
	wavelength_range=None,params_ranges=None,params_priors=None,fit_method='mcmc',gal_z=None,nrands_z=10,
	add_igm_absorption=0,igm_type=0,spec_sigma=2.6,poly_order=8, likelihood='gauss',dof=2.0,nwalkers=100,nsteps=600,
	nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0, spec_chi_sigma_clip=3.0,cosmo=0,H0=70.0,Om0=0.3,
	del_wave_nebem=10.0,store_full_samplers=1,name_out_fits=None):
	"""Function for performing SED fitting to a single spectrophotometric SED.

	:param obs_flux: 
		Input fluxes in multiple bands. The format is 1D array with a number of elements of the array should be the sama as that of obs_flux_err and filters.
		The fluxes should be in the unit of erg/s/cm^2/Angstrom.

	:param obs_flux_err:
		Input flux uncertainties in multiple bands. The flux uncertainties should be in the unit of erg/s/cm^2/Angstrom.

	:param filters: 
		List of photometric filters. The list of filters recognized by piXedfit can be accesses using :func:`piXedfit.utils.filtering.list_filters`. 
		Please see `this page <https://pixedfit.readthedocs.io/en/latest/manage_filters.html>`_ for information on managing filters that include listing available filters, adding, and removing filters. 

	:param spec_wave:
		1D array of wavelength of the input spectrum.
	
	:param spec_flux: 
		Flux grids of the input spectrum. The fluxes should be in the unit of erg/s/cm^2/Angstrom.

	:param spec_flux_err: 
		Flux uncertainties of the input spectrum. The flux uncertainties should be in the unit of erg/s/cm^2/Angstrom.

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the SED fitting process using the function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of model spectra will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param wavelength_range:
		Range of wavelength within which the observed spectrum will be considered in the SED fitting. 
		The accepted format is [wmin,wmax] with wmin and wmax are minimum and maximum wavelengths.

	:param params_ranges:
		Ranges of the parameter defined (i.e., outputted) using the :func:`params_ranges` in the :class:`priors` class.

	:param params_priors:
		Forms of adopted priros. The acceptable format is a list, such as params_priors=[prior1, prior2, prior3] where prior1, prior2, and prior3 are output of functions in the :class:`priors` class.

	:param fit_method:
		Choice of method for the SED fitting. Options are: (a)'mcmc' for Markov Chain Monte Carlo, and (b)'rdsps' for Random Dense Sampling of Parameter Space.

	:param gal_z: 
		Redshift of the galaxy. If gal_z=None, redshift will be a free parameter in the fitting. 

	:param spec_sigma: 
		Spectral resolution (in Angstrom) of the input spectrum.

	:param poly_order: 
		The degree of the legendre polynomial function to be used for correcting the shape (normalization) of the model spectra.  

	:param nrands_z:
		Number of random redshifts to be generated (within the chosen range as set in the params_range) in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.
		This is only relevant if gal_z=None (i.e., photometric redshift will be activated).

	:param add_igm_absorption:
		Switch for the IGM absorption. Options are: 0 for switch off and 1 for switch on.

	:param igm_type: 
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).
	
	:param likelihood: 
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: 'gauss' for the Gaussian form and 'student_t' for the student's t form.

	:param dof: 
		Degree of freedom (nu) in the Student's t likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: 
		Number of walkers in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: 
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: 
		Number of first steps of each walkers that will be cut when collecting the final sampler chains. Only relevant if fit_method='mcmc' and store_full_samplers=1.

	:param nproc: 
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: 
		Number of models to be used in the initial fitting in the MCMC method. Only relevant if fit_method='mcmc'.  

	:param perc_chi2:
		A percentile in the set of models sorted based on the chi-square values that will be considered in the calculation of the best-fit parameters (i.e., posterior-weighted averages) in the RDSPS fitting. 
		This parameter is only relevant if fit_method='rdsps'.
	
	:param spec_chi_sigma_clip: 
		Standard deviation (sigma) to be adopted in the sigma clipping to the spectrum data points that are regarded as outliers before 
		calculating chi-square in the SED fitting process. The sigma clipping is carried out based on the distribution of chi values (sum((D-M)/Derr)). 

	:param cosmo: 
		Choices for the cosmology. Options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: 
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: 
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param del_wave_nebem:
		This parameter defines the Wavelength region (+/- del_wave_nebem) around the emission lines that will be excluded in the fitting of spectral continuum between the model spectrum and the observed one.  

	:param store_full_samplers:
		Flag indicating whether full sampler models will be stored into the output FITS file or not. Options are: 1 or True for storing the full samplers and 0 or False otherwise.

	:param name_out_fits:
		Name of the output FITS file. This parameter is optional. If None, a default name will be adopted.
	"""

	# get number of filters:
	nbands = len(filters)

	# file of filter list
	name_filters_list = randname("filters_list",".dat")
	write_filters_list(name_filters_list,filters)
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc

	if params_ranges is None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors is None:
		params_priors = []

	if gal_z is None or gal_z<=0.0:
		gal_z = -99.0
		free_z = 1
	else:
		free_z = 0

	# configuration file
	name_config = randname("config_file",".dat")
	flg_write = write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc,spec_sigma=spec_sigma,poly_order=poly_order,
					del_wave_nebem=del_wave_nebem,spec_chi_sigma_clip=spec_chi_sigma_clip)
	os.system('mv %s %s' % (name_config,temp_dir))

	# input SED in HDF5
	inputSED_file = randname("inputSED_file",".hdf5")
	# cut spectrum if required
	if wavelength_range is not None:
		idx0 = np.where((spec_wave>=wavelength_range[0]) & (spec_wave<=wavelength_range[1]))
		spec_wave = spec_wave[idx0[0]]
		spec_flux = spec_flux[idx0[0]]
		spec_flux_err = spec_flux_err[idx0[0]]
	write_input_specphoto_hdf5(inputSED_file,obs_flux,obs_flux_err,spec_wave,spec_flux,spec_flux_err)
	os.system('mv %s %s' % (inputSED_file,temp_dir))

	# output files name:
	if name_out_fits is None:
		if fit_method=='mcmc' or fit_method=='MCMC':
			name_out_fits = randname("mcmc_fit",".fits")
		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_out_fits = randname("rdsps_fit",".fits")

	if fit_method=='mcmc' or fit_method=='MCMC':
		name_samplers_hdf5 = randname("samplers_",".hdf5")

		os.system("mpirun -n %d python %s./mc_p1_sp.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																		inputSED_file,name_samplers_hdf5))
		os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
		
		if store_full_samplers==1 or store_full_samplers==True:
			os.system("mpirun -n %d python %s./mc_p2_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																			name_samplers_hdf5,name_out_fits))
		elif store_full_samplers==0 or store_full_samplers==False:
			os.system("mpirun -n %d python %s./mc_nsmp_p2_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_samplers_hdf5,name_out_fits))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()

	elif fit_method=='rdsps' or fit_method=='RDSPS':
		if store_full_samplers==1 or store_full_samplers==True:
			if free_z==0:
				os.system("mpirun -n %d python %s./rd_fz_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				inputSED_file,name_out_fits))
			elif free_z==1:
				os.system("mpirun -n %d python %s./rd_vz_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				inputSED_file,name_out_fits))
		elif store_full_samplers==0 or store_full_samplers==False:
			if free_z==0:
				os.system("mpirun -n %d python %s./rd_fz_nsmp_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				inputSED_file,name_out_fits))
			elif free_z==1:
				os.system("mpirun -n %d python %s./rd_vz_nsmp_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				inputSED_file,name_out_fits))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()

	else:
		print ("The input fit_method is not recognized!")
		sys.exit()

	# free disk space:
	os.system("rm %s%s" % (temp_dir,name_config))
	os.system("rm %s%s" % (temp_dir,name_filters_list))
	os.system("rm %s%s" % (temp_dir,inputSED_file))
	if fit_method=='mcmc' or fit_method=='MCMC':
		os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))
	if len(flg_write)>0:
		for ii in range(0,len(flg_write)):
			os.system("rm %s%s" % (temp_dir,flg_write[ii]))

	return name_out_fits


def SEDfit_from_binmap_specphoto(fits_binmap,binid_range=None,bin_ids=None,wavelength_range=None,models_spec=None,
	params_ranges=None,params_priors=None,fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,
	spec_sigma=2.6,poly_order=8,likelihood='gauss',dof=3.0,nwalkers=100,nsteps=600,nsteps_cut=50,nproc=10,
	initfit_nmodels_mcmc=30000,perc_chi2=90.0,spec_chi_sigma_clip=3.0, cosmo=0,H0=70.0,Om0=0.3,del_wave_nebem=10.0,
	store_full_samplers=1,name_out_fits=None):
	"""Function for performing SED fitting to a spectrophotometric data cube. The data cube should has been binned using the 
	function :func:`pixel_binning`.

	:param fits_binmap:
		Input FITS file of the spectrophotometric data cube that has been binned.

	:param binid_range:
		Range of bin IDs which the SEDs are going to be fit. The accepted format is [idmin,idmax]. The ID starts from 0. 
		If None, the SED fitting will be done to all of the spatial bins in the galaxy.

	:param bin_ids:
		Bin IDs which the SEDs are going to be fit. The accepted format is 1D array. 
		The ID starts from 0. Both binid_range and bin_ids can't be None. If both of them are not None, the bin_ids will be used. 

	:param wavelength_range:
		Range of wavelength within which the observed spectrum will be considered in the SED fitting. 
		The accepted format is [wmin,wmax] with wmin and wmax are minimum and maximum wavelengths.

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the SED fitting process using the function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of model spectra will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:
		Ranges of the parameter defined (i.e., outputted) using the :func:`params_ranges` in the :class:`priors` class.

	:param params_priors:
		Forms of adopted priros. The acceptable format is a list, such as params_priors=[prior1, prior2, prior3] where prior1, prior2, and prior3 are output of functions in the :class:`priors` class.

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: 'mcmc' and 'rdsps'.

	:param gal_z:
		Redshift of the galaxy. If gal_z=None, then redshift is taken from the header of the FITS file. 
		If gal_z in the FITS header is negative, the redshift is set to be free in the SED fitting (i.e., photometric redshift). 
	
	:param nrands_z:
		Number of random redshifts to be generated (within the chosen range as set in the params_range) in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.
		This is only relevant if gal_z=None (i.e., photometric redshift will be activated).

	:param add_igm_absorption:
		Switch for the IGM absorption. Options are: 0 for switch off and 1 for switch on.

	:param igm_type: 
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).
	
	:param spec_sigma: 
		Spectral resolution (in Angstrom) of the input spectrum.

	:param poly_order: 
		The degree of the legendre polynomial function to be used for correcting the shape (normalization) of the model spectra.

	:param likelihood: 
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: 'gauss' for the Gaussian form and 'student_t' for the student's t form.

	:param dof: 
		Degree of freedom (nu) in the Student's t likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: 
		Number of walkers in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: 
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: 
		Number of first steps of each walkers that will be cut when collecting the final sampler chains. Only relevant if fit_method='mcmc' and store_full_samplers=1.

	:param nproc: 
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: 
		Number of models to be used in the initial fitting in the MCMC method. Only relevant if fit_method='mcmc'.  

	:param perc_chi2:
		A percentile in the set of models sorted based on the chi-square values that will be considered in the calculation of the best-fit parameters (i.e., posterior-weighted averages) in the RDSPS fitting. 
		This parameter is only relevant if fit_method='rdsps'.
	
	:param spec_chi_sigma_clip: 
		Standard deviation (sigma) to be adopted in the sigma clipping to the spectrum data points that are regarded as outliers before 
		calculating chi-square in the SED fitting process. The sigma clipping is carried out based on the distribution of chi values (sum((D-M)/Derr)). 

	:param cosmo: 
		Choices for the cosmology. Options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: 
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: 
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param del_wave_nebem:
		This parameter defines the Wavelength region (+/- del_wave_nebem) around the emission lines that will be excluded in the fitting of spectral continuum between the model spectrum and the observed one.  

	:param store_full_samplers:
		Flag indicating whether full sampler models will be stored into the output FITS file or not. Options are: 1 or True for storing the full samplers and 0 or False otherwise.

	:param name_out_fits: 
		Names of output FITS files. This parameter is optional. If not None it must be in a list format with the same number of elements as the number of bins to be performed SED fitting. 
		Example: name_out_fits = ['bin1.fits', 'bin2.fits', ..., 'binN.fits']. If None, default names will be adopted.
	"""

	# open the input FITS file
	hdu = fits.open(fits_binmap)
	header = hdu[0].header
	if header['specphot']==0:
		print ("This function only fits spectrophotometric data cube!")
		sys.exit()
	# get number bins that have photometric and spectrophotometric data
	nbins_photo = int(header['nbinsph'])
	nbins_spec = int(header['nbinssp'])
	# get set of filters
	nbands = int(header['nfilters'])
	filters = []
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header[str_temp])
	# spatial bin maps
	binmap_photo = hdu['photo_bin_map'].data
	binmap_spec = hdu['spec_bin_map'].data
	# unit of flux
	unit = float(header['unit'])		# in erg/s/cm2/A
	# wavelength of the spectra
	spec_wave = hdu['spec_wave'].data
	nwaves = len(spec_wave)
	# allocate arrays for photometric and spectrophotometric SEDs of spatial bins
	bin_photo_flux = np.zeros((nbins_photo,nbands))
	bin_photo_flux_err = np.zeros((nbins_photo,nbands))
	bin_spec_flux = np.zeros((nbins_photo,nwaves))
	bin_spec_flux_err = np.zeros((nbins_photo,nwaves))
	for bb in range(0,nbins_photo):
		bin_id = bb + 1
		rows, cols = np.where(binmap_photo==bin_id)
		bin_photo_flux[bb] = hdu['bin_photo_flux'].data[:,rows[0],cols[0]]*unit
		bin_photo_flux_err[bb] = hdu['bin_photo_fluxerr'].data[:,rows[0],cols[0]]*unit
		rows, cols = np.where(binmap_spec==bin_id)
		if len(rows)>0:
			bin_spec_flux[bb] = hdu['bin_spec_flux'].data[:,rows[0],cols[0]]*unit
			bin_spec_flux_err[bb] = hdu['bin_spec_fluxerr'].data[:,rows[0],cols[0]]*unit
	hdu.close()

	# file of filter list
	name_filters_list = randname("filters_list",".dat")
	write_filters_list(name_filters_list,filters)
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc

	if params_ranges is None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors is None:
		params_priors = []

	if gal_z is None or gal_z<=0.0:
		gal_z = float(header['z'])
		if gal_z<=0.0:
			gal_z = -99.0
			free_z = 1
		else:
			free_z = 0
	else:
		free_z = 0

	# configuration file
	name_config = randname("config_file",".dat")
	flg_write = write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc,spec_sigma=spec_sigma,poly_order=poly_order,
					del_wave_nebem=del_wave_nebem,spec_chi_sigma_clip=spec_chi_sigma_clip)
	os.system('mv %s %s' % (name_config,temp_dir))

	nbins_calc = 0
	if bin_ids is not None:
		if binid_range is not None:
			print ("Both bin_ids and binid_range are not empty, so calculation will be done based on bin_ids.")
		bin_ids = np.asarray(bin_ids)
		nbins_calc = len(bin_ids)

	elif bin_ids is None:
		if binid_range is None:
			print ("Both bin_ids and binid_range are empty, so SED fitting will be done to all the bins.")
			bin_ids = np.arange(nbins_photo)
		elif binid_range is not None:
			binid_min = binid_range[0]
			binid_max = binid_range[1]
			bin_ids = np.arange(int(binid_min), int(binid_max))

	if name_out_fits is not None and 0<len(name_out_fits)<nbins_calc:
		print ("The number of elements in name_out_fits should be the same as the number of bins to be calculated!")
		sys.exit()

	count_id = 0
	for idx_bin in bin_ids:

		# name for output FITS file
		if name_out_fits is None:
			if fit_method=='mcmc' or fit_method=='MCMC':
				name_out_fits1 = "mcmc_bin%d.fits" % (idx_bin+1)
			elif fit_method=='rdsps' or fit_method=='RDSPS':
				name_out_fits1 = "rdsps_bin%d.fits" % (idx_bin+1)
		else:
			name_out_fits1 = name_out_fits[int(count_id)]

		# spectrophotometric
		if np.sum(bin_spec_flux[int(idx_bin)])>0:
			# input SED in HDF5
			inputSED_file = randname("inputSED_file",".hdf5")
			# cut spectrum if required
			if wavelength_range is not None:
				idx0 = np.where((spec_wave>=wavelength_range[0]) & (spec_wave<=wavelength_range[1]))
				spec_wave = spec_wave[idx0[0]]
				spec_flux = bin_spec_flux[int(idx_bin)][idx0[0]]
				spec_flux_err = bin_spec_flux_err[int(idx_bin)][idx0[0]]
			else:
				spec_flux = bin_spec_flux[int(idx_bin)]
				spec_flux_err = bin_spec_flux_err[int(idx_bin)]

			write_input_specphoto_hdf5(inputSED_file,bin_photo_flux[int(idx_bin)],bin_photo_flux_err[int(idx_bin)],
										spec_wave,spec_flux,spec_flux_err)
			os.system('mv %s %s' % (inputSED_file,temp_dir))

			if fit_method=='mcmc' or fit_method=='MCMC':
				name_samplers_hdf5 = randname("samplers_",".hdf5")

				os.system("mpirun -n %d python %s./mc_p1_sp.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,inputSED_file,name_samplers_hdf5))
				os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
				
				if store_full_samplers==1 or store_full_samplers==True:
					os.system("mpirun -n %d python %s./mc_p2_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
				elif store_full_samplers==0 or store_full_samplers==False:
					os.system("mpirun -n %d python %s./mc_nsmp_p2_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()
				os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))

			elif fit_method=='rdsps' or fit_method=='RDSPS':
				if store_full_samplers==1 or store_full_samplers==True:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits1))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits1))
				elif store_full_samplers==0 or store_full_samplers==False:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_nsmp_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits1))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_nsmp_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits1))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()
			else:
				print ("The input fit_method is not recognized!")
				sys.exit()

			os.system("rm %s%s" % (temp_dir,inputSED_file))

		# photometric SED
		else:
			# store input SED into a text file
			name_SED_txt = randname("inputSED_file",".dat")
			write_input_singleSED(name_SED_txt,bin_photo_flux[int(idx_bin)],bin_photo_flux_err[int(idx_bin)])
			os.system('mv %s %s' % (name_SED_txt,temp_dir))

			if fit_method=='mcmc' or fit_method=='MCMC':
				name_samplers_hdf5 = randname("samplers_",".hdf5")
				os.system("mpirun -n %d python %s./mc_p1.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,name_SED_txt,name_samplers_hdf5))
				os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
				if store_full_samplers==1 or store_full_samplers==True:
					os.system("mpirun -n %d python %s./mc_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
				elif store_full_samplers==0 or store_full_samplers==False:
					os.system("mpirun -n %d python %s./mc_nsmp_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

				count_id = count_id + 1
				os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))

			elif fit_method=='rdsps' or fit_method=='RDSPS':
				if store_full_samplers==1 or store_full_samplers==True:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
				elif store_full_samplers==0 or store_full_samplers==False:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_nsmp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_nsmp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

			os.system("rm %s%s" % (temp_dir,name_SED_txt))

		count_id = count_id + 1

	# free disk space:
	os.system("rm %s%s" % (temp_dir,name_filters_list))
	os.system("rm %s%s" % (temp_dir,name_config))
	if len(flg_write)>0:
		for ii in range(0,len(flg_write)):
			os.system("rm %s%s" % (temp_dir,flg_write[ii]))


def SEDfit_from_binmap(fits_binmap,binid_range=None,bin_ids=None,models_spec=None,params_ranges=None,params_priors=None,
	fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,likelihood='gauss',dof=3.0,nwalkers=100,
	nsteps=600,nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0,cosmo=0,H0=70.0,Om0=0.3,store_full_samplers=1,
	name_out_fits=None):

	"""Function for performing SED fitting to a photometric data cube. The data cube should has been binned using the function :func:`pixel_binning`.

	:param fits_binmap:
		Input FITS file of the spectrophotometric data cube that has been binned.

	:param binid_range:
		Range of bin IDs which the SEDs are going to be fit. The accepted format is [idmin,idmax]. The ID starts from 0. 
		If None, the SED fitting will be done to all of the spatial bins in the galaxy.

	:param bin_ids:
		Bin IDs which the SEDs are going to be fit. The accepted format is 1D array. 
		The ID starts from 0. Both binid_range and bin_ids can't be None. If both of them are not None, the bin_ids will be used. 

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the SED fitting process using the function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of model spectra will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:
		Ranges of the parameter defined (i.e., outputted) using the :func:`params_ranges` in the :class:`priors` class.

	:param params_priors:
		Forms of adopted priros. The acceptable format is a list, such as params_priors=[prior1, prior2, prior3] where prior1, prior2, and prior3 are output of functions in the :class:`priors` class.

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: 'mcmc' and 'rdsps'.

	:param gal_z:
		Redshift of the galaxy. If gal_z=None, then redshift is taken from the header of the FITS file. 
		If gal_z in the FITS header is negative, the redshift is set to be free in the SED fitting (i.e., photometric redshift). 
	
	:param nrands_z:
		Number of random redshifts to be generated (within the chosen range as set in the params_range) in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.
		This is only relevant if gal_z=None (i.e., photometric redshift will be activated).

	:param add_igm_absorption:
		Switch for the IGM absorption. Options are: 0 for switch off and 1 for switch on.

	:param igm_type: 
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).
	
	:param likelihood: 
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: 'gauss' for the Gaussian form and 'student_t' for the student's t form.

	:param dof: 
		Degree of freedom (nu) in the Student's t likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: 
		Number of walkers in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: 
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: 
		Number of first steps of each walkers that will be cut when collecting the final sampler chains. Only relevant if fit_method='mcmc' and store_full_samplers=1.

	:param nproc: 
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: 
		Number of models to be used in the initial fitting in the MCMC method. Only relevant if fit_method='mcmc'.  

	:param perc_chi2:
		A percentile in the set of models sorted based on the chi-square values that will be considered in the calculation of the best-fit parameters (i.e., posterior-weighted averages) in the RDSPS fitting. 
		This parameter is only relevant if fit_method='rdsps'.
	
	:param cosmo: 
		Choices for the cosmology. Options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: 
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: 
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param store_full_samplers:
		Flag indicating whether full sampler models will be stored into the output FITS file or not. Options are: 1 or True for storing the full samplers and 0 or False otherwise.

	:param name_out_fits: 
		Names of output FITS files. This parameter is optional. If not None it must be in a list format with the same number of elements as the number of bins to be performed SED fitting. 
		Example: name_out_fits = ['bin1.fits', 'bin2.fits', ..., 'binN.fits']. If None, default names will be adopted.
	"""

	# open pixel binning maps
	hdu = fits.open(fits_binmap)
	header = hdu[0].header
	bin_map = hdu['bin_map'].data
	bin_flux = hdu['bin_flux'].data
	bin_flux_err = hdu['bin_fluxerr'].data
	hdu.close()
	unit = float(header['unit'])
	# transpose from (wave,y,x) => (y,x,wave)
	bin_flux_trans = np.transpose(bin_flux, axes=(1,2,0))*unit
	bin_flux_err_trans = np.transpose(bin_flux_err, axes=(1,2,0))*unit
	# get filters
	nbands = int(header['nfilters'])
	filters = []
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb 
		filters.append(header[str_temp])

	# number of bins:
	nbins = int(header['nbins'])

	# file of filter list
	name_filters_list = randname("filters_list",".dat")
	write_filters_list(name_filters_list,filters)
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc

	if params_ranges is None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors is None:
		params_priors = []

	if gal_z is None or gal_z<=0.0:
		gal_z = float(header['z'])
		if gal_z<=0.0:
			gal_z = -99.0
			free_z = 1
		else:
			free_z = 0
	else:
		free_z = 0

	# configuration file
	name_config = randname("config_file",".dat")
	flg_write = write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc)
	os.system('mv %s %s' % (name_config,temp_dir))

	nbins_calc = 0
	if bin_ids is not None:
		if len(binid_range) > 0:
			print ("Both bin_ids and binid_range are not empty, calculation will be done based on bin_ids.")
		bin_ids = np.asarray(bin_ids)
		nbins_calc = len(bin_ids)

	elif bin_ids is None:
		if binid_range is None:
			print ("Both bin_ids and binid_range are empty, SED fitting will be done to all the bins.")
			bin_ids = np.arange(nbins)
		elif binid_range is not None:
			binid_min = binid_range[0]
			binid_max = binid_range[1]
			bin_ids = np.arange(int(binid_min), int(binid_max))

	if name_out_fits is not None and 0<len(name_out_fits)<nbins_calc:
		print ("The number of elements in name_out_fits should be the same as the number of bins to be calculated!")
		sys.exit()

	if fit_method=='mcmc' or fit_method=='MCMC':
		count_id = 0
		for idx_bin in bin_ids:
			# SED of bin
			rows, cols = np.where(bin_map==idx_bin+1)

			# input SED text file
			name_SED_txt = randname("inputSED_file",".dat")
			write_input_singleSED(name_SED_txt,bin_flux_trans[rows[0]][cols[0]],bin_flux_err_trans[rows[0]][cols[0]])
			os.system('mv %s %s' % (name_SED_txt,temp_dir))

			# name of output FITS file
			if name_out_fits is None:
				name_out_fits1 = "mcmc_bin%d.fits" % (idx_bin+1)
			else:
				name_out_fits1 = name_out_fits[int(count_id)]

			name_samplers_hdf5 = randname("samplers_",".hdf5")

			os.system("mpirun -n %d python %s./mc_p1.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,name_SED_txt,name_samplers_hdf5))
			os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
			
			if store_full_samplers==1 or store_full_samplers==True:
				os.system("mpirun -n %d python %s./mc_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
			elif store_full_samplers==0 or store_full_samplers==False:
				os.system("mpirun -n %d python %s./mc_nsmp_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
			else:
				print ("Input store_full_samplers not recognized!")
				sys.exit()

			count_id = count_id + 1

			os.system("rm %s%s" % (temp_dir,name_SED_txt))
			os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))

	elif fit_method=='rdsps' or fit_method=='RDSPS':
		# single bin
		if len(bin_ids) == 1:
			rows, cols = np.where(bin_map==int(bin_ids[0])+1)

			name_SED_txt = randname("inputSED_file",".dat")
			write_input_singleSED(name_SED_txt,bin_flux_trans[rows[0]][cols[0]],bin_flux_err_trans[rows[0]][cols[0]])
			os.system('mv %s %s' % (name_SED_txt,temp_dir))

			# output file names:
			if name_out_fits is None:
				name_out_fits1 = "rdsps_bin%d.fits" % (bin_ids[0]+1)
			else:
				name_out_fits1 = name_out_fits[0]

			if store_full_samplers==1 or store_full_samplers==True:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
			elif store_full_samplers==0 or store_full_samplers==False:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz_nsmp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_nsmp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,name_out_fits1))
			else:
				print ("Input store_full_samplers not recognized!")
				sys.exit()

			os.system("rm %s%s" % (temp_dir,name_SED_txt))

		# multiple bins
		else:
			# input SEDs: store to hdf5 file
			name_inputSEDs = randname("input_SEDs_",".hdf5")
			with h5py.File(name_inputSEDs, 'w') as f:
				m = f.create_group('obs_seds')
				m.attrs['nbins_calc'] = len(bin_ids)

				fl = m.create_group('flux')
				fle = m.create_group('flux_err')
				for ii in range(0,len(bin_ids)):
					rows, cols = np.where(bin_map==bin_ids[ii]+1)
					str_temp = 'b%d_f' % ii
					fl.create_dataset(str_temp, data=np.array(bin_flux_trans[rows[0]][cols[0]]))

					str_temp = 'b%d_ferr' % ii
					fle.create_dataset(str_temp, data=np.array(bin_flux_err_trans[rows[0]][cols[0]]))
			os.system('mv %s %s' % (name_inputSEDs,temp_dir))

			# name outputs:
			if name_out_fits is None:
				name_outs = randname("name_outs",".dat")
				file_out = open(name_outs,"w")
				for idx_bin in bin_ids:
					name0 = "rdsps_bin%d.fits" % (idx_bin+1)
					file_out.write("%s\n" % name0)
				file_out.close()
			else:
				name_outs = randname("name_outs",".dat")
				file_out = open(name_outs,"w")
				for zz in range(0,len(name_out_fits)):
					file_out.write("%s\n" % name_out_fits[zz])
				file_out.close()
			os.system('mv %s %s' % (name_outs,temp_dir))

			if store_full_samplers==1 or store_full_samplers==True:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																					name_inputSEDs,name_outs))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																					name_inputSEDs,name_outs))

			elif store_full_samplers==0 or store_full_samplers==False:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz_nsmp_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_inputSEDs,name_outs))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_nsmp_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_inputSEDs,name_outs))
			else:
				print ("Input store_full_samplers not recognized!")
				sys.exit()
			os.system("rm %s%s" % (temp_dir,name_inputSEDs))
			os.system("rm %s%s" % (temp_dir,name_outs))
	else:
		print ("Input fit_method is not recognized!")
		sys.exit()

	# free disk space:
	os.system("rm %s%s" % (temp_dir,name_filters_list))
	os.system("rm %s%s" % (temp_dir,name_config))
	if len(flg_write)>0:
		for ii in range(0,len(flg_write)):
			os.system("rm %s%s" % (temp_dir,flg_write[ii]))


def SEDfit_pixels_from_fluxmap(fits_fluxmap,x_range=None,y_range=None,models_spec=None,params_ranges=None,params_priors=None,
	fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,likelihood='gauss',dof=2.0,nwalkers=100,nsteps=600,
	nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=10.0,cosmo=0,H0=70.0,Om0=0.3,store_full_samplers=1):

	"""Function for performing SED fitting to a photometric data cube on the pixel-by-pixel basis. 

	:param fits_fluxmap:
		Input FITS file of photometric data cube. It is the ouput (or should have the same format as that of the output) 
		of the image processing by :func:`images_processing` function.

	:param x_range:
		Range of x-axis coordinate within which SED fitting will be performed to the pixels. The format is [xmin,xmax]. 
		If x_range=None, the whole x-axis range of the data cube is considered.  

	:param y_range:
		Range of y-axis coordinate within which SED fitting will be performed to the pixels. The format is [ymin,ymax]. 
		If y_range=None, the whole y-axis range of the data cube is considered.

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the SED fitting process using the function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of model spectra will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:
		Ranges of the parameter defined (i.e., outputted) using the :func:`params_ranges` in the :class:`priors` class.

	:param params_priors:
		Forms of adopted priros. The acceptable format is a list, such as params_priors=[prior1, prior2, prior3] where prior1, prior2, and prior3 are output of functions in the :class:`priors` class.

	:param fit_method:
		Choice for the fitting method. Options are: 'mcmc' and 'rdsps'.

	:param gal_z:
		Redshift of the galaxy. If gal_z=None, then redshift is taken from the header of the FITS file. 
		If gal_z in the FITS header is negative, the redshift is set to be free in the SED fitting (i.e., photometric redshift). 
	
	:param nrands_z:
		Number of random redshifts to be generated (within the chosen range as set in the params_range) in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.
		This is only relevant if gal_z=None (i.e., photometric redshift will be activated).

	:param add_igm_absorption:
		Switch for the IGM absorption. Options are: 0 for switch off and 1 for switch on.

	:param igm_type: 
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).

	:param likelihood: 
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: 'gauss' for the Gaussian form and 'student_t' for the student's t form.

	:param dof: 
		Degree of freedom (nu) in the Student's t likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: 
		Number of walkers in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: 
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: 
		Number of first steps of each walkers that will be cut when collecting the final sampler chains. Only relevant if fit_method='mcmc' and store_full_samplers=1.

	:param nproc: 
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: 
		Number of models to be used in the initial fitting in the MCMC method. Only relevant if fit_method='mcmc'.  

	:param perc_chi2:
		A percentile in the set of models sorted based on the chi-square values that will be considered in the calculation of the best-fit parameters (i.e., posterior-weighted averages) in the RDSPS fitting. 
		This parameter is only relevant if fit_method='rdsps'.
	
	:param cosmo: 
		Choices for the cosmology. Options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: 
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: 
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param store_full_samplers:
		Flag indicating whether full sampler models will be stored into the output FITS file or not. Options are: 1 or True for storing the full samplers and 0 or False otherwise.
	"""

	# open the input FITS file
	hdu = fits.open(fits_fluxmap)
	header = hdu[0].header
	gal_region = hdu['galaxy_region'].data 
	map_flux = hdu['flux'].data 
	map_flux_err = hdu['flux_err'].data 
	hdu.close()
	unit = float(header['unit'])
	# transpose from (wave,y,x) => (y,x,wave)
	map_flux_trans = np.transpose(map_flux, axes=(1,2,0))*unit
	map_flux_err_trans = np.transpose(map_flux_err, axes=(1,2,0))*unit
	dim_y = gal_region.shape[0]
	dim_x = gal_region.shape[1]
	
	nbands = int(header['nfilters'])
	filters = []
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header[str_temp])
	
	name_filters_list = randname("filters_list",".dat")
	write_filters_list(name_filters_list,filters)
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	# Define x- and y-ranges
	if x_range is None:
		xmin, xmax = 0, dim_x
	else:
		if x_range[0]<0 or x_range[1]>dim_x:
			print ("Can't perform SED fitting to region beyond the region of the data cube!")
			sys.exit()
		else:
			xmin, xmax = x_range[0], x_range[1] 

	if y_range is None:
		ymin, ymax = 0, dim_y
	else:
		if y_range[0]<0 or y_range[1]>dim_y:
			print ("Can't perform SED fitting to region beyond the region of the data cube!")
			sys.exit()
		else:
			ymin, ymax = y_range[0], y_range[1]

	if gal_z is None or gal_z<=0.0:
		gal_z = float(header['z'])
		if gal_z<=0.0:
			gal_z = -99.0
			free_z = 1
		else:
			free_z = 0
	else:
		free_z = 0

	# configuration file
	name_config = randname("config_file",".dat")
	flg_write = write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc,spec_sigma=spec_sigma,poly_order=poly_order,
					del_wave_nebem=del_wave_nebem,spec_chi_sigma_clip=spec_chi_sigma_clip)
	os.system('mv %s %s' % (name_config,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc
	
	#==> fitting process
	if fit_method=='mcmc' or fit_method=='MCMC':
		for yy in range(ymin,ymax):
			for xx in range(xmin,xmax):
				if gal_region[yy][xx] == 1:
					obs_flux = map_flux_trans[yy][xx]
					obs_flux_err = map_flux_err_trans[yy][xx]

					# input SED text file
					name_SED_txt = randname("inputSED_file",".dat")
					write_input_singleSED(name_SED_txt,obs_flux,obs_flux_err)
					os.system('mv %s %s' % (name_SED_txt,temp_dir))

					# name of output FITS file
					name_out_fits = "pix_y%d_x%d_mcmc.fits" % (yy,xx)
					name_samplers_hdf5 = randname("samplers_",".hdf5")
					os.system("mpirun -n %d python %s./mc_p1.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																				name_SED_txt,name_samplers_hdf5))
					os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
					
					if store_full_samplers==1 or store_full_samplers==True:
						os.system("mpirun -n %d python %s./mc_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																					name_samplers_hdf5,name_out_fits))
					elif store_full_samplers==0 or store_full_samplers==False:
						os.system("mpirun -n %d python %s./mc_nsmp_p2.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_samplers_hdf5,name_out_fits))
					else:
						print ("Input store_full_samplers not recognized!")
						sys.exit()

					os.system('rm %s%s' % (temp_dir,name_SED_txt))
					os.system('rm %s%s' % (temp_dir,name_samplers_hdf5))

	elif fit_method=='rdsps' or fit_method=='RDSPS':
		name_outs = randname("name_outs_",".dat")
		file_out = open(name_outs,"w")
		# input SEDs: store to hdf5 file
		name_inputSEDs = randname("input_SEDs_",".hdf5")
		with h5py.File(name_inputSEDs, 'w') as f:
			m = f.create_group('obs_seds')
			fl = m.create_group('flux')
			fle = m.create_group('flux_err')
			nbins_calc = 0
			for yy in range(ymin,ymax):
				for xx in range(xmin,xmax):
					if gal_region[yy][xx] == 1:
						obs_flux = map_flux_trans[yy][xx]
						obs_flux_err = map_flux_err_trans[yy][xx]
						str_temp = 'b%d_f' % nbins_calc
						fl.create_dataset(str_temp, data=np.array(obs_flux))
						str_temp = 'b%d_ferr' % nbins_calc
						fle.create_dataset(str_temp, data=np.array(obs_flux_err))
						# name output
						name0 = "pix_y%d_x%d_rdsps.fits" % (yy,xx)
						file_out.write("%s\n" % name0)
						nbins_calc = nbins_calc + 1
			m.attrs['nbins_calc'] = nbins_calc
		file_out.close()

		os.system('mv %s %s' % (name_inputSEDs,temp_dir))
		os.system('mv %s %s' % (name_outs,temp_dir))

		if store_full_samplers==1 or store_full_samplers==True:
			if free_z==0:
				os.system("mpirun -n %d python %s./rd_fz_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_inputSEDs,name_outs))
			elif free_z==1:
				os.system("mpirun -n %d python %s./rd_vz_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_inputSEDs,name_outs))
		elif store_full_samplers==0 or store_full_samplers==False:
			if free_z==0:
				os.system("mpirun -n %d python %s./rd_fz_nsmp_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																					name_inputSEDs,name_outs))
			elif free_z==1:
				os.system("mpirun -n %d python %s./rd_vz_nsmp_blk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																					name_inputSEDs,name_outs))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()

		os.system("rm %s%s" % (temp_dir,name_inputSEDs))
		os.system("rm %s%s" % (temp_dir,name_outs))

	else:
		print ("Input fit_method is not recognized!")
		sys.exit()

	# free disk space:
	os.system("rm %s%s" % (temp_dir,name_filters_list))
	os.system("rm %s%s" % (temp_dir,name_config))
	if len(flg_write)>0:
		for ii in range(0,len(flg_write)):
			os.system("rm %s%s" % (temp_dir,flg_write[ii]))


def maps_parameters(fits_binmap, bin_ids, name_sampler_fits, fits_fluxmap=None, refband_SFR=None, refband_SM=None, 
	refband_Mdust=None, name_out_fits=None):
	"""Function for constructing maps of properties of a galaxy from the collectin of fitting results of the spatial bins within the galaxy.

	:param fits_binmap:
		Input FITS file of the spectrophotometric data cube that has been binned.

	:param bin_ids:
		Bin indices of the FITS files listed in name_sampler_fits input. Allowed format is a 1D array. The id starts from 0. 

	:param name_sampler_fits:
		List of the names of the FITS files containing the fitting results of spatial bins. This should have the same number of element as that of bin_ids. 
		The number of element doesn't necessarily the same as the number of bins. A missing bin will be ignored in the creation of the maps of properties.

	:param fits_fluxmap:
		FITS file containing reduced maps of multiband fluxes, which is output of the :func:`flux_map` in the :class:`images_processing` class in the :mod:`piXedfit_images` module.

	:param refband_SFR:
		Index of band in the multiband set that is used for reference in dividing map of SFR in bin space into map of SFR in pixel space.
		If None, the band with shortest wavelength is selected.

	:param refband_SM:
		Index of band in the multiband set that is used for reference in dividing map of stellar mass in bin space into map of stellar mass in pixel space.
		If None, the band with longest wavelength is selected.

	:param refband_Mdust:
		Index of band/filter in the multiband set that is used for reference in dividing map of dust mass in bin space into map of dust mass in pixel space.
		If None, the band with longest wavelength is selected.

	:param name_out_fits: 
		Desired name for the output FITS file. If None, a default name will be used.  
	"""

	# open the FITS file containing the pixel binning map
	hdu = fits.open(fits_binmap)
	unit_bin = float(hdu[0].header['unit'])
	if hdu[0].header['SPECPHOT'] == 0:
		nbins = int(hdu[0].header['nbins'])
		binmap = hdu['bin_map'].data
		bin_flux = hdu['bin_flux'].data*unit_bin
		bin_flux_err = hdu['bin_fluxerr'].data*unit_bin
	elif hdu[0].header['SPECPHOT'] == 1:
		nbins = int(hdu[0].header['NBINSPH'])
		binmap = hdu['PHOTO_BIN_MAP'].data
		bin_flux = hdu['BIN_PHOTO_FLUX'].data*unit_bin
		bin_flux_err = hdu['BIN_PHOTO_FLUXERR'].data*unit_bin
	gal_z = float(hdu[0].header['z'])
	nbands = int(hdu[0].header['nfilters'])
	hdu.close()

	dim_y = binmap.shape[0]
	dim_x = binmap.shape[1]

	# flag for excluded bins
	name_sampler_fits1 = []
	for bb in range(0,nbins):
		name_sampler_fits1.append("temp")
	bin_excld_flag = np.zeros(nbins) + 1
	for bb in range(0,len(bin_ids)):
		bin_excld_flag[int(bin_ids[bb])] = 0
		name_sampler_fits1[int(bin_ids[bb])] = name_sampler_fits[bb]

	# open FITS file containing multiband fluxes maps
	hdu = fits.open(fits_fluxmap)
	unit_pix = float(hdu[0].header['unit'])
	if hdu[0].header['SPECPHOT'] == 0:
		galaxy_region = hdu['galaxy_region'].data
		pix_flux = hdu['flux'].data*unit_pix
		pix_flux_err = hdu['flux_err'].data*unit_pix
	elif hdu[0].header['SPECPHOT'] == 1:
		galaxy_region = hdu['PHOTO_REGION'].data
		pix_flux = hdu['PHOTO_FLUX'].data*unit_pix
		pix_flux_err = hdu['PHOTO_FLUXERR'].data*unit_pix
	hdu.close()

	# number of files
	nfiles = len(name_sampler_fits)

	#=> get some information
	hdu = fits.open(name_sampler_fits[nfiles-1])
	fit_method = hdu[0].header['fitmethod']
	# list of parameters
	params = []
	for pp in range(0,int(hdu[0].header['nparams'])):
		params.append(hdu[0].header['param%d' % pp])
	if fit_method == 'mcmc':
		params.append('log_sfr')
		params.append('log_mw_age')
		if int(hdu[0].header['duste_stat'])==1:
			params.append('log_dustmass')
		if int(hdu[0].header['add_agn'])==1:
			params.append('log_fagn_bol')
	nparams = len(params)
	# indices
	if fit_method == 'rdsps':
		indexes = ["mean", "mean_err"]
	elif fit_method == 'mcmc':
		indexes = ["p16","p50","p84"]
	nindexes = len(indexes)
	hdu.close()

	# allocate memory
	bfit_param = {}
	for pp in range(0,nparams):
		bfit_param[params[pp]] = {}
		for ii in range(0,nindexes):
			bfit_param[params[pp]][indexes[ii]] = np.zeros(nbins) - 99.0

	# get bfit_param in bin space
	for ii in range(0,nbins):
		if bin_excld_flag[ii] != 1:
			hdu = fits.open(name_sampler_fits1[ii])
			for pp in range(0,nparams):
				if fit_method == 'rdsps':
					bfit_param[params[pp]]["mean"][ii] = hdu['fit_params'].data[params[pp]][0]
					bfit_param[params[pp]]["mean_err"][ii] = hdu['fit_params'].data[params[pp]][1]
				elif fit_method == 'mcmc':
					bfit_param[params[pp]]["p16"][ii] = hdu['fit_params'].data[params[pp]][0]
					bfit_param[params[pp]]["p50"][ii] = hdu['fit_params'].data[params[pp]][1]
					bfit_param[params[pp]]["p84"][ii] = hdu['fit_params'].data[params[pp]][2]
			hdu.close()

	#=> Store to FITS file
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['gal_z'] = gal_z
	hdr['nbins'] = nbins
	count_HDU = 2
	# bin space
	for pp in range(0,nparams):
		for ii in range(0,nindexes):
			idx_str = "bin-%s-%s" % (params[pp],indexes[ii])
			hdr['HDU%d' % int(count_HDU)] = idx_str
			count_HDU = count_HDU + 1
	# pixel space
	params_pix = []
	for pp in range(0,nparams):
		if params[pp] == 'log_sfr' or params[pp] == 'log_mass' or params[pp]=='log_dustmass':
			params_pix.append(params[pp])
			for ii in range(0,nindexes):
				idx_str = "pix-%s-%s" % (params[pp],indexes[ii])
				hdr['HDU%d' % int(count_HDU)] = idx_str
				count_HDU = count_HDU + 1
	hdr['nHDU'] = count_HDU
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)

	# get number of parameters to be distributed to pixel space
	nparams_pix = len(params_pix)

	hdul.append(fits.ImageHDU(galaxy_region, name='galaxy_region'))

	# maps in bin space
	for pp in range(0,nparams):
		for ii in range(0,nindexes):
			idx_str = "bin-%s-%s" % (params[pp],indexes[ii])
			map_prop = np.zeros((dim_y,dim_x))
			for bb in range(0,nbins):
				rows, cols = np.where(binmap==bb+1)
				map_prop[rows,cols] = bfit_param[params[pp]][indexes[ii]][bb]
			hdul.append(fits.ImageHDU(map_prop, name=idx_str))

	# reference bands:
	if refband_SFR is None:
		refband_SFR = 0
	if refband_SM is None:
		refband_SM = nbands-1
	if refband_Mdust is None:
		refband_Mdust = nbands-1

	refband_params_pix = {}
	refband_params_pix['log_sfr'] = refband_SFR
	refband_params_pix['log_mass'] = refband_SM
	for pp in range(0,nparams_pix):
		if params_pix[pp] == 'log_dustmass':
			refband_params_pix['log_dustmass'] = refband_Mdust

	# maps in pixel space
	for pp in range(0,nparams_pix):
		for ii in range(0,nindexes):
			idx_str = "pix-%s-%s" % (params_pix[pp],indexes[ii])
			map_prop = np.zeros((dim_y,dim_x))

			if fit_method == 'rdsps':
				if indexes[ii] == 'mean':
					for bb in range(0,nbins):
						rows, cols = np.where(binmap==bb+1)
						bin_val = np.power(10.0,bfit_param[params_pix[pp]][indexes[ii]][bb])
						pix_val = bin_val*pix_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]/bin_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
						map_prop[rows,cols] = np.log10(pix_val)
					hdul.append(fits.ImageHDU(map_prop, name=idx_str))
				else:
					for bb in range(0,nbins):
						rows, cols = np.where(binmap==bb+1)
						bin_val = np.power(10.0,bfit_param[params_pix[pp]]["mean"][bb])
						bin_valerr = np.power(10.0,bfit_param[params_pix[pp]][indexes[ii]][bb])
						bin_f = bin_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
						bin_ferr =  bin_flux_err[int(refband_params_pix[params_pix[pp]])][rows,cols]
						pix_f = pix_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
						pix_ferr = pix_flux_err[int(refband_params_pix[params_pix[pp]])][rows,cols]
						pix_val = bin_val*np.sqrt((bin_val*bin_val/bin_valerr/bin_valerr) + (bin_f*bin_f/bin_ferr/bin_ferr) + (pix_f*pix_f/pix_ferr/pix_ferr))
						map_prop[rows,cols] = np.log10(pix_val)
					hdul.append(fits.ImageHDU(map_prop, name=idx_str))

			elif fit_method == 'mcmc':
				for bb in range(0,nbins):
					rows, cols = np.where(binmap==bb+1)
					bin_val = np.power(10.0,bfit_param[params_pix[pp]][indexes[ii]][bb])
					pix_val = bin_val*pix_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]/bin_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
					map_prop[rows,cols] = np.log10(pix_val)
				hdul.append(fits.ImageHDU(map_prop, name=idx_str))
					
	if name_out_fits is None:
		name_out_fits = "fitres_%s" % fits_binmap
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def maps_parameters_fit_pixels(fits_fluxmap, pix_x, pix_y, name_sampler_fits, name_out_fits=None):
	"""Function for calculating maps of properties from the fitting results obtained with the MCMC method on pixel basis.

	:param fits_fluxmap: 
		FITS file containing reduced maps of multiband fluxes, which is output of the :func:`flux_map` in the :class:`images_processing` class in the :mod:`piXedfit_images` module.

	:param pix_x:
		x coordinates of pixels associated with the FITS files (fitting results) in name_sampler_fits input.

	:param pix_y:
		y coordinates of pixels associated with the FITS files (fitting results) in name_sampler_fits input.

	:param name_sampler_fits: 
		List of names of the FITS files containing the fitting results. 
		The three inputs of pix_x, pix_y, and name_sampler_fits should have the same number of elements.

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
	"""

	npixs = len(name_sampler_fits)

	if len(pix_x)!=npixs or len(pix_y)!=npixs:
		print ("pix_x and pix_y should have the same number of elements as name_sampler_fits!")
		sys.exit()

	# open FITS file containing maps of multiband fluxes
	hdu = fits.open(fits_fluxmap)
	galaxy_region = hdu['galaxy_region'].data
	gal_z = float(hdu[0].header['z'])
	hdu.close()

	dim_y = galaxy_region.shape[0]
	dim_x = galaxy_region.shape[1]

	#=> get some information
	hdu = fits.open(name_sampler_fits[npixs-1])
	fit_method = hdu[0].header['fitmethod']
	# list of parameters
	params = []
	for pp in range(0,int(hdu[0].header['nparams'])):
		params.append(hdu[0].header['param%d' % pp])
	if fit_method == 'mcmc':
		params.append('log_sfr')
		params.append('log_mw_age')
		if int(hdu[0].header['duste_stat'])==1:
			params.append('log_dustmass')
		if int(hdu[0].header['add_agn'])==1:
			params.append('log_fagn_bol')
	nparams = len(params)
	# indices
	if fit_method == 'rdsps':
		indexes = ["mean", "mean_err"]
	elif fit_method == 'mcmc':
		indexes = ["p16","p50","p84"]
	nindexes = len(indexes)
	hdu.close()

	# allocate memory
	bfit_param = {}
	for pp in range(0,nparams):
		bfit_param[params[pp]] = {}
		for ii in range(0,nindexes):
			bfit_param[params[pp]][indexes[ii]] = np.zeros(nbins) - 99.0

	# get bfit_param
	for ii in range(0,npixs):
		hdu = fits.open(name_sampler_fits[ii])
		for pp in range(0,nparams):
			if fit_method == 'rdsps':
				bfit_param[params[pp]]["mean"][ii] = hdu['fit_params'].data[params[pp]][0]
				bfit_param[params[pp]]["mean_err"][ii] = hdu['fit_params'].data[params[pp]][1]
			elif fit_method == 'mcmc':
					bfit_param[params[pp]]["p16"][ii] = hdu['fit_params'].data[params[pp]][0]
					bfit_param[params[pp]]["p50"][ii] = hdu['fit_params'].data[params[pp]][1]
					bfit_param[params[pp]]["p84"][ii] = hdu['fit_params'].data[params[pp]][2]
		hdu.close()

	# store the maps to FITS file
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['gal_z'] = gal_z
	count_HDU = 2
	# pixel space
	for pp in range(0,nparams):
		for ii in range(0,nindexes):
			idx_str = "pix-%s-%s" % (params[pp],indexes[ii])
			hdr['HDU%d' % int(count_HDU)] = idx_str
			count_HDU = count_HDU + 1
	hdr['nHDU'] = count_HDU
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)

	# galaxy's region
	hdul.append(fits.ImageHDU(galaxy_region, name='galaxy_region'))

	# maps of properties
	for pp in range(0,nparams):
		for ii in range(0,nindexes):
			idx_str = "pix-%s-%s" % (params[pp],indexes[ii])
			map_prop = np.zeros((dim_y,dim_x)) - 99.0
			for jj in range(0,npixs):
				map_prop[int(pix_y[jj])][int(pix_x[jj])] = bfit_param[params[pp]][indexes[ii]][jj]
			hdul.append(fits.ImageHDU(map_prop, name=idx_str))
					
	if name_out_fits is None:
		name_out_fits = "fitres_%s" % fits_binmap
	hdul.writeto(name_out_fits, overwrite=True)


def get_params(free_z, sfh_form, duste_switch, dust_law, add_agn):

	params = ['logzsol', 'log_tau']
	# SFH
	if sfh_form==2 or sfh_form==3:
		params.append('log_t0')
	elif sfh_form==4:
		params.append('log_alpha')
		params.append('log_beta')
	params.append('log_age')

	# dust attenuation
	if dust_law==0:
		params.append('dust_index')
		params.append('dust1')
	params.append('dust2')

	# dust emission
	if duste_switch==1:
		params.append('log_gamma')
		params.append('log_umin')
		params.append('log_qpah')

	# AGN:
	if add_agn == 1:
		params.append('log_fagn')
		params.append('log_tauagn')

	# redshift
	if free_z == 1:
		params.append('z')

	params.append('log_mass')

	nparams = len(params)

	return params, nparams


