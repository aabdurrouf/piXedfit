import numpy as np
from math import pi, pow, sqrt, cos, sin
import sys, os
import h5py
from random import randint
from astropy.io import fits
from .fitutils import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

os.environ["OMP_NUM_THREADS"] = "1"

__all__ = ["singleSEDfit", "singleSEDfit_specphoto", "SEDfit_from_binmap", "SEDfit_from_binmap_specphoto", 
			"SEDfit_pixels_from_fluxmap", "maps_parameters", "maps_parameters_fit_pixels", "get_params", "priors"]

CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
temp_dir = PIXEDFIT_HOME+'/data/temp/'


class priors:
	"""Functions for defining priors.
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
		"""
		return self.ranges

	def uniform(self, param):
		"""Function for assigning uniform prior to a parameter.

		:param param:
			The parameter that will be assigned the prior.

		:returns prior:
		"""
		prior = [param, "uniform"]
		return prior 

	def gaussian(self, param, loc, scale):
		"""Function for assigning gaussian prior to a parameter.

		:param param:
			The parameter that will be assigned the prior.

		:param loc:

		:param scale:

		:returns prior:
		"""
		prior = [param, "gaussian", loc, scale]
		return prior

	def studentt(self, param, df, loc, scale):
		"""Function for assigning Student's t prior to a parameter.

		:param param:
			The parameter that will be assigned the prior.

		:param df:

		:param loc:

		:param scale:

		:returns prior:
		"""
		prior = [param, "studentt", df, loc, scale]
		return prior

	def gamma(self, param, a, loc, scale):
		"""Function for assigning a prior in the form of Gamma function to a parameter.

		:param param:
			The parameter that will be assigned the prior.

		:param a:

		:param loc:

		:param scale:

		:returns prior:
		"""
		prior = [param, "gamma", a, loc, scale]
		return prior

	def arbitrary(self, param, values, prob):
		"""Function for assigning an arbitrary prior.

		:param param:

		:param values:

		:param prob:

		"""
		namepr = randname("arbtprior",".dat")
		write_arbitprior(namepr,values,prob)
		os.system('mv %s %s' % (namepr,temp_dir))
		prior = [param, "arbitrary", namepr]
		return prior

	def joint_with_mass(self, param, log_mass, param_values, scale):
		"""Function for assigning a joint prior between a parameter and log_mass

		:param param:

		:param log_mass:

		:param param_values:

		:param scale:
		"""
		namepr = randname("jprmass",".dat")
		write_joint_prior(namepr,log_mass,param_values)
		os.system('mv %s %s' % (namepr,temp_dir))
		prior = [param, "joint_with_mass", namepr, scale]
		return prior



def singleSEDfit(obs_flux,obs_flux_err,filters,models_spec,params_ranges=None,params_priors=None,fit_method='mcmc',
	gal_z=None,nrands_z=10,add_igm_absorption=0, igm_type=0,likelihood='gauss',dof=2.0,nwalkers=100,nsteps=600,nsteps_cut=50,
	nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0,cosmo=0,H0=70.0,Om0=0.3,store_full_samplers=1,name_out_fits=None):
	"""Function for performing fitting to a single photometric SED.

	:param obs_flux: (1D array; float)
		Input fluxes in multiple bands. The number of elements of the array should be the sama as that of obs_flux_err and filters.

	:param obs_flux_err: (1D array; float)
		Input flux uncertainties in multiple bands. 

	:param filters: (list; string)
		List of photometric filters. The list of filters recognized by piXedfit can be accesses using :func:`piXedfit.utils.filtering.list_filters`.

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the fitting using :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of models will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:

	:param params_priors:
	
	:param fit_method: (default: 'mcmc')
		Method in SED fitting. Options are: (a)'mcmc' for Markov Chain Monte Carlo, and (b)'rdsps' for Random Dense Sampling of Parameter Space.

	:param gal_z: 
		Redshift of the galaxy. If gal_z=None, then redshift is set to be a free parameter in the fitting.  
		
	:param params_range:
		Ranges of parameters. The format of this input argument is python dictionary. 
		The range for redshift ('z') only relevant if gal_z=None (i.e., redshift becomes free parameter). 

	:param nrands_z: (default: 10)
		Number of random redshifts to be generated (within the chosen range as set in the params_range) in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.
		This is only relevant if gal_z=None (i.e., photometric redshift is activated).

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption. Options are: (a)0 for switch off, and (b)1 for switch on.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: (default: 30000)
		Number of models to be used for initial fitting in the MCMC method.  

	:param perc_chi2: (default: 90.0)
		Lowest chi-square Percentage from the full model SEDs that are considered in the calculation of posterior-weighted averaging. 
		This parameter is only relevant for the RDSPS fitting method and it is not applicable for the MCMC method.

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param store_full_samplers: (default: 1 or True)
		Flag indicating whether full sampler models will be stored into the output FITS file or not. 
		Options are: (a) 1 or True and (b) 0 or False.

	:param name_out_fits: (optional, default: None)
		Name of output FITS file. This parameter is optional. 
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

	if params_ranges==None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors==None:
		params_priors = []

	if gal_z==None or gal_z<=0.0:
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
	if name_out_fits == None:
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


def singleSEDfit_specphoto(obs_flux,obs_flux_err,filters,spec_wave,spec_flux,spec_flux_err,models_spec,params_ranges=None,
	params_priors=None,fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,spec_sigma=2.6,poly_order=8, 
	likelihood='gauss',dof=2.0,nwalkers=100,nsteps=600,nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0, 
	spec_chi_sigma_clip=3.0,cosmo=0,H0=70.0,Om0=0.3,del_wave_nebem=10.0,store_full_samplers=1,name_out_fits=None):
	"""Function for performing SED fitting to a single photometric SED.

	:param obs_flux: (1D array; float)
		Input fluxes in multiple bands. The number of elements of the array should be the sama as that of obs_flux_err and filters.

	:param obs_flux_err: (1D array; float)
		Input flux uncertainties in multiple bands. 

	:param filters: (list; string)
		List of photometric filters. The list of filters recognized by piXedfit can be accesses using :func:`piXedfit.utils.filtering.list_filters`.

	:param spec_wave: (1D array; float)
		Wavelength grids of the input spectrum.
	
	:param spec_flux: (1D array; float)
		Flux grids of the input spectrum.

	:param spec_flux_err: (1D array; float)
		Flux uncertainties of the input spectrum.

	:param models_spec:
		Model spectral templates in the rest-frame generated prior to the fitting using :func:`piXedfit.piXedfit_model.save_models_rest_spec`. 
		This set of models will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:

	:param params_priors:

	:param fit_method: (default: 'mcmc')
		Method in SED fitting. Options are: (a)'mcmc' for Markov Chain Monte Carlo, and (b)'rdsps' for Random Dense Sampling of Parameter Space.

	:param gal_z: (float)
		Redshift of the galaxy. If gal_z=None, then redshift is set to be a free parameter in the fitting.

	:param spec_sigma: (default:2.6; float)
		Spectral resolution in Angstrom of the input spectrum.

	:param poly_order: (default:8.0; integer)
		Degree of the legendre polynomial function. 

	:param params_range: (Dictionary)
		Ranges of parameters. The format of this input argument is python dictionary. 
		The range for redshift ('z') only relevant if gal_z=None (i.e., redshift becomes free parameter). 

	:param nrands_z: (default: 10; integer)
		Number of random redshift to be generated in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0; float)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: (default: 100; integer)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600; integer)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (default: 50; integer)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param nproc: (default: 10; integer)
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: (default: 30000; integer)
		Number of models to be used for initial fitting in the MCMC method. 

	:param perc_chi2: (default: 90.0; float)
		Lowest chi-square Percentage from the full model SEDs that are considered in the calculation of posterior-weighted averaging. 
		This parameter is only relevant for the RDSPS fitting method and it is not applicable for the MCMC method.
	
	:param spec_chi_sigma_clip: (default: 3.0; float)

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0; float)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3; float)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param del_wave_nebem: (default: 10; float)

	:param store_full_samplers: (default: 1 or True)
		Flag indicating whether full sampler models will be stored into the output FITS file or not. 
		Options are: (a) 1 or True and (b) 0 or False.

	:param name_out_fits: (optional, default: None; string)
		Name of output FITS file. This parameter is optional. 
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

	if params_ranges==None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors==None:
		params_priors = []

	if gal_z==None or gal_z<=0.0:
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
	write_input_specphoto_hdf5(inputSED_file,obs_flux,obs_flux_err,spec_wave,spec_flux,spec_flux_err)
	os.system('mv %s %s' % (inputSED_file,temp_dir))

	# output files name:
	if name_out_fits == None:
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


def SEDfit_from_binmap_specphoto(fits_binmap,binid_range=[],bin_ids=[],models_spec=None,params_ranges=None,params_priors=None,
	fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,spec_sigma=2.6,poly_order=8,likelihood='gauss',
	dof=3.0,nwalkers=100,nsteps=600,nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0,spec_chi_sigma_clip=3.0, 
	cosmo=0,H0=70.0,Om0=0.3,del_wave_nebem=10.0,store_full_samplers=1,name_out_fits=[]):
	"""A function for performing SED fitting to set of spatially resolved SEDs from the reduced data cube that is produced after the pixel binning. 

	:param fits_binmap:
		Input FITS file of reduced data cube after pixel binning. This FITS file is the one that is output by :func:`pixel_binning_photo` function in the :mod:`piXedfit_bin` module. 
		This is a mandatory parameter.

	:param binid_range:
		Range of bin IDs that are going to be fit. Allowed format is [idmin,idmax]. The id starts from 0. If empty, [], fitting will be done to SEDs of all spatial bins.

	:param bin_ids:
		Bin ids whose the SEDs are going to be fit. Allowed format is a 1D array. The id starts from 0. Both binid_range and bin_ids can't be empty, []. If both of them are not empty, the bin_ids will be used. 

	:param models_spec:
		Model spectral templates in the rest-frame that is generated prior to the fitting. This set of models will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: (1)'mcmc', and (2)'rdsps'.

	:param gal_z:
		Redshift of the galaxy. If gal_z=None, then redshift is taken from the header of the FITS file. 
		If gal_z in the FITS file header is negatiive, then redshift is set to be free. 

	:param spec_sigma: (default: 2.6)
		Spectral resolution.

	:param poly_order: (default: 5.0)

	:param params_range:
		Ranges of parameters. The format of this input argument is python dictionary. 
		The range for redshift ('z') only relevant if gal_z=None (i.e., redshift becomes free parameter).

	:param nrands_z: (default: 10)
		Number of random redshift to be generated in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (optional, default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: (default: 30000)
		Number of models to be used for initial fitting in the MCMC method. 

	:param perc_chi2: (optional, default: 90.0)
		Lowest chi-square Percentage from the full model SEDs that are considered in the calculation of posterior-weighted averaging. 
		This parameter is only relevant for the RDSPS fitting method and it is not applicable for the MCMC method.

	:param spec_chi_sigma_clip: (default: 3.0)

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param store_full_samplers: (default: 1 or True)
		Flag indicating whether full sampler models will be stored into the output FITS file or not. 
		Options are: (a) 1 or True and (b) 0 or False.

	:param name_out_fits: (optional, default: [])
		Names of output FITS files. This parameter is optional. If not empty, it must be in a list format with number of elements is the same as the number of bins to be fit. 
		Example: name_out_fits = ['bin1.fits', 'bin2.fits', ..., 'binN.fits'].
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

	if params_ranges==None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors==None:
		params_priors = []

	if gal_z==None or gal_z<=0.0:
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
	if len(bin_ids)>0:
		if len(binid_range) > 0:
			print ("Both bin_ids and binid_range are not empty, so calculation will be done based on bin_ids.")
		bin_ids = np.asarray(bin_ids)
		nbins_calc = len(bin_ids)

	elif len(bin_ids)==0:
		if len(binid_range) == 0:
			print ("Both bin_ids and binid_range are empty, so SED fitting will be done to all the bins.")
			bin_ids = np.arange(nbins_photo)
		elif len(binid_range) > 0:
			binid_min = binid_range[0]
			binid_max = binid_range[1]
			bin_ids = np.arange(int(binid_min), int(binid_max))

	if 0<len(name_out_fits)<nbins_calc:
		print ("The number of elements in name_out_fits should be the same as the number of bins to be calculated!")
		sys.exit()

	count_id = 0
	for idx_bin in bin_ids:

		# name for output FITS file
		if len(name_out_fits) == 0:
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
			write_input_specphoto_hdf5(inputSED_file,bin_photo_flux[int(idx_bin)],bin_photo_flux_err[int(idx_bin)],
										spec_wave,bin_spec_flux[int(idx_bin)],bin_spec_flux_err[int(idx_bin)])
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


def SEDfit_from_binmap(fits_binmap,binid_range=[],bin_ids=[],models_spec=None,params_ranges=None,params_priors=None,
	fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,likelihood='gauss',dof=3.0,nwalkers=100,
	nsteps=600,nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,perc_chi2=90.0,cosmo=0,H0=70.0,Om0=0.3,
	store_full_samplers=1,name_out_fits=[]):

	"""A function for performing SED fitting to set of spatially resolved SEDs from the reduced data cube that is produced after the pixel binning. 

	:param fits_binmap:
		Input FITS file of reduced data cube after pixel binning. This FITS file is the one that is output by :func:`pixel_binning_photo` function in the :mod:`piXedfit_bin` module. 
		This is a mandatory parameter.

	:param binid_range:
		Range of bin IDs that are going to be fit. Allowed format is [idmin,idmax]. The id starts from 0. If empty, [], fitting will be done to SEDs of all spatial bins.

	:param bin_ids:
		Bin ids whose the SEDs are going to be fit. Allowed format is a 1D array. The id starts from 0. Both binid_range and bin_ids can't be empty, []. If both of them are not empty, the bin_ids will be used. 

	:param models_spec:
		Model spectral templates in the rest-frame that is generated prior to the fitting. This set of models will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:

	:param params_priors:

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: (1)'mcmc', and (2)'rdsps'.

	:param gal_z:
		Redshift of the galaxy. If gal_z=None, then redshift is taken from the header of the FITS file. 
		If gal_z in the FITS file header is negatiive, then redshift is set to be free. 

	:param params_range:
		Ranges of parameters. The format of this input argument is python dictionary. 
		The range for redshift ('z') only relevant if gal_z=None (i.e., redshift becomes free parameter). 

	:param nrands_z: (default: 10)
		Number of random redshift to be generated in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (optional, default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. 
		Only relevant if fit_method='mcmc'.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: (default: 30000)
		Number of models to be used for initial fitting in the MCMC method.

	:param perc_chi2: (optional, default: 90.0)
		Lowest chi-square Percentage from the full model SEDs that are considered in the calculation of posterior-weighted averaging. 
		This parameter is only relevant for the RDSPS fitting method and it is not applicable for the MCMC method.

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param store_full_samplers: (default: 1 or True)
		Flag indicating whether full sampler models will be stored into the output FITS file or not. 
		Options are: (a) 1 or True and (b) 0 or False.

	:param name_out_fits: (optional, default: [])
		Names of output FITS files. This parameter is optional. If not empty, it must be in a list format with number of elements is the same as the number of bins to be fit. 
		Example: name_out_fits = ['bin1.fits', 'bin2.fits', ..., 'binN.fits'].
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

	if params_ranges==None:
		pr = priors()
		params_ranges = pr.params_ranges()

	if params_priors==None:
		params_priors = []

	if gal_z==None or gal_z<=0.0:
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
	if len(bin_ids)>0:
		if len(binid_range) > 0:
			print ("Both bin_ids and binid_range are not empty, calculation will be done based on bin_ids.")
		bin_ids = np.asarray(bin_ids)
		nbins_calc = len(bin_ids)

	elif len(bin_ids)==0:
		if len(binid_range) == 0:
			print ("Both bin_ids and binid_range are empty, SED fitting will be done to all the bins.")
			bin_ids = np.arange(nbins)
		elif len(binid_range) > 0:
			binid_min = binid_range[0]
			binid_max = binid_range[1]
			bin_ids = np.arange(int(binid_min), int(binid_max))

	if 0<len(name_out_fits)<nbins_calc:
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
			if len(name_out_fits) == 0:
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
			if len(name_out_fits) == 0:
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
			if len(name_out_fits) == 0:
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


def SEDfit_pixels_from_fluxmap(fits_fluxmap,x_range=[],y_range=[],models_spec=None,params_ranges=None,params_priors=None,
	fit_method='mcmc',gal_z=None,nrands_z=10,add_igm_absorption=0,igm_type=0,likelihood='gauss',dof=2.0,nwalkers=100,nsteps=600,
	nsteps_cut=50,nproc=10,initfit_nmodels_mcmc=30000,cosmo=0,H0=70.0,Om0=0.3,perc_chi2=10.0,store_full_samplers=1):

	"""A function for performing SED fitting on pixel-by-pixel basis. 

	:param fits_fluxmap:
		Input FITS file of reduced 3D data cube of multiband fluxes. It is the ouput (or should have the same format as that of the output) 
		of the image processing by :func:`images_processing` function of the :mod:`piXedfit_images` module.

	:param x_range:
		Range of x-axis coordinate within which SED fitting will be performed to the pixels. The format is [xmin,xmax]. 
		If x_range=[], the whole x-axis range of the data cube is considered.  

	:param y_range:
		Range of y-axis coordinate within which SED fitting will be performed to the pixels. The format is [ymin,ymax]. 
		If y_range=[], the whole y-axis range of the data cube is considered.

	:param models_spec:
		Model spectral templates in the rest-frame that is generated prior to the fitting. This set of models will be used in the main fitting step if fit_method='rdsps' or initial fitting if fit_method='mcmc'. 

	:param params_ranges:

	:param params_priors:

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: (1)'mcmc', and (2)'rdsps'.

	:param gal_z: 
		Redshift of the galaxy. If gal_z=None, then redshift is taken from the FITS file header. 
		If gal_z from the header is negative, then redshift is set to be free. **As for the current version of **piXedfit**, photo-z hasn't been implemented**.

	:param params_range:
		Ranges of parameters. The format of this input argument is python dictionary. 
		The range for redshift ('z') only relevant if gal_z=None (i.e., redshift becomes free parameter). 

	:param nrands_z: (default: 10)
		Number of random redshift to be generated in the main fitting if fit_method='rdsps' or initial fitting if fit_method='mcmc'.

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (optional, default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

	:param initfit_nmodels_mcmc: (default: 30000)
		Number of models to be used for initial fitting in the MCMC method. 

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param perc_chi2: (default: 10.0)
		Lowest chi-square Percentage from the full model SEDs that are considered in the calculation of posterior-weighted averaging. 
		This parameter is only relevant for the RDSPS fitting method and it is not applicable for the MCMC method.

	:param store_full_samplers: (default: 1 or True)
		Flag indicating whether full sampler models will be stored into the output FITS file or not. 
		Options are: (a) 1 or True and (b) 0 or False.
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
	if len(x_range)==0:
		xmin, xmax = 0, dim_x
	else:
		if x_range[0]<0 or x_range[1]>dim_x:
			print ("Can't perform SED fitting to region beyond the region of the data cube!")
			sys.exit()
		else:
			xmin, xmax = x_range[0], x_range[1] 

	if len(y_range)==0:
		ymin, ymax = 0, dim_y
	else:
		if y_range[0]<0 or y_range[1]>dim_y:
			print ("Can't perform SED fitting to region beyond the region of the data cube!")
			sys.exit()
		else:
			ymin, ymax = y_range[0], y_range[1]

	if gal_z==None or gal_z<=0.0:
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


def maps_parameters(fits_binmap=None, bin_ids=[], name_sampler_fits=[], fits_fluxmap=None, refband_SFR=None, refband_SM=None, 
	refband_Mdust=None, name_out_fits=None):
	"""Function that can be used for constructing maps of properties from the fitting results of spatial bins.

	:param fits_binmap: (Mandatory, default=None)
		FITS file containing the reduced data after the pixel binning process, which is output of the :func:`pixel_binning_photo` function in the :mod:`piXedfit_bin` module.

	:param bin_ids:
		Bin indices of the FITS files listed in name_sampler_fits input. Allowed format is a 1D array. The id starts from 0. 

	:param name_sampler_fits:
		List of the names of the FITS files containing the fitting results of spatial bins. This should have the same number of element as that of bin_ids. 
		The number of element doesn't necessarily the same as the number of bins. A missing bin will be ignored in the creation of the maps of properties.

	:param fits_fluxmap: (Mandatory, default: None)
		FITS file containing reduced maps of multiband fluxes, which is output of the :func:`flux_map` in the :class:`images_processing` class in the :mod:`piXedfit_images` module.

	:param refband_SFR: (default: None)
		Index of band in the multiband set that is used for reference in dividing map of SFR in bin space into map of SFR in pixel space.
		If None, the band with shortest wavelength is selected.

	:param refband_SM: (default: None)
		Index of band in the multiband set that is used for reference in dividing map of stellar mass in bin space into map of stellar mass in pixel space.
		If None, the band with longest wavelength is selected.

	:param refband_Mdust: (default: None)
		Index of band/filter in the multiband set that is used for reference in dividing map of dust mass in bin space into map of dust mass in pixel space.
		If None, the band with longest wavelength is selected.

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
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
	if refband_SFR == None:
		refband_SFR = 0
	if refband_SM == None:
		refband_SM = nbands-1
	if refband_Mdust == None:
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
					
	if name_out_fits == None:
		name_out_fits = "fitres_%s" % fits_binmap
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def maps_parameters_fit_pixels(fits_fluxmap=None, pix_x=[], pix_y=[], name_sampler_fits=[], name_out_fits=None):
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
					
	if name_out_fits == None:
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


