import numpy as np
from math import pi, pow, sqrt, cos, sin
import sys, os
import h5py
from random import randint
from astropy.io import fits

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

os.environ["OMP_NUM_THREADS"] = "1"

__all__ = ["singleSEDfit", "singleSEDfit_specphoto", "SEDfit_from_binmap", "SEDfit_from_binmap_specphoto", 
			"SEDfit_pixels_from_fluxmap", "inferred_params_mcmc_list", "inferred_params_rdsps_list",
			"get_inferred_params_mcmc", "get_inferred_params_rdsps", "maps_parameters", "maps_parameters_fit_pixels", 
			"get_params", "priors"]


def nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut):
	ngrids2 = (nwalkers*nsteps) - (nwalkers*nsteps_cut)        
	for ii in range(0,nproc):
		if ngrids2%nproc == 0:
			nproc_new = nproc
			break
		nproc = nproc  - 1
	return nproc_new


class priors:
	"""Functions for defining priors.
	"""
	def __init__(self, ranges={'z':[0.0,1.0],'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-3.0,1.14],
		'log_alpha':[-2.0,2.0],'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0], 
		'dust2':[0.0,4.0],'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],
		'log_tauagn':[0.7, 2.18]}):

		def_ranges={'z':[0.0,1.0],'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-3.0,1.14],'log_alpha':[-2.0,2.0],
			'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0],'dust2':[0.0,4.0],
			'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],
			'log_tauagn':[0.7, 2.18]}

		# get keys in input params_range:
		keys = list(ranges.keys())

		# merge with the default one
		ranges1 = def_ranges
		for ii in range(0,len(keys)):
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

def write_filters_list(name,filters):
	file_out = open(name,"w")
	for bb in range(0,len(filters)):
		file_out.write("%s\n" % filters[bb]) 
	file_out.close()

def write_input_singleSED(name,obs_flux,obs_flux_err):
	file_out = open(name,"w")
	for bb in range(0,len(obs_flux)):
		file_out.write("%e  %e\n" % (obs_flux[bb],obs_flux_err[bb]))
	file_out.close()

def write_input_specphoto_hdf5(name,obs_flux,obs_flux_err,spec_wave,spec_flux,spec_flux_err):
	with h5py.File(name, 'w') as f:
		f.create_dataset('obs_flux', data=np.array(obs_flux), compression="gzip")
		f.create_dataset('obs_flux_err', data=np.array(obs_flux_err), compression="gzip")
		f.create_dataset('spec_wave', data=np.array(spec_wave), compression="gzip")
		f.create_dataset('spec_flux', data=np.array(spec_flux), compression="gzip")
		f.create_dataset('spec_flux_err', data=np.array(spec_flux_err), compression="gzip")

def write_conf_file(name,params_ranges,priors_coll,nwalkers,nsteps,nsteps_cut,nproc,cosmo,
	H0,Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
	perc_chi2,initfit_nmodels_mcmc,spec_sigma=None,poly_order=None,del_wave_nebem=None,
	spec_chi_sigma_clip=None):

	file_out = open(name,"w")
	file_out.write("nwalkers %d\n" % nwalkers)
	file_out.write("nsteps %d\n" % nsteps)
	file_out.write("nsteps_cut %d\n" % nsteps_cut)
	file_out.write("ori_nproc %d\n" % nproc)
	# cosmology
	if cosmo=='flat_LCDM' or cosmo==0:
		cosmo1 = 0
	elif cosmo=='WMAP5' or cosmo==1:
		cosmo1 = 1
	elif cosmo=='WMAP7' or cosmo==2:
		cosmo1 = 2
	elif cosmo=='WMAP9' or cosmo==3:
		cosmo1 = 3
	elif cosmo=='Planck13' or cosmo==4:
		cosmo1 = 4
	elif cosmo=='Planck15' or cosmo==5:
		cosmo1 = 5
	#elif cosmo=='Planck18' or cosmo==6:
	#	cosmo1 = 6
	else:
		print ("Input cosmo is not recognized!")
		sys.exit()
	file_out.write("cosmo %d\n" % cosmo1)
	file_out.write("H0 %lf\n" % H0)
	file_out.write("Om0 %lf\n" % Om0)
	if fit_method=='rdsps' or fit_method=='RDSPS':
		file_out.write("likelihood %s\n" % likelihood)
		file_out.write("dof %lf\n" % dof)
	file_out.write("models_spec %s\n" % models_spec)
	file_out.write("gal_z %lf\n" % gal_z)
	file_out.write("nrands_z %d\n" % nrands_z)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)
	file_out.write("igm_type %d\n" % igm_type)
	if fit_method=='rdsps' or fit_method=='RDSPS':
		file_out.write("perc_chi2 %lf\n" % perc_chi2)
	if spec_sigma != None:
		file_out.write("spec_sigma %lf\n" % spec_sigma)
	if poly_order != None:
		file_out.write("poly_order %d\n" % poly_order)
	if del_wave_nebem != None:
		file_out.write("del_wave_nebem %lf\n" % del_wave_nebem)
	if spec_chi_sigma_clip != None:
		file_out.write("spec_chi_sigma_clip %lf\n" % spec_chi_sigma_clip)
	file_out.write("initfit_nmodels_mcmc %d\n" % initfit_nmodels_mcmc)
	# ranges of parameters
	file_out.write("pr_z_min %lf\n" % params_ranges['z'][0])
	file_out.write("pr_z_max %lf\n" % params_ranges['z'][1])
	file_out.write("pr_logzsol_min %lf\n" % params_ranges['logzsol'][0])
	file_out.write("pr_logzsol_max %lf\n" % params_ranges['logzsol'][1])
	file_out.write("pr_log_tau_min %lf\n" % params_ranges['log_tau'][0])
	file_out.write("pr_log_tau_max %lf\n" % params_ranges['log_tau'][1])
	file_out.write("pr_log_t0_min %lf\n" % params_ranges['log_t0'][0])
	file_out.write("pr_log_t0_max %lf\n" % params_ranges['log_t0'][1])
	file_out.write("pr_log_alpha_min %lf\n" % params_ranges['log_alpha'][0])
	file_out.write("pr_log_alpha_max %lf\n" % params_ranges['log_alpha'][1])
	file_out.write("pr_log_beta_min %lf\n" % params_ranges['log_beta'][0])
	file_out.write("pr_log_beta_max %lf\n" % params_ranges['log_beta'][1])
	file_out.write("pr_log_age_min %lf\n" % params_ranges['log_age'][0])
	file_out.write("pr_log_age_max %lf\n" % params_ranges['log_age'][1])
	file_out.write("pr_dust_index_min %lf\n" % params_ranges['dust_index'][0])
	file_out.write("pr_dust_index_max %lf\n" % params_ranges['dust_index'][1])
	file_out.write("pr_dust1_min %lf\n" % params_ranges['dust1'][0])
	file_out.write("pr_dust1_max %lf\n" % params_ranges['dust1'][1])
	file_out.write("pr_dust2_min %lf\n" % params_ranges['dust2'][0])
	file_out.write("pr_dust2_max %lf\n" % params_ranges['dust2'][1])
	file_out.write("pr_log_gamma_min %lf\n" % params_ranges['log_gamma'][0])
	file_out.write("pr_log_gamma_max %lf\n" % params_ranges['log_gamma'][1])
	file_out.write("pr_log_umin_min %lf\n" % params_ranges['log_umin'][0])
	file_out.write("pr_log_umin_max %lf\n" % params_ranges['log_umin'][1])
	file_out.write("pr_log_qpah_min %lf\n" % params_ranges['log_qpah'][0])
	file_out.write("pr_log_qpah_max %lf\n" % params_ranges['log_qpah'][1])
	file_out.write("pr_log_fagn_min %lf\n" % params_ranges['log_fagn'][0])
	file_out.write("pr_log_fagn_max %lf\n" % params_ranges['log_fagn'][1])
	file_out.write("pr_log_tauagn_min %lf\n" % params_ranges['log_tauagn'][0])
	file_out.write("pr_log_tauagn_max %lf\n" % params_ranges['log_tauagn'][1])
	file_out.write("pr_nparams %d\n" % len(priors_coll))
	for ii in range(0,len(priors_coll)):
		priors = priors_coll[ii]
		param = priors[0]
		form = priors[1]
		file_out.write("pr_param%d %s\n" % (ii,param))
		file_out.write("pr_form_%s %s\n" % (param,form))
		if form == 'gaussian':
			loc, scale = priors[2], priors[3]
			file_out.write("pr_form_%s_gauss_loc %lf\n" % (param,loc))
			file_out.write("pr_form_%s_gauss_scale %lf\n" % (param,scale))
		elif form == 'studentt':
			df, loc, scale = priors[2], priors[3], priors[4]
			file_out.write("pr_form_%s_stdt_df %lf\n" % (param,df))
			file_out.write("pr_form_%s_stdt_loc %lf\n" % (param,loc))
			file_out.write("pr_form_%s_stdt_scale %lf\n" % (param,scale))
		elif form == 'gamma':
			a, loc, scale = priors[2], priors[3], priors[4]
			file_out.write("pr_form_%s_gamma_a %lf\n" % (param,a))
			file_out.write("pr_form_%s_gamma_loc %lf\n" % (param,loc))
			file_out.write("pr_form_%s_gamma_scale %lf\n" % (param,scale))
	file_out.close()


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

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'

	# get number of filters:
	nbands = len(filters)

	# file of filter list
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
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
	name_config = "config_file%d.dat" % (randint(0,10000))
	write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc)
	os.system('mv %s %s' % (name_config,temp_dir))

	# input SED text file
	name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
	write_input_singleSED(name_SED_txt,obs_flux,obs_flux_err)
	os.system('mv %s %s' % (name_SED_txt,temp_dir))

	# output files name:
	if name_out_fits == None:
		random_int = (randint(0,20000))
		if fit_method=='mcmc' or fit_method=='MCMC':
			name_out_fits = "mcmc_fit%d.fits" % random_int
		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_out_fits = "rdsps_fit%d.fits" % random_int

	if fit_method=='mcmc' or fit_method=='MCMC':
		random_int = (randint(0,20000))
		name_samplers_hdf5 = "samplers_%d.hdf5" % random_int

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

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'

	# get number of filters:
	nbands = len(filters)

	# file of filter list
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
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
	name_config = "config_file%d.dat" % (randint(0,10000))
	write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
					Om0,fit_method,likelihood,dof,models_spec,gal_z,nrands_z,add_igm_absorption,igm_type,
					perc_chi2,initfit_nmodels_mcmc,spec_sigma=spec_sigma,poly_order=poly_order,
					del_wave_nebem=del_wave_nebem,spec_chi_sigma_clip=spec_chi_sigma_clip)
	os.system('mv %s %s' % (name_config,temp_dir))

	# input SED in HDF5
	inputSED_file = "inputSED_file%d.hdf5" % (randint(0,20000))
	write_input_specphoto_hdf5(inputSED_file,obs_flux,obs_flux_err,spec_wave,spec_flux,spec_flux_err)
	os.system('mv %s %s' % (inputSED_file,temp_dir))

	# output files name:
	if name_out_fits == None:
		random_int = (randint(0,20000))
		if fit_method=='mcmc' or fit_method=='MCMC':
			name_out_fits = "mcmc_fit%d.fits" % random_int
		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_out_fits = "rdsps_fit%d.fits" % random_int

	if fit_method=='mcmc' or fit_method=='MCMC':
		random_int = (randint(0,20000))
		name_samplers_hdf5 = "samplers_%d.hdf5" % random_int

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

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'

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
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
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

	# configuration file
	name_config = "config_file%d.dat" % (randint(0,10000))
	write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
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
			inputSED_file = "inputSED_file%d.hdf5" % (randint(0,20000))
			write_input_specphoto_hdf5(inputSED_file,bin_photo_flux[int(idx_bin)],bin_photo_flux_err[int(idx_bin)],
										spec_wave,bin_spec_flux[int(idx_bin)],bin_spec_flux_err[int(idx_bin)])
			os.system('mv %s %s' % (inputSED_file,temp_dir))

			if fit_method=='mcmc' or fit_method=='MCMC':
				random_int = (randint(0,20000))
				name_samplers_hdf5 = "samplers_%d.hdf5" % random_int

				os.system("mpirun -n %d python %s./mc_p1_sp.py %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,inputSED_file,name_samplers_hdf5))
				os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
				
				if store_full_samplers==1 or store_full_samplers==True:
					os.system("mpirun -n %d python %s./mc_p2_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
				elif store_full_samplers==0 or store_full_samplers==False:
					os.system("mpirun -n %d python %s./mc_nsmp_p2_sp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_samplers_hdf5,name_out_fits1))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

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

			# free disk space:
			os.system("rm %s%s" % (temp_dir,name_config))
			os.system("rm %s%s" % (temp_dir,name_filters_list))
			os.system("rm %s%s" % (temp_dir,inputSED_file))
			if fit_method=='mcmc' or fit_method=='MCMC':
				os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))

		# photometric SED
		else:
			# store input SED into a text file
			name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
			write_input_singleSED(name_SED_txt,bin_photo_flux[int(idx_bin)],bin_photo_flux_err[int(idx_bin)])
			os.system('mv %s %s' % (name_SED_txt,temp_dir))

			if fit_method=='mcmc' or fit_method=='MCMC':
				name_samplers_hdf5 = "samplers_%d.hdf5" % (randint(0,20000))

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

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'

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

	# file of filter list
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
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

	# configuration file
	name_config = "config_file%d.dat" % (randint(0,10000))
	write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
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
			name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
			write_input_singleSED(name_SED_txt,bin_flux_trans[rows[0]][cols[0]],bin_flux_err_trans[rows[0]][cols[0]])
			os.system('mv %s %s' % (name_SED_txt,temp_dir))

			# name of output FITS file
			if len(name_out_fits) == 0:
				name_out_fits1 = "mcmc_bin%d.fits" % (idx_bin+1)
			else:
				name_out_fits1 = name_out_fits[int(count_id)]

			random_int = (randint(0,20000))
			name_samplers_hdf5 = "samplers_%d.hdf5" % random_int

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

			name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
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
			name_inputSEDs = "input_SEDs_%d.hdf5" % (randint(0,10000))
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
				name_outs = "name_outs_%d.dat" % (randint(0,10000))
				file_out = open(name_outs,"w")
				for idx_bin in bin_ids:
					name0 = "rdsps_bin%d.fits" % (idx_bin+1)
					file_out.write("%s\n" % name0)
				file_out.close()
			else:
				name_outs = "name_outs_%d.dat" % (randint(0,10000))
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

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'

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
	
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
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

	# configuration file
	name_config = "config_file%d.dat" % (randint(0,10000))
	write_conf_file(name_config,params_ranges,params_priors,nwalkers,nsteps,nsteps_cut,nproc,cosmo,H0,
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
					name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
					write_input_singleSED(name_SED_txt,obs_flux,obs_flux_err)
					os.system('mv %s %s' % (name_SED_txt,temp_dir))

					# name of output FITS file
					name_out_fits = "pix_y%d_x%d_mcmc.fits" % (yy,xx)

					random_int = (randint(0,20000))
					name_samplers_hdf5 = "samplers_%d.hdf5" % random_int

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
		name_outs = "name_outs_%d.dat" % (randint(0,10000))
		file_out = open(name_outs,"w")
		# input SEDs: store to hdf5 file
		name_inputSEDs = "input_SEDs_%d.hdf5" % (randint(0,10000))
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


def inferred_params_mcmc_list(list_name_fits=[],name_out_fits=None):
	"""Function for calculating inferred parameters (i.e., median posteriors) of fitting results.
	The expected input is a list of FITS files contining the MCMC samplers which are output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions. 
	The output of this function is a FITS file containing summary of inferred parameters.

	:param list_name_fits: (Mandatory, default=[])
		List of names of the input FITS files. The FITS files are output of fitting with MCMC method using :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.
		All the FITS files should have identical structure. In other word, they are results of fitting with the same set of free parameters.

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.
	"""

	nfiles = len(list_name_fits)

	hdu = fits.open(list_name_fits[0])
	header = hdu[0].header
	nparams = int(header['nparams'])
	free_z = header['free_z']
	ncols = int(header['ncols'])

	if nfiles>1:
		for ii in range(0,nfiles):
			hdu0 = fits.open(list_name_fits[ii])
			header0 = hdu0[0].header
			nparams0 = int(header0['nparams'])
			free_z0 = header0['free_z']
			if nparams!=nparams0 or free_z!=free_z0:
				print ("Only set of fitting results with the same fitting parameters are allowed!")
				sys.exit()
			hdu0.close()

	# get parameters
	tfields = int(hdu[1].header['TFIELDS'])
	params = []
	for ii in range(1,tfields):
		str_temp = 'TTYPE%d' % (ii+1)
		params.append(hdu[1].header[str_temp])
	nparams_new = len(params)
	print("List of parameters:")
	print (params)
	hdu.close()

	# define indexes
	indexes = ["p16","p50","p84"]
	nindexes = len(indexes)

	# allocate memory:
	arrays_fitres = {}
	string_idx = []
	for pp in range(0,nparams_new):
		for ii in range(0,nindexes):
			str_temp = "%s-%s" % (params[pp],indexes[ii])
			arrays_fitres[str_temp] = np.zeros(nfiles)
			string_idx.append(str_temp)
	nstring_idx = len(string_idx)

	# calcultions
	for jj in range(0,nfiles):
		hdu = fits.open(list_name_fits[jj])
		data_samplers = hdu[1].data 
		hdu.close()

		idx_excld = np.where(data_samplers['log_sfr']<=-29.0)

		for pp in range(0,nparams_new):
			data_samp = np.delete(data_samplers[params[pp]], idx_excld[0])
			for ii in range(0,nindexes):
				str_temp = "%s-%s" % (params[pp],indexes[ii])

				if indexes[ii] == 'p16':
					arrays_fitres[str_temp][jj] = np.percentile(data_samp, 16)
				elif indexes[ii] == 'p50':
					arrays_fitres[str_temp][jj] = np.percentile(data_samp, 50)
				elif indexes[ii] == 'p84':
					arrays_fitres[str_temp][jj] = np.percentile(data_samp, 84)

	# make output FITS file to store the results
	hdr = fits.Header()
	hdr['nfiles'] = nfiles
	ids = np.zeros(nfiles)
	for ii in range(0,nfiles):
		str_temp = 'file%d' % ii
		hdr[str_temp] = list_name_fits[ii]
		ids[ii] = ii + 1
	for pp in range(0,nparams_new):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]
	for ii in range(0,nindexes):
		str_temp = 'index%d' % ii
		hdr[str_temp] = indexes[ii]
	primary_hdu = fits.PrimaryHDU(header=hdr)

	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(ids))
	cols0.append(col)

	for ii in range(0,nstring_idx):
		col = fits.Column(name=string_idx[ii], format='D', array=np.array(arrays_fitres[string_idx[ii]]))
		cols0.append(col)

	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdul = fits.HDUList([primary_hdu, hdu])

	if name_out_fits == None:
		name_out_fits = 'bestfit_parameters.fits'
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def inferred_params_rdsps_list(list_name_fits=[], perc_chi2=10.0, name_out_fits=None):
	"""Function for calculating inferred parameters (i.e., median posteriors) of fitting results obtained with RDSPS method.
	The expected input is a list of FITS files contining the model properties which are output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions. 
	The output of this function is a FITS file containing summary of inferred parameters.

	:param list_name_fits: (Mandatory, default=[])
		List of names of the input FITS files. The FITS files are output of fitting with RDSPS method using :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.
		All the FITS files should have identical structure. In other word, they are results of fitting with the same set of free parameters.

	:param perc_chi2: (default: 10.0)
		Lowest chi-square Percentage from the full model SEDs that are considered in the calculation of posterior-weighted averaging. 

	:param name_out_fits: (default: None)
		Desired name for the output FITS file.
	"""
	
	nseds = len(list_name_fits)

	# open one FITS file
	hdu = fits.open(list_name_fits[0])
	duste_stat = hdu[0].header['duste_stat']

	# get parameters
	tfields = int(hdu[1].header['TFIELDS'])
	params = []
	for ii in range(1,tfields-2):				## exclude 'chi2' and 'prob'
		str_temp = 'TTYPE%d' % (ii+1)
		params.append(hdu[1].header[str_temp])
	nparams = len(params)
	hdu.close()

	# allocate memory
	bfit_params = {}
	bfit_params_err = {}
	for pp in range(0,nparams):
		bfit_params[params[pp]] = np.zeros(nseds)
		bfit_params_err[params[pp]] = np.zeros(nseds)

	# iteration
	length_string = np.zeros(nseds)
	for ii in range(0,nseds):
		length_string[ii] = len(list_name_fits[ii])
		hdu = fits.open(list_name_fits[ii])
		data_samplers = hdu[1].data
		hdu.close()

		crit_chi2 = np.percentile(data_samplers['chi2'], perc_chi2)
		idx_sel = np.where((data_samplers['chi2']<=crit_chi2) & (data_samplers['log_sfr']>-29.0) & (np.isnan(data_samplers['lnprob'])==False) & (np.isinf(data_samplers['lnprob'])==False))

		array_lnprob = data_samplers['lnprob'][idx_sel[0]] - max(data_samplers['lnprob'][idx_sel[0]])  # normalize
		array_prob = np.exp(array_lnprob)
		array_prob = array_prob/np.sum(array_prob)						 # normalize
		tot_prob = np.sum(array_prob)

		for pp in range(0,nparams):
			array_val = data_samplers[params[pp]][idx_sel[0]]

			mean_val = np.sum(array_val*array_prob)/tot_prob
			mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
			std_val = sqrt(abs(mean_val2 - (mean_val**2)))

			bfit_params[params[pp]][ii] = mean_val
			bfit_params_err[params[pp]][ii] = std_val

	# store into FITS file
	hdr = fits.Header()
	hdr['nparams'] = nparams
	hdr['nseds'] = nseds
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul = fits.HDUList()
	hdul.append(primary_hdu)

	id_seds = np.linspace(1, nseds, nseds)
	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(id_seds))
	cols0.append(col)
	str_temp = 'A%d' % max(length_string)
	col = fits.Column(name='name', format=str_temp, array=np.array(list_name_fits))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdul.append(fits.BinTableHDU.from_columns(cols, name='input_fits'))

	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(id_seds))
	cols0.append(col)
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(bfit_params[params[pp]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdul.append(fits.BinTableHDU.from_columns(cols, name='mean'))

	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(id_seds))
	cols0.append(col)
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(bfit_params_err[params[pp]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdul.append(fits.BinTableHDU.from_columns(cols, name='std'))

	if name_out_fits == None:
		name_out_fits = 'bfit_params.fits'
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def get_inferred_params_mcmc(list_name_sampler_fits=[], bin_excld_flag=[]):
	"""A function for calculating inferred parameters from the fitting results with MCMC method.
	"""
	nfiles = len(list_name_sampler_fits)

	# get list of parameters
	hdu = fits.open(list_name_sampler_fits[nfiles-1])
	tfields = int(hdu[1].header['TFIELDS'])
	params = []
	for ii in range(1,tfields):
		str_temp = 'TTYPE%d' % (ii+1)
		params.append(hdu[1].header[str_temp])
	nparams = len(params)
	hdu.close()

	indexes = ["p16","p50","p84"]
	nindexes = len(indexes)

	# allocate memory
	bfit_param = {}
	for pp in range(0,nparams):
		bfit_param[params[pp]] = {}
		for ii in range(0,nindexes):
			bfit_param[params[pp]][indexes[ii]] = np.zeros(nfiles) - 99.0

	# calculation
	for ii in range(0,nfiles):
		if bin_excld_flag[ii] != 1:
			hdu = fits.open(list_name_sampler_fits[ii])
			ncols = int(hdu[0].header['ncols'])
			params0 = []
			for jj in range(2,ncols+1):
				str_temp = "col%d" % jj
				params0.append(hdu[0].header[str_temp])
			nparams0 = len(params0)
			data_samplers = hdu[1].data 

			idx_excld = np.where(data_samplers['log_sfr']<=-29.0)

			# iteration
			for pp in range(0,nparams0):
				data_samp = np.delete(data_samplers[params0[pp]], idx_excld[0])
				for kk in range(0,nindexes):
					if indexes[kk] == 'p16':
						bfit_param[params0[pp]][indexes[kk]][ii] = np.percentile(data_samp, 16)
					elif indexes[kk] == 'p50':
						bfit_param[params0[pp]][indexes[kk]][ii] = np.percentile(data_samp, 50)
					elif indexes[kk] == 'p84':
						bfit_param[params0[pp]][indexes[kk]][ii] = np.percentile(data_samp, 84)
			hdu.close()

	return bfit_param,params


def get_inferred_params_rdsps(list_name_sampler_fits=[], bin_excld_flag=[], perc_chi2=10.0):
	nfiles = len(list_name_sampler_fits)

	# get parameters
	hdu = fits.open(list_name_sampler_fits[0]) 
	duste_stat = hdu[0].header['duste_stat']
	params = []
	for pp in range(0,int(hdu[0].header['nparams'])):
		str_temp = 'param%d' % pp
		params.append(hdu[0].header[str_temp])
	hdu.close()
	params.append('log_mass')
	params.append('log_sfr')
	params.append('log_mw_age')
	if duste_stat == 'duste':
		params.append('log_dustmass')
	nparams = len(params)

	indexes = ["mean", "mean_err"]

	bfit_param = {}
	for pp in range(0,nparams):
		bfit_param[params[pp]] = {}
		bfit_param[params[pp]]["mean"] = np.zeros(nfiles) - 99.0
		bfit_param[params[pp]]["mean_err"] = np.zeros(nfiles) - 99.0

	for ii in range(0,nfiles):
		if bin_excld_flag[ii] != 1:
			hdu = fits.open(list_name_sampler_fits[ii])
			data_samplers = hdu[1].data
			hdu.close()

			crit_chi2 = np.percentile(data_samplers['chi2'], perc_chi2)
			idx_sel = np.where((data_samplers['chi2']<=crit_chi2) & (data_samplers['log_sfr']>-29.0) & (np.isnan(data_samplers['lnprob'])==False) & (np.isinf(data_samplers['lnprob'])==False))

			array_lnprob = data_samplers['lnprob'][idx_sel[0]] - max(data_samplers['lnprob'][idx_sel[0]])  # normalize
			array_prob = np.exp(array_lnprob)
			array_prob = array_prob/np.sum(array_prob)						 # normalize
			tot_prob = np.sum(array_prob)

			for pp in range(0,nparams):
				array_val = data_samplers[params[pp]][idx_sel[0]]

				mean_val = np.sum(array_val*array_prob)/tot_prob
				mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
				std_val = sqrt(abs(mean_val2 - (mean_val**2)))

				bfit_param[params[pp]]["mean"][ii] = mean_val
				bfit_param[params[pp]]["mean_err"][ii] = std_val

	return bfit_param,params,indexes


def maps_parameters(fits_binmap=None, bin_ids=[], name_sampler_fits=[], fits_fluxmap=None, refband_SFR=None, refband_SM=None, 
	refband_Mdust=None, perc_chi2=80.0, name_out_fits=None):
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

	:param perc_chi2: (optional, default=80.0)
		A parameter that set the percentile cut of the random model SEDs. The cut is applied after the models are sorted based on their chi-square values. 
		This parameter defines up to what percentile the models will be cut. This parameter is only used if the fitting results are obtained with the RDSPS method.  

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
	"""

	# open the FITS file containing the pixel binning map
	hdu = fits.open(fits_binmap)
	pix_bin_flag = hdu['bin_map'].data
	unit_bin = float(hdu[0].header['unit'])
	nbins = int(hdu[0].header['nbins'])
	gal_z = float(hdu[0].header['z'])
	nbands = int(hdu[0].header['nfilters'])
	bin_flux = hdu['bin_flux'].data*unit_bin
	bin_flux_err = hdu['bin_fluxerr'].data*unit_bin
	hdu.close()

	dim_y = pix_bin_flag.shape[0]
	dim_x = pix_bin_flag.shape[1]

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
	galaxy_region = hdu['galaxy_region'].data
	unit_pix = float(hdu[0].header['unit'])
	pix_flux = hdu['flux'].data*unit_pix
	pix_flux_err = hdu['flux_err'].data*unit_pix
	hdu.close()

	# check the fitting method and whether the sampler chains are stored or not
	hdu = fits.open(name_sampler_fits[0])
	fit_method = hdu[0].header['fitmethod']
	store_full_samplers = int(hdu[0].header['storesamp'])
	hdu.close()

	if fit_method == 'rdsps':
		indexes = ["mean", "mean_err"]
		nindexes = len(indexes)

		#=> get the fitting results
		if store_full_samplers == 1:
			bfit_param,params,indexes = get_inferred_params_rdsps(list_name_sampler_fits=name_sampler_fits1, bin_excld_flag=bin_excld_flag, perc_chi2=perc_chi2)

		elif store_full_samplers == 0:
			nfiles = len(name_sampler_fits)

			# get params
			hdu = fits.open(name_sampler_fits[nfiles-1])
			params = []
			for ii in range(2,int(hdu[0].header['ncols']+1)):
				str_temp = 'col%d' % ii
				params.append(hdu[0].header[str_temp])
			hdu.close()
			nparams = len(params)

			# allocate memory
			bfit_param = {}
			for pp in range(0,nparams):
				bfit_param[params[pp]] = {}
				for ii in range(0,nindexes):
					bfit_param[params[pp]][indexes[ii]] = np.zeros(nbins) - 99.0

			# get bfit_param
			for ii in range(0,nbins):
				if bin_excld_flag[ii] != 1:
					hdu = fits.open(name_sampler_fits1[ii])
					for pp in range(0,nparams):
						bfit_param[params[pp]]["mean"][ii] = hdu[1].data[params[pp]][0]
						bfit_param[params[pp]]["mean_err"][ii] = hdu[1].data[params[pp]][1]
					hdu.close()

		nparams = len(params)
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
				str_temp = 'HDU%d' % int(count_HDU)
				hdr[str_temp] = idx_str
				count_HDU = count_HDU + 1
		# pixel space
		params_pix = []
		for pp in range(0,nparams):
			if params[pp] == 'log_sfr' or params[pp] == 'log_mass' or params[pp]=='log_dustmass':
				params_pix.append(params[pp])
				for ii in range(0,nindexes):
					idx_str = "pix-%s-%s" % (params[pp],indexes[ii])
					str_temp = 'HDU%d' % int(count_HDU)
					hdr[str_temp] = idx_str
					count_HDU = count_HDU + 1
		hdr['nHDU'] = count_HDU
		primary_hdu = fits.PrimaryHDU(header=hdr)
		hdul.append(primary_hdu)

		# get number of parameters to be distributed to pixel space
		nparams_pix = len(params_pix)

		hdul.append(fits.ImageHDU(galaxy_region, name='galaxy_region'))

		#maps in bin space
		for pp in range(0,nparams):
			for ii in range(0,nindexes):
				idx_str = "bin-%s-%s" % (params[pp],indexes[ii])
				map_prop = np.zeros((dim_y,dim_x))
				for bb in range(0,nbins):
					rows, cols = np.where(pix_bin_flag==bb+1)
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

				if indexes[ii] == 'mean':
					for bb in range(0,nbins):
						rows, cols = np.where(pix_bin_flag==bb+1)
						bin_val = np.power(10.0,bfit_param[params_pix[pp]][indexes[ii]][bb])
						pix_val = bin_val*pix_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]/bin_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
						map_prop[rows,cols] = np.log10(pix_val)
					hdul.append(fits.ImageHDU(map_prop, name=idx_str))
				else:
					for bb in range(0,nbins):
						rows, cols = np.where(pix_bin_flag==bb+1)
						bin_val = np.power(10.0,bfit_param[params_pix[pp]]["mean"][bb])
						bin_valerr = np.power(10.0,bfit_param[params_pix[pp]][indexes[ii]][bb])
						bin_f = bin_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
						bin_ferr =  bin_flux_err[int(refband_params_pix[params_pix[pp]])][rows,cols]
						pix_f = pix_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
						pix_ferr = pix_flux_err[int(refband_params_pix[params_pix[pp]])][rows,cols]
						pix_val = bin_val*np.sqrt((bin_val*bin_val/bin_valerr/bin_valerr) + (bin_f*bin_f/bin_ferr/bin_ferr) + (pix_f*pix_f/pix_ferr/pix_ferr))
						map_prop[rows,cols] = np.log10(pix_val)
					hdul.append(fits.ImageHDU(map_prop, name=idx_str))
					
		if name_out_fits == None:
			name_out_fits = "fitres_%s" % fits_binmap
		hdul.writeto(name_out_fits, overwrite=True)

	elif fit_method == 'mcmc':
		indexes = ["p16","p50","p84"]
		nindexes = len(indexes)

		#=> get the fitting results
		if store_full_samplers == 1:
			bfit_param,params = get_inferred_params_mcmc(list_name_sampler_fits=name_sampler_fits1,bin_excld_flag=bin_excld_flag)

		elif store_full_samplers == 0:
			nfiles = len(name_sampler_fits)

			# get params
			hdu = fits.open(name_sampler_fits[nfiles-1])
			params = []
			for ii in range(2,int(hdu[0].header['ncols']+1)):
				str_temp = 'col%d' % ii
				params.append(hdu[0].header[str_temp])
			hdu.close()
			nparams = len(params)

			# allocate memory
			bfit_param = {}
			for pp in range(0,nparams):
				bfit_param[params[pp]] = {}
				for ii in range(0,nindexes):
					bfit_param[params[pp]][indexes[ii]] = np.zeros(nbins) - 99.0

			# get bfit_param
			for ii in range(0,nbins):
				if bin_excld_flag[ii] != 1:
					hdu = fits.open(name_sampler_fits1[ii])
					for pp in range(0,nparams):
						bfit_param[params[pp]]["p16"][ii] = hdu[1].data[params[pp]][0]
						bfit_param[params[pp]]["p50"][ii] = hdu[1].data[params[pp]][1]
						bfit_param[params[pp]]["p84"][ii] = hdu[1].data[params[pp]][2]
					hdu.close()

		nparams = len(params)
		# Make FITS file:
		hdul = fits.HDUList()
		hdr = fits.Header()
		hdr['gal_z'] = gal_z
		hdr['nbins'] = nbins
		count_HDU = 2
		# bin space
		for pp in range(0,nparams):
			for ii in range(0,nindexes):
				idx_str = "bin-%s-%s" % (params[pp],indexes[ii])
				str_temp = 'HDU%d' % int(count_HDU)
				hdr[str_temp] = idx_str
				count_HDU = count_HDU + 1
		# pixel space
		params_pix = []
		for pp in range(0,nparams):
			if params[pp] == 'log_sfr' or params[pp] == 'log_mass' or params[pp]=='log_dustmass':
				params_pix.append(params[pp])
				for ii in range(0,nindexes):
					idx_str = "pix-%s-%s" % (params[pp],indexes[ii])
					str_temp = 'HDU%d' % int(count_HDU)
					hdr[str_temp] = idx_str
					count_HDU = count_HDU + 1
		hdr['nHDU'] = count_HDU
		primary_hdu = fits.PrimaryHDU(header=hdr)
		hdul.append(primary_hdu)

		# get number of parameters to be distributed to pixel space
		nparams_pix = len(params_pix)

		hdul.append(fits.ImageHDU(galaxy_region, name='galaxy_region'))

		#maps in bin space
		for pp in range(0,nparams):
			for ii in range(0,nindexes):
				idx_str = "bin-%s-%s" % (params[pp],indexes[ii])
				map_prop = np.zeros((dim_y,dim_x))
				for bb in range(0,nbins):
					rows, cols = np.where(pix_bin_flag==bb+1)
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

				for bb in range(0,nbins):
					rows, cols = np.where(pix_bin_flag==bb+1)
					bin_val = np.power(10.0,bfit_param[params_pix[pp]][indexes[ii]][bb])
					pix_val = bin_val*pix_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]/bin_flux[int(refband_params_pix[params_pix[pp]])][rows,cols]
					map_prop[rows,cols] = np.log10(pix_val)
				hdul.append(fits.ImageHDU(map_prop, name=idx_str))
					
		if name_out_fits == None:
			name_out_fits = "fitres_%s" % fits_binmap
		hdul.writeto(name_out_fits, overwrite=True)


	return name_out_fits


def maps_parameters_fit_pixels(fits_fluxmap=None, pix_x=[], pix_y=[], name_sampler_fits=[], perc_chi2=80.0, name_out_fits=None):
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

	:param perc_chi2: (optional, default=80.0)
		A parameter that set the percentile cut of the random model SEDs. The cut is applied after the models are sorted based on their chi-square values. 
		This parameter defines up to what percentile the models will be cut. This parameter is only used if the fitting results are obtained with the RDSPS method.  

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
	"""

	npixs = len(name_sampler_fits)

	if len(pix_x)!=npixs or len(pix_y)!=npixs:
		print ("pix_x and pix_y should have the same number of elements as that of name_sampler_fits!")
		sys.exit()

	# open FITS file containing maps of multiband fluxes
	hdu = fits.open(fits_fluxmap)
	galaxy_region = hdu['galaxy_region'].data

	gal_z = float(hdu[0].header['z'])
	hdu.close()

	dim_y = galaxy_region.shape[0]
	dim_x = galaxy_region.shape[1]

	# check the fitting method and whether the sampler chains are stored or not
	hdu = fits.open(name_sampler_fits[0])
	fit_method = hdu[0].header['fitmethod']
	store_full_samplers = int(hdu[0].header['storesamp'])
	hdu.close()

	if fit_method=='rdsps':
		indexes = ["mean", "mean_err"]
		nindexes = len(indexes)

		#=> get the fitting results
		if store_full_samplers == 1:
			bfit_param,params,indexes = get_inferred_params_rdsps(list_name_sampler_fits=name_sampler_fits, bin_excld_flag=bin_excld_flag, 
																	perc_chi2=perc_chi2)

		elif store_full_samplers == 0:
			# get params
			hdu = fits.open(name_sampler_fits[npixs-1])
			params = []
			for ii in range(2,int(hdu[0].header['ncols']+1)):
				str_temp = 'col%d' % ii
				params.append(hdu[0].header[str_temp])
			hdu.close()
			nparams = len(params)

			# allocate memory
			bfit_param = {}
			for pp in range(0,nparams):
				bfit_param[params[pp]] = {}
				for ii in range(0,nindexes):
					bfit_param[params[pp]][indexes[ii]] = np.zeros(npixs) - 99.0

			# get bfit_param
			for ii in range(0,npixs):
				hdu = fits.open(name_sampler_fits[ii])
				for pp in range(0,nparams):
					bfit_param[params[pp]]["mean"][ii] = hdu[1].data[params[pp]][0]
					bfit_param[params[pp]]["mean_err"][ii] = hdu[1].data[params[pp]][1]
				hdu.close()

	elif fit_method=='mcmc':
		indexes = ["p16","p50","p84"]
		nindexes = len(indexes)

		#=> get the fitting results
		if store_full_samplers == 1:
			bin_excld_flag = np.zeros(npixs)
			bfit_param,params = get_inferred_params_mcmc(list_name_sampler_fits=name_sampler_fits,bin_excld_flag=bin_excld_flag)

		elif store_full_samplers == 0:
			# get params
			hdu = fits.open(name_sampler_fits[npixs-1])
			params = []
			for ii in range(2,int(hdu[0].header['ncols']+1)):
				str_temp = 'col%d' % ii
				params.append(hdu[0].header[str_temp])
			hdu.close()
			nparams = len(params)

			# allocate memory
			bfit_param = {}
			for pp in range(0,nparams):
				bfit_param[params[pp]] = {}
				for ii in range(0,nindexes):
					bfit_param[params[pp]][indexes[ii]] = np.zeros(npixs) - 99.0

			# get bfit_param
			for ii in range(0,npixs):
				hdu = fits.open(name_sampler_fits[ii])
				for pp in range(0,nparams):
					bfit_param[params[pp]]["p16"][ii] = hdu[1].data[params[pp]][0]
					bfit_param[params[pp]]["p50"][ii] = hdu[1].data[params[pp]][1]
					bfit_param[params[pp]]["p84"][ii] = hdu[1].data[params[pp]][2]
				hdu.close()

	# store the maps to FITS file
	nparams = len(params)
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['gal_z'] = gal_z
	count_HDU = 2
	# pixel space
	for pp in range(0,nparams):
		for ii in range(0,nindexes):
			idx_str = "pix-%s-%s" % (params[pp],indexes[ii])
			str_temp = 'HDU%d' % int(count_HDU)
			hdr[str_temp] = idx_str
			count_HDU = count_HDU + 1
	hdr['nHDU'] = count_HDU
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)

	hdul.append(fits.ImageHDU(galaxy_region, name='galaxy_region'))

	# maps of properties
	for pp in range(0,nparams):
		for ii in range(0,nindexes):
			idx_str = "bin-%s-%s" % (params[pp],indexes[ii])
			map_prop = np.zeros((dim_y,dim_x)) - 99.0
			for jj in range(0,npixs):
				yy = int(pix_y[jj])
				xx = int(pix_x[jj])
				map_prop[yy][xx] = bfit_param[params[pp]][indexes[ii]][jj]
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


