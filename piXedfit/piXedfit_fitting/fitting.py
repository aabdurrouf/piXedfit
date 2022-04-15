import numpy as np
from math import pi, pow, sqrt, cos, sin
import sys, os
from random import randint
from astropy.io import fits
from scipy.interpolate import interp1d

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

os.environ["OMP_NUM_THREADS"] = "1"

__all__ = ["singleSEDfit", "SEDfit_from_binmap", "SEDfit_pixels_from_fluxmap", "inferred_params_mcmc_list", 
			"inferred_params_rdsps_list","get_inferred_params_mcmc", "map_params_mcmc", "get_inferred_params_rdsps", 
			"map_params_rdsps", "map_params_rdsps_from_list",  "map_params_fit_pixels_mcmc", 
			"map_params_fit_pixels_rdsps"]


def nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut):
	ngrids2 = (nwalkers*nsteps) - (nwalkers*nsteps_cut)        
	for ii in range(0,nproc):
		if ngrids2%nproc == 0:
			nproc_new = nproc
			break
		nproc = nproc  - 1
	return nproc_new


def singleSEDfit(obs_flux=[],obs_flux_err=[],filters=[],gal_z=-99.0,imf_type=1,sfh_form=4,dust_ext_law=1,add_igm_absorption=0,igm_type=0,
	duste_switch=0,add_neb_emission=1,add_agn=0,gas_logu=-2.0,logzsol_range=[-2.0,0.2],log_tau_range=[-1.0,1.2],log_age_range=[-1.0,1.14],
	log_alpha_range=[-2.0,2.0],log_beta_range=[-2.0,2.0],log_t0_range=[-1.0,1.14],dust_index_range=[-0.7,-0.7],dust1_range=[0.0,3.0],dust2_range=[0.0,3.0],
	log_gamma_range=[-3.0,-0.824],log_umin_range=[-1.0,1.176],log_qpah_range=[-1.0,0.845], z_range=[-99.0,-99.0],log_fagn_range=[-5.0,0.48],
	log_tauagn_range=[0.70,2.18],del_lognorm=[-2.0,2.0],fit_method='mcmc',likelihood='gauss', dof=2.0, gauss_likelihood_form=0, name_saved_randmod=None, 
	nrandmod=0, redc_chi2_initfit=2.0, nwalkers=100, nsteps=600, nsteps_cut=50, width_initpos=0.08, nproc=10, cosmo=0,H0=70.0,Om0=0.3,perc_chi2=10.0, 
	store_full_samplers=1, name_out_fits=None):
	"""Function for performing SED fitting to a single photometric SED.

	:param obs_flux:
		1D array containing the multiband fluxes. The number of elements should be the same as that of the input filters.

	:param obs_flux_err:
		1D array containing the multiband flux errors. The number of elements should be the same as that of the input filters.

	:param filters:
		1D list of photometric filters names. The accepted naming for the filters can be seen using :func:`list_filters` function in the :mod:`utils.filtering` module. 

	:param gal_z: (**Mandatory input, in the current version**)
		Redshift of the galaxy. If gal_z=-99.0, then redshift is set to be a free parameter in the fitting. As for the current version of **piXedfit**, photo-z hasn't been implemented.   

	:param imf_type: (default: 1)
		Choice for the IMF. Should be integer. Options are: (1)0 for Salpeter (1955), (2)1 for Chabrier (2003), and (3)2 for Kroupa (2001).

	:param sfh_form: (default: 4)
		Choice for the SFH model. Options are: (1)0 or 'tau_sfh', (2)1 or 'delayed_tau_sfh', (3)2 or 'log_normal_sfh', (4)3 or 'gaussian_sfh', (5)4 or 'double_power_sfh'.

	:param dust_ext_law: (default: 1)
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param add_igm_absorption: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Only relevant if add_igm_absorption=1. Options are: (1)0 or 'madau1995' for Madau (1995), and (2)1 or 'inoue2014' for Inoue et al. (2014).

	:param duste_switch: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission: (default: 1)
		Choice for turning on (1) or off (0) the nebular emission modeling.

	:param add_agn: (default: 0)
		Choice for turning on (1) or off (0) the AGN dusty torus emission modeling.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param del_lognorm: (default: [-2.0,2.0])
		Left and right width for the prior range in normalization (i.e., M*), 
		such that the prior range in M* is log(M*)=[log(s_best)+del_lognorm[0],log(s_best)+del_lognorm[1]].
		Only relevant if fit_method='mcmc'.

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: (1)'mcmc', and (2)'rdsps'.
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param gauss_likelihood_form: (default: 0)
		Choice for the Gaussian likelihood function. Options are: (1)0 for full/original form, and (2)1 for reduced/simplified form. 
		Only relevant for 2 cases: (1) fit_method='rdsps' and likelihood='gauss', and (2) fit_method='mcmc'.
		gauss_likelihood_form=0 means the Gaussian likelihood function uses the original/full Gaussian function form, whereas gauss_likelihood_form=1 means the likelihood function uses simplified form: P=exp(-0.5*chi^2).

	:param name_saved_randmod: (Mandatory)
		Name of the FITS file that contains pre-calculated model SEDs. This is mandatory, meaning that a set of model SEDs (stored in FITS file) should be 
		generated before performing SED fitting. The task of generatig set of random model SEDs can be done using :func:`save_models` function in :mod:`piXedfit_model` module.

	:param nrandmod: (default: 0)
		Number of random model SEDs to be generated for the initial fitting. This paremeter is only relevant if name_saved_randmod=None.
		If name_saved_randmod=None, model SEDs will be generated for conducting initial fitting. If equal to nrandmod=0, then it is set to nparams*10000

	:param redc_chi2_initfit: (default: 2.0)
		Desired reduced chi-square of the best-fit model SED in the initial fitting. This is set to estimate bulk systemtic error associated with 
		the input SED and models. It is quite possible that flux uncertainties of observed SEDs are underestimated because of underestimating or not accounting for the systematic errors. 
		These systematic errors can be due to both observational factors (e.g., associated with images processing, PSF-matching, spatial resampling and reprojection) and SED modeling aspects.
		See Section 4.1. (2nd last paragraph) in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.      

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param width_initpos: (default: 0.08)
		A factor to be multiplied with the piror range in order to set width (i.e.,sigma) of the gaussian "ball" (i.e., distribution) representing inital positions of the MCMC walkers.
		Only relevant if fit_method='mcmc'. See Section 4.2.1 in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

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

	:param name_out_fits: (optional, default: None)
		Name of output FITS file. This parameter is optional. 
	
	:returns name_out_fits:
		Output FITS file.
	"""

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'


	# get number of filters:
	nbands = len(filters)

	obs_flux = np.asarray(obs_flux)
	obs_flux_err = np.asarray(obs_flux_err)

	# make text file containing list of filters:
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
	file_out = open(name_filters_list,"w")
	for ii in range(0,nbands):
		file_out.write("%s\n" % filters[ii]) 
	file_out.close()
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc

	# make configuration file:
	name_config = "config_file%d.dat" % (randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("imf_type %d\n" % imf_type)
	file_out.write("add_neb_emission %d\n" % add_neb_emission)
	file_out.write("gas_logu %lf\n" % gas_logu)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)

	# SFH choice
	if sfh_form=='tau_sfh' or sfh_form==0:
		sfh_form1 = 0
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		sfh_form1 = 1
	elif sfh_form=='log_normal_sfh' or sfh_form==2:
		sfh_form1 = 2
	elif sfh_form=='gaussian_sfh' or sfh_form==3:
		sfh_form1 = 3
	elif sfh_form=='double_power_sfh' or sfh_form==4:
		sfh_form1 = 4
	else:
		print ("SFH choice is not recognized!")
		sys.exit()
	file_out.write("sfh_form %d\n" % sfh_form1)

	# dust law
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		dust_ext_law1 = 0
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		dust_ext_law1 = 1
	else:
		print ("dust_ext_law is not recognized!")
		sys.exit()
	file_out.write("dust_ext_law %d\n" % dust_ext_law1)

	# IGM type
	if igm_type=='madau1995' or igm_type==0:
		igm_type1 = 0
	elif igm_type=='inoue2014' or igm_type==1:
		igm_type1 = 1
	else:
		print ("igm_type is not recognized!")
		sys.exit()
	file_out.write("igm_type %d\n" % igm_type1)

	file_out.write("duste_switch %d\n" % duste_switch)
	file_out.write("add_agn %d\n" % add_agn)
	file_out.write("pr_logzsol_min %lf\n" % logzsol_range[0])
	file_out.write("pr_logzsol_max %lf\n" % logzsol_range[1])
	file_out.write("pr_log_tau_min %lf\n" % log_tau_range[0])
	file_out.write("pr_log_tau_max %lf\n" % log_tau_range[1])
	file_out.write("pr_log_alpha_min %lf\n" % log_alpha_range[0])
	file_out.write("pr_log_alpha_max %lf\n" % log_alpha_range[1])
	file_out.write("pr_log_beta_min %lf\n" % log_beta_range[0])
	file_out.write("pr_log_beta_max %lf\n" % log_beta_range[1])
	file_out.write("pr_log_t0_min %lf\n" % log_t0_range[0])
	file_out.write("pr_log_t0_max %lf\n" % log_t0_range[1])
	file_out.write("pr_log_age_min %lf\n" % log_age_range[0])
	file_out.write("pr_log_age_max %lf\n" % log_age_range[1])
	file_out.write("pr_dust_index_min %lf\n" % dust_index_range[0])
	file_out.write("pr_dust_index_max %lf\n" % dust_index_range[1])
	file_out.write("pr_dust1_min %lf\n" % dust1_range[0])
	file_out.write("pr_dust1_max %lf\n" % dust1_range[1])
	file_out.write("pr_dust2_min %lf\n" % dust2_range[0])
	file_out.write("pr_dust2_max %lf\n" % dust2_range[1])
	file_out.write("pr_z_min %lf\n" % z_range[0])
	file_out.write("pr_z_max %lf\n" % z_range[1])
	file_out.write("pr_log_gamma_min %lf\n" % log_gamma_range[0])
	file_out.write("pr_log_gamma_max %lf\n" % log_gamma_range[1])
	file_out.write("pr_log_umin_min %lf\n" % log_umin_range[0])
	file_out.write("pr_log_umin_max %lf\n" % log_umin_range[1])
	file_out.write("pr_log_qpah_min %lf\n" % log_qpah_range[0])
	file_out.write("pr_log_qpah_max %lf\n" % log_qpah_range[1])
	file_out.write("pr_log_fagn_min %lf\n" % log_fagn_range[0])
	file_out.write("pr_log_fagn_max %lf\n" % log_fagn_range[1])
	file_out.write("pr_log_tauagn_min %lf\n" % log_tauagn_range[0])
	file_out.write("pr_log_tauagn_max %lf\n" % log_tauagn_range[1])
	file_out.write("pr_del_lognorm_min %lf\n" % del_lognorm[0])
	file_out.write("pr_del_lognorm_max %lf\n" % del_lognorm[1])
	file_out.write("nwalkers %d\n" % nwalkers)
	file_out.write("nsteps %d\n" % nsteps)
	file_out.write("nsteps_cut %d\n" % nsteps_cut)
	file_out.write("width_initpos %lf\n" % width_initpos)
	file_out.write("ori_nproc %d\n" % nproc)
	file_out.write("nrandmod %d\n" % nrandmod)
	file_out.write("redc_chi2_initfit %lf\n" % redc_chi2_initfit)

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

	if name_saved_randmod == None:
		print ("name_saved_randmod is mandatory input!")
		sys.exit()
	file_out.write("name_saved_randmod %s\n" % name_saved_randmod)

	file_out.write("gauss_likelihood_form %d\n" % gauss_likelihood_form)
	# redshift
	file_out.write("gal_z %lf\n" % gal_z)

	if fit_method=='rdsps':
		file_out.write("perc_chi2 %lf\n" % perc_chi2)

	file_out.close()

	# store configuration file into temp directory:
	os.system('mv %s %s' % (name_config,temp_dir))

	# input SED text file
	name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
	file_out = open(name_SED_txt,"w")
	for ii in range(0,nbands):
		file_out.write("%e  %e\n" % (obs_flux[ii],obs_flux_err[ii]))
	file_out.close()
	os.system('mv %s %s' % (name_SED_txt,temp_dir))

	# more..
	random_int = (randint(0,10000))
	name_params_list = "params_list_%d.dat" % random_int
	name_sampler_list = "sampler_%d.dat" % random_int
	name_modif_obs_photo_SED = "modif_obs_photo_SED_%d.dat" % random_int

	# output files name:
	if name_out_fits == None:
		random_int = (randint(0,10000))
		if fit_method=='mcmc' or fit_method=='MCMC':
			name_out_fits = "mcmc_fit%d.fits" % random_int
		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_out_fits = "rdsps_fit%d.fits" % random_int

	if fit_method=='mcmc' or fit_method=='MCMC':
		if store_full_samplers==1 or store_full_samplers==True:
			os.system("mpirun -n %d python %s./mcmc_pcmod_p1.py %s %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																							name_SED_txt,name_params_list,name_sampler_list,
																							name_modif_obs_photo_SED))
			os.system('mv %s %s' % (name_params_list,temp_dir))
			os.system('mv %s %s' % (name_sampler_list,temp_dir))
			os.system('mv %s %s' % (name_modif_obs_photo_SED,temp_dir))
			os.system("mpirun -n %d python %s./mcmc_pcmod_p2.py %s %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_params_list,name_sampler_list,
																						name_modif_obs_photo_SED,name_out_fits))
		elif store_full_samplers==0 or store_full_samplers==False:
			os.system("mpirun -n %d python %s./mcmc_pcmod_nsamp_p1.py %s %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																							name_SED_txt,name_params_list,name_sampler_list,
																							name_modif_obs_photo_SED))
			os.system('mv %s %s' % (name_params_list,temp_dir))
			os.system('mv %s %s' % (name_sampler_list,temp_dir))
			os.system('mv %s %s' % (name_modif_obs_photo_SED,temp_dir))
			os.system("mpirun -n %d python %s./mcmc_pcmod_nsamp_p2.py %s %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_params_list,name_sampler_list,
																						name_modif_obs_photo_SED,name_out_fits))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()


	elif fit_method=='rdsps' or fit_method=='RDSPS':
		if store_full_samplers==1 or store_full_samplers==True:
			os.system("mpirun -n %d python %s./rdsps_pcmod.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,
																			name_out_fits))
		elif store_full_samplers==0 or store_full_samplers==False:
			os.system("mpirun -n %d python %s./rdsps_pcmod_nsamp.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,name_SED_txt,
																			name_out_fits))
		else:
			print ("Input store_full_samplers not recognized!")
			sys.exit()

	else:
		print ("The input fit_method is not recognized!")
		sys.exit()

	# free disk space:
	os.system("rm %s%s" % (temp_dir,name_config))
	os.system("rm %s%s" % (temp_dir,name_filters_list))
	os.system("rm %s%s" % (temp_dir,name_params_list))
	os.system("rm %s%s" % (temp_dir,name_sampler_list))
	os.system("rm %s%s" % (temp_dir,name_modif_obs_photo_SED))
	os.system("rm %s%s" % (temp_dir,name_SED_txt))

	return name_out_fits


def SEDfit_from_binmap(fits_binmap=None,binid_range=[],bin_ids=[],filters=None,gal_z=-99.0,imf_type=1,sfh_form=4,dust_ext_law=1,add_igm_absorption=0,
	igm_type=0,duste_switch=0,add_neb_emission=1,add_agn=0,gas_logu=-2.0,logzsol_range=[-2.0,0.2],log_tau_range=[-1.0,1.2],log_age_range=[-1.0,1.14],
	log_alpha_range=[-2.0,2.0],log_beta_range=[-2.0,2.0],log_t0_range=[-1.0,1.14],dust_index_range=[-0.7,-0.7],dust1_range=[0.0,3.0],
	dust2_range=[0.0,3.0],log_gamma_range=[-3.0,-0.824],log_umin_range=[-1.0,1.176],log_qpah_range=[-1.0,0.845], z_range=[-99.0,-99.0],
	log_fagn_range=[-5.0,0.48],log_tauagn_range=[0.70,2.18],del_lognorm=[-2.0,2.0],fit_method='mcmc',likelihood='gauss', dof=3.0, 
	gauss_likelihood_form=0, name_saved_randmod=None, nrandmod=0, redc_chi2_initfit=2.0, nwalkers=100, nsteps=600, nsteps_cut=50, 
	width_initpos=0.08, nproc=10, cosmo=0,H0=70.0,Om0=0.3,perc_chi2=10.0,store_full_samplers=1,name_out_fits=[]):

	"""A function for performing SED fitting to set of spatially resolved SEDs from the reduced data cube that is produced after the pixel binning. 

	:param fits_binmap:
		Input FITS file of reduced data cube after pixel binning. This FITS file is the one that is output by :func:`pixel_binning_photo` function in the :mod:`piXedfit_bin` module. 
		This is a mandatory parameter.

	:param binid_range: (default: [])
		Range of bin IDs that are going to be fit. Allowed format is [idmin,idmax]. The id starts from 0. If empty, [], fitting will be done to SEDs of all spatial bins.

	:param bin_ids: (default: [])
		Bin ids whose the SEDs are going to be fit. Allowed format is a 1D array. The id starts from 0. Both binid_range and bin_ids can't be empty, []. If both of them are not empty, the bin_ids will be used. 

	:param filters: (optional, default: None)
		List of photometric filters of which the fluxes are considered in the fitting. If filters=None, then the same set of filters as that stored in the header of the input FITS file would be used.

	:param gal_z: (default: -99.0)
		Redshift of the galaxy. If gal_z=-99.0, then redshift is taken from the FITS file. 
		If still gal_z=-99.0, then redshift is set to be free. **As for the current version of **piXedfit**, photo-z hasn't been implemented**.

	:param imf_type: (default: 1)
		Choice for the IMF. Should be integer. Options are: (1)0 for Salpeter (1955), (2)1 for Chabrier (2003), and (3)2 for Kroupa (2001).

	:param sfh_form: (default: 4)
		Choice for the SFH model. Options are: (1)0 or 'tau_sfh', (2)1 or 'delayed_tau_sfh', (3)2 or 'log_normal_sfh', (4)3 or 'gaussian_sfh', (5)4 or 'double_power_sfh'.

	:param dust_ext_law: (default: 1)
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param add_igm_absorption: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Only relevant if add_igm_absorption=1. Options are: (1)0 or 'madau1995' for Madau (1995), and (2)1 or 'inoue2014' for Inoue et al. (2014).

	:param duste_switch: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission: (default: 1)
		Choice for turning on (1) or off (0) the nebular emission modeling.

	:param add_agn: (default: 0)
		Choice for turning on (1) or off (0) the AGN dusty torus emission modeling.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param del_lognorm: (default: [-2.0,2.0])
		Left and right width for the prior range in normalization (i.e., M*), 
		such that the prior range in M* is log(M*)=[log(s_best)+del_lognorm[0],log(s_best)+del_lognorm[1]].
		Only relevant if fit_method='mcmc'.

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: (1)'mcmc', and (2)'rdsps'.
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param gauss_likelihood_form: (default: 0)
		Choice for the Gaussian likelihood function. Options are: (1)0 for full/original form, and (2)1 for reduced/simplified form. 
		Only relevant for 2 cases: (1) fit_method='rdsps' and likelihood='gauss', and (2) fit_method='mcmc'.
		gauss_likelihood_form=0 means the Gaussian likelihood function uses the original/full Gaussian function form, whereas gauss_likelihood_form=1 means the likelihood function uses simplified form: P=exp(-0.5*chi^2).

	:param name_saved_randmod: (Mandatory)
		Name of the FITS file that contains pre-calculated model SEDs. As for the current version of **piXedfit**, this parameter is mandatory, meaning that a set of model SEDs (stored in FITS file) should be 
		generated before performing SED fitting. The task of generatig set of random model SEDs can be done using :func:`save_models` function in :mod:`piXedfit_model` module.

	:param nrandmod: (default: 0)
		Number of random model SEDs to be generated for the initial fitting. This paremeter is only relevant if name_saved_randmod=None.
		If name_saved_randmod=None, model SEDs will be generated for conducting initial fitting. If equal to nrandmod=0, then it is set to nparams*10000

	:param redc_chi2_initfit: (default: 2.0)
		Desired reduced chi-square of the best-fit model SED in the initial fitting. This is set to estimate bulk systemtic error associated with 
		the input SED and models. It is quite possible that flux uncertainties of observed SEDs are underestimated because of underestimating or not accounting for the systematic errors. 
		These systematic errors can be due to both observational factors (e.g., associated with images processing, PSF-matching, spatial resampling and reprojection) and SED modeling aspects.
		See Section 4.1. (2nd last paragraph) in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.  

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (optional, default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param width_initpos: (default: 0.08)
		A factor to be multiplied with the piror range in order to set width (i.e.,sigma) of the gaussian "ball" (i.e., distribution) representing inital positions of the MCMC walkers.
		Only relevant if fit_method='mcmc'. See Section 4.2.1 in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

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

	:param name_out_fits: (optional, default: [])
		Names of output FITS files. This parameter is optional. If not empty, it must be in a list format with number of elements is the same as the number of bins to be fit. 
		Example: name_out_fits = ['bin1.fits', 'bin2.fits', ..., 'binN.fits'].
	"""

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_fitting/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'

	# open the FITS file of pixel binning
	hdu = fits.open(fits_binmap)
	header = hdu[0].header
	pix_bin_flag = hdu[1].data
	bin_flux = hdu['bin_flux'].data
	bin_flux_err = hdu['bin_fluxerr'].data
	hdu.close()

	# redshift
	if gal_z <= 0:
		gal_z = float(header['z'])
	
	# get original filters
	nbands_ori = int(header['nfilters'])
	filters_ori = []
	for bb in range(0,nbands_ori):
		str_temp = 'fil%d' % bb
		filters_ori.append(header[str_temp])

	if filters == None:
		nbands = nbands_ori
		filters = filters_ori
	else:
		nbands = len(filters)

	idx_fil = np.zeros(nbands)
	for ii in range(0,nbands):
		for jj in range(0,nbands_ori):
			if filters[ii] == filters_ori[jj]:
				idx_fil[ii] = jj
	idx_fil = idx_fil.astype(int)

	# transpose from (wave,y,x) => (y,x,wave)
	bin_flux_trans = np.transpose(bin_flux, axes=(1,2,0))
	bin_flux_err_trans = np.transpose(bin_flux_err, axes=(1,2,0))

	# get observed SEDs of all bins
	nbins = int(np.max(pix_bin_flag))
	unit = float(header['unit'])
	obs_flux_all = np.zeros((nbins,nbands))
	obs_flux_err_all = np.zeros((nbins,nbands))
	for bb in range(0,nbins):
		rows, cols = np.where(pix_bin_flag==bb+1)
		obs_flux_all[bb] = bin_flux_trans[rows[0]][cols[0]][idx_fil]*unit
		obs_flux_err_all[bb] = bin_flux_err_trans[rows[0]][cols[0]][idx_fil]*unit

	# make text file containing list of filters:
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
	file_out = open(name_filters_list,"w")
	for ii in range(0,nbands):
		file_out.write("%s\n" % filters[ii]) 
	file_out.close()
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	# make configuration file:
	name_config = "config_file%d.dat" % (randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("imf_type %d\n" % imf_type)

	# SFH choice
	if sfh_form=='tau_sfh' or sfh_form==0:
		sfh_form1 = 0
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		sfh_form1 = 1
	elif sfh_form=='log_normal_sfh' or sfh_form==2:
		sfh_form1 = 2
	elif sfh_form=='gaussian_sfh' or sfh_form==3:
		sfh_form1 = 3
	elif sfh_form=='double_power_sfh' or sfh_form==4:
		sfh_form1 = 4
	else:
		print ("SFH choice is not recognized!")
		sys.exit()
	file_out.write("sfh_form %d\n" % sfh_form1)

	# dust law
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		dust_ext_law1 = 0
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		dust_ext_law1 = 1
	else:
		print ("dust_ext_law is not recognized!")
		sys.exit()
	file_out.write("dust_ext_law %d\n" % dust_ext_law1)

	# IGM type
	if igm_type=='madau1995' or igm_type==0:
		igm_type1 = 0
	elif igm_type=='inoue2014' or igm_type==1:
		igm_type1 = 1
	else:
		print ("igm_type is not recognized!")
		sys.exit()
	file_out.write("igm_type %d\n" % igm_type1)

	file_out.write("add_neb_emission %d\n" % add_neb_emission)
	file_out.write("gas_logu %lf\n" % gas_logu)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)
	file_out.write("duste_switch %d\n" % duste_switch)
	file_out.write("add_agn %d\n" % add_agn)
	file_out.write("pr_logzsol_min %lf\n" % logzsol_range[0])
	file_out.write("pr_logzsol_max %lf\n" % logzsol_range[1])
	file_out.write("pr_log_tau_min %lf\n" % log_tau_range[0])
	file_out.write("pr_log_tau_max %lf\n" % log_tau_range[1])
	file_out.write("pr_log_alpha_min %lf\n" % log_alpha_range[0])
	file_out.write("pr_log_alpha_max %lf\n" % log_alpha_range[1])
	file_out.write("pr_log_beta_min %lf\n" % log_beta_range[0])
	file_out.write("pr_log_beta_max %lf\n" % log_beta_range[1])
	file_out.write("pr_log_t0_min %lf\n" % log_t0_range[0])
	file_out.write("pr_log_t0_max %lf\n" % log_t0_range[1])
	file_out.write("pr_log_age_min %lf\n" % log_age_range[0])
	file_out.write("pr_log_age_max %lf\n" % log_age_range[1])
	file_out.write("pr_dust_index_min %lf\n" % dust_index_range[0])
	file_out.write("pr_dust_index_max %lf\n" % dust_index_range[1])
	file_out.write("pr_dust1_min %lf\n" % dust1_range[0])
	file_out.write("pr_dust1_max %lf\n" % dust1_range[1])
	file_out.write("pr_dust2_min %lf\n" % dust2_range[0])
	file_out.write("pr_dust2_max %lf\n" % dust2_range[1])
	file_out.write("pr_z_min %lf\n" % z_range[0])
	file_out.write("pr_z_max %lf\n" % z_range[1])
	file_out.write("pr_log_gamma_min %lf\n" % log_gamma_range[0])
	file_out.write("pr_log_gamma_max %lf\n" % log_gamma_range[1])
	file_out.write("pr_log_umin_min %lf\n" % log_umin_range[0])
	file_out.write("pr_log_umin_max %lf\n" % log_umin_range[1])
	file_out.write("pr_log_qpah_min %lf\n" % log_qpah_range[0])
	file_out.write("pr_log_qpah_max %lf\n" % log_qpah_range[1])
	file_out.write("pr_log_fagn_min %lf\n" % log_fagn_range[0])
	file_out.write("pr_log_fagn_max %lf\n" % log_fagn_range[1])
	file_out.write("pr_log_tauagn_min %lf\n" % log_tauagn_range[0])
	file_out.write("pr_log_tauagn_max %lf\n" % log_tauagn_range[1])
	file_out.write("pr_del_lognorm_min %lf\n" % del_lognorm[0])
	file_out.write("pr_del_lognorm_max %lf\n" % del_lognorm[1])
	file_out.write("nwalkers %d\n" % nwalkers)
	file_out.write("nsteps %d\n" % nsteps)
	file_out.write("nsteps_cut %d\n" % nsteps_cut)
	file_out.write("width_initpos %lf\n" % width_initpos)
	file_out.write("ori_nproc %d\n" % nproc)
	file_out.write("nrandmod %d\n" % nrandmod)
	file_out.write("redc_chi2_initfit %lf\n" % redc_chi2_initfit)

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

	if name_saved_randmod == None:
		print ("name_saved_randmod is mandatory input!")
		sys.exit()
	file_out.write("name_saved_randmod %s\n" % name_saved_randmod)

	file_out.write("gauss_likelihood_form %d\n" % gauss_likelihood_form)
	# redshift
	file_out.write("gal_z %lf\n" % gal_z)

	if fit_method=='rdsps':
		file_out.write("perc_chi2 %lf\n" % perc_chi2) 

	file_out.close()

	# store configuration file into temp directory
	os.system('mv %s %s' % (name_config,temp_dir))

	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc

	nbins_calc = 0
	if len(bin_ids)>0:
		if len(binid_range) > 0:
			print ("Both bin_ids and binid_range are not empty, so calculation will be done based on bin_ids.")
		bin_ids = np.asarray(bin_ids)
		nbins_calc = len(bin_ids)

	elif len(bin_ids)==0:
		if len(binid_range) == 0:
			print ("Both bin_ids and binid_range are empty, so SED fitting will be done to all the bins.")
			bin_ids = np.zeros(nbins)
			for ii in range(0,nbins):
				bin_ids[ii] = ii
		elif len(binid_range) > 0:
			binid_min = binid_range[0]
			binid_max = binid_range[1]
			nbins_calc = binid_max - binid_min
			bin_ids = np.zeros(nbins_calc)
			count_id = 0
			for ii in range(binid_min,binid_max):
				bin_ids[int(count_id)] = ii
				count_id = count_id + 1

	if 0<len(name_out_fits)<nbins_calc:
		print ("The number of elements in name_out_fits should be the same as the number of bins to be calculated!")
		sys.exit()

	if fit_method=='mcmc' or fit_method=='MCMC':
		count_id = 0
		for idx_bin in bin_ids:
			# SED of bin
			obs_flux = obs_flux_all[int(idx_bin)]
			obs_flux_err = obs_flux_err_all[int(idx_bin)]

			# input SED text file
			name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
			file_out = open(name_SED_txt,"w")
			for ii in range(0,nbands):
				file_out.write("%e  %e\n" % (obs_flux[ii],obs_flux_err[ii]))
			file_out.close()
			os.system('mv %s %s' % (name_SED_txt,temp_dir))

			# name of output FITS file
			if len(name_out_fits) == 0:
				name_out_fits1 = "mcmc_bin%d.fits" % (idx_bin+1)
			else:
				name_out_fits1 = name_out_fits[int(count_id)]

			# more..
			random_int = (randint(0,10000))
			name_params_list = "params_list_%d.dat" % random_int
			name_sampler_list = "sampler_%d.dat" % random_int
			name_modif_obs_photo_SED = "modif_obs_photo_SED_%d.dat" % random_int

			if store_full_samplers==1 or store_full_samplers==True:
				os.system("mpirun -n %d python %s./mcmc_pcmod_p1.py %s %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																							name_SED_txt,name_params_list,name_sampler_list,
																							name_modif_obs_photo_SED))
				os.system('mv %s %s' % (name_params_list,temp_dir))
				os.system('mv %s %s' % (name_sampler_list,temp_dir))
				os.system('mv %s %s' % (name_modif_obs_photo_SED,temp_dir))
				os.system("mpirun -n %d python %s./mcmc_pcmod_p2.py %s %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_params_list,name_sampler_list,
																						name_modif_obs_photo_SED,name_out_fits1))
				os.system("rm %s%s" % (temp_dir,name_SED_txt))
				os.system("rm %s%s" % (temp_dir,name_params_list))
				os.system("rm %s%s" % (temp_dir,name_sampler_list))
				os.system("rm %s%s" % (temp_dir,name_modif_obs_photo_SED))

			elif store_full_samplers==0 or store_full_samplers==False:
				os.system("mpirun -n %d python %s./mcmc_pcmod_nsamp_p1.py %s %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																							name_SED_txt,name_params_list,name_sampler_list,
																							name_modif_obs_photo_SED))
				os.system('mv %s %s' % (name_params_list,temp_dir))
				os.system('mv %s %s' % (name_sampler_list,temp_dir))
				os.system('mv %s %s' % (name_modif_obs_photo_SED,temp_dir))
				os.system("mpirun -n %d python %s./mcmc_pcmod_nsamp_p2.py %s %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																						name_params_list,name_sampler_list,
																						name_modif_obs_photo_SED,name_out_fits1))
				os.system("rm %s%s" % (temp_dir,name_SED_txt))
				os.system("rm %s%s" % (temp_dir,name_params_list))
				os.system("rm %s%s" % (temp_dir,name_sampler_list))
				os.system("rm %s%s" % (temp_dir,name_modif_obs_photo_SED))
			else:
				print ("Input store_full_samplers not recognized!")
				sys.exit()

			count_id = count_id + 1

	elif fit_method=='rdsps' or fit_method=='RDSPS':
		# input SEDs
		name_inputSEDs = "input_SEDs_%d.dat" % (randint(0,10000))
		file_out = open(name_inputSEDs,"w")
		for idx_bin in bin_ids:
			for bb in range(0,nbands):
				file_out.write("%e  " % obs_flux_all[int(idx_bin)][bb])
			for bb in range(0,nbands-1):
				file_out.write("%e  " % obs_flux_err_all[int(idx_bin)][bb])
			file_out.write("%e\n" % obs_flux_err_all[int(idx_bin)][nbands-1])
		file_out.close()
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
			os.system("mpirun -n %d python %s./rdsps_pcmod_bulk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_inputSEDs,name_outs))
		elif store_full_samplers==0 or store_full_samplers==False:
			os.system("mpirun -n %d python %s./rdsps_pcmod_nsamp_bulk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
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


def SEDfit_pixels_from_fluxmap(fits_fluxmap=None,x_range=[],y_range=[],filters=None,gal_z=-99.0,imf_type=1,sfh_form=4,dust_ext_law=1,add_igm_absorption=0,
	igm_type=0,duste_switch=0,add_neb_emission=1,add_agn=0,gas_logu=-2.0,logzsol_range=[-2.0,0.2],log_tau_range=[-1.0,1.2],log_age_range=[-1.0,1.14],
	log_alpha_range=[-2.0,2.0],log_beta_range=[-2.0,2.0],log_t0_range=[-1.0,1.14],dust_index_range=[-0.7,-0.7],dust1_range=[0.0,3.0],
	dust2_range=[0.0,3.0],log_gamma_range=[-3.0,-0.824],log_umin_range=[-1.0,1.176],log_qpah_range=[-1.0,0.845], z_range=[-99.0,-99.0],
	log_fagn_range=[-5.0,0.48],log_tauagn_range=[0.70,2.18],del_lognorm=[-2.0,2.0],fit_method='mcmc',likelihood='gauss', dof=3.0, 
	gauss_likelihood_form=0, name_saved_randmod=None, nrandmod=0, redc_chi2_initfit=2.0, nwalkers=100, nsteps=600, nsteps_cut=50, 
	width_initpos=0.08, nproc=10, cosmo=0,H0=70.0,Om0=0.3,perc_chi2=10.0,store_full_samplers=1):

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

	:param filters: (optional, default: None)
		List of photometric filters of which the fluxes are considered in the fitting. If filters=None, 
		then the same set of filters as that in the data cube would be used.

	:param gal_z: (default: -99.0)
		Redshift of the galaxy. If gal_z=-99.0, then redshift is taken from the FITS file. 
		If still gal_z=-99.0, then redshift is set to be free. **As for the current version of **piXedfit**, photo-z hasn't been implemented**.

	:param imf_type: (default: 1)
		Choice for the IMF. Should be integer. Options are: (1)0 for Salpeter (1955), (2)1 for Chabrier (2003), and (3)2 for Kroupa (2001).

	:param sfh_form: (default: 4)
		Choice for the SFH model. Options are: (1)0 or 'tau_sfh', (2)1 or 'delayed_tau_sfh', (3)2 or 'log_normal_sfh', (4)3 or 'gaussian_sfh', (5)4 or 'double_power_sfh'.

	:param dust_ext_law: (default: 1)
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param add_igm_absorption: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Only relevant if add_igm_absorption=1. 
		Options are: (1)0 or 'madau1995' for Madau (1995), and (2)1 or 'inoue2014' for Inoue et al. (2014).

	:param duste_switch: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission: (default: 1)
		Choice for turning on (1) or off (0) the nebular emission modeling.

	:param add_agn: (default: 0)
		Choice for turning on (1) or off (0) the AGN dusty torus emission modeling.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param del_lognorm: (default: [-2.0,2.0])
		Left and right width for the prior range in normalization (i.e., M*), 
		such that the prior range in M* is log(M*)=[log(s_best)+del_lognorm[0],log(s_best)+del_lognorm[1]].
		Only relevant if fit_method='mcmc'.

	:param fit_method: (default: 'mcmc')
		Choice for the fitting method. Options are: (1)'mcmc', and (2)'rdsps'.
	
	:param likelihood: (default: 'gauss')
		Choice of likelihood function for the RDSPS method. Only relevant if the fit_method='rdsps'. Options are: (1)'gauss', and (2) 'student_t'.

	:param dof: (default: 2.0)
		Degree of freedom (nu) in the Student's likelihood function. Only relevant if the fit_method='rdsps' and likelihood='student_t'.

	:param gauss_likelihood_form: (default: 0)
		Choice for the Gaussian likelihood function. Options are: (1)0 for full/original form, and (2)1 for reduced/simplified form. 
		Only relevant for 2 cases: (1) fit_method='rdsps' and likelihood='gauss', and (2) fit_method='mcmc'.
		gauss_likelihood_form=0 means the Gaussian likelihood function uses the original/full Gaussian function form, whereas gauss_likelihood_form=1 means the likelihood function uses simplified form: P=exp(-0.5*chi^2).

	:param name_saved_randmod: (Mandatory)
		Name of the FITS file that contains pre-calculated model SEDs. As for the current version of **piXedfit**, this parameter is mandatory, meaning that a set of model SEDs (stored in FITS file) should be 
		generated before performing SED fitting. The task of generatig set of random model SEDs can be done using :func:`save_models` function in :mod:`piXedfit_model` module.

	:param nrandmod: (default: 0)
		Number of random model SEDs to be generated for the initial fitting. This paremeter is only relevant if name_saved_randmod=None.
		If name_saved_randmod=None, model SEDs will be generated for conducting initial fitting. If equal to nrandmod=0, then it is set to nparams*10000

	:param redc_chi2_initfit: (default: 2.0)
		Desired reduced chi-square of the best-fit model SED in the initial fitting. This is set to estimate bulk systemtic error associated with 
		the input SED and models. It is quite possible that flux uncertainties of observed SEDs are underestimated because of underestimating or not accounting for the systematic errors. 
		These systematic errors can be due to both observational factors (e.g., associated with images processing, PSF-matching, spatial resampling and reprojection) and SED modeling aspects.
		See Section 4.1. (2nd last paragraph) in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.  

	:param nwalkers: (default: 100)
		Number of walkers to be set in the MCMC process. This parameter is only applicable if fit_method='mcmc'.

	:param nsteps: (default: 600)
		Number of steps for each walker in the MCMC process. Only relevant if fit_method='mcmc'.

	:param nsteps_cut: (optional, default: 50)
		Number of first steps of each walkers that will be cut when constructing the final sampler chains. Only relevant if fit_method='mcmc'.

	:param width_initpos: (default: 0.08)
		A factor to be multiplied with the piror range in order to set width (i.e.,sigma) of the gaussian "ball" (i.e., distribution) representing inital positions of the MCMC walkers.
		Only relevant if fit_method='mcmc'. See Section 4.2.1 in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

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

	# dimension
	dim_y = gal_region.shape[0]
	dim_x = gal_region.shape[1]

	# redshift
	if gal_z <= 0:
		gal_z = float(header['z'])
	
	# get original filters
	nbands_ori = int(header['nfilters'])
	filters_ori = []
	for bb in range(0,nbands_ori):
		str_temp = 'fil%d' % bb
		filters_ori.append(header[str_temp])

	if filters == None:
		nbands = nbands_ori
		filters = filters_ori
	else:
		nbands = len(filters)

	idx_fil = np.zeros(nbands)
	for ii in range(0,nbands):
		for jj in range(0,nbands_ori):
			if filters[ii] == filters_ori[jj]:
				idx_fil[ii] = jj
	idx_fil = idx_fil.astype(int)

	# transpose from (wave,y,x) => (y,x,wave)
	map_flux_trans = np.transpose(map_flux, axes=(1,2,0))
	map_flux_err_trans = np.transpose(map_flux_err, axes=(1,2,0))

	# Define x- and y-ranges
	if len(x_range)==0:
		xmin, xmax = 0, dim_x
	else:
		if x_range[0]<0 or x_range[1]>dim_x:
			print ("Can't perform SED fitting to region beyond the dimension of the data cube!")
			sys.exit()
		else:
			xmin, xmax = x_range[0], x_range[1] 

	if len(y_range)==0:
		ymin, ymax = 0, dim_y
	else:
		if y_range[0]<0 or y_range[1]>dim_y:
			print ("Can't perform SED fitting to region beyond the dimension of the data cube!")
			sys.exit()
		else:
			ymin, ymax = y_range[0], y_range[1]

	# unit of flux:
	unit = float(header['unit'])

	# make text file containing list of filters:
	name_filters_list = "filters_list%d.dat" % (randint(0,10000))
	file_out = open(name_filters_list,"w")
	for ii in range(0,nbands):
		file_out.write("%s\n" % filters[ii]) 
	file_out.close()
	os.system('mv %s %s' % (name_filters_list,temp_dir))

	# make configuration file:
	name_config = "config_file%d.dat" % (randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("imf_type %d\n" % imf_type)

	# SFH choice
	if sfh_form=='tau_sfh' or sfh_form==0:
		sfh_form1 = 0
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		sfh_form1 = 1
	elif sfh_form=='log_normal_sfh' or sfh_form==2:
		sfh_form1 = 2
	elif sfh_form=='gaussian_sfh' or sfh_form==3:
		sfh_form1 = 3
	elif sfh_form=='double_power_sfh' or sfh_form==4:
		sfh_form1 = 4
	else:
		print ("SFH choice is not recognized!")
		sys.exit()
	file_out.write("sfh_form %d\n" % sfh_form1)

	# dust law
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		dust_ext_law1 = 0
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		dust_ext_law1 = 1
	else:
		print ("dust_ext_law is not recognized!")
		sys.exit()
	file_out.write("dust_ext_law %d\n" % dust_ext_law1)

	# IGM type
	if igm_type=='madau1995' or igm_type==0:
		igm_type1 = 0
	elif igm_type=='inoue2014' or igm_type==1:
		igm_type1 = 1
	else:
		print ("igm_type is not recognized!")
		sys.exit()
	file_out.write("igm_type %d\n" % igm_type1)

	file_out.write("add_neb_emission %d\n" % add_neb_emission)
	file_out.write("gas_logu %lf\n" % gas_logu)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)
	file_out.write("duste_switch %d\n" % duste_switch)
	file_out.write("add_agn %d\n" % add_agn)
	file_out.write("pr_logzsol_min %lf\n" % logzsol_range[0])
	file_out.write("pr_logzsol_max %lf\n" % logzsol_range[1])
	file_out.write("pr_log_tau_min %lf\n" % log_tau_range[0])
	file_out.write("pr_log_tau_max %lf\n" % log_tau_range[1])
	file_out.write("pr_log_alpha_min %lf\n" % log_alpha_range[0])
	file_out.write("pr_log_alpha_max %lf\n" % log_alpha_range[1])
	file_out.write("pr_log_beta_min %lf\n" % log_beta_range[0])
	file_out.write("pr_log_beta_max %lf\n" % log_beta_range[1])
	file_out.write("pr_log_t0_min %lf\n" % log_t0_range[0])
	file_out.write("pr_log_t0_max %lf\n" % log_t0_range[1])
	file_out.write("pr_log_age_min %lf\n" % log_age_range[0])
	file_out.write("pr_log_age_max %lf\n" % log_age_range[1])
	file_out.write("pr_dust_index_min %lf\n" % dust_index_range[0])
	file_out.write("pr_dust_index_max %lf\n" % dust_index_range[1])
	file_out.write("pr_dust1_min %lf\n" % dust1_range[0])
	file_out.write("pr_dust1_max %lf\n" % dust1_range[1])
	file_out.write("pr_dust2_min %lf\n" % dust2_range[0])
	file_out.write("pr_dust2_max %lf\n" % dust2_range[1])
	file_out.write("pr_z_min %lf\n" % z_range[0])
	file_out.write("pr_z_max %lf\n" % z_range[1])
	file_out.write("pr_log_gamma_min %lf\n" % log_gamma_range[0])
	file_out.write("pr_log_gamma_max %lf\n" % log_gamma_range[1])
	file_out.write("pr_log_umin_min %lf\n" % log_umin_range[0])
	file_out.write("pr_log_umin_max %lf\n" % log_umin_range[1])
	file_out.write("pr_log_qpah_min %lf\n" % log_qpah_range[0])
	file_out.write("pr_log_qpah_max %lf\n" % log_qpah_range[1])
	file_out.write("pr_log_fagn_min %lf\n" % log_fagn_range[0])
	file_out.write("pr_log_fagn_max %lf\n" % log_fagn_range[1])
	file_out.write("pr_log_tauagn_min %lf\n" % log_tauagn_range[0])
	file_out.write("pr_log_tauagn_max %lf\n" % log_tauagn_range[1])
	file_out.write("pr_del_lognorm_min %lf\n" % del_lognorm[0])
	file_out.write("pr_del_lognorm_max %lf\n" % del_lognorm[1])
	file_out.write("nwalkers %d\n" % nwalkers)
	file_out.write("nsteps %d\n" % nsteps)
	file_out.write("nsteps_cut %d\n" % nsteps_cut)
	file_out.write("width_initpos %lf\n" % width_initpos)
	file_out.write("ori_nproc %d\n" % nproc)
	file_out.write("nrandmod %d\n" % nrandmod)
	file_out.write("redc_chi2_initfit %lf\n" % redc_chi2_initfit)

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

	if name_saved_randmod == None:
		print ("name_saved_randmod is mandatory input!")
		sys.exit()
	file_out.write("name_saved_randmod %s\n" % name_saved_randmod)

	file_out.write("gauss_likelihood_form %d\n" % gauss_likelihood_form)
	# redshift
	file_out.write("gal_z %lf\n" % gal_z)

	if fit_method=='rdsps':
		file_out.write("perc_chi2 %lf\n" % perc_chi2) 

	file_out.close()

	# store configuration file into temp directory
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
					# observed SED:
					obs_flux = map_flux_trans[yy][xx][idx_fil]*unit
					obs_flux_err = map_flux_err_trans[yy][xx][idx_fil]*unit

					# input SED text file
					name_SED_txt = "inputSED_file%d.dat" % (randint(0,20000))
					file_out = open(name_SED_txt,"w")
					for ii in range(0,nbands):
						file_out.write("%e  %e\n" % (obs_flux[ii],obs_flux_err[ii]))
					file_out.close()
					os.system('mv %s %s' % (name_SED_txt,temp_dir))

					# name of output FITS file
					name_out_fits = "pix_y%d_x%d_mcmc.fits" % (yy,xx)

					# more..
					random_int = (randint(0,10000))
					name_params_list = "params_list_%d.dat" % random_int
					name_sampler_list = "sampler_%d.dat" % random_int
					name_modif_obs_photo_SED = "modif_obs_photo_SED_%d.dat" % random_int

					if store_full_samplers==1 or store_full_samplers==True:
						os.system("mpirun -n %d python %s./mcmc_pcmod_p1.py %s %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																									name_SED_txt,name_params_list,name_sampler_list,
																									name_modif_obs_photo_SED))
						os.system('mv %s %s' % (name_params_list,temp_dir))
						os.system('mv %s %s' % (name_sampler_list,temp_dir))
						os.system('mv %s %s' % (name_modif_obs_photo_SED,temp_dir))
						os.system("mpirun -n %d python %s./mcmc_pcmod_p2.py %s %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																								name_params_list,name_sampler_list,
																								name_modif_obs_photo_SED,name_out_fits))
						os.system("rm %s%s" % (temp_dir,name_SED_txt))
						os.system("rm %s%s" % (temp_dir,name_params_list))
						os.system("rm %s%s" % (temp_dir,name_sampler_list))
						os.system("rm %s%s" % (temp_dir,name_modif_obs_photo_SED))

					elif store_full_samplers==0 or store_full_samplers==False:
						os.system("mpirun -n %d python %s./mcmc_pcmod_nsamp_p1.py %s %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,
																									name_SED_txt,name_params_list,name_sampler_list,
																									name_modif_obs_photo_SED))
						os.system('mv %s %s' % (name_params_list,temp_dir))
						os.system('mv %s %s' % (name_sampler_list,temp_dir))
						os.system('mv %s %s' % (name_modif_obs_photo_SED,temp_dir))
						os.system("mpirun -n %d python %s./mcmc_pcmod_nsamp_p2.py %s %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																								name_params_list,name_sampler_list,
																								name_modif_obs_photo_SED,name_out_fits))
						os.system("rm %s%s" % (temp_dir,name_SED_txt))
						os.system("rm %s%s" % (temp_dir,name_params_list))
						os.system("rm %s%s" % (temp_dir,name_sampler_list))
						os.system("rm %s%s" % (temp_dir,name_modif_obs_photo_SED))
					else:
						print ("Input store_full_samplers not recognized!")
						sys.exit()

	elif fit_method=='rdsps' or fit_method=='RDSPS':
		# input SEDs
		name_inputSEDs = "input_SEDs_%d.dat" % (randint(0,10000))
		file_out1 = open(name_inputSEDs,"w")

		# name output
		name_outs = "name_outs_%d.dat" % (randint(0,10000))
		file_out2 = open(name_outs,"w")

		for yy in range(ymin,ymax):
			for xx in range(xmin,xmax):
				if gal_region[yy][xx] == 1:
					# observed SED
					obs_flux = map_flux_trans[yy][xx][idx_fil]*unit
					obs_flux_err = map_flux_err_trans[yy][xx][idx_fil]*unit
					for bb in range(0,nbands):
						file_out1.write("%e  " % obs_flux[bb])
					for bb in range(0,nbands-1):
						file_out1.write("%e  " % obs_flux_err[bb])
					file_out1.write("%e\n" % obs_flux_err[nbands-1])

					# name output
					name0 = "pix_y%d_x%d_rdsps.fits" % (yy,xx)
					file_out2.write("%s\n" % name0)

		file_out1.close()
		file_out2.close()

		os.system('mv %s %s' % (name_inputSEDs,temp_dir))
		os.system('mv %s %s' % (name_outs,temp_dir))

		if store_full_samplers==1 or store_full_samplers==True:
			os.system("mpirun -n %d python %s./rdsps_pcmod_bulk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
																				name_inputSEDs,name_outs))
		elif store_full_samplers==0 or store_full_samplers==False:
			os.system("mpirun -n %d python %s./rdsps_pcmod_nsamp_bulk.py %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,
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
		idx_sel = np.where((data_samplers['chi2']<=crit_chi2) & (data_samplers['log_sfr']>-29.0) & (np.isnan(data_samplers['prob'])==False) & (np.isinf(data_samplers['prob'])==False))

		array_prob = data_samplers['prob'][idx_sel[0]]
		tot_prob = np.sum(array_prob)

		params_bfits = np.zeros((nparams,2))
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


def map_params_mcmc(fits_binmap=None, bin_ids=[], name_sampler_fits=[], fits_fluxmap=None, refband_SFR=None, 
	refband_SM=None, refband_Mdust=None, name_out_fits=None):
	"""Function for calculating maps of properties from the fitting results obtained with the MCMC method.

	:param fits_binmap:
		FITS file containing the reduced data after the pixel binning process, which is output of the :func:`pixel_binning_photo` function in the :mod:`piXedfit_bin` module.

	:param bin_ids:
		Bin indices of the FITS files listed in name_sampler_fits input. Allowed format is a 1D array. The id starts from 0. 

	:param name_sampler_fits:
		List of the names of the FITS files containing the fitting results of spatial bins. This should have the same number of element as that of bin_ids. 
		The number of element doesn't necessarily the same as the number of bins. A missing bin will be ignored in the creation of the maps of properties.

	:param fits_fluxmap:
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

	:param bin_id_exclude: (optional, default: [])
		Indexes of bins that are going to be excluded or ignored in the derivation of the maps of properties. 

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
	"""

	if len(bin_ids) != len(name_sampler_fits):
		print ("Both bin_ids and name_sampler_fits are required and they should have the same number of elements!")
		sys.exit()

	# open the FITS file containing the pixel binning map
	hdu = fits.open(fits_binmap)
	pix_bin_flag = hdu[1].data
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

	# inspect whether the FITS fitting results contains sampler chains or only the inferred parameters
	hdu = fits.open(name_sampler_fits[0])
	header_samplers = hdu[0].header
	if header_samplers['col1'] == 'rows':
		store_full_samplers = 0
	elif header_samplers['col1'] == 'id':
		store_full_samplers = 1
	hdu.close()

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
			idx_sel = np.where((data_samplers['chi2']<=crit_chi2) & (data_samplers['log_sfr']>-29.0) & (np.isnan(data_samplers['prob'])==False) & (np.isinf(data_samplers['prob'])==False))

			array_prob = data_samplers['prob'][idx_sel[0]]
			tot_prob = np.sum(array_prob)

			for pp in range(0,nparams):
				array_val = data_samplers[params[pp]][idx_sel[0]]

				mean_val = np.sum(array_val*array_prob)/tot_prob
				mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
				std_val = sqrt(abs(mean_val2 - (mean_val**2)))

				bfit_param[params[pp]]["mean"][ii] = mean_val
				bfit_param[params[pp]]["mean_err"][ii] = std_val

	return bfit_param,params,indexes



def map_params_rdsps(fits_binmap=None, bin_ids=[], name_sampler_fits=[], fits_fluxmap=None, refband_SFR=None, refband_SM=None, 
	refband_Mdust=None, bin_id_exclude=[], perc_chi2=10.0, name_out_fits=None):
	"""Function for calculating maps of properties from the fitting results obtained with the RDSPS method.

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

	:param bin_id_exclude: (optional, default: [])
		Indexes of bins that are going to be excluded or ignored in the derivation of the maps of properties.

	:param perc_chi2: (optional, default=5.0)
		A parameter that set the percentile cut of the random model SEDs. The cut is applied after the models are sorted based on their chi-square values. 
		This parameter defines up to what percentile the models will be cut.  

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
	"""

	# open the FITS file containing the pixel binning map
	hdu = fits.open(fits_binmap)
	pix_bin_flag = hdu[1].data
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

	# inspect whether the FITS fitting results contains sampler chains or only the inferred parameters
	hdu = fits.open(name_sampler_fits[0])
	header_samplers = hdu[0].header
	if header_samplers['col1'] == 'rows':
		store_full_samplers = 0
	elif header_samplers['col1'] == 'id':
		store_full_samplers = 1
	hdu.close()

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
	hdr['nfilters'] = nbands
	for bb in range(0,nbands):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = filters[bb]

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

	return name_out_fits


def map_params_rdsps_from_list(fits_binmap=None, fits_fluxmap=None, fit_results=None, refband_SFR=None, refband_SM=None, 
	refband_Mdust=None, name_out_fits=None):
	"""Function for calculating maps of properties from the fitting results obtained with the RDSPS method.

	:param fits_binmap: (Mandatory, default=None)
		FITS file containing the reduced data after the pixel binning process, which is output of the :func:`pixel_binning_photo` function in the :mod:`piXedfit_bin` module.

	:param fits_fluxmap: (Mandatory, default: None)
		FITS file containing reduced maps of multiband fluxes, which is output of the :func:`flux_map` in the :class:`images_processing` class in the :mod:`piXedfit_images` module.

	:param fit_results: (Mandatory, default: None)
		FITS file containing results of fitting with RDSPS.

	:param refband_SFR: (default: None)
		Index of band in the multiband set that is used for reference in dividing map of SFR in bin space into map of SFR in pixel space.
		If None, the band with shortest wavelength is selected.

	:param refband_SM: (default: None)
		Index of band in the multiband set that is used for reference in dividing map of stellar mass in bin space into map of stellar mass in pixel space.
		If None, the band with longest wavelength is selected.

	:param refband_Mdust: (default: None)
		Index of band/filter in the multiband set that is used for reference in dividing map of dust mass in bin space into map of dust mass in pixel space.
		If None, the band with longest wavelength is selected.

	:param bin_id_exclude: (optional, default: [])
		Indexes of bins that are going to be excluded or ignored in the derivation of the maps of properties.

	:param idx_exclude: (optional, defult: [])
		List of model indexes that are intended to be exlcuded in the calculation of posterior-weighted averaging.

	:param perc_chi2: (optional, default=5.0)
		A parameter that set the percentile cut of the random model SEDs. The cut is applied after the models are sorted based on their chi-square values. 
		This parameter defines up to what percentile the models will be cut.  

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.   
	"""

	# open the FITS file containing pixel binning map
	hdu = fits.open(fits_binmap)
	header = hdu[0].header
	pix_bin_flag = hdu[1].data
	unit_bin = float(header['unit'])
	bin_flux = hdu['bin_flux'].data*unit_bin
	bin_flux_err = hdu['bin_fluxerr'].data*unit_bin
	nbins = int(header['nbins'])
	hdu.close()

	# open FITS file containing multiband fluxes maps
	hdu = fits.open(fits_fluxmap)
	header = hdu[0].header
	unit_pix = float(header['unit'])
	galaxy_region = hdu['galaxy_region'].data
	pix_flux = hdu['flux'].data*unit_pix
	pix_flux_err = hdu['flux_err'].data*unit_pix
	gal_z = float(header['z'])
	nbands = int(header['nfilters'])
	filters = []
	for ii in range(0,nbands):
		str_temp = 'fil%d' % ii 
		filters.append(header[str_temp])
	hdu.close()

	dim_y = pix_bin_flag.shape[0]
	dim_x = pix_bin_flag.shape[1]

	# open FITS file containing results of fitting 
	hdu = fits.open(fit_results)
	fit_header = hdu[0].header
	fit_files = hdu[1].data
	fit_mean = hdu['mean'].data
	fit_std = hdu['std'].data
	hdu.close()

	# check if some bins are missing:
	bin_idx = np.zeros(nbins) - 99.0
	for ii in range(0,nbins):
		str_temp0 = 'rdsps_bin%d.fits' % (ii+1)
		str_temp1 = 'rdsps_bin%d.fits.gz' % (ii+1)
		idx0 = np.where((fit_files['name']==str_temp0) | (fit_files['name']==str_temp1))
		if len(idx0[0])>0:
			bin_idx[ii] = idx0[0][0]

	# get parameters:
	nparams = int(fit_header['nparams'])
	params = []
	for ii in range(0,nparams):
		str_temp = 'param%d' % ii
		params.append(fit_header[str_temp])

	indexes = ["mean", "mean_err"]
	nindexes = len(indexes)

	bfit_param = {}
	for pp in range(0,nparams):
		bfit_param[params[pp]] = {}
		bfit_param[params[pp]]["mean"] = np.zeros(nbins) - 99.0
		bfit_param[params[pp]]["mean_err"] = np.zeros(nbins) - 99.0
		for ii in range(0,nbins):
			if bin_idx[ii]>=0:
				bfit_param[params[pp]]["mean"][ii] = fit_mean[params[pp]][int(bin_idx[ii])] 
				bfit_param[params[pp]]["mean_err"][ii] = fit_std[params[pp]][int(bin_idx[ii])]


	# Make FITS file:
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['gal_z'] = gal_z
	hdr['nbins'] = nbins
	hdr['nfilters'] = nbands
	for bb in range(0,nbands):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = filters[bb]

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

	return name_out_fits



def map_params_fit_pixels_mcmc(fits_fluxmap=None, pix_x=[], pix_y=[], name_sampler_fits=[], name_out_fits=None):
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
		print ("pix_x and pix_y should have the same number of elements as that of name_sampler_fits!")
		sys.exit()

	# open FITS file containing maps of multiband fluxes
	hdu = fits.open(fits_fluxmap)
	galaxy_region = hdu['galaxy_region'].data
	gal_z = float(hdu[0].header['z'])
	hdu.close()

	dim_y = galaxy_region.shape[0]
	dim_x = galaxy_region.shape[1]

	# inspect whether the FITS fitting results contains sampler chains or only the inferred parameters
	hdu = fits.open(name_sampler_fits[0])
	header_samplers = hdu[0].header
	if header_samplers['col1'] == 'rows':
		store_full_samplers = 0
	elif header_samplers['col1'] == 'id':
		store_full_samplers = 1
	hdu.close()

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

	nparams = len(params)
	# Make FITS file:
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

	return name_out_fits



def map_params_fit_pixels_rdsps(fits_fluxmap=None, pix_x=[], pix_y=[], name_sampler_fits=[], name_out_fits=None):
	"""Function for calculating maps of properties from the fitting results obtained with the RDSPS method on pixel basis.

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
		print ("pix_x and pix_y should have the same number of elements as that of name_sampler_fits!")
		sys.exit()

	# open FITS file containing maps of multiband fluxes
	hdu = fits.open(fits_fluxmap)
	galaxy_region = hdu['galaxy_region'].data
	gal_z = float(hdu[0].header['z'])
	hdu.close()

	dim_y = galaxy_region.shape[0]
	dim_x = galaxy_region.shape[1]

	# inspect whether the FITS fitting results contains sampler chains or only the inferred parameters
	hdu = fits.open(name_sampler_fits[0])
	header_samplers = hdu[0].header
	if header_samplers['col1'] == 'rows':
		store_full_samplers = 0
	elif header_samplers['col1'] == 'id':
		store_full_samplers = 1
	hdu.close()

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

	nparams = len(params)
	# Make FITS file:
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

	return name_out_fits



