import h5py
import sys, os
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm
from math import gamma

from ..piXedfit_model.model_utils import default_params_val, calc_mw_age, construct_SFH


__all__ = ["nproc_reduced", "randname", "write_filters_list", "write_input_singleSED_photo", "write_input_specphoto_hdf5", 
			"write_arbitprior", "write_joint_prior", "write_conf_file", "read_config_file_fit", "get_nproc", "define_free_z",
			"get_name_out_fits", "remove_files", "run_fitting", "write_input_spec_hdf5", "make_bins_name_out_fits", 
			"define_free_z_bins_fits", "save_fitting_results_db", "compute_chisq", "fit_sed_chisq", "fit_sed_batch", "bayesian_fit_sed_batch"]

def nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut):
	ngrids2 = (nwalkers*nsteps) - (nwalkers*nsteps_cut)        
	for ii in range(0,nproc):
		if ngrids2%nproc == 0:
			nproc_new = nproc
			break
		nproc = nproc  - 1
	return nproc_new

def randname(initial,ext):
	return initial+str(np.random.randint(50000))+ext

def write_filters_list(temp_dir,filters):
	name_filters_list = randname("filters_list",".dat")
	file_out = open(name_filters_list,"w")
	for bb in range(len(filters)):
		file_out.write("%s\n" % filters[bb]) 
	file_out.close()
	os.system('mv %s %s' % (name_filters_list,temp_dir))
	return name_filters_list

def write_input_singleSED_photo(temp_dir,obs_flux,obs_flux_err):
	inputSED_file = randname("inputSED_file",".dat")
	file_out = open(inputSED_file,"w")
	for bb in range(0,len(obs_flux)):
		file_out.write("%e  %e\n" % (obs_flux[bb],obs_flux_err[bb]))
	file_out.close()
	os.system('mv %s %s' % (inputSED_file,temp_dir))
	return inputSED_file

def write_input_specphoto_hdf5(temp_dir,obs_flux,obs_flux_err,spec_wave,spec_flux,spec_flux_err,wavelength_range):
	inputSED_file = randname("inputSED_file",".hdf5")

	if wavelength_range is not None:
		idx0 = np.where((spec_wave>=wavelength_range[0]) & (spec_wave<=wavelength_range[1]))
		spec_wave = spec_wave[idx0[0]]
		spec_flux = spec_flux[idx0[0]]
		spec_flux_err = spec_flux_err[idx0[0]]

	with h5py.File(inputSED_file, 'w') as f:
		f.create_dataset('obs_flux', data=np.array(obs_flux), compression="gzip")
		f.create_dataset('obs_flux_err', data=np.array(obs_flux_err), compression="gzip")
		f.create_dataset('spec_wave', data=np.array(spec_wave), compression="gzip")
		f.create_dataset('spec_flux', data=np.array(spec_flux), compression="gzip")
		f.create_dataset('spec_flux_err', data=np.array(spec_flux_err), compression="gzip")

	os.system('mv %s %s' % (inputSED_file,temp_dir))
	return inputSED_file

def write_input_spec_hdf5(temp_dir,spec_wave,spec_flux,spec_flux_err,wavelength_range):
	inputSED_file = randname("inputSED_file",".hdf5")

	if wavelength_range is not None:
		idx0 = np.where((spec_wave>=wavelength_range[0]) & (spec_wave<=wavelength_range[1]))
		spec_wave = spec_wave[idx0[0]]
		spec_flux = spec_flux[idx0[0]]
		spec_flux_err = spec_flux_err[idx0[0]]

	with h5py.File(inputSED_file, 'w') as f:
		f.create_dataset('spec_wave', data=np.array(spec_wave), compression="gzip")
		f.create_dataset('spec_flux', data=np.array(spec_flux), compression="gzip")
		f.create_dataset('spec_flux_err', data=np.array(spec_flux_err), compression="gzip")

	os.system('mv %s %s' % (inputSED_file,temp_dir))
	return inputSED_file

def write_arbitprior(name,values,prob):
	file_out = open(name,"w")
	for ii in range(0,len(values)):
		file_out.write("%e %e\n" % (values[ii],prob[ii]))
	file_out.close()

def write_joint_prior(name,values1,values2):
	file_out = open(name,"w")
	for ii in range(0,len(values1)):
		file_out.write("%e %e\n" % (values1[ii],values2[ii]))
	file_out.close()

def get_nproc(nproc,fit_method,nwalkers,nsteps,nsteps_cut):
	if fit_method=='mcmc' or fit_method=='MCMC':
		nproc_new = nproc_reduced(nproc,nwalkers,nsteps,nsteps_cut)
	elif fit_method=='rdsps' or fit_method=='RDSPS':
		nproc_new = nproc
	return nproc_new

def define_free_z(gal_z):
	if gal_z is None or gal_z<=0.0:
		gal_z = -99.0
		free_z = 1
	else:
		free_z = 0

	return gal_z, free_z

def define_free_z_bins_fits(fits_binmap, free_z, gal_z):
	if free_z == 0:
		if gal_z is None or gal_z<=0.0:
			hdu = fits.open(fits_binmap)
			gal_z = float(hdu[0].header['z'])
			hdu.close()
	elif free_z == 1:
		gal_z = -99.0

	return gal_z


def get_name_out_fits(name_out_fits,fit_method):
	if name_out_fits is None:
		if fit_method=='mcmc' or fit_method=='MCMC':
			name_out_fits = randname("mcmc_fit",".fits")
		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_out_fits = randname("rdsps_fit",".fits")

	return name_out_fits

def remove_files(temp_dir,name_config=None,name_filters_list=None,inputSED_file=None,
	flg_write=None,name_samplers_hdf5=None,name_file_lsf=None):

	if name_config is not None:
		os.system("rm %s%s" % (temp_dir,name_config))
	if name_filters_list is not None:
		os.system("rm %s%s" % (temp_dir,name_filters_list))
	if inputSED_file is not None:
		os.system("rm %s%s" % (temp_dir,inputSED_file))
	if flg_write is not None and len(flg_write)>0:
		for ii in range(len(flg_write)):
			os.system("rm %s%s" % (temp_dir,flg_write[ii]))
	if name_samplers_hdf5 is not None:
		os.system("rm %s%s" % (temp_dir,name_samplers_hdf5))
	if name_file_lsf is not None:
		os.system("rm %s%s" % (temp_dir,name_file_lsf))

def write_conf_file(temp_dir,params_ranges=None,params_priors=None,nwalkers=None,nsteps=None,nsteps_cut=None,
	nproc=None,cosmo=None,H0=None,Om0=None,fit_method=None,likelihood=None,dof=None,gal_z=None,nrands_z=None,
	add_igm_absorption=None,igm_type=None,perc_chi2=None,initfit_nmodels_mcmc=None,smooth_velocity=True,
	sigma_smooth=0.0,spec_resolution=None,smooth_lsf=False,lsf_wave=None,lsf_sigma=None,poly_order=None,
	del_wave_nebem=None,spec_chi_sigma_clip=None):

	name_config = randname("config_file",".dat")
	file_out = open(name_config,"w")
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
	file_out.write("gal_z %lf\n" % gal_z)
	file_out.write("nrands_z %d\n" % nrands_z)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)
	file_out.write("igm_type %d\n" % igm_type)
	if fit_method=='rdsps' or fit_method=='RDSPS':
		file_out.write("perc_chi2 %lf\n" % perc_chi2)
	file_out.write("initfit_nmodels_mcmc %d\n" % initfit_nmodels_mcmc)

	if smooth_velocity == True or smooth_velocity == 1:
		file_out.write("smooth_velocity 1\n")
	elif smooth_velocity == False or smooth_velocity == 0:
		file_out.write("smooth_velocity 0\n")

	if smooth_velocity == True or smooth_velocity == 1:
		if sigma_smooth is None:
			if spec_resolution is None:
				print ('spec_resolution is required parameter if smooth_velocity=True and sigma_smooth=None!')
				sys.exit()
			else:
				from astropy import constants as const
				sigma_smooth = const.c.value/1e+3/spec_resolution

	file_out.write("sigma_smooth %lf\n" % sigma_smooth)

	if smooth_lsf == True or smooth_lsf == 1:
		file_out.write("smooth_lsf 1\n")
		# make file storing line spread function
		name_file_lsf = random_name("smooth_lsf",".dat")
		file_out1 = open(name_file_lsf,"w")
		for ii in range(len(lsf_wave)):
			file_out1.write("%e %e\n" % (lsf_wave[ii],lsf_sigma[ii]))
		file_out1.close()
		file_out.write("name_file_lsf %s\n" % name_file_lsf)
		os.system('mv %s %s' % (name_file_lsf,temp_dir))
	elif smooth_lsf == False or smooth_lsf == 0:
		file_out.write("smooth_lsf 0\n")
		name_file_lsf = None

	if poly_order is not None:
		file_out.write("poly_order %d\n" % poly_order)
	if del_wave_nebem is not None:
		file_out.write("del_wave_nebem %lf\n" % del_wave_nebem)
	if spec_chi_sigma_clip is not None:
		file_out.write("spec_chi_sigma_clip %lf\n" % spec_chi_sigma_clip)
	
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
	file_out.write("pr_gas_logu_min %lf\n" % params_ranges['gas_logu'][0])
	file_out.write("pr_gas_logu_max %lf\n" % params_ranges['gas_logu'][1])
	if params_ranges['gas_logz'] is not None:
		file_out.write("pr_gas_logz_min %lf\n" % params_ranges['gas_logz'][0])
		file_out.write("pr_gas_logz_max %lf\n" % params_ranges['gas_logz'][1])
		file_out.write("free_gas_logz 1\n")
	elif params_ranges['gas_logz'] is None:
		file_out.write("free_gas_logz 0\n")
	file_out.write("pr_nparams %d\n" % len(params_priors))

	flg_write = []
	for ii in range(0,len(params_priors)):
		priors = params_priors[ii]
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
		elif form == 'arbitrary':
			file_out.write("pr_form_%s_arbit_name %s\n" % (param,priors[2]))
			flg_write.append(priors[2])
		elif form == 'joint_with_mass':
			file_out.write("pr_form_%s_jtmass_name %s\n" % (param,priors[2]))
			file_out.write("pr_form_%s_jtmass_scale %s\n" % (param,priors[3]))
			# Whether the stellar mass is in surface density or not
			if priors[4]==True or priors[4]==1:
				file_out.write("pr_form_%s_jtmass_sd 1\n" % param)
			else:
				file_out.write("pr_form_%s_jtmass_sd 0\n" % param)
	file_out.close()
	os.system('mv %s %s' % (name_config,temp_dir))

	return flg_write, name_config, name_file_lsf


def read_config_file_fit(temp_dir,config_file,params0):
	from astropy.cosmology import FlatLambdaCDM, WMAP5, WMAP7, WMAP9, Planck13, Planck15

	def_params_val = default_params_val()

	data = np.genfromtxt(temp_dir+config_file, dtype=str)
	config_data = {}
	for ii in range(0,len(data[:,0])):
		str_temp = data[:,0][ii]
		config_data[str_temp] = data[:,1][ii]

	cosmo, H0, Om0, gal_z = int(config_data['cosmo']), float(config_data['H0']), float(config_data['Om0']), float(config_data['gal_z'])
	if gal_z<=0.0:
		free_z = 1
		DL_Gpc = 0.0
	elif gal_z>0.0:
		free_z = 0
		# luminosity distance
		if cosmo=='flat_LCDM' or cosmo==0:
			cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
			DL_Gpc0 = cosmo1.luminosity_distance(gal_z)      # in unit of Mpc
		elif cosmo=='WMAP5' or cosmo==1:
			DL_Gpc0 = WMAP5.luminosity_distance(gal_z)
		elif cosmo=='WMAP7' or cosmo==2:
			DL_Gpc0 = WMAP7.luminosity_distance(gal_z)
		elif cosmo=='WMAP9' or cosmo==3:
			DL_Gpc0 = WMAP9.luminosity_distance(gal_z)
		elif cosmo=='Planck13' or cosmo==4:
			DL_Gpc0 = Planck13.luminosity_distance(gal_z)
		elif cosmo=='Planck15' or cosmo==5:
			DL_Gpc0 = Planck15.luminosity_distance(gal_z)
		#elif cosmo=='Planck18' or cosmo==6:
		#	DL_Gpc0 = Planck18.luminosity_distance(gl_z)
		DL_Gpc = DL_Gpc0.value/1.0e+3

		def_params_val['z'] = gal_z

	# remove 'z' if free_z=0
	if free_z == 0:
		params0.remove('z')

	free_gas_logz = int(config_data['free_gas_logz'])
	if free_gas_logz == 0:
		params0.remove('gas_logz')

	if float(config_data['pr_gas_logu_min']) == float(config_data['pr_gas_logu_max']):
		params0.remove('gas_logu')

	nparams0 = len(params0)
	params, fix_params, priors_min0, priors_max0 = [], [], [], []
	for pp in range(0,nparams0-1): 						# without log_mass
		str_temp1 = 'pr_%s_min' % params0[pp]
		str_temp2 = 'pr_%s_max' % params0[pp]
		if float(config_data[str_temp1]) == float(config_data[str_temp2]):
			def_params_val[params0[pp]] = float(config_data[str_temp1])
			fix_params.append(params0[pp])
		elif float(config_data[str_temp1]) < float(config_data[str_temp2]):
			params.append(params0[pp])
			priors_min0.append(float(config_data[str_temp1]))
			priors_max0.append(float(config_data[str_temp2]))
		else:
			print ("invalid (inverted) range for %s!" % params0[pp])
	params.append('log_mass')
	nparams = len(params)

	priors_min = np.zeros(nparams)
	priors_max = np.zeros(nparams)
	priors_min[0:nparams-1] = np.asarray(priors_min0)
	priors_max[0:nparams-1] = np.asarray(priors_max0)

	nmodels = int(config_data['initfit_nmodels_mcmc'])

	# get preferred priors
	nparams_in_prior = int(config_data['pr_nparams'])
	params_in_prior = []
	for pp in range(0,nparams_in_prior):
		params_in_prior.append(config_data['pr_param%d' % pp])

	params_priors = {}
	for pp in range(0,nparams):
		params_priors[params[pp]] = {}
		if params[pp] in params_in_prior:
			params_priors[params[pp]]['form'] = config_data['pr_form_%s' % params[pp]]
			if params_priors[params[pp]]['form'] == 'gaussian':
				params_priors[params[pp]]['loc'] = float(config_data['pr_form_'+params[pp]+'_gauss_loc'])
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_'+params[pp]+'_gauss_scale'])
			elif params_priors[params[pp]]['form'] == 'studentt':
				params_priors[params[pp]]['df'] = float(config_data['pr_form_'+params[pp]+'_stdt_df'])
				params_priors[params[pp]]['loc'] = float(config_data['pr_form_'+params[pp]+'_stdt_loc'])
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_'+params[pp]+'_stdt_scale'])
			elif params_priors[params[pp]]['form'] == 'gamma':
				params_priors[params[pp]]['a'] = float(config_data['pr_form_'+params[pp]+'_gamma_a'])
				params_priors[params[pp]]['loc'] = float(config_data['pr_form_'+params[pp]+'_gamma_loc'])
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_'+params[pp]+'_gamma_scale'])
			elif params_priors[params[pp]]['form'] == 'arbitrary':
				data = np.loadtxt(temp_dir+config_data['pr_form_'+params[pp]+'_arbit_name'])
				params_priors[params[pp]]['values'] = data[:,0]
				params_priors[params[pp]]['prob'] = data[:,1]
			elif params_priors[params[pp]]['form'] == 'joint_with_mass':
				data = np.loadtxt(temp_dir+config_data['pr_form_'+params[pp]+'_jtmass_name'])
				params_priors[params[pp]]['lmass'] = data[:,0]
				params_priors[params[pp]]['pval'] = data[:,1]
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_'+params[pp]+'_jtmass_scale'])
				params_priors[params[pp]]['mass_sd'] = int(config_data['pr_form_'+params[pp]+'_jtmass_sd'])
		else:
			params_priors[params[pp]]['form'] = 'uniform'

	# igm absorption
	add_igm_absorption = int(config_data['add_igm_absorption'])
	igm_type = int(config_data['igm_type'])

	# number of walkers, steps, and nsteps_cut
	nwalkers = int(config_data['nwalkers'])
	nsteps = int(config_data['nsteps'])
	nsteps_cut = int(config_data['nsteps_cut'])

	if free_z == 1:
		nrands_z = int(config_data['nrands_z'])
		pr_z_min = float(config_data['pr_z_min'])
		pr_z_max = float(config_data['pr_z_max'])
	elif free_z == 0:
		nrands_z = 10
		pr_z_min = 0.0
		pr_z_max = 1.0

	# spectral smoothing parameters
	smooth_velocity = int(config_data['smooth_velocity'])
	sigma_smooth = float(config_data['sigma_smooth'])
	smooth_lsf = int(config_data['smooth_lsf'])
	if smooth_lsf == 1:
		name_file_lsf = config_data['name_file_lsf']
	else:
		name_file_lsf = 'none'

	# more parameters in the spectrophotometric SED fitting
	if 'poly_order' in config_data:
		poly_order = int(config_data['poly_order'])
	else:
		poly_order = 1

	if 'del_wave_nebem' in config_data:
		del_wave_nebem = float(config_data['del_wave_nebem'])
	else:
		del_wave_nebem = 15.0

	if 'spec_chi_sigma_clip' in config_data:
		spec_chi_sigma_clip = float(config_data['spec_chi_sigma_clip'])
	else:
		spec_chi_sigma_clip = 3.0

	return def_params_val, cosmo, H0, Om0, gal_z, free_z, DL_Gpc, params, fix_params, nparams, priors_min, priors_max, nmodels, params_priors, add_igm_absorption, igm_type, nwalkers, nsteps, nsteps_cut, nrands_z, pr_z_min, pr_z_max, free_gas_logz, smooth_velocity, sigma_smooth, smooth_lsf, name_file_lsf, poly_order, del_wave_nebem, spec_chi_sigma_clip 


def run_fitting(temp_dir,obs_flux,obs_flux_err,filters,spec_wave,spec_flux,spec_flux_err,wavelength_range,fit_method,free_z,
	nproc,nproc_new,CODE_dir,name_config,models_spec,store_full_samplers,name_out_fits,bin_area=1.0):

	name_out_fits = get_name_out_fits(name_out_fits,fit_method)

	if obs_flux is None:
		name_filters_list = None
		# spec only
		inputSED_file = write_input_spec_hdf5(temp_dir,spec_wave,spec_flux,spec_flux_err,wavelength_range)

		if fit_method=='mcmc' or fit_method=='MCMC':
			name_samplers_hdf5 = randname("samplers_",".hdf5")

			os.system("mpirun -n %d python %s./mc_p1_s.py %s %s %s %s %lf" % (nproc,CODE_dir,name_config,inputSED_file,name_samplers_hdf5,models_spec,bin_area))
			os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
			
			if store_full_samplers==1 or store_full_samplers==True:
				os.system("mpirun -n %d python %s./mc_p2_s.py %s %s" % (nproc_new,CODE_dir,name_samplers_hdf5,name_out_fits))
			elif store_full_samplers==0 or store_full_samplers==False:
				os.system("mpirun -n %d python %s./mc_nsmp_p2_s.py %s %s" % (nproc_new,CODE_dir,name_samplers_hdf5,name_out_fits))
			else:
				print ("Input store_full_samplers not recognized!")
				sys.exit()

		elif fit_method=='rdsps' or fit_method=='RDSPS':
			name_samplers_hdf5 = None
			
			if store_full_samplers==1 or store_full_samplers==True:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz_s.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_s.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
			elif store_full_samplers==0 or store_full_samplers==False:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz_nsmp_s.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_nsmp_s.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
			else:
				print ("Input store_full_samplers not recognized!")
				sys.exit()

		else:
			print ("The input fit_method is not recognized!")
			sys.exit()

	else:
		name_filters_list = write_filters_list(temp_dir,filters)
		if spec_wave is None:
			# photo only
			inputSED_file = write_input_singleSED_photo(temp_dir,obs_flux,obs_flux_err)

			if fit_method=='mcmc' or fit_method=='MCMC':
				name_samplers_hdf5 = randname("samplers_",".hdf5")

				os.system("mpirun -n %d python %s./mc_p1.py %s %s %s %s %s %lf" % (nproc,CODE_dir,name_filters_list,name_config,inputSED_file,name_samplers_hdf5,models_spec,bin_area))
				os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
				
				if store_full_samplers==1 or store_full_samplers==True:
					os.system("mpirun -n %d python %s./mc_p2.py %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_samplers_hdf5,name_out_fits))
				elif store_full_samplers==0 or store_full_samplers==False:
					os.system("mpirun -n %d python %s./mc_nsmp_p2.py %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_samplers_hdf5,name_out_fits))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

			elif fit_method=='rdsps' or fit_method=='RDSPS':
				name_samplers_hdf5 = None
				
				if store_full_samplers==1 or store_full_samplers==True:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
				elif store_full_samplers==0 or store_full_samplers==False:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_nsmp.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_nsmp.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

			else:
				print ("The input fit_method is not recognized!")
				sys.exit()

		else:
			# spec+photo
			inputSED_file = write_input_specphoto_hdf5(temp_dir,obs_flux,obs_flux_err,spec_wave,spec_flux,spec_flux_err,wavelength_range)

			if fit_method=='mcmc' or fit_method=='MCMC':
				name_samplers_hdf5 = randname("samplers_",".hdf5")

				os.system("mpirun -n %d python %s./mc_p1_sp.py %s %s %s %s %s %lf" % (nproc,CODE_dir,name_filters_list,name_config,inputSED_file,name_samplers_hdf5,models_spec,bin_area))
				os.system('mv %s %s' % (name_samplers_hdf5,temp_dir))
				
				if store_full_samplers==1 or store_full_samplers==True:
					os.system("mpirun -n %d python %s./mc_p2_sp.py %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_samplers_hdf5,name_out_fits))
				elif store_full_samplers==0 or store_full_samplers==False:
					os.system("mpirun -n %d python %s./mc_nsmp_p2_sp.py %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_samplers_hdf5,name_out_fits))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

			elif fit_method=='rdsps' or fit_method=='RDSPS':
				name_samplers_hdf5 = None
				
				if store_full_samplers==1 or store_full_samplers==True:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_sp.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_sp.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
				elif store_full_samplers==0 or store_full_samplers==False:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_nsmp_sp.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_nsmp_sp.py %s %s %s %s %s %lf" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec,bin_area))
				else:
					print ("Input store_full_samplers not recognized!")
					sys.exit()

			else:
				print ("The input fit_method is not recognized!")
				sys.exit()

	return name_filters_list, inputSED_file, name_samplers_hdf5, name_out_fits


def make_bins_name_out_fits(name_out_fits,bin_ids,fit_method):

	if name_out_fits is None:
		name_out_fits1 = []

		if fit_method=='mcmc' or fit_method=='MCMC':
			for bin_id in bin_ids:
				name_out_fits1.append("mcmc_bin%d.fits" % (bin_id+1))

		elif fit_method=='rdsps' or fit_method=='RDSPS':
			for bin_id in bin_ids:
				name_out_fits1.append("rdsps_bin%d.fits" % (bin_id+1))

	else:
		if 0<len(name_out_fits)<len(bin_ids):
			print ("The number of elements in name_out_fits should be the same as the number of bins to be calculated!")
			sys.exit()
		else:
			name_out_fits1 = name_out_fits

	return name_out_fits1
		

def save_fitting_results_db(sedfit,priors,name_out_fits,mocksp,filters):

	from ..piXedfit_model import calc_mw_age
	from dense_basis.pre_grid import makespec_atlas
	from ..piXedfit_images import convert_flux_unit
	from scipy.interpolate import interp1d
	from piXedfit.utils.filtering import filtering, cwave_filters

	from astropy.cosmology import FlatLambdaCDM
	cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

	sfh_500, sfh_160, sfh_840, common_time0 = sedfit.evaluate_posterior_SFH(sedfit.z[1])
	idx_sel = np.where((np.isnan(common_time0)==False) & (np.isinf(common_time0)==False) & (np.isnan(sfh_500)==False) & (np.isinf(sfh_500)==False))
	sfh_50, sfh_16, sfh_84, common_time = sfh_500[idx_sel[0]], sfh_160[idx_sel[0]], sfh_840[idx_sel[0]], common_time0[idx_sel[0]]

	common_time_lbt = np.flip(common_time,0)
	
	mw_age = np.zeros(3)
	if len(common_time)>0:
		mw_age[0] = calc_mw_age(sfh_form='arbitrary_sfh',age=max(common_time),sfh_t=common_time,sfh_sfr=sfh_16)
		mw_age[1] = calc_mw_age(sfh_form='arbitrary_sfh',age=max(common_time),sfh_t=common_time,sfh_sfr=sfh_50)
		mw_age[2] = calc_mw_age(sfh_form='arbitrary_sfh',age=max(common_time),sfh_t=common_time,sfh_sfr=sfh_84)
		mw_age_sort = np.sort(mw_age)
	else:
		mw_age_sort = mw_age

	dt = 100/(priors.Nparam+1)
	t_names = []
	for ii in range(priors.Nparam):
		t_names.append('t%d' % round((ii+1)*dt))

	## retrive best-fit model SEDs
	nmods = 100

	mod_spec_flam = []
	mod_photo_flam = []

	mod_spec_flam_fz = []
	mod_photo_flam_fz = []

	mod_chi2 = []
	
	sort_idx_likelihood = np.argsort(sedfit.likelihood)

	lam, spec_fnu =  makespec_atlas(sedfit.atlas, sort_idx_likelihood[-1], priors, mocksp, cosmo, filter_list=[], filt_dir=[], return_spec=True)
	ref_spec_wave = lam*(1.0 + sedfit.atlas['zval'][sort_idx_likelihood[-1]])
	bfit_spec_flam = convert_flux_unit(ref_spec_wave, spec_fnu*sedfit.norm_fac, init_unit='uJy', final_unit='erg/s/cm2/A')
	mod_spec_flam.append(bfit_spec_flam)

	bfit_photo_flam = filtering(ref_spec_wave, bfit_spec_flam, filters)
	mod_photo_flam.append(bfit_photo_flam)

	for ii in range(nmods):
		lam, spec_fnu =  makespec_atlas(sedfit.atlas, sort_idx_likelihood[-(ii+1)], priors, mocksp, cosmo, filter_list=[], filt_dir=[], return_spec=True)
		
		redshifted_lam = lam*(1.0 + sedfit.atlas['zval'][sort_idx_likelihood[-(ii+1)]])
		spec_flam = convert_flux_unit(redshifted_lam, spec_fnu*sedfit.norm_fac, init_unit='uJy', final_unit='erg/s/cm2/A')
		f = interp1d(redshifted_lam, spec_flam, fill_value="extrapolate")
		mod_spec_flam.append(f(ref_spec_wave))
		mod_photo_flam.append(filtering(redshifted_lam, spec_flam, filters))

		mod_chi2.append(sedfit.chi2_array[sort_idx_likelihood[-(ii+1)]])

	mod_spec_flam = np.asarray(mod_spec_flam)
	mod_photo_flam = np.asarray(mod_photo_flam)

	p16_mod_spec_flam = np.percentile(mod_spec_flam, 16, axis=0)
	p50_mod_spec_flam = np.percentile(mod_spec_flam, 50, axis=0)
	p84_mod_spec_flam = np.percentile(mod_spec_flam, 84, axis=0)

	p16_mod_photo_flam = np.percentile(mod_photo_flam, 16, axis=0)
	p50_mod_photo_flam = np.percentile(mod_photo_flam, 50, axis=0)
	p84_mod_photo_flam = np.percentile(mod_photo_flam, 84, axis=0)

	photo_cwave = cwave_filters(filters)

	obs_sed_flam = convert_flux_unit(photo_cwave, sedfit.sed, init_unit='uJy', final_unit='erg/s/cm2/A')
	obs_sed_err_flam = convert_flux_unit(photo_cwave, sedfit.sed_err, init_unit='uJy', final_unit='erg/s/cm2/A')
	
	hdr = fits.Header()
	hdr['Nparam'] = priors.Nparam
	hdr['mass_min'] = priors.mass_min
	hdr['mass_max'] = priors.mass_max
	hdr['sfr_prior_type'] = priors.sfr_prior_type
	hdr['met_treatment'] = priors.met_treatment
	hdr['met_min'] = priors.Z_min 
	hdr['met_max'] = priors.Z_max 
	hdr['dust_prior'] = priors.dust_prior
	hdr['Av_min'] = priors.Av_min
	hdr['Av_max'] = priors.Av_max
	hdr['z_min'] = priors.z_min 
	hdr['z_max'] = priors.z_max 
	hdr['min_chi2'] = mod_chi2[0]
	for ii in range(priors.Nparam):
		hdr['sfh_t%d' % ii] = t_names[ii] 
	primary_hdu = fits.PrimaryHDU(header=hdr)

	## fitting parameters
	cols0 = []
	col = fits.Column(name='rows', format='3A', array=['p16','p50','p84'])
	cols0.append(col)
	col = fits.Column(name='log_mass', format='D', array=np.array(sedfit.mstar))
	cols0.append(col)
	col = fits.Column(name='log_sfr', format='D', array=np.array(sedfit.sfr))
	cols0.append(col)
	col = fits.Column(name='Av', format='D', array=np.array(sedfit.Av))
	cols0.append(col)
	col = fits.Column(name='logzsol', format='D', array=np.array(sedfit.Z))
	cols0.append(col)
	col = fits.Column(name='z', format='D', array=np.array(sedfit.z))
	cols0.append(col)
	col = fits.Column(name='log_mw_age', format='D', array=np.array(np.log10(mw_age_sort)))
	cols0.append(col)

	sfh_tuple = np.asarray(sedfit.sfh_tuple)
	sfh_tuple_t = np.transpose(sfh_tuple, axes=(1,0))

	col = fits.Column(name='log_mass_form', format='D', array=np.array(sfh_tuple_t[0]))
	cols0.append(col)

	for ii in range(priors.Nparam):
		col = fits.Column(name=t_names[ii], format='D', array=np.array(sfh_tuple_t[ii+3]))
		cols0.append(col)

	cols = fits.ColDefs(cols0)
	hdu1 = fits.BinTableHDU.from_columns(cols, name='fit_params')

	## SFH
	cols0 = []
	col = fits.Column(name='lbt', format='D', array=np.array(common_time_lbt))
	cols0.append(col)
	col = fits.Column(name='sfh_16', format='D', array=np.array(sfh_16))
	cols0.append(col)
	col = fits.Column(name='sfh_50', format='D', array=np.array(sfh_50))
	cols0.append(col)
	col = fits.Column(name='sfh_84', format='D', array=np.array(sfh_84))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu2 = fits.BinTableHDU.from_columns(cols, name='sfh')

	## median of posterior model spectrum: with varying z
	cols0 = []
	col = fits.Column(name='spec_wave', format='D', array=np.array(ref_spec_wave))
	cols0.append(col)
	col = fits.Column(name='spec_p16', format='D', array=np.array(p16_mod_spec_flam))
	cols0.append(col)
	col = fits.Column(name='spec_p50', format='D', array=np.array(p50_mod_spec_flam))
	cols0.append(col)
	col = fits.Column(name='spec_p84', format='D', array=np.array(p84_mod_spec_flam))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu3 = fits.BinTableHDU.from_columns(cols, name='spec')

	## best-fit model spectrum
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(ref_spec_wave))
	cols0.append(col)
	col = fits.Column(name='spec', format='D', array=np.array(bfit_spec_flam))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu4 = fits.BinTableHDU.from_columns(cols, name='bfit_spec')

	## observed and best-fit photometric SEDs
	cols0 = []
	col = fits.Column(name='photo_wave', format='D', array=np.array(photo_cwave))
	cols0.append(col)
	col = fits.Column(name='sed', format='D', array=np.array(obs_sed_flam))
	cols0.append(col)
	col = fits.Column(name='sed_err', format='D', array=np.array(obs_sed_err_flam))
	cols0.append(col)	
	col = fits.Column(name='photo_p16', format='D', array=np.array(p16_mod_photo_flam))
	cols0.append(col)
	col = fits.Column(name='photo_p50', format='D', array=np.array(p50_mod_photo_flam))
	cols0.append(col)
	col = fits.Column(name='photo_p84', format='D', array=np.array(p84_mod_photo_flam))
	cols0.append(col)
	col = fits.Column(name='bfit_photo', format='D', array=np.array(bfit_photo_flam))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu5 = fits.BinTableHDU.from_columns(cols, name='photo')


	hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4, hdu5])
	hdul.writeto(name_out_fits, overwrite=True)	


def compute_chisq(i_model, model_flux, obs_flux, obs_flux_err):
    w = 1.0 / (obs_flux_err**2)
    numerator = np.sum(w * obs_flux * model_flux)
    denominator = np.sum(w * model_flux**2)

    if denominator == 0:
        norm = 0.0
        chi_sq = np.inf
    else:
        norm = numerator / denominator
        chi_sq = np.sum(((obs_flux - norm * model_flux) / obs_flux_err)**2)

    return i_model, chi_sq, norm

def fit_sed_chisq(obs_flux, obs_flux_err, model_seds, n_cpu=None):
    if n_cpu is None:
        from joblib import cpu_count
        n_cpu = cpu_count()

    results = Parallel(n_jobs=n_cpu)(
        delayed(compute_chisq)(i, model_seds[i], obs_flux, obs_flux_err)
        for i in range(model_seds.shape[0])
    )

    return min(results, key=lambda x: x[1])  # (idx, chi², norm)

def fit_sed_batch_old(obs_fluxes, obs_flux_errs, model_seds, n_cpu=None):
    return [
        fit_sed_chisq(obs_flux, obs_flux_err, model_seds, n_cpu=n_cpu)
        for obs_flux, obs_flux_err in zip(obs_fluxes, obs_flux_errs)
    ]

def fit_sed_batch(obs_fluxes, obs_flux_errs, model_seds, n_cpu=None):
    """
    Batch fitting multiple observed SEDs with tqdm progress bar.
    """
    results = []
    for i in tqdm(range(len(obs_fluxes)), desc="Batch fitting"):
        result = fit_sed_chisq(obs_fluxes[i], obs_flux_errs[i], model_seds, n_cpu=n_cpu)
        results.append(result)
    return results


def calculate_likelihood(obs_flux, obs_flux_err, model_seds, chi_sq_list, likelihood_type='gauss', dof=2.0):
    """
    Calculates the likelihood for each model SED based on the chi-square.
    
    :param obs_flux: 1D array of observed fluxes.
    :param obs_flux_err: 1D array of observed flux uncertainties.
    :param model_seds: 2D array of model fluxes (N_models x N_bands).
    :param chi_sq_list: 1D array of chi-square values for each model.
    :param likelihood_type: 'gauss' for Gaussian, 'student_t' for Student's t.
    :param dof: Degree of freedom for Student's t distribution.
    
    :returns likelihood_list: 1D array of likelihoods.
    :returns chi_sq_list: 1D array of chi-square values.
    """
    
    likelihood_list = np.zeros(len(chi_sq_list))
    
    if likelihood_type.lower() == 'gauss':
        # Likelihood L is proportional to exp(-chi^2 / 2) for Gaussian noise
        likelihood_list = np.exp(-0.5 * np.asarray(chi_sq_list))
    
    elif likelihood_type.lower() == 'student_t':
        # Likelihood for Student's t distribution (see pixedfit docs or reference)
        
        # We need the number of data points
        N_data = len(obs_flux)
        nu = dof
        
        # T_nu = (Gamma((nu + 1)/2) / (Gamma(nu/2) * sqrt(nu*pi))) * (1 + (x^2)/nu)^(-(nu+1)/2)
        # For simplicity in relative likelihoods, we can use the part dependent on chi^2
        # Likelihood L is proportional to (1 + chi^2 / nu)^(-(nu+1)/2)
        
        # Calculate log-likelihood
        log_term = (1.0 + np.asarray(chi_sq_list) / nu)
        log_likelihood_list = -0.5 * (nu + N_data) * np.log(log_term)
        
        # Scale to avoid underflow: normalize by max log-likelihood
        max_log_like = np.max(log_likelihood_list[np.isfinite(log_likelihood_list)])
        likelihood_list = np.exp(log_likelihood_list - max_log_like)

    else:
        raise ValueError("Unknown likelihood type: choose 'gauss' or 'student_t'")

    # Set NaN/Inf likelihoods to zero for robust parameter weighting
    likelihood_list[~np.isfinite(likelihood_list)] = 0.0
    
    # Normalize likelihoods so their sum is 1 (effectively a normalized posterior
    # since we assume a uniform prior on the models, which is true for RDSPS/initial MCMC steps)
    if np.sum(likelihood_list) > 0:
        likelihood_list = likelihood_list / np.sum(likelihood_list)

    return likelihood_list, chi_sq_list


def calculate_priors(params_val_list, param_priors, normalized_params_val_list):
    """
    Calculates the prior probability (or un-normalized prior PDF value) for each model.
    
    :param params_val_list: Original list of model parameter dictionaries.
    :param param_priors: Dictionary defining the priors.
    :param normalized_params_val_list: List of model parameter dictionaries AFTER log_mass normalization.
    
    :returns prior_list: 1D array of multiplicative prior probability factor for each model.
    """
    N_models = len(params_val_list)
    prior_list = np.ones(N_models)
    
    # We must iterate over models and apply each prior sequentially as a multiplicative factor
    for i in range(N_models):
        current_prior = 1.0
        
        # Determine the current (normalized) log_mass
        log_mass = normalized_params_val_list[i].get('log_mass')
        
        # Check if log_mass is a valid number (i.e., this model survived the normalization check)
        if np.isnan(log_mass) or log_mass is None:
             # If log_mass is invalid, this model is essentially rejected (prior = 0)
             prior_list[i] = 0.0
             continue

        for param, prior_def in param_priors.items():
            param_val = params_val_list[i].get(param)
            
            # Skip if parameter is not in the model or has an invalid value
            if param_val is None or np.isnan(param_val):
                continue
                
            form = prior_def['form']
            
            if form == 'gaussian':
                loc = prior_def['loc']
                scale = prior_def['scale']
                
                # Calculate PDF value for a Gaussian distribution: 
                # P(x) = 1 / (sigma * sqrt(2*pi)) * exp(-0.5 * ((x-mu)/sigma)^2)
                # Since we care about relative probabilities, we can drop the constant factor 1 / (sigma * sqrt(2*pi))
                term = (param_val - loc) / scale
                prior_prob = np.exp(-0.5 * term**2)
                
            elif form == 'joint_with_mass':
                # Allowed only for 'log_mw_age' and 'logzsol'
                if param not in ['log_mw_age', 'logzsol']:
                    continue
                    
                corr_log_mass = prior_def['corr_log_mass']
                corr_pval = prior_def['corr_pval']
                scale = prior_def['scale'] # Scatter (sigma)
                
                # Interpolate the median relation: find the expected 'loc' for the current log_mass
                from scipy.interpolate import interp1d
                median_relation_func = interp1d(corr_log_mass, corr_pval, kind='linear', fill_value='extrapolate')
                loc_from_mass = median_relation_func(log_mass)

                # The prior is a Gaussian centered at this interpolated 'loc' with the input 'scale'
                term = (param_val - loc_from_mass) / scale
                prior_prob = np.exp(-0.5 * term**2)
                
            else:
                # 'uniform' or any other unsupported explicit form -> effectively a constant (1.0)
                prior_prob = 1.0
                
            # Multiply the prior probabilities
            current_prior *= prior_prob
        
        prior_list[i] = current_prior
        
    # Scale final list to prevent numerical underflow if values are very small
    # Note: We apply this normalization to the prior values, not the likelihood
    max_prior = np.max(prior_list[prior_list > 0])
    if max_prior > 0.0:
        prior_list[prior_list > 0] /= max_prior

    return prior_list


def bayesian_inference_parameters(normalized_params_val_list, total_posterior_list, perc_chi2=90.0):
    """
    Calculates the inferred (best-fit) parameters using posterior-weighted statistics.
    
    :param normalized_params_val_list: List of dictionaries, each containing parameters for a model,
                                       with log_mass, log_sfr_inst, and log_sfr_100 already scaled by norm.
    :param total_posterior_list: 1D array of normalized posterior probability (Likelihood * Prior)
                                 for each model.
    :param perc_chi2: Percentile of models based on likelihood mass to consider.
    
    :returns inferred_params: Dictionary of inferred parameters (median, p16, p84).
    """
    
    # 1. Prepare parameter arrays
    if len(normalized_params_val_list) == 0:
        return {}
        
    param_names = list(normalized_params_val_list[0].keys())
    
    model_params_all = {}
    for param_name in param_names:
        # Convert list to array, replacing 'None' with 'np.nan'
        raw_values = [p.get(param_name) for p in normalized_params_val_list]
        clean_values = np.array([v if v is not None else np.nan for v in raw_values])
        model_params_all[param_name] = clean_values

    # 2. Filter models based on posterior mass (Bayesian method)
    # Sort models by posterior (descending)
    posterior_arr = np.asarray(total_posterior_list)
    sort_idx = np.argsort(posterior_arr)[::-1]
    sorted_posteriors = posterior_arr[sort_idx]
    
    # Cumulative sum of sorted posteriors
    cumulative_posterior = np.cumsum(sorted_posteriors)
    
    # Determine which models to keep based on the percentile cut.
    target_cumulative_mass = perc_chi2 / 100.0
    
    if cumulative_posterior[-1] == 0:
        models_to_keep_indices = []
    else:
        normalized_cumulative = cumulative_posterior / cumulative_posterior[-1]
        last_index_to_keep = np.searchsorted(normalized_cumulative, target_cumulative_mass)
        models_to_keep_indices = sort_idx[:last_index_to_keep + 1]

    # Fallback/Edge Case Handling
    if len(models_to_keep_indices) == 0:
        if len(sort_idx) > 0 and sorted_posteriors[0] > 0:
             models_to_keep_indices = [sort_idx[0]]
        else:
             inferred_params = {}
             for param_name in param_names:
                  inferred_params[param_name] = [np.nan, np.nan, np.nan] # p16, p50, p84
             return inferred_params

    # Filter the parameters and posteriors
    filtered_params = {p: model_params_all[p][models_to_keep_indices] for p in param_names}
    filtered_posteriors = posterior_arr[models_to_keep_indices]
    
    # Re-normalize posteriors of the kept models
    total_posterior_mass = np.sum(filtered_posteriors)
    if total_posterior_mass == 0:
        weights = np.ones(len(filtered_posteriors)) / len(filtered_posteriors)
    else:
        weights = filtered_posteriors / total_posterior_mass

    # 3. Calculate posterior-weighted percentiles (P16, P50, P84)
    inferred_params = {}
    
    for param_name in param_names:
        param_values = filtered_params[param_name]
        
        # Remove NaN values from the slice and adjust weights accordingly
        valid_mask = ~np.isnan(param_values)
        if np.sum(valid_mask) == 0:
            inferred_params[param_name] = [np.nan, np.nan, np.nan]
            continue
            
        param_values = param_values[valid_mask]
        current_weights = weights[valid_mask]
        
        # Re-normalize weights if some points were removed
        if np.sum(current_weights) > 0:
             current_weights /= np.sum(current_weights)
        else:
             inferred_params[param_name] = [np.nan, np.nan, np.nan]
             continue

        # Sort values and align weights
        sort_idx = np.argsort(param_values)
        sorted_values = param_values[sort_idx]
        sorted_weights = current_weights[sort_idx]
        
        # Calculate cumulative distribution function (CDF)
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find the values corresponding to P16, P50, and P84
        p16 = np.interp(0.16, cumulative_weights, sorted_values)
        p50 = np.interp(0.50, cumulative_weights, sorted_values)
        p84 = np.interp(0.84, cumulative_weights, sorted_values)
        
        inferred_params[param_name] = [p16, p50, p84]

    return inferred_params


def calculate_log_mw_age_and_sfr_for_models(params_val_list, sfh_form=4):
    """
    Calculates log10(mass-weighted age) for a list of model parameters.
    This function parallels the logic inside piXedfit_model.
    """
    log_mw_ages = []
    log_sfrs_inst = []
    log_sfrs_100 = []
    
    # Use single-threaded loop for simplicity, parallelization can be added if performance is critical
    for params_val in params_val_list:
        try:
            # 1. Convert log-params to linear scale (as expected by SFH functions)
            age = np.power(10.0, params_val['log_age'])
            tau = np.power(10.0, params_val.get('log_tau', 0.0))
            t0 = np.power(10.0, params_val.get('log_t0', 0.0))
            alpha = np.power(10.0, params_val.get('log_alpha', 0.0))
            beta = np.power(10.0, params_val.get('log_beta', 0.0))
            formed_mass = np.power(10.0, params_val.get('log_mass', 0.0))
            
            mw_age = calc_mw_age(sfh_form=sfh_form, age=age, tau=tau, t0=t0, alpha=alpha, beta=beta, formed_mass=formed_mass)
            sfh_t, sfh_sfr = construct_SFH(sfh_form=sfh_form, t0=t0, tau=tau, alpha=alpha, beta=beta, age=age, formed_mass=formed_mass, del_t=0.001)
            
            log_mw_ages.append(np.log10(mw_age))
            log_sfrs_inst.append(np.log10(sfh_sfr[-1]))
            log_sfrs_100.append(np.log10(np.mean(sfh_sfr[-1-100:-1])))
            
        except Exception as e:
            # If calculation fails (e.g., math domain error, zero mass), use NaN
            log_mw_ages.append(np.nan)
            log_sfrs_inst.append(np.nan)
            log_sfrs_100.append(np.nan)
            
    return log_mw_ages, log_sfrs_inst, log_sfrs_100



def bayesian_fit_sed_batch(obs_fluxes, obs_flux_errs, model_seds, params_val_list, param_priors=None, 
						likelihood_type='gauss', dof=2.0, perc_chi2=90.0, n_cpu=None, sfh_form=4):
    """
    Batch fitting multiple observed SEDs using the Bayesian likelihood-weighted method.
    
    Optimized: Priors independent of normalization (Gaussian/Uniform) are pre-calculated 
    *for each SED* if the priors change. Joint priors are calculated in-batch using 
    the best-fit normalized mass as the anchor.
    
    :param param_priors: Can be a single dictionary (same priors for all SEDs) or 
                         a list of dictionaries (one prior set per SED).
    """
    
    if n_cpu is None:
        from joblib import cpu_count
        n_cpu = cpu_count()
        
    if param_priors is None:
        param_priors = {}
    
    N_models = len(params_val_list)
    N_obs = len(obs_fluxes)
    
    # --- HANDLE PARAM_PRIORS INPUT TYPE ---
    if isinstance(param_priors, dict):
        # Convert a single dict to a list of N_obs identical dicts
        priors_list = [param_priors] * N_obs
    elif isinstance(param_priors, list):
        if len(param_priors) != N_obs:
            raise ValueError(f"When param_priors is a list, its length ({len(param_priors)}) must match the number of SEDs ({N_obs}).")
        priors_list = param_priors
    else:
        raise TypeError("param_priors must be a single dictionary or a list of dictionaries.")


    # 1. --- PRE-CALCULATE log_mw_age AND SFRs FOR ALL MODELS (UN-NORMALIZED) ---
    print("Pre-calculating model-dependent parameters (log_mw_age, log_sfr) once...")
    log_mw_ages, log_sfrs_inst, log_sfrs_100 = calculate_log_mw_age_and_sfr_for_models(params_val_list, sfh_form)
    
    # --- AUGMENT params_val_list INTO A RAW DICTIONARY OF ARRAYS ---
    model_params_raw = {}
    if N_models > 0:
        for k in params_val_list[0].keys():
            model_params_raw[k] = np.array([p.get(k) for p in params_val_list])
    
        # Add calculated parameters
        model_params_raw['log_mw_age'] = np.array(log_mw_ages)
        model_params_raw['log_sfr_inst'] = np.array(log_sfrs_inst) 
        model_params_raw['log_sfr_100'] = np.array(log_sfrs_100)
    else:
        # Handle case with no models
        return []

    # 2. --- START BATCH FITTING LOOP (one loop per observed SED) ---
    print("Starting batch SED fitting with SED-specific priors...")
    all_fit_results = []
    
    iterator = tqdm(zip(obs_fluxes, obs_flux_errs, priors_list), total=N_obs, desc="Batch SED Fitting (Bayesian)")
    for obs_flux, obs_flux_err, current_param_priors in iterator:
        
        # 2a. Calculate Chi-square and Normalization for all models/SED
        def compute_all_chisq_for_one_sed(obs_f, obs_ferr, model_seds, n_jobs):
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_chisq)(i, model_seds[i], obs_f, obs_ferr)
                for i in range(model_seds.shape[0])
            )
            return zip(*results) 

        _, chi_sq_list, norm_list = compute_all_chisq_for_one_sed(obs_flux, obs_flux_err, model_seds, n_cpu)
        norm_arr = np.array(norm_list)
        chi_sq_arr = np.array(chi_sq_list)
        
        # Find the *best-fit* normalization (anchor)
        min_chi_sq_idx = np.argmin(chi_sq_arr)
        
        # 2b. Calculate Likelihood for all models
        likelihood_list, _ = calculate_likelihood(
            obs_flux, obs_flux_err, model_seds, chi_sq_arr, likelihood_type, dof
        )
        
        # 2c. Calculate Normalized log_mass (for Anchor & Final Output)
        log_mass_raw_arr = model_params_raw['log_mass']
        log_norm_arr = np.log10(norm_arr, out=np.full_like(norm_arr, np.nan), where=norm_arr > 0.0)
        
        normalized_log_mass_arr = log_mass_raw_arr + log_norm_arr
        
        # Find the best-fit normalized log_mass (The Anchor Value)
        best_fit_normalized_log_mass = normalized_log_mass_arr[min_chi_sq_idx]
        
        
        # 2d. --- BATCH CALCULATE SED-SPECIFIC PRIORS ---
        prior_list_independent = np.ones(N_models)
        joint_prior_defs = {}

        for param, prior_def in current_param_priors.items():
            form = prior_def['form']
            if form == 'uniform':
                continue
                
            elif form == 'gaussian':
                loc = prior_def['loc']
                scale = prior_def['scale']
                param_values = model_params_raw.get(param)
                
                if param_values is not None:
                    # Vectorized Gaussian PDF (un-normalized P(x) proportional to exp(-0.5 * z^2))
                    term = (param_values - loc) / scale
                    prior_prob = np.exp(-0.5 * term**2)
                    
                    prior_list_independent *= prior_prob

            elif form == 'joint_with_mass':
                joint_prior_defs[param] = prior_def
        
        # Normalize the independent prior component to avoid underflow
        max_prior_indep = np.max(prior_list_independent[np.isfinite(prior_list_independent)])
        if max_prior_indep > 0.0:
            prior_list_independent[np.isfinite(prior_list_independent)] /= max_prior_indep
        else:
            prior_list_independent[:] = 1.0


        # 2e. Anchor-Based Joint Prior Calculation
        prior_list_joint = np.ones(N_models)
        
        if len(joint_prior_defs) > 0:
            if np.isnan(best_fit_normalized_log_mass):
                prior_list_joint[:] = 0.0
            else:
                from scipy.interpolate import interp1d
                for param, prior_def in joint_prior_defs.items():
                    if param not in ['log_mw_age', 'logzsol']:
                         continue

                    corr_log_mass = prior_def['corr_log_mass']
                    corr_pval = prior_def['corr_pval']
                    scale = prior_def['scale']
                    
                    # Interpolate the median relation: find the expected 'loc' for the anchor mass
                    median_relation_func = interp1d(corr_log_mass, corr_pval, kind='linear', fill_value='extrapolate')
                    loc_from_mass = median_relation_func(best_fit_normalized_log_mass)

                    param_values = model_params_raw.get(param)
                    
                    if param_values is not None:
                        # Vectorized Gaussian Prior centered at the *anchor location*
                        term = (param_values - loc_from_mass) / scale
                        prior_prob = np.exp(-0.5 * term**2)
                        
                        prior_list_joint *= prior_prob

        # 2f. Total Prior: independent * joint
        prior_list_total = prior_list_independent * prior_list_joint
        # Normalize the total prior component to avoid underflow
        max_prior_total = np.max(prior_list_total[np.isfinite(prior_list_total)])
        if max_prior_total > 0.0:
            prior_list_total[np.isfinite(prior_list_total)] /= max_prior_total
        else:
            prior_list_total[:] = 0.0

        # 2g. Calculate Total Posterior
        total_posterior_list = likelihood_list * prior_list_total
        
        # Normalize the posterior to sum to 1
        total_posterior_sum = np.sum(total_posterior_list[np.isfinite(total_posterior_list)])
        if total_posterior_sum > 0:
            total_posterior_list = total_posterior_list / total_posterior_sum
        else:
             total_posterior_list = np.zeros_like(total_posterior_list)
             
        # 2h. Prepare Normalized Parameter List (for Bayesian inference)
        normalized_params_val_list = []
        for i in range(N_models):
            normalized_pv = {}
            for k, v_arr in model_params_raw.items():
                normalized_pv[k] = v_arr[i]
                
            # Overwrite mass/SFR with normalized values
            normalized_pv['log_mass'] = normalized_log_mass_arr[i]
            
            # log_sfr_inst normalization
            if 'log_sfr_inst' in normalized_pv and not np.isnan(log_norm_arr[i]):
                 normalized_pv['log_sfr_inst'] += log_norm_arr[i]
            else:
                 normalized_pv['log_sfr_inst'] = np.nan
                 
            # log_sfr_100 normalization
            if 'log_sfr_100' in normalized_pv and not np.isnan(log_norm_arr[i]):
                 normalized_pv['log_sfr_100'] += log_norm_arr[i]
            else:
                 normalized_pv['log_sfr_100'] = np.nan
            
            if norm_arr[i] <= 0.0:
                 normalized_pv['log_mass'] = np.nan
                 normalized_pv['log_sfr_inst'] = np.nan
                 normalized_pv['log_sfr_100'] = np.nan

            normalized_params_val_list.append(normalized_pv)

        # 2i. Perform Bayesian parameter inference
        inferred_params = bayesian_inference_parameters(
            normalized_params_val_list, total_posterior_list, perc_chi2
        )
        
        all_fit_results.append({
            'inferred_params': inferred_params,
            'min_chi_sq': chi_sq_arr[min_chi_sq_idx],
            'best_fit_model_idx': min_chi_sq_idx,
            'model_likelihoods': likelihood_list,
            'model_priors': prior_list_total,
            'model_posteriors': total_posterior_list,
            'model_chi_sq': chi_sq_arr,
            'model_norms': norm_arr,
        })

    return all_fit_results



def bayesian_fit_sed_batch_old1(obs_fluxes, obs_flux_errs, model_seds, params_val_list, param_priors=None, 
						likelihood_type='gauss', dof=2.0, perc_chi2=90.0, n_cpu=None, sfh_form=4):
    """
    Batch fitting multiple observed SEDs using the Bayesian likelihood-weighted method.
    
    Optimized: Priors independent of normalization (Gaussian/Uniform) are pre-calculated.
    Joint priors are calculated in-batch using the best-fit normalized mass as the anchor.
    """
    
    if n_cpu is None:
        from joblib import cpu_count
        n_cpu = cpu_count()
        
    if param_priors is None:
        param_priors = {}
    
    N_models = len(params_val_list)
    
    # 1. --- PRE-CALCULATE log_mw_age AND SFRs FOR ALL MODELS (UN-NORMALIZED) ---
    print("Pre-calculating model-dependent parameters (log_mw_age, log_sfr)...")
    log_mw_ages, log_sfrs_inst, log_sfrs_100 = calculate_log_mw_age_and_sfr_for_models(params_val_list, sfh_form)
    
    # --- AUGMENT params_val_list WITH log_mw_age, log_sfr_inst, log_sfr_100, and 'raw' log_mass ---
    # Convert to a more efficient structure (e.g., NumPy record array or dict of arrays) for batch ops.
    model_params_raw = {}
    for k in params_val_list[0].keys():
        model_params_raw[k] = np.array([p.get(k) for p in params_val_list])
    
    # Add calculated parameters
    model_params_raw['log_mw_age'] = np.array(log_mw_ages)
    model_params_raw['log_sfr_inst'] = np.array(log_sfrs_inst) 
    model_params_raw['log_sfr_100'] = np.array(log_sfrs_100)
    
    
    # 2. --- BATCH PRE-CALCULATE PRIORS INDEPENDENT OF NORMALIZATION (Gaussian/Uniform) ---
    prior_list_independent = np.ones(N_models)
    joint_prior_defs = {}

    for param, prior_def in param_priors.items():
        form = prior_def['form']
        if form == 'uniform':
            # Uniform prior is 1.0, already handled by initialization
            continue
            
        elif form == 'gaussian':
            loc = prior_def['loc']
            scale = prior_def['scale']
            param_values = model_params_raw.get(param)
            
            if param_values is not None:
                # Vectorized Gaussian PDF (un-normalized P(x) proportional to exp(-0.5 * z^2))
                term = (param_values - loc) / scale
                prior_prob = np.exp(-0.5 * term**2)
                
                # Apply as a multiplicative factor
                prior_list_independent *= prior_prob

        elif form == 'joint_with_mass':
            # Store joint prior definitions for in-loop calculation
            joint_prior_defs[param] = prior_def
            # Need interp1d inside the loop, so initialize here if not done
            from scipy.interpolate import interp1d
            
    # Normalize the independent prior component to avoid underflow
    max_prior_indep = np.max(prior_list_independent[np.isfinite(prior_list_independent)])
    if max_prior_indep > 0.0:
        prior_list_independent[np.isfinite(prior_list_independent)] /= max_prior_indep
    else:
        # If all independent priors are 0 (e.g., due to log(0)), treat as uniform 1.0
        prior_list_independent[:] = 1.0


    # 3. --- START BATCH FITTING LOOP (one loop per observed SED) ---
    print("Starting batch SED fitting with anchor-based priors...")
    all_fit_results = []
    
    for obs_flux, obs_flux_err in tqdm(zip(obs_fluxes, obs_flux_errs), total=len(obs_fluxes), desc="Batch SED Fitting (Bayesian)"):
        
        # 3a. Calculate Chi-square and Normalization for all models/SED
        def compute_all_chisq_for_one_sed(obs_f, obs_ferr, model_seds, n_jobs):
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_chisq)(i, model_seds[i], obs_f, obs_ferr)
                for i in range(model_seds.shape[0])
            )
            return zip(*results) 

        _, chi_sq_list, norm_list = compute_all_chisq_for_one_sed(obs_flux, obs_flux_err, model_seds, n_cpu)
        norm_arr = np.array(norm_list)
        chi_sq_arr = np.array(chi_sq_list)
        
        # Find the *best-fit* normalization (anchor)
        min_chi_sq_idx = np.argmin(chi_sq_arr)
        best_fit_norm = norm_arr[min_chi_sq_idx]
        
        # 3b. Calculate Likelihood for all models
        likelihood_list, _ = calculate_likelihood(
            obs_flux, obs_flux_err, model_seds, chi_sq_arr, likelihood_type, dof
        )
        
        # 3c. Calculate Normalized log_mass (The Anchor Value)
        # log_mass_raw: the mass that formed the stellar population (M_formed)
        log_mass_raw_arr = model_params_raw['log_mass']
        log_norm_arr = np.log10(norm_arr, out=np.full_like(norm_arr, np.nan), where=norm_arr > 0.0)
        
        # This is the observation's log_mass: log10(M_formed * normalization)
        normalized_log_mass_arr = log_mass_raw_arr + log_norm_arr
        
        # Find the best-fit normalized log_mass (The Anchor Value)
        # Note: We take the best-fit model's raw log_mass and apply its norm
        best_fit_normalized_log_mass = normalized_log_mass_arr[min_chi_sq_idx]
        
        
        # 3d. Anchor-Based Joint Prior Calculation
        prior_list_joint = np.ones(N_models)
        
        if len(joint_prior_defs) > 0:
            if np.isnan(best_fit_normalized_log_mass):
                # If the anchor mass is invalid, all joint priors are zero
                prior_list_joint[:] = 0.0
            else:
                for param, prior_def in joint_prior_defs.items():
                    # Check if the anchor parameter is available and valid
                    if param not in ['log_mw_age', 'logzsol']:
                         continue

                    # Anchor interpolation setup
                    corr_log_mass = prior_def['corr_log_mass']
                    corr_pval = prior_def['corr_pval']
                    scale = prior_def['scale']
                    
                    # Interpolate the median relation: find the expected 'loc' for the best-fit normalized log_mass
                    median_relation_func = interp1d(corr_log_mass, corr_pval, kind='linear', fill_value='extrapolate')
                    loc_from_mass = median_relation_func(best_fit_normalized_log_mass)

                    # Model parameter values to check against the anchor
                    param_values = model_params_raw.get(param)
                    
                    if param_values is not None:
                        # Vectorized Gaussian Prior centered at the *anchor location*
                        term = (param_values - loc_from_mass) / scale
                        prior_prob = np.exp(-0.5 * term**2)
                        
                        # Apply as a multiplicative factor
                        prior_list_joint *= prior_prob

        # 3e. Total Prior: independent * joint
        prior_list_total = prior_list_independent * prior_list_joint
        # Normalize the total prior component to avoid underflow
        max_prior_total = np.max(prior_list_total[np.isfinite(prior_list_total)])
        if max_prior_total > 0.0:
            prior_list_total[np.isfinite(prior_list_total)] /= max_prior_total
        else:
            prior_list_total[:] = 0.0 # Prior is 0 if no valid priors found

        # 3f. Calculate Total Posterior
        total_posterior_list = likelihood_list * prior_list_total
        
        # Normalize the posterior to sum to 1
        total_posterior_sum = np.sum(total_posterior_list[np.isfinite(total_posterior_list)])
        if total_posterior_sum > 0:
            total_posterior_list = total_posterior_list / total_posterior_sum
        else:
             total_posterior_list = np.zeros_like(total_posterior_list)
             
        # 3g. Prepare Normalized Parameter List (for Bayesian inference)
        # This part still requires zipping and is best done as a list of dicts.
        normalized_params_val_list = []
        for i in range(N_models):
            # Create a dictionary of normalized parameters for the current model
            normalized_pv = {}
            # Start with raw parameters (non-mass/SFR parameters are not normalized)
            for k, v_arr in model_params_raw.items():
                normalized_pv[k] = v_arr[i]
                
            # Overwrite mass/SFR with normalized values
            normalized_pv['log_mass'] = normalized_log_mass_arr[i]
            
            # log_sfr_inst normalization
            if 'log_sfr_inst' in normalized_pv and not np.isnan(log_norm_arr[i]):
                 normalized_pv['log_sfr_inst'] += log_norm_arr[i]
            else:
                 normalized_pv['log_sfr_inst'] = np.nan
                 
            # log_sfr_100 normalization
            if 'log_sfr_100' in normalized_pv and not np.isnan(log_norm_arr[i]):
                 normalized_pv['log_sfr_100'] += log_norm_arr[i]
            else:
                 normalized_pv['log_sfr_100'] = np.nan
            
            # Handle models with zero likelihood/norm (set non-fixed/non-age params to NaN)
            if norm_arr[i] <= 0.0:
                 # In this case, the posterior is 0, so the inference will ignore it,
                 # but we can set mass/sfr to NaN for clarity.
                 normalized_pv['log_mass'] = np.nan
                 normalized_pv['log_sfr_inst'] = np.nan
                 normalized_pv['log_sfr_100'] = np.nan

            normalized_params_val_list.append(normalized_pv)

        # 3h. Perform Bayesian parameter inference
        inferred_params = bayesian_inference_parameters(
            normalized_params_val_list, total_posterior_list, perc_chi2
        )
        
        all_fit_results.append({
            'inferred_params': inferred_params,
            'min_chi_sq': chi_sq_arr[min_chi_sq_idx],
            'best_fit_model_idx': min_chi_sq_idx,
            'model_likelihoods': likelihood_list,
            'model_priors': prior_list_total,
            'model_posteriors': total_posterior_list,
            'model_chi_sq': chi_sq_arr,
            'model_norms': norm_arr,
        })

    return all_fit_results
