import h5py
import sys, os
import numpy as np
from astropy.io import fits

from ..piXedfit_model.model_utils import default_params_val

__all__ = ["nproc_reduced", "randname", "write_filters_list", "write_input_singleSED_photo", "write_input_specphoto_hdf5", 
			"write_arbitprior", "write_joint_prior", "write_conf_file", "read_config_file_fit", "get_nproc", 
			"define_free_z", "get_name_out_fits", "remove_files", "run_fitting", "write_input_spec_hdf5", 
			"make_bins_name_out_fits", "define_free_z_bins_fits"]

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

def define_free_z_bins_fits(free_z, gal_z):
	if free_z == 0:
		if gal_z is None or gal_z<=0.0:
			gal_z = float(header['z'])
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
				params_priors[params[pp]]['loc'] = float(config_data['pr_form_%s_gauss_loc' % params[pp]])
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_gauss_scale' % params[pp]])
			elif params_priors[params[pp]]['form'] == 'studentt':
				params_priors[params[pp]]['df'] = float(config_data['pr_form_%s_stdt_df' % params[pp]])
				params_priors[params[pp]]['loc'] = float(config_data['pr_form_%s_stdt_loc' % params[pp]])
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_stdt_scale' % params[pp]])
			elif params_priors[params[pp]]['form'] == 'gamma':
				params_priors[params[pp]]['a'] = float(config_data['pr_form_%s_gamma_a' % params[pp]])
				params_priors[params[pp]]['loc'] = float(config_data['pr_form_%s_gamma_loc' % params[pp]])
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_gamma_scale' % params[pp]])
			elif params_priors[params[pp]]['form'] == 'arbitrary':
				data = np.loadtxt(temp_dir+config_data['pr_form_%s_arbit_name' % params[pp]])
				params_priors[params[pp]]['values'] = data[:,0]
				params_priors[params[pp]]['prob'] = data[:,1]
			elif params_priors[params[pp]]['form'] == 'joint_with_mass':
				data = np.loadtxt(temp_dir+config_data['pr_form_%s_jtmass_name' % params[pp]])
				params_priors[params[pp]]['lmass'] = data[:,0]
				params_priors[params[pp]]['pval'] = data[:,1]
				params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_jtmass_scale' % params[pp]])
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
	nproc,nproc_new,CODE_dir,name_config,models_spec,store_full_samplers,name_out_fits):

	name_out_fits = get_name_out_fits(name_out_fits,fit_method)

	if obs_flux is None:
		name_filters_list = None
		# spec only
		inputSED_file = write_input_spec_hdf5(temp_dir,spec_wave,spec_flux,spec_flux_err,wavelength_range)

		if fit_method=='mcmc' or fit_method=='MCMC':
			name_samplers_hdf5 = randname("samplers_",".hdf5")

			os.system("mpirun -n %d python %s./mc_p1_s.py %s %s %s %s" % (nproc,CODE_dir,name_config,inputSED_file,name_samplers_hdf5,models_spec))
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
					os.system("mpirun -n %d python %s./rd_fz_s.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_s.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
			elif store_full_samplers==0 or store_full_samplers==False:
				if free_z==0:
					os.system("mpirun -n %d python %s./rd_fz_nsmp_s.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
				elif free_z==1:
					os.system("mpirun -n %d python %s./rd_vz_nsmp_s.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
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

				os.system("mpirun -n %d python %s./mc_p1.py %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,inputSED_file,name_samplers_hdf5,models_spec))
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
						os.system("mpirun -n %d python %s./rd_fz.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
				elif store_full_samplers==0 or store_full_samplers==False:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_nsmp.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_nsmp.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
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

				os.system("mpirun -n %d python %s./mc_p1_sp.py %s %s %s %s %s" % (nproc,CODE_dir,name_filters_list,name_config,inputSED_file,name_samplers_hdf5,models_spec))
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
						os.system("mpirun -n %d python %s./rd_fz_sp.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_sp.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
				elif store_full_samplers==0 or store_full_samplers==False:
					if free_z==0:
						os.system("mpirun -n %d python %s./rd_fz_nsmp_sp.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
					elif free_z==1:
						os.system("mpirun -n %d python %s./rd_vz_nsmp_sp.py %s %s %s %s %s" % (nproc_new,CODE_dir,name_filters_list,name_config,inputSED_file,name_out_fits,models_spec))
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
		











