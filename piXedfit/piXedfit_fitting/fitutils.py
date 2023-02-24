import numpy as np
import h5py
from astropy.io import fits


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
	flg_write = []
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
		elif form == 'arbitrary':
			file_out.write("pr_form_%s_arbit_name %s\n" % (param,priors[2]))
			flg_write.append(priors[2])
		elif form == 'joint_with_mass':
			file_out.write("pr_form_%s_jtmass_name %s\n" % (param,priors[2]))
			file_out.write("pr_form_%s_jtmass_scale %s\n" % (param,priors[3]))
	file_out.close()

	return flg_write



