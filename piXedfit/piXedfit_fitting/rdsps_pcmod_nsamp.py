import numpy as np
from math import log10, pow, sqrt 
import sys, os
import fsps
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from functools import reduce

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.posteriors import model_leastnorm, calc_chi2, gauss_prob, gauss_prob_reduced, student_t_prob
from piXedfit.piXedfit_model import calc_mw_age, generate_modelSED_spec_decompose
from piXedfit.utils.filtering import cwave_filters


def bayesian_sedfit_gauss():
	# open the input FITS file
	hdu = fits.open(name_saved_randmod)
	data_randmod = hdu[1].data
	hdu.close()

	#redcd_chi2 = float(config_data['redc_chi2_initfit'])

	numDataPerRank = int(npmod_seds/size)
	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	sampler_log_mass_temp = np.zeros(numDataPerRank)
	sampler_log_sfr_temp = np.zeros(numDataPerRank)
	sampler_log_mw_age_temp = np.zeros(numDataPerRank)
	if duste_switch == 'duste':
		sampler_logdustmass_temp = np.zeros(numDataPerRank)
	if add_agn == 1:
		sampler_log_fagn_bol_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		for pp in range(0,nparams): 
			temp_val = data_randmod[params[pp]][int(ii)]
			mod_params_temp[pp][int(count)] = temp_val

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = data_randmod[filters[bb]][int(ii)]
		norm0 = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		mod_fluxes0 = norm0*fluxes

		sampler_log_mass_temp[int(count)] = data_randmod['log_mass'][int(ii)]+log10(norm0)
		sampler_log_sfr_temp[int(count)] = data_randmod['log_sfr'][int(ii)]+log10(norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = data_randmod['log_dustmass'][int(ii)]+log10(norm0)
		if add_agn == 1:
			sampler_log_fagn_bol_temp[int(count)] = data_randmod['log_fagn_bol'][int(ii)]

		# calculate MW-age
		formed_mass = pow(10.0,data_randmod['log_mass'][int(ii)])
		age = pow(10.0,data_randmod['log_age'][int(ii)])
		tau = pow(10.0,data_randmod['log_tau'][int(ii)])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0
		if sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
			t0 = pow(10.0,data_randmod['log_t0'][int(ii)])
		if sfh_form == 'double_power_sfh':
			alpha = pow(10.0,data_randmod['log_alpha'][int(ii)])
			beta = pow(10.0,data_randmod['log_beta'][int(ii)])
		mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
											age=age,formed_mass=formed_mass)
		sampler_log_mw_age_temp[int(count)] = log10(mw_age)

		# calculate chi-square and prob
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,mod_fluxes0)

		if gauss_likelihood_form == 0:
			prob0 = gauss_prob(obs_fluxes,obs_flux_err,mod_fluxes0)
		elif gauss_likelihood_form == 1:
			prob0 = gauss_prob_reduced(obs_fluxes,obs_flux_err,mod_fluxes0)

		mod_chi2_temp[int(count)] = chi2
		mod_prob_temp[int(count)] = prob0
		for bb in range(0,nbands):
			mod_fluxes_temp[bb][int(count)] = mod_fluxes0[bb]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	mod_params = np.zeros((nparams,numDataPerRank*size))
	mod_fluxes = np.zeros((nbands,numDataPerRank*size))
	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_prob = np.zeros(numDataPerRank*size)

	# additional parameters
	sampler_log_mass = np.zeros(numDataPerRank*size)
	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_log_mw_age = np.zeros(numDataPerRank*size)
	if duste_switch == 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn == 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

	# additional properties:
	comm.Gather(sampler_log_mass_temp, sampler_log_mass, root=0)
	comm.Gather(sampler_log_sfr_temp, sampler_log_sfr, root=0)
	comm.Gather(sampler_log_mw_age_temp, sampler_log_mw_age, root=0)
	if duste_switch == 'duste':
		comm.Gather(sampler_logdustmass_temp, sampler_logdustmass, root=0)
	if add_agn == 1:
		comm.Gather(sampler_log_fagn_bol_temp, sampler_log_fagn_bol, root=0)
	
	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = mod_fluxes[bb][idx0]

		print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))
			status_add_err[0] = 1
		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		# Broadcast
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)
		 
	elif status_add_err[0] == 1:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

		# transpose
		mod_fluxes1 = np.transpose(mod_fluxes, axes=(1,0))     # become [idx-model][idx-band] 

		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')

		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_prob_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			fluxes = mod_fluxes1[int(ii)]

			chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)

			if gauss_likelihood_form == 0:
				prob0 = gauss_prob(obs_fluxes,modif_obs_flux_err,fluxes)
			elif gauss_likelihood_form == 1:
				prob0 = gauss_prob_reduced(obs_fluxes,modif_obs_flux_err,fluxes)

			mod_chi2_temp[int(count)] = chi2
			mod_prob_temp[int(count)] = prob0
			count = count + 1

		mod_prob = np.zeros(numDataPerRank*size)
		mod_chi2 = np.zeros(numDataPerRank*size)
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_prob_temp, mod_prob, root=0)

		# Broadcast
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		## end of status_add_err[0] == 1

	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn != 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	return mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol


def bayesian_sedfit_student_t():
	# open the input FITS file
	hdu = fits.open(name_saved_randmod)
	data_randmod = hdu[1].data
	hdu.close()

	#redcd_chi2 = float(config_data['redc_chi2_initfit'])

	numDataPerRank = int(npmod_seds/size)
	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	sampler_log_mass_temp = np.zeros(numDataPerRank)
	sampler_log_sfr_temp = np.zeros(numDataPerRank)
	sampler_log_mw_age_temp = np.zeros(numDataPerRank)
	if duste_switch == 'duste':
		sampler_logdustmass_temp = np.zeros(numDataPerRank)
	if add_agn == 1:
		sampler_log_fagn_bol_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		for pp in range(0,nparams): 
			temp_val = data_randmod[params[pp]][int(ii)]
			mod_params_temp[pp][int(count)] = temp_val

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = data_randmod[filters[bb]][int(ii)]

		norm0 = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		mod_fluxes0 = norm0*fluxes

		sampler_log_mass_temp[int(count)] = data_randmod['log_mass'][int(ii)]+log10(norm0)
		sampler_log_sfr_temp[int(count)] = data_randmod['log_sfr'][int(ii)]+log10(norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = data_randmod['log_dustmass'][int(ii)]+log10(norm0)
		if add_agn == 1:
			sampler_log_fagn_bol_temp[int(count)] = data_randmod['log_fagn_bol'][int(ii)]

		# mass-weighted age
		formed_mass = pow(10.0,data_randmod['log_mass'][int(ii)])
		age = pow(10.0,data_randmod['log_age'][int(ii)])
		tau = pow(10.0,data_randmod['log_tau'][int(ii)])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0
		if sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
			t0 = pow(10.0,data_randmod['log_t0'][int(ii)])
		if sfh_form == 'double_power_sfh':
			alpha = pow(10.0,data_randmod['log_alpha'][int(ii)])
			beta = pow(10.0,data_randmod['log_beta'][int(ii)])
		mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
											age=age,formed_mass=formed_mass)
		sampler_log_mw_age_temp[int(count)] = log10(mw_age)

		# calculate chi-square and prob.
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,mod_fluxes0)
		chi = (obs_fluxes-mod_fluxes0)/obs_flux_err
		prob0 = student_t_prob(dof,chi)

		mod_chi2_temp[int(count)] = chi2
		mod_prob_temp[int(count)] = prob0
		for bb in range(0,nbands):
			mod_fluxes_temp[bb][int(count)] = mod_fluxes0[bb]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	mod_params = np.zeros((nparams,numDataPerRank*size))
	mod_fluxes = np.zeros((nbands,numDataPerRank*size))
	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_prob = np.zeros(numDataPerRank*size)

	sampler_log_mass = np.zeros(numDataPerRank*size)
	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_log_mw_age = np.zeros(numDataPerRank*size)
	if duste_switch == 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn == 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)
				
	# gather the scattered data
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)
	# additional properties
	comm.Gather(sampler_log_mass_temp, sampler_log_mass, root=0)
	comm.Gather(sampler_log_sfr_temp, sampler_log_sfr, root=0)
	comm.Gather(sampler_log_mw_age_temp, sampler_log_mw_age, root=0)
	if duste_switch == 'duste':
		comm.Gather(sampler_logdustmass_temp, sampler_logdustmass, root=0)
	if add_agn == 1:
		comm.Gather(sampler_log_fagn_bol_temp, sampler_log_fagn_bol, root=0)
	
	# allocate memory
	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = mod_fluxes[bb][idx0]

		print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2: 
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))

			status_add_err[0] = 1

		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	# Broadcast
	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		# Broadcast
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

	elif status_add_err[0] == 1:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

		mod_fluxes1 = np.transpose(mod_fluxes, axes=(1,0))     # become [idx-model][idx-band] 

		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_prob_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			fluxes = mod_fluxes1[int(ii)]

			# calculate model's chi2 and prob.
			chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)
			chi = (obs_fluxes-fluxes)/modif_obs_flux_err
			prob0 = student_t_prob(dof,chi)

			mod_chi2_temp[int(count)] = chi2
			mod_prob_temp[int(count)] = prob0
			count = count + 1

		mod_prob = np.zeros(numDataPerRank*size)
		mod_chi2 = np.zeros(numDataPerRank*size)
				
		## gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_prob_temp, mod_prob, root=0)

		# Broadcast
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		# end of status_add_err[0] == 1

	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn != 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	return mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol


def store_to_fits(sampler_params=None,sampler_log_mass=None,sampler_log_sfr=None,sampler_log_mw_age=None,
	sampler_logdustmass=None,sampler_log_fagn_bol=None,mod_chi2=None,mod_prob=None,fits_name_out=None):

	#==> Get best-fit parameters
	crit_chi2 = np.percentile(mod_chi2, perc_chi2)
	idx_sel = np.where((mod_chi2<=crit_chi2) & (sampler_log_sfr>-29.0) & (np.isnan(mod_prob)==False) & (np.isinf(mod_prob)==False))

	array_prob = mod_prob[idx_sel[0]]
	tot_prob = np.sum(array_prob)

	params_bfits = np.zeros((nparams,2))
	for pp in range(0,nparams):
		array_val = sampler_params[params[pp]][idx_sel[0]]

		mean_val = np.sum(array_val*array_prob)/tot_prob
		mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
		std_val = sqrt(abs(mean_val2 - (mean_val**2)))

		params_bfits[pp][0] = mean_val
		params_bfits[pp][1] = std_val

	#log_mass
	array_val = sampler_log_mass[idx_sel[0]]
	mean_val = np.sum(array_val*array_prob)/tot_prob
	mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
	std_val = sqrt(abs(mean_val2 - (mean_val**2)))
	log_mass_mean = mean_val
	log_mass_std = std_val

	#log_sfr
	array_val = sampler_log_sfr[idx_sel[0]]
	mean_val = np.sum(array_val*array_prob)/tot_prob
	mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
	std_val = sqrt(abs(mean_val2 - (mean_val**2)))
	log_sfr_mean = mean_val
	log_sfr_std = std_val

	#log_mw_age
	array_val = sampler_log_mw_age[idx_sel[0]]
	mean_val = np.sum(array_val*array_prob)/tot_prob
	mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
	std_val = sqrt(abs(mean_val2 - (mean_val**2)))
	log_mw_age_mean = mean_val
	log_mw_age_std = std_val

	#dust
	if duste_switch == 'duste':
		array_val = sampler_logdustmass[idx_sel[0]]
		mean_val = np.sum(array_val*array_prob)/tot_prob
		mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
		std_val = sqrt(abs(mean_val2 - (mean_val**2)))
		logdustmass_mean = mean_val
		logdustmass_std = std_val

	#agn
	if add_agn == 1: 
		array_val = sampler_log_fagn_bol[idx_sel[0]]
		mean_val = np.sum(array_val*array_prob)/tot_prob
		mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
		std_val = sqrt(abs(mean_val2 - (mean_val**2)))
		log_fagn_bol_mean = mean_val
		log_fagn_bol_std = std_val


	#==> Get best-fit model spectrum
	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	idx, min_val = min(enumerate(mod_chi2), key=operator.itemgetter(1))
	# best-fit chi-square
	bfit_chi2 = mod_chi2[idx]

	# call fsps
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	# generate the spectrum
	params_val = def_params_val
	for pp in range(0,nparams):
		params_val[params[pp]] = sampler_params[params[pp]][idx]
	params_val['log_mass'] = sampler_log_mass[idx]

	spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo_str,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=='erg/s/cm2/A')

	# get the photometric SED:
	bfit_photo_SED = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)


	# store to FITS file
	hdr = fits.Header()
	hdr['imf'] = imf
	hdr['nparams'] = nparams
	hdr['sfh_form'] = sfh_form
	hdr['dust_ext_law'] = dust_ext_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
	if duste_switch == 'duste':
		if fix_dust_index == 1:
			hdr['dust_index'] = fix_dust_index_val
	hdr['add_neb_emission'] = add_neb_emission
	if add_neb_emission == 1:
		hdr['gas_logu'] = gas_logu
	hdr['add_agn'] = add_agn
	hdr['add_igm_absorption'] = add_igm_absorption
	hdr['likelihood_form'] = likelihood_form
	if likelihood_form == 'student_t':
		hdr['dof'] = dof
	if add_igm_absorption == 1:
		hdr['igm_type'] = igm_type
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		hdr[str_temp] = filters[bb]
		str_temp = 'flux%d' % bb
		hdr[str_temp] = obs_fluxes[bb]
		str_temp = 'flux_err%d' % bb 
		hdr[str_temp] = obs_flux_err[bb]
	if free_z == 0:
		hdr['gal_z'] = gal_z
		hdr['free_z'] = 0
	elif free_z == 1:
		hdr['free_z'] = 1
	hdr['cosmo'] = cosmo_str
	hdr['H0'] = H0
	hdr['Om0'] = Om0

	# parameters
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]

	# chi-square
	hdr['redcd_chi2'] = bfit_chi2/nbands
	hdr['perc_chi2'] = perc_chi2

	# add columns
	cols0 = []
	col_count = 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'rows'
	col = fits.Column(name='rows', format='4A', array=['mean','std'])
	cols0.append(col)

	#=> basic params
	for pp in range(0,nparams):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = params[pp]
		col = fits.Column(name=params[pp], format='D', array=np.array([params_bfits[pp][0],params_bfits[pp][1]]))
		cols0.append(col)

	#=> log_mass
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mass'
	col = fits.Column(name='log_mass', format='D', array=np.array([log_mass_mean,log_mass_std]))
	cols0.append(col)

	#=> log_sfr
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_sfr'
	col = fits.Column(name='log_sfr', format='D', array=np.array([log_sfr_mean,log_sfr_std]))
	cols0.append(col)

	#=> log_mw_age
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mw_age'
	col = fits.Column(name='log_mw_age', format='D', array=np.array([log_mw_age_mean,log_mw_age_std]))
	cols0.append(col)

	#=> dust
	if duste_switch == 'duste':
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_dustmass'
		col = fits.Column(name='log_dustmass', format='D', array=np.array([logdustmass_mean,logdustmass_std]))
		cols0.append(col)

	#agn
	if add_agn == 1: 
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_fagn_bol'
		col = fits.Column(name='log_fagn_bol', format='D', array=np.array([log_fagn_bol_mean,log_fagn_bol_std]))
		cols0.append(col)

	hdr['ncols'] = col_count

	# combine header
	primary_hdu = fits.PrimaryHDU(header=hdr)

	# combine binary table HDU1
	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)

	#==> make new table for best-fit spectra
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(spec_SED['wave']))
	cols0.append(col)
	col = fits.Column(name='flux_total', format='D', array=np.array(spec_SED['flux_total']))
	cols0.append(col)
	col = fits.Column(name='flux_stellar', format='D', array=np.array(spec_SED['flux_stellar']))
	cols0.append(col)
	if add_neb_emission == 1:
		col = fits.Column(name='flux_nebe', format='D', array=np.array(spec_SED['flux_nebe']))
		cols0.append(col)
	if duste_switch == 'duste':
		col = fits.Column(name='flux_duste', format='D', array=np.array(spec_SED['flux_duste']))
		cols0.append(col)
	if add_agn == 1:
		col = fits.Column(name='flux_agn', format='D', array=np.array(spec_SED['flux_agn']))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu1 = fits.BinTableHDU.from_columns(cols)

	#==> make new table for best-fit photometry
	photo_cwave = cwave_filters(filters)

	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(photo_cwave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(bfit_photo_SED))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu2 = fits.BinTableHDU.from_columns(cols)

	# combine all
	hdul = fits.HDUList([primary_hdu, hdu, hdu1, hdu2])
	hdul.writeto(fits_name_out, overwrite=True)


"""
USAGE: mpirun -np [npros] python ./rdsps_pcmod.py (1)name_filters_list (2)name_config (3)name_SED_txt (4)name_out_fits
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank() 

# configuration file
#global config_data
config_file = str(sys.argv[2])
data = np.genfromtxt(temp_dir+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]

# filters
global filters, nbands
name_filters = str(sys.argv[1])
filters = np.genfromtxt(temp_dir+name_filters, dtype=str)
nbands = len(filters)

# FITS file containing pre-calculated model SEDs
name_saved_randmod = config_data['name_saved_randmod']

# data of pre-calculated model SEDs
global header_randmod, data_randmod
hdu = fits.open(name_saved_randmod)
header_randmod = hdu[0].header
data_randmod = hdu[1].data
hdu.close()

# number of model SEDs:
global npmod_seds
npmod_seds = int(header_randmod['nrows'])
npmod_seds = int(npmod_seds/size)*size

global add_neb_emission
add_neb_emission = int(header_randmod['add_neb_emission'])

global gas_logu
gas_logu = float(header_randmod['gas_logu'])

global sfh_form
sfh_form = header_randmod['sfh_form']

# redshift
global gal_z
gal_z = float(header_randmod['gal_z'])

global imf
imf = int(header_randmod['imf_type'])

global duste_switch
duste_switch = header_randmod['duste_switch']

global fix_dust_index, fix_dust_index_val
if duste_switch == 'duste':
	if 'dust_index' in header_randmod:
		fix_dust_index_val = float(header_randmod['dust_index'])
		fix_dust_index = 1
	else:
		fix_dust_index_val = 0
		fix_dust_index = 0

global dust_ext_law
dust_ext_law = header_randmod['dust_ext_law']

global add_igm_absorption,igm_type
add_igm_absorption = int(header_randmod['add_igm_absorption'])
#igm_type = int(header_randmod['igm_type'])
if add_igm_absorption == 1:
	igm_type = int(header_randmod['igm_type'])

global add_agn 
add_agn = int(header_randmod['add_agn'])

global likelihood_form
likelihood_form = config_data['likelihood']

# degree of freedom in the student's t likelihood function -> only relevant if likelihood_form='student_t'
global dof
dof = float(config_data['dof'])

global gauss_likelihood_form
gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

# input SED
global obs_fluxes, obs_flux_err
inputSED_txt = str(sys.argv[3])
data = np.loadtxt(temp_dir+inputSED_txt)
obs_fluxes = np.asarray(data[:,0])
obs_flux_err = np.asarray(data[:,1])

global free_z
if gal_z <= 0:
	free_z = 1
else:
	free_z = 0

# comology
global cosmo_str, H0, Om0
cosmo_str = header_randmod['cosmo']
H0 = float(header_randmod['H0'])
Om0 = float(header_randmod['Om0'])

# list of parameters
global params, nparams
nparams = int(header_randmod['nparams'])
params = []
for ii in range(0,nparams):
	str_temp = 'param%d' % ii
	params.append(header_randmod[str_temp])


global redcd_chi2
redcd_chi2 = float(config_data['redc_chi2_initfit'])

global perc_chi2
perc_chi2 = float(config_data['perc_chi2'])


# running the calculation
if likelihood_form == 'gauss':
	mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol = bayesian_sedfit_gauss()
elif likelihood_form == 'student_t':
	mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol = bayesian_sedfit_student_t()
else:
	print ("likelihood_form is not recognized!")
	sys.exit()

if np.sum(modif_obs_flux_err) != 0:
	obs_flux_err = modif_obs_flux_err

# change the format to dictionary
nsamples = len(mod_params[0])
sampler_params = {}
for pp in range(0,nparams):
	sampler_params[params[pp]] = np.zeros(nsamples)
	sampler_params[params[pp]] = mod_params[pp]

# store to fits file
if rank == 0:
	fits_name_out = str(sys.argv[4])
	store_to_fits(sampler_params=sampler_params,sampler_log_mass=sampler_log_mass,sampler_log_sfr=sampler_log_sfr,
				sampler_log_mw_age=sampler_log_mw_age,sampler_logdustmass=sampler_logdustmass, sampler_log_fagn_bol=sampler_log_fagn_bol,
				mod_chi2=mod_chi2,mod_prob=mod_prob,fits_name_out=fits_name_out)


