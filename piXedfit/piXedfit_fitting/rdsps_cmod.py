import numpy as np
from math import log10, pow
import sys, os
import fsps
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from astropy.cosmology import *
from functools import reduce

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.filtering import match_filters_array
from piXedfit.utils.posteriors import model_leastnorm, calc_chi2, gauss_prob, gauss_prob_reduced, student_t_prob
from piXedfit.piXedfit_model import generate_modelSED_propphoto_nomwage_fit, calc_mw_age, generate_modelSED_spec_fit


def bayesian_sedfit_gauss():
	redcd_chi2 = float(config_data['redc_chi2_initfit'])
	
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
		params_val = def_params_val
		for pp in range(0,nparams): 
			temp_val = np.random.uniform(priors_min[pp],priors_max[pp])
			mod_params_temp[pp][int(count)] = temp_val
			params_val[params[pp]] = temp_val

		SED_prop,photo_SED = generate_modelSED_propphoto_nomwage_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,filters=filters,
												add_igm_absorption=add_igm_absorption,igm_type=igm_type,params_fsps=params_fsps,
												params_val=params_val,DL_Gpc=DL_Gpc,cosmo=cosmo_str,H0=H0,Om0=Om0,free_z=free_z,
												trans_fltr_int=trans_fltr_int)

		fluxes = photo_SED['flux']
		norm0 = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		mod_fluxes0 = norm0*fluxes

		sampler_log_mass_temp[int(count)] = log10(norm0)
		sampler_log_sfr_temp[int(count)] = log10(SED_prop['SFR']*norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = log10(SED_prop['dust_mass']*norm0)
		if add_agn == 1:
			sampler_log_fagn_bol_temp[int(count)] = SED_prop['log_fagn_bol']

		# calculate mass-weighted age
		age = pow(10.0,params_val['log_age'])
		tau = pow(10.0,params_val['log_tau'])
		t0 = pow(10.0,params_val['log_t0'])
		alpha = pow(10.0,params_val['log_alpha'])
		beta = pow(10.0,params_val['log_beta'])
		mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
											age=age,formed_mass=SED_prop['SM']*norm0)
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
				# modifiy observed flux errors:
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

		# end of status_add_err[0] == 1

	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn != 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	return mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol


def bayesian_sedfit_student_t():
	redcd_chi2 = float(config_data['redc_chi2_initfit'])
	
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
		params_val = def_params_val
		for pp in range(0,nparams): 
			temp_val = np.random.uniform(priors_min[pp],priors_max[pp])
			mod_params_temp[pp][int(count)] = temp_val
			params_val[params[pp]] = temp_val

		# get model SED and properties
		SED_prop,photo_SED = generate_modelSED_propphoto_nomwage_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,filters=filters,
												add_igm_absorption=add_igm_absorption,igm_type=igm_type,params_fsps=params_fsps,
												params_val=params_val,DL_Gpc=DL_Gpc,cosmo=cosmo_str,H0=H0,Om0=Om0,free_z=free_z,
												trans_fltr_int=trans_fltr_int)

		fluxes = photo_SED['flux']
		norm0 = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		mod_fluxes0 = norm0*fluxes

		sampler_log_mass_temp[int(count)] = log10(norm0)
		sampler_log_sfr_temp[int(count)] = log10(SED_prop['SFR']*norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = log10(SED_prop['dust_mass']*norm0)
		if add_agn == 1:
			sampler_log_fagn_bol_temp[int(count)] = SED_prop['log_fagn_bol']

		# calculate mass-weighted age:
		age = pow(10.0,params_val['log_age'])
		tau = pow(10.0,params_val['log_tau'])
		t0 = pow(10.0,params_val['log_t0'])
		alpha = pow(10.0,params_val['log_alpha'])
		beta = pow(10.0,params_val['log_beta'])
		mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
											age=age,formed_mass=SED_prop['SM']*norm0)
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
				
		# gather the scattered data
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


def store_to_fits(nsamples=None,sampler_params=None,sampler_log_mass=None,sampler_log_sfr=None,sampler_log_mw_age=None,
	sampler_logdustmass=None,sampler_log_fagn_bol=None,mod_fluxes=None,mod_chi2=None,mod_prob=None,cosmo_str='flat_LCDM',H0=70.0,Om0=0.3,fits_name_out=None): 

	sampler_id = np.linspace(1, nsamples, nsamples)

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
	hdr['nrows'] = nsamples

	# parameters
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]

	col_count = 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'id'
	for pp in range(0,nparams):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = params[pp]

	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mass'
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_sfr'
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mw_age'

	if duste_switch == 'duste':
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_dustmass'

	if add_agn == 1:
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_fagn_bol'

	for bb in range(0,nbands):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = filters[bb]

	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'chi2'
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'prob'
	hdr['ncols'] = col_count

	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(sampler_id))
	cols0.append(col)
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(sampler_params[params[pp]]))
		cols0.append(col)
	col = fits.Column(name='log_mass', format='D', array=np.array(sampler_log_mass))
	cols0.append(col)
	col = fits.Column(name='log_sfr', format='D', array=np.array(sampler_log_sfr))
	cols0.append(col)
	col = fits.Column(name='log_mw_age', format='D', array=np.array(sampler_log_mw_age))
	cols0.append(col)

	if duste_switch == 'duste':
		col = fits.Column(name='log_dustmass', format='D', array=np.array(sampler_logdustmass))
		cols0.append(col)

	if add_agn == 1:
		col = fits.Column(name='log_fagn_bol', format='D', array=np.array(sampler_log_fagn_bol))
		cols0.append(col)

	for bb in range(0,nbands):
		col = fits.Column(name=filters[bb], format='D', array=np.array(mod_fluxes[bb]))
		cols0.append(col)

	col = fits.Column(name='chi2', format='D', array=np.array(mod_chi2))
	cols0.append(col)
	col = fits.Column(name='prob', format='D', array=np.array(mod_prob))
	cols0.append(col)

	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)
	primary_hdu = fits.PrimaryHDU(header=hdr)

	hdul = fits.HDUList([primary_hdu, hdu])
	hdul.writeto(fits_name_out, overwrite=True)

"""
USAGE: mpirun -np [npros] python ./rdsps_cmod.py (1)name_filters_list (2)name_config (3)name_SED_txt (4)name_out_fits
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# configuration file
global config_data
config_file = str(sys.argv[2])
data = np.genfromtxt(temp_dir+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]
# get filters
global filters, nbands
name_filters = str(sys.argv[1])
filters = np.genfromtxt(temp_dir+name_filters, dtype=str)
nbands = len(filters)
# nebular emission
global add_neb_emission
add_neb_emission = int(config_data['add_neb_emission'])
# gas_logu
global gas_logu
gas_logu = float(config_data['gas_logu'])

# SFH form
global sfh_form
if int(config_data['sfh_form']) == 0:
	sfh_form = 'tau_sfh'
elif int(config_data['sfh_form']) == 1:
	sfh_form = 'delayed_tau_sfh'
elif int(config_data['sfh_form']) == 2:
	sfh_form = 'log_normal_sfh'
elif int(config_data['sfh_form']) == 3:
	sfh_form = 'gaussian_sfh'
elif int(config_data['sfh_form']) == 4:
	sfh_form = 'double_power_sfh'

# redshift
global gal_z
gal_z = float(config_data['gal_z'])

# input SED
global obs_fluxes, obs_flux_err
inputSED_txt = str(sys.argv[3])
data = np.loadtxt(temp_dir+inputSED_txt)
obs_fluxes = np.asarray(data[:,0])
obs_flux_err = np.asarray(data[:,1])

global imf
imf = int(config_data['imf_type'])

global duste_switch
if int(config_data['duste_switch']) == 0:
	duste_switch = 'noduste'
elif int(config_data['duste_switch']) == 1:
	duste_switch = 'duste'

global dust_ext_law
if int(config_data['dust_ext_law']) == 0:
	dust_ext_law = 'CF2000'
elif int(config_data['dust_ext_law']) == 1:
	dust_ext_law = 'Cal2000'

global add_igm_absorption,igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

# whether dust_index is set fix or not
global fix_dust_index, fix_dust_index_val
if float(config_data['pr_dust_index_min']) == float(config_data['pr_dust_index_max']):     # dust_index is set fixed
	fix_dust_index = 1
	fix_dust_index_val = float(config_data['pr_dust_index_min'])
elif float(config_data['pr_dust_index_min']) != float(config_data['pr_dust_index_max']):   # dust_index varies
	fix_dust_index = 0
	fix_dust_index_val = 0

global add_agn 
add_agn = int(config_data['add_agn'])

# likelihood form
global likelihood_form
likelihood_form = config_data['likelihood']

# degree of freedom of the student's t distribution -> only relevant if likelihood_form='student_t'
global dof
dof = float(config_data['dof'])

global gauss_likelihood_form
gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

# cosmology
global cosmo_str, H0, Om0
cosmo = int(config_data['cosmo'])
if cosmo==0: 
	cosmo_str = 'flat_LCDM' 
elif cosmo==1:
	cosmo_str = 'WMAP5'
elif cosmo==2:
	cosmo_str = 'WMAP7'
elif cosmo==3:
	cosmo_str = 'WMAP9'
elif cosmo==4:
	cosmo_str = 'Planck13'
elif cosmo==5:
	cosmo_str = 'Planck15'
#elif cosmo==6:
#	cosmo_str = 'Planck18'
else:
	print ("The cosmo input is not recognized!")
	sys.exit()
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])

global free_z, DL_Gpc
if gal_z<=0:
	free_z = 1
	DL_Gpc = 0
else:
	free_z = 0
	# luminosity distance
	if cosmo_str=='flat_LCDM':
		cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
		DL_Gpc0 = cosmo1.luminosity_distance(gal_z)      # in Mpc
	elif cosmo_str=='WMAP5':
		DL_Gpc0 = WMAP5.luminosity_distance(gal_z)
	elif cosmo_str=='WMAP7':
		DL_Gpc0 = WMAP7.luminosity_distance(gal_z)
	elif cosmo_str=='WMAP9':
		DL_Gpc0 = WMAP9.luminosity_distance(gal_z)
	elif cosmo_str=='Planck13':
		DL_Gpc0 = Planck13.luminosity_distance(gal_z)
	elif cosmo_str=='Planck15':
		DL_Gpc0 = Planck15.luminosity_distance(gal_z)
	#elif cosmo_str=='Planck18':
	#	DL_Gpc0 = Planck18.luminosity_distance(gl_z)
	DL_Gpc = DL_Gpc0.value/1.0e+3

# default parameter set
global def_params, def_params_val
def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2',
				'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

#def_params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,
#					'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,'dust_index':-99.0,'log_age':-99.0,
#					'log_alpha':-99.0, 'log_beta':-99.0, 'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0}

def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

if fix_dust_index == 1:
	def_params_val['dust_index'] = fix_dust_index_val
if fix_dust_index == 0:
	def_params_val['dust_index'] = -0.7
if free_z == 0:
	def_params_val['z'] = gal_z

global def_params_fsps, params_assoc_fsps, status_log
def_params_fsps = ['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 
				'log_qpah','log_fagn', 'log_tauagn']
params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

# call FSPS
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
if duste_switch == 'duste':
	sp.params["add_dust_emission"] = True
elif duste_switch == 'noduste':
	sp.params["add_dust_emission"] = False
if add_neb_emission == 1:
	sp.params["add_neb_emission"] = True
	sp.params['gas_logu'] = gas_logu
elif add_neb_emission == 0:
	sp.params["add_neb_emission"] = False
if add_agn == 0:
	sp.params["fagn"] = 0
elif add_agn == 1:
	sp.params["fagn"] = 1

if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh':
	if sfh_form == 'tau_sfh':
		sp.params["sfh"] = 1
	elif sfh_form == 'delayed_tau_sfh':
		sp.params["sfh"] = 4
	sp.params["const"] = 0
	sp.params["sf_start"] = 0
	sp.params["sf_trunc"] = 0
	sp.params["fburst"] = 0
	sp.params["tburst"] = 30.0
	if dust_ext_law == 'CF2000' :
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_ext_law == 'Cal2000':
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh':
	sp.params["sfh"] = 3
	if dust_ext_law == 'CF2000' :
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_ext_law == 'Cal2000':
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0

# define number of parameters
global params, nparams
if free_z == 0:
	if sfh_form == 'tau_sfh' or sfh_form == 'delayed_tau_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# fix-z, mainSFH, noduste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_age','dust2']
					nparams = len(params)
				elif add_agn == 1:
					# fix-z, mainSFH, noduste, Cal2000, AGN
					params = ['logzsol','log_tau','log_age','dust2','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# fix-z, mainSFH, duste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah']
					nparams = len(params)
				elif add_agn == 1:
					# fix-z, mainSFH, duste, Cal2000, AGN
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, duste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, duste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)

	elif sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# fix-z, otherSFH, noduste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2']
					nparams = len(params)
				elif add_agn == 1:
					# fix-z, otherSFH, noduste, Cal2000, AGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# fix-z, mainSFH, duste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah']
					nparams = len(params)
				elif add_agn == 1:
					# fix-z, mainSFH, duste, Cal2000, AGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, duste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, duste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)

	elif sfh_form == 'double_power_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# fix-z, mainSFH, noduste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2']
					nparams = len(params)
				elif add_agn == 1:
					# fix-z, mainSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# fix-z, mainSFH, duste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah']
					nparams = len(params)
				elif add_agn == 1:
					# fix-z, mainSFH, duste, Cal2000, AGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, duste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						# fix-z, mainSFH, duste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)

elif free_z == 1:
	if sfh_form == 'tau_sfh' or sfh_form == 'delayed_tau_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# free-z, mainSFH, noduste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_age','dust2','z']
					nparams = len(params)
				elif add_agn == 1:
					# free-z, mainSFH, noduste, Cal2000, AGN
					params = ['logzsol','log_tau','log_age','dust2','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# free-z, mainSFH, noduste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, noduste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# free-z, mainSFH, noduste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, noduste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# free-z, mainSFH, duste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah','z']
					nparams = len(params)
				elif add_agn == 1:
					# free-z, mainSFH, duste, Cal2000, AGN
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# free-z, mainSFH, duste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, duste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# free-z, mainSFH, duste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, duste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)

	elif sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# free-z, otherSFH, noduste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','z']
					nparams = len(params)
				elif add_agn == 1:
					# free-z, otherSFH, noduste, Cal2000, AGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# free-z, mainSFH, noduste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, noduste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# free-z, mainSFH, noduste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, noduste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# free-z, mainSFH, duste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah','z']
					nparams = len(params)
				elif add_agn == 1:
					# free-z, mainSFH, duste, Cal2000, AGN
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# free-z, mainSFH, duste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, duste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# free-z, mainSFH, duste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, duste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']       ### 14 parameters
						nparams = len(params)

	elif sfh_form == 'double_power_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# free-z, mainSFH, noduste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','z']
					nparams = len(params)
				elif add_agn == 1:
					# free-z, mainSFH, noduste, Cal2000, AGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# free-z, mainSFH, noduste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, noduste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# free-z, mainSFH, noduste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, noduste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					# free-z, mainSFH, duste, Cal2000, noAGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah','z']
					nparams = len(params)
				elif add_agn == 1:
					# free-z, mainSFH, duste, Cal2000, AGN
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						# free-z, mainSFH, duste, CF2000, fix dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, duste, CF2000, fix dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						# free-z, mainSFH, duste, CF2000, vary dust_index, noAGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						# free-z, mainSFH, duste, CF2000, vary dust_index, AGN
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)

if rank == 0:
	print ("parameters set: ")
	print (params)
	print ("number of parameters: %d" % nparams)

global params_fsps, nparams_fsps
params_fsps = []
for ii in range(0,len(def_params_fsps)):
	for jj in range(0,nparams):
		if def_params_fsps[ii] == params[jj]:
			params_fsps.append(params[jj])
nparams_fsps = len(params_fsps)

# priors ranges for the parameters: including normalization
global priors_min, priors_max
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for ii in range(0,nparams):      
	str_temp = 'pr_%s_min' % params[ii]
	priors_min[ii] = float(config_data[str_temp])
	str_temp = 'pr_%s_max' % params[ii]
	priors_max[ii] = float(config_data[str_temp])

# if redshift is fix: free_z=0, 
# match wavelength points of transmission curves with that of the spectrum at a given redshift
global trans_fltr_int
if free_z == 0:
	params_val = def_params_val
	for pp in range(0,nparams-1):   				# log_mass is excluded
		params_val[params[pp]] = priors_min[pp]

	spec_SED = generate_modelSED_spec_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,add_igm_absorption=add_igm_absorption,
		igm_type=igm_type,params_fsps=params_fsps, params_val=params_val,DL_Gpc=DL_Gpc)

	nwave = len(spec_SED['wave'])
	if rank == 0:
		trans_fltr_int = match_filters_array(spec_SED['wave'],filters)
	elif rank>0:
		trans_fltr_int = np.zeros((nbands,nwave))

	comm.Bcast(trans_fltr_int, root=0)
elif free_z == 1:
	trans_fltr_int = None
 
global npmod_seds
nrandmod = int(config_data['nrandmod'])
if nrandmod == 0:
	npmod_seds0 = nparams*10000
elif nrandmod >= 0:
	npmod_seds0 = nrandmod
npmod_seds = int(npmod_seds0/size)*size

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
	store_to_fits(nsamples=nsamples,sampler_params=sampler_params,sampler_log_mass=sampler_log_mass,sampler_log_sfr=sampler_log_sfr,
				sampler_log_mw_age=sampler_log_mw_age,sampler_logdustmass=sampler_logdustmass,sampler_log_fagn_bol=sampler_log_fagn_bol, 
				mod_fluxes=mod_fluxes,mod_chi2=mod_chi2,mod_prob=mod_prob,cosmo_str=cosmo_str,H0=H0,Om0=Om0,fits_name_out=fits_name_out)


