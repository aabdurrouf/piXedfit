import numpy as np
from math import log10, pow, sqrt 
import sys, os
import h5py
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from astropy.cosmology import *
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
from scipy.stats import norm as normal
from scipy.stats import t, gamma

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

from piXedfit.utils.posteriors import model_leastnorm, ln_gauss_prob, ln_student_t_prob
from piXedfit.utils.filtering import interp_filters_curves, filtering_interp_filters, cwave_filters 
from piXedfit.utils.redshifting import cosmo_redshifting
from piXedfit.utils.igm_absorption import igm_att_madau, igm_att_inoue
from piXedfit.piXedfit_spectrophotometric import spec_smoothing, match_spectra_poly_legendre_fit
from piXedfit.piXedfit_model import get_no_nebem_wave_fit, generate_modelSED_spec


def bayesian_sedfit_gauss():
	f = h5py.File(models_spec, 'r')

	# get spectral wavelength
	wave = f['mod/spec/wave'][:]

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_spec_flux_temp = np.zeros((nwaves_clean,numDataPerRank))
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get spectral fluxes
		str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][int(ii)]
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type == 0:
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type == 1:
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes

		chi_photo = (norm_fluxes-obs_fluxes)/obs_flux_err
		chi2_photo = np.sum(np.square(chi_photo))
		
		# cut and normalize model spectrum
		# smoothing model spectrum to meet resolution of IFS
		conv_mod_spec_wave,conv_mod_spec_flux = spec_smoothing(redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]*norm,spec_sigma)

		# get model continuum
		func = interp1d(conv_mod_spec_wave,conv_mod_spec_flux)
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# get ratio of the obs and mod continuum
		spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

		# fit with legendre polynomial
		poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

		# apply to the model
		conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

		chi2 = chi2_photo + chi2_spec

		idx1 = np.where((chi_spec0>=lower) & (chi_spec0<=upper))
		m_merge = conv_mod_spec_flux_clean[idx1[0]].tolist() + norm_fluxes.tolist()
		d_merge = spec_flux_clean[idx1[0]].tolist() + obs_fluxes.tolist()
		derr_merge = spec_flux_err_clean[idx1[0]].tolist() + obs_flux_err.tolist()
		lnlikeli = ln_gauss_prob(d_merge,derr_merge,m_merge)

		mod_chi2_temp[int(count)] = chi2
		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)
		mod_prob_temp[int(count)] = lnlikeli
		mod_fluxes_temp[:,int(count)] = fluxes  # before normalized
		mod_spec_flux_temp[:,int(count)] = conv_mod_spec_flux_clean/norm # before normalized

		# get parameters
		for pp in range(0,nparams):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()

	mod_params = np.zeros((nparams,nmodels))
	mod_fluxes = np.zeros((nbands,nmodels))
	mod_spec_flux = np.zeros((nwaves_clean,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)
	mod_prob = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)

	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)
	for bb in range(0,nwaves_clean):
		comm.Gather(mod_spec_flux_temp[bb], mod_spec_flux[bb], root=0)

	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)
	modif_spec_flux_err_clean = np.zeros(nwaves_clean)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2_photo), key=itemgetter(1))
		fluxes = mod_fluxes[:,idx0]
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)

		if mod_chi2_photo[idx0]/nbands > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = np.sum(np.square(((norm*fluxes) -obs_fluxes)/modif_obs_flux_err))
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			status_add_err[0] = 1
		elif mod_chi2_photo[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

		if mod_redcd_chi2_spec[idx0] > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_spec_flux_err_clean = np.sqrt(np.square(spec_flux_err_clean) + np.square(sys_err_frac*spec_flux_clean))
				chi_spec0 = ((norm*mod_spec_flux[:,idx0])-spec_flux_clean)/modif_spec_flux_err_clean
				chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
				chi2_spec = np.sum(np.square(chi_spec))
				if chi2_spec/len(chi_spec) <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			status_add_err[0] = 1
		elif mod_redcd_chi2_spec[idx0] <= redcd_chi2:
			status_add_err[0] = 0

	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		# Broadcast
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_chi2_photo, root=0)
		comm.Bcast(mod_redcd_chi2_spec, root=0)
		comm.Bcast(mod_prob, root=0)
		 
	elif status_add_err[0] == 1:
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_spec_flux, root=0)
		comm.Bcast(modif_obs_flux_err, root=0) 
		comm.Bcast(modif_spec_flux_err_clean, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_chi2_photo_temp = np.zeros(numDataPerRank)
		mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)

		mod_prob_temp = np.zeros(numDataPerRank)
		mod_params_temp = np.zeros((nparams,numDataPerRank))

		count = 0
		for ii in recvbuf_idx:
			if modif_obs_flux_err[0]>0:
				norm = model_leastnorm(obs_fluxes,modif_obs_flux_err,mod_fluxes[:,int(count)])
			else:
				norm = model_leastnorm(obs_fluxes,obs_flux_err,mod_fluxes[:,int(count)])
			norm_fluxes = norm*mod_fluxes[:,int(count)]
			chi_photo = (norm_fluxes-obs_fluxes)/obs_flux_err
			chi2_photo = np.sum(np.square(chi_photo))

			if modif_spec_flux_err_clean[0]>0:
				chi_spec0 = ((norm*mod_spec_flux[:,int(count)])-spec_flux_clean)/modif_spec_flux_err_clean
			else:
				chi_spec0 = ((norm*mod_spec_flux[:,int(count)])-spec_flux_clean)/spec_flux_err_clean
			chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
			chi2_spec = np.sum(np.square(chi_spec))

			chi2 = chi2_photo + chi2_spec

			idx1 = np.where((chi_spec0>=lower) & (chi_spec0<=upper))
			m_merge = mod_spec_flux[:,int(count)][idx1[0]].tolist() + norm_fluxes.tolist()
			d_merge = spec_flux_clean[idx1[0]].tolist() + obs_fluxes.tolist()
			derr_merge = spec_flux_err_clean[idx1[0]].tolist() + obs_flux_err.tolist()
			lnlikeli = ln_gauss_prob(d_merge,derr_merge,m_merge)

			mod_chi2_temp[int(count)] = chi2
			mod_chi2_photo_temp[int(count)] = chi2_photo
			mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)
			mod_prob_temp[int(count)] = lnlikeli

			# get parameters
			for pp in range(0,nparams):
				str_temp = 'mod/par/%s' % params[pp]
				if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
					mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
				else:
					mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

			count = count + 1

		mod_prob = np.zeros(nmodels)
		mod_chi2 = np.zeros(nmodels)
		mod_chi2_photo = np.zeros(nmodels)
		mod_redcd_chi2_spec = np.zeros(nmodels)
		mod_params = np.zeros((nparams,nmodels))
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
		comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)

		comm.Gather(mod_prob_temp, mod_prob, root=0)
		for pp in range(0,nparams):
			comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

		# Broadcast
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_chi2_photo, root=0)
		comm.Bcast(mod_redcd_chi2_spec, root=0)
		comm.Bcast(mod_prob, root=0)
		comm.Bcast(mod_params, root=0)

		## end of status_add_err[0] == 1

	f.close()

	return mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob, modif_obs_flux_err, modif_spec_flux_err_clean


def bayesian_sedfit_student_t():
	f = h5py.File(models_spec, 'r')

	# get spectral wavelength
	wave = f['mod/spec/wave'][:]

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_spec_flux_temp = np.zeros((nwaves_clean,numDataPerRank))
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get spectral fluxes
		str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][int(ii)]
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type == 0:
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type == 1:
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)

		chi_photo = ((norm*fluxes)-obs_fluxes)/obs_flux_err
		chi2_photo = np.sum(np.square(chi_photo))
		
		# cut and normalize model spectrum
		# smoothing model spectrum to meet resolution of IFS
		conv_mod_spec_wave,conv_mod_spec_flux = spec_smoothing(redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]*norm,spec_sigma)

		# get model continuum
		func = interp1d(conv_mod_spec_wave,conv_mod_spec_flux)
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# get ratio of the obs and mod continuum
		spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

		# fit with legendre polynomial
		poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

		# apply to the model
		conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

		# total chi-square
		chi2 = chi2_photo + chi2_spec

		# probability
		chi_merge = chi_photo.tolist() + chi_spec.tolist()
		chi_merge = np.asarray(chi_merge)
		lnlikeli = ln_student_t_prob(dof,chi_merge)

		mod_chi2_temp[int(count)] = chi2
		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)
		mod_prob_temp[int(count)] = lnlikeli
		mod_fluxes_temp[:,int(count)] = fluxes  # before normalized
		mod_spec_flux_temp[:,int(count)] = conv_mod_spec_flux_clean/norm # before normalized

		# get parameters
		for pp in range(0,nparams):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()

	mod_params = np.zeros((nparams,nmodels))
	mod_fluxes = np.zeros((nbands,nmodels))
	mod_spec_flux = np.zeros((nwaves_clean,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)
	mod_prob = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)

	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)
	for bb in range(0,nwaves_clean):
		comm.Gather(mod_spec_flux_temp[bb], mod_spec_flux[bb], root=0)

	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)
	modif_spec_flux_err_clean = np.zeros(nwaves_clean)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2_photo), key=itemgetter(1))
		fluxes = mod_fluxes[:,idx0]
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)

		if mod_chi2_photo[idx0]/nbands > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = np.sum(np.square(((norm*fluxes)-obs_fluxes)/modif_obs_flux_err))
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			status_add_err[0] = 1
		elif mod_chi2_photo[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

		if mod_redcd_chi2_spec[idx0] > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_spec_flux_err_clean = np.sqrt(np.square(spec_flux_err_clean) + np.square(sys_err_frac*spec_flux_clean))
				chi_spec0 = ((norm*mod_spec_flux[:,idx0])-spec_flux_clean)/modif_spec_flux_err_clean
				chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
				chi2_spec = np.sum(np.square(chi_spec))
				if chi2_spec/len(chi_spec) <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			status_add_err[0] = 1
		elif mod_redcd_chi2_spec[idx0] <= redcd_chi2:
			status_add_err[0] = 0

	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		# Broadcast
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_chi2_photo, root=0)
		comm.Bcast(mod_redcd_chi2_spec, root=0)
		comm.Bcast(mod_prob, root=0)
		 
	elif status_add_err[0] == 1:
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_spec_flux, root=0)
		comm.Bcast(modif_obs_flux_err, root=0) 
		comm.Bcast(modif_spec_flux_err_clean, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_chi2_photo_temp = np.zeros(numDataPerRank)
		mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)

		mod_prob_temp = np.zeros(numDataPerRank)
		mod_params_temp = np.zeros((nparams,numDataPerRank))

		count = 0
		for ii in recvbuf_idx:
			if modif_obs_flux_err[0]>0:
				norm = model_leastnorm(obs_fluxes,modif_obs_flux_err,mod_fluxes[:,int(count)])
			else:
				norm = model_leastnorm(obs_fluxes,obs_flux_err,mod_fluxes[:,int(count)])
			chi_photo = ((norm*mod_fluxes[:,int(count)])-obs_fluxes)/obs_flux_err
			chi2_photo = np.sum(np.square(chi_photo))

			if modif_spec_flux_err_clean[0]>0:
				chi_spec0 = ((norm*mod_spec_flux[:,int(count)])-spec_flux_clean)/modif_spec_flux_err_clean
			else:
				chi_spec0 = ((norm*mod_spec_flux[:,int(count)])-spec_flux_clean)/spec_flux_err_clean
			chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
			chi2_spec = np.sum(np.square(chi_spec))

			chi2 = chi2_photo + chi2_spec
			
			# probability
			chi_merge = chi_photo.tolist() + chi_spec.tolist()
			chi_merge = np.asarray(chi_merge)
			lnlikeli = ln_student_t_prob(dof,chi_merge)

			mod_chi2_temp[int(count)] = chi2
			mod_chi2_photo_temp[int(count)] = chi2_photo
			mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)
			mod_prob_temp[int(count)] = lnlikeli

			# get parameters
			for pp in range(0,nparams):
				str_temp = 'mod/par/%s' % params[pp]
				if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
					mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
				else:
					mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

			count = count + 1

		mod_prob = np.zeros(nmodels)
		mod_chi2 = np.zeros(nmodels)
		mod_chi2_photo = np.zeros(nmodels)
		mod_redcd_chi2_spec = np.zeros(nmodels)
		mod_params = np.zeros((nparams,nmodels))
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
		comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)

		comm.Gather(mod_prob_temp, mod_prob, root=0)
		for pp in range(0,nparams):
			comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

		# Broadcast
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_chi2_photo, root=0)
		comm.Bcast(mod_redcd_chi2_spec, root=0)
		comm.Bcast(mod_prob, root=0)
		comm.Bcast(mod_params, root=0)

		## end of status_add_err[0] == 1

	f.close()
	
	return mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob, modif_obs_flux_err, modif_spec_flux_err_clean


def store_to_fits(sampler_params,mod_chi2,mod_chi2_photo,mod_redcd_chi2_spec,mod_prob,fits_name_out):

	# get best-fit model
	idx, min_val = min(enumerate(mod_chi2), key=itemgetter(1))
	bfit_chi2 = mod_chi2[idx]
	bfit_rchi2_photo = mod_chi2_photo[idx]/nbands
	bfit_rchi2_spec = mod_redcd_chi2_spec[idx]

	f = h5py.File(models_spec, 'r')
	# best-fit SED
	wave = f['mod/spec/wave'][:]
	str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][idx]
	extnc_spec = f[str_temp][:]
	f.close()
	# redshifting
	redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)
	# IGM absorption
	if add_igm_absorption == 1:
		if igm_type == 0:
			trans = igm_att_madau(redsh_wave,gal_z)
			redsh_spec = redsh_spec*trans
		elif igm_type == 1:
			trans = igm_att_inoue(redsh_wave,gal_z)
			redsh_spec = redsh_spec*trans
	# filtering
	fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
	# calculate normalization
	norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
	bfit_photo_fluxes = norm*fluxes

	# generate model spectrum free of emission lines
	params_val = def_params_val
	for pp in range(0,nparams):
		params_val[params[pp]] = sampler_params[params[pp]][idx]
	params_val['z'] = gal_z
	spec_SED = generate_modelSED_spec(imf_type=imf,duste_switch=duste_switch,add_neb_emission=0,
						dust_law=dust_law,sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
						igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	redsh_wave1,redsh_spec1 = spec_SED['wave'], spec_SED['flux']
		
	# cut and normalize model spectrum
	# smoothing model spectrum to meet resolution of IFS
	conv_mod_spec_wave,conv_mod_spec_flux = spec_smoothing(redsh_wave1[idx_mod_wave[0]],redsh_spec1[idx_mod_wave[0]]*norm,spec_sigma)

	# get model continuum
	func = interp1d(conv_mod_spec_wave,conv_mod_spec_flux)
	conv_mod_spec_flux_clean = func(spec_wave_clean)

	# get ratio of the obs and mod continuum
	spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

	# fit with legendre polynomial
	poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

	# apply to the model
	conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean

	# remove bad spectral points
	chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
	chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)

	# get best-fit model spectrum and polynomial corrrection factor
	corr_factor = poly_legendre(redsh_wave1[idx_mod_wave[0]])
	bfit_spec_wave = conv_mod_spec_wave
	bfit_spec_nwaves = len(bfit_spec_wave)
	bfit_spec_flux = corr_factor*conv_mod_spec_flux

	#==> Get median likelihood parameters
	# get initial prediction for stellar mass
	idx_sel = np.where((np.isnan(mod_prob)==False) & (np.isinf(mod_prob)==False))
	array_lnprob = mod_prob[idx_sel[0]] - max(mod_prob[idx_sel[0]])  # normalize
	array_prob = np.exp(array_lnprob)
	array_prob = array_prob/np.sum(array_prob)						 # normalize
	tot_prob = np.sum(array_prob)
	array_val = sampler_params['log_mass'][idx_sel[0]]
	mean_lmass = np.sum(array_val*array_prob)/tot_prob
	mean_lmass2 = np.sum(np.square(array_val)*array_prob)/tot_prob
	std_lmass = sqrt(abs(mean_lmass2 - (mean_lmass**2)))

	# add more if parameters are joint with mass
	if len(params_prior_jtmass)>0:
		for pp in range(0,len(params_prior_jtmass)):
			loc = np.interp(mean_lmass,params_priors[params_prior_jtmass[pp]]['lmass'],params_priors[params_prior_jtmass[pp]]['pval'])
			scale = params_priors[params_prior_jtmass[pp]]['scale']
			mod_prob += np.log(normal.pdf(sampler_params[params_prior_jtmass[pp]],loc=loc,scale=scale))

	for pp in range(0,nparams):
		if params_priors[params[pp]]['form'] == 'gaussian':
			mod_prob += np.log(normal.pdf(sampler_params[params[pp]],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
		elif params_priors[params[pp]]['form'] == 'studentt':
			mod_prob += np.log(t.pdf(sampler_params[params[pp]],params_priors[params[pp]]['df'],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
		elif params_priors[params[pp]]['form'] == 'gamma':
			mod_prob += np.log(gamma.pdf(sampler_params[params[pp]],params_priors[params[pp]]['a'],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
		elif params_priors[params[pp]]['form'] == 'arbitrary':
			mod_prob += np.log(np.interp(sampler_params[params[pp]],params_priors[params[pp]]['values'],params_priors[params[pp]]['prob']))

	crit_chi2 = np.percentile(mod_chi2[np.logical_not(np.isnan(mod_chi2))], perc_chi2)
	idx_sel = np.where((mod_chi2<=crit_chi2) & (np.isnan(mod_prob)==False) & (np.isinf(mod_prob)==False))

	array_lnprob = mod_prob[idx_sel[0]] - max(mod_prob[idx_sel[0]])  # normalize
	array_prob = np.exp(array_lnprob)
	array_prob = array_prob/np.sum(array_prob)						 # normalize
	tot_prob = np.sum(array_prob)

	params_bfits = np.zeros((nparams,2))
	for pp in range(0,nparams):
		if params[pp] == 'log_mass':
			params_bfits[pp][0] = mean_lmass
			params_bfits[pp][1] = std_lmass
		else:
			array_val = sampler_params[params[pp]][idx_sel[0]]
			mean_val = np.sum(array_val*array_prob)/tot_prob
			mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
			std_val = sqrt(abs(mean_val2 - (mean_val**2)))
			params_bfits[pp][0] = mean_val
			params_bfits[pp][1] = std_val

	# store the result to a FITS file
	hdr = fits.Header()
	hdr['imf'] = imf
	hdr['nparams'] = nparams
	hdr['sfh_form'] = sfh_form
	hdr['dust_law'] = dust_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
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
	hdr['gal_z'] = gal_z
	hdr['free_z'] = 0
	hdr['cosmo'] = cosmo
	hdr['H0'] = H0
	hdr['Om0'] = Om0
	hdr['redcd_chi2'] = bfit_chi2/(nbands+len(chi_spec))
	hdr['redcd_chi2_spec'] = bfit_rchi2_spec
	hdr['redcd_chi2_photo'] = bfit_rchi2_photo
	hdr['perc_chi2'] = perc_chi2
	for pp in range(0,nparams):
		hdr['param%d' % pp] = params[pp]
	hdr['fitmethod'] = 'rdsps'
	hdr['storesamp'] = 0
	hdr['specphot'] = 1
	primary_hdu = fits.PrimaryHDU(header=hdr)

	#=> parameters inferred from SED fitting with RDSPS method
	cols0 = []
	col = fits.Column(name='rows', format='4A', array=['mean','std'])
	cols0.append(col)
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array([params_bfits[pp][0],params_bfits[pp][1]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu1 = fits.BinTableHDU.from_columns(cols, name='fit_params')

	#==> add table for parameters that have minimum chi-square
	cols0 = []
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array([sampler_params[params[pp]][idx]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu2 = fits.BinTableHDU.from_columns(cols, name='minchi2_params')

	#==> observed photometric SED
	cols0 = []
	col = fits.Column(name='flux', format='D', array=np.array(obs_fluxes))
	cols0.append(col)
	col = fits.Column(name='flux_err', format='D', array=np.array(obs_flux_err))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu3 = fits.BinTableHDU.from_columns(cols, name='obs_photo')

	#==> observed spectrum
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(spec_wave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(spec_flux))
	cols0.append(col)
	col = fits.Column(name='flux_err', format='D', array=np.array(spec_flux_err))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu4 = fits.BinTableHDU.from_columns(cols, name='obs_spec')

	#==> best-fit photometric SED
	photo_cwave = cwave_filters(filters)
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(photo_cwave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(bfit_photo_fluxes))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu5 = fits.BinTableHDU.from_columns(cols, name='bfit_photo')
	
	#==> best-fit spectrum
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(bfit_spec_wave[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(bfit_spec_flux[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu6 = fits.BinTableHDU.from_columns(cols, name='bfit_spec')

	#==> correction factor
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(bfit_spec_wave[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	col = fits.Column(name='corr_factor', format='D', array=np.array(corr_factor[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu7 = fits.BinTableHDU.from_columns(cols, name='corr_factor')

	#==> best-fit model spectrum to the observed photometric SED
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(redsh_wave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(redsh_spec*norm))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu8 = fits.BinTableHDU.from_columns(cols, name='bfit_mod_spec')

	hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8])
	hdul.writeto(fits_name_out, overwrite=True)

"""
USAGE: mpirun -np [npros] python ./rdsps_pcmod.py (1)name_filters_list (2)name_config (3)name_SED_txt (4)name_out_fits
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank()

global def_params_val
def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,
				'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,
				'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0} 

# configuration file
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

global likelihood_form
likelihood_form = config_data['likelihood']

# degree of freedom in the student's t likelihood function, only relevant if likelihood_form='student_t'
global dof
dof = float(config_data['dof'])

global perc_chi2
perc_chi2 = float(config_data['perc_chi2'])

# cosmology
global cosmo, H0, Om0
cosmo = int(config_data['cosmo'])
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])

# redshift
global gal_z, DL_Gpc
gal_z = float(config_data['gal_z'])
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

# input SED
global obs_fluxes, obs_flux_err, spec_wave, spec_flux, spec_flux_err
inputSED_file = str(sys.argv[3])
f = h5py.File(temp_dir+inputSED_file, 'r')
obs_fluxes = f['obs_flux'][:]
obs_flux_err = f['obs_flux_err'][:]
spec_wave = f['spec_wave'][:]
spec_flux = f['spec_flux'][:]
spec_flux_err = f['spec_flux_err'][:]
f.close()

# remove bad spectral fluxes
idx0 = np.where((np.isnan(spec_flux)==False) & (np.isnan(spec_flux_err)==False) & (spec_flux>0) & (spec_flux_err>0))
spec_wave = spec_wave[idx0[0]]
spec_flux = spec_flux[idx0[0]]
spec_flux_err = spec_flux_err[idx0[0]]

# wavelength range of the observed spectrum
nwaves = len(spec_wave)
min_spec_wave = min(spec_wave)
max_spec_wave = max(spec_wave)

# spectral resolution
global spec_sigma
spec_sigma = float(config_data['spec_sigma'])

# order of the Legendre polynomial
global poly_order
poly_order = int(config_data['poly_order'])

# add systematic error accommodating various factors, including modeling uncertainty, assume systematic error of 0.1
sys_err_frac = 0.1
obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
spec_flux_err = np.sqrt(np.square(spec_flux_err) + np.square(sys_err_frac*spec_flux))

# get wavelength free of emission lines
global del_wave_nebem, spec_wave_clean, spec_flux_clean, spec_flux_err_clean, nwaves_clean
del_wave_nebem = float(config_data['del_wave_nebem'])
spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
spec_flux_clean = np.delete(spec_flux, waveid_excld)
spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)
nwaves_clean = len(spec_wave_clean)

# clipping for bad spectral points in the chi-square calculation
global spec_chi_sigma_clip
spec_chi_sigma_clip = float(config_data['spec_chi_sigma_clip'])

# HDF5 file containing pre-calculated model SEDs
global models_spec
models_spec = config_data['models_spec']

# data of pre-calculated model SEDs
f = h5py.File(models_spec, 'r')

# get list of parameters
global nparams, params
nparams = int(f['mod'].attrs['nparams_all'])  # include all possible parameters
params = []
for pp in range(0,nparams):
	str_temp = 'par%d' % pp
	attrs = f['mod'].attrs[str_temp]
	if isinstance(attrs, str) == False:
		attrs = attrs.decode()
	params.append(attrs)

if rank==0:
	print ("Number of parameters: %d" % nparams)
	print ("List of parameters: ")
	print (params)

# number of model SEDs
global nmodels_parent
nmodels_parent = int(f['mod'].attrs['nmodels'])

# get the prior ranges of the parameters.
# for RDSPS fitting, a fix parameters can't be declared when the fitting run
# it must be applied when building the model rest-frame spectral templates
global priors_min, priors_max
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for pp in range(0,nparams):
	str_temp1 = "pr_%s_min" % params[pp]
	if str_temp1 in config_data:
		str_temp2 = "pr_%s_max" % params[pp]
		priors_min[pp] = float(config_data[str_temp1])
		priors_max[pp] = float(config_data[str_temp2])
	else:
		str_temp = 'mod/par/%s' % params[pp]
		priors_min[pp] = min(f[str_temp][:])
		priors_max[pp] = max(f[str_temp][:])

# get model indexes satisfying the preferred ranges
status_idx = np.zeros(nmodels_parent)
for pp in range(0,nparams):						
	str_temp = 'mod/par/%s' % params[pp]
	idx0 = np.where((f[str_temp][:]>=priors_min[pp]) & (f[str_temp][:]<=priors_max[pp]))
	status_idx[idx0[0]] = status_idx[idx0[0]] + 1

global idx_parmod_sel, nmodels
idx_parmod_sel = np.where(status_idx==nparams)	
nmodels = int(len(idx_parmod_sel[0])/size)*size
idx_parmod_sel = idx_parmod_sel[0:int(nmodels)]

if rank == 0:
	print ("Number of parent models in models_spec: %d" % nmodels_parent)
	print ("Number of models to be used for fitting: %d" % nmodels)

# modeling configurations
global imf, sfh_form, dust_law, duste_switch, add_neb_emission, add_agn, gas_logu
imf = f['mod'].attrs['imf_type']
sfh_form = f['mod'].attrs['sfh_form']
dust_law = f['mod'].attrs['dust_law']
duste_switch = f['mod'].attrs['duste_switch']
add_neb_emission = f['mod'].attrs['add_neb_emission']
add_agn = f['mod'].attrs['add_agn']
gas_logu = f['mod'].attrs['gas_logu']

# cut model spectrum to match range given by the IFS spectra
global idx_mod_wave, nwaves_model, redsh_mod_wave
rest_wave = f['mod/spec/wave'][:]
redsh_mod_wave0 = (1.0+gal_z)*rest_wave
idx_mod_wave = np.where((redsh_mod_wave0>min_spec_wave-30) & (redsh_mod_wave0<max_spec_wave+30))
nwaves_model = len(idx_mod_wave[0])
redsh_mod_wave = redsh_mod_wave0[idx_mod_wave[0]]
f.close()

# get preferred priors
nparams_in_prior = int(config_data['pr_nparams'])
params_in_prior = []
for pp in range(0,nparams_in_prior):
	params_in_prior.append(config_data['pr_param%d' % pp])

global params_priors, params_prior_jtmass
params_priors = {}
params_prior_jtmass = []
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
			params_prior_jtmass.append(params[pp])
	else:
		params_priors[params[pp]]['form'] = 'uniform'

# igm absorption
global add_igm_absorption,igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

global interp_filters_waves, interp_filters_trans
interp_filters_waves,interp_filters_trans = interp_filters_curves(filters)

# running the calculation
global redcd_chi2
redcd_chi2 = 2.0

# running the calculation
if likelihood_form == 'gauss':
	mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob, modif_obs_flux_err, modif_spec_flux_err_clean = bayesian_sedfit_gauss()
elif likelihood_form == 'student_t':
	mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob, modif_obs_flux_err, modif_spec_flux_err_clean = bayesian_sedfit_student_t()
else:
	print ("likelihood_form is not recognized!")
	sys.exit()

if np.sum(modif_obs_flux_err) != 0:
	obs_flux_err = modif_obs_flux_err

if np.sum(modif_spec_flux_err_clean) != 0:
	spec_flux_err_clean = modif_spec_flux_err_clean

# change the format to dictionary
sampler_params = {}
for pp in range(0,nparams):
	sampler_params[params[pp]] = mod_params[pp]

# store to fits file
if rank == 0:
	fits_name_out = str(sys.argv[4])
	store_to_fits(sampler_params,mod_chi2,mod_chi2_photo,mod_redcd_chi2_spec,mod_prob,fits_name_out)
	sys.stdout.write('\n')
