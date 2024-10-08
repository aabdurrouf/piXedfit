import numpy as np
import sys, os
import fsps
import h5py
from math import log10, pow, sqrt 
from mpi4py import MPI
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
from operator import itemgetter

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.piXedfit_model import calc_mw_age, get_dust_mass_mainSFH_fit, get_dust_mass_fagnbol_mainSFH_fit 
from piXedfit.piXedfit_model import get_dust_mass_othSFH_fit, get_sfr_dust_mass_othSFH_fit 
from piXedfit.piXedfit_model import get_sfr_dust_mass_fagnbol_othSFH_fit, construct_SFH
from piXedfit.piXedfit_model import get_no_nebem_wave_fit, generate_modelSED_spec_restframe_fit
from piXedfit.piXedfit_model import generate_modelSED_spec_decompose
from piXedfit.piXedfit_model import default_params_val, set_initial_fsps, get_params_fsps


def store_to_fits(nsamples=None,sampler_params=None,sampler_log_sfr=None,sampler_log_mw_age=None,
	sampler_logdustmass=None,sampler_log_fagn_bol=None,fits_name_out=None): 

	#==> get median best-fit model spectrophotometric SED
	nchains = 300
	idx_sel = np.where((sampler_log_sfr>-29.0) & (sampler_params['log_age']<1.2))
	nchains = int(nchains/size)*size
	numDataPerRank = int(nchains/size)
	idx_mpi = np.random.uniform(0,len(idx_sel[0])-1,nchains)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	# for best-fit native model spectrum
	bfit_spec_tot_temp = np.zeros((nwaves_spec,numDataPerRank))
	bfit_spec_stellar_temp = np.zeros((nwaves_spec,numDataPerRank))
	bfit_spec_duste_temp = np.zeros((nwaves_spec,numDataPerRank))
	bfit_spec_agn_temp = np.zeros((nwaves_spec,numDataPerRank))
	bfit_spec_nebe_temp = np.zeros((nwaves_spec,numDataPerRank))
	bfit_spec_nebe_cont_temp = np.zeros((nwaves_spec,numDataPerRank))
	bfit_spec_wave = np.zeros(nwaves_spec)

	bfit_mod_spec_tot_temp = np.zeros((nwaves_mod,numDataPerRank))
	bfit_mod_spec_stellar_temp = np.zeros((nwaves_mod,numDataPerRank))
	bfit_mod_spec_duste_temp = np.zeros((nwaves_mod,numDataPerRank))
	bfit_mod_spec_agn_temp = np.zeros((nwaves_mod,numDataPerRank))
	bfit_mod_spec_nebe_temp = np.zeros((nwaves_mod,numDataPerRank))
	bfit_mod_spec_nebe_cont_temp = np.zeros((nwaves_mod,numDataPerRank))
	bfit_mod_spec_wave = np.zeros(nwaves_mod)

	bfit_corr_factor_temp = np.zeros((nwaves_spec,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		params_val = def_params_val
		param_stat = 0
		for pp in range(0,nparams):
			val0 = sampler_params[params[pp]][int(idx_sel[0][int(ii)])]
			params_val[params[pp]] = val0
			if val0>=priors_min[pp] and val0<=priors_max[pp]:
				param_stat = param_stat + 1

		if param_stat == nparams:
			# get wavelength free of emission lines
			spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(params_val['z'],spec_wave,del_wave_nebem)
			spec_flux_clean = np.delete(spec_flux, waveid_excld)
			spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)

			spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val,imf=imf,duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_law=dust_law,add_agn=add_agn,
							add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,
							Om0=Om0,sfh_form=sfh_form,funit='erg/s/cm2/A')
			bfit_mod_spec_wave = spec_SED['wave']
			bfit_mod_spec_tot_temp[:,int(count)] = spec_SED['flux_total']
			bfit_mod_spec_stellar_temp[:,int(count)] = spec_SED['flux_stellar']
			if add_neb_emission == 1:
				bfit_mod_spec_nebe_temp[:,int(count)] = spec_SED['flux_nebe']
				bfit_mod_spec_nebe_cont_temp[:,int(count)] = spec_SED['flux_nebe_cont']
			if duste_switch==1:
				bfit_mod_spec_duste_temp[:,int(count)] = spec_SED['flux_duste']
			if add_agn == 1:
				bfit_mod_spec_agn_temp[:,int(count)] = spec_SED['flux_agn']

			idx_mod_wave = np.where((spec_SED['wave']>min_spec_wave-100) & (spec_SED['wave']<max_spec_wave+100))
			conv_mod_spec_wave, conv_mod_spec_flux = spec_SED['wave'][idx_mod_wave[0]], spec_SED['flux_total'][idx_mod_wave[0]]

			# get model continuum, excluding emission lines
			func = interp1d(conv_mod_spec_wave, conv_mod_spec_flux, fill_value='extrapolate')
			conv_mod_spec_flux_clean = func(spec_wave_clean)

			# calculate chi-square value
			chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
			chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
			mod_chi2_temp[int(count)] = np.sum(np.square(chi_spec))

			# get ratio
			spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

			# fit with legendre polynomial and get the correction factor
			poly_legendre1 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)
			corr_factor = poly_legendre1(spec_wave)
			bfit_corr_factor_temp[:,int(count)] = corr_factor

			bfit_spec_wave = spec_wave

			func = interp1d(spec_SED['wave'],spec_SED['flux_total'], fill_value='extrapolate')
			bfit_spec_tot_temp[:,int(count)] = corr_factor*func(bfit_spec_wave)

			func = interp1d(spec_SED['wave'],spec_SED['flux_stellar'], fill_value='extrapolate')
			bfit_spec_stellar_temp[:,int(count)] = corr_factor*func(bfit_spec_wave)

			if add_neb_emission == 1:
				func = interp1d(spec_SED['wave'],spec_SED['flux_nebe'], fill_value='extrapolate')
				bfit_spec_nebe_temp[:,int(count)] = corr_factor*func(bfit_spec_wave)

				func = interp1d(spec_SED['wave'],spec_SED['flux_nebe_cont'], fill_value='extrapolate')
				bfit_spec_nebe_cont_temp[:,int(count)] = corr_factor*func(bfit_spec_wave)

			if duste_switch==1:
				if max_spec_wave/(params_val['z']+1.0) > 1e+4:
					func = interp1d(spec_SED['wave'], spec_SED['flux_duste'], fill_value='extrapolate')
					bfit_spec_duste_temp[:,int(count)] = corr_factor*func(bfit_spec_wave)
				else:
					bfit_spec_duste_temp[:,int(count)] = spec_SED['flux_duste']
			if add_agn == 1:
				if max_spec_wave/(params_val['z']+1.0) > 1e+4:
					func = interp1d(spec_SED['wave'], spec_SED['flux_agn'], fill_value='extrapolate')
					bfit_spec_agn_temp[:,int(count)] = corr_factor*func(bfit_spec_wave)
				else:
					bfit_spec_agn_temp[:,int(count)] = spec_SED['flux_agn']

		count = count + 1

	bfit_spec_tot = np.zeros((nwaves_spec,nchains))
	bfit_spec_stellar = np.zeros((nwaves_spec,nchains))
	bfit_spec_duste = np.zeros((nwaves_spec,nchains))
	bfit_spec_agn = np.zeros((nwaves_spec,nchains))
	bfit_spec_nebe = np.zeros((nwaves_spec,nchains))
	bfit_spec_nebe_cont = np.zeros((nwaves_spec,nchains))

	bfit_mod_spec_tot = np.zeros((nwaves_mod,nchains))
	bfit_mod_spec_stellar = np.zeros((nwaves_mod,nchains))
	bfit_mod_spec_duste = np.zeros((nwaves_mod,nchains))
	bfit_mod_spec_agn = np.zeros((nwaves_mod,nchains))
	bfit_mod_spec_nebe = np.zeros((nwaves_mod,nchains))
	bfit_mod_spec_nebe_cont = np.zeros((nwaves_mod,nchains))

	bfit_corr_factor = np.zeros((nwaves_spec,nchains))
	mod_chi2 = np.zeros(nchains)

	for bb in range(0,nwaves_spec):
		comm.Gather(bfit_spec_tot_temp[bb], bfit_spec_tot[bb], root=0)
		comm.Gather(bfit_spec_stellar_temp[bb], bfit_spec_stellar[bb], root=0)
		comm.Gather(bfit_corr_factor_temp[bb], bfit_corr_factor[bb], root=0)

	if add_neb_emission == 1:
		for bb in range(0,nwaves_spec):
			comm.Gather(bfit_spec_nebe_temp[bb], bfit_spec_nebe[bb], root=0)
			comm.Gather(bfit_spec_nebe_cont_temp[bb], bfit_spec_nebe_cont[bb], root=0)

	if duste_switch == 1:
		for bb in range(0,nwaves_spec):
			comm.Gather(bfit_spec_duste_temp[bb], bfit_spec_duste[bb], root=0)

	if add_agn == 1:
		for bb in range(0,nwaves_spec):
			comm.Gather(bfit_spec_agn_temp[bb], bfit_spec_agn[bb], root=0)

	for bb in range(0,nwaves_mod):
		comm.Gather(bfit_mod_spec_tot_temp[bb], bfit_mod_spec_tot[bb], root=0)
		comm.Gather(bfit_mod_spec_stellar_temp[bb], bfit_mod_spec_stellar[bb], root=0)

	if add_neb_emission == 1:
		for bb in range(0,nwaves_mod):
			comm.Gather(bfit_mod_spec_nebe_temp[bb], bfit_mod_spec_nebe[bb], root=0)
			comm.Gather(bfit_mod_spec_nebe_cont_temp[bb], bfit_mod_spec_nebe_cont[bb], root=0)

	if duste_switch == 1:
		for bb in range(0,nwaves_mod):
			comm.Gather(bfit_mod_spec_duste_temp[bb], bfit_mod_spec_duste[bb], root=0)

	if add_agn == 1:
		for bb in range(0,nwaves_mod):
			comm.Gather(bfit_mod_spec_agn_temp[bb], bfit_mod_spec_agn[bb], root=0)

	comm.Gather(mod_chi2_temp, mod_chi2, root=0)

	if rank == 0:
		# get percentiles
		p16_corr_factor = np.percentile(bfit_corr_factor,16,axis=1)
		p50_corr_factor = np.percentile(bfit_corr_factor,50,axis=1)
		p84_corr_factor = np.percentile(bfit_corr_factor,84,axis=1)

		p16_spec_tot = np.percentile(bfit_spec_tot,16,axis=1)
		p50_spec_tot = np.percentile(bfit_spec_tot,50,axis=1)
		p84_spec_tot = np.percentile(bfit_spec_tot,84,axis=1)

		p16_spec_stellar = np.percentile(bfit_spec_stellar,16,axis=1)
		p50_spec_stellar = np.percentile(bfit_spec_stellar,50,axis=1)
		p84_spec_stellar = np.percentile(bfit_spec_stellar,84,axis=1)

		p16_mod_spec_tot = np.percentile(bfit_mod_spec_tot,16,axis=1)
		p50_mod_spec_tot = np.percentile(bfit_mod_spec_tot,50,axis=1)
		p84_mod_spec_tot = np.percentile(bfit_mod_spec_tot,84,axis=1)

		p16_mod_spec_stellar = np.percentile(bfit_mod_spec_stellar,16,axis=1)
		p50_mod_spec_stellar = np.percentile(bfit_mod_spec_stellar,50,axis=1)
		p84_mod_spec_stellar = np.percentile(bfit_mod_spec_stellar,84,axis=1)

		# get best-fit model spectrum from chi-square minimization
		idx_sel1 = np.where(mod_chi2>0) 
		idx0, min_val = min(enumerate(mod_chi2[idx_sel1[0]]), key=itemgetter(1))
		idx1 = idx_sel1[0][idx0]
		chimin_spec_tot = bfit_spec_tot[:,idx1]
		chimin_spec_stellar = bfit_spec_stellar[:,idx1]

		if add_neb_emission == 1:
			p16_spec_nebe = np.percentile(bfit_spec_nebe,16,axis=1)
			p50_spec_nebe = np.percentile(bfit_spec_nebe,50,axis=1)
			p84_spec_nebe = np.percentile(bfit_spec_nebe,84,axis=1)

			p16_spec_nebe_cont = np.percentile(bfit_spec_nebe_cont,16,axis=1)
			p50_spec_nebe_cont = np.percentile(bfit_spec_nebe_cont,50,axis=1)
			p84_spec_nebe_cont = np.percentile(bfit_spec_nebe_cont,84,axis=1)

			p16_mod_spec_nebe = np.percentile(bfit_mod_spec_nebe,16,axis=1)
			p50_mod_spec_nebe = np.percentile(bfit_mod_spec_nebe,50,axis=1)
			p84_mod_spec_nebe = np.percentile(bfit_mod_spec_nebe,84,axis=1)

			p16_mod_spec_nebe_cont = np.percentile(bfit_mod_spec_nebe_cont,16,axis=1)
			p50_mod_spec_nebe_cont = np.percentile(bfit_mod_spec_nebe_cont,50,axis=1)
			p84_mod_spec_nebe_cont = np.percentile(bfit_mod_spec_nebe_cont,84,axis=1)

			chimin_spec_nebe = bfit_spec_nebe[:,idx1]
			chimin_spec_nebe_cont = bfit_spec_nebe_cont[:,idx1]

		if duste_switch == 1:
			p16_spec_duste = np.percentile(bfit_spec_duste,16,axis=1)
			p50_spec_duste = np.percentile(bfit_spec_duste,50,axis=1)
			p84_spec_duste = np.percentile(bfit_spec_duste,84,axis=1)

			p16_mod_spec_duste = np.percentile(bfit_mod_spec_duste,16,axis=1)
			p50_mod_spec_duste = np.percentile(bfit_mod_spec_duste,50,axis=1)
			p84_mod_spec_duste = np.percentile(bfit_mod_spec_duste,84,axis=1)

			chimin_spec_duste = bfit_spec_duste[:,idx1]

		if add_agn == 1:
			p16_spec_agn = np.percentile(bfit_spec_agn,16,axis=1)
			p50_spec_agn = np.percentile(bfit_spec_agn,50,axis=1)
			p84_spec_agn = np.percentile(bfit_spec_agn,84,axis=1)

			p16_mod_spec_agn = np.percentile(bfit_mod_spec_agn,16,axis=1)
			p50_mod_spec_agn = np.percentile(bfit_mod_spec_agn,50,axis=1)
			p84_mod_spec_agn = np.percentile(bfit_mod_spec_agn,84,axis=1)

			chimin_spec_agn = bfit_spec_agn[:,idx1]

		# sampler_id:
		sampler_id = np.linspace(1, nsamples, nsamples)

		# make header
		hdr = fits.Header()
		hdr['imf'] = imf
		hdr['nparams'] = nparams
		hdr['sfh_form'] = sfh_form
		hdr['dust_law'] = dust_law
		hdr['duste_stat'] = duste_switch
		hdr['add_neb_emission'] = add_neb_emission
		hdr['add_agn'] = add_agn
		hdr['add_igm_absorption'] = add_igm_absorption
		hdr['free_gas_logz'] = free_gas_logz
		hdr['cosmo'] = cosmo_str
		hdr['H0'] = H0
		hdr['Om0'] = Om0
		hdr['smooth_velocity'] = smooth_velocity
		hdr['sigma_smooth'] = sigma_smooth
		hdr['smooth_lsf'] = smooth_lsf
		if smooth_lsf == 1:
			hdr['name_file_lsf'] = name_file_lsf
		hdr['poly_order'] = poly_order
		hdr['del_wave_nebem'] = del_wave_nebem
		hdr['spec_chi_sigma_clip'] = spec_chi_sigma_clip
		if add_igm_absorption == 1:
			hdr['igm_type'] = igm_type
		if free_z == 0:
			hdr['gal_z'] = gal_z
			hdr['free_z'] = 0
		elif free_z == 1:
			hdr['free_z'] = 1
		hdr['nrows'] = nsamples
		# add free parameters
		for pp in range(0,nparams):
			hdr['param%d' % pp] = params[pp]
		# add fix parameters, if any
		hdr['nfixpar'] = nfix_params
		if nfix_params > 0:
			for pp in range(0,nfix_params):
				hdr['fpar%d' % pp] = fix_params[pp]
				hdr['fpar%d_val' % pp] = fix_params_val[pp]
		hdr['fitmethod'] = 'mcmc'
		hdr['storesamp'] = 1
		hdr['specphot'] = 2         # spectrum only
		primary_hdu = fits.PrimaryHDU(header=hdr)

		#==> sampler chains
		cols0 = []
		col = fits.Column(name='id', format='K', array=np.array(sampler_id))
		cols0.append(col)
		for pp in range(0,nparams):
			col = fits.Column(name=params[pp], format='D', array=np.array(sampler_params[params[pp]]))
			cols0.append(col)
		col = fits.Column(name='log_sfr', format='D', array=np.array(sampler_log_sfr))
		cols0.append(col)
		col = fits.Column(name='log_mw_age', format='D', array=np.array(sampler_log_mw_age))
		cols0.append(col)
		if duste_switch==1:
			col = fits.Column(name='log_dustmass', format='D', array=np.array(sampler_logdustmass))
			cols0.append(col)
		if add_agn == 1:
			col = fits.Column(name='log_fagn_bol', format='D', array=np.array(sampler_log_fagn_bol))
			cols0.append(col)
		cols = fits.ColDefs(cols0)
		hdu1 = fits.BinTableHDU.from_columns(cols, name='samplers')

		#==> inferred parameters
		cols0 = []
		col = fits.Column(name='rows', format='3A', array=['p16','p50','p84'])
		cols0.append(col)
		# basic params
		for pp in range(0,nparams):
			p16 = np.percentile(sampler_params[params[pp]],16)
			p50 = np.percentile(sampler_params[params[pp]],50)
			p84 = np.percentile(sampler_params[params[pp]],84)
			col = fits.Column(name=params[pp], format='D', array=np.array([p16,p50,p84]))
			cols0.append(col)
		# SFR
		p16 = np.percentile(sampler_log_sfr,16)
		p50 = np.percentile(sampler_log_sfr,50)
		p84 = np.percentile(sampler_log_sfr,84)
		col = fits.Column(name='log_sfr', format='D', array=np.array([p16,p50,p84]))
		cols0.append(col)
		# mass-weighted age
		p16 = np.percentile(sampler_log_mw_age,16)
		p50 = np.percentile(sampler_log_mw_age,50)
		p84 = np.percentile(sampler_log_mw_age,84)
		col = fits.Column(name='log_mw_age', format='D', array=np.array([p16,p50,p84]))
		cols0.append(col)
		# dust mass
		if duste_switch==1:
			p16 = np.percentile(sampler_logdustmass,16)
			p50 = np.percentile(sampler_logdustmass,50)
			p84 = np.percentile(sampler_logdustmass,84)
			col = fits.Column(name='log_dustmass', format='D', array=np.array([p16,p50,p84]))
			cols0.append(col)
		# AGN
		if add_agn == 1:
			p16 = np.percentile(sampler_log_fagn_bol,16)
			p50 = np.percentile(sampler_log_fagn_bol,50)
			p84 = np.percentile(sampler_log_fagn_bol,84)
			col = fits.Column(name='log_fagn_bol', format='D', array=np.array([p16,p50,p84]))
			cols0.append(col)
		# combine
		cols = fits.ColDefs(cols0)
		hdu2 = fits.BinTableHDU.from_columns(cols, name='fit_params')

		#==> observed spectrum
		cols0 = []
		col = fits.Column(name='wave', format='D', array=np.array(spec_wave))
		cols0.append(col)
		col = fits.Column(name='flux', format='D', array=np.array(spec_flux))
		cols0.append(col)
		col = fits.Column(name='flux_err', format='D', array=np.array(spec_flux_err))
		cols0.append(col)
		cols = fits.ColDefs(cols0)
		hdu3 = fits.BinTableHDU.from_columns(cols, name='obs_spec')

		#==> best-fit model spectra from posteriors
		cols0 = []
		col = fits.Column(name='wave', format='D', array=np.array(bfit_spec_wave))
		cols0.append(col)

		col = fits.Column(name='tot_p16', format='D', array=np.array(p16_spec_tot))
		cols0.append(col)
		col = fits.Column(name='tot_p50', format='D', array=np.array(p50_spec_tot))
		cols0.append(col)
		col = fits.Column(name='tot_p84', format='D', array=np.array(p84_spec_tot))
		cols0.append(col)

		col = fits.Column(name='stellar_p16', format='D', array=np.array(p16_spec_stellar))
		cols0.append(col)
		col = fits.Column(name='stellar_p50', format='D', array=np.array(p50_spec_stellar))
		cols0.append(col)
		col = fits.Column(name='stellar_p84', format='D', array=np.array(p84_spec_stellar))
		cols0.append(col)

		if add_neb_emission == 1:
			col = fits.Column(name='nebe_p16', format='D', array=np.array(p16_spec_nebe))
			cols0.append(col)
			col = fits.Column(name='nebe_p50', format='D', array=np.array(p50_spec_nebe))
			cols0.append(col)
			col = fits.Column(name='nebe_p84', format='D', array=np.array(p84_spec_nebe))
			cols0.append(col)

			col = fits.Column(name='nebe_cont_p16', format='D', array=np.array(p16_spec_nebe_cont))
			cols0.append(col)
			col = fits.Column(name='nebe_cont_p50', format='D', array=np.array(p50_spec_nebe_cont))
			cols0.append(col)
			col = fits.Column(name='nebe_cont_p84', format='D', array=np.array(p84_spec_nebe_cont))
			cols0.append(col)

		if duste_switch == 1:
			col = fits.Column(name='duste_p16', format='D', array=np.array(p16_spec_duste))
			cols0.append(col)
			col = fits.Column(name='duste_p50', format='D', array=np.array(p50_spec_duste))
			cols0.append(col)
			col = fits.Column(name='duste_p84', format='D', array=np.array(p84_spec_duste))
			cols0.append(col)

		if add_agn == 1:
			col = fits.Column(name='agn_p16', format='D', array=np.array(p16_spec_agn))
			cols0.append(col)
			col = fits.Column(name='agn_p50', format='D', array=np.array(p50_spec_agn))
			cols0.append(col)
			col = fits.Column(name='agn_p84', format='D', array=np.array(p84_spec_agn))
			cols0.append(col)

		cols = fits.ColDefs(cols0)
		hdu4 = fits.BinTableHDU.from_columns(cols, name='bfit_spec')

		#==> best-fit model spectra from chi-square minimization
		cols0 = []
		col = fits.Column(name='wave', format='D', array=np.array(bfit_spec_wave))
		cols0.append(col)
		col = fits.Column(name='total', format='D', array=np.array(chimin_spec_tot))
		cols0.append(col)
		col = fits.Column(name='stellar', format='D', array=np.array(chimin_spec_stellar))
		cols0.append(col)

		if add_neb_emission == 1:
			col = fits.Column(name='nebe', format='D', array=np.array(chimin_spec_nebe))
			cols0.append(col)

			col = fits.Column(name='nebe_cont', format='D', array=np.array(chimin_spec_nebe_cont))
			cols0.append(col)

		if duste_switch == 1:
			col = fits.Column(name='duste', format='D', array=np.array(chimin_spec_duste))
			cols0.append(col)

		if add_agn == 1:
			col = fits.Column(name='agn', format='D', array=np.array(chimin_spec_agn))
			cols0.append(col)

		cols = fits.ColDefs(cols0)
		hdu5 = fits.BinTableHDU.from_columns(cols, name='chimin_spec')

		#==> best-fit correction factor
		cols0 = []
		col = fits.Column(name='wave', format='D', array=np.array(spec_wave))
		cols0.append(col)
		col = fits.Column(name='p16', format='D', array=np.array(p16_corr_factor))
		cols0.append(col)
		col = fits.Column(name='p50', format='D', array=np.array(p50_corr_factor))
		cols0.append(col)
		col = fits.Column(name='p84', format='D', array=np.array(p84_corr_factor))
		cols0.append(col)
		cols = fits.ColDefs(cols0)
		hdu6 = fits.BinTableHDU.from_columns(cols, name='corr_factor')

		#==> best-fit native model spectra
		cols0 = []
		col = fits.Column(name='wave', format='D', array=np.array(bfit_mod_spec_wave))
		cols0.append(col)

		col = fits.Column(name='tot_p16', format='D', array=np.array(p16_mod_spec_tot))
		cols0.append(col)
		col = fits.Column(name='tot_p50', format='D', array=np.array(p50_mod_spec_tot))
		cols0.append(col)
		col = fits.Column(name='tot_p84', format='D', array=np.array(p84_mod_spec_tot))
		cols0.append(col)

		col = fits.Column(name='stellar_p16', format='D', array=np.array(p16_mod_spec_stellar))
		cols0.append(col)
		col = fits.Column(name='stellar_p50', format='D', array=np.array(p50_mod_spec_stellar))
		cols0.append(col)
		col = fits.Column(name='stellar_p84', format='D', array=np.array(p84_mod_spec_stellar))
		cols0.append(col)

		if add_neb_emission == 1:
			col = fits.Column(name='nebe_p16', format='D', array=np.array(p16_mod_spec_nebe))
			cols0.append(col)
			col = fits.Column(name='nebe_p50', format='D', array=np.array(p50_mod_spec_nebe))
			cols0.append(col)
			col = fits.Column(name='nebe_p84', format='D', array=np.array(p84_mod_spec_nebe))
			cols0.append(col)

			col = fits.Column(name='nebe_cont_p16', format='D', array=np.array(p16_mod_spec_nebe_cont))
			cols0.append(col)
			col = fits.Column(name='nebe_cont_p50', format='D', array=np.array(p50_mod_spec_nebe_cont))
			cols0.append(col)
			col = fits.Column(name='nebe_cont_p84', format='D', array=np.array(p84_mod_spec_nebe_cont))
			cols0.append(col)

		if duste_switch == 1:
			col = fits.Column(name='duste_p16', format='D', array=np.array(p16_mod_spec_duste))
			cols0.append(col)
			col = fits.Column(name='duste_p50', format='D', array=np.array(p50_mod_spec_duste))
			cols0.append(col)
			col = fits.Column(name='duste_p84', format='D', array=np.array(p84_mod_spec_duste))
			cols0.append(col)

		if add_agn == 1:
			col = fits.Column(name='agn_p16', format='D', array=np.array(p16_mod_spec_agn))
			cols0.append(col)
			col = fits.Column(name='agn_p50', format='D', array=np.array(p50_mod_spec_agn))
			cols0.append(col)
			col = fits.Column(name='agn_p84', format='D', array=np.array(p84_mod_spec_agn))
			cols0.append(col)

		cols = fits.ColDefs(cols0)
		hdu7 = fits.BinTableHDU.from_columns(cols, name='bfit_mod')

		hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
		hdul.writeto(fits_name_out, overwrite=True)	


def calc_sampler_mwage(nsamples=0,sampler_pop_mass=[],sampler_tau=[],sampler_t0=[],
	sampler_alpha=[],sampler_beta=[],sampler_age=[]):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_mw_age0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		pop_mass = sampler_pop_mass[int(ii)]
		tau = sampler_tau[int(ii)]
		age = sampler_age[int(ii)]
		if sfh_form==0 or sfh_form==1:
			sampler_mw_age0[int(count)] = calc_mw_age(sfh_form=sfh_form,tau=tau,age=age,formed_mass=pop_mass)
		elif sfh_form==2 or sfh_form==3:
			t0 = sampler_t0[int(ii)]
			sampler_mw_age0[int(count)] = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,age=age,formed_mass=pop_mass)
		elif sfh_form==4:
			alpha = sampler_alpha[int(ii)]
			beta = sampler_beta[int(ii)]
			sampler_mw_age0[int(count)] = calc_mw_age(sfh_form=sfh_form,tau=tau,alpha=alpha,beta=beta,age=age,formed_mass=pop_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_mw_age = np.zeros(numDataPerRank*size)
	comm.Gather(sampler_mw_age0, sampler_mw_age, root=0)
	comm.Bcast(sampler_mw_age, root=0)

	sampler_log_mw_age = np.log10(sampler_mw_age)
	return sampler_log_mw_age


def calc_sampler_dustmass_fagnbol_mainSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_logdustmass0 = np.zeros(numDataPerRank)
	sampler_log_fagn_bol0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		dust_mass, fagn_bol = get_dust_mass_fagnbol_mainSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		sampler_log_fagn_bol0[int(count)] = log10(fagn_bol) 

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_logdustmass = np.zeros(numDataPerRank*size)
	sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_logdustmass, root=0)

	comm.Gather(sampler_log_fagn_bol0, sampler_log_fagn_bol, root=0)
	comm.Bcast(sampler_log_fagn_bol, root=0)

	return sampler_logdustmass, sampler_log_fagn_bol


def calc_sampler_dustmass_mainSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_logdustmass0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# dust-mass
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		dust_mass = get_dust_mass_mainSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_logdustmass = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	return sampler_logdustmass

def calc_sampler_dustmass_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_logdustmass0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# dust-mass
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		dust_mass = get_dust_mass_othSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_logdustmass = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	return sampler_logdustmass

def calc_sampler_SFR_dustmass_fagnbol_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_log_sfr0 = np.zeros(numDataPerRank)
	sampler_logdustmass0 = np.zeros(numDataPerRank)
	sampler_log_fagn_bol0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 

		SFR, dust_mass, log_fagn_bol = get_sfr_dust_mass_fagnbol_othSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		sampler_log_sfr0[int(count)] = log10(SFR)
		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		sampler_log_fagn_bol0[int(count)] = log_fagn_bol

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_logdustmass = np.zeros(numDataPerRank*size)
	sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_log_sfr0, sampler_log_sfr, root=0)
	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Gather(sampler_log_fagn_bol0, sampler_log_fagn_bol, root=0)

	comm.Bcast(sampler_log_sfr, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	comm.Bcast(sampler_log_fagn_bol, root=0)

	return sampler_log_sfr, sampler_logdustmass, sampler_log_fagn_bol


def calc_sampler_SFR_dustmass_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_log_sfr0 = np.zeros(numDataPerRank)
	sampler_logdustmass0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# dust-mass
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		SFR,dust_mass = get_sfr_dust_mass_othSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)
		sampler_log_sfr0[int(count)] = log10(SFR)
		
		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_logdustmass = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_log_sfr0, sampler_log_sfr, root=0)
	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_log_sfr, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	return (sampler_log_sfr,sampler_logdustmass)

def calc_sampler_SFR_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_log_sfr0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		tau = pow(10.0,sampler_params['log_tau'][int(ii)])
		age = pow(10.0,sampler_params['log_age'][int(ii)])
		formed_mass = pow(10.0,sampler_params['log_mass'][int(ii)])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0

		if sfh_form==2 or sfh_form==3:
			t0 = pow(10.0,sampler_params['log_t0'][int(ii)])
		if sfh_form==4:
			alpha = pow(10.0,sampler_params['log_alpha'][int(ii)])
			beta = pow(10.0,sampler_params['log_beta'][int(ii)])
		
		t, SFR_t = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,beta=beta,age=age,formed_mass=formed_mass)

		idx_excld = np.where((SFR_t<=0) | (np.isnan(SFR_t)==True) | (np.isinf(SFR_t)==True))
		t = np.delete(t, idx_excld[0])
		SFR_t = np.delete(SFR_t, idx_excld[0])
		if len(t)>0:
			sampler_log_sfr0[int(count)] = log10(SFR_t[len(t)-1])

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	sampler_log_sfr = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_log_sfr0, sampler_log_sfr, root=0)
	comm.Bcast(sampler_log_sfr, root=0)
	return sampler_log_sfr



"""
USAGE: mpirun -np [npros] python ./mcmc_pcmod_p2.py (1)data samplers hdf5 file (2)name_out_finalfit
"""

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

temp_dir = PIXEDFIT_HOME+'/data/temp/'

# default parameter set
global def_params_val
def_params_val = default_params_val()

global gal_z, add_igm_absorption, igm_type, cosmo, cosmo_str, H0, Om0
global imf, sfh_form, dust_law, duste_switch, add_neb_emission, add_agn, nwaves_mod, free_gas_logz

# open results from mcmc fitting
f = h5py.File(temp_dir+str(sys.argv[1]), 'r')

imf = f['samplers'].attrs['imf']
sfh_form = f['samplers'].attrs['sfh_form']
dust_law = f['samplers'].attrs['dust_law']
duste_switch = f['samplers'].attrs['duste_switch']
add_neb_emission = f['samplers'].attrs['add_neb_emission']
add_agn = f['samplers'].attrs['add_agn']
nwaves_mod = int(f['samplers'].attrs['nwaves_mod'])
free_gas_logz = f['samplers'].attrs['free_gas_logz']
gal_z = float(f['samplers'].attrs['gal_z'])
add_igm_absorption = f['samplers'].attrs['add_igm_absorption']
igm_type = f['samplers'].attrs['igm_type']
cosmo = f['samplers'].attrs['cosmo']
H0 = f['samplers'].attrs['H0']
Om0 = f['samplers'].attrs['Om0']

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

if gal_z<=0.0:
	free_z = 1
elif gal_z>0.0:
	free_z = 0
	def_params_val['z'] = gal_z

global smooth_velocity, sigma_smooth, smooth_lsf, name_file_lsf
smooth_velocity = f['samplers'].attrs['smooth_velocity']
sigma_smooth = f['samplers'].attrs['sigma_smooth']
smooth_lsf = f['samplers'].attrs['smooth_lsf']
if smooth_lsf == True or smooth_lsf == 1:
	name_file_lsf = f['samplers'].attrs['name_file_lsf']
	data = np.loadtxt(temp_dir+name_file_lsf)
	lsf_wave, lsf_sigma = data[:,0], data[:,1]
else:
	lsf_wave, lsf_sigma = None, None

global poly_order, del_wave_nebem, spec_chi_sigma_clip
poly_order = f['samplers'].attrs['poly_order']
del_wave_nebem = f['samplers'].attrs['del_wave_nebem']
spec_chi_sigma_clip = f['samplers'].attrs['spec_chi_sigma_clip']

# get list of free parameters and their ranges
global params, nparams, priors_min, priors_max
nparams = int(f['samplers'].attrs['nparams'])
params = []
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for pp in range(0,nparams):
	attrs = f['samplers'].attrs['par%d' % pp]
	if isinstance(attrs, str) == False:
		attrs = attrs.decode()
	params.append(attrs)
	priors_min[pp] = float(f['samplers'].attrs['min_%s' % attrs])
	priors_max[pp] = float(f['samplers'].attrs['max_%s' % attrs])

# get list of fix parameters, if any
global nfix_params, fix_params, fix_params_val
nfix_params = int(f['samplers'].attrs['nfix_params'])
if nfix_params>0:
	fix_params = []
	fix_params_val = np.zeros(nfix_params)
	for pp in range(0,nfix_params):
		attrs = f['samplers'].attrs['fpar%d' % pp]
		if isinstance(attrs, str) == False:
			attrs = attrs.decode()
		fix_params.append(attrs)

		fix_params_val[pp] = float(f['samplers'].attrs['fpar%d_val' % pp])
		# modify default parameters for next FSPS call
		def_params_val[fix_params[pp]] = float(f['samplers'].attrs['fpar%d_val' % pp])

# get observed sed: photometry and spectroscopy
global spec_wave, spec_flux, spec_flux_err, nwaves_spec
spec_wave = f['spec/wave'][:]
spec_flux = f['spec/flux'][:]
spec_flux_err = f['spec/flux_err'][:]
nwaves_spec = len(spec_wave)

# get sampler chains
nsamples = len(f['samplers/logzsol'][:])

sampler_params = {}
for pp in range(0,nparams):
	sampler_params[params[pp]] = np.zeros(nsamples)

for pp in range(0,nparams):
	str_temp = 'samplers/%s' % params[pp]
	sampler_params[params[pp]] = f[str_temp][:]
f.close()

# wavelength range of the observed spectrum
global min_spec_wave, max_spec_wave
nwaves = len(spec_wave)
min_spec_wave = min(spec_wave)
max_spec_wave = max(spec_wave)

# call FSPS
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
sp = set_initial_fsps(sp,duste_switch,add_neb_emission,add_agn,sfh_form,dust_law,smooth_velocity=smooth_velocity,
						sigma_smooth=sigma_smooth,smooth_lsf=smooth_lsf,lsf_wave=lsf_wave,lsf_sigma=lsf_sigma)

global params_fsps, nparams_fsps
params_fsps, nparams_fsps = get_params_fsps(params)

sampler_pop_mass = np.power(10.0,sampler_params['log_mass'])
sampler_age = np.power(10.0,sampler_params['log_age'])
sampler_tau = np.power(10.0,sampler_params['log_tau'])

# calculate SFR, mw-age, dust mass, log_fagn_bol
sampler_log_sfr = None
sampler_log_mw_age = None
sampler_logdustmass = None
sampler_log_fagn_bol = None
if sfh_form==0 or sfh_form==1:
	# SFR
	sampler_SFR_exp = 1.0/np.exp(sampler_age/sampler_tau)
	if sfh_form==0:
		sampler_log_sfr = np.log10(sampler_pop_mass*sampler_SFR_exp/sampler_tau/(1.0-sampler_SFR_exp)/1e+9)
	if sfh_form==1:
		sampler_log_sfr = np.log10(sampler_pop_mass*sampler_age*sampler_SFR_exp/((sampler_tau*sampler_tau)-((sampler_age*sampler_tau)+(sampler_tau*sampler_tau))*sampler_SFR_exp)/1e+9)
	# MW-age
	sampler_log_mw_age = calc_sampler_mwage(nsamples=nsamples,sampler_pop_mass=sampler_pop_mass,sampler_tau=sampler_tau,sampler_age=sampler_age)
	# dust-mass
	if duste_switch==1:
		if add_agn == 1:
			sampler_logdustmass, sampler_log_fagn_bol = calc_sampler_dustmass_fagnbol_mainSFH(nsamples=nsamples,sampler_params=sampler_params)
		elif add_agn == 0:
			sampler_logdustmass = calc_sampler_dustmass_mainSFH(nsamples=nsamples,sampler_params=sampler_params)

elif sfh_form==2 or sfh_form==3:
	sampler_t0 = np.power(10.0,sampler_params['log_t0'])
	# MW-age
	sampler_log_mw_age = calc_sampler_mwage(nsamples=nsamples,sampler_pop_mass=sampler_pop_mass,sampler_tau=sampler_tau,sampler_t0=sampler_t0,sampler_age=sampler_age)
	# SFR and dust-mass
	if duste_switch==1:
		if add_agn == 1:
			sampler_log_sfr, sampler_logdustmass, sampler_log_fagn_bol = calc_sampler_SFR_dustmass_fagnbol_othSFH(nsamples=nsamples,sampler_params=sampler_params)
		elif add_agn == 0:
			sampler_log_sfr, sampler_logdustmass = calc_sampler_SFR_dustmass_othSFH(nsamples=nsamples,sampler_params=sampler_params)
	else:
		sampler_log_sfr = calc_sampler_SFR_othSFH(nsamples=nsamples,sampler_params=sampler_params)


elif sfh_form==4:
	sampler_alpha = np.power(10.0,sampler_params['log_alpha'])
	sampler_beta = np.power(10.0,sampler_params['log_beta'])
	# MW-age
	sampler_log_mw_age = calc_sampler_mwage(nsamples=nsamples,sampler_pop_mass=sampler_pop_mass,sampler_tau=sampler_tau,
													sampler_alpha=sampler_alpha, sampler_beta=sampler_beta, sampler_age=sampler_age)
	# SFR and dust-mass
	if duste_switch==1:
		if add_agn == 1:
			sampler_log_sfr, sampler_logdustmass, sampler_log_fagn_bol = calc_sampler_SFR_dustmass_fagnbol_othSFH(nsamples=nsamples,sampler_params=sampler_params)
		elif add_agn == 0:
			sampler_log_sfr, sampler_logdustmass = calc_sampler_SFR_dustmass_othSFH(nsamples=nsamples,sampler_params=sampler_params)
	else:
		sampler_log_sfr = calc_sampler_SFR_othSFH(nsamples=nsamples,sampler_params=sampler_params)

fits_name_out = str(sys.argv[2])
store_to_fits(nsamples=nsamples,sampler_params=sampler_params,sampler_log_sfr=sampler_log_sfr,
					sampler_log_mw_age=sampler_log_mw_age,sampler_logdustmass=sampler_logdustmass,
					sampler_log_fagn_bol=sampler_log_fagn_bol,fits_name_out=fits_name_out)

if rank==0:
	sys.stdout.write('\n')

