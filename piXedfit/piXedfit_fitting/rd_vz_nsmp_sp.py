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

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.posteriors import model_leastnorm, ln_gauss_prob, ln_student_t_prob
from piXedfit.utils.filtering import interp_filters_curves, filtering_interp_filters, cwave_filters 
from piXedfit.utils.redshifting import cosmo_redshifting
from piXedfit.utils.igm_absorption import igm_att_madau, igm_att_inoue
from piXedfit.piXedfit_spectrophotometric import spec_smoothing, match_spectra_poly_legendre_fit
from piXedfit.piXedfit_model import get_no_nebem_wave_fit


def bayesian_sedfit_gauss(gal_z,zz):
	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get model spectral wavelength 
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by the IFS spectra
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get the spectral fluxes
		str_temp = 'mod/spec/f%d' % int(ii)
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

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

		if gauss_likelihood_form == 0:
			idx1 = np.where((chi_spec0>=lower) & (chi_spec0<=upper))
			m_merge = conv_mod_spec_flux_clean[idx1[0]].tolist() + norm_fluxes.tolist()
			d_merge = spec_flux_clean[idx1[0]].tolist() + obs_fluxes.tolist()
			derr_merge = spec_flux_err_clean[idx1[0]].tolist() + obs_flux_err.tolist()
			lnprob0 = ln_gauss_prob(d_merge,derr_merge,m_merge)
		elif gauss_likelihood_form == 1:
			lnprob0 = -0.5*chi2

		mod_chi2_temp[int(count)] = chi2
		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)
		mod_prob_temp[int(count)] = lnprob0

		# get parameters
		for pp in range(0,nparams-1):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: z %d of %d (%d%%) and model %d of %d (%d%%)' % (rank,zz+1,nrands_z,(zz+1)*100/nrands_z,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)
	mod_prob = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams-1):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	# Broadcast
	comm.Bcast(mod_chi2, root=0)
	comm.Bcast(mod_chi2_photo, root=0)
	comm.Bcast(mod_redcd_chi2_spec, root=0)
	comm.Bcast(mod_prob, root=0)
	comm.Bcast(mod_params, root=0)

	# add redshift
	mod_params[int(nparams)-1] = np.zeros(nmodels)+gal_z

	f.close()

	return mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob


def bayesian_sedfit_student_t(gal_z,zz):
	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get spectral wavelength
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by the IFS spectra
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get the spectral fluxes
		str_temp = 'mod/spec/f%d' % int(ii)
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

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
		lnprob0 = ln_student_t_prob(dof,chi_merge)

		mod_chi2_temp[int(count)] = chi2
		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)
		mod_prob_temp[int(count)] = lnprob0

		# get parameters
		for pp in range(0,nparams-1):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: z %d of %d (%d%%) and model %d of %d (%d%%)' % (rank,zz+1,nrands_z,(zz+1)*100/nrands_z,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)
	mod_prob = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams-1):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	# Broadcast
	comm.Bcast(mod_chi2, root=0)
	comm.Bcast(mod_chi2_photo, root=0)
	comm.Bcast(mod_redcd_chi2_spec, root=0)
	comm.Bcast(mod_prob, root=0)
	comm.Bcast(mod_params, root=0)

	# add redshift
	mod_params[int(nparams)-1] = np.zeros(nmodels)+gal_z

	f.close()
	
	return mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob


def store_to_fits(sampler_params,mod_chi2,mod_chi2_photo,mod_redcd_chi2_spec,mod_prob,fits_name_out):
	# get best-fit model
	idx, min_val = min(enumerate(mod_chi2), key=itemgetter(1))
	bfit_chi2 = mod_chi2[idx]
	bfit_rchi2_photo = mod_chi2_photo[idx]/nbands
	bfit_rchi2_spec = mod_redcd_chi2_spec[idx]

	# best-fit z
	gal_z = sampler_params['z'][idx]

	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get spectral wavelength
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by the IFS spectra
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))

	# get spectral fluxes
	str_temp = 'mod/spec/f%d' % (idx % nmodels)   # modulo
	extnc_spec = f[str_temp][:]
	
	# redshifting
	redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)
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

	# remove bad spectral points
	chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
	chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)

	# get best-fit model spectrum anc polynomial corrrection factor
	corr_factor = poly_legendre(redsh_wave[idx_mod_wave[0]])
	bfit_spec_wave = conv_mod_spec_wave
	bfit_spec_nwaves = len(bfit_spec_wave)
	bfit_spec_flux = corr_factor*conv_mod_spec_flux

	#==> Get median likelihood parameters
	crit_chi2 = np.percentile(mod_chi2, perc_chi2)
	idx_sel = np.where((mod_chi2<=crit_chi2) & (sampler_params['log_sfr']>-29.0) & (np.isnan(mod_prob)==False) & (np.isinf(mod_prob)==False))

	array_lnprob = mod_prob[idx_sel[0]] - max(mod_prob[idx_sel[0]])  # normalize
	array_prob = np.exp(array_lnprob)
	tot_prob = np.sum(array_prob)
	array_prob = array_prob/tot_prob								# normalize

	params_bfits = np.zeros((nparams,2))
	for pp in range(0,nparams):
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
	hdr['dust_ext_law'] = dust_ext_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
	if duste_switch==1:
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
	hdr['free_z'] = 1
	hdr['cosmo'] = cosmo
	hdr['H0'] = H0
	hdr['Om0'] = Om0

	# parameters
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]

	# chi-square
	hdr['redcd_chi2'] = bfit_chi2/(nbands+len(chi_spec))
	hdr['redcd_chi2_spec'] = bfit_rchi2_spec
	hdr['redcd_chi2_photo'] = bfit_rchi2_photo

	hdr['perc_chi2'] = perc_chi2

	# add columns
	cols0 = []
	col_count = 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'rows'
	col = fits.Column(name='rows', format='4A', array=['mean','std'])
	cols0.append(col)

	#=> parameters
	for pp in range(0,nparams):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = params[pp]
		col = fits.Column(name=params[pp], format='D', array=np.array([params_bfits[pp][0],params_bfits[pp][1]]))
		cols0.append(col)

	hdr['ncols'] = col_count

	# combine header
	primary_hdu = fits.PrimaryHDU(header=hdr)

	# combine binary table HDU1: parameters derived from fitting rdsps
	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols, name='fit_params')

	#==> add table for parameters that have minimum chi-square
	cols0 = []
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array([sampler_params[params[pp]][idx]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu1 = fits.BinTableHDU.from_columns(cols, name='minchi2_params')

	#==> observed spectrum
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(spec_wave))
	cols0.append(col)

	col = fits.Column(name='flux', format='D', array=np.array(spec_flux))
	cols0.append(col)

	col = fits.Column(name='flux_err', format='D', array=np.array(spec_flux_err))
	cols0.append(col)

	cols = fits.ColDefs(cols0)
	hdu2 = fits.BinTableHDU.from_columns(cols, name='obs_spec')
	
	#==> best-fit spectrum
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(bfit_spec_wave[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(bfit_spec_flux[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu3 = fits.BinTableHDU.from_columns(cols, name='bfit_spec')

	#==> correction factor
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(bfit_spec_wave[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	col = fits.Column(name='corr_factor', format='D', array=np.array(corr_factor[10:bfit_spec_nwaves-10]))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu4 = fits.BinTableHDU.from_columns(cols, name='corr_factor')

	#==> best-fit photometric SED
	photo_cwave = cwave_filters(filters)

	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(photo_cwave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(bfit_photo_fluxes))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu5 = fits.BinTableHDU.from_columns(cols, name='bfit_photo')

	hdul = fits.HDUList([primary_hdu, hdu, hdu1, hdu2, hdu3, hdu4, hdu5])
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

global gauss_likelihood_form
gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

global perc_chi2
perc_chi2 = float(config_data['perc_chi2'])

# cosmology
global cosmo, H0, Om0
cosmo = int(config_data['cosmo'])
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])

# generate random redshifts
global rand_z, nrands_z
pr_z_min = float(config_data['pr_z_min'])
pr_z_max = float(config_data['pr_z_max'])
nrands_z = int(config_data['nrands_z'])
rand_z = np.random.uniform(pr_z_min, pr_z_max, nrands_z)

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
global min_spec_wave, max_spec_wave
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

global del_wave_nebem
del_wave_nebem = float(config_data['del_wave_nebem'])

# clipping for bad spectral points in the chi-square calculation
global spec_chi_sigma_clip
spec_chi_sigma_clip = float(config_data['spec_chi_sigma_clip'])

# HDF5 file containing pre-calculated model SEDs
global models_spec
models_spec = config_data['models_spec']

# data of pre-calculated model SEDs
f = h5py.File(models_spec, 'r')

# number of model SEDs
global nmodels
nmodels = int(f['mod'].attrs['nmodels']/size)*size

# get list of parameters
global nparams, params
nparams = int(f['mod'].attrs['nparams_all'])  # include all possible parameters
params = []
for pp in range(0,nparams):
	str_temp = 'par%d' % pp 
	params.append(f['mod'].attrs[str_temp])
# add redshift
params.append('z')
nparams = nparams + 1

# modeling configurations
global imf, sfh_form, dust_ext_law, duste_switch, add_neb_emission, add_agn, gas_logu, fix_dust_index, fix_dust_index_val
imf = f['mod'].attrs['imf_type']
sfh_form = f['mod'].attrs['sfh_form']
dust_ext_law = f['mod'].attrs['dust_ext_law']
duste_switch = f['mod'].attrs['duste_switch']
add_neb_emission = f['mod'].attrs['add_neb_emission']
add_agn = f['mod'].attrs['add_agn']
gas_logu = f['mod'].attrs['gas_logu']
if duste_switch==1:
	if 'dust_index' in params:
		fix_dust_index = 0 
	else:
		fix_dust_index = 1 
		fix_dust_index_val = f['mod'].attrs['dust_index']
f.close()

# igm absorption
global add_igm_absorption,igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

global interp_filters_waves, interp_filters_trans
interp_filters_waves,interp_filters_trans = interp_filters_curves(filters)

# running the calculation
global redcd_chi2
redcd_chi2 = 2.0

# iteration for calculations
nmodels_merge = int(nmodels*nrands_z)
mod_params_merge = np.zeros((nparams,nmodels_merge))
mod_chi2_merge = np.zeros(nmodels_merge)
mod_chi2_photo_merge = np.zeros(nmodels_merge)
mod_redcd_chi2_spec_merge = np.zeros(nmodels_merge)
mod_prob_merge = np.zeros(nmodels_merge)

for zz in range(0,nrands_z):
	# redshift
	gal_z = rand_z[zz]

	# running the calculation
	if likelihood_form == 'gauss':
		mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob = bayesian_sedfit_gauss(gal_z,zz)
	elif likelihood_form == 'student_t':
		mod_params, mod_chi2, mod_chi2_photo, mod_redcd_chi2_spec, mod_prob = bayesian_sedfit_student_t(gal_z,zz)
	else:
		print ("likelihood_form is not recognized!")
		sys.exit()

	for pp in range(0,nparams):
		mod_params_merge[pp,int(zz*nmodels):int((zz+1)*nmodels)] = mod_params[pp][:]

	mod_chi2_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_chi2[:]
	mod_chi2_photo_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_chi2_photo[:]
	mod_redcd_chi2_spec_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_redcd_chi2_spec[:]
	mod_prob_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_prob[:]

# change the format to dictionary
sampler_params = {}
for pp in range(0,nparams):
	sampler_params[params[pp]] = mod_params_merge[pp]

# store to fits file
if rank == 0:
	fits_name_out = str(sys.argv[4])
	store_to_fits(sampler_params,mod_chi2_merge,mod_chi2_photo_merge,mod_redcd_chi2_spec_merge,mod_prob_merge,fits_name_out)
	sys.stdout.write('\n')

