import numpy as np 
import sys, os
from numpy import unravel_index
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Gaussian2DKernel
from astropy.stats import SigmaClip
from astropy.modeling.models import Gaussian2D
from scipy.interpolate import interp1d

from ..piXedfit_images.images_utils import get_largest_FWHM_PSF, k_lmbd_Fitz1986_LMC
from ..piXedfit_model.model_utils import get_no_nebem_wave_fit


__all__ = ["spec_smoothing", "match_spectra_poly_legendre", "match_spectra_poly_legendre_fit", "match_spectra_poly_legendre_extrapolate", 
		   "get_wavelength_grid_nirspec", "sort_by_filter", "spectral_matching_specphoto", "check_disperser_filter", "get_spec_resolution_disperser_filter"]



def correct_Galacticdust(Gal_EBV, wave, spec, spec_err):
	"""A function to correct for Galactic foreground dust extinction
	"""
	Alambda = k_lmbd_Fitz1986_LMC(wave)*Gal_EBV
	corr_spec = spec*np.power(10.0,0.4*Alambda)
	corr_spec_err = spec_err*np.power(10.0,0.4*Alambda)

	return corr_spec, corr_spec_err


def spec_smoothing(wave,flux,spec_sigma):
	"""Function for convolving a spectrum to meet a given spectral resolution
	"""
	min_wave = int(min(wave))
	max_wave = int(max(wave))

	wave_lin = np.linspace(min_wave,max_wave,max_wave-min_wave+1)
	f = interp1d(wave, flux, fill_value="extrapolate")
	flux_wave_lin = f(wave_lin)
	flux_wave_lin[0], flux_wave_lin[len(wave_lin)-1] = flux_wave_lin[1], flux_wave_lin[len(wave_lin)-2]

	spec_kernel = Gaussian1DKernel(stddev=spec_sigma)
			
	conv_flux = convolve_fft(flux_wave_lin, spec_kernel)

	# return the spectrum to the original wavelength sampling
	f = interp1d(wave_lin, conv_flux, fill_value="extrapolate")
	smoothed_flux = f(wave)
	smoothed_flux[0], smoothed_flux[len(wave)-1] = smoothed_flux[1], smoothed_flux[len(wave)-2]

	return wave, smoothed_flux


def match_spectra_poly_legendre(in_spec_wave=[],in_spec_flux=[],ref_spec_wave=[],ref_spec_flux=[],
	final_wave=None,wave_clean=None,z=0.001,del_wave_nebem=10.0,order=3):
	""" Function for matching normalization og two spectra by multiplying one spectra with a smooth factor, 
	which is derived from polynomial interpolation to the continuum flux ratio as a function of wavelength
	"""

	# remove nan and inf values from the input spectra
	in_spec_wave = np.array(in_spec_wave)
	in_spec_flux = np.array(in_spec_flux)

	in_spec_wave = in_spec_wave[np.isfinite(in_spec_flux)]
	in_spec_flux = in_spec_flux[np.isfinite(in_spec_flux)]

	if final_wave is None:
		final_wave = in_spec_wave

	if wave_clean is None:
		# get wavelength grids (in the final wave sampling) that are free from emission lines:
		wave_clean,wave_mask = get_no_nebem_wave_fit(z,final_wave,del_wave_nebem)

	# get flux continuum excess between the two spectra in the final wave sampling:
	f = interp1d(in_spec_wave,in_spec_flux, fill_value='extrapolate')
	in_spec_flux_clean = f(wave_clean)
	f = interp1d(ref_spec_wave,ref_spec_flux, fill_value='extrapolate')
	ref_spec_flux_clean = f(wave_clean)

	#flux_ratio = ref_spec_flux_clean/in_spec_flux_clean
	flux_ratio = ref_spec_flux_clean - in_spec_flux_clean       ####

	idx = np.where((np.isnan(flux_ratio)==False) & (np.isinf(flux_ratio)==False))
	if len(idx[0])>0:
		flux_ratio = flux_ratio[idx[0]]
		wave_clean = wave_clean[idx[0]]

	poly_legendre = np.polynomial.legendre.Legendre.fit(wave_clean, flux_ratio, order)
	factor = poly_legendre(final_wave)

	f = interp1d(in_spec_wave,in_spec_flux, fill_value='extrapolate')
	#final_flux = f(final_wave)*factor
	final_flux = f(final_wave) + factor         ####

	ratio_spec_wave = wave_clean
	ratio_spec_flux = flux_ratio 

	return final_wave, final_flux, ratio_spec_wave, ratio_spec_flux, factor


def match_spectra_poly_legendre_extrapolate(in_spec_wave=[],in_spec_flux=[],ref_spec_wave=[],ref_spec_flux=[],
	final_wave=None,wave_clean=None,factor_wave_extrapolate=None,z=0.001,del_wave_nebem=10.0,order=3):
	""" Function for matching normalization og two spectra by multiplying one spectra with a smooth factor, 
	which is derived from polynomial interpolation to the continuum flux ratio as a function of wavelength
	"""

	if final_wave is None:
		final_wave = in_spec_wave

	if wave_clean is None:
		# get wavelength grids (in the final wave sampling) that are free from emission lines:
		wave_clean,wave_mask = get_no_nebem_wave_fit(z,final_wave,del_wave_nebem)

	# get flux ratio of continuum between the two spectra in the final wave sampling:
	f = interp1d(in_spec_wave,in_spec_flux, fill_value='extrapolate')
	in_spec_flux_clean = f(wave_clean)
	f = interp1d(ref_spec_wave,ref_spec_flux, fill_value='extrapolate')
	ref_spec_flux_clean = f(wave_clean)

	#flux_ratio = ref_spec_flux_clean/in_spec_flux_clean
	flux_ratio = ref_spec_flux_clean - in_spec_flux_clean       ####

	idx = np.where((np.isnan(flux_ratio)==False) & (np.isinf(flux_ratio)==False))
	if len(idx[0])>0:
		flux_ratio = flux_ratio[idx[0]]
		wave_clean = wave_clean[idx[0]]

	poly_legendre = np.polynomial.legendre.Legendre.fit(wave_clean, flux_ratio, order)
	factor = poly_legendre(final_wave)

	f = interp1d(in_spec_wave,in_spec_flux, fill_value='extrapolate')
	#final_flux = f(final_wave)*factor
	final_flux = f(final_wave) + factor         ####

	ratio_spec_wave = wave_clean
	ratio_spec_flux = flux_ratio 

	# extrapolate?
	if factor_wave_extrapolate is None:
		factor_wave_extrapolate = None
		factor_extrapolate = None 
	else:
		factor_extrapolate = poly_legendre(factor_wave_extrapolate) 

	return final_wave, final_flux, ratio_spec_wave, ratio_spec_flux, factor, factor_wave_extrapolate, factor_extrapolate


def match_spectra_poly_legendre_fit(in_spec_wave,in_spec_flux,ref_spec_wave,ref_spec_flux,
	wave_clean,z,del_wave_nebem,order):
	""" Function for matching normalization og two spectra by multiplying one spectra with a smooth factor, 
	which is derived from polynomial interpolation to the continuum flux ratio as a function of wavelength
	"""

	from scipy.stats import sigmaclip

	if len(wave_clean)==0:
		# get wavelength grids (in the final wave sampling) that are free from emission lines:
		wave_clean,wave_mask = get_no_nebem_wave_fit(z,final_wave,del_wave_nebem)

	# get flux ratio of continuum between the two spectra in the final wave sampling:
	f = interp1d(in_spec_wave,in_spec_flux, fill_value='extrapolate')
	in_spec_flux_clean = f(wave_clean)

	f = interp1d(ref_spec_wave,ref_spec_flux, fill_value='extrapolate')
	ref_spec_flux_clean = f(wave_clean)

	#flux_ratio = ref_spec_flux_clean/in_spec_flux_clean
	flux_ratio = ref_spec_flux_clean - in_spec_flux_clean       ####

	res0 = (ref_spec_flux_clean-in_spec_flux_clean)/ref_spec_flux_clean
	res,lower,upper = sigmaclip(res0, low=2.5, high=2.5)

	# remove bad fluxes
	idx = np.where((np.isnan(flux_ratio)==False) & (np.isinf(flux_ratio)==False) & (flux_ratio>0.0) & (res0>=lower) & (res0<=upper))
	flux_ratio = flux_ratio[idx[0]]
	wave_clean = wave_clean[idx[0]]

	# fit polynomial function
	final_wave = in_spec_wave
	order = int(order)
	poly_legendre = np.polynomial.legendre.Legendre.fit(wave_clean, flux_ratio, order)
	factor = poly_legendre(final_wave)

	# get final re-scaled spectrum
	#final_flux = in_spec_flux*factor
	final_flux = in_spec_flux + factor         ####

	# get model/reference spectrum

	return final_wave, final_flux, factor, ref_spec_flux_clean


def check_disperser_filter(disperser_filter):
	"""Check if the given disperser and filter combination is valid."""
	valid_combinations = [
		'g140m-f070lp', 'g140m-f100lp', 'g235m-f170lp', 'g395m-f290lp',
		'g140h-f070lp', 'g140h-f100lp', 'g235h-f170lp', 'g395h-f290lp',
		'prism-clear'
	]

	if not isinstance(disperser_filter, str):
		raise ValueError("disperser_filter must be a string.")
	if '-' not in disperser_filter:
		raise ValueError("disperser_filter must be in the format 'disperser-filter'.")
	
	if disperser_filter not in valid_combinations:
		raise ValueError(f"Invalid disperser-filter combination: {disperser_filter}. "
						 "Valid combinations are: " + ", ".join(valid_combinations))

def get_wavelength_grid_nirspec(header):
    crval = header['CRVAL3']  # Starting wavelength (at ref pixel)
    crpix = header['CRPIX3']  # Reference pixel (1-based)
    cdelt = header['CDELT3']  # Wavelength increment per pixel
    naxis = header['NAXIS3']  # Number of wavelength channels
    wave_um = crval + (np.arange(naxis) + 1 - crpix) * cdelt  # in microns
    return wave_um

def sort_by_filter(input_list, sort_filter):
    def get_filter_key(item):
        # Extract the filter part (after the dash)
        return sort_filter.index(item.split('-')[1]) if item.split('-')[1] in sort_filter else len(sort_filter)
    
    return sorted(input_list, key=get_filter_key)

def get_spec_resolution_disperser_filter(disperser_filter):
	"""Get the spectral resolution for a given disperser and filter combination."""
	# Define the spectral resolution for each disperser and filter combination
	channels1 = ['g140m-f070lp', 'g140m-f100lp', 'g235m-f170lp', 'g395m-f290lp']
	channels2 = ['g140h-f070lp', 'g140h-f100lp', 'g235h-f170lp', 'g395h-f290lp']
	channels3 = ['prism-clear']

	if disperser_filter in channels1:
		spec_resolution = 1000
	elif disperser_filter in channels2:
		spec_resolution = 2700
	elif disperser_filter in channels3:
		spec_resolution = 100

	return spec_resolution


def spectral_matching_specphoto(gal_z, filters, photo_seds, photo_seds_err, spec_wave_angstrom, spec_seds, spec_seds_err, ncpu=10, nmodels=100000, 
								spec_resolution=2000, del_wave_nebem=20.0, poly_order=3, verbose=False):
	"""A function for spectral matching between photometric SED and spectroscopic SED
	spec_wave_angstrom: Wavelength grid of the spectroscopic SED in Angstrom
	spec_seds: Spectroscopic SEDs in erg/s/cm^2/Angstrom (N, nwaves)
	spec_seds_err: Spectroscopic SEDs errors in erg/s/cm^2/Angstrom (N, nwaves)
	photo_seds: Photometric SEDs in erg/s/cm^2/Angstrom (N, nbands)
	photo_seds_err: Photometric SEDs errors in erg/s/cm^2/Angstrom (N, nbands)
	"""

	from astropy.cosmology import FlatLambdaCDM

	from ..piXedfit_model import model_sed_worker, run_parallel_model_seds, generate_modelSED_photo, generate_modelSED_spec, generate_model_seds_parallel
	from ..piXedfit_fitting import fit_sed_batch

	import fsps
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=1, add_neb_emission=1)

	nseds = len(photo_seds)

	imf_type=1
	duste_switch=0
	add_neb_emission=1
	add_neb_continuum=1
	dust_law=0
	sfh_form=1        # delayed tau SFH
	add_agn=0
	add_igm_absorption=1
	igm_type=0
	smooth_velocity=True
	sigma_smooth=0.0
	spec_resolution=3000
	smooth_lsf=False
	lsf_wave=None
	lsf_sigma=None
	cosmo='flat_LCDM'
	H0=70.0
	Om0=0.3

	def_params_val = {'log_mass':0.0,'z':gal_z,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54, 'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,
                  'dust2':0.0,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1, 'log_t0':0.4,'log_tau':0.4,'logzsol':0.0,
                  'gas_logu':-2.0,'gas_logz':None}
	
	age_univ = FlatLambdaCDM(H0=70.0, Om0=0.3).age(gal_z)
	max_log_age = np.log10(age_univ.value)
	
	mod_dust2 = np.random.uniform(0.0, 3.0, nmodels)
	mod_log_age = np.random.uniform(-2.0, max_log_age, nmodels)
	mod_log_tau = np.random.uniform(-1.0, 1.5, nmodels)
	mod_logzsol = np.random.uniform(-2.0, 0.2, nmodels)

	params_val_list = []
	for i in range(nmodels):
		params_val = def_params_val.copy()
		params_val['dust2'] = mod_dust2[i]
		params_val['log_age'] = mod_log_age[i]
		params_val['log_tau'] = mod_log_tau[i]
		params_val['logzsol'] = mod_logzsol[i]
		params_val_list.append(params_val)

	# Set your shared inputs once
	shared_inputs = (
		filters,
		imf_type,
		duste_switch,
		add_neb_emission,
		dust_law,
		sfh_form,
		add_agn,
		add_igm_absorption,
		igm_type,
		smooth_velocity,
		sigma_smooth,
		spec_resolution,
		smooth_lsf,
		lsf_wave,
		lsf_sigma,
		cosmo,
		H0,
		Om0
	)

	if verbose:
		print("Generating model SEDs in parallel...")
	# generate model SEDs
	#if __name__ == '__main__':
	#	# define or import `params_val_list` and `shared_inputs` here
	#	model_seds = run_parallel_model_seds(params_val_list, shared_inputs, num_cpus=ncpu)
	#	model_seds = np.array(model_seds)

	model_seds = generate_model_seds_parallel(params_val_list, shared_inputs, num_cpus=ncpu)
	model_seds = np.array(model_seds)

	if verbose:
		print ('Running SED fitting to photometric SEDs of pixels...')
	# run SED fitting. fit_results: [mod_idx, chi-square, normalization]
	fit_results = fit_sed_batch(photo_seds, photo_seds_err, model_seds, n_cpu=ncpu)

	if verbose:
		print ('Rescaling spectra of pixels...')

	rescaled_spec_seds = []
	rescaled_factor = []

	for ii in range(nseds):
		idx1 = fit_results[ii][0]	

		# get the best-fit model spectrum
		bfit_params_val = def_params_val.copy()
		bfit_params_val['dust2'] = mod_dust2[idx1]
		bfit_params_val['log_age'] = mod_log_age[idx1]
		bfit_params_val['log_tau'] = mod_log_tau[idx1]
		bfit_params_val['logzsol'] = mod_logzsol[idx1]
		bfit_params_val['log_mass'] = np.log10(fit_results[ii][2])

		bfit_model_spec = generate_modelSED_spec(sp=sp,imf_type=imf_type,duste_switch=duste_switch,add_neb_emission=add_neb_emission,dust_law=dust_law,sfh_form=sfh_form,
							add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type, smooth_velocity=True, sigma_smooth=None, 
							spec_resolution=spec_resolution, smooth_lsf=smooth_lsf, lsf_wave=lsf_wave, lsf_sigma=lsf_sigma, cosmo=cosmo, H0=H0, Om0=Om0, 
							params_val=bfit_params_val, add_neb_continuum=add_neb_continuum)
		mod_spec_wave = bfit_model_spec['wave']    # in Angstrom
		mod_spec_flux = bfit_model_spec['flux']    # in erg/s/cm^2/Angstrom
		
		# cut the model spectrum
		idx_wave = np.where((mod_spec_wave >= spec_wave_angstrom[0]-20) & (mod_spec_wave <= spec_wave_angstrom[-1]+20))[0]
		mod_spec_wave_cut = mod_spec_wave[idx_wave]
		mod_spec_flux_cut = mod_spec_flux[idx_wave]

		final_wave, final_flux, ratio_spec_wave, ratio_spec_flux, factor = match_spectra_poly_legendre(in_spec_wave=spec_wave_angstrom, in_spec_flux=spec_seds[ii], 
																					ref_spec_wave=mod_spec_wave_cut, ref_spec_flux=mod_spec_flux_cut,
																					final_wave=spec_wave_angstrom, z=gal_z, del_wave_nebem=del_wave_nebem, order=poly_order)
		rescaled_spec_seds.append(final_flux)
		rescaled_factor.append(factor)

		if verbose:
			sys.stdout.write('\r')
			sys.stdout.write('pix: %d of %d (%d%%)' % (ii+1,nseds,(ii+1)*100/nseds))
			sys.stdout.flush()
	if verbose:
		sys.stdout.write('\n')


	return np.asarray(rescaled_spec_seds), np.asarray(rescaled_factor)
		

