import numpy as np 
import sys, os
from numpy import unravel_index
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Gaussian2DKernel
from astropy.stats import SigmaClip
from astropy.modeling.models import Gaussian2D
from scipy.interpolate import interp1d, interp2d

from ..piXedfit_images.images_utils import get_largest_FWHM_PSF, k_lmbd_Fitz1986_LMC
from ..piXedfit_model.model_utils import get_no_nebem_wave_fit


__all__ = ["spec_smoothing", "match_spectra_poly_legendre", "match_spectra_poly_legendre_fit"]



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
	final_wave=[],wave_clean=[],z=0.001,del_wave_nebem=10.0,order=3):
	""" Function for matching normalization og two spectra by multiplying one spectra with a smooth factor, 
	which is derived from polynomial interpolation to the continuum flux ratio as a function of wavelength
	"""

	if len(final_wave)==0:
		final_wave = ref_spec_wave

	if len(wave_clean)==0:
		# get wavelength grids (in the final wave sampling) that are free from emission lines:
		wave_clean,wave_mask = get_no_nebem_wave_fit(z,final_wave,del_wave_nebem)

	# get flux ratio of continuum between the two spectra in the final wave sampling:
	f = interp1d(in_spec_wave,in_spec_flux)
	in_spec_flux_clean = f(wave_clean)
	f = interp1d(ref_spec_wave,ref_spec_flux)
	ref_spec_flux_clean = f(wave_clean)

	flux_ratio = ref_spec_flux_clean/in_spec_flux_clean

	idx = np.where((np.isnan(flux_ratio)==False) & (np.isinf(flux_ratio)==False))
	if len(idx[0])>0:
		flux_ratio = flux_ratio[idx[0]]
		wave_clean = wave_clean[idx[0]]

	poly_legendre = np.polynomial.legendre.Legendre.fit(wave_clean, flux_ratio, order)
	factor = poly_legendre(final_wave)

	f = interp1d(in_spec_wave,in_spec_flux)
	final_flux = f(final_wave)*factor

	ratio_spec_wave = wave_clean
	ratio_spec_flux = flux_ratio 

	return final_wave, final_flux, ratio_spec_wave, ratio_spec_flux, factor


def match_spectra_poly_legendre_fit(in_spec_wave,in_spec_flux,ref_spec_wave,ref_spec_flux,
	wave_clean,z,del_wave_nebem,order):
	""" Function for matching normalization og two spectra by multiplying one spectra with a smooth factor, 
	which is derived from polynomial interpolation to the continuum flux ratio as a function of wavelength
	"""

	if len(wave_clean)==0:
		# get wavelength grids (in the final wave sampling) that are free from emission lines:
		wave_clean,wave_mask = get_no_nebem_wave_fit(z,final_wave,del_wave_nebem)

	# get flux ratio of continuum between the two spectra in the final wave sampling:
	f = interp1d(in_spec_wave,in_spec_flux)
	in_spec_flux_clean = f(wave_clean)
	f = interp1d(ref_spec_wave,ref_spec_flux)
	ref_spec_flux_clean = f(wave_clean)

	flux_ratio = ref_spec_flux_clean/in_spec_flux_clean

	# remove bad fluxes
	idx = np.where((np.isnan(flux_ratio)==False) & (np.isinf(flux_ratio)==False))
	if len(idx[0])>0:
		flux_ratio = flux_ratio[idx[0]]
		wave_clean = wave_clean[idx[0]]

	# fit polynomial function
	final_wave = in_spec_wave
	poly_legendre = np.polynomial.legendre.Legendre.fit(wave_clean, flux_ratio, order)
	factor = poly_legendre(final_wave)

	# get final re-scaled spectrum
	final_flux = in_spec_flux*factor

	# get model/reference spectrum


	return final_wave, final_flux, factor, ref_spec_flux_clean

