import numpy as np 
import math
import sys, os
import operator
#import photutils
#import astropy
from numpy import unravel_index
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve_fft
from astropy.stats import SigmaClip
from astropy.modeling.models import Gaussian2D
from scipy import interpolate
from reproject import reproject_exact
from photutils import Background2D, MedianBackground
from photutils import CosineBellWindow, HanningWindow, create_matching_kernel
from astropy.cosmology import *

from ..utils.filtering import cwave_filters


__all__ = ["sort_filters", "kpc_per_pixel", "k_lmbd_Fitz1986_LMC", "EBV_foreground_dust", "skybg_sdss", "get_gain_dark_variance",
			"var_img_sdss", "var_img_GALEX", "var_img_2MASS", "var_img_WISE", "var_img_from_unc_img",
			"var_img_from_weight_img", "segm_sextractor", "mask_region_bgmodel", "subtract_background", "get_psf_fwhm",
			"get_largest_FWHM_PSF", "ellipse_fit", "draw_ellipse", "ellipse_sma", "crop_ellipse_galregion",
			"crop_ellipse_galregion_fits", "crop_stars", "crop_stars_galregion_fits", "crop_square_region_fluxmap", 
			"crop_image_given_radec", "crop_image_given_xy", "sci_var_img_miniJPAS", "create_kernels_miniJPAS", 
			"check_avail_kernel", "create_kernel_gaussian"]


def sort_filters(filters):
	photo_wave = cwave_filters(filters)
	id_sort = np.argsort(photo_wave)

	sorted_filters = []
	for ii in range(0,len(filters)):
		sorted_filters.append(filters[id_sort[ii]])

	return sorted_filters

def kpc_per_pixel(z=0.01,arcsec_per_pix=1.5,cosmo='flat_LCDM',H0=70.0,Om0=0.3):
	"""Function for calculating a physical scale (in kpc) corresponding to a pixel size of an imaging data.

	:param z: (default: 0.01)
		Redshift of the galaxy.

	:param arsec_per_pix: (default: 1.5)
		Pixel size in arcsecond.

	:param cosmo: (default: 'flat_LCDM')
		Choices for the cosmology. The options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5, (7)'Planck18' or 6.
		These are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:returns kpc_per_pix:
		corresponding physical scale in kpc unit of the given pixel size. 
	"""

	if cosmo=='flat_LCDM' or cosmo==0:
		cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
		kpc_per_arcmin = cosmo1.kpc_proper_per_arcmin(z)
	elif cosmo=='WMAP5' or cosmo==1:
		kpc_per_arcmin = WMAP5.kpc_proper_per_arcmin(z)
	elif cosmo=='WMAP7' or cosmo==2:
		kpc_per_arcmin = WMAP7.kpc_proper_per_arcmin(z)
	elif cosmo=='WMAP9' or cosmo==3:
		kpc_per_arcmin = WMAP9.kpc_proper_per_arcmin(z)
	elif cosmo=='Planck13' or cosmo==4:
		kpc_per_arcmin = Planck13.kpc_proper_per_arcmin(z)
	elif cosmo=='Planck15' or cosmo==5:
		kpc_per_arcmin = Planck15.kpc_proper_per_arcmin(z)
	elif cosmo=='Planck18' or cosmo==6:
		kpc_per_arcmin = Planck18.kpc_proper_per_arcmin(z)

	arcmin_per_pix = arcsec_per_pix/60.0
	kpc_per_pix = kpc_per_arcmin.value*arcmin_per_pix 

	return kpc_per_pix


def k_lmbd_Fitz1986_LMC(wavelength_Ang):
	"""A function for calculting dust extc. curve of Fitzpatrick et al. 1986.
	To be used for correction of foreground Galactic dust ettenuation 
	"""

	if np.isscalar(wavelength_Ang)==False:
		lmbd_micron = np.asarray(wavelength_Ang)/1e+4
		inv_lmbd_micron = 1.0/lmbd_micron

		k = np.zeros(len(wavelength_Ang))

		idx = np.where(inv_lmbd_micron>=5.9)
		par1 = np.square(inv_lmbd_micron[idx[0]]-(4.608*4.608*lmbd_micron[idx[0]]))
		k[idx[0]] = -0.69 + (0.89/lmbd_micron[idx[0]]) + (2.55/(par1+(0.994*0.994))) + (0.5*((0.539*(inv_lmbd_micron[idx[0]]-5.9)*(inv_lmbd_micron[idx[0]]-5.9)) + (0.0564*(inv_lmbd_micron[idx[0]]-5.9)*(inv_lmbd_micron[idx[0]]-5.9)*(inv_lmbd_micron[idx[0]]-5.9)))) + 3.1
		
		idx = np.where((inv_lmbd_micron<5.9) & (inv_lmbd_micron>3.3))
		par1 = np.square(inv_lmbd_micron[idx[0]]-(4.608*4.608*lmbd_micron[idx[0]]))
		k[idx[0]] = -0.69 + (0.89/lmbd_micron[idx[0]]) + (3.55/(par1+(0.994*0.994))) + 3.1

		idx = np.where((inv_lmbd_micron<=3.3) & (inv_lmbd_micron>=1.1))
		yy = inv_lmbd_micron[idx[0]]-1.82
		ax = 1 + (0.17699*yy) - (0.50447*yy*yy) - (0.02427*yy*yy*yy) + (0.72085*yy*yy*yy*yy) + (0.01979*yy*yy*yy*yy*yy) - (0.77530*yy*yy*yy*yy*yy*yy) + (0.32999*yy*yy*yy*yy*yy*yy*yy)
		bx = (1.41338*yy) + (2.28305*yy*yy) + (1.07233*yy*yy*yy) - (5.38434*yy*yy*yy*yy) - (0.62251*yy*yy*yy*yy*yy) + (5.30260*yy*yy*yy*yy*yy*yy) - (2.09002*yy*yy*yy*yy*yy*yy*yy)
		k[idx[0]] = (3.1*ax) + bx

		idx = np.where(inv_lmbd_micron<1.1)
		ax = 0.574*np.power(inv_lmbd_micron[idx[0]],1.61)
		bx = -0.527*np.power(inv_lmbd_micron[idx[0]],1.61)
		k[idx[0]] = (3.1*ax) + bx
	else:
		lmbd_micron = wavelength_Ang/10000.0
		inv_lmbd_micron = 1.0/lmbd_micron

		if inv_lmbd_micron>=5.9:
			par1 = (inv_lmbd_micron-(4.608*4.608*lmbd_micron))*(inv_lmbd_micron-(4.608*4.608*lmbd_micron))
			k = -0.69 + (0.89/lmbd_micron) + (2.55/(par1+(0.994*0.994))) + (0.5*((0.539*(inv_lmbd_micron-5.9)*(inv_lmbd_micron-5.9)) + (0.0564*(inv_lmbd_micron-5.9)*(inv_lmbd_micron-5.9)*(inv_lmbd_micron-5.9)))) + 3.1
		elif inv_lmbd_micron<5.9 and inv_lmbd_micron>3.3:
			par1 = (inv_lmbd_micron-(4.608*4.608*lmbd_micron))*(inv_lmbd_micron-(4.608*4.608*lmbd_micron))
			k = -0.69 + (0.89/lmbd_micron) + (3.55/(par1+(0.994*0.994))) + 3.1
		elif inv_lmbd_micron<=3.3 and inv_lmbd_micron>=1.1:
			yy = inv_lmbd_micron-1.82
			ax = 1 + 0.17699*yy - 0.50447*yy*yy - 0.02427*yy*yy*yy + 0.72085*yy*yy*yy*yy + 0.01979*yy*yy*yy*yy*yy - 0.77530*yy*yy*yy*yy*yy*yy + 0.32999*yy*yy*yy*yy*yy*yy*yy
			bx = 1.41338*yy + 2.28305*yy*yy + 1.07233*yy*yy*yy - 5.38434*yy*yy*yy*yy - 0.62251*yy*yy*yy*yy*yy + 5.30260*yy*yy*yy*yy*yy*yy - 2.09002*yy*yy*yy*yy*yy*yy*yy
			k = 3.1*ax + bx
		elif inv_lmbd_micron<1.1:
			ax = 0.574*math.pow(inv_lmbd_micron,1.61)
			bx = -0.527*pow(inv_lmbd_micron,1.61)
			k = 3.1*ax + bx

	return k


def EBV_foreground_dust(Alambda_SDSS):
	"""A function for estimating E(B-V) of the foreground Galactic dust attenuation
	given A_lambda in 5 SDSS bands. Assuming Fitzpatrick et al. 1986
	"""

	## central wavelengths of the SDSS 5 bands: 
	filters = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']
	#wave_SDSS =filtering.cwave_filters(filters)
	wave_SDSS = cwave_filters(filters)

	## calculate average E(B-V):
	#ebv_SDSS = np.zeros(5)
	#ebv_SDSS[0] = Alambda_SDSS[0]/k_lmbd_Fitz1986_LMC(wave_SDSS[0])
	#ebv_SDSS[1] = Alambda_SDSS[1]/k_lmbd_Fitz1986_LMC(wave_SDSS[1])
	#ebv_SDSS[2] = Alambda_SDSS[2]/k_lmbd_Fitz1986_LMC(wave_SDSS[2])
	#ebv_SDSS[3] = Alambda_SDSS[3]/k_lmbd_Fitz1986_LMC(wave_SDSS[3])
	#ebv_SDSS[4] = Alambda_SDSS[4]/k_lmbd_Fitz1986_LMC(wave_SDSS[4])
	ebv_SDSS = Alambda_SDSS/k_lmbd_Fitz1986_LMC(wave_SDSS)

	ave_ebv = np.mean(ebv_SDSS)

	return ave_ebv


def skybg_sdss(fits_image):
	"""A function for reconstructing background image of an SDSS image.
	The sample background image is stored in HDU2 of an SDSS image.
	As output, a fits file containing a background image is produced 

	:param fits_image:
		An SDSS image in fits file

	:returns output:
		A dictionary format that contains background image and the name 
	"""
	#print ("[Construct sky-background of an SDSS image: %s]" % fits_image)
	hdu = fits.open(fits_image)
	hdu2_data = hdu[2].data
	ALLSKY = hdu2_data[0][0]
	XINTERP = hdu2_data[0][1]
	YINTERP = hdu2_data[0][2]

	dim_y0 = ALLSKY.shape[0]
	dim_x0 = ALLSKY.shape[1]
	x = np.arange(0,dim_x0,1)
	y = np.arange(0,dim_y0,1)
	f = interpolate.interp2d(x, y, ALLSKY, kind='linear')

	dim_y = hdu[0].shape[0]
	dim_x = hdu[0].shape[1]
	full_sky = f(XINTERP,YINTERP)

	## write to a fits file:
	out_fits_name = "skybg_%s" % fits_image
	header = fits.getheader(fits_image)
	fits.writeto(out_fits_name, full_sky, header, overwrite=True)
	#print ("produce %s" % out_fits_name)

	output = {}
	output['skybg'] = full_sky
	output['skybg_name'] = out_fits_name

	return output

#### define function to get gain and dark_variance from input run and camcol in SDSS:
def get_gain_dark_variance(band,run,camcol):
    #### get gain:
    if camcol == 1:
        if band==1:
            gain = 1.62
        elif band==2:
            gain = 3.32
        elif band==3:
            gain = 4.71
        elif band==4:
            gain = 5.165
        elif band==5:
            gain = 4.745   
    elif camcol == 2: 
        if band==1:
            if run > 1100:
                gain = 1.825
            else:
                gain = 1.595;
        elif band==2:
            gain = 3.855
        elif band==3:
            gain = 4.6
        elif band==4:
            gain = 6.565 
        elif band==5:
            gain = 5.155      
    elif camcol == 3:
        if band==1:
            gain = 1.59 
        elif band==2:
            gain = 3.845
        elif band==3:
            gain = 4.72
        elif band==4:
            gain = 4.86 
        elif band==5:
            gain = 4.885
    elif camcol == 4:
        if band==1:
            gain = 1.6 
        elif band==2:
            gain = 3.995
        elif band==3:
            gain = 4.76
        elif band==4:
            gain = 4.885
        elif band==5:
            gain = 4.775
    elif camcol == 5:
        if band==1:
            gain = 1.47 
        elif band==2:
            gain = 4.05 
        elif band==3:
            gain = 4.725 
        elif band==4:
            gain = 4.64
        elif band==5:
            gain = 3.48      
    elif camcol == 6:
        if band==1:
            gain = 2.17 
        elif band==2:
            gain = 4.035 
        elif band==3:
            gain = 4.895 
        elif band==4:
            gain = 4.76 
        elif band==5:
            gain = 4.69

    #### get dark variance:
    if camcol == 1:
        if band==1:
            dark_variance = 9.61
        elif band==2:
            dark_variance = 15.6025 
        elif band==3:
            dark_variance = 1.8225 
        elif band==4:
            dark_variance = 7.84 
        elif band==5:
            dark_variance = 0.81 
    elif camcol == 2:
        if band==1:
            dark_variance = 12.6025
        elif band==2:
            dark_variance = 1.44
        elif band==3:
            dark_variance = 1
        elif band==4:
            if run < 1500:
                dark_variance = 5.76
            elif run > 1500:
                dark_variance = 6.25
        elif band==5:
            dark_variance = 1     
    elif camcol == 3:
        if band==1:
            dark_variance = 8.7025 
        elif band==2:
            dark_variance = 1.3225 
        elif band==3:
            dark_variance = 1.3225 
        elif band==4:
            dark_variance = 4.6225 
        elif band==5:
            dark_variance = 1
    elif camcol == 4:
        if band==1:
            dark_variance = 12.6025 
        elif band==2:
            dark_variance = 1.96 
        elif band==3:
            dark_variance = 1.3225
        elif band==4:
            if run < 1500:
                dark_variance = 6.25
            elif run > 1500:
                dark_variance = 7.5625
        elif band==5:
            if run < 1500:
                dark_variance = 9.61
            elif run > 1500:
                dark_variance = 12.6025  
    elif camcol == 5:
        if band==1:
            dark_variance = 9.3025
        elif band==2:
            dark_variance = 1.1025 
        elif band==3:
            dark_variance = 0.81 
        elif band==4:
            dark_variance = 7.84
        elif band==5:
            if run < 1500:
                dark_variance = 1.8225
            elif run > 1500:
                dark_variance = 2.1025
    elif camcol == 6:
        if band==1:
            dark_variance = 7.0225 
        elif band==2:
            dark_variance = 1.8225 
        elif band==3:
            dark_variance = 0.9025
        elif band==4:
            dark_variance = 5.0625 
        elif band==5:
            dark_variance = 1.21

    return gain, dark_variance


def var_img_sdss(fits_image=None,filter_name=None,name_out_fits=None):
	"""A function for calculating variance image of an SDSS image

	:param fits_image:
		fits file containing an SDSS image.

	:param band_char:
		A filter name in string.

	:returns name_out_fits:
		Name of output FITS file.
	"""

	#print ("[Construct variance/sigma-square image of an SDSS image: %s]" % fits_image)
	if filter_name == 'sdss_u':
		band = 1
	elif filter_name == 'sdss_g':
		band = 2
	elif filter_name == 'sdss_r':
		band = 3
	elif filter_name == 'sdss_i':
		band = 4
	elif filter_name == 'sdss_z':
		band = 5
	else:
		print ("The filter_name is not recognized!")
		sys.exit()

	# get image count and flat-field calibrator or nmgy:
	image_data = fits.open(fits_image)
	nmgy = image_data[1].data
	image_data.close()
	image_count = fits.getdata(fits_image)
	# get run and camcol
	header = image_data[0].header
	run = int(header['run'])
	camcol = int(header['camcol'])
	# dimension of SDSS image
	SDSS_xmax = image_data[0].shape[1]
	SDSS_ymax = image_data[0].shape[0]
	# get the full-image sky count
	output = skybg_sdss(fits_image)
	sky_count = output['skybg']
	#sky_count = skybg_data 

	# get the gain and dark variance
	gain, dark_variance = get_gain_dark_variance(band,run,camcol)
	# calculate full-image sigma-square:
	sigma_sq_full = np.zeros((SDSS_ymax,SDSS_xmax))
	for yy in range(0,SDSS_ymax):
		for xx in range(0,SDSS_xmax):
			DN = (image_count[yy][xx]/nmgy[xx]) + sky_count[yy][xx]
			DN_err = ((DN/gain) + dark_variance)*nmgy[xx]*nmgy[xx]     ### in nanomaggy^2
			sigma_sq_full[yy][xx] = DN_err

	# store the result into fits file:
	header = fits.getheader(fits_image)

	if name_out_fits == None:
		name_out_fits = 'var_%s' % fits_image
	fits.writeto(name_out_fits,sigma_sq_full,header,overwrite=True)

	#output = {}
	#output['sigmasq'] = sigma_sq_full
	#output['sigmasq_name'] = sigma_sq_name
	#return output
	return name_out_fits


def var_img_GALEX(filter_name='galex_fuv',sci_img=None,skybg_img=None,name_out_fits=None):
	"""Function for calculating and producing a variance image of an GALEX image

	:param filter_name: (default: 'galex_fuv')
		Filter name in string. Allowed options are: 'galex_fuv' and 'galex_nuv'.

	:param sci_img: (default: None)
		FITS file containing the science image.

	:param skybg_img: (default: None)
		FITS file of the background image.

	:param name_out: (default: None)
		Desired name for the output variance image.

	:returns name_out_fits:
		Name of output FITS file.
	"""

	# get science image:
	hdu = fits.open(sci_img)
	sci_img_data = hdu[0].data
	sci_img_header = hdu[0].header
	hdu.close()

	# get sky background image:
	hdu = fits.open(skybg_img)
	skybg_img_data = hdu[0].data
	hdu.close()

	# get exposure time:
	exp_time = float(sci_img_header['EXPTIME'])

	val0 = sci_img_data + skybg_img_data
	if filter_name == 'galex_fuv':
		sigma_sq_img_data = ((val0*exp_time) + np.square(0.050*val0*exp_time))/exp_time/exp_time
	elif filter_name == 'galex_nuv':
		sigma_sq_img_data = ((val0*exp_time) + np.square(0.027*val0*exp_time))/exp_time/exp_time
								
	if name_out_fits == None:
		name_out_fits = 'var_%s' % sci_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	#return sigma_sq_img_data
	return name_out_fits


def var_img_2MASS(sci_img=None,skyrms_img=None,skyrms_img_data=[],skyrms_value=None,name_out_fits=None):
	"""Function for deriving a variance image of 2MASS image. The estimation of uncertainty is based on information from
	http://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/#coadd

	:param sci_img: (default: None)
		FITS file containing the science image.

	:param skyrms_img: (default: None)
		FITS file containing the RMS background image. If skyrms_img_data==[] or skyrms_value==None, 
		this parameter should be provided.

	:param skyrms_img_data: (default: [] or empty)
		2D array containing the RMS background image, such as that output by subtract_background (the output['skybgrms']).

	:param skyrms_value: (default: None)
		Scalar value of median/mean of the RMS background image, in case 2D data is not available or 
		only median/mean value over the whole field is sufficient.

	:param name_out: (default: None)
		Desired name for the output FITS file. It is not mandatory parameter.

	:returns name_out_fits:
		Name of output FITS file. 
	"""

	# get science image:
	hdu = fits.open(sci_img)
	sci_img_data = hdu[0].data
	sci_img_header = hdu[0].header
	hdu.close()

	# get sky RMS image:
	if np.sum(skyrms_img_data) == 0 and skyrms_value==None:
		hdu = fits.open(skyrms_img)
		skyrms_img_data = hdu[0].data
		hdu.close()

	# typical gain and other coefficients for calculating flux error of a 2MASS image:
	gain_2mass = 10.0
	Nc = 6.0
	kc = 1.7

	# flux error: taken from http://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/#coadd
	SNR_l0 = sci_img_data/gain_2mass/Nc
	if skyrms_value != None:
		SNR_l1 = 1.0*np.square(2.0*kc*skyrms_value)
		SNR_l2 = np.square(1.0*0.024*skyrms_value)
	else:
		SNR_l1 = 1.0*np.square(2.0*kc*skyrms_img_data)
		SNR_l2 = np.square(1.0*0.024*skyrms_img_data)
	sigma_sq_img_data = SNR_l0 + SNR_l1 + SNR_l2

	if name_out_fits == None:
		name_out_fits = 'var_%s' % sci_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	#return sigma_sq_img_data
	return name_out_fits


def var_img_WISE(filter_name='wise_w1',sci_img=None,unc_img=None,skyrms_img=None,name_out_fits=None):
	"""Function for deriving variance image of an WISE image. The uncertainty estimation is based on information from
	http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html

	:param filter_name: (default: 'wise_w1')
		Filter name. Options are: 'wise_w1', 'wise_w2', 'wise_w3', and 'wise_w4'

	:param sci_img:
		FITS file containing the science image.

	:param unc_img:
		FITS file containing the uncertainty image.

	:param skyrms_img:
		FITS file containing the RMS background image.

	:param name_out:
		Desired name for the output FITS file.
	
	:returns name_out_fits:
		Name of output FITS file. 
	"""

	# get science image:
	hdu = fits.open(sci_img)
	sci_img_data = hdu[0].data
	sci_img_header = hdu[0].header
	hdu.close()

	# get sky RMS image:
	hdu = fits.open(skyrms_img)
	skyrms_img_data = hdu[0].data
	hdu.close()

	# get uncertainty image:
	hdu = fits.open(unc_img)
	unc_img_data = hdu[0].data
	hdu.close()

	if filter_name=='wise_w1':
		f_0 = 306.682
		sigma_0 = 4.600
		sigma_magzp = 0.006
	elif filter_name=='wise_w2':
		f_0 = 170.663
		sigma_0 = 2.600
		sigma_magzp = 0.007
	elif filter_name=='wise_w3':
		f_0 = 29.0448
		sigma_0 = 0.436
		sigma_magzp = 0.015
	elif filter_name=='wise_w4':
		f_0 = 8.2839
		sigma_0 = 0.124
		sigma_magzp = 0.012

	# based on information obtained from: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
	Fcorr = 5.0
	sigma_i = unc_img_data
	sigma_B = skyrms_img_data
	#sigma_conf = 0.0
	sigma_src = np.sqrt(Fcorr*(np.square(sigma_i) + (0.5*math.pi*sigma_B*sigma_B)))
	sigma_sq_img_data = np.square(sci_img_data)*((sigma_0*sigma_0/f_0/f_0) + (0.8483*sigma_magzp*sigma_magzp) + np.square(sigma_src)) ## in unit of DN

	if name_out_fits == None:
		name_out_fits = 'var_%s' % unc_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	#return sigma_sq_img_data
	return name_out_fits


def var_img_from_unc_img(unc_image=None, header=None, name_out_fits=None):
	"""Function for creating variance image from an input of uncertainty image.

	:param unc_img:
		Input FITS file containing the uncertainty image.

	:param header: (optional, default: None)
		Input FITS file header. This is optional parameter.

	:returns name_out_fits:
		Name of output FITS file.
	"""

	hdu = fits.open(unc_image)
	if header == None:
		header = hdu[0].header
	data_unc_image = hdu[0].data
	hdu.close()
	var_image = np.square(data_unc_image)

	# store to fits file:
	if name_out_fits == None:
		name_out_fits = "var_%s" % unc_image
	fits.writeto(name_out_fits, var_image, header=header, overwrite=True)

	return name_out_fits


def var_img_from_weight_img(wht_image=None, header=None, name_out_fits=None):
	"""Function for creating variance image from an input of weight image, which is defined as inverse variance.

	:param wht_image:
		Input FITS file containing weight image (i.e., inverse variance).

	:param header: (optional, default: None)
		Input FITS file header. This is optional parameter. This is optional parameter. 

	:returns name_out_fits:
		Name of output FITS file.
	"""
	hdu = fits.open(wht_image)
	if header == None:
		header = hdu[0].header
	data_image = hdu[0].data
	hdu.close()
	var_image = 1.0/data_image
	# store into fits file:
	if name_out_fits == None:
		name_out_fits = "var_%s" % wht_image
	fits.writeto(name_out_fits, var_image, header=header, overwrite=True)

	return name_out_fits


def sci_var_img_miniJPAS(filters=[],img=[],zp=[],zp_err=[]):
	"""Function for creating science images and variance images from miniJPAS data. The produced images are in flux density unit erg/s/cm^-2/Ang.

	:param filters:
		List of photometric filters names in string format. The accepted naming for the filters can be seen using :func:`list_filters` function in the :mod:`utils.filtering` module. 
		It is not mandatory to give the filters names in the wavelength order (from shortest to longest).

	:param img:
		List of the names of the input images.

	:param zp:
		List of zero-points.  

	:param zp_err:
		List of zero-point uncertainties.

	:returns sci_img:
		Dictionary containing names of produced science images. The format is sci_img['filter_name']='sci_img_name'.

	:returns var_img:
		Dictionary containing names of produced variance images. The format is var_img['filter_name']='var_img_name'.
	"""

	c = 2.998e+18        ## in A/s

	# get central wavelength of the filters:
	photo_wave = cwave_filters(filters)

	sci_img = {}
	var_img = {}
	img_unit = {}
	for bb in range(0,len(filters)):
		hdu = fits.open(img[bb])
		data_image = hdu[0].data
		header = hdu[0].header
		hdu.close()

		data_image_positive = np.sqrt(np.square(data_image))

		#=> science image
		mAB = -2.5*np.log10(data_image_positive) + zp[bb]
		f_nu = np.power(10.0,-4.0*(mAB+48.6)/10.0)
		f_lmd = c*f_nu/photo_wave[bb]/photo_wave[bb]
		# give negative sign fro previously negative values
		f_lmd_new = f_lmd*data_image/np.absolute(data_image)		# in erg/s/cm^2/Ang.
		# store into fits file:
		name_out = "sci_%s" % img[bb]
		fits.writeto(name_out, f_lmd_new, header=header, overwrite=True)
		sci_img[filters[bb]] = name_out

		# variance image
		mAB = -2.5*np.log10(data_image_positive) + zp[bb] + zp_err[bb]
		f_nu = np.power(10.0,-4.0*(mAB+48.6)/10.0)
		f_lmd1 = c*f_nu/photo_wave[bb]/photo_wave[bb]				# in erg/s/cm^2/Ang.

		mAB = -2.5*np.log10(data_image_positive) + zp[bb] - zp_err[bb]
		f_nu = np.power(10.0,-4.0*(mAB+48.6)/10.0)
		f_lmd2 = c*f_nu/photo_wave[bb]/photo_wave[bb]				# in erg/s/cm^2/Ang.

		del_f_lmd = 0.5*np.absolute(f_lmd2-f_lmd1)
		f_lmd_sq = del_f_lmd*del_f_lmd

		# store into fits file:
		name_out = "var_%s" % img[bb]
		fits.writeto(name_out, f_lmd_sq, header=header, overwrite=True)
		var_img[filters[bb]] = name_out

		# image unit: 0 means flux, 1 means surface brightness
		img_unit[filters[bb]] = 0


	return sci_img, var_img, img_unit


def create_kernels_miniJPAS(filters=[], psf_img=[], pix_scale=0.2267, alpha_cosbell=0.6):
	"""Function for creating kernels for PSF matching among miniJPAS data. This function uses some functions in Photutils package. 

	:param filters:
		List of photometric filters names in string format. The accepted naming for the filters can be seen using :func:`list_filters` function in the :mod:`utils.filtering` module. 
		It is not mandatory to give the filters names in the wavelength order (from shortest to longest).

	:param psf_img:
		Names of PSF images.

	:param pix_scale:
		Pixel size in arcsecond.

	:param alpha_cosbell:
		The alpha parameter (the percentage of array that are tapered) in the Cosine Bell window function (see `CosineBellWindow <https://photutils.readthedocs.io/en/stable/api/photutils.psf.matching.CosineBellWindow.html#photutils.psf.matching.CosineBellWindow>`_).

	:returns kernels:
		Dictionary containing names of produced kernels.

	:returns psf_fwhm:
		Dictionary containing PSF FWHMs of the PFS images. This information is taken from the header of those FITS files. 
	"""

	fwhm = np.zeros(len(filters))
	for bb in range(0,len(filters)):
		hdu = fits.open(psf_img[bb])
		fwhm[bb] = float(hdu[1].header['PSF_FWHM'])*pix_scale		# in arcsec
		hdu.close()

	# get the largest PSF size:
	idx_max, max_val = max(enumerate(fwhm), key=operator.itemgetter(1))

	# get reference PSF:
	hdu = fits.open(psf_img[idx_max])
	psf_final = hdu[1].data
	hdu.close()

	# create kernels:
	kernels = {}
	psf_fwhm = {}
	for bb in range(0,len(filters)):
		psf_fwhm[filters[bb]] = fwhm[bb]
		if bb == idx_max:
			kernels[filters[bb]] = None
		else:
			hdu = fits.open(psf_img[bb])
			psf_init = hdu[1].data
			header = hdu[1].data
			hdu.close()

			window = CosineBellWindow(alpha=alpha_cosbell)
			kernel_data = create_matching_kernel(psf_init, psf_final, window=window)

			# store into fits file:
			name_out = "kernel_%s_to_%s.fits" % (filters[bb],filters[idx_max])
			fits.writeto(name_out, kernel_data, overwrite=True)
			kernels[filters[bb]] = name_out

	return kernels, psf_fwhm


  
def segm_sextractor(fits_image=None,detect_thresh=1.5,detect_minarea=20,
	deblend_nthresh=32.0,deblend_mincont=0.005):
	"""A function to get segmentation map of a galaxy using SExtractor.
	This function uses sewpy Python wrapper.
	"""

	#if sewpypath == None:
	#	print ("sewpypath should be specified!. It is a path to sewpy installation directory.")
	#	sys.exit()

	#sys.path.insert(0, os.path.abspath(sewpypath))
	import logging
	logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)
	import sewpy
	sexpath='sex'
	name_segm = "segm_temp.fits"
	sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "FLUX_APER(3)", "FLAGS"],
			config={"DETECT_THRESH":detect_thresh, "DETECT_MINAREA":detect_minarea, "DEBLEND_NTHRESH":deblend_nthresh,
			"DEBLEND_MINCONT":deblend_mincont, "CHECKIMAGE_TYPE":"SEGMENTATION", 
			"CHECKIMAGE_NAME":name_segm},sexpath=sexpath)
	out = sew(fits_image)
	# get the segmentation map in a form of 2D array:
	hdu = fits.open(name_segm)
	segm_map = hdu[0].data
	hdu.close()

	return segm_map


def mask_region_bgmodel(fits_image=None,detect_thresh=1.5,detect_minarea=20):
	"""A function to get segmentation map containing regions associated with 
	sources within an image. This segmentation map is then used as reference
	for regions masking in the backround subtraction process

	"""

	# get segmentation map of the field:
	segm_map = segm_sextractor(fits_image,detect_thresh,detect_minarea)

	dim_y = segm_map.shape[0]
	dim_x = segm_map.shape[1]
	mask_region = np.zeros((dim_y,dim_x))
	for yy in range(0,dim_y):
		for xx in range(0,dim_x):
			if segm_map[int(yy)][int(xx)]>0:
				mask_region[int(yy)][int(xx)] = 1

	return mask_region


def subtract_background(fits_image=None, hdu_idx=0, sigma=3.0, box_size=None, mask_region=[], 
						mask_sextractor_sources=True, detect_thresh=3.0, detect_minarea=50):
	"""Function for estimating 2D background and subtracting it from an image. This function also produce RMS image.
	Basically, the input image is gridded and the sigma clipping is done to each bin/grid.

	:param fits_image:
		Input image.

	:param hdu_idx: (default: 0)
		The FITS file extension where the image is stored. Default is 0 (HDU0).

	:param sigma: (default: 3.0)
		Sigma clipping threshold value.

	:param box_size: (default: None)
		Gridding size. 

	:param mask_region: (optional, default: [])
		Region within the image that are going to be excluded. 
		mask_region should be 2D array with the same size as the input image.

	:param mask_sextractor_sources: (default: True)
		If True, source detection and segmentation will be performed with SExtractor 
		and the regions associated with the detected sources will be excluded. 
		This can reduce the contamination by astronomical sources.

	:param detect_thresh: (default: 3.0)
		Detection threshold. This is the same as DETECT_THRESH parameter in SExtractor.

	:param detect_minarea: (default: 50)
		Minimum number of pixels above threshold triggering detection. This is the same as DETECT_MINAREA parameter in SExtractor.

	"""

	# open the input image:
	hdu = fits.open(fits_image)
	data_image = hdu[int(hdu_idx)].data
	header = hdu[int(hdu_idx)].header
	hdu.close()

	# define box size: depending on the dimension of the image:
	dim_x = data_image.shape[1]
	dim_y = data_image.shape[0]
	if box_size == None:
		box_size = [int(dim_y/10),int(dim_x/10)]
	elif box_size != None:
		box_size = box_size

	if mask_sextractor_sources==True or mask_sextractor_sources==1: 
		mask_region0 = mask_region_bgmodel(fits_image=fits_image,detect_thresh=detect_thresh,detect_minarea=detect_minarea)

		if len(mask_region) == 0:
			mask_region1 = mask_region0
		else:
			if mask_region.shape[0]!=dim_y or mask_region.shape[1]!=dim_x:
				print ("dimension of mask_region should be the same with the dimension of fits_image!")
				sys.exit()
			else:
				mask_region1 = np.zeros((dim_y,dim_x))
				rows, cols = np.where((mask_region0==1) or (mask_region==1))
				mask_region1[rows,cols] = 1

		#print ("[Estimating skybackground of %s]" % fits_image)
		sigma_clip = SigmaClip(sigma=sigma)
		bkg_estimator = MedianBackground()
		bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), mask=mask_region1,
							sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

	elif mask_sextractor_sources==False or mask_sextractor_sources==0:
		if len(mask_region)==0:
			sigma_clip = SigmaClip(sigma=sigma)
			bkg_estimator = MedianBackground()
			bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

		else:
			sigma_clip = SigmaClip(sigma=sigma)
			bkg_estimator = MedianBackground()
			bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), mask=mask_region, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

	# define mask_region: including source extraction using sextractor
	#if mask_sextractor_sources==True or mask_sextractor_sources==1: 
	#	mask_region0 = mask_region_bgmodel(fits_image=fits_image,detect_thresh=detect_thresh,detect_minarea=detect_minarea)
	#elif mask_sextractor_sources==False or mask_sextractor_sources==0:
	#	mask_region0 = np.zeros((dim_y,dim_x))

	#if len(mask_region) == 0:
	#	mask_region1 = mask_region0
	#else:
	#	if mask_region.shape[0]!=dim_y or mask_region.shape[1]!=dim_x:
	#		print ("dimension of mask_region should be the same with the dimension of fits_image!")
	#		sys.exit()
	#	else:
	#		mask_region1 = np.zeros((dim_y,dim_x))
	#		for yy in range(0,dim_y):
	#			for xx in range(0,dim_x):
	#				if mask_region0[yy][xx]==1 or mask_region[yy][xx]==1:
	#					mask_region1[yy][xx] = 1

	#print ("[Estimating skybackground of %s]" % fits_image)
	#sigma_clip = SigmaClip(sigma=sigma)
	#bkg_estimator = MedianBackground()
	#bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), mask=mask_region1,
	#					sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


	skybg_image = bkg.background
	# store to fits file:
	name_out_skybg = "skybg_%s" % fits_image
	fits.writeto(name_out_skybg, skybg_image, header, overwrite=True)
	print ("produce %s" % name_out_skybg)

	# get the background rms noise image:
	skybgrms_image = bkg.background_rms
	name_out_skybgrms = "skybgrms_%s" % fits_image
	fits.writeto(name_out_skybgrms, skybgrms_image, header, overwrite=True)
	print ("produce %s" % name_out_skybgrms)

	# calculate background subtracted image:
	skybgsub_image = data_image - skybg_image
	name_out_skybgsub = "skybgsub_%s" % fits_image
	fits.writeto(name_out_skybgsub, skybgsub_image, header, overwrite=True)
	print ("produce %s" % name_out_skybgsub)

	#output = {}
	#output['skybg'] = skybg_image
	#output['skybg_name'] = name_out_skybg
	#output['skybgrms'] = skybgrms_image
	#output['skybgrms_name'] = name_out_skybgrms
	#output['skybgsub'] = skybgsub_image
	#output['skybgsub_name'] = name_out_skybgsub
	#return output


def check_avail_kernel(filter_init=None, filter_final=None):
	filters_def = ['galex_fuv', 'galex_nuv', 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z', '2mass_j', '2mass_h', '2mass_k', 
					'spitzer_irac_36', 'spitzer_irac_45', 'spitzer_irac_58', 'spitzer_irac_80', 'spitzer_mips_24', 'spitzer_mips_70', 
					'spitzer_mips_160', 'herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160', 'herschel_spire_250', 'herschel_spire_350', 
					'herschel_spire_500', 'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4']

	status = 0
	for ii in range(0,len(filters_def)):
		if filters_def[ii] == filter_init:
			status = status + 1
		if filters_def[ii] == filter_final:
			status = status + 1

	if status == 2:
		status_kernel = 1
	else:
		status_kernel = 0

	return status_kernel


def get_psf_fwhm(filters=[]):
	col_fwhm_psf = {}
	col_fwhm_psf['galex_fuv'] = 4.48
	col_fwhm_psf['galex_nuv'] = 5.05
	col_fwhm_psf['sdss_u'] = 1.4
	col_fwhm_psf['sdss_g'] = 1.4
	col_fwhm_psf['sdss_r'] = 1.2
	col_fwhm_psf['sdss_i'] = 1.2
	col_fwhm_psf['sdss_z'] = 1.2
	col_fwhm_psf['2mass_j'] = 3.4
	col_fwhm_psf['2mass_h'] = 3.4
	col_fwhm_psf['2mass_k'] = 3.5
	col_fwhm_psf['spitzer_irac_36'] = 1.9
	col_fwhm_psf['spitzer_irac_45'] = 1.81
	col_fwhm_psf['spitzer_irac_58'] = 2.11
	col_fwhm_psf['spitzer_irac_80'] = 2.82
	col_fwhm_psf['spitzer_mips_24'] = 6.43
	col_fwhm_psf['spitzer_mips_70'] = 18.74
	col_fwhm_psf['spitzer_mips_160'] = 38.78
	col_fwhm_psf['herschel_pacs_70'] = 5.67
	col_fwhm_psf['herschel_pacs_100'] = 7.04
	col_fwhm_psf['herschel_pacs_160'] = 11.18
	col_fwhm_psf['herschel_spire_250'] = 18.15
	col_fwhm_psf['herschel_spire_350'] = 24.88
	col_fwhm_psf['herschel_spire_500'] = 36.09
	col_fwhm_psf['wise_w1'] = 5.79
	col_fwhm_psf['wise_w2'] = 6.37
	col_fwhm_psf['wise_w3'] = 6.60
	col_fwhm_psf['wise_w4'] = 11.89

	nbands = len(filters)
	psf_fwhm = np.zeros(nbands)
	for ii in range(0,nbands):
		if filters[ii] in col_fwhm_psf:
			psf_fwhm[ii] = col_fwhm_psf[filters[ii]]

	return psf_fwhm



def get_largest_FWHM_PSF(filters=None, col_fwhm_psf=None, seeing_wfcam_y=None, seeing_wfcam_j=None, seeing_wfcam_h=None, seeing_wfcam_k=None,
	seeing_vircam_z=None, seeing_vircam_y=None, seeing_vircam_j=None, seeing_vircam_h=None, seeing_vircam_ks=None):
	"""A function to find a band that has largest PSF size
	"""
	if col_fwhm_psf == None:
		col_fwhm_psf = {}
		col_fwhm_psf['galex_fuv'] = 4.48
		col_fwhm_psf['galex_nuv'] = 5.05
		col_fwhm_psf['sdss_u'] = 1.4
		col_fwhm_psf['sdss_g'] = 1.4
		col_fwhm_psf['sdss_r'] = 1.2
		col_fwhm_psf['sdss_i'] = 1.2
		col_fwhm_psf['sdss_z'] = 1.2
		col_fwhm_psf['2mass_j'] = 3.4
		col_fwhm_psf['2mass_h'] = 3.4
		col_fwhm_psf['2mass_k'] = 3.5

		col_fwhm_psf['wfcam_y'] = seeing_wfcam_y
		col_fwhm_psf['wfcam_j'] = seeing_wfcam_j
		col_fwhm_psf['wfcam_h'] = seeing_wfcam_h
		col_fwhm_psf['wfcam_k'] = seeing_wfcam_k

		col_fwhm_psf['vircam_z'] = seeing_vircam_z
		col_fwhm_psf['vircam_y'] = seeing_vircam_y
		col_fwhm_psf['vircam_j'] = seeing_vircam_j
		col_fwhm_psf['vircam_h'] = seeing_vircam_h
		col_fwhm_psf['vircam_ks'] = seeing_vircam_ks

		col_fwhm_psf['spitzer_irac_36'] = 1.9
		col_fwhm_psf['spitzer_irac_45'] = 1.81
		col_fwhm_psf['spitzer_irac_58'] = 2.11
		col_fwhm_psf['spitzer_irac_80'] = 2.82
		col_fwhm_psf['spitzer_mips_24'] = 6.43
		col_fwhm_psf['spitzer_mips_70'] = 18.74
		col_fwhm_psf['spitzer_mips_160'] = 38.78
		col_fwhm_psf['herschel_pacs_70'] = 5.67
		col_fwhm_psf['herschel_pacs_100'] = 7.04
		col_fwhm_psf['herschel_pacs_160'] = 11.18
		col_fwhm_psf['herschel_spire_250'] = 18.15
		col_fwhm_psf['herschel_spire_350'] = 24.88
		col_fwhm_psf['herschel_spire_500'] = 36.09
		col_fwhm_psf['wise_w1'] = 5.79
		col_fwhm_psf['wise_w2'] = 6.37
		col_fwhm_psf['wise_w3'] = 6.60
		col_fwhm_psf['wise_w4'] = 11.89

	nbands = len(filters)
	max_fwhm = -10.0
	for bb in range(0,nbands):
		if col_fwhm_psf[filters[bb]] > max_fwhm:
			max_fwhm = col_fwhm_psf[filters[bb]]
			idx_fil_max = bb

	return idx_fil_max


def create_kernel_gaussian(psf_fwhm_init=None, psf_fwhm_final=None, alpha_cosbell=1.5, pixsize_PSF_target=0.25, size=[101,101]):
	y_cent = (size[0]-1)/2
	x_cent = (size[1]-1)/2 

	# Make PSF. estimate sigma in unit of pixel:
	# by definition fwhm = 2.355*sigma
	sigma = psf_fwhm_init/2.355/pixsize_PSF_target			# in pixel
	y, x = np.mgrid[0:size[0], 0:size[1]]
	gm1 = Gaussian2D(100, x_cent, y_cent, sigma, sigma)
	model_psf_init = gm1(x, y)

	sigma = psf_fwhm_final/2.355/pixsize_PSF_target
	y, x = np.mgrid[0:size[0], 0:size[1]]
	gm1 = Gaussian2D(100, x_cent, y_cent, sigma, sigma)
	model_psf_final = gm1(x, y)

	### calculate the kernel:
	window = CosineBellWindow(alpha=alpha_cosbell)
	kernel = create_matching_kernel(model_psf_init, model_psf_final, window=window)

	return kernel



def ellipse_fit(data=None, init_x0=None, init_y0=None, init_sma=10.0, 
	init_ell=0.3, init_pa=45.0, rmax=30.0):
	"""Function for performing an elliptical aperture fitting to a galaxy in a particular band. 
	The aim is to get a suitable elliptical aperture around a certain radius (along the semi-major axis) of the galaxy.
	This function uses elliptical isophote analysis of the `photutils <https://photutils.readthedocs.io/en/stable/isophote.html>`_ package.

	:param data:
		Input of 2D array containing data of the image in a particular band.

	:param init_x0: (default: None)
		Initial estimate for the central coordinate in x-axis of the elliptical isophote. If None, the init_x0 is taken from the central coordinate of the image.  

	:param init_y0: (default: None)
		Initial estimate for the central coordinate in y-axis of the elliptical isophote. If None, the init_y0 is taken from the central coordinate of the image.

	:param init_sma: (default: 10.0)
		Initial radius in pixel (along the semi-major axis) for the initial guess in isophotal fitting. This is to be used for setting initial ellipse geometry in 
		the elliptical isophote fitting with `photutils <https://photutils.readthedocs.io/en/stable/isophote.html>`_.   

	:param init_ell: (default: 0.3)
		Initial ellipticity for the initial ellipse geometry.

	:param init_pa: (default: 45.0)
		Initial position angle for the initial ellipse geometry.

	:param rmax: (default: 30.0)
		Desired radius in pixel (along the semi-major axis) of the elliptical aperture. 

	:returns x0:
		Central coordinate in x-axis of the elliptical aperture.

	:returns y0:
		Central coordinate in y-axis of the elliptical aperture.

	:returns ell:
		Ellipticity of the elliptical aperture.

	:returns pa:
		Position angle of the elliptical aperture.
	"""
	
	from photutils import EllipticalAperture
	from photutils.isophote import EllipseGeometry
	from photutils.isophote import Ellipse

	## estimate central pixel:
	if init_x0 == None:
		init_x0 = (data.shape[1]-1)/2
	if init_y0 == None:
		init_y0 = (data.shape[0]-1)/2

	geometry = EllipseGeometry(x0=init_x0, y0=init_y0, sma=init_sma, eps=init_ell,
								pa=init_pa*np.pi/180.0)
	ellipse = Ellipse(data, geometry)
	isolist = ellipse.fit_image()

	nell = len(isolist.sma)
	if max(isolist.sma)>rmax:
		idx_sma = nell - 1
	else:
		abs_dist = np.absolute(isolist.sma - rmax)
		idx_sma, min_val = min(enumerate(abs_dist), key=operator.itemgetter(1))

	ell = isolist.eps[idx_sma]
	pa = isolist.pa[idx_sma]
	x0 = isolist.x0[idx_sma]
	y0 = isolist.y0[idx_sma]

	pa = (pa*180.0/math.pi) - 90.0

	return x0, y0, ell, pa


def draw_ellipse(x_cent,y_cent,a,e,pa):
	""" Function for producing x- and y- coordinates of a line associated witn an ellipse aperture. 

	:param x_cent:
		Central coordinate in x-axis of the ellipse.

	:param y_cent:
		Central coordinate in y-axis of the ellipse.

	:param a:
		Radius in pixel (along semi-major axis) of the ellipse.

	:param e:
		Ellipticity.

	:param pa:
		Position angle of the ellipse. This is measured in degree counterclockwise from the positive y-axis. 

	:returns ellipse_xy:
		2D array containing coordinates of pixels associated with the ellipse aperture.
		ellipse_xy[0] is the x coordinates, while ellipse_xy[1] is the y coordinates.
		To plot, do plt.plot(ellipse_xy[0],ellipse_xy[1]) with plt is `matplotlib.pyplot`.  
	"""

	# convert from degree to radian:
	pa = pa*math.pi/180.0
	x_temp = []
	y_temp = []
	count = 0
	# the positive x side:
	y0 = -1.0*a
	while y0<=a:
		x0 = (1.0-e)*math.sqrt((a*a) - (y0*y0))
		count = count + 1
		x_temp.append(x0)
		y_temp.append(y0)
		y0 = y0 + 0.05
	num_points = count
	# the negative x side:
	for ii in range(num_points,0,-1):
		x_temp.append(-1.0*x_temp[int(ii)-1])
		y_temp.append(y_temp[int(ii)-1])
        
	# store the ellipse's coordinates:
	ellipse_xy = []
	for xx in range(0,2):
		ellipse_xy.append([])
		for ii in range(0,2*num_points):
			if int(xx)==0:
				# transform to x-y plane:
				x0 = x_temp[int(ii)]*math.cos(pa) - y_temp[int(ii)]*math.sin(pa)
				ellipse_xy[int(xx)].append(x0+x_cent)
			elif int(xx)==1:
				y0 = x_temp[int(ii)]*math.sin(pa) + y_temp[int(ii)]*math.cos(pa)
				ellipse_xy[int(xx)].append(y0+y_cent)

	return ellipse_xy


def ellipse_sma(ell,pa,x_norm,y_norm):
	"""A function for calculating semi-major axes of pixels for a given ellipse configuration 
	defined by the ellipticity (ell) and position angle (pa)

	:param ell:
		The ellipticity of the ellipse.

	:param pa:
		The position angle of the ellipse.

	:param x_norm and y_norm:
		The pixels coordinates after subtracted with the central coorsinate (x_cent,y_cent).
		Typically: x_norm = x - x_cent and y_norm = y - y_cent

	:returns sma:
		Radii of the pixels (along the semi-major axis).  
	"""

	# convert to the x'-y' plane:
	x_norm_rot = np.asarray(x_norm)*math.cos(pa*math.pi/180.0) + np.asarray(y_norm)*math.sin(pa*math.pi/180.0)
	y_norm_rot = -1.0*np.asarray(x_norm)*math.sin(pa*math.pi/180.0) + np.asarray(y_norm)*math.cos(pa*math.pi/180.0)
	# get the semi-major axis of the pixel at x'-y' plane:
	sma = np.sqrt((y_norm_rot*y_norm_rot) + (x_norm_rot*x_norm_rot/(1.0-ell)/(1.0-ell)))
	return sma


def old_crop_ellipse_galregion(gal_region0,x_cent,y_cent,ell,pa,rmax):
	"""A function for cropping a galaxy's region within an ellipse aperture.
	The input should be in 2D array with values of either 1 (meaning 
	that the pixel is belong to the galaxy's region) or 0 (meaning 
	that the pixel is not belong to the galaxy's region)

	:param gal_region0:
		Original galaxy's region. Should be in 2D array as explained above

	:param x_cent and y_cent:
		The central coordinate of the galaxy

	:param ell:
		The ellipticity of the ellipse aperture used as reference in cropping

	:param pa:
		The position angle of the ellipse aperture used as reference in cropping

	:param rmax:
		The radius (along the semi-major axis) of tne ellipse aperture

	:returns gal_region:
		The cropped galaxy's region
	"""

	dim_y = gal_region0.shape[0]
	dim_x = gal_region0.shape[1]

	npixs = dim_y*dim_x
	pix_x_norm = np.zeros(npixs)
	pix_y_norm = np.zeros(npixs)
	idx = 0
	for yy in range(0,dim_y):
		for xx in range(0,dim_x):
			pix_x_norm[int(idx)] = xx - x_cent
			pix_y_norm[int(idx)] = yy - y_cent
			idx = idx + 1

	sma = ellipse_sma(ell,pa,pix_x_norm,pix_y_norm)
	gal_region = np.zeros((dim_y,dim_x))
	for ii in range(0,npixs):
		xx = pix_x_norm[int(ii)] + x_cent
		yy = pix_y_norm[int(ii)] + y_cent
		if gal_region0[int(yy)][int(xx)] == 1 and sma[ii]<=rmax:
			gal_region[int(yy)][int(xx)] = 1

	return gal_region


def crop_ellipse_galregion(gal_region0,x_cent,y_cent,ell,pa,rmax):
	"""A function for cropping a galaxy's region within an ellipse aperture.
	The input should be in 2D array with values of either 1 (meaning 
	that the pixel is belong to the galaxy's region) or 0 (meaning 
	that the pixel is not belong to the galaxy's region)

	:param gal_region0:
		Original galaxy's region. Should be in 2D array as explained above.

	:param x_cent:
		The central x coordinate of the galaxy.

	:param y_cent:
		The central y coordinate of the galaxy.

	:param ell:
		The ellipticity of the ellipse aperture used as reference in cropping.

	:param pa:
		The position angle of the ellipse aperture used as reference in cropping.

	:param rmax:
		The radius (along the semi-major axis) of tne ellipse aperture.

	:returns gal_region:
		The cropped galaxy's region.
	"""

	dim_y = gal_region0.shape[0]
	dim_x = gal_region0.shape[1]

	x = np.linspace(0,dim_x-1,dim_x)
	y = np.linspace(0,dim_y-1,dim_y)
	xx, yy = np.meshgrid(x,y)
	xx_norm, yy_norm = xx-x_cent, yy-y_cent

	data2D_sma = ellipse_sma(ell,pa,xx_norm,yy_norm)

	rows,cols = np.where((data2D_sma<=rmax) & (gal_region0==1))

	gal_region = np.zeros((dim_y,dim_x))
	gal_region[rows,cols] = 1

	return gal_region 


def crop_ellipse_galregion_fits(input_fits=None,x_cent=None,y_cent=None,
								ell=None,pa=None,rmax=25.0,name_out_fits=None):
	"""Function for cropping a galaxy's region within a desired ellipse aperture and produce a new FITS file.
	The input should be the FITS file of reduced maps of multiband fluxes (output of the :func:`flux_map` method in the :class:`images_processing` class).

	:param input_fits:
		Input FITS file containing the reduced maps of multiband fluxes. This FITS file must the output of the :func:`flux_map` method in the :class:`images_processing` class. 

	:param x_cent:
		Central coordinate in x-axis of the galaxy. If None, central coordinate of the galaxy is assumed to be the same as the central coordinate of the image.

	:param y_cent:
		Central coordinate in y-axis of the galaxy. If None, central coordinate of the galaxy is assumed to be the same as the central coordinate of the image.

	:param ell:
		Ellipticity of the ellipse aperture used as reference in cropping,

	:param pa:
		Position angle of the ellipse aperture used as reference in cropping.

	:param rmax:
		Radius (along the semi-major axis) of tne ellipse aperture used as reference in cropping.

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file. This is optional parameter.
	"""
	
	# open the input FITS file
	hdu = fits.open(input_fits)
	header = hdu[0].header
	gal_region0 = hdu['galaxy_region'].data
	map_flux0 = hdu['flux'].data
	map_flux_err0 = hdu['flux_err'].data
	stamp_img = hdu['stamp_image'].data 
	stamp_hdr = hdu['stamp_image'].header
	hdu.close()

	# get dimension
	dim_y = gal_region0.shape[0]
	dim_x = gal_region0.shape[1]

	# central coordinate of the galaxy
	if x_cent == None:
		x_cent = (dim_x-1)/2
	if y_cent == None:
		y_cent = (dim_y-1)/2

	# get modified galaxy's region
	gal_region = crop_ellipse_galregion(gal_region0,x_cent,y_cent,ell,pa,rmax) 

	# crop the flux maps following the modified gal_region
	nbands = int(header['nfilters'])
	map_flux = np.zeros((nbands,dim_y,dim_x))
	map_flux_err = np.zeros((nbands,dim_y,dim_x))
	for bb in range(0,nbands):
		for yy in range(0,dim_y):
			for xx in range(0,dim_x):
				if gal_region[int(yy)][int(xx)] == 1:
					map_flux[bb][int(yy)][int(xx)] = map_flux0[bb][int(yy)][int(xx)]
					map_flux_err[bb][int(yy)][int(xx)] = map_flux_err0[bb][int(yy)][int(xx)]

	# store to FITS file
	hdul = fits.HDUList()
	primary_hdu = fits.PrimaryHDU(header=header)
	hdul.append(primary_hdu)
	# add mask map to the HDU list
	hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
	# add fluxes maps to the HDU list
	hdul.append(fits.ImageHDU(map_flux, name='flux'))
	# add flux errors maps to the HDU list
	hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
	# add one of the stamp image (the first band):
	hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
	# write to fits file
	if name_out_fits == None:
		name_out_fits = 'crop_%s' % input_fits
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def crop_stars(gal_region0=[],x_cent=[],y_cent=[],radius=[]):
	"""A function for cropping foreground stars within a galaxy's region of interst.
	The input is aa galaxy region in 2D array with values of either 1 (meaning 
	that the pixel is belong to the galaxy's region) or 0 (meaning 
	that the pixel is not belong to the galaxy's region)   

	:param gal_region0:
		The 2D array of input galaxy's region.

	:param x_cent and y_cent:
		Arrays containing central coordinates of the stars.

	:param radius:
		Arrays containing the estimated radii of the stars.

	:returns gal_region:
		2D array containing output galaxy's region after stars subtraction. 
	"""
	nstars = len(x_cent)

	dim_y = gal_region0.shape[0]
	dim_x = gal_region0.shape[1]

	gal_region = gal_region0
	for yy in range(0,dim_y):
		for xx in range(0,dim_x):

			if gal_region[yy][xx] == 1:
				status_in = 0
				for ii in range(0,nstars):
					rad = math.sqrt((yy-y_cent[ii])**2.0 + (xx-x_cent[ii])**2.0)
					if rad<=radius[ii]:
						status_in = status_in + 1

				if status_in>0:
					gal_region[yy][xx] = 0

	return gal_region


def crop_stars_galregion_fits(input_fits=None,output_fits=None,x_cent=[],y_cent=[],radius=[]):
	"""A function for cropping foreground stars within a galaxy's region of interst.
	The input is a FITS file containing reduced multiband fluxes maps output of flux_map() function

	:param input_fits:
		The input FITS file

	:param output_fits:
		Desired name for the output FITS file

	:param x_cent and y_cent:
		Arrays containing cenral coordinates of the stars

	:param radius:
		Arrays containing the estimated radii of the stars.   
	"""

	# get the initial galaxy's region:
	hdu = fits.open(input_fits)
	header = hdu[0].header
	gal_region0 = hdu['galaxy_region'].data
	map_flux0 = hdu['flux'].data
	map_flux_err0 = hdu['flux_err'].data
	stamp_img = hdu['stamp_image'].data 
	stamp_hdr = hdu['stamp_image'].header
	hdu.close()

	# get modified galaxy's region:
	gal_region = crop_stars(gal_region0=gal_region0,x_cent=x_cent,y_cent=y_cent,radius=radius)

	# crop the flux maps following the modified gal_region:
	nbands = int(header['nfilters'])
	dim_y = gal_region0.shape[0]
	dim_x = gal_region0.shape[1]
	map_flux = np.zeros((nbands,dim_y,dim_x))
	map_flux_err = np.zeros((nbands,dim_y,dim_x))
	for bb in range(0,nbands):
		for yy in range(0,dim_y):
			for xx in range(0,dim_x):
				if gal_region[int(yy)][int(xx)] == 1:
					map_flux[bb][int(yy)][int(xx)] = map_flux0[bb][int(yy)][int(xx)]
					map_flux_err[bb][int(yy)][int(xx)] = map_flux_err0[bb][int(yy)][int(xx)]

	# store to fits file:
	hdul = fits.HDUList()
	primary_hdu = fits.PrimaryHDU(header=header)
	hdul.append(primary_hdu)
	# add mask map to the HDU list:
	hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
	# add fluxes maps to the HDU list:
	hdul.append(fits.ImageHDU(map_flux, name='flux'))
	# add flux errors maps to the HDU list:
	hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
	# add one of the stamp image (the first band):
	hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
	# write to fits file:
	if output_fits == None:
		output_fits = 'crop_%s' % input_fits
	hdul.writeto(output_fits, overwrite=True)

	return output_fits

def crop_square_region_fluxmap(input_fits=None,xrange=[],yrange=[],name_out_fits=None):

	# get the initial galaxy's region:
	hdu = fits.open(input_fits)
	header = hdu[0].header
	gal_region0 = hdu['galaxy_region'].data
	map_flux0 = hdu['flux'].data
	map_flux_err0 = hdu['flux_err'].data
	stamp_img = hdu['stamp_image'].data 
	stamp_hdr = hdu['stamp_image'].header
	hdu.close()

	# crop the maps:
	nbands = int(header['nfilters'])
	#dim_y = gal_region0.shape[0]
	#dim_x = gal_region0.shape[1]
	dim_y = int(yrange[1]) - int(yrange[0])
	dim_x = int(xrange[1]) - int(xrange[0])
	gal_region = np.zeros((dim_y,dim_x))
	map_flux = np.zeros((nbands,dim_y,dim_x))
	map_flux_err = np.zeros((nbands,dim_y,dim_x))
	for bb in range(0,nbands):
		idx_y = 0
		for yy in range(int(yrange[0]),int(yrange[1])):
			idx_x = 0
			for xx in range(int(xrange[0]),int(xrange[1])):
				gal_region[int(idx_y)][int(idx_x)] = gal_region0[yy][xx]
				map_flux[bb][int(idx_y)][int(idx_x)] = map_flux0[bb][yy][xx]
				map_flux_err[bb][int(idx_y)][int(idx_x)] = map_flux_err0[bb][yy][xx]
				idx_x = idx_x + 1
			idx_y = idx_y + 1

	# store to fits file:
	hdul = fits.HDUList()
	primary_hdu = fits.PrimaryHDU(header=header)
	hdul.append(primary_hdu)
	hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
	hdul.append(fits.ImageHDU(map_flux, name='flux'))
	hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
	hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))

	if name_out_fits == None:
		name_out_fits = 'crop_%s' % input_fits
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def crop_image_given_radec(img_name=None, ra=None, dec=None, stamp_size=[], name_out_fits=None):
	"""Function for cropping an image around a given position (RA, DEC)
	"""

	hdu = fits.open(img_name)[0]
	wcs = WCS(hdu.header)
	gal_x, gal_y = wcs.wcs_world2pix(ra, dec, 1)
	position = (gal_x,gal_y)
	cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
	hdu.data = cutout.data
	hdu.header.update(cutout.wcs.to_header())

	if name_out_fits == None:
		name_out_fits = 'crop_%s' % img_name

	hdu.writeto(name_out_fits, overwrite=True)
	print ("[produce %s]" % name_out_fits)


def crop_image_given_xy(img_name=None, x=None, y=None, stamp_size=[], name_out_fits=None):
	"""Function for cropping an image around a given position (x, y)
	"""

	hdu = fits.open(img_name)[0]
	wcs = WCS(hdu.header)
	position = (x,y)
	cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
	hdu.data = cutout.data
	hdu.header.update(cutout.wcs.to_header())

	if name_out_fits == None:
		name_out_fits = 'crop_%s' % img_name

	hdu.writeto(name_out_fits, overwrite=True)
	print ("[produce %s]" % name_out_fits)





