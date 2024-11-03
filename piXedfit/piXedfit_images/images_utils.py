import numpy as np 
from math import pi, pow, sqrt, cos, sin 
import sys, os
from astropy.io import fits 
from astropy.table import Table
from astropy.cosmology import *
import warnings
warnings.filterwarnings('ignore')

from ..utils.filtering import cwave_filters


__all__ = ["sort_filters", "kpc_per_pixel", "k_lmbd_Fitz1986_LMC", "EBV_foreground_dust", "skybg_sdss", "get_gain_dark_variance", "var_img_sdss", 
	   "var_img_GALEX", "var_img_2MASS", "var_img_WISE", "create_sci_var_HSC_images", "create_sci_var_VIRCAM_images",
	   "var_img_from_unc_img","var_img_from_weight_img", "mask_region_bgmodel", "subtract_background", "get_psf_fwhm",
	   "get_largest_FWHM_PSF", "ellipse_fit", "draw_ellipse", "ellipse_sma", "crop_ellipse_galregion", "crop_ellipse_galregion_fits", 
	   "crop_stars", "crop_stars_galregion_fits", "crop_image_given_radec", "segm_sep", "crop_image_given_xy", "check_avail_kernel", 
	   "create_kernel_gaussian", "crop_image", "create_psf_matching_kernel","raise_errors", "calc_pixsize", "get_img_pixsizes", 
	   "in_kernels", "get_flux_or_sb", "crop_2D_data", "remove_naninf_image_2dinterpolation", "check_dir", "add_dir", 
	   "check_name_remove_dir", "mapping_multiplots", "open_fluxmap_fits", "plot_maps_fluxes", "convert_flux_unit", 
	   "get_pixels_SED_fluxmap", "plot_SED_pixels", "get_total_SED", "get_curve_of_growth", "get_SNR_radial_profile", 
	   "plot_SNR_radial_profile", "get_flux_radial_profile", "photometry_within_aperture", "draw_aperture_on_maps_fluxes", 
	   "central_brightest_pixel", "curve_of_growth_psf", "rotate_pixels", "get_rectangular_region", "radial_profile_psf", 
	   "test_psfmatching_kernel", "remove_naninfzeroneg_image_2dinterpolation"]


def sort_filters(filters):
	photo_wave = cwave_filters(filters)
	id_sort = np.argsort(photo_wave)

	sorted_filters = []
	for ii in range(0,len(filters)):
		sorted_filters.append(filters[id_sort[ii]])

	return sorted_filters

def kpc_per_pixel(z,arcsec_per_pix,cosmo='flat_LCDM',H0=70.0,Om0=0.3):
	"""Function for calculating a physical scale (in kpc) corresponding to a single pixel in an image.

	:param z:
		Redshift of the galaxy.

	:param arsec_per_pix:
		Pixel size in arcsecond.

	:param cosmo:
		Choices for the cosmology. The options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5, (g)'Planck18' or 6.
		These are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0:
		The Hubble constant at z=0. Only relevant if cosmo='flat_LCDM'.

	:param Om0:
		The Omega matter at z=0.0. Only relevant if cosmo='flat_LCDM'.

	:returns kpc_per_pix:
		Output physical scale in kpc. 
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
			ax = 0.574*pow(inv_lmbd_micron,1.61)
			bx = -0.527*pow(inv_lmbd_micron,1.61)
			k = 3.1*ax + bx

	return k


def EBV_foreground_dust(ra, dec):
	"""Function for estimating E(B-V) dust attenuation due to the foreground Galactic dust attenuation at a given coordinate on the sky.

	:param ra:
		Right ascension coordinate in degree.

	:param dec:
		Declination coordinate in degree.

	:returns ebv:
		E(B-V) value.
	"""

	from astroquery.irsa_dust import IrsaDust
	import astropy.coordinates as coord
	import astropy.units as u

	coo = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
	table = IrsaDust.get_extinction_table(coo)

	Alambda_SDSS = np.zeros(5)
	Alambda_SDSS[0] = np.array(table['A_SandF'][table['Filter_name']=='SDSS u'])[0]
	Alambda_SDSS[1] = np.array(table['A_SandF'][table['Filter_name']=='SDSS g'])[0]
	Alambda_SDSS[2] = np.array(table['A_SandF'][table['Filter_name']=='SDSS r'])[0]
	Alambda_SDSS[3] = np.array(table['A_SandF'][table['Filter_name']=='SDSS i'])[0]
	Alambda_SDSS[4] = np.array(table['A_SandF'][table['Filter_name']=='SDSS z'])[0]

	# central wavelengths of the SDSS 5 bands: 
	filters = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']
	wave_SDSS = cwave_filters(filters)

	# calculate average E(B-V):
	ebv_SDSS = Alambda_SDSS/k_lmbd_Fitz1986_LMC(wave_SDSS)
	ebv = np.mean(ebv_SDSS)
	return ebv
    

def skybg_sdss(fits_image):
	"""A function for reconstructing background image of an SDSS image.
	A low resolution of background image is stored in HDU2 of an SDSS image.
	This function perform a bilinear interpolation to this sample background image. 

	:param fits_image:
		Input SDSS image. This image should be the corrected frame type provided in the SDSS website. 

	:returns output:
		A dictionary that contains background image and its name 
	"""

	from scipy import interpolate

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


def var_img_sdss(fits_image,filter_name,name_out_fits=None):
	"""A function for constructing a variance image from an input SDSS image

	:param fits_image:
		Input SDSS image (corrected frame type).

	:param filter_name:
		A string of filter name. Options are: 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', and 'sdss_z'.

	:returns name_out_fits:
		Name of output FITS file. If None, a generic name will be used.
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

	if name_out_fits is None:
		name_out_fits = 'var_%s' % fits_image
	fits.writeto(name_out_fits,sigma_sq_full,header,overwrite=True)

	return name_out_fits


def var_img_GALEX(sci_img,filter_name,name_out_fits=None):
	"""Function for calculating variance image from an input GALEX image

	:param sci_img:
		Input GALEX science image (i.e., background subtracted). 
		This type of image is provided in the GALEX website as indicated with "-intbgsub" (i.e., background subtracted intensity map).

	:param filter_name:
		A string of filter name. Options are: 'galex_fuv' and 'galex_nuv'.

	:param name_out_fits:
		Desired name for the output variance image. If None, a generic name will be used.
	"""

	# get science image:
	hdu = fits.open(sci_img)
	sci_img_data = hdu[0].data
	sci_img_header = hdu[0].header
	hdu.close()

	# get exposure time:
	exp_time = float(sci_img_header['EXPTIME'])

	if filter_name == 'galex_fuv':
		var_img_data = (0.05*0.05*sci_img_data*sci_img_data) + (np.absolute(sci_img_data)/exp_time)
	elif filter_name == 'galex_nuv':
		var_img_data = (0.027*0.027*sci_img_data*sci_img_data) + (np.absolute(sci_img_data)/exp_time)

	if name_out_fits is None:
		name_out_fits = 'var_%s' % sci_img
	fits.writeto(name_out_fits, var_img_data, sci_img_header, overwrite=True)

	return name_out_fits


def old_var_img_GALEX(sci_img,skybg_img,filter_name,name_out_fits=None):
	"""Function for calculating variance image from an input GALEX image

	:param sci_img:
		Input GALEX science image (i.e., background subtracted). 
		This type of image is provided in the GALEX website as indicated with "-intbgsub" (i.e., background subtracted intensity map). 

	:param skybg_img:
		Input sky background image .

	:param filter_name:
		A string of filter name. Options are: 'galex_fuv' and 'galex_nuv'.

	:param name_out_fits:
		Desired name for the output variance image. If None, a generic name will be used.
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

	rows, cols = np.where(val0==0.0)
	val0[rows,cols] = skybg_img_data[rows,cols]

	if filter_name == 'galex_fuv':
		sigma_sq_img_data = (np.absolute(val0*exp_time) + np.square(0.050*val0*exp_time))/exp_time/exp_time
	elif filter_name == 'galex_nuv':
		sigma_sq_img_data = (np.absolute(val0*exp_time) + np.square(0.027*val0*exp_time))/exp_time/exp_time
								
	if name_out_fits is None:
		name_out_fits = 'var_%s' % sci_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	#return sigma_sq_img_data
	return name_out_fits


def var_img_2MASS(sci_img,skyrms_img=None,skyrms_value=None,name_out_fits=None):
	"""Function for constructing a variance image from an input 2MASS image. 
	The estimation of flux uncertainty follows the information provided on the `2MASS website <http://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/#coadd>`_.

	:param sci_img:
		Input background-subtracted science image. 

	:param skyrms_img:
		FITS file of the RMS background image. If skyrms_value=None, this parameter should be provided. 
		The background subtraction and calculation of RMS image can be performed using the :func:`subtract_background` function.

	:param skyrms_value:
		Scalar value of median or mean of the RMS background image. This input will be considered when skyrms_img=None. 

	:param name_out_fits:
		Desired name for the output FITS file.
	"""

	# get science image:
	hdu = fits.open(sci_img)
	sci_img_data = hdu[0].data
	sci_img_header = hdu[0].header
	hdu.close()

	# typical gain and other coefficients for calculating flux error of a 2MASS image:
	gain_2mass = 10.0
	Nc = 6.0
	kc = 1.7

	# flux error: taken from http://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/#coadd
	SNR_l0 = sci_img_data/gain_2mass/Nc
	if skyrms_img is not None:
		hdu = fits.open(skyrms_img)
		skyrms_img_data = hdu[0].data
		hdu.close()
		SNR_l1 = 1.0*np.square(2.0*kc*skyrms_img_data)
		SNR_l2 = np.square(1.0*0.024*skyrms_img_data)
	else:
		SNR_l1 = 1.0*np.square(2.0*kc*skyrms_value)
		SNR_l2 = np.square(1.0*0.024*skyrms_value)
	sigma_sq_img_data = SNR_l0 + SNR_l1 + SNR_l2

	if name_out_fits is None:
		name_out_fits = 'var_%s' % sci_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	return name_out_fits


def var_img_WISE(sci_img,unc_img,filter_name,skyrms_img,name_out_fits=None):
	"""Function for constructing variance image from an input WISE image. 
	The uncertainty estimation follows the information from the `WISE website <http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html>`_

	:param sci_img:
		Input background-subtracted science image.

	:param unc_img:
		Input uncertainty image. This type of WISE image is provided in the 
		`IRSA website <https://irsa.ipac.caltech.edu/applications/wise/>`_ and indicated with '-unc-' keyword.

	:param filter_name: 		
		The filter name. Options are: 'wise_w1', 'wise_w2', 'wise_w3', and 'wise_w4'

	:param skyrms_img:
		Input RMS background image. This image is produced in the background subtraction with the :func:`subtract_background` function. 

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file. If None, a generic name will be made.
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
	sigma_src = np.sqrt(Fcorr*(np.square(sigma_i) + (0.5*pi*sigma_B*sigma_B)))
	sigma_sq_img_data = np.square(sci_img_data)*((sigma_0*sigma_0/f_0/f_0) + (0.8483*sigma_magzp*sigma_magzp)) + np.square(sigma_src) ## in unit of DN

	if name_out_fits is None:
		name_out_fits = 'var_%s' % unc_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	return name_out_fits


def create_sci_var_HSC_images(filters, images={}, dir_images=None):
	# get number of filters
	nbands = len(filters)

	# get central wavelength of the filters
	photo_wave = cwave_filters(filters)

	# magnitude zero-point
	magzp = 27.0

	# directory of images
	if dir_images is None:
		dir_images = './'

	for bb in range(0,nbands):
		# open input image
		hdu = fits.open(dir_images+images[filters[bb]])
		sci_img_data0 = hdu[1].data
		header_sci = hdu[1].header 
		var_img_data0 = hdu[3].data 
		header_var = hdu[3].header
		hdu.close()

		##=> convert the pixel values into erg/s/cm2/A
		factor = np.power(10.0,-0.4*(48.6+magzp))*2.998e+18/photo_wave[bb]/photo_wave[bb]

		# science image
		sci_img_data = sci_img_data0*factor

		# variance image
		var_img_data = factor*factor*var_img_data0

		# produce FITS file for science image
		name_out_fits = 'sci_%s' % images[filters[bb]].replace('.gz','')
		fits.writeto(name_out_fits, sci_img_data, header_sci, overwrite=True)
		print ('produced '+name_out_fits)

		# produce FITS file for variance image
		name_out_fits = 'var_%s' % images[filters[bb]].replace('.gz','')
		fits.writeto(name_out_fits, var_img_data, header_var, overwrite=True)
		print ('produced '+name_out_fits)


def create_sci_var_VIRCAM_images(filters, images={}, weights={}, dir_images=None):
	# get number of filters
	nbands = len(filters)

	# get central wavelength of filters
	photo_wave = cwave_filters(filters)

	# directory of images
	if dir_images is None:
		dir_images = './'

	for bb in range(0,nbands):
		# AB vs Vega excess: based on information from http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/filter-set
		# and https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
		if filters[bb] == 'vircam_z':
			dmag = 0.502
		elif filters[bb] == 'vircam_y':
			dmag = 0.600
		elif filters[bb] == 'vircam_j':
			dmag = 0.916
		elif filters[bb] == 'vircam_h':
			dmag = 1.366
		elif filters[bb] == 'vircam_ks':
			dmag = 1.827

		#==> open science image
		hdu = fits.open(dir_images+images[filters[bb]])
		sci_img_data0 = hdu[1].data
		header_sci = hdu[1].header
		magzp = float(hdu[1].header['MAGZPT'])
		hdu.close()

		# conversion to erg/s/cm2/A
		factor = np.power(10.0,-0.4*(48.6+magzp+dmag))*2.998e+18/photo_wave[bb]/photo_wave[bb]
		sci_img_data = sci_img_data0*factor

		#==> open weight image (i.e., inverse variance)
		hdu = fits.open(dir_images+weights[filters[bb]])
		wht_img_data = hdu[1].data 
		header_var = hdu[1].header
		magzp = float(hdu[1].header['MAGZPT'])
		hdu.close()

		# Calculate variance image in units of erg/s/cm2/A square
		factor = np.power(10.0,-0.4*(48.6+magzp+dmag))*2.998e+18/photo_wave[bb]/photo_wave[bb]
		var_img_data = factor*factor/wht_img_data

		# produce FITS file for science image
		name_out_fits = 'sci_%s' % images[filters[bb]].replace('.gz','')
		fits.writeto(name_out_fits, sci_img_data, header_sci, overwrite=True)
		print ('produced '+name_out_fits)

		# produce FITS file for variance image
		name_out_fits = 'var_%s' % weights[filters[bb]].replace('.gz','')
		fits.writeto(name_out_fits, var_img_data, header_var, overwrite=True)
		print ('produced '+name_out_fits)


def var_img_from_unc_img(unc_image, name_out_fits=None):
	"""Function for constructing a variance image from an input of uncertainty image.
	This function simply takes square of the uncertainty image and store it into a new FITS file while retaining the header information.

	:param unc_img:
		Input uncertainty image.

	:param name_out_fits: (optional, default: None)
		Name of output FITS file. If None, a generic name will be generated.
	"""

	hdu = fits.open(unc_image)
	header = hdu[0].header
	data_unc_image = hdu[0].data
	hdu.close()

	var_image = np.square(data_unc_image)

	# store to fits file:
	if name_out_fits is None:
		name_out_fits = "var_%s" % unc_image
	fits.writeto(name_out_fits, var_image, header=header, overwrite=True)

	return name_out_fits


def var_img_from_weight_img(wht_image, name_out_fits=None):
	"""Function for constructing a variance image from an input weight (i.e., inverse variance) image.
	This funciton simply takes inverse of the weight image and store it into a new FITS file while 
	retaining the header information.

	:param wht_image:
		Input of weight image (i.e., inverse variance).

	:returns name_out_fits: (optional, default: None)
		Name of output FITS file. If None, a generic name will be made.
	"""
	hdu = fits.open(wht_image)
	header = hdu[0].header
	data_image = hdu[0].data
	hdu.close()

	var_data = 1.0/np.absolute(data_image)

	# store into fits file:
	if name_out_fits is None:
		name_out_fits = "var_%s" % wht_image
	fits.writeto(name_out_fits, var_data, header=header, overwrite=True)

	return name_out_fits


def segm_sep(fits_image=None, thresh=1.5, var=None, minarea=5, deblend_nthresh=32, deblend_cont=0.005):
	import sep 

	hdu = fits.open(fits_image)
	data_img = hdu[0].data 
	hdu.close()

	data_img = data_img.byteswap(inplace=True).newbyteorder()

	if var is None:
		rows,cols = np.where((np.isnan(data_img)==False) & (np.isinf(data_img)==False))
		err = np.percentile(data_img[rows,cols], 2.5)
	else:
		hdu = fits.open(var)
		data_var = hdu[0].data 
		hdu.close()

		data_var = data_var.byteswap(inplace=True).newbyteorder()
		err = np.sqrt(data_var)


	objects, segm_map = sep.extract(data=data_img, thresh=thresh, err=err, minarea=minarea, 
									deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, 
									segmentation_map=True)

	return segm_map


def mask_region_bgmodel(fits_image=None, thresh=1.5, var=None, minarea=5, deblend_nthresh=32, deblend_cont=0.005):

	segm_map = segm_sep(fits_image=fits_image, thresh=thresh, var=var, minarea=minarea, 
						deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont)

	dim_y = segm_map.shape[0]
	dim_x = segm_map.shape[1]
	mask_region = np.zeros((dim_y,dim_x))

	rows, cols = np.where(segm_map>0)
	mask_region[rows,cols] = 1

	return mask_region


def subtract_background(fits_image, hdu_idx=0, sigma=3.0, box_size=None, mask_region=None, mask_sources=True, 
	var_image=None, thresh=1.5, minarea=5, deblend_nthresh=32, deblend_cont=0.005):
	"""Function for estimating 2D background and subtracting it from the input image. This function also produce RMS image. 
	This function adopts the Background2D function from the photutils package. To estimate 2D background, 
	the input image is gridded and sigma clipping is done to each bin (grid). Then 2D interpolation is performed to construct the 2D background image. 
	A more information can be seen at `photutils <https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html#photutils.background.Background2D>`_ website.  

	:param fits_image:
		Input image.

	:param hdu_idx:
		The extension (HDU) where the image is stored. Default is 0 (HDU0).

	:param sigma: 
		Sigma clipping threshold value.

	:param box_size:
		The box size for the image gridding. The format is: [ny, nx]. If None, both axes will be divided into 10 grids. 

	:param mask_region: 
		Region within the image that are going to be excluded. 
		mask_region should be 2D array with the same size as the input image.

	:param mask_sources: 
		If True, source detection and segmentation will be performed with SEP (Pyhton version of SExtractor) 
		and the regions associated with the detected sources will be excluded. This help reducing contamination by astronomical sources.

	:param var_image: 
		Variance image (in FITS file format) to be used in the sources detection process. This input argument is only relevant if mask_sources=True.

	:param thresh: 
		Detection threshold for the sources detection. If variance image is supplied, the threshold value for a given pixel is 
		interpreted as a multiplicative factor of the uncertainty (i.e. square root of the variance) on that pixel. 
		If var_image=None, the threshold is taken to be 2.5 percentile of the pixel values in the image. 

	:param minarea:
		Minimum number of pixels above threshold triggering detection. 

	:param deblend_nthresh:
		Number of thresholds used for object deblending.

	:param deblend_cont:
		Minimum contrast ratio used for object deblending. To entirely disable deblending, set to 1.0.
	"""

	from astropy.stats import SigmaClip
	from photutils import Background2D, MedianBackground

	# open the input image:
	hdu = fits.open(fits_image)
	data_image = hdu[int(hdu_idx)].data
	header = hdu[int(hdu_idx)].header
	hdu.close()

	# define box size: depending on the dimension of the image:
	dim_x = data_image.shape[1]
	dim_y = data_image.shape[0]
	if box_size is None:
		box_size = [int(dim_y/10),int(dim_x/10)]
	elif box_size != None:
		box_size = box_size

	if mask_sources==True or mask_sources==1:
		mask_region0 = mask_region_bgmodel(fits_image=fits_image, thresh=thresh, var=var_image, 
											minarea=minarea, deblend_nthresh=deblend_nthresh, 
											deblend_cont=deblend_cont)

		if mask_region is None:
			mask_region1 = mask_region0
		else:
			if mask_region.shape[0]!=dim_y or mask_region.shape[1]!=dim_x:
				print ("dimension of mask_region should be the same with the dimension of fits_image!")
				sys.exit()
			else:
				mask_region1 = np.zeros((dim_y,dim_x)).astype(int)
				rows, cols = np.where((mask_region0==1) | (mask_region==1))
				mask_region1[rows,cols] = 1

		sigma_clip = SigmaClip(sigma=sigma)
		bkg_estimator = MedianBackground()
		bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), mask=mask_region1,
							sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

	elif mask_sources==False or mask_sources==0:
		if mask_region is None:
			sigma_clip = SigmaClip(sigma=sigma)
			bkg_estimator = MedianBackground()
			bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

		else:
			sigma_clip = SigmaClip(sigma=sigma)
			bkg_estimator = MedianBackground()
			bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), mask=mask_region, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


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


def unknown_images(filters):
	list_filters = ['galex_fuv', 'galex_nuv', 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i',
					'sdss_z', '2mass_j', '2mass_h', '2mass_k', 'spitzer_irac_36', 'spitzer_irac_45',
					'spitzer_irac_58', 'spitzer_irac_80', 'spitzer_mips_24', 'spitzer_mips_70',
					'spitzer_mips_160', 'herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160',
					'herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500', 'wise_w1', 'wise_w2',
					'wise_w3', 'wise_w4']
	unknown = []
	for fil in filters:
		if (fil in list_filters) == False:
			unknown.append(fil)

	return unknown 


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


def get_largest_FWHM_PSF(filters=None):
	"""A function to find a band that has largest PSF size
	"""
	from operator import itemgetter

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
	fil_fwhm = np.zeros(nbands)
	for bb in range(0,nbands):
		fil_fwhm[bb] = col_fwhm_psf[filters[bb]]

	idx_fil_max, max_val = max(enumerate(fil_fwhm), key=itemgetter(1))

	return idx_fil_max


def calc_pixsize(fits_image):
	""" Function to get pixel size of an image

	:param fits_image:
		Input image.

	:returns pixsize_arcsec:
		Pixel size in arc second.
	"""

	from astropy.wcs.utils import proj_plane_pixel_area
	from astropy.wcs import WCS

	h = fits.open(fits_image)
	w = WCS(h[0].header)
	area = proj_plane_pixel_area(w)
	pixsize_deg = np.sqrt(area)
	pixsize_arcsec = pixsize_deg*3600.0
	h.close()
	
	return pixsize_arcsec


def get_img_pixsizes(img_pixsizes,filters,sci_img,flux_or_sb,flag_psfmatch,flag_reproject):

	img_pixsizes1 = {}

	if img_pixsizes is not None:
		for bb in range(0,len(filters)):
			if filters[bb] in img_pixsizes:
				img_pixsizes1[filters[bb]] = img_pixsizes[filters[bb]]
			else:
				if flag_psfmatch==1 and flag_reproject==1 and flux_or_sb[filters[bb]]==0:
					img_pixsizes1[filters[bb]] = -99.0
				else:
					img_pixsizes1[filters[bb]] = calc_pixsize(sci_img[filters[bb]])
	else:
		for bb in range(0,len(filters)):
			if flag_psfmatch==1 and flag_reproject==1 and flux_or_sb[filters[bb]]==0:
				img_pixsizes1[filters[bb]] = -99.0
			else:
				img_pixsizes1[filters[bb]] = calc_pixsize(sci_img[filters[bb]])

	return img_pixsizes1


def raise_errors(filters, kernels, flag_psfmatch, img_unit, img_scale):
	unknown = unknown_images(filters)
	if len(unknown)>0:
		if kernels is None and flag_psfmatch==0:
			print ("PSF matching kernels for the following filters are not available by default. In this case, input kernels should be supplied!")
			print (unknown)
			sys.exit()

		if img_unit is None or img_scale is None:
			print ("Unit of the pixel values of the following imaging data are not recognized. In this case, input img_unit and img_scale should be provided!")
			print (unknown)
			sys.exit()

def in_kernels(kernels,sorted_filters):
	kernels1 = {}
	if kernels is None:
		for ii in range(0,len(sorted_filters)):
			kernels1[sorted_filters[ii]] = None
	else:
		for ii in range(0,len(sorted_filters)):
			if sorted_filters[ii] in kernels:
				kernels1[sorted_filters[ii]] = kernels[sorted_filters[ii]]
			else:
				kernels1[sorted_filters[ii]] = None

	return kernels1

def get_flux_or_sb(filters,img_unit):
	flux_or_sb0 = {}
	flux_or_sb0['galex_fuv'] = 0
	flux_or_sb0['galex_nuv'] = 0
	flux_or_sb0['sdss_u'] = 0
	flux_or_sb0['sdss_g'] = 0
	flux_or_sb0['sdss_r'] = 0
	flux_or_sb0['sdss_i'] = 0
	flux_or_sb0['sdss_z'] = 0
	flux_or_sb0['2mass_j'] = 0
	flux_or_sb0['2mass_h'] = 0
	flux_or_sb0['2mass_k'] = 0
	flux_or_sb0['wise_w1'] = 0
	flux_or_sb0['spitzer_irac_36'] = 1
	flux_or_sb0['spitzer_irac_45'] = 1
	flux_or_sb0['wise_w2'] = 0
	flux_or_sb0['spitzer_irac_58'] = 1
	flux_or_sb0['spitzer_irac_80'] = 1
	flux_or_sb0['wise_w3'] = 0
	flux_or_sb0['wise_w4'] = 0
	flux_or_sb0['spitzer_mips_24'] = 1
	flux_or_sb0['herschel_pacs_70'] = 0
	flux_or_sb0['herschel_pacs_160'] = 0
	flux_or_sb0['herschel_spire_250'] = 1
	flux_or_sb0['herschel_spire_350'] = 1

	flux_or_sb = {}
	for bb in range(0,len(filters)):
		if filters[bb] in flux_or_sb0:
			flux_or_sb[filters[bb]] = flux_or_sb0[filters[bb]]
		else:
			if filters[bb] in img_unit:
				if img_unit[filters[bb]] == 'erg/s/cm2/A' or img_unit[filters[bb]] == 'Jy':
					flux_or_sb[filters[bb]] = 0
				elif img_unit[filters[bb]] == 'MJy/sr':
					flux_or_sb[filters[bb]] =  1
				else:
					print ("Inputted img_unit[%s] is not recognized!" % filters[bb])
					sys.exit()
			else:
				print ("Input img_unit is required for this imaging data!")
				sys.exit()

	return flux_or_sb


def create_kernel_gaussian(psf_fwhm_init=None, psf_fwhm_final=None, alpha_cosbell=1.5, pixsize_PSF_target=0.25, size=[101,101]):

	from astropy.modeling.models import Gaussian2D
	from photutils import CosineBellWindow, create_matching_kernel

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


def crop_image(data, new_dimx, new_dimy):
	dimy, dimx = data.shape[0], data.shape[1]
	#x_cent, y_cent = (dimx-1)/2, (dimy-1)/2
	y_cent, x_cent = np.unravel_index(data.argmax(), data.shape)

	delx, dely = (new_dimx-1)/2, (new_dimy-1)/2

	row_start, row_end = y_cent-dely, y_cent+dely+1
	col_start, col_end = x_cent-delx, x_cent+delx+1

	new_data = data[int(row_start):int(row_end), int(col_start):int(col_end)]

	return new_data


def create_psf_matching_kernel(init_PSF_name, target_PSF_name, pixscale_init_PSF, 
	pixscale_target_PSF, window='top_hat', window_arg=1.0):
	"""A function for creating convolution kernel for PSF matching given initial and target PSFs.

	:param init_PSF_name:
		Image of input/initial PSF.

	:param target_PSF_name:
		Image of target PSF.

	:param pixscale_init_PSF:
		Pixel size of the initial PSF in arcsec.

	:param pixscale_target_PSF:
		Pixel size of the target PSF in arcsec.

	:param window:
		Options are 'top_hat' and 'cosine_bell'.
	
	:param window_arg:
		Coefficient value of the window function, following Photutils. 

	:param kernel:
		The data of convolution kernel in 2D array. 
	"""

	from photutils import CosineBellWindow, TopHatWindow, create_matching_kernel
	from photutils.psf.matching import resize_psf

	init_PSF = fits.open(init_PSF_name)[0].data
	target_PSF = fits.open(target_PSF_name)[0].data

	# resize PSFs
	pixscale1 = 0
	if pixscale_init_PSF>pixscale_target_PSF:
		init_PSF = resize_psf(init_PSF, pixscale_init_PSF, pixscale_target_PSF, order=3)
		pixscale1 = pixscale_target_PSF
	elif pixscale_init_PSF<pixscale_target_PSF:
		target_PSF = resize_psf(target_PSF, pixscale_target_PSF, pixscale_init_PSF, order=3)
		pixscale1 = pixscale_init_PSF

	if init_PSF.shape[0]>target_PSF.shape[0]:
		init_PSF = crop_image(init_PSF, target_PSF.shape[1], target_PSF.shape[0])
	elif init_PSF.shape[0]<target_PSF.shape[0]:
		target_PSF = crop_image(target_PSF, init_PSF.shape[1], init_PSF.shape[0])
	
	if window == 'cosine_bell':
		window = CosineBellWindow(alpha=window_arg)
	elif window == 'top_hat':
		window = TopHatWindow(window_arg)
	else:
		print ('Window type is not recognized!')
		sys.exit()

	kernel = create_matching_kernel(init_PSF, target_PSF, window=window)

	if pixscale1 > 0:
		kernel = resize_psf(kernel, pixscale1, pixscale_init_PSF, order=3)

	return kernel


def radial_profile_psf(psf, pixsize, e=0.0, pa=45.0, dr_arcsec=None):
	"""
	Function to get the radial profile of PSF. 

	:param psf:
		Input PSF, either in a FITS file or the data array (2D).

	:pixsize:
		Pixel size in arc second.

	:param e: (default = 0)
		Ellipticity of the apertures. The default is zero (i.e., circular).

	:param pa:
		The position angle of the elliptical apertures.

	:param dr_arsec:
		Radial increment in units of arc second.

	:returns curve_rad:
		The radius array of the PSF radial profile.

	:returns curve_val:
		The normalized fluxes of the PSF radial profile.
	"""

	if dr_arcsec is None:
		dr_arcsec = pixsize

	if isinstance(psf, str) == True:
		data_psf = fits.open(psf)[0].data
	else:
		data_psf = psf

	dimy, dimx = data_psf.shape[0], data_psf.shape[1]

	## normalize
	data_psf = data_psf/np.sum(data_psf)

	#x_cent, y_cent = (dimx-1.0)/2.0, (dimy-1.0)/2.0
	y_cent, x_cent = np.unravel_index(data_psf.argmax(), data_psf.shape)

	x = np.linspace(0,dimx-1,dimx)
	y = np.linspace(0,dimy-1,dimy)
	xx, yy = np.meshgrid(x,y)
	xx_norm, yy_norm = xx-x_cent, yy-y_cent

	data2D_sma = ellipse_sma(e,pa,xx_norm,yy_norm)*pixsize     # in unit of arcsec

	for xx in range(0,round(x_cent)):
		for yy in range(0,dimy):
			data2D_sma[yy][xx] = -1.0*data2D_sma[yy][xx]

	curve_rad = []
	curve_val = []

	r1 = np.min(data2D_sma)
	r2 = r1 + dr_arcsec 
	while r2<=np.max(data2D_sma):
		rows, cols = np.where((data2D_sma>=r1) & (data2D_sma<r2))

		curve_rad.append(0.5*(r1+r2))
		curve_val.append(np.mean(data_psf[rows,cols]))

		r1 = r2
		r2 = r2 + dr_arcsec

	curve_rad = np.asarray(curve_rad)
	curve_val = np.asarray(curve_val)

	# normalize
	curve_val = curve_val/max(curve_val)
	
	return curve_rad, curve_val


def curve_of_growth_psf(psf, e=0.0, pa=45.0, dr=1.0):
	"""
	:param dr:
		Radial increment in units of pixel.
	"""

	hdu = fits.open(psf)
	data_psf = hdu[0].data
	dimy, dimx = data_psf.shape[0], data_psf.shape[1]
	hdu.close()

	## normalize
	data_psf = data_psf/np.sum(data_psf)

	#x_cent, y_cent = (dimx-1.0)/2.0, (dimy-1.0)/2.0
	y_cent, x_cent = np.unravel_index(data_psf.argmax(), data_psf.shape)

	x = np.linspace(0,dimx-1,dimx)
	y = np.linspace(0,dimy-1,dimy)
	xx, yy = np.meshgrid(x,y)
	xx_norm, yy_norm = xx-x_cent, yy-y_cent

	data2D_sma = ellipse_sma(e,pa,xx_norm,yy_norm)

	curve_rad = []
	curve_val = []

	r1 = 0
	r2 = r1 + dr 
	cumul_val = 0
	while r2<=np.max(data2D_sma):
		rows, cols = np.where((data2D_sma>=r1) & (data2D_sma<r2))

		curve_rad.append(0.5*(r1+r2))
		cumul_val = cumul_val + np.sum(data_psf[rows,cols])
		curve_val.append(cumul_val)

		r1 = r2
		r2 = r2 + dr

	curve_rad = np.asarray(curve_rad)
	curve_val = np.asarray(curve_val)
	
	return curve_rad, curve_val


def test_psfmatching_kernel(init_PSF_name, target_PSF_name, kernel, pixscale_init_PSF, pixscale_target_PSF, 
	dr_arcsec=None, xrange_arcsec=[-0.5,0.5], savefig=False, name_plot=None):
	""" Function for testing the convolution kernel. This includes convolving the initial PSF with the kernel,
	comparing the radial profiles of the convolved initial PSF and the target PSF along with the original initial PSF.

	:param init_PSF_name:
		Image of the initial PSF.

	:param target_PSF_name:
		Image of the target PSF.

	:param kernel:
		The convolution kernel data (2D array).

	:param pixscale_init_PSF:
		Pixel size of the initial PSF in arcsec.

	:param pixscale_target_PSF:
		Pixel size of the target PSF in arcsec.

	:param dr_arsec:
		Radial increment for the radial profile in arcsec.

	:param xrange_arcsec:
		Range of x-axis in arsec.

	:param savefig:
		Decide whether to save the plot or not.

	:param name_plot:
		Name for the output plot.
	"""

	from astropy.convolution import convolve_fft
	import matplotlib.pyplot as plt

	psfmatch_init_psf = convolve_fft(fits.open(init_PSF_name)[0].data, kernel, allow_huge=True)


	fig1 = plt.figure(figsize=(8,6))
	f1 = plt.subplot()
	plt.xlim(xrange_arcsec[0],xrange_arcsec[1])

	curve_rad, curve_val = radial_profile_psf(init_PSF_name, pixscale_init_PSF, dr_arcsec=dr_arcsec)
	plt.plot(curve_rad, curve_val, lw=2, label='initial PSF', color='black')

	curve_rad, curve_val = radial_profile_psf(target_PSF_name, pixscale_target_PSF, dr_arcsec=dr_arcsec)
	plt.plot(curve_rad, curve_val, lw=2, label='target PSF', color='red')

	curve_rad, curve_val = radial_profile_psf(psfmatch_init_psf, pixscale_init_PSF, dr_arcsec=dr_arcsec)
	plt.plot(curve_rad, curve_val, lw=2, label='test', color='blue')

	plt.legend(fontsize=15)

	if savefig is True:
		if name_plot is None:
			name_plot = 'test_psfmatching_kernel.png'
		plt.savefig(name_plot)
	else:
		plt.show()


def ellipse_fit(data=None, init_x0=None, init_y0=None, init_sma=10.0, 
	init_ell=0.3, init_pa=45.0, rmax=30.0):
	"""Function for performing an elliptical aperture fitting to a galaxy in a particular band. 
	The aim is to get a suitable elliptical aperture around a certain radius (along the semi-major axis) of the galaxy.
	This function uses elliptical isophote analysis of the `photutils <https://photutils.readthedocs.io/en/stable/isophote.html>`_ package.

	:param data:
		Input of 2D array containing data of the image in a particular band.

	:param init_x0: 
		Initial estimate for the central coordinate in x-axis of the elliptical isophote. If None, the init_x0 is taken from the central coordinate of the image.  

	:param init_y0: 
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
	from operator import itemgetter

	## estimate central pixel:
	if init_x0 is None:
		init_x0 = (data.shape[1]-1)/2
	if init_y0 is None:
		init_y0 = (data.shape[0]-1)/2

	geometry = EllipseGeometry(x0=init_x0, y0=init_y0, sma=init_sma, eps=init_ell,
								pa=init_pa*pi/180.0)
	ellipse = Ellipse(data, geometry)
	isolist = ellipse.fit_image()

	nell = len(isolist.sma)
	if max(isolist.sma)>rmax:
		idx_sma = nell - 1
	else:
		abs_dist = np.absolute(isolist.sma - rmax)
		idx_sma, min_val = min(enumerate(abs_dist), key=itemgetter(1))

	ell = isolist.eps[idx_sma]
	pa = isolist.pa[idx_sma]
	x0 = isolist.x0[idx_sma]
	y0 = isolist.y0[idx_sma]

	pa = (pa*180.0/pi) - 90.0

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
	pa = pa*pi/180.0
	x_temp = []
	y_temp = []
	count = 0
	# the positive x side:
	y0 = -1.0*a
	while y0<=a:
		x0 = (1.0-e)*sqrt((a*a) - (y0*y0))
		count = count + 1
		x_temp.append(x0)
		y_temp.append(y0)
		y0 = y0 + 0.05
	num_points = count
	# the negative x side:
	for ii in range(num_points,0,-1):
		x_temp.append(-1.0*x_temp[ii-1])
		y_temp.append(y_temp[ii-1])
        
	# store the ellipse's coordinates:
	ellipse_xy = []
	for xx in range(0,2):
		ellipse_xy.append([])
		for ii in range(0,2*num_points):
			if xx==0:
				# transform to x-y plane:
				x0 = x_temp[ii]*cos(pa) - y_temp[ii]*sin(pa)
				ellipse_xy[xx].append(x0+x_cent)
			elif int(xx)==1:
				y0 = x_temp[ii]*sin(pa) + y_temp[ii]*cos(pa)
				ellipse_xy[xx].append(y0+y_cent)

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
	x_norm, y_norm = np.asarray(x_norm), np.asarray(y_norm)

	x_norm_rot = x_norm*np.cos(pa*np.pi/180.0) + y_norm*np.sin(pa*np.pi/180.0)
	y_norm_rot = -1.0*x_norm*np.sin(pa*np.pi/180.0) + y_norm*np.cos(pa*np.pi/180.0)
	sma = np.sqrt((y_norm_rot*y_norm_rot) + (x_norm_rot*x_norm_rot/(1.0-ell)/(1.0-ell)))

	return sma


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


def crop_ellipse_galregion_fits(input_fits,x_cent=None,y_cent=None,
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
	if x_cent is None:
		x_cent = (dim_x-1)/2
	if y_cent is None:
		y_cent = (dim_y-1)/2

	# get modified galaxy's region
	gal_region = crop_ellipse_galregion(gal_region0,x_cent,y_cent,ell,pa,rmax)

	# number of filters
	nbands = int(header['nfilters'])

	rows, cols = np.where(gal_region==1)

	map_flux = np.zeros((nbands,dim_y,dim_x))
	map_flux_err = np.zeros((nbands,dim_y,dim_x))
	for bb in range(0,nbands):
		map_flux[bb][rows,cols] = map_flux0[bb][rows,cols]
		map_flux_err[bb][rows,cols] = map_flux_err0[bb][rows,cols]

	# store to FITS file
	hdul = fits.HDUList()
	hdul.append(fits.ImageHDU(data=map_flux, header=header, name='flux'))
	hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
	hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
	hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
	
	# write to fits file
	if name_out_fits is None:
		name_out_fits = 'crop_%s' % input_fits
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def crop_stars(gal_region=[],x_cent=[],y_cent=[],radius=[]):
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

	dim_y = gal_region.shape[0]
	dim_x = gal_region.shape[1]

	x = np.linspace(0,dim_x-1,dim_x)
	y = np.linspace(0,dim_y-1,dim_y)
	xx, yy = np.meshgrid(x,y)

	for ii in range(0,nstars):
		xx_norm, yy_norm = xx-x_cent[ii], yy-y_cent[ii]
		data2D_rad = np.sqrt(np.square(xx_norm) + np.square(yy_norm))

		rows, cols = np.where((data2D_rad<=radius[ii]) & (gal_region==1))
		gal_region[rows,cols] = 0

	return gal_region


def crop_stars_galregion_fits(input_fits,x_cent,y_cent,radius,name_out_fits=None):
	"""Function for cropping foreground stars within a galaxy's region of interst.

	:param input_fits:
		The input FITS file containing the data cube that is produced using the :func:`flux_map` funciton in the :class:`images_processing` class.

	:param x_cent:
		1D array of x coordinates of the center of the stars.

	:param y_cent:
		1D array of y coordinates of the center of the stars.

	:param radius:
		1D array of the estimated circular radii of the stars. 

	:param name_out_fits:
		Desired name for the output FITS file. If None, a generic name will be made.  
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

	# number of filters
	nbands = int(header['nfilters'])

	# dimension
	dim_y = gal_region0.shape[0]
	dim_x = gal_region0.shape[1]

	# get modified galaxy's region:
	gal_region = crop_stars(gal_region=gal_region0,x_cent=x_cent,y_cent=y_cent,radius=radius)

	rows, cols = np.where(gal_region==1)

	map_flux = np.zeros((nbands,dim_y,dim_x))
	map_flux_err = np.zeros((nbands,dim_y,dim_x))
	for bb in range(0,nbands):
		map_flux[bb][rows,cols] = map_flux0[bb][rows,cols]
		map_flux_err[bb][rows,cols] = map_flux_err0[bb][rows,cols]

	# store to fits file:
	hdul = fits.HDUList()
	hdul.append(fits.ImageHDU(data=map_flux, header=header, name='flux'))
	hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
	hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
	hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
	
	# write to fits file:
	if name_out_fits is None:
		name_out_fits = 'crop_%s' % input_fits
	hdul.writeto(name_out_fits, overwrite=True)

	#return output_fits


def crop_image_given_radec(image, ra, dec, idx_hdu=0, stamp_size=[], name_out_fits=None):
	"""Function for cropping an image around a given coordinate (RA, DEC).

	:param image:
		Input image.

	:param ra:
		Coordinate RA.

	:param dec:
		Coordinate DEC.

	:param idx_hdu:
		Index of the FITS file extension in which the image is stored.

	:param stamp_size:
		Size [ny,nx] of cropped image.

	:param name_out_fits:
		Name for output cropped image.
	"""

	from astropy.wcs import WCS
	from astropy.nddata import Cutout2D

	hdu = fits.open(image)[int(idx_hdu)]
	wcs = WCS(hdu.header)
	gal_x, gal_y = wcs.wcs_world2pix(ra, dec, 1)
	position = (gal_x,gal_y)
	cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
	hdu.data = cutout.data
	hdu.header.update(cutout.wcs.to_header())

	if name_out_fits is None:
		name_out_fits = 'crop_%s' % image

	hdu.writeto(name_out_fits, overwrite=True)
	return name_out_fits


def crop_image_given_xy(img_name=None, x=None, y=None, stamp_size=[], name_out_fits=None):
	"""Function for cropping an image around a given position (x, y)
	"""

	from astropy.wcs import WCS
	from astropy.nddata import Cutout2D

	hdu = fits.open(img_name)[0]
	wcs = WCS(hdu.header)
	position = (x,y)
	cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
	hdu.data = cutout.data
	hdu.header.update(cutout.wcs.to_header())

	if name_out_fits is None:
		name_out_fits = 'crop_%s' % img_name

	hdu.writeto(name_out_fits, overwrite=True)
	print ("[produce %s]" % name_out_fits)


def crop_2D_data(in_data=None, data_x_cent=None, data_y_cent=None, new_size_x=None, new_size_y=None):
	del_y = int((new_size_y-1)/2)
	row_start = data_y_cent - del_y
	row_end = data_y_cent + del_y + 1

	del_x = int((new_size_x-1)/2)
	col_start = data_x_cent - del_x
	col_end = data_x_cent + del_x + 1

	new_data = in_data[row_start:row_end, col_start:col_end]

	return new_data


def remove_naninf_image_2dinterpolation(data_image):
	from scipy.interpolate import griddata

	rows_nan, cols_nan = np.where((np.isnan(data_image)==True) | (np.isinf(data_image)==True))
	if len(rows_nan)>0:
		y = np.arange(0,data_image.shape[0])
		x = np.arange(0,data_image.shape[1])
		xx, yy = np.meshgrid(x, y)

		rows, cols = np.where((np.isnan(data_image)==False) & (np.isinf(data_image)==False))
		data_image_new = griddata((rows,cols), data_image[rows,cols], (yy, xx), method='cubic')

		rows_nan, cols_nan = np.where((np.isnan(data_image_new)==True) | (np.isinf(data_image_new)==True))
		if len(rows_nan)>0:
			rows, cols = np.where((np.isnan(data_image_new)==False) & (np.isinf(data_image_new)==False))
			#data_image_new = griddata((rows,cols), data_image_new[rows,cols], (yy, xx), method='cubic')
			data_image_new[rows_nan,cols_nan] = np.percentile(data_image_new[rows,cols],50)

		return data_image_new
	else:
		return data_image

def remove_naninfzeroneg_image_2dinterpolation(data_image):
	from scipy.interpolate import griddata

	rows_nan, cols_nan = np.where((np.isnan(data_image)==True) | (np.isinf(data_image)==True) | (data_image<=0))
	if len(rows_nan)>0:
		y = np.arange(0,data_image.shape[0])
		x = np.arange(0,data_image.shape[1])
		xx, yy = np.meshgrid(x, y)

		rows, cols = np.where((np.isnan(data_image)==False) & (np.isinf(data_image)==False) & (data_image>0))
		data_image_new = griddata((rows,cols), data_image[rows,cols], (yy, xx), method='cubic')

		rows_nan, cols_nan = np.where((np.isnan(data_image_new)==True) | (np.isinf(data_image_new)==True) | (data_image_new<=0))
		if len(rows_nan)>0:
			rows, cols = np.where((np.isnan(data_image_new)==False) & (np.isinf(data_image_new)==False) & (data_image_new>0))
			#data_image_new = griddata((rows,cols), data_image_new[rows,cols], (yy, xx), method='cubic')
			data_image_new[rows_nan,cols_nan] = np.percentile(data_image_new[rows,cols],50)

		return data_image_new
	else:
		return data_image


def check_dir(dir_file):
	if dir_file is None:
		dir_file = './'

	if dir_file[len(dir_file)-1] != '/':
		dir_file = dir_file + '/'

	return dir_file


def add_dir(sci_img, var_img, dir_images, filters):
	nbands = len(filters)
	for bb in range(nbands):
		sci_img[filters[bb]] = dir_images + sci_img[filters[bb]]
		var_img[filters[bb]] = dir_images + var_img[filters[bb]]

	return sci_img, var_img


def check_name_remove_dir(file_name, dir_images):
	# keep only the file name without the directory
	return file_name.split('/')[-1]


def mapping_multiplots(nplots,ncols):
	nrows = int(nplots/ncols)
	if nplots % ncols > 0:
		nrows = nrows + 1

	map_plots = np.zeros((int(nrows)+2,int(ncols)+2))
	count = 0
	for yy in range(1,nrows+1):
		for xx in range(1,ncols+1):
			count = count + 1
			if count <= nplots:
				map_plots[yy][xx] = count
	
	return map_plots, nrows


def open_fluxmap_fits(flux_maps_fits):
	""" Function to extract data from the FITS file of fluxes maps produced from the image processing.

	:param flux_maps_fits:
		Input FITS file containing the flux maps.

	:returns filters:
		The set of filters.

	:returns gal_region:
		The galaxy's spatial region of interest.

	:returns flux_map:
		The data cube of flux maps: (nbands,ny,nx).

	:returns flux_err_map:
		The data cube of flux uncertainties maps: (nbands,ny,nx).

	:returns unit_flux:
		The flux unit. In this case, the scale factor to erg/s/cm^2/Angstrom.
	"""

	hdu = fits.open(flux_maps_fits)
	unit_flux = float(hdu[0].header['unit'])
	gal_region = hdu['GALAXY_REGION'].data
	flux_map = hdu['FLUX'].data*unit_flux 
	flux_err_map = hdu['FLUX_ERR'].data*unit_flux
	nbands = int(hdu[0].header['nfilters'])
	filters = []
	for bb in range(nbands):
		filters.append(hdu[0].header['fil%d' % bb])
	hdu.close()

	return filters, gal_region, flux_map, flux_err_map, unit_flux


def plot_maps_fluxes(flux_maps_fits, ncols=5, savefig=True, name_plot_mapflux=None, vmin=-22, vmax=-15, name_plot_mapfluxerr=None):
	""" Function for plotting maps of multiband fluxes.

	:param flux_maps_fits:
		Input FITS file of multiband flux maps.

	:param ncols:
		Number of columns in the plots.

	:param savefig:
		Decide whether to save the plot or not.

	:param name_plot_mapflux:
		Name of the output plot for the maps of multiband fluxes.

   	:param vmin:
    		Minimum flux limit for the color bar in logarithmic scale.

      	:param vmax:
       		Maximum flux limit for the color bar in logarithmic scale.

	:param name_plot_mapfluxerr:
		Name of the output plot for the maps of multiband flux uncertainties.
	"""

	import matplotlib.pyplot as plt

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	#=> plotting
	map_plots, nrows = mapping_multiplots(nbands,ncols)

	# plotting maps of fluxes
	fig1 = plt.figure(figsize=(ncols*4,nrows*4))
	for bb in range(0,nbands):
		yy, xx = np.where(map_plots==bb+1)
		f1 = fig1.add_subplot(nrows, ncols, bb+1)
		if map_plots[yy[0]][xx[0]-1] == 0:
			plt.ylabel('[pixel]', fontsize=15)
		if map_plots[yy[0]+1][xx[0]] == 0:
			plt.xlabel('[pixel]', fontsize=15)

		im = plt.imshow(np.log10(flux_map[bb]), origin='lower', cmap='nipy_spectral', vmin=vmin, vmax=vmax)
		f1.text(0.5, 0.93, '%s' % filters[bb], horizontalalignment='center', verticalalignment='center',
			transform = f1.transAxes, fontsize=16, color='black')

	cax = fig1.add_axes([0.3, 0.86, 0.4, 0.03])
	cb = fig1.colorbar(im, cax=cax, orientation="horizontal")
	cb.ax.tick_params(labelsize=14)
	cax.xaxis.set_ticks_position("top")
	cax.xaxis.set_label_position("top")
	cb.set_label(r'log(Flux density [erg $\rm{ s}^{-1}\rm{cm}^{-2}\AA^{-1}$])', fontsize=22)

	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, hspace=0.1, wspace=0.05)

	if savefig is True:
		if name_plot_mapflux is None:
			name_plot_mapflux = 'maps_fluxes.png'
		plt.savefig(name_plot_mapflux)
	else:
		plt.show()

	# plotting maps of flux uncertainties
	fig1 = plt.figure(figsize=(ncols*4,nrows*4))
	for bb in range(0,nbands):
		yy, xx = np.where(map_plots==bb+1)
		f1 = fig1.add_subplot(nrows, ncols, bb+1)
		if map_plots[yy[0]][xx[0]-1] == 0:
			plt.ylabel('[pixel]', fontsize=15)
		if map_plots[yy[0]+1][xx[0]] == 0:
			plt.xlabel('[pixel]', fontsize=15)

		im = plt.imshow(np.log10(flux_err_map[bb]), origin='lower', cmap='nipy_spectral', vmin=vmin, vmax=vmax)
		f1.text(0.5, 0.93, '%s' % filters[bb], horizontalalignment='center', verticalalignment='center',
			transform = f1.transAxes, fontsize=16, color='black')

	cax = fig1.add_axes([0.3, 0.86, 0.4, 0.03])
	cb = fig1.colorbar(im, cax=cax, orientation="horizontal")
	cb.ax.tick_params(labelsize=14)
	cax.xaxis.set_ticks_position("top")
	cax.xaxis.set_label_position("top")
	cb.set_label(r'log(Flux uncertainty [erg $\rm{ s}^{-1}\rm{cm}^{-2}\AA^{-1}$])', fontsize=22)

	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, hspace=0.1, wspace=0.05)

	if savefig is True:
		if name_plot_mapfluxerr is None:
			name_plot_mapfluxerr = 'maps_flux_errors.png'
		plt.savefig(name_plot_mapfluxerr)
	else:
		plt.show()


def convert_flux_unit(wave, flux, init_unit='erg/s/cm2/A', final_unit='Jy'):
	""" Function for converting flux unit

	:param wave:
		Wavelength grids.

	:param flux:
		Flux grids.

	:param init_unit:
		Initial unit of the input fluxes. Options are: 'erg/s/cm2/A', 'erg/s/cm2', 'Jy', 'uJy'.

	:param final_unit:
		Final unit for conversion. Options are: 'erg/s/cm2/A', 'erg/s/cm2', 'Jy', 'uJy'.
	"""

	wave = np.asarray(wave)
	flux = np.asarray(flux)

	if init_unit == 'erg/s/cm2/A':
		if final_unit == 'erg/s/cm2/A':
			final_flux = flux
		elif final_unit == 'erg/s/cm2':
			final_flux = flux*wave
		elif final_unit == 'Jy':
			final_flux = flux*wave*wave/1.0e-23/2.998e+18
		elif final_unit == 'uJy':
			final_flux = flux*wave*wave/1.0e-6/1.0e-23/2.998e+18
		else:
			print ('The final_unit is not recognized!')
			sys.exit()

	elif init_unit == 'erg/s/cm2':
		if final_unit == 'erg/s/cm2/A':
			final_flux = flux/wave
		elif final_unit == 'erg/s/cm2':
			final_flux = flux
		elif final_unit == 'Jy':
			final_flux = flux*wave/1.0e-23/2.998e+18
		elif final_unit == 'uJy':
			final_flux = flux*wave/1.0e-6/1.0e-23/2.998e+18
		else:
			print ('The final_unit is not recognized!')
			sys.exit()

	elif init_unit == 'Jy':
		if final_unit == 'erg/s/cm2/A':
			final_flux = flux*1.0e-23*2.998e+18/wave/wave
		elif final_unit == 'erg/s/cm2':
			final_flux = flux*1.0e-23*2.998e+18/wave
		elif final_unit == 'Jy':
			final_flux = flux
		elif final_unit == 'uJy':
			final_flux = 1e+6*flux
		else:
			print ('The final_unit is not recognized!')
			sys.exit()

	elif init_unit == 'uJy':
		if final_unit == 'erg/s/cm2/A':
			final_flux = flux*1.0e-23*2.998e+18/wave/wave/1.0e+6
		elif final_unit == 'erg/s/cm2':
			final_flux = flux*1.0e-23*2.998e+18/wave/1.0e+6
		elif final_unit == 'Jy':
			final_flux = flux/1.0e+6
		elif final_unit == 'uJy':
			final_flux = flux
		else:
			print ('The final_unit is not recognized!')
			sys.exit()
	else:
		print ('The init_unit is not recognized!')
		sys.exit()

	return final_flux


def get_pixels_SED_fluxmap(flux_maps_fits, pix_x=None, pix_y=None, all_pixels=False):
	""" Function to get SEDs of pixels.

	:param flux_maps_fits:
		Input FITS file of the multiband fluxes.

	:param pix_x:
		One dimensional array of x coordinates. Only relevant if all_pixels=False.

	:param pix_y:
		One dimensional array of the y coordinates. Only relevant if all_pixels=False.

	:param all_pixels: (optional)
		An option to get SEDs of all pixels.

	:returns pix_x:
		x coordinates.

	:returns pix_y:
		y coordinates.

	:returns pix_SED_flux:
		Fluxes of the pixels: (npixs,nbands)

	:returns pix_SED_flux_err:
		Flux uncertainties of the pixels: (npixels,nbands)

	:returns photo_wave:
		Central wavelength of the filters.
	"""

	from ..utils.filtering import cwave_filters

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)

	# get central wavelength of filters
	photo_wave = cwave_filters(filters)

	# transpose from (band,y,x) => (y,x,band)
	flux_map_trans = np.transpose(flux_map, axes=(1,2,0))
	flux_err_map_trans = np.transpose(flux_err_map, axes=(1,2,0))

	if all_pixels == True:
		rows, cols = np.where(gal_region == 1)
		pix_y, pix_x = rows, cols
	else:
		if np.isscalar(pix_x) == True:
			pix_x, pix_y = [round(pix_x)], [round(pix_y)]

	pix_SED_flux = []
	pix_SED_flux_err = []
	for ii in range(0,len(pix_x)):
		pix_SED_flux.append(flux_map_trans[int(pix_y[ii])][int(pix_x[ii])])
		pix_SED_flux_err.append(flux_err_map_trans[int(pix_y[ii])][int(pix_x[ii])])

	pix_SED_flux = np.asarray(pix_SED_flux)
	pix_SED_flux_err = np.asarray(pix_SED_flux_err)

	return pix_x, pix_y, pix_SED_flux, pix_SED_flux_err, photo_wave


def plot_SED_pixels(flux_maps_fits, pix_x=None, pix_y=None, all_pixels=False, logscale_y=True, logscale_x=True, 
	wunit='angstrom', yrange=None, xrange=None, savefig=True, name_plot=None):
	""" Function for plotting SEDs of pixels.

	:param flux_maps_fits:
		Input FITS file of the multiband fluxes.

	:param pix_x:
		One dimensional array of x coordinates. Only relevant if all_pixels=False.

	:param pix_y:
		One dimensional array of the y coordinates. Only relevant if all_pixels=False.

	:param all_pixels: (optional)
		An option to get SEDs of all pixels.

	:param logscale_y: (optional)
		Option to set the y axis in logarithmic scale.

	:param logscale_x: (optional)
		Option to set the x axis in logarithmic scale.

	:param wunit: (optional)
		Wavelength unit. Options are 'angstrom' and 'micron'.

	:param yrange: (optional)
		Range in y axis.

	:param xrange: (optional)
		Range in x axis.

	:param savefig: (optional)
		Option to save the plot.

	:param name_plot: (optional)
		Name for the output plot.
	"""

	import matplotlib.pyplot as plt 

	# get SEDs of pixels
	pix_x, pix_y, pix_SED_flux, pix_SED_flux_err, photo_wave= get_pixels_SED_fluxmap(flux_maps_fits, pix_x, pix_y, all_pixels=all_pixels)

	# Plotting:
	fig1 = plt.figure(figsize=(12,6))
	f1 = plt.subplot()
	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')

	if wunit == 'micron':
		plt.xlabel(r"Wavelength [$\mu$m]", fontsize=18)
	elif wunit == 'angstrom':
		plt.xlabel(r"Wavelength [$\AA$]", fontsize=18)
	else:
		print ('wunit is not recognized!')
		sys.exit()

	plt.ylabel(r"Flux density [erg $\rm{s}^{-1}\rm{cm}^{-2}\AA^{-1}$]", fontsize=18)
	plt.setp(f1.get_xticklabels(), fontsize=12)
	plt.setp(f1.get_yticklabels(), fontsize=12)

	if yrange is not None:
		plt.ylim(yrange[0],yrange[1])
 
	if xrange is not None:
		plt.xlim(xrange[0],xrange[1])

	for ii in range(len(pix_SED_flux)):
		if wunit == 'micron':
			plt.errorbar(photo_wave/1e+4, pix_SED_flux[ii], yerr=pix_SED_flux_err[ii], fmt='-o', lw=2, alpha=0.5)
		elif wunit == 'angstrom':
			plt.errorbar(photo_wave, pix_SED_flux[ii], yerr=pix_SED_flux_err[ii], fmt='-o', lw=2, alpha=0.5)

	if savefig is True:
		if name_plot is None:
			name_plot = 'plot_SED_pixels.png'
		plt.savefig(name_plot)
	else:
		plt.show()


def get_total_SED(flux_maps_fits):
	""" Function to calculate total (i.e., integrated) SED from input maps of multiband fluxes.

	:param flux_maps_fits:
		Input FITS file containing maps of fluxes produced from the image processing.

	:returns tot_SED_flux:
		Total fluxes in multiple bands. The format is 1D array.

	:returns tot_SED_flux_err:
		Total flux uncertainties in multiple bands. The format is 1D array.

	:returns photo_wave:
		The central wavelength of the filters.
	"""

	from ..utils.filtering import cwave_filters

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	# get central wavelength of filters
	photo_wave = cwave_filters(filters)

	rows, cols = np.where(gal_region==1)
	tot_SED_flux = np.zeros(nbands)
	tot_SED_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		tot_SED_flux[bb] = np.sum(flux_map[bb][rows,cols])
		tot_SED_flux_err[bb] = np.sqrt(np.sum(np.square(flux_err_map[bb][rows,cols])))

	return tot_SED_flux, tot_SED_flux_err, photo_wave


def get_curve_of_growth(flux_maps_fits, filter_id, e=0.0, pa=45.0, cent_x=None, cent_y=None):
	""" Function to calculate curve of growth of flux in a particular band.
	"""

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	rows, cols = np.where(gal_region==1)
	dimy, dimx = gal_region.shape[0], gal_region.shape[1]

	if cent_y is None:
		cent_y = (dimy-1.0)/2.0
	if cent_x is None:
		cent_x = (dimx-1.0)/2.0

	pix_x_norm, pix_y_norm = cols - cent_x, rows - cent_y

	pix_sma = ellipse_sma(e,pa,pix_x_norm,pix_y_norm)

	# sort the radius
	sort_index = np.argsort(pix_sma)
	curve_pix_rad = pix_sma[sort_index]

	# get cumulative sum of fluxes
	curve_pix_flux = flux_map[int(filter_id)][rows,cols]
	curve_cumul_flux = np.cumsum(curve_pix_flux[sort_index])

	return curve_pix_rad, curve_cumul_flux


def get_flux_radial_profile(flux_maps_fits, filter_id, e=0.0, pa=45.0, cent_x=None, cent_y=None):
	""" Function to calculate radial profile of flux in a particular band.
	"""

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	rows, cols = np.where(gal_region==1)
	dimy, dimx = gal_region.shape[0], gal_region.shape[1]

	if cent_y is None:
		cent_y = (dimy-1.0)/2.0
	if cent_x is None:
		cent_x = (dimx-1.0)/2.0

	pix_x_norm, pix_y_norm = cols - cent_x, rows - cent_y

	pix_sma0 = ellipse_sma(e,pa,pix_x_norm,pix_y_norm)
	sort_index = np.argsort(pix_sma0)
	pix_sma = pix_sma0[sort_index]

	pix_flux = flux_map[int(filter_id)][rows,cols][sort_index]

	return pix_sma, pix_flux


def get_SNR_radial_profile(flux_maps_fits, e=0.0, pa=45.0, cent_x=None, cent_y=None):
	""" Function to get radial profile of S/N ratio in all filters
	"""

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	rows, cols = np.where(gal_region==1)
	dimy, dimx = gal_region.shape[0], gal_region.shape[1]

	if cent_y is None:
		cent_y = (dimy-1.0)/2.0
	if cent_x is None:
		cent_x = (dimx-1.0)/2.0

	pix_x_norm, pix_y_norm = cols - cent_x, rows - cent_y
	pix_sma = ellipse_sma(e,pa,pix_x_norm,pix_y_norm)
	pix_SNR = np.zeros((nbands,len(pix_sma)))

	for bb in range(0,nbands):
		pix_SNR[bb] = flux_map[bb][rows,cols]/flux_err_map[bb][rows,cols]

	return pix_sma, pix_SNR, filters


def plot_SNR_radial_profile(flux_maps_fits, e=0.0, pa=45.0, cent_x=None, cent_y=None, yrange=None, xrange=None, savefig=True, name_plot=None):
	""" Function for plotting S/N ratios of pixels.

	:param flux_maps_fits:
		Input FITS file produced from the image processing.

	:param e:
		Ellipticity of the elliptical apertures that will be used for deriving the S/N ratios.

	:param pa:
		The position angle of the elliptical apertures.

	:param cent_x:
		The x coordinate of the central pixel.

	:param cent_y:
		The y coordinate of the central pixel.

	:param yrange:
		Range in y-axis for the plot.

	:param xrange:
		Range in x-axis for the plot.

	:param savefig:
		Decide whether to save the plot or not.

	:param name_plot:
		Name for the output plot.
	"""

	import matplotlib.pyplot as plt

	pix_sma, pix_SNR, filters = get_SNR_radial_profile(flux_maps_fits, e=e, pa=pa, cent_x=cent_x, cent_y=cent_y)
	nbands = len(filters)

	# plotting
	fig1 = plt.figure(figsize=(10,6))
	f1 = plt.subplot()

	if yrange is not None:
		plt.ylim(yrange[0],yrange[1])
 
	if xrange is not None:
		plt.xlim(xrange[0],xrange[1])

	plt.setp(f1.get_yticklabels(), fontsize=13, visible=True)
	plt.setp(f1.get_xticklabels(), fontsize=13, visible=True)
	plt.xlabel(r"Radius [pixel]", fontsize=20)
	plt.ylabel(r"log(S/N Ratio)", fontsize=20)

	cmap = plt.get_cmap('jet', nbands)

	for bb in range(0,nbands):
		plt.scatter(pix_sma, np.log10(pix_SNR[bb]), s=10, alpha=0.5, color=cmap(bb), label='%s' % filters[bb])

	plt.legend(fontsize=10, ncol=2)

	if savefig is True:
		if name_plot is None:
			name_plot = 'plot_pixels_SNR.png'
		plt.savefig(name_plot)
	else:
		plt.show()


def photometry_within_aperture(flux_maps_fits, e=0.0, pa=45.0, cent_x=None, cent_y=None, radius=None):
	""" Function to get photometry within a given aperture (elliptical/circular) from input flux maps

	:param flux_maps_fits:
		input FITS file containing the maps of fluxes produced from the image processing.

	:param e:
		Ellipticity of the apertures.

	:param pa:
		Position angle of the elliptical apertures.

	:param cent_x:
		The x coordinate of the central pixel.

	:param cent_y:
		The y coordinate of the central pixel.

	:param radius:
		The radius of the aperture.

	:returns tot_fluxes:
		Output total fluxes.

	:returns tot_flux_errors:
		Output total flux uncertainties.

	"""

	# get data from the input FITS file
	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	dimy, dimx = gal_region.shape[0], gal_region.shape[1]

	# get pixels within the aperture
	if cent_y is None:
		cent_y = (dimy-1.0)/2.0
	if cent_x is None:
		cent_x = (dimx-1.0)/2.0

	x = np.linspace(0,dimx-1,dimx)
	y = np.linspace(0,dimy-1,dimy)
	xx, yy = np.meshgrid(x,y)
	xx_norm, yy_norm = xx - cent_x, yy - cent_y

	data2D_sma = ellipse_sma(e,pa,xx_norm,yy_norm)
	rows, cols = np.where((gal_region==1) & (data2D_sma<=radius))

	pix_x, pix_y, pix_SED_flux, pix_SED_flux_err, photo_wave = get_pixels_SED_fluxmap(flux_maps_fits, pix_x=cols, pix_y=rows)

	tot_fluxes = np.sum(pix_SED_flux, axis=0)
	tot_flux_errors = np.sqrt(np.sum(np.square(pix_SED_flux_err), axis=0))

	return tot_fluxes, tot_flux_errors


def draw_aperture_on_maps_fluxes(flux_maps_fits, ncols=6, e=[0.0], pa=[45.0], cent_x=None, cent_y=None, radius=[5.0], 
	colors=None, lw=3, savefig=True, name_plot=None):
	""" Function for drawing apertures on top of the multiband fluxes maps. This function can plot more than one aperture.

	:param flux_maps_fits:
		input FITS file containing the maps of fluxes produced from the image processing.

	:param e:
		Ellipticity of the apertures. Input in list format if want to make multiple apertures. 

	:param pa:
		Position angle of the elliptical apertures. Input in list format if want to make multiple apertures. 

	:param cent_x:
		The x coordinate of the central pixel. Input in list format if want to make multiple apertures. 

	:param cent_y:
		The y coordinate of the central pixel. Input in list format if want to make multiple apertures. 

	:param radius:
		The radius of the aperture. Input in list format if want to make multiple apertures. 

	:param colors:
		Colors of the apertures. Input in list format if want to make multiple apertures. 

	:param lw:
		Line width of the apertures in the plot.

	:param savefig:
		Decide whether to save the plot or not.

	:param name_plot:
		Name for the plot.
	"""

	import matplotlib.pyplot as plt
	from piXedfit.piXedfit_images import draw_ellipse

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	#=> plotting
	map_plots, nrows = mapping_multiplots(nbands,ncols)

	# plotting maps of fluxes
	fig1 = plt.figure(figsize=(ncols*4,nrows*4))
	for bb in range(0,nbands):
		yy, xx = np.where(map_plots==bb+1)
		f1 = fig1.add_subplot(nrows, ncols, bb+1)
		if map_plots[yy[0]][xx[0]-1] == 0:
			plt.ylabel('[pixel]', fontsize=15)
		if map_plots[yy[0]+1][xx[0]] == 0:
			plt.xlabel('[pixel]', fontsize=15)

		plt.imshow(np.log10(flux_map[bb]), origin='lower', cmap='nipy_spectral')

		for ii in range(0,len(e)):
			ellipse_xy = draw_ellipse(cent_x[ii], cent_y[ii], radius[ii], e[ii], pa[ii])
			if colors is None:
				plt.plot(ellipse_xy[0], ellipse_xy[1], lw=lw)
			else:
				plt.plot(ellipse_xy[0], ellipse_xy[1], lw=lw, color=colors[ii])

		f1.text(0.5, 0.93, '%s' % filters[bb], horizontalalignment='center', verticalalignment='center', transform = f1.transAxes, fontsize=16, color='black')

	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

	if savefig is True:
		if name_plot is None:
			name_plot = 'apertures.png'
		plt.savefig(name_plot)
	else:
		plt.show()

def rotate_pixels(x, y, x_cent, y_cent, theta):
	x1 = (x-x_cent)*np.cos(theta*np.pi/180.0) - (y-y_cent)*np.sin(theta*np.pi/180.0) + x_cent
	y1 = (x-x_cent)*np.sin(theta*np.pi/180.0) + (y-y_cent)*np.cos(theta*np.pi/180.0) + y_cent
	return x1, y1


def linear_func_of_twopoints(x1, x2, y1, y2):
	gradient, intercept = (y2-y1)/(x2-x1), (x2*y1 - x1*y2)/(x2-x1)
	return gradient, intercept 


def get_rectangular_region(stamp_img, x=None, y=None, ra=None, dec=None):
	from astropy.wcs import WCS

	hdu = fits.open(stamp_img)
	wcs = WCS(hdu[0].header)
	data_img = hdu[0].data
	dimy, dimx = data_img.shape[0], data_img.shape[1]
	hdu.close()

	if x is not None and y is not None:
		x1, y1 = x[0], y[0]
		x2, y2 = x[1], y[1]
		x3, y3 = x[2], y[2]
		x4, y4 = x[3], y[3]
	else:
		x1, y1 = wcs.wcs_world2pix(ra[0], dec[0], 1)
		x2, y2 = wcs.wcs_world2pix(ra[1], dec[1], 1)
		x3, y3 = wcs.wcs_world2pix(ra[2], dec[2], 1)
		x4, y4 = wcs.wcs_world2pix(ra[3], dec[3], 1)

	x, y = np.asarray([x1, x2, x3, x4]), np.asarray([y1, y2, y3, y4])

	# find center
	cent_x, cent_y = int((max(x)+min(x))/2.0), int((max(y)+min(y))/2.0)

	idx_sort_x = np.argsort(x)

	x1, y1 = x[idx_sort_x[0]], y[idx_sort_x[0]]
	x2, y2 = x[idx_sort_x[1]], y[idx_sort_x[1]]
	x3, y3 = x[idx_sort_x[2]], y[idx_sort_x[2]]
	x4, y4 = x[idx_sort_x[3]], y[idx_sort_x[3]]

	f12_m, f12_b = linear_func_of_twopoints(x1, x2, y1, y2)
	f13_m, f13_b = linear_func_of_twopoints(x1, x3, y1, y3) 
	f24_m, f24_b = linear_func_of_twopoints(x2, x4, y2, y4)
	f34_m, f34_b = linear_func_of_twopoints(x3, x4, y3, y4)

	gal_region0 = np.zeros((dimy,dimx))
	rows, cols = np.where(gal_region0==0) 

	map12_y, map12_fy = np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
	map12_y[rows,cols] = rows
	map12_fy[rows,cols] = f12_m*cols + f12_b

	map13_y, map13_fy = np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
	map13_y[rows,cols] = rows
	map13_fy[rows,cols] = f13_m*cols + f13_b

	map24_y, map24_fy = np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
	map24_y[rows,cols] = rows
	map24_fy[rows,cols] = f24_m*cols + f24_b

	map34_y, map34_fy = np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
	map34_y[rows,cols] = rows
	map34_fy[rows,cols] = f34_m*cols + f34_b

	### 12
	if cent_y > f12_m*cent_x + f12_b:
		rows1, cols1 = np.where(map12_y >= map12_fy - 1)
		gal_region0[rows1,cols1] = 1
	else:
		rows1, cols1 = np.where(map12_y <= map12_fy + 1)
		gal_region0[rows1,cols1] = 1

	### 13
	if cent_y > f13_m*cent_x + f13_b:
		rows1, cols1 = np.where(map13_y >= map13_fy - 1)
		temp = gal_region0[rows1,cols1]
		gal_region0[rows1,cols1] = temp + 1
	else:
		rows1, cols1 = np.where(map13_y <= map13_fy + 1)
		temp = gal_region0[rows1,cols1]
		gal_region0[rows1,cols1] = temp + 1

	### 24
	if cent_y > f24_m*cent_x + f24_b:
		rows1, cols1 = np.where(map24_y >= map24_fy - 1)
		temp = gal_region0[rows1,cols1]
		gal_region0[rows1,cols1] = temp + 1
	else:
		rows1, cols1 = np.where(map24_y <= map24_fy + 1)
		temp = gal_region0[rows1,cols1]
		gal_region0[rows1,cols1] = temp + 1

	### 34
	if cent_y > f34_m*cent_x + f34_b:
		rows1, cols1 = np.where(map34_y >= map34_fy - 1)
		temp = gal_region0[rows1,cols1]
		gal_region0[rows1,cols1] = temp + 1
	else:
		rows1, cols1 = np.where(map34_y <= map34_fy + 1)
		temp = gal_region0[rows1,cols1]
		gal_region0[rows1,cols1] = temp + 1

	gal_region = np.zeros((dimy,dimx))
	rows2, cols2 = np.where(gal_region0 == 4)
	gal_region[rows2,cols2] = 1

	return gal_region, x1, x2, x3, x4, y1, y2, y3, y4


def get_rectangular_region_old(stamp_img, x=None, y=None, ra=None, dec=None, theta=11.9):

    from astropy.wcs import WCS
    
    hdu = fits.open(stamp_img)
    wcs = WCS(hdu[0].header)
    data_img = hdu[0].data
    dimy, dimx = data_img.shape[0], data_img.shape[1]
    hdu.close()
    
    if x is not None and y is not None:
        x1, y1 = x[0], y[0]
        x2, y2 = x[1], y[1]
        x3, y3 = x[2], y[2]
        x4, y4 = x[3], y[3]
    else:
        x1, y1 = wcs.wcs_world2pix(ra[0], dec[0], 1)
        x2, y2 = wcs.wcs_world2pix(ra[1], dec[1], 1)
        x3, y3 = wcs.wcs_world2pix(ra[2], dec[2], 1)
        x4, y4 = wcs.wcs_world2pix(ra[3], dec[3], 1)

    x_cent = 0.5*(x4+x2)
    y_cent = 0.5*(y1+y3)

    x1_rot, y1_rot = rotate_pixels(x1,y1,x_cent,y_cent,theta)
    x2_rot, y2_rot = rotate_pixels(x2,y2,x_cent,y_cent,theta)
    x3_rot, y3_rot = rotate_pixels(x3,y3,x_cent,y_cent,theta)
    x4_rot, y4_rot = rotate_pixels(x4,y4,x_cent,y_cent,theta)
    
    xmin = round(0.5*(x3_rot+x4_rot))
    xmax = round(0.5*(x1_rot+x2_rot))

    ymin = round(0.5*(y1_rot+y4_rot))
    ymax = round(0.5*(y2_rot+y3_rot))
    
    slit_region0 = np.zeros((dimy,dimx))
    slit_region0[ymin:ymax, xmin:xmax] = 1
    
    ## rotate
    rows, cols = np.where(slit_region0==1)
    theta1 = -1.0*theta
    cols1, rows1 = rotate_pixels(cols, rows, x_cent, y_cent, theta1)
    cols_u = np.zeros(len(cols1)) + 0.5
    rows_u = np.zeros(len(cols1)) + 0.5
    cols_d = np.zeros(len(cols1)) - 0.5
    rows_d = np.zeros(len(cols1)) - 0.5
    
    cols_temp, rows_temp = cols1+cols_u, rows1+rows_u
    cols2, rows2 = cols1.tolist()+cols_temp.tolist(), rows1.tolist()+rows_temp.tolist()
    
    cols2, rows2 = np.asarray(cols2), np.asarray(rows2)
    
    cols_temp, rows_temp = cols1+cols_d, rows1+rows_d
    cols3, rows3 = cols2.tolist()+cols_temp.tolist(), rows2.tolist()+rows_temp.tolist()
    
    cols3, rows3 = np.asarray(cols3), np.asarray(rows3)
    
    cols_temp, rows_temp = cols1+cols_u, rows1+rows_d
    cols4, rows4 = cols3.tolist()+cols_temp.tolist(), rows3.tolist()+rows_temp.tolist()
    
    cols4, rows4 = np.asarray(cols4), np.asarray(rows4)
    
    cols_temp, rows_temp = cols1+cols_d, rows1+rows_u
    cols5, rows5 = cols4.tolist()+cols_temp.tolist(), rows4.tolist()+rows_temp.tolist()
    
    cols5, rows5 = np.asarray(cols5), np.asarray(rows5)

    slit_region = np.zeros((dimy,dimx))
    slit_region[np.round_(rows5).astype(int), np.round_(cols5).astype(int)] = 1
    
    return slit_region, x1, x2, x3, x4, y1, y2, y3, y4


def central_brightest_pixel(flux_maps_fits, filter_id, xrange=None, yrange=None):
	""" Function to get the central brightest pixel within a map of flux in a particular band.

	:param flux_maps_fits:
		Input FITS file of the flux maps produced from the image processing.

	:param filter_id:
		The filter of the image where the central pixel is to be found.

	:param xrange:
		Range in x-axis for searching area.

	:param yrange:
		Range in y-axis for searching area.
	"""

	filters, gal_region, flux_map, flux_err_map, unit_flux = open_fluxmap_fits(flux_maps_fits)
	nbands = len(filters)

	if xrange is None:
		xrange = [0,gal_region.shape[1]]
	if yrange is None:
		yrange = [0,gal_region.shape[0]]

	cent_y, cent_x = np.unravel_index(flux_map[int(filter_id)][yrange[0]:yrange[1], xrange[0]:xrange[1]].argmax(), flux_map[int(filter_id)][yrange[0]:yrange[1], xrange[0]:xrange[1]].shape)
	cent_x, cent_y = cent_x+xrange[0], cent_y+yrange[0]

	return cent_x, cent_y	

















