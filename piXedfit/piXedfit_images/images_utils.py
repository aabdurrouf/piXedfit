import numpy as np 
from math import pi, pow, sqrt, cos, sin 
import sys, os
from astropy.io import fits 
from astroquery.sdss import SDSS
from astroquery.mast import Observations
from astroquery.esa.hubble import ESAHubble
from astroquery.ipac.irsa import sha
import urllib
import pyvo
import gzip, shutil, glob
from astropy.table import Table
from astropy.cosmology import *
import warnings
warnings.filterwarnings('ignore')

from ..utils.filtering import cwave_filters


__all__ = ["sort_filters", "kpc_per_pixel", "k_lmbd_Fitz1986_LMC", "EBV_foreground_dust", "Sloan", "TwoMASS", "WISE", "Galex", "HST", "Spitzer","skybg_sdss", "get_gain_dark_variance", "var_img_sdss", "var_img_GALEX", "var_img_2MASS", "var_img_WISE", "var_img_from_unc_img",
			"var_img_from_weight_img", "mask_region_bgmodel", "subtract_background", "get_psf_fwhm","get_largest_FWHM_PSF", "ellipse_fit", "draw_ellipse", "ellipse_sma", "crop_ellipse_galregion",
			"crop_ellipse_galregion_fits", "crop_stars", "crop_stars_galregion_fits",  "crop_image_given_radec", "segm_sep", "crop_image_given_xy", "check_avail_kernel", "create_kernel_gaussian",
			"raise_errors", "get_img_pixsizes", "in_kernels", "get_flux_or_sb", "crop_2D_data"]


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
			ax = 0.574*pow(inv_lmbd_micron,1.61)
			bx = -0.527*pow(inv_lmbd_micron,1.61)
			k = 3.1*ax + bx

	return k


def EBV_foreground_dust(ra, dec):
	"""Function for estimating E(B-V) associated with the foreground Galactic dust attenuation.

	:param ra:
		Right ascension in degree.

	:param dec:
		Declination in degree.

	:returns ebv:
		E(B-V) associated with the foreground Galactic dust.
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


def Sloan(pos,bands = ['u','g','r','i','z'],size=20):
	"""A tool to download images from SDSS.

	:param pos: 
		Target's coordinate, in the astropy.coordinates form.

	:param bands: (defaults: ['u','g','r','i','z'])
		Request image in which filter. Default to download every band

	:param size:  (defaults: 20[arcsec])
		Search cone size. Note that this size IS NOT image size.

	"""
	
	# Query the region
	xid = SDSS.query_region(pos, spectro=False,radius = size * u.arcsec)

	# a to c are dummy variables to drop repeated images
	a = xid.to_pandas()
	b = a.drop_duplicates(subset=['run','rerun','camcol','field'])   # remove repeated images
	c = Table([b.ra,b.dec,b.objid,b.run,b.rerun,b.camcol,b.field],names = ('ra','dec','objid','run','rerun','camcol','field'))

	# Start downloading here
	for band in bands:
		im = SDSS.get_images(matches=c, band = band)
		i = 0 # Numbering each file
		for image in im:
			image.writeto(f'SDSS_{band}_{i}.fits', overwrite = True)
			print(f"{band} image downloaded")
			i += 1


def TwoMASS(pos,size = 0.1):
	"""A tool to download TwoMASS images from IRSA.

	:param pos: 
		Target's coordinate, in the astropy.coordinates form.

	:param size:  (defaults: 0.1[deg])
		Search cone size. Note that this size IS NOT image size.
	"""

	# Connect to pyvo service
	image_service = pyvo.regsearch(servicetype='image', keywords=['2mass'])
	image_table = image_service[0].search(pos=pos, size=size) # index 0 is fixed according to regsearch result
	im_table = image_table.to_table()

	# Dummy variavles to remove repeat images
	a = im_table.to_pandas()
	b = a[a['format'].str.contains('fits')].drop_duplicates(subset=['band','hem','date','scan','image']) # filter for fits file, exclude html files

	# Start downloading
	for _, row in b.sort_values(['band','date']).iterrows(): # By sorting to couple same scan images
		no = 0
		while os.path.exists(f"{row.band}_{no}.fits.gz"): # file numbering
			no += 1
		urllib.request.urlretrieve(f"{row.download}", f"{row.band}_{no}.fits.gz") # Download images from IRSA
    
    # Because images from IRSA are zipped, we need to unzip them
	for file in glob.glob("*.gz"):
		with gzip.open(file, 'r') as f_in, open(file.replace(".gz", ""), 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)  # unzipped gz files
		os.remove(file)   # delete .gz files 

def WISE(pos, size = 0.1, pix = 800):
	"""A tool to download allwise images from IRSA.

	:param pos:
		Target's coordinate, in the astropy.coordinates form.

	:param size: (defaults: 0.1[deg])
		Search cone size. Note that this size IS NOT image size.

	:param pix: (defaults to 800)
		This does crop on the image. If you want to download the whole image from WISE, please set pix = 0
		Recommend not to download the original image which is way too large (~60 MB).

	"""

	# Connect to pyvo service
	image_service = pyvo.regsearch(servicetype='image', keywords=['allwise'])
	image_table = image_service[0].search(pos=pos, size=size) # index 0 is fixed according to regsearch result
	im_table = image_table.to_table()

	# Dummy variables to remove repeat images
	a = im_table.to_pandas()
	b = a[a['sia_fmt'].str.contains('fits')].drop_duplicates(subset=['sia_url']) # filter for fits file, exclude html files


	for _, row in b.sort_values(['sia_bp_id','coadd_id']).iterrows():
		no = 0

		while os.path.exists(f"{row.sia_bp_id}_{no}.fits.gz"): # file numbering
			no += 1

		if pix != 0: # doing crop on the url

			urllib.request.urlretrieve(f"{row.sia_url}?center={pos.ra.degree},{pos.dec.degree}&size={pix}pix", f"{row.sia_bp_id}_{no}.fits.gz") 
			urllib.request.urlretrieve(f"{row.unc_url}?center={pos.ra.degree},{pos.dec.degree}&size={pix}pix", f"{row.sia_bp_id}_unc_{no}.fits.gz") 

		elif pix == 0: # request original image
			urllib.request.urlretrieve(f"{row.sia_url}", f"{row.sia_bp_id}_{no}.fits") 
			urllib.request.urlretrieve(f"{row.unc_url}", f"{row.sia_bp_id}_unc_{no}.fits") 

	# Unzip all the images
	for file in glob.glob("*.gz"):
		with gzip.open(file, 'r') as f_in, open(file.replace(".gz", ""), 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)  # unzipped gz files
		os.remove(file)   # delete gz files  


def Galex(pos, size = 0.1, unzip = True):
	"""A tool to download allwise images from GALEX.

	:param pos:
		Target's coordinate, in the astropy.coordinates form.
	:param size: (defaults to 0.1 [deg] )
		Search cone size. Note that this size IS NOT image size.	
	:param unzip: (defaults to True)
		Whether to unzip the download file.
	"""

		# Full list
	obs_table = Observations.query_region(pos,radius= size)
	data_products_by_obs = Observations.get_product_list(obs_table[:])
	# query list
	data_products = data_products_by_obs[(data_products_by_obs['obs_collection'] == 'GALEX')]

	# Preventing from repeating download
	# Here we only download three datatypes: intensity map, sky background map and background subtracted image
	a = data_products.to_pandas()
	b = a[a.productFilename.str.contains('int.fits|skybg.fits|intbgsub.fits')]

	# Start downloading
	for _, row in b.iterrows():
		Observations.download_file(row.dataURI)

	if unzip:
		for file in glob.glob("*.gz"):
			with gzip.open(file, 'r') as f_in, open(file.replace(".gz", ""), 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)  # unzipped gz files
			os.remove(file)   # delete gz files  
	else:
		pass


def HST(pos,size = 1, save = True, output_fmt = 'csv'):
	"""A tool to download allwise images from Hubble Space Telescope.

	:param pos:
		Target's coordinate, in the astropy.coordinates form.
	:param size: (defaults to 1 [arcmin] )
		Search cone size. Note that this size IS NOT image size.
	:param save: (defaults to True)
		Whether to save searching result.
	:param output_fmt: (defaults to 'csv')
		Search result file extension, by default 'csv'
        Other formats: 'csv', 'votable', 'xml'
	"""

	if output_fmt not in ['csv','votable','xml']:
		raise TypeError("output format is not supported. Please use one of these: csv, votable, xml")

	# Start Hubble service
	esahubble = ESAHubble()
	table = esahubble.cone_search_criteria(radius = size , coordinates = pos , save = save, output_format= output_fmt ,obs_collection= "HST", filename = f'Search_Result_Table.{output_fmt}')

	# Select HST images
	a = table.to_pandas()
	b = a[(a.collection == 'HST') & (a.data_product_type == 'image')]
	b.to_csv("Download_Table.csv")

	# Start downloading
	for obs_id in b.observation_id.unique():
		esahubble.download_product(observation_id= obs_id, filename = f"data_for_{obs_id}.tar")

def Spitzer(pos, size = 1/120):
	"""A tool to download allwise images from Spitzer.

	:param pos:
		Target's coordinate, in the astropy.coordinates form.
	:param size: (defaults to 1/120 [deg] )
		Search cone size. Note that this size IS NOT image size.
	"""

	# Start searching
	table = sha.query(coord = pos , size = size)
	
	# Filtering images
	table[table['filetype'] == 'Image']
	# Save filtered result
	table.write('Download_detail.csv', format='csv', overwrite = True)

	# Start downloading
	for url in table[table['filetype'] == b' Image   ']['accessUrl'][:2]:
		sha.save_file(url.strip())
    



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
	"""A function for constructing a variance image of SDSS image

	:param fits_image:
		Input SDSS image (corrected frame type).

	:param filter_name:
		A string of filter name. Options are: 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', and 'sdss_z'.

	:returns name_out_fits: (optional, default: None)
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

	if name_out_fits == None:
		name_out_fits = 'var_%s' % fits_image
	fits.writeto(name_out_fits,sigma_sq_full,header,overwrite=True)

	return name_out_fits


def var_img_GALEX(sci_img,skybg_img,filter_name,name_out_fits=None):
	"""Function for calculating variance image from an input GALEX image

	:param sci_img:
		Input GALEX science image (i.e., background subtracted). 
		This type of image is provided in the GALEX website as indicated with "-intbgsub" (i.e., background subtracted intensity map). 

	:param skybg_img:
		Input sky background image .

	:param filter_name:
		A string of filter name. Options are: 'galex_fuv' and 'galex_nuv'.

	:param name_out_fits: (optional, default: None)
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
								
	if name_out_fits == None:
		name_out_fits = 'var_%s' % sci_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	#return sigma_sq_img_data
	return name_out_fits


def var_img_2MASS(sci_img,skyrms_img=[],skyrms_img_data=[],skyrms_value=None,name_out_fits=None):
	"""Function for deriving a variance image from a 2MASS image. 
	The estimation of uncertainty is based on information from the `2MASS website <http://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/#coadd>`_.

	:param sci_img:
		Input science image (i.e., background subtracted). 

	:param skyrms_img:
		FITS file of the RMS background image. If skyrms_img_data==[] or skyrms_value==None, 
		this parameter should be provided. The background subtraction and calculation of RMS image can be done using 
		the :func:`subtract_background` function.

	:param skyrms_img_data:
		2D array of the RMS background image.

	:param skyrms_value:
		Scalar value of median or mean of the RMS background image. 
		If the 2D data of RMS is not provided, this value will be used.

	:param name_out_fits: (optional, default: None)
		Desired name for the output FITS file.
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

	return name_out_fits


def var_img_WISE(sci_img,unc_img,filter_name,skyrms_img,name_out_fits=None):
	"""Function for constructing variance image from an input WISE image. 
	The uncertainty estimation is based on information from the `WISE website <http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html>`_

	:param sci_img:
		Input science image (i.e., background subtracted).

	:param unc_img:
		Input uncertainty image. This type of image is provided in the 
		`IRSA website <https://irsa.ipac.caltech.edu/applications/wise/>`_ and indicated with '-unc-' keyword.

	:param filter_name: 		
		A string of filter name. Options are: 'wise_w1', 'wise_w2', 'wise_w3', and 'wise_w4'

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
	sigma_sq_img_data = np.square(sci_img_data)*((sigma_0*sigma_0/f_0/f_0) + (0.8483*sigma_magzp*sigma_magzp) + np.square(sigma_src)) ## in unit of DN

	if name_out_fits == None:
		name_out_fits = 'var_%s' % unc_img
	fits.writeto(name_out_fits, sigma_sq_img_data, sci_img_header, overwrite=True)

	return name_out_fits


def var_img_from_unc_img(unc_image, name_out_fits=None):
	"""Function for constructing a variance image from an input of uncertainty image.
	This function simply takes square of the uncertainty image and store it into a FITS file while retaining the 
	header information.

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
	if name_out_fits == None:
		name_out_fits = "var_%s" % unc_image
	fits.writeto(name_out_fits, var_image, header=header, overwrite=True)

	return name_out_fits


def var_img_from_weight_img(wht_image, name_out_fits=None):
	"""Function for constructing a variance image from an input weight (i.e., inverse variance) image.
	This funciton will simply take inverse of the weight image and store it into a new FITS file while 
	retaining the header information.

	:param wht_image:
		Input of weight image (i.e., inverse variance).

	:returns name_out_fits: (optional, default: None)
		Name of output FITS file. If None, a generic name will be used.
	"""
	hdu = fits.open(wht_image)
	data_image = hdu[0].data
	hdu.close()

	var_image = 1.0/data_image
	# store into fits file:
	if name_out_fits == None:
		name_out_fits = "var_%s" % wht_image
	fits.writeto(name_out_fits, var_image, header=header, overwrite=True)

	return name_out_fits



def segm_sep(fits_image=None, thresh=1.5, var=None, minarea=5, deblend_nthresh=32, deblend_cont=0.005):
	import sep 

	hdu = fits.open(fits_image)
	data_img = hdu[0].data 
	hdu.close()

	data_img = data_img.byteswap(inplace=True).newbyteorder()

	if var==None:
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



def subtract_background(fits_image, hdu_idx=0, sigma=3.0, box_size=None, mask_region=[], mask_sources=True, 
	var=None, thresh=1.5, minarea=5, deblend_nthresh=32, deblend_cont=0.005):
	"""Function for estimating 2D background and subtracting it from the input image. This function also produce RMS image. 
	This function adopts the Background2D function from the photutils package. To estimate 2D background, 
	the input image is gridded and sigma clipping is done to each bin/grid. Then 2D interpolation is performed to
	construct 2D background image. A calculation in this function is based on the `Background2D <https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html#photutils.background.Background2D>`_ 
	of the photutils.  

	:param fits_image:
		Input image.

	:param hdu_idx: (int, optional, default: 0)
		The FITS file extension where the image is stored. Default is 0 (HDU0).

	:param sigma: (float, optional, default: 3.0)
		Sigma clipping threshold value.

	:param box_size: (int or array_like, optional, default: None)
		The box size along each axis in the image gridding. The format is: [ny, nx]. If None, both axes will be divided into 10 grids. 

	:param mask_region: (array_like, optional, default: [])
		Region within the image that are going to be excluded. 
		mask_region should be 2D array with the same size as the input image.

	:param mask_sources: (array_like, default: True)
		If True, source detection and segmentation will be performed with SEP (Pyhton version of SExtractor) 
		and the regions associated with the detected sources will be excluded. This help reducing contamination by astronomical sources.

	:param var: (optional, optional, default: None)
		Variance image (in FITS file format) to be used in the sources detection process. This input argument is only relevant if mask_sources=True.

	:param thresh: (float, optional, default: 1.5)
		Detection threshold for the sources detection. If variance image is supplied, the threshold value for a given pixel is 
		interpreted as a multiplicative factor of the uncertainty (i.e. square root of the variance) on that pixel. 
		If var=None, the threshold is taken to be 2.5 percentile of the pixel values in the image. 

	:param minarea: (int, optional, default: 5)
		Minimum number of pixels above threshold triggering detection. 

	:param deblend_nthresh: (float, optional, default: 32)
		The same as deblend_nthresh parameter in the SEP.

	:param deblend_cont: (float, optional, default: 0.005)
		The same as deblend_cont parameter in the SEP.

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
	if box_size == None:
		box_size = [int(dim_y/10),int(dim_x/10)]
	elif box_size != None:
		box_size = box_size

	if mask_sources==True or mask_sources==1: 
		if var==None:
			err = None 
		else:
			err = np.sqrt(var)
		mask_region0 = mask_region_bgmodel(fits_image=fits_image, thresh=thresh, var=var, 
											minarea=minarea, deblend_nthresh=deblend_nthresh, 
											deblend_cont=deblend_cont)

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

		sigma_clip = SigmaClip(sigma=sigma)
		bkg_estimator = MedianBackground()
		bkg = Background2D(data_image, (box_size[0], box_size[1]), filter_size=(3, 3), mask=mask_region1,
							sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

	elif mask_sources==False or mask_sources==0:
		if len(mask_region)==0:
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

	if bool(img_pixsizes)==True:
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
		if bool(kernels)==False and flag_psfmatch==0:
			print ("PSF matching kernels for the following filters are not available by default. In this case, input kernels should be supplied!")
			print (unknown)
			sys.exit()

		if bool(img_unit)==False or bool(img_scale)==False:
			print ("Unit of the following imaging data are not recognized. In this case, input img_unit and img_scale should be provided!")
			print (unknown)
			sys.exit()

def in_kernels(kernels,sorted_filters):
	kernels1 = {}
	if bool(kernels) == False:
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
	if init_x0 == None:
		init_x0 = (data.shape[1]-1)/2
	if init_y0 == None:
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

	x_norm_rot = np.asarray(x_norm)*cos(pa*pi/180.0) + np.asarray(y_norm)*sin(pa*pi/180.0)
	y_norm_rot = -1.0*np.asarray(x_norm)*sin(pa*pi/180.0) + np.asarray(y_norm)*cos(pa*pi/180.0)
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
	if x_cent == None:
		x_cent = (dim_x-1)/2
	if y_cent == None:
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
	if name_out_fits == None:
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


def crop_stars_galregion_fits(input_fits,x_cent=[],y_cent=[],radius=[],name_out_fits=None):
	"""A function for cropping foreground stars within a galaxy's region of interst.
	The input is a FITS file of reduced multiband fluxes maps output of flux_map() function

	:param input_fits:
		The input FITS file.

	:param x_cent and y_cent:
		Arrays containing cenral coordinates of the stars.

	:param radius:
		Arrays containing the estimated radii of the stars. 

	:param name_out_fits:
		Desired name for the output FITS file. If None, a generic name will be used.  
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
	if name_out_fits == None:
		name_out_fits = 'crop_%s' % input_fits
	hdul.writeto(name_out_fits, overwrite=True)

	return output_fits


def crop_image_given_radec(img_name=None, ra=None, dec=None, stamp_size=[], name_out_fits=None):
	"""Function for cropping an image around a given position (RA, DEC)
	"""

	from astropy.wcs import WCS
	from astropy.nddata import Cutout2D

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

	from astropy.wcs import WCS
	from astropy.nddata import Cutout2D

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


def crop_2D_data(in_data=None, data_x_cent=None, data_y_cent=None, new_size_x=None, new_size_y=None):
	del_y = int((new_size_y-1)/2)
	row_start = data_y_cent - del_y
	row_end = data_y_cent + del_y + 1

	del_x = int((new_size_x-1)/2)
	col_start = data_x_cent - del_x
	col_end = data_x_cent + del_x + 1

	new_data = in_data[row_start:row_end, col_start:col_end]

	return new_data




