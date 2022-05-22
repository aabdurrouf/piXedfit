import numpy as np 
from math import sqrt, pow
import sys, os
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve_fft
from reproject import reproject_exact
from photutils.psf.matching import resize_psf

from ..utils.filtering import cwave_filters
from .images_utils import *  

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']


__all__ = ["images_processing"]


class images_processing:
	"""A Python class for processing of multiband imaging data. The processing basically includes PSF-matching to homogenize the spatial (angular) resolution of the multiband imaging data 
	and spatial-resampling and reprojection to homogenize the pixel size and spatil reprojection of the mulltiband imaging data. A list of imaging data sets that can be handle using this class in the current version of piXedfit 
	can be seen at :ref:`List of imaging data <list-imagingdata>`.  

	:param filters:
		List of photometric filters names in string format. The accepted naming for the filters can be seen using :func:`list_filters` function in the :mod:`utils.filtering` module. 
		It is not mandatory to give the filters names in the wavelength order (from shortest to longest).

	:param sci_img:
		Dictionary containing names of the science images. 

	:param var_img:
		Dictionary containing names of the variance images.

	:param gal_ra:
		Coordinate Right Ascension (RA) of the target galaxy.

	:param gal_dec:
		Coordinate Declination (DEC) of the target galaxy.

	:param img_unit: (optional)
		Unit of pixel value in the multiband images. The input format is python dictionary. 
		This input will only be considered (and required) if the input images are not among the default list of recognized imaging data 
		in piXedfit (i.e. GALEX, SDSS, 2MASS, WISE, Spitzer, and Herschel).  
		The allowed units are: (1)"erg/s/cm2/A", (2) "Jy", and (3) "MJy/sr".

	:param img_scale: (optional)
		Scale of the pixel value with respect to the unit in img_unit. For instance, if image is in unit of MJy, 
		the img_unit can be set to be "Jy" and img_scale is set to be 1e+6. This input is only relevant if the input images are not among the default list of recognized images 
		in piXedfit. The format of this input should be in python dictionary.  

	:param flag_psfmatch:
		Flag stating whether the multiband imaging data have been PSF-matched or not. The options are: (1) 0 means hasn't been PSF-matched, and (2)1 means has been PSF-matched.

	:param flag_reproject:
		Flag stating whether the multiband imaging data have been spatially-resampled and matched in the projection. The options are: (1)0 means not yet, and (2)1 means has been carried out. 

	:param flag_crop:
		Flag stating whether the multiband imaging data have been cropped around the target galaxy. The options are: (1)0 means not yet, and (2)1 means has been cropped. 
		If flag_crop=0, cropping will be done according to the input stamp_size. If flag_crop=1, cropping will not be done. 

	:param img_pixsizes: (optional)
		Pixel sizes (in arcsecond) of the input imaging data. This input should be in dictionary format. 
		If not provided, pixel size will be calculated based on the WCS information in the header of the FITS file.

	:param kernels: (optional)
		Dictionary containing names of FITS files for the kernels to be used for the PSF matching process. 
		If None, internal convolution kernels in **piXedfit** will be used, given that the imaging data is recognized by piXedfit. 
		Otherwise, input kernels should be supplied.  
		If external kerenels avaiable, the input should be in dictionary format like the input sci_img, 
		but the number of element should be Nb-1, where Nb is the number of photometric bands.   

	:param gal_z:
		Galaxy's redshift. This is not used in any calculation during the image processing and calculating fluxes maps
		But only intended to be saved in the heder of the produced FITS file. 

	:param stamp_size: (default: [101,101])
		Desired size for the reduced maps of multiband fluxes. This is a list data type with 2 elements. Accepted struture is: [dim_y,dim_x]. Only relevant if flag_crop=0. 
	
	:param remove_files: (default: True)
		If True, the unnecessary image files produced during the image processing will be removed. This can save disk space. 
		If False, those files will not be removed.   
	"""

	def __init__(self,filters=[],sci_img={},var_img={},gal_ra=None,gal_dec=None,img_unit={},img_scale={},flag_psfmatch=0,
				flag_reproject=0,flag_crop=0, img_pixsizes={}, kernels={}, gal_z=None, stamp_size=[101,101],remove_files=True):

		raise_errors(filters, kernels, flag_psfmatch, img_unit, img_scale)

		# sorting filters:
		sorted_filters = sort_filters(filters)

		kernels = in_kernels(kernels,sorted_filters)
		flux_or_sb = get_flux_or_sb(filters,img_unit)
		img_pixsizes = get_img_pixsizes(img_pixsizes,filters,sci_img,flux_or_sb,flag_psfmatch,flag_reproject)

		self.filters = sorted_filters
		self.sci_img = sci_img
		self.var_img = var_img
		self.gal_ra = gal_ra
		self.gal_dec = gal_dec
		self.flux_or_sb = flux_or_sb
		self.img_unit = img_unit
		self.img_scale = img_scale
		self.flag_psfmatch = flag_psfmatch
		self.flag_reproject = flag_reproject
		self.flag_crop = flag_crop
		self.img_pixsizes = img_pixsizes	
		self.gal_z = gal_z
		self.stamp_size = stamp_size
		self.remove_files = remove_files
		self.kernels = kernels

	def reduced_stamps(self):
		"""Function within the Class that runs the image processing that includes PSF matching, 
		spatial resampling and reprojection, and cropping around the target galaxy.

		:returns output_stamps:
			Dictionary containing name of postage stamps of reduced multiband images. 
		"""

		from operator import itemgetter

		temp_file_names = []

		####============== (1) GET BASIC INPUT ============#####
		# get the list of filters:
		filters = self.filters
		nbands = len(filters)

		# science images:
		sci_img_name = self.sci_img
		if len(sci_img_name) != nbands:
			print ("Number of science images should be the same as the number of filters!")
			sys.exit()

		# variance images:
		var_img_name = self.var_img
		if len(var_img_name) != nbands:
			print ("Number of variance images should be the same as the number of filters!")
			sys.exit()

		flag_psfmatch = self.flag_psfmatch
		flag_reproject = self.flag_reproject
		flag_crop = self.flag_crop
		img_pixsizes = self.img_pixsizes
		flux_or_sb = self.flux_or_sb
		kernels = self.kernels	
		gal_ra = self.gal_ra
		gal_dec = self.gal_dec
		stamp_size = self.stamp_size

		# get index of filter that has image with largest pixel scale
		fil_pixsizes = np.zeros(nbands)
		for bb in range(0,nbands):
			fil_pixsizes[bb] = img_pixsizes[filters[bb]]
		idfil_align, max_val = max(enumerate(fil_pixsizes), key=itemgetter(1))
		####============== End of (1) GET BASIC INPUT ============#####

		if flag_psfmatch == 1:
			psfmatch_sci_img_name = {}
			psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				psfmatch_sci_img_name[filters[bb]] = sci_img_name[filters[bb]]
				psfmatch_var_img_name[filters[bb]] = var_img_name[filters[bb]]
			idfil_psfmatch = 0

		elif flag_psfmatch == 0:
			####============== (2) PSF matching ============#####
			##==> (a) Get filter index with largest PSF size:
			idfil_psfmatch = get_largest_FWHM_PSF(filters=filters)
			print ("[PSF matching to %s]" % filters[idfil_psfmatch])

			##==> (b) Get Kernels:
			# All the kernels should be brought to 0.25"/pixel sampling:
			status_kernel_resize = 1
			kernel_data = {}
			for bb in range(0,nbands):
				if bb != idfil_psfmatch:
					if kernels[filters[bb]] != None:
						hdu = fits.open(kernels[filters[bb]])
						kernel_data[filters[bb]] = hdu[0].data/hdu[0].data.sum()
						hdu.close()

						status_kernel_resize = 0				

					elif kernels[filters[bb]] == None:
						status_kernel = check_avail_kernel(filter_init=filters[bb], filter_final=filters[idfil_psfmatch])

						if status_kernel == 1:
							dir_file = PIXEDFIT_HOME+'/data/kernels/'
							kernel_name0 = 'kernel_%s_to_%s.fits.gz' % (filters[bb],filters[idfil_psfmatch])
							hdu = fits.open(dir_file+kernel_name0)
							kernel_data[filters[bb]] = hdu[0].data/hdu[0].data.sum()
							hdu.close()

							status_kernel_resize = 1			# the kernels are in pix-size=0.25", so need to be adjusted with the pix-size of image 

						elif status_kernel == 0:
							print ("Kernel for PSF matching %s--%s is not available by default, so the input kernels is required for this!")
							sys.exit()

			##==> (c) PSF matching:
			# allocate
			psfmatch_sci_img_name = {}
			psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				psfmatch_sci_img_name[filters[bb]] = None
				psfmatch_var_img_name[filters[bb]] = None

			for bb in range(0,nbands):
				# for filter that has largest PSF
				if bb == idfil_psfmatch:
					psfmatch_sci_img_name[filters[bb]] = sci_img_name[filters[bb]]
					psfmatch_var_img_name[filters[bb]] = var_img_name[filters[bb]]
				# for others
				elif bb != idfil_psfmatch:
					#=> cropping around the target galaxy to minimize calculation time
					dim_y0 = stamp_size[0]
					dim_x0 = stamp_size[1]
					dim_y1 = int(dim_y0*1.5*img_pixsizes[filters[idfil_align]]/img_pixsizes[filters[bb]])
					dim_x1 = int(dim_x0*1.5*img_pixsizes[filters[idfil_align]]/img_pixsizes[filters[bb]])

					#++> science image
					hdu = fits.open(sci_img_name[filters[bb]])[0]
					wcs = WCS(hdu.header)
					gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
					position = (gal_x,gal_y)
					cutout = Cutout2D(hdu.data, position=position, size=(dim_y1,dim_x1), wcs=wcs)
					hdu.data = cutout.data 
					hdu.header.update(cutout.wcs.to_header())
					name_out = "crop_%s" % sci_img_name[filters[bb]]
					hdu.writeto(name_out, overwrite=True)
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

					#++> variance image
					hdu = fits.open(var_img_name[filters[bb]])[0]
					wcs = WCS(hdu.header)
					gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
					position = (gal_x,gal_y)
					cutout = Cutout2D(hdu.data, position=position, size=(dim_y1,dim_x1), wcs=wcs)
					hdu.data = cutout.data 
					hdu.header.update(cutout.wcs.to_header())
					name_out = "crop_%s" % var_img_name[filters[bb]]
					hdu.writeto(name_out, overwrite=True)
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

					# resize/resampling kernel image to match the sampling of the image
					kernel_resize0 = resize_psf(kernel_data[filters[bb]], 0.250, img_pixsizes[filters[bb]], order=3)

					# crop kernel to reduce memory usage: roughly match the size of the cropped image that will be convolved with the kernel
					name_temp = "crop_%s" % sci_img_name[filters[bb]]
					hdu_temp = fits.open(name_temp)
					dim_y_temp = hdu_temp[0].data.shape[0]
					dim_x_temp = hdu_temp[0].data.shape[1]
					if dim_y_temp>=dim_x_temp:
						dim_temp = dim_y_temp
					else:
						dim_temp = dim_x_temp
					hdu_temp.close()

					if kernel_resize0.shape[0] > dim_temp + 5:
						# get coordinate of brightest pixel
						bright_y,bright_x = np.unravel_index(kernel_resize0.argmax(), kernel_resize0.shape)

						# new dimension:
						if (dim_temp + 5)%2 == 0:
							dim_y = dim_temp + 5 - 1
							dim_x = dim_temp + 5 - 1
						else:
							dim_y = dim_temp + 5
							dim_x = dim_temp + 5

						kernel_resize1 = crop_2D_data(in_data=kernel_resize0, data_x_cent=bright_x, 
													data_y_cent=bright_y, new_size_x=dim_x, new_size_y=dim_y)

					else:
						kernel_resize1 = kernel_resize0

					# Normalize the kernel to have integrated value of 1.0
					kernel_resize = kernel_resize1/np.sum(kernel_resize1)

					print ("[PSF matching]")
					#++> science image
					name_fits = "crop_%s" % sci_img_name[filters[bb]]
					hdu = fits.open(name_fits)
					psfmatch_data = convolve_fft(hdu[0].data, kernel_resize, allow_huge=True)
					name_out = "psfmatch_%s" % name_fits
					psfmatch_sci_img_name[filters[bb]] = name_out
					fits.writeto(name_out,psfmatch_data,hdu[0].header, overwrite=True)
					hdu.close()
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

					#++> variance image
					name_fits = "crop_%s" % var_img_name[filters[int(bb)]]
					hdu = fits.open(name_fits)
					psfmatch_data = convolve_fft(hdu[0].data, kernel_resize, allow_huge=True)
					name_out = "psfmatch_%s" % name_fits
					psfmatch_var_img_name[filters[bb]] = name_out
					fits.writeto(name_out,psfmatch_data,hdu[0].header,overwrite=True)
					hdu.close()
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

			####============== End of (c) PSF matching ============#####

		if flag_reproject==1 and flag_crop==0:
			# Just crop the images
			align_psfmatch_sci_img_name = {}
			align_psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				#++> science image
				hdu = fits.open(psfmatch_sci_img_name[filters[bb]])[0]
				wcs = WCS(hdu.header)
				gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
				position = (gal_x,gal_y)
				cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
				hdu.data = cutout.data
				hdu.header.update(cutout.wcs.to_header())
				name_out = 'stamp_%s' % psfmatch_sci_img_name[filters[bb]]
				hdu.writeto(name_out, overwrite=True)
				align_psfmatch_sci_img_name[filters[bb]] = name_out
				print ("[produce %s]" % name_out)

				#++> variance image
				hdu = fits.open(psfmatch_var_img_name[filters[bb]])[0]
				wcs = WCS(hdu.header)
				gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
				position = (gal_x,gal_y)
				cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
				hdu.data = cutout.data
				hdu.header.update(cutout.wcs.to_header())
				name_out = 'stamp_%s' % psfmatch_var_img_name[filters[bb]]
				hdu.writeto(name_out, overwrite=True)
				align_psfmatch_var_img_name[filters[bb]] = name_out
				print ("[produce %s]" % name_out)


		if flag_reproject==1 and flag_crop==1:
			align_psfmatch_sci_img_name = {}
			align_psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				#++> science image
				hdu = fits.open(psfmatch_sci_img_name[filters[bb]])
				name_out = 'stamp_%s' % psfmatch_sci_img_name[filters[bb]]
				fits.writeto(name_out, hdu[0].data, header=hdu[0].header, overwrite=True)
				align_psfmatch_sci_img_name[filters[bb]] = name_out
				print ("[produce %s]" % name_out)
				hdu.close()

				#++> variance image
				hdu = fits.open(psfmatch_var_img_name[filters[bb]])
				name_out = 'stamp_%s' % psfmatch_var_img_name[filters[bb]]
				fits.writeto(name_out, hdu[0].data, header=hdu[0].header, overwrite=True)
				align_psfmatch_var_img_name[filters[bb]] = name_out
				print ("[produce %s]" % name_out)
				hdu.close()


		if flag_reproject==0:
			####============== (3) Spatial reprojection and resampling ============#####
			print ("[images reprojection and resampling]")
			print ("align images to the reprojection and sampling of %s: %lf arcsec/pixel" % (filters[idfil_align],img_pixsizes[filters[idfil_align]]))
			align_psfmatch_sci_img_name = {}
			align_psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				align_psfmatch_sci_img_name[filters[bb]] = None
				align_psfmatch_var_img_name[filters[bb]] = None

			# for image with largest pixel size: just crop the image
			#++> science image
			hdu = fits.open(psfmatch_sci_img_name[filters[idfil_align]])[0]
			wcs = WCS(hdu.header)
			gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
			position = (gal_x,gal_y)
			cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
			hdu.data = cutout.data
			hdu.header.update(cutout.wcs.to_header())
			name_out = 'stamp_%s' % psfmatch_sci_img_name[filters[idfil_align]]
			hdu.writeto(name_out, overwrite=True)
			align_psfmatch_sci_img_name[filters[idfil_align]] = name_out
			print ("[produce %s]" % name_out)

			#++> variance image
			hdu = fits.open(psfmatch_var_img_name[filters[idfil_align]])[0]
			wcs = WCS(hdu.header)
			gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
			position = (gal_x,gal_y)
			cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
			hdu.data = cutout.data
			hdu.header.update(cutout.wcs.to_header())
			name_out = 'stamp_%s' % psfmatch_var_img_name[filters[idfil_align]]
			hdu.writeto(name_out, overwrite=True)
			align_psfmatch_var_img_name[filters[idfil_align]] = name_out
			print ("[produce %s]" % name_out)


			# for other filters:
			# get header of stamp image that has largest pixel scale
			hdu = fits.open(align_psfmatch_sci_img_name[filters[idfil_align]])
			header_for_align = hdu[0].header
			hdu.close()

			for bb in range(0,nbands):
				if bb != idfil_align:
					#++> science image
					hdu = fits.open(psfmatch_sci_img_name[filters[bb]])
					if flux_or_sb[filters[bb]] == 0: 							# flux
						data_image = hdu[0].data/img_pixsizes[filters[bb]]/img_pixsizes[filters[bb]]
						align_data_image0, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
						align_data_image = align_data_image0*img_pixsizes[filters[idfil_align]]*img_pixsizes[filters[idfil_align]]
					elif flux_or_sb[filters[bb]] == 1:  						# surface brightness
						data_image = hdu[0].data
						align_data_image, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
					name_out = "stamp_%s" % psfmatch_sci_img_name[filters[bb]]
					fits.writeto(name_out,align_data_image,header_for_align,overwrite=True)
					hdu.close()
					align_psfmatch_sci_img_name[filters[bb]] = name_out
					print ("[produce %s]" % name_out)

					#++> variance image
					hdu = fits.open(psfmatch_var_img_name[filters[bb]])
					if flux_or_sb[filters[bb]] == 0:  						# flux
						data_image = hdu[0].data/img_pixsizes[filters[bb]]/img_pixsizes[filters[bb]]
						align_data_image0, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
						align_data_image = align_data_image0*img_pixsizes[filters[idfil_align]]*img_pixsizes[filters[idfil_align]]
					elif flux_or_sb[filters[bb]] == 1:  						# surface brightness
						data_image = hdu[0].data
						align_data_image, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
					name_out = "stamp_%s" % psfmatch_var_img_name[filters[bb]]
					fits.writeto(name_out,align_data_image,header_for_align,overwrite=True)
					hdu.close()
					align_psfmatch_var_img_name[filters[bb]] = name_out
					print ("[produce %s]" % name_out)

			####============== (End of 3) Reprojection and resampling ============#####

		# outputs
		output_stamps = {}
		for bb in range(0,nbands):
			str_temp = "name_img_%s" % filters[bb]
			output_stamps[str_temp] = align_psfmatch_sci_img_name[filters[bb]]

			str_temp = "name_var_%s" % filters[bb]
			output_stamps[str_temp] = align_psfmatch_var_img_name[filters[bb]]

			output_stamps['idfil_align'] = idfil_align
			output_stamps['idfil_psfmatch'] = idfil_psfmatch

		# remove files
		if self.remove_files==True:
			for zz in range(0,len(temp_file_names)):
				os.system("rm %s" % temp_file_names[zz])

		return output_stamps


	def segmentation_sep(self, output_stamps, thresh=1.5, minarea=30, deblend_nthresh=32, deblend_cont=0.005):
		"""Get segmentation maps of a galaxy in multiple bands using the SEP (a Python version of the SExtractor). 

		:param output_stamps:
			output_stamps output from the :func:`reduced_stamps` method.

		:param thresh: (float, optional, default: 1.5)
			Detection threshold for the sources detection. If variance image is supplied, the threshold value for a given pixel is 
			interpreted as a multiplicative factor of the uncertainty (i.e. square root of the variance) on that pixel. 
			If var=None, the threshold is taken to be 2.5 percentile of the pixel values in the image.

		:param minarea: (float, optional, default: 5)
			Minimum number of pixels (above threshold) required for a detected object. 

		:param deblend_nthresh: (optional, default: 32)
			The same as deblend_nthresh parameter in the SEP.

		:param deblend_cont: (float, optional, default: 0.005)
			The same as deblend_cont parameter in the SEP.

		:returns segm_maps:
			Output segmentation maps.
		"""

		import sep 

		filters = self.filters
		nbands = len(filters)

		# get input science images
		name_img = []
		for bb in range(0,nbands):
			str_temp = "name_img_%s" % filters[bb]
			name_img.append(output_stamps[str_temp])

		# get input variance images
		name_var = []
		for bb in range(0,nbands):
			str_temp = "name_var_%s" % filters[bb]
			name_var.append(output_stamps[str_temp])

		segm_maps = []
		for bb in range(0,nbands):
			# data of science image
			hdu = fits.open(name_img[bb])
			data_img = hdu[0].data 
			hdu.close()

			# date of variance image
			hdu = fits.open(name_var[bb])
			data_var = hdu[0].data 
			hdu.close()

			data_img = data_img.byteswap(inplace=True).newbyteorder()
			data_var = data_var.byteswap(inplace=True).newbyteorder()

			rows,cols = np.where((np.isnan(data_var)==False) & (np.isinf(data_var)==False))
			med_var = np.median(data_var[rows,cols])
			med_err = np.sqrt(med_var)

			objects, segm_map0 = sep.extract(data=data_img, thresh=thresh, err=med_err, minarea=minarea, 
											deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, 
											segmentation_map=True)

			if np.max(segm_map0)>1:
				dim_y, dim_x = data_img.shape[0], data_img.shape[1]
				y_cent = (dim_y-1)/2
				x_cent = (dim_x-1)/2

				segm_map1 = np.zeros((dim_y,dim_x))
				rows, cols = np.where(segm_map0==segm_map0[int(y_cent)][int(x_cent)])
				segm_map1[rows,cols] = 1
			else:
				segm_map1 = segm_map0

			segm_maps.append(segm_map1)


		return segm_maps


	def galaxy_region(self, segm_maps=[], use_ellipse=False, x_cent=None,
	 					y_cent=None, ell=0, pa=45.0, radius_sma=30.0):
		"""Define galaxy's region of interest for further analysis.

		:param segm_maps: (list of string, optional, default: [])
			Input segmentation maps in a list format. If the galaxy's region is to be defined based 
			on segmentation maps obtained with SEP, this input argument is required.

		:param use_ellipse: (boolean, optional, default: False)
			Alternative of defining galaxy's region using elliptical aperture centered at the target galaxy.
			Set use_ellipse=True if you want to use this option.

		:param x_cent: (float, optional, default: None)
			x coordinate of the ellipse center. If x_cent=None, the ellipse center is assumed 
			to be the same as the image center. 

		:param y_cent: (float, optional, default: None)
			y coordinate of the ellipse center. If y_cent=None, the ellipse center is assumed 
			to be the same as the image center.

		:param ell: (float, optional, default: 0.0)
			Ellipticity of the elliptical aperture.

		:param pa: (float, optional, default: 45.0)
			Position angle of the elliptical aperture.

		:param radius_sma: (float, optional, default: 30.0)
			Radal distance along the semi-major axis of the elliptical aperture. This radius is in pixel unit.

		:returns gal_region: (2D array)
			Output galaxy's region of interest.
		"""
		stamp_size = self.stamp_size

		dim_y = int(stamp_size[0])
		dim_x = int(stamp_size[1])

		gal_region = np.zeros((dim_y,dim_x))


		if use_ellipse==False or use_ellipse==0:
			if len(segm_maps)>0:
				# use the segmentation maps: merge them
				for bb in range(0,len(segm_maps)):
					rows, cols = np.where(segm_maps[bb] == 1)
					gal_region[rows,cols] = 1
			else:
				print ("In case of not using elliptical aperture, segm_maps input is required!")
				sys.exit()

		elif use_ellipse==True or use_ellipse==1:
			# use elliptical aperture
			if y_cent == None:
				y_cent = (dim_y-1)/2
			if x_cent == None:
				x_cent = (dim_x-1)/2

			x = np.linspace(0,dim_x-1,dim_x)
			y = np.linspace(0,dim_y-1,dim_y)
			xx, yy = np.meshgrid(x,y)
			xx_norm, yy_norm = xx-x_cent, yy-y_cent

			data2D_sma = ellipse_sma(ell,pa,xx_norm,yy_norm)

			rows,cols = np.where(data2D_sma<=radius_sma)

			gal_region[rows,cols] = 1

		else:
			print ("The inputted use_ellipse is not recognized!")
			sys.exit()

		return gal_region


	def flux_map(self, output_stamps, gal_region, Gal_EBV=None, scale_unit=1.0e-17, 
		mag_zp_2mass=[], unit_spire='Jy_per_beam', name_out_fits=None):
		"""Function for calculating maps of multiband fluxes

		:param output_stamps:
			Dictionary containing reduced multiband images produced by the :func:`reduced_stamps` function.

		:param gal_region:
			2D array containing the galaxy's region of interest. The vlues should be 0 for masked region and 1 for the galaxy's region of interest.
			It can be taken from the output of the :func:`galaxy_region` function. But, user can also defined its own.

		:param Gal_EBV: (float, optional, default:None)
			The E(B-V) dust attenuation due to the foreground Galactic dust. This is optional parameter.

		:param scale_unit: (float, optional, defult: 1.0e-17)
			Normalized unit for the fluxes in the output fits file. The unit is flux density in erg/s/cm^2/Ang.

		:param mag_zp_2mass: (float array_like, optional, default: [])
			Magnitude zero-points of 2MASS images. Sshoud be in 1D array with three elements: [magzp-j,magzp-h,magzp-k]. This is optional parameter.
			If not given (i.e. [] or empty), the values will be taken from the FITS header information.

		:param unit_spire: (string, optional, default: 'Jy_per_beam')
			Unit of SPIRE images, in case Herschel/SPIRE image is included in the analysis. Options are: ['Jy_per_beam', 'MJy_per_sr', 'Jy_per_pixel']  

		:param name_out_fits: (string, optional, default: None)
			Desired name for the output FITS file. If None, a generic name will be used.
		"""

		from operator import itemgetter

		###================ (1) get basic information ===============####
		filters = self.filters
		nbands = len(filters)
		img_pixsizes = self.img_pixsizes
		sci_img = self.sci_img
		var_img = self.var_img	
		gal_ra = self.gal_ra
		gal_dec = self.gal_dec
		gal_z = self.gal_z

		flag_psfmatch = self.flag_psfmatch
		flag_reproject = self.flag_reproject

		img_unit = self.img_unit
		img_scale = self.img_scale		

		# get image dimension and example of stamp_img
		str_temp = "name_img_%s" % filters[0]
		name_img = output_stamps[str_temp]
		hdu = fits.open(name_img)
		stamp_img = hdu[0].data
		dim_y = stamp_img.shape[0]
		dim_x = stamp_img.shape[1]
		stamp_hdr = hdu[0].header
		hdu.close()

		# some info
		idfil_align = output_stamps['idfil_align']
		idfil_psfmatch = output_stamps['idfil_psfmatch']
		fil_align = filters[int(idfil_align)]
		fil_psfmatch = filters[int(idfil_psfmatch)]
		final_pix_size = img_pixsizes[fil_align]
		psf_fwhm1 = get_psf_fwhm(filters=[fil_psfmatch])
		final_psf_fwhm = psf_fwhm1[0]

		# get effective/central wavelength of the filters
		photo_wave = cwave_filters(filters)

		eff_wave = {}
		for bb in range(0,nbands):
			str_temp = "cw_%s" % filters[bb]
			eff_wave[filters[bb]] = photo_wave[bb]

		# calculate Alambda for Galactic dust extinction correction
		if Gal_EBV == None:
			Gal_EBV = EBV_foreground_dust(gal_ra, gal_dec)

		Alambda = {}
		for bb in range(0,nbands):
			Alambda[filters[bb]] = k_lmbd_Fitz1986_LMC(eff_wave[filters[bb]])*Gal_EBV

		# get index of filter that has image with largest pixel scale:
		fil_pixsizes = np.zeros(nbands)
		for bb in range(0,nbands):
			fil_pixsizes[bb] = img_pixsizes[filters[bb]]
		idfil_align, max_val = max(enumerate(fil_pixsizes), key=itemgetter(1))
		###================ End of (1) get basic information ===============####

		###================ (2) Calculation of flux map ===============####
		# allocate memory
		map_flux = np.zeros((nbands,dim_y,dim_x)) - 99.0
		map_flux_err = np.zeros((nbands,dim_y,dim_x)) - 99.0
		for bb in range(0,nbands):
			# get science image
			str_temp = "name_img_%s" % filters[bb]
			hdu = fits.open(output_stamps[str_temp])
			sci_img_data = hdu[0].data
			hdu.close()

			# get variance image
			str_temp = "name_var_%s" % filters[bb]
			hdu = fits.open(output_stamps[str_temp]) 
			var_img_data = hdu[0].data
			hdu.close()

			#==> get magnitude zero-point and fluxes for zero-magnitudes zero point conversion: of 2MASS:
			if filters[bb]=='2mass_j' or filters[bb]=='2mass_h' or filters[bb]=='2mass_k':
				# get magnitude zero-point
				if len(mag_zp_2mass) == 0:
					name_init_image = sci_img[filters[bb]]
					hdu = fits.open(name_init_image)
					MAGZP_2mass = float(hdu[0].header["MAGZP"])
					hdu.close()
				else:
					if filters[bb]=='2mass_j':
						MAGZP_2mass = mag_zp_2mass[0]
					elif filters[bb]=='2mass_h':
						MAGZP_2mass = mag_zp_2mass[1]
					elif filters[bb]=='2mass_k':
						MAGZP_2mass = mag_zp_2mass[2]

				# get flux at magnitude zero-point
				if filters[bb]=='2mass_j':
					FLUXZP_2mass = 3.129e-13              #in W/cm^2/micron
				elif filters[bb]=='2mass_h':
					FLUXZP_2mass = 1.133E-13
				elif filters[bb]=='2mass_k':
					FLUXZP_2mass = 4.283E-14

			#==> get DN to Jy correction factors for WISE bands
			if filters[bb]=='wise_w1' or filters[bb]=='wise_w2' or filters[bb]=='wise_w3' or filters[bb]=='wise_w4':
				if filters[bb]=='wise_w1':
					DN_to_Jy = 1.9350e-06
				elif filters[bb]=='wise_w2':
					DN_to_Jy = 2.7048E-06
				elif filters[bb]=='wise_w3':
					DN_to_Jy = 2.9045e-06
				elif filters[bb]=='wise_w4':
					DN_to_Jy = 5.2269E-05

			#==> get beam area of Herschel SPIRE
			if filters[bb]=='herschel_spire_250' or filters[bb]=='herschel_spire_350' or filters[bb]=='herschel_spire_500':
				# get beam area in arcsec^2:
				if filters[bb]=='herschel_spire_250':
					beam_area = 469.3542
				elif filters[bb]=='herschel_spire_350':
					beam_area = 831.275
				elif filters[bb]=='herschel_spire_500':
					beam_area = 1804.3058


			##====> calculation of maps of fluxes and flux uncertainties:

			# galaxy's region
			rows, cols = np.where(gal_region == 1)

			# correction factor for Galactic dust extinction
			Gal_dust_corr_factor = pow(10.0,0.4*Alambda[filters[bb]])

			#--> GALEX/FUV
			if filters[bb] == 'galex_fuv':
				map_flux[bb][rows,cols] = sci_img_data[rows,cols]*1.40e-15*Gal_dust_corr_factor   # in erg/s/cm^2/Ang.
				map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*1.40e-15*Gal_dust_corr_factor   # in erg/s/cm^2/Ang.

			#--> GALEX/NUV
			elif filters[bb] == 'galex_nuv':
				map_flux[bb][rows,cols] = sci_img_data[rows,cols]*2.06e-16*Gal_dust_corr_factor   # in erg/s/cm^2/Ang.
				map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*2.06e-16*Gal_dust_corr_factor   # in erg/s/cm^2/Ang.
				
			#--> SDSS
			elif filters[bb]=='sdss_u' or filters[bb]=='sdss_g' or filters[bb]=='sdss_r' or filters[bb]=='sdss_i' or filters[bb]=='sdss_z': 
				f0 = sci_img_data[rows,cols]*0.000003631                              # in Jy
				flux0 = f0*0.00002998/eff_wave[filters[bb]]/eff_wave[filters[bb]]     # in erg/s/cm^2/Ang.
				map_flux[bb][rows,cols] = flux0*Gal_dust_corr_factor				

				f0 = np.sqrt(np.absolute(var_img_data[rows,cols]))*0.000003631        # in Jy
				flux0 = f0*0.00002998/eff_wave[filters[bb]]/eff_wave[filters[bb]]     # in erg/s/cm^2/Ang.
				map_flux_err[bb][rows,cols] = flux0*Gal_dust_corr_factor

			#--> 2MASS
			# the image is in DN
			elif filters[bb]=='2mass_j' or filters[bb]=='2mass_h' or filters[bb]=='2mass_k':
				#=> flux
				rows1, cols1 = np.where((gal_region==1) & (sci_img_data>0))
				map_flux[bb][rows1,cols1] = FLUXZP_2mass*np.power(10.0,0.4*((2.5*np.log10(sci_img_data[rows1,cols1]))-MAGZP_2mass))*1.0e+3*Gal_dust_corr_factor  # in erg/s/cm^2/Ang. 

				rows2, cols2 = np.where((gal_region==1) & (sci_img_data<=0))
				map_flux[bb][rows2,cols2] = -1.0*FLUXZP_2mass*np.power(10.0,0.4*((2.5*np.log10(-1.0*sci_img_data[rows2,cols2]))-MAGZP_2mass))*1.0e+3*Gal_dust_corr_factor  # in erg/s/cm^2/Ang. 

				#=> flux error
				map_flux_err[bb][rows,cols] = FLUXZP_2mass*np.power(10.0,0.4*((2.5*np.log10(np.sqrt(np.absolute(var_img_data[rows,cols]))))-MAGZP_2mass))*1.0e+3*Gal_dust_corr_factor  # in erg/s/cm^2/Ang. 

			#--> Spitzer: IRAC and MIPS
			# Spitzer image is in Mjy/sr
			elif filters[bb]=='spitzer_irac_36' or filters[bb]=='spitzer_irac_45' or filters[bb]=='spitzer_irac_58' or filters[bb]=='spitzer_irac_80' or filters[bb]=='spitzer_mips_24' or filters[bb]=='spitzer_mips_70' or filters[bb]=='spitzer_mips_160':
				f0 = sci_img_data[rows,cols]*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]	# in unit of Jy
				map_flux[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   													# in erg/s/cm^2/Ang.

				f0 = np.sqrt(np.absolute(var_img_data[rows,cols]))*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   # in unit of Jy
				map_flux_err[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   							 # in erg/s/cm^2/Ang.

			#--> WISE 
			# image is in DN. DN_to_Jy is conversion factor from DN to Jy
			elif filters[bb]=='wise_w1' or filters[bb]=='wise_w2' or filters[bb]=='wise_w3' or filters[bb]=='wise_w4':	
				map_flux[bb][rows,cols] = sci_img_data[rows,cols]*DN_to_Jy*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   							# in erg/s/cm^2/Ang.						
				map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*DN_to_Jy*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   	# in erg/s/cm^2/Ang.

			#--> Herschel PACS
			# image is in Jy/pixel or Jy --> this is not surface brightness unit but flux density
			elif filters[bb]=='herschel_pacs_70' or filters[bb]=='herschel_pacs_100' or filters[bb]=='herschel_pacs_160':
				map_flux[bb][rows,cols] = sci_img_data[rows,cols]*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   								# in erg/s/cm^2/Ang.
				map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]		# in erg/s/cm^2/Ang.

			#--> Herschel SPIRE
			elif filters[bb]=='herschel_spire_250' or filters[bb]=='herschel_spire_350' or filters[bb]=='herschel_spire_500':
				if unit_spire == 'Jy_per_beam':
					# image is in Jy/beam -> surface brightness unit
					# Jy/pixel = Jy/beam x beam/arcsec^2 x arcsec^2/pixel  -> flux density
					f0 = sci_img_data[rows,cols]*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]/beam_area  	# in Jy
					map_flux[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   			# in erg/s/cm^2/Ang.
					
					f0 = np.sqrt(np.absolute(var_img_data[rows,cols]))*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]/beam_area  # in Jy
					map_flux_err[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]			# in erg/s/cm^2/Ang.
				
				### in case the data is in Mjy/sr
				elif unit_spire == 'MJy_per_sr':
					f0 = sci_img_data[rows,cols]*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]	# in unit of Jy
					map_flux[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   			# in erg/s/cm^2/Ang.
					
					f0 = np.sqrt(np.absolute(var_img_data[rows,cols]))*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   # in unit of Jy
					map_flux_err[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]   		# in erg/s/cm^2/Ang.
					
				### in case the data is in Jy/pixel or Jy
				elif unit_spire == 'Jy_per_pixel':
					map_flux[bb][rows,cols] = sci_img_data[rows,cols]*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]		# in erg/s/cm^2/Ang.
					map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]	# in erg/s/cm^2/Ang.
				
				else:
					print ("unit of Herschel images is not recognized!")
					sys.exit()

			else:
				if img_unit[filters[bb]]=='erg/s/cm2/A':
					map_flux[bb][rows,cols] = sci_img_data[rows,cols]*Gal_dust_corr_factor*img_scale[filters[bb]]
					map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*Gal_dust_corr_factor*img_scale[filters[bb]]

				elif img_unit[filters[bb]]=='Jy':
					map_flux[bb][rows,cols] = sci_img_data[rows,cols]*1.0e-23*2.998e+18*Gal_dust_corr_factor*img_scale[filters[bb]]/eff_wave[filters[bb]]/eff_wave[filters[bb]]
					map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*1.0e-23*2.998e+18*Gal_dust_corr_factor*img_scale[filters[bb]]/eff_wave[filters[bb]]/eff_wave[filters[bb]]

				elif img_unit[filters[bb]]=='MJy/sr':
					f0 = sci_img_data[rows,cols]*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]							# in unit of Jy
					map_flux[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor*img_scale[filters[bb]]/eff_wave[filters[bb]]/eff_wave[filters[bb]]   			# in erg/s/cm^2/Ang.

					f0 = np.sqrt(np.absolute(var_img_data[rows,cols]))*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   	# in unit of Jy
					map_flux_err[bb][rows,cols] = f0*1.0e-23*2.998e+18*Gal_dust_corr_factor*img_scale[filters[bb]]/eff_wave[filters[bb]]/eff_wave[filters[bb]]   		# in erg/s/cm^2/Ang.

				else:
					print ("Inputted img_unit[%s] is not recognized!" % filters[bb])
					sys.exit()

			### end for bb: nbands


		# scaling the flux maps:
		#print ("[scaling the flux map to a unit of %e erg/s/cm^2/Ang.]" % scale_unit)
		map_flux = map_flux/scale_unit
		map_flux_err = map_flux_err/scale_unit

		###================ End of (2) Calculation of flux map ===============####

		###================ (3) Store into fits file ===============####
		# get header of one of the stamp image:
		str_temp = "name_img_%s" % filters[0]
		hdu = fits.open(output_stamps[str_temp])
		header_stamp_image = hdu[0].header
		hdu.close()

		hdul = fits.HDUList()
		hdr = fits.Header()
		hdr['nfilters'] = nbands
		if gal_ra != None:
			hdr['RA'] = gal_ra

		if gal_dec != None:
			hdr['DEC'] = gal_dec

		if gal_z != None:
			hdr['z'] = gal_z
		elif gal_z == None:
			hdr['z'] = 0

		hdr['unit'] = scale_unit
		hdr['bunit'] = 'erg/s/cm^2/A'
		hdr['GalEBV'] = Gal_EBV
		hdr['struct'] = '(band,y,x)'

		if flag_reproject == 0:
			hdr['fsamp'] = fil_align
		if final_pix_size > 0:
			hdr['pixsize'] = final_pix_size
		if flag_psfmatch == 0:
			hdr['fpsfmtch'] = fil_psfmatch
			hdr['psffwhm'] = final_psf_fwhm

		hdr['specphot'] = 0
		for bb in range(0,nbands):
			str_temp = 'fil%d' % bb
			hdr[str_temp] = filters[bb]

		hdul.append(fits.ImageHDU(data=map_flux, header=hdr, name='flux'))
		hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
		hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
		hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
		
		if name_out_fits == None:
			name_out_fits = 'fluxmap.fits'
		hdul.writeto(name_out_fits, overwrite=True)

		return name_out_fits




