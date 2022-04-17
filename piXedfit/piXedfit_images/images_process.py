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
from .images_utils import sort_filters, k_lmbd_Fitz1986_LMC, unknown_images, check_avail_kernel, get_psf_fwhm, get_largest_FWHM_PSF, crop_2D_data  

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

	:param img_unit:
		Dictionary containing units of the input imaging data. The options aree: (1)0 if the image is in a flux unit, and (2)1 if the image is in surface brightness unit.

	:param flag_psfmatch:
		Flag stating whether the multiband imaging data have been PSF-matched or not. The options are: (1) 0 means hasn't been PSF-matched, and (2)1 means has been PSF-matched.

	:param flag_reproject:
		Flag stating whether the multiband imaging data have been spatially-resampled and matched in the projection. The options are: (1)0 means not yet, and (2)1 means has been carried out. 

	:param flag_crop:
		Flag stating whether the multiband imaging data have been cropped around the target galaxy. The options are: (1)0 means not yet, and (2)1 means has been cropped. 
		If flag_crop=0, cropping will be done according to the input stamp_size. If flag_crop=1, cropping will not be done. 

	:param img_pixsizes:
		Dictionary containing pixel sizes (in arcsecond) of the input imaging data.

	:param kernels: (optional, default: None)
		Dictionary containing names of FITS files for the kernels to be used for the PSF matching process. 
		If None, internal convolution kernels in **piXedfit** will be used. 
		If external kerenels avaiable, the input should be in dictionary format like the input sci_img, 
		but the number of element should be Nb-1, where Nb is the number of photometric bands.   

	:param gal_ra:
		Coordinate Right Ascension (RA) of the target galaxy.

	:param gal_dec:
		Coordinate Declination (DEC) of the target galaxy.

	:param gal_z: (default: None)
		Galaxy's redshift. This is not used in any calculation during the image processing and calculating fluxes maps
		But only intended to be saved in the heder of the produced FITS file. 

	:param stamp_size: (default: [101,101])
		Desired size for the reduced maps of multiband fluxes. This is a list data type with 2 elements. Accepted struture is: [dim_y,dim_x]. Only relevant if flag_crop=0. 
	
	:param remove_files: (default: True)
		If True, the unnecessary image files produced during the image processing will be removed. This can save disk space. 
		If False, those files will not be removed.   
	"""

	def __init__(self,filters=[],sci_img={},var_img={},img_unit={},flag_psfmatch=0,flag_reproject=0,flag_crop=0, 
				img_pixsizes={},kernels=None,gal_ra=None,gal_dec=None,gal_z=None,stamp_size=[101,101],remove_files=True):

		unknown = unknown_images(filters)
		if len(unknown)>0 and kernels==None and flag_psfmatch==0:
			print ("PSF matching kernels for the following filters are not available by default. In this case, input kernels should be supplied!")
			print (unknown)
			sys.exit()

		# sorting filters:
		sorted_filters = sort_filters(filters)

		self.filters = sorted_filters
		self.sci_img = sci_img
		self.var_img = var_img
		self.img_unit = img_unit
		self.flag_psfmatch = flag_psfmatch
		self.flag_reproject = flag_reproject
		self.flag_crop = flag_crop
		self.img_pixsizes = img_pixsizes	
		self.gal_ra = gal_ra
		self.gal_dec = gal_dec
		self.gal_z = gal_z
		self.stamp_size = stamp_size
		self.remove_files = remove_files

		# kernels:
		if kernels == None:
			kernels = {}
			for ii in range(0,len(sorted_filters)):
				kernels[sorted_filters[ii]] = None
		self.kernels = kernels

	def reduced_stamps(self):
		"""Function within the Class that runs the image processing that includes PSF matching, 
		spatial resampling and reprojection, and cropping around the target galaxy.

		:returns output_stamps:
			Dictionary containing name of postage stamps of reduced multiband images. 
		"""

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
		img_unit = self.img_unit
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
					if img_unit[filters[bb]] == 0: 							# flux
						data_image = hdu[0].data/img_pixsizes[filters[bb]]/img_pixsizes[filters[bb]]
						align_data_image0, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
						align_data_image = align_data_image0*img_pixsizes[filters[idfil_align]]*img_pixsizes[filters[idfil_align]]
					elif img_unit[filters[bb]] == 1:  						# surface brightness
						data_image = hdu[0].data
						align_data_image, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
					name_out = "stamp_%s" % psfmatch_sci_img_name[filters[bb]]
					fits.writeto(name_out,align_data_image,header_for_align,overwrite=True)
					hdu.close()
					align_psfmatch_sci_img_name[filters[bb]] = name_out
					print ("[produce %s]" % name_out)

					#++> variance image
					hdu = fits.open(psfmatch_var_img_name[filters[bb]])
					if img_unit[filters[bb]] == 0:  						# flux
						data_image = hdu[0].data/img_pixsizes[filters[bb]]/img_pixsizes[filters[bb]]
						align_data_image0, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
						align_data_image = align_data_image0*img_pixsizes[filters[idfil_align]]*img_pixsizes[filters[idfil_align]]
					elif img_unit[filters[bb]] == 1:  						# surface brightness
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


	def segmentation_sextractor(self,output_stamps=None,detect_thresh=1.5,detect_minarea=10,deblend_nthresh=32,
								deblend_mincont=0.005):

		"""A function for deriving segmentation maps of a galaxy in multiple bands using the SExtractor through `sewpy <https://sewpy.readthedocs.io/en/latest/>`_ Python wrapper.
		To use this function, sewpy package should be installed. 

		:param output_stamps:
			output_stamps output from the :func:`reduced_stamps`.

		:param detect_thresh: (default: 1.5)
			Detection threshold. This is the same as DETECT_THRESH parameter in SExtractor.

		:param detect_minarea: (default: 10)
			Minimum number of pixels above threshold triggering detection. This is the same as DETECT_MINAREA parameter in SExtractor.

		:param deblend_nthresh: (default: 32)
			Number of deblending sub-thresholds. This is the same as DEBLEND_NTHRESH parameter in SExtractor.

		:param deblend_mincont: (default: 0.005)
			Minimum contrast parameter for deblending. This is the same as DEBLEND_MINCONT parameter in SExtractor.

		:returns segm_map:
			Output segmentation maps.

		:returns segm_map_name:
			Names of output FITS files containing the segmentation maps.
		"""

		import logging
		logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

		import sewpy
		sexpath='sex'

		filters = self.filters
		nbands = len(filters)
		# get set of images that will be used for this analysis
		name_img = []
		for bb in range(0,nbands):
			str_temp = "name_img_%s" % filters[bb]
			name_img.append(output_stamps[str_temp])

		# get image dimension
		hdu = fits.open(name_img[0])
		image_data = hdu[0].data
		dim_y = image_data.shape[0]
		dim_x = image_data.shape[1]
		hdu.close()

		segm_map_name = []
		segm_map = np.zeros((nbands,dim_y,dim_x))
		for bb in range(0,nbands):
			name_segm = "segm_sext_%s" % name_img[bb]
			segm_map_name.append(name_segm)
			sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "FLUX_APER(3)", "FLAGS"],
				config={"DETECT_THRESH":detect_thresh, "DETECT_MINAREA":detect_minarea, "DEBLEND_NTHRESH":deblend_nthresh,
				"DEBLEND_MINCONT":deblend_mincont, "CHECKIMAGE_TYPE":"SEGMENTATION", 
				"CHECKIMAGE_NAME":name_segm},sexpath=sexpath)
			out = sew(name_img[int(bb)])
			print ("SExtractor detection and segemnetation for %s" % name_img[bb])
			print (out["table"])
			# get the segmentation map in form of 2D array:
			hdu = fits.open(name_segm)
			segm_map[bb] = hdu[0].data
			hdu.close()

		return segm_map, segm_map_name


	def galaxy_region(self, name_segmentation_maps):
		"""A function to get initial definition of the galaxy's region of interest by merging together the segmentation maps.

		:param name_segmentation_maps:
			List of the segmentation maps.

		:returns gal_region:
			Final merged-segmentation map.
		"""

		nmaps = len(name_segmentation_maps)

		# image size
		hdu = fits.open(name_segmentation_maps[0])
		dim_y = hdu[0].data.shape[0]
		dim_x = hdu[0].data.shape[1]
		hdu.close()

		gal_region = np.zeros((dim_y,dim_x))

		for bb in range(0,nmaps):
			hdu = fits.open(name_segmentation_maps[bb])
			rows, cols = np.where(hdu[0].data == 1)
			gal_region[rows,cols] = 1
			hdu.close()

		return gal_region


	def flux_map(self, output_stamps=None, gal_region=None, Gal_EBV=0, scale_unit=1.0e-17, 
		mag_zp_2mass=[], unit_spire='Jy_per_beam', name_out_fits=None):
		"""Function for calculating maps of multiband fluxes

		:param output_stamps:
			Dictionary containing reduced multiband images produced by the :func:`reduced_stamps` function.

		:param gal_region:
			2D array containing the galaxy's region of interest. The vlues should be 0 for masked region and 1 for the galaxy's region of interest.
			It can be taken from the output of the :func:`galaxy_region` function. But, user can also defined its own.

		:param Gal_EBV: (optional, default: 0)
			The E(B-V) dust attenuation due to the foreground Galactic dust. This is optional parameter.

		:param scale_unit: (defult: 1.0e-17)
			Normalized unit for the fluxes in the output fits file. The unit is flux density in erg/s/cm^2/Ang.

		:param mag_zp_2mass: (optional, default: [])
			Magnitude zero-points of 2MASS images. Sshoud be in 1D array with three elements: [magzp-j,magzp-h,magzp-k]. This is optional parameter.
			If not given (i.e. [] or empty), the values will be taken from the FITS header information.

		:param unit_spire: (default: 'Jy_per_beam')
			Unit of SPIRE images, in case Herschel/SPIRE image is included in the analysis. Options are: ['Jy_per_beam', 'MJy_per_sr', 'Jy_per_pixel']  

		:param name_out_fits:
			Desired name for the output FITS file.
		"""

		###================ (1) get basic information ===============####
		filters = self.filters
		nbands = len(filters)
		img_pixsizes = self.img_pixsizes
		sci_img = self.sci_img
		var_img = self.var_img	
		gal_ra = self.gal_ra
		gal_dec = self.gal_dec
		gal_z = self.gal_z			

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
		Alambda = {}
		for bb in range(0,nbands):
			if Gal_EBV == 0:
				Alambda[filters[bb]] = 0.0
			else:
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
				map_flux[bb][rows1,cols1] = FLUXZP_2mass*pow(10.0,0.4*((2.5*np.log10(sci_img_data[rows1,cols1]))-MAGZP_2mass))*1.0e+3*Gal_dust_corr_factor  # in erg/s/cm^2/Ang. 

				rows2, cols2 = np.where((gal_region==1) & (sci_img_data<=0))
				map_flux[bb][rows2,cols2] = -1.0*FLUXZP_2mass*pow(10.0,0.4*((2.5*np.log10(-1.0*sci_img_data[rows2,cols2]))-MAGZP_2mass))*1.0e+3*Gal_dust_corr_factor  # in erg/s/cm^2/Ang. 

				#=> flux error
				map_flux_err[bb][rows,cols] = FLUXZP_2mass*pow(10.0,0.4*((2.5*np.log10(np.sqrt(np.absolute(var_img_data[rows,cols]))))-MAGZP_2mass))*1.0e+3*Gal_dust_corr_factor  # in erg/s/cm^2/Ang. 

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
					
				### in case the data is in Jy/pixel or Jy --> this is not surface brightness unit but flux density
				elif unit_spire == 'Jy_per_pixel':
					map_flux[bb][rows,cols] = sci_img_data[rows,cols]*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]		# in erg/s/cm^2/Ang.
					map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*1.0e-23*2.998e+18*Gal_dust_corr_factor/eff_wave[filters[bb]]/eff_wave[filters[bb]]	# in erg/s/cm^2/Ang.
				
				else:
					print ("unit of Herschel images is not recognized!")
					sys.exit()

			#--> other images: assumed they are already in erg/s/cm^2/A
			else:
				map_flux[bb][rows,cols] = sci_img_data[rows,cols]*Gal_dust_corr_factor
				map_flux_err[bb][rows,cols] = np.sqrt(np.absolute(var_img_data[rows,cols]))*Gal_dust_corr_factor

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
		hdr['RA'] = gal_ra
		hdr['DEC'] = gal_dec
		if gal_z != None:
			hdr['z'] = gal_z
		elif gal_z == None:
			hdr['z'] = 0
		hdr['unit'] = scale_unit
		hdr['bunit'] = 'erg/s/cm^2/A'
		hdr['GalEBV'] = Gal_EBV
		hdr['struct'] = '(band,y,x)'
		hdr['fsamp'] = fil_align
		hdr['pixsize'] = final_pix_size
		hdr['fpsfmtch'] = fil_psfmatch
		hdr['psffwhm'] = final_psf_fwhm
		for bb in range(0,nbands):
			str_temp = 'fil%d' % int(bb)
			hdr[str_temp] = filters[int(bb)]
		primary_hdu = fits.PrimaryHDU(header=hdr)
		hdul.append(primary_hdu)
		hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
		hdul.append(fits.ImageHDU(map_flux, name='flux'))
		hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
		hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
		if name_out_fits == None:
			name_out_fits = 'fluxmap.fits'
		hdul.writeto(name_out_fits, overwrite=True)

		return name_out_fits




