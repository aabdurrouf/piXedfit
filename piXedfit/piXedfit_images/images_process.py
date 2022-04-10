import numpy as np 
import math
import sys, os
#import operator
#import photutils
#import astropy
from numpy import unravel_index
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve_fft
from astropy.stats import SigmaClip
from scipy import interpolate
from reproject import reproject_exact
from photutils import Background2D, MedianBackground
from photutils import CosineBellWindow, HanningWindow, create_matching_kernel
from photutils.psf.matching import resize_psf

from ..utils.filtering import cwave_filters
from .images_utils import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']


__all__ = ["images_processing"]


class images_processing:
	"""A Python class that can be used for processing of multiband imaging data. The processing basically includes PSF-matching to homogenize the spatial (angular) resolution of the multiband imaging data 
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
		Dictionary containing names of FITS files for the kernels to be used for the PSF matching process. If None, internal convolution kernels in **piXedfit** will be used. 
		If external kerenels avaiable, the input should be in dictionary format like the input sci_img, but the number of element should be Nb-1, where Nb is the number of photometric bands.   

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

	def __init__(self, filters=[],sci_img={},var_img={},img_unit={},flag_psfmatch=0, flag_reproject=0, flag_crop=0, 
		img_pixsizes={},kernels=None,psf_fwhm=None,gal_ra=None,gal_dec=None,gal_z=None,stamp_size=[101,101],
		remove_files=True):

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
		#self.kernels = kernels	
		self.gal_ra = gal_ra
		self.gal_dec = gal_dec
		self.gal_z = gal_z
		self.stamp_size = stamp_size
		self.remove_files = remove_files
		#self.nbands = len(filters)

		# kernels:
		if kernels == None:
			kernels = {}
			for ii in range(0,len(sorted_filters)):
				kernels[sorted_filters[ii]] = None
		self.kernels = kernels
		self.psf_fwhm = psf_fwhm

	def reduced_stamps(self):
		"""Function within the Class which runs the image processing that includes PSF matching, spatial resampling and reprojection, and cropping around the target galaxy.

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
		sci_img_data = {}
		for bb in range(0,nbands):
			hdu = fits.open(sci_img_name[filters[bb]])
			sci_img_data[filters[bb]] = hdu[0].data
			hdu.close()

		# variance images:
		var_img_name = self.var_img
		if len(var_img_name) != nbands:
			print ("Number of variance images should be the same as the number of filters!")
			sys.exit()
		var_img_data = {}
		for bb in range(0,nbands):
			var_img_data[filters[bb]] = None

		flag_psfmatch = self.flag_psfmatch
		flag_reproject = self.flag_reproject
		flag_crop = self.flag_crop
		img_pixsizes = self.img_pixsizes
		img_unit = self.img_unit
		kernels = self.kernels	
		gal_ra = self.gal_ra
		gal_dec = self.gal_dec
		stamp_size = self.stamp_size
		#sewpypath = self.sewpypath
		psf_fwhm = self.psf_fwhm

		# get index of filter that has image with largest pixel scale:
		max_val = -10.0
		for bb in range(0,nbands):
			if img_pixsizes[filters[bb]] >= max_val:
				max_val = img_pixsizes[filters[bb]]
				idfil_align = bb
		####============== End of (1) GET BASIC INPUT ============#####

		if flag_psfmatch == 1:
			psfmatch_sci_img_data = {}
			psfmatch_sci_img_name = {}
			psfmatch_var_img_data = {}
			psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				psfmatch_sci_img_name[filters[bb]] = sci_img_name[filters[bb]]
				psfmatch_sci_img_data[filters[bb]] = sci_img_data[filters[bb]]
				psfmatch_var_img_name[filters[bb]] = var_img_name[filters[bb]]
				psfmatch_var_img_data[filters[bb]] = var_img_data[filters[bb]]

		if flag_psfmatch == 0:
			####============== (2) PSF matching ============#####
			# get PSF sizes of WFCAM
			seeing_wfcam_y = None
			seeing_wfcam_j = None
			seeing_wfcam_h = None
			seeing_wfcam_k = None
			for bb in range(0,nbands):
				if filters[bb] == 'wfcam_y':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_wfcam_y = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'wfcam_j':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_wfcam_j = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'wfcam_h':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_wfcam_h = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'wfcam_k':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_wfcam_k = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()

			# get PSF sizes of VIRCAM:
			seeing_vircam_z = None
			seeing_vircam_y = None
			seeing_vircam_j = None
			seeing_vircam_h = None
			seeing_vircam_ks = None
			for bb in range(0,nbands):
				if filters[bb] == 'vircam_z':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_vircam_z = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'vircam_y':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_vircam_y = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'vircam_j':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_vircam_j = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'vircam_h':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_vircam_h = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()
				elif filters[bb] == 'vircam_ks':
					hdu = fits.open(sci_img_name[filters[bb]])
					seeing_vircam_ks = float(hdu[0].header['SEEING'])*img_pixsizes[filters[bb]]
					hdu.close()

			######==> (a) Get filter index which has largest PSF size:
			idfil_psfmatch = get_largest_FWHM_PSF(filters=filters, col_fwhm_psf=psf_fwhm, seeing_wfcam_y=seeing_wfcam_y, seeing_wfcam_j=seeing_wfcam_j, 
													seeing_wfcam_h=seeing_wfcam_h, seeing_wfcam_k=seeing_wfcam_k,
													seeing_vircam_z=seeing_vircam_z, seeing_vircam_y=seeing_vircam_y,
													seeing_vircam_j=seeing_vircam_j, seeing_vircam_h=seeing_vircam_h,
													seeing_vircam_ks=seeing_vircam_ks)
			print ("[PSF matching to %s]" % filters[int(idfil_psfmatch)])

			######==> (b) Get Kernels:
			# All the kernels should be brought to 0.25"/pixel sampling:
			status_kernel_resize = 1
			kernel_data = {}
			for bb in range(0,nbands):
				if bb != idfil_psfmatch:
					if kernels[filters[bb]] != None:
						hdu = fits.open(kernels[filters[bb]])
						kernel_data0 = hdu[0].data
						kernel_data[filters[bb]] = kernel_data0/kernel_data0.sum()
						hdu.close()

						status_kernel_resize = 0				# assuming the kernel image has the same pixel scale as the imaging data to be convolved

					elif kernels[filters[bb]] == None:
						status_kernel = check_avail_kernel(filter_init=filters[bb], filter_final=filters[int(idfil_psfmatch)])

						if status_kernel == 1:
							dir_file = PIXEDFIT_HOME+'/data/kernels/'
							kernel_name0 = 'kernel_%s_to_%s.fits.gz' % (filters[bb],filters[int(idfil_psfmatch)])
							hdu = fits.open(dir_file+kernel_name0)
							kernel_data0 = hdu[0].data
							kernel_data[filters[bb]] = kernel_data0/np.sum(kernel_data0)
							hdu.close()

							status_kernel_resize = 1			# the kernels are in pix-size=0.25 arcsec, so need to be adjusted with the pix-size of image 

						elif status_kernel == 0:
							if psf_fwhm == None:
								print ('Error: psf_fwhm=None!. If kernel for PSF matching %s-->%s is not available, their PSF FWHMs should be provided' % (filters[bb],filters[int(idfil_psfmatch)]))
								sys.exit()
							elif psf_fwhm != None:
								if psf_fwhm[filters[bb]] == None:
									print ('Error: psf_fwhm[%s]=None!. If kernel for PSF matching %s-->%s is not available, their PSF FWHMs should be provided' % (filters[bb],filters[bb],filters[int(idfil_psfmatch)]))
									sys.exit()
								else:
									## create kernel assuming Gaussian PSFs
									kernel_data[filters[bb]] = create_kernel_gaussian(psf_fwhm_init=psf_fwhm[filters[bb]], psf_fwhm_final=psf_fwhm[filters[int(idfil_psfmatch)]], 
																	alpha_cosbell=0.35, pixsize_PSF_target=img_pixsizes[filters[bb]], size=[101,101])



			###==> (c) PSF matching:
			# make arrays: set the default to None
			psfmatch_sci_img_data = {}
			psfmatch_sci_img_name = {}
			psfmatch_var_img_data = {}
			psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				psfmatch_sci_img_data[filters[bb]] = None
				psfmatch_sci_img_name[filters[bb]] = None
				psfmatch_var_img_data[filters[bb]] = None
				psfmatch_var_img_name[filters[bb]] = None

			for bb in range(0,nbands):
				# for filter that has largest PSF:
				if bb == idfil_psfmatch:
					psfmatch_sci_img_name[filters[bb]] = sci_img_name[filters[bb]]
					psfmatch_sci_img_data[filters[bb]] = sci_img_data[filters[bb]]
					psfmatch_var_img_name[filters[bb]] = var_img_name[filters[bb]]
					psfmatch_var_img_data[filters[bb]] = var_img_data[filters[bb]]
				# for other images:
				elif bb != idfil_psfmatch:
					###=> cropping around galaxy in question to minimize calculation time:
					print ("[Cropping images around target galaxy to minimize PSF matching time]")
					dim_y0 = stamp_size[0]
					dim_x0 = stamp_size[1]
					dim_y1 = int(dim_y0*1.5*img_pixsizes[filters[int(idfil_align)]]/img_pixsizes[filters[bb]])
					dim_x1 = int(dim_x0*1.5*img_pixsizes[filters[int(idfil_align)]]/img_pixsizes[filters[bb]])

					#++> for science image:
					hdu = fits.open(sci_img_name[filters[bb]])[0]
					wcs = WCS(hdu.header)
					gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
					position = (gal_x,gal_y)
					cutout = Cutout2D(hdu.data, position=position, size=(dim_y1,dim_x1), wcs=wcs)
					hdu.data = cutout.data 
					hdu.header.update(cutout.wcs.to_header())
					# write to fits file:
					name_out = "crop_%s" % sci_img_name[filters[bb]]
					hdu.writeto(name_out, overwrite=True)
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

					#++> for variance image:
					hdu = fits.open(var_img_name[filters[bb]])[0]
					wcs = WCS(hdu.header)
					gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
					position = (gal_x,gal_y)
					cutout = Cutout2D(hdu.data, position=position, size=(dim_y1,dim_x1), wcs=wcs)
					hdu.data = cutout.data 
					hdu.header.update(cutout.wcs.to_header())
					# write to fits file:
					name_out = "crop_%s" % var_img_name[filters[bb]]
					hdu.writeto(name_out, overwrite=True)
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

					# resize/resampling kernel image to match the sampling of the image:
					kernel_resize0 = resize_psf(kernel_data[filters[bb]], 0.250, img_pixsizes[filters[bb]], order=3)
					#print ("[resize %s to match the sampling of %s: %lf arcsec/pixel] => dimension: (%d x %d)" % (kernel_name[filters[bb]],
					#																					filters[bb],img_pixsizes[filters[bb]],
					#																					kernel_resize0.shape[0],
					#																					kernel_resize0.shape[1]))

					# crop the kernel size to reduce the memory usage:
					# roughly match with the size of the cropped image that will be convolved with the kernel:
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
						# get the coordinate of brightest pixel:
						bright_y,bright_x = unravel_index(kernel_resize0.argmax(), kernel_resize0.shape)
						print ("brightest pixel: (%d,%d)" % (bright_x,bright_y))

						# new dimension:
						if (dim_temp + 5)%2 == 0:
							dim_y = dim_temp + 5 - 1
							dim_x = dim_temp + 5 - 1
						else:
							dim_y = dim_temp + 5
							dim_x = dim_temp + 5

						y_cent = (dim_y-1)/2
						x_cent = (dim_x-1)/2
						print ("desired new dimension: (%d x %d)" % (dim_y,dim_x))
						print ("desired central position: (%d,%d)" % (x_cent,y_cent))

						kernel_resize1 = np.zeros((dim_y,dim_x))
						for yy in range(0,dim_y):
							for xx in range(0,dim_x):
								old_x = (xx - x_cent) + bright_x
								old_y = (yy - y_cent) + bright_y
								if old_x>kernel_resize0.shape[1]-1:
									old_x = kernel_resize0.shape[1] - 1
								if old_y>kernel_resize0.shape[0]-1:
									old_y = kernel_resize0.shape[0] - 1
								kernel_resize1[yy][xx] = kernel_resize0[int(old_y)][int(old_x)]
					else:
						kernel_resize1 = kernel_resize0

					# Normalize the kernel to have integrated value of 1.0
					kernel_resize = kernel_resize1/np.sum(kernel_resize1)

					print ("[PSF matching]")
					#++> for science image:
					name_fits = "crop_%s" % sci_img_name[filters[bb]]
					hdu = fits.open(name_fits)
					psfmatch_sci_img_data[filters[bb]] = convolve_fft(hdu[0].data, kernel_resize, allow_huge=True)
					name_out = "psfmatch_%s" % name_fits
					psfmatch_sci_img_name[filters[bb]] = name_out
					fits.writeto(name_out,psfmatch_sci_img_data[filters[bb]],hdu[0].header, overwrite=True)
					hdu.close()
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

					#++> for variance image:
					name_fits = "crop_%s" % var_img_name[filters[int(bb)]]
					hdu = fits.open(name_fits)
					psfmatch_var_img_data[filters[bb]] = convolve_fft(hdu[0].data, kernel_resize, allow_huge=True)
					name_out = "psfmatch_%s" % name_fits
					psfmatch_var_img_name[filters[bb]] = name_out
					fits.writeto(name_out,psfmatch_var_img_data[filters[bb]],hdu[0].header, overwrite=True)
					hdu.close()
					print ("[produce %s]" % name_out)
					temp_file_names.append(name_out)

			####============== End of (c) PSF matching ============#####

		if flag_reproject==1 and flag_crop==0:
			# Just crop the images
			align_psfmatch_sci_img_data = {}
			align_psfmatch_sci_img_name = {}
			align_psfmatch_var_img_data = {}
			align_psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				#++> for science image:
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
				align_psfmatch_sci_img_data[filters[bb]] = cutout.data
				print ("[produce %s]" % name_out)

				#++> for variance image:
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
				align_psfmatch_var_img_data[filters[bb]] = cutout.data
				print ("[produce %s]" % name_out)


		if flag_reproject==1 and flag_crop==1:
			align_psfmatch_sci_img_data = {}
			align_psfmatch_sci_img_name = {}
			align_psfmatch_var_img_data = {}
			align_psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				#++> for science image:
				hdu = fits.open(psfmatch_sci_img_name[filters[bb]])
				name_out = 'stamp_%s' % psfmatch_sci_img_name[filters[bb]]
				fits.writeto(name_out, hdu[0].data, header=hdu[0].header, overwrite=True)
				align_psfmatch_sci_img_name[filters[bb]] = name_out
				align_psfmatch_sci_img_data[filters[bb]] = hdu[0].data
				print ("[produce %s]" % name_out)
				hdu.close()
				#temp_file_names.append(name_out)

				#++> for variance image:
				hdu = fits.open(psfmatch_var_img_name[filters[bb]])
				name_out = 'stamp_%s' % psfmatch_var_img_name[filters[bb]]
				fits.writeto(name_out, hdu[0].data, header=hdu[0].header, overwrite=True)
				align_psfmatch_var_img_name[filters[bb]] = name_out
				align_psfmatch_var_img_data[filters[bb]] = hdu[0].data
				print ("[produce %s]" % name_out)
				hdu.close()
				#temp_file_names.append(name_out)


		if flag_reproject==0:
			####============== (3) Spatial reprojection and resampling ============#####
			print ("[images reprojection and resampling]")
			print ("align images to the reprojection and sampling of %s: %lf arcsec/pixel" % (filters[int(idfil_align)],img_pixsizes[filters[int(idfil_align)]]))
			align_psfmatch_sci_img_data = {}
			align_psfmatch_sci_img_name = {}
			align_psfmatch_var_img_data = {}
			align_psfmatch_var_img_name = {}
			for bb in range(0,nbands):
				align_psfmatch_sci_img_data[filters[bb]] = None
				align_psfmatch_sci_img_name[filters[bb]] = None
				align_psfmatch_var_img_data[filters[bb]] = None
				align_psfmatch_var_img_name[filters[bb]] = None

			# for image with largest pixel size: just crop the image
			#++> for science image:
			hdu = fits.open(psfmatch_sci_img_name[filters[int(idfil_align)]])[0]
			wcs = WCS(hdu.header)
			gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
			position = (gal_x,gal_y)
			cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
			hdu.data = cutout.data
			hdu.header.update(cutout.wcs.to_header())
			name_out = 'stamp_%s' % psfmatch_sci_img_name[filters[int(idfil_align)]]
			hdu.writeto(name_out, overwrite=True)
			align_psfmatch_sci_img_name[filters[int(idfil_align)]] = name_out
			align_psfmatch_sci_img_data[filters[int(idfil_align)]] = cutout.data
			print ("[produce %s]" % name_out)

			#++> for variance image:
			hdu = fits.open(psfmatch_var_img_name[filters[int(idfil_align)]])[0]
			wcs = WCS(hdu.header)
			gal_x, gal_y = wcs.wcs_world2pix(gal_ra, gal_dec, 1)
			position = (gal_x,gal_y)
			cutout = Cutout2D(hdu.data, position=position, size=stamp_size, wcs=wcs)
			hdu.data = cutout.data
			hdu.header.update(cutout.wcs.to_header())
			name_out = 'stamp_%s' % psfmatch_var_img_name[filters[int(idfil_align)]]
			hdu.writeto(name_out, overwrite=True)
			align_psfmatch_var_img_name[filters[int(idfil_align)]] = name_out
			align_psfmatch_var_img_data[filters[int(idfil_align)]] = cutout.data
			print ("[produce %s]" % name_out)


			# for other filters:
			# get header of stamp image that has largest pixel scale:
			hdu = fits.open(align_psfmatch_sci_img_name[filters[int(idfil_align)]])
			header_for_align = hdu[0].header
			hdu.close()

			for bb in range(0,nbands):
				if bb != idfil_align:
					#++> for science image:
					hdu = fits.open(psfmatch_sci_img_name[filters[bb]])
					if img_unit[filters[bb]] == 0:  ## flux
						data_image = hdu[0].data/img_pixsizes[filters[bb]]/img_pixsizes[filters[bb]]
						align_data_image0, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
						align_data_image = align_data_image0*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]
					elif img_unit[filters[bb]] == 1:  ## surface brightness
						data_image = hdu[0].data
						align_data_image, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
					name_out = "stamp_%s" % psfmatch_sci_img_name[filters[bb]]
					fits.writeto(name_out,align_data_image,header_for_align,overwrite=True)
					hdu.close()
					align_psfmatch_sci_img_name[filters[bb]] = name_out
					align_psfmatch_sci_img_data[filters[bb]] = align_data_image
					print ("[produce %s]" % name_out)
					#temp_file_names.append(name_out)

					#++> for variance image:
					hdu = fits.open(psfmatch_var_img_name[filters[bb]])
					if img_unit[filters[bb]] == 0:  ## flux
						data_image = hdu[0].data/img_pixsizes[filters[bb]]/img_pixsizes[filters[bb]]
						align_data_image0, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
						align_data_image = align_data_image0*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]
					elif img_unit[filters[bb]] == 1:  ## surface brightness
						data_image = hdu[0].data
						align_data_image, footprint = reproject_exact((data_image,hdu[0].header), header_for_align)
					### add systematic error:
					name_out = "stamp_%s" % psfmatch_var_img_name[filters[bb]]
					fits.writeto(name_out,align_data_image,header_for_align,overwrite=True)
					hdu.close()
					align_psfmatch_var_img_name[filters[bb]] = name_out
					align_psfmatch_var_img_data[filters[bb]] = align_data_image
					print ("[produce %s]" % name_out)
					#temp_file_names.append(name_out)

			####============== (End of 3) Reprojection and resampling ============#####

		### store the output:
		output_stamps = {}
		for bb in range(0,nbands):
			#++> for science image:
			str_temp = "name_img_%s" % filters[bb]
			output_stamps[str_temp] = align_psfmatch_sci_img_name[filters[bb]]

			#++> for variance image:
			str_temp = "name_var_%s" % filters[bb]
			output_stamps[str_temp] = align_psfmatch_var_img_name[filters[bb]]

			#++> additional info
			output_stamps['idfil_align'] = idfil_align
			output_stamps['idfil_psfmatch'] = idfil_psfmatch

		## remove files:
		if self.remove_files==True:
			for zz in range(0,len(temp_file_names)):
				os.system("rm %s" % temp_file_names[zz])

		return output_stamps

	#def segmentation_sextractor(self,output_stamps=None,detect_thresh=1.5,detect_minarea=10,deblend_nthresh=32,
	#							deblend_mincont=0.005,sewpypath=None):
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

		#if sewpypath == None:
		#	print ("sewpypath should be specified!. It is a path to sewpy installation directory.")
		#	sys.exit()

		#sys.path.insert(0, os.path.abspath(sewpypath))
		import logging
		logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

		import sewpy
		sexpath='sex'

		filters = self.filters
		#nbands = self.nbands
		nbands = len(filters)
		## get set of images that will be used for this analysis:
		name_img = []
		for bb in range(0,nbands):
			str_temp = "name_img_%s" % filters[bb]
			name_img.append(output_stamps[str_temp])

		## get image dimension:
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
			## get the segmentation map in form of 2D array:
			hdu = fits.open(name_segm)
			segm_map[int(bb)] = hdu[0].data
			hdu.close()

		return segm_map, segm_map_name

	def galaxy_region(self, name_segmentation_maps):
		"""A function to get initial definition of the galaxy's region of interest by merging together the segmentation maps.

		:param name_segmentation_maps:
			List of names of the segmentation maps.

		:returns gal_region:
			Final merged-segmentation map.
		"""

		nmaps = len(name_segmentation_maps)
		# get image dimension:
		hdu = fits.open(name_segmentation_maps[0])
		segm_map0 = hdu[0].data
		hdu.close()
		dim_y = segm_map0.shape[0]
		dim_x = segm_map0.shape[1]

		segmentation_maps = np.zeros((nmaps,dim_y,dim_x))
		for bb in range(0,nmaps):
			hdu = fits.open(name_segmentation_maps[bb])
			segmentation_maps[bb] = hdu[0].data
			hdu.close()

		gal_region = np.zeros((dim_y,dim_x))
		for yy in range(0,dim_y):
			for xx in range(0,dim_x):
				status_pix = 0
				for bb in range(0,nmaps):
					if segmentation_maps[bb][int(yy)][int(xx)] == 1:
						status_pix = 1
				if status_pix == 1:
					gal_region[int(yy)][int(xx)] = 1

		return gal_region

	def flux_map(self, output_stamps=None, gal_region=None, Gal_EBV=0, mag_zp_2mass=[], flag_HST_all=0, 
		PHOTFLAM_HST=[], flag_miniJPAS_all=0, unit_spire='Jy_per_beam', scale_unit=1.0e-17, sys_err_factor=0.0, name_out_fits=None):
		"""Function for calculating maps of multiband fluxes

		:param output_stamps:
			Dictionary containing reduced multiband images produced by the :func:`reduced_stamps` function.

		:param gal_region:
			2D array containing the galaxy's region of interest. The vlues should be 0 for masked region and 1 for the galaxy's region of interest.
			It can be taken from the output of the :func:`galaxy_region` function. But, user can also defined its own.

		:param Gal_EBV: (optional, default: 0)
			The E(B-V) dust attenuation due to the foreground Galactic dust. This is optional parameter.

		:param mag_zp_2mass: (optional, default: [])
			Magnitude zero-points of 2MASS images. Sshoud be in 1D array with three elements: [magzp-j,magzp-h,magzp-k]. This is optional parameter.
			If not given (i.e. [] or empty), the values will be taken from the FITS header information.

		:param flag_HST_all: (default: 0)
			Flag stating whether the all multiband images are from HST (value: 1) or not (value: 0). This flag can help assisting the calculation process especially if all the iamging data are from HST.

		:param PHOTFLAM_HST: (optional, default: [])
			List of PHOTFLAM values (i.e., zero points for converting pixel value into flux in erg/s/cm^2/A) in HST image. This parameter is only relevent (i.e., considered) when the flag_HST_all=0.
			The number of elements in the list should equal to the number of HST imaging data. This parameter is not mandatory (optional). If this parameter is empty, the PHOTFLAM value will be taken directly from the header of the original FITS files.

		:param unit_spire: (default: 'Jy_per_beam')
			Unit of SPIRE images, in case SPIRE image is included in the analysis. Options are: ['Jy_per_beam', 'MJy_per_sr', 'Jy_per_pixel']

		:param scale_unit: (defult: 1.0e-17)
			Normalized unit for the fluxes in the output fits file. The unit is flux density in erg/s/cm^2/Ang. 

		:param sys_err_factor: (default: 0.0)
			An estimate for systematic error, in which the bulk is assumed to be a factor from the flux (in every band). 

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

		# get effective/central wavelength of the filters:
		#photo_wave = filtering.cwave_filters(filters)
		photo_wave = cwave_filters(filters)

		eff_wave = {}
		for bb in range(0,nbands):
			str_temp = "cw_%s" % filters[bb]
			eff_wave[filters[bb]] = photo_wave[bb]

		# calculate Alambda for dust extinction correction:
		Alambda = {}
		for bb in range(0,nbands):
			if Gal_EBV == 0:
				Alambda[filters[bb]] = 0.0
			else:
				Alambda[filters[bb]] = k_lmbd_Fitz1986_LMC(eff_wave[filters[bb]])*Gal_EBV

		# get index of filter that has image with largest pixel scale:
		max_val = -10.0
		for bb in range(0,nbands):
			if img_pixsizes[filters[bb]] > max_val:
				max_val = img_pixsizes[filters[bb]]
				idfil_align = bb
		###================ End of (1) get basic information ===============####


		###================ (2) Calculation of flux map ===============####
		print ("[Deriving maps of multiband fluxes and flux uncertainties]")

		# allocate memory:
		map_flux = np.zeros((nbands,dim_y,dim_x)) - 99.0
		map_flux_err = np.zeros((nbands,dim_y,dim_x)) - 99.0
		for bb in range(0,nbands):
			print ("[Calculating fluxes maps for %s]" % filters[bb])

			# get science image:
			str_temp = "name_img_%s" % filters[bb]
			hdu = fits.open(output_stamps[str_temp])
			sci_img_data = hdu[0].data
			hdu.close()

			# get variance image:
			str_temp = "name_var_%s" % filters[bb]
			hdu = fits.open(output_stamps[str_temp]) 
			var_img_data = hdu[0].data
			hdu.close()

			###==> get magnitude zero-point and fluxes for zero-magnitudes zero point conversion: of 2MASS:
			if filters[bb]=='2mass_j' or filters[bb]=='2mass_h' or filters[bb]=='2mass_k':
				## get magnitude zero-point:
				if len(mag_zp_2mass) == 0:
					name_init_image = sci_img[filters[bb]]
					hdu = fits.open(name_init_image)
					MAGZP_2mass = float(hdu[0].header["MAGZP"])
					hdu.close()
				else:
					if filters[bb]=='2mass_j':
						MAGZP_2mass = mag_zp_2mass[0]
					elif filters[int(bb)]=='2mass_h':
						MAGZP_2mass = mag_zp_2mass[1]
					elif filters[int(bb)]=='2mass_k':
						MAGZP_2mass = mag_zp_2mass[2]

				## get flux at magnitude zero-point:
				if filters[bb]=='2mass_j':
					FLUXZP_2mass = 3.129e-13  #in W/cm^2/micron
				elif filters[int(bb)]=='2mass_h':
					FLUXZP_2mass = 1.133E-13
				elif filters[int(bb)]=='2mass_k':
					FLUXZP_2mass = 4.283E-14

			###==> get DN to Jy correction factors for WISE bands:
			if filters[bb]=='wise_w1' or filters[bb]=='wise_w2' or filters[bb]=='wise_w3' or filters[bb]=='wise_w4':
				if filters[bb]=='wise_w1':
					DN_to_Jy = 1.9350e-06
				elif filters[bb]=='wise_w2':
					DN_to_Jy = 2.7048E-06
				elif filters[bb]=='wise_w3':
					DN_to_Jy = 2.9045e-06
				elif filters[bb]=='wise_w4':
					DN_to_Jy = 5.2269E-05

			###==> get beam area of Herschel SPIRE:
			if filters[bb]=='herschel_spire_250' or filters[bb]=='herschel_spire_350' or filters[bb]=='herschel_spire_500':
				# get beam area in arcsec^2:
				if filters[bb]=='herschel_spire_250':
					beam_area = 469.3542
				elif filters[bb]=='herschel_spire_350':
					beam_area = 831.275
				elif filters[bb]=='herschel_spire_500':
					beam_area = 1804.3058

			###==> Get PHOTFLAM of HST image:
			PHOTFLAM = -99.0
			if flag_HST_all == 1:
				if len(PHOTFLAM_HST) == 0:
					name_init_image = sci_img[filters[bb]]
					hdu = fits.open(name_init_image)
					PHOTFLAM = float(hdu[0].header['PHOTFLAM'])
					hdu.close()
				else:
					PHOTFLAM = PHOTFLAM_HST[bb]

			if flag_HST_all == 0:
				name_init_image = sci_img[filters[bb]]
				hdu = fits.open(name_init_image)
				if 'PHOTFLAM' in hdu[0].header:
					PHOTFLAM = float(hdu[0].header['PHOTFLAM'])
				hdu.close()


			######====> calculation flux pixel-by-pixel:
			for yy in range(0,dim_y):
				for xx in range(0,dim_x):
					if gal_region[yy][xx] == 1:
						###==> For GALEX/FUV:
						if filters[bb] == 'galex_fuv':
							# flux:
							flux0 = sci_img_data[yy][xx]*1.40e-15  ## in erg/s/cm^2/Ang.
							map_flux[bb][yy][xx] = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])
							# flux error:
							flux0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*1.40e-15  ## in erg/s/cm^2/Ang.
							map_flux_err[bb][yy][xx] = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])

						###==> For GALEX/NUV:
						elif filters[bb] == 'galex_nuv':
							# flux:
							flux0 = sci_img_data[yy][xx]*2.06e-16  ## in erg/s/cm^2/Ang.
							map_flux[bb][yy][xx] = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])
							# flux error:
							flux0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*2.06e-16  ## in erg/s/cm^2/Ang.
							map_flux_err[bb][yy][xx] = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])

						###==> SDSS:
						elif filters[bb]=='sdss_u' or filters[bb]=='sdss_g' or filters[bb]=='sdss_r' or filters[bb]=='sdss_i' or filters[bb]=='sdss_z': 
							# flux:
							f0 = sci_img_data[yy][xx]*0.000003631       #### in Jy
							flux0 = f0*0.00002998/eff_wave[filters[bb]]/eff_wave[filters[bb]]  ## in erg/s/cm^2/Ang.
							map_flux[bb][yy][xx] = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])
							
							# flux error:
							f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*0.000003631       #### in Jy
							flux0 = f0*0.00002998/eff_wave[filters[bb]]/eff_wave[filters[bb]]  ## in erg/s/cm^2/Ang.
							map_flux_err[bb][yy][xx] = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])

						###==> 2MASS:
						### the image is in DN
						elif filters[bb]=='2mass_j' or filters[bb]=='2mass_h' or filters[bb]=='2mass_k':
							# flux:
							if sci_img_data[yy][xx] > 0:
								mag = MAGZP_2mass - 2.5*np.log10(sci_img_data[yy][xx])
								flux0 = FLUXZP_2mass*math.pow(10.0,-0.4*mag) ## in W/cm^2/micron
								flux0 = flux0*1.0e+3   ## in erg/s/cm^2/Ang.
								flux0 = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])
							elif sci_img_data[yy][xx] <= 0:
								mag = MAGZP_2mass - 2.5*np.log10(-1.0*sci_img_data[yy][xx])
								flux0 = -1.0*FLUXZP_2mass*math.pow(10.0,-0.4*mag) ## in W/cm^2/micron
								flux0 = flux0*1.0e+3   ## in erg/s/cm^2/Ang.
								flux0 = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])
							map_flux[bb][yy][xx] = flux0 
							# flux error:
							mag = MAGZP_2mass - 2.5*np.log10(math.sqrt(np.absolute(var_img_data[yy][xx])))
							flux0 = FLUXZP_2mass*math.pow(10.0,-0.4*mag) ## in W/cm^2/micron
							flux0 = flux0*1.0e+3   ## in erg/s/cm^2/Ang.
							flux0 = flux0*math.pow(10.0,0.4*Alambda[filters[bb]])
							map_flux_err[bb][yy][xx] = flux0 

						###==> Spitzer: IRAC and MIPS
						### Spitzer image is in Mjy/sr
						elif filters[bb]=='spitzer_irac_36' or filters[bb]=='spitzer_irac_45' or filters[bb]=='spitzer_irac_58' or filters[bb]=='spitzer_irac_80' or filters[bb]=='spitzer_mips_24' or filters[bb]=='spitzer_mips_70' or filters[bb]=='spitzer_mips_160':
							# flux:
							f0 = sci_img_data[yy][xx]*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   ## in unit of Jy
							flux0 = f0*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
							map_flux[bb][yy][xx] = flux0
							# flux error:
							f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   ## in unit of Jy
							flux0 = f0*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
							map_flux_err[bb][yy][xx] = flux0

						###==> WISE: 
						### image is in DN. DN_to_Jy is conversion factor from DN to Jy
						elif filters[bb]=='wise_w1' or filters[bb]=='wise_w2' or filters[bb]=='wise_w3' or filters[bb]=='wise_w4':
							# flux:
							f0 = sci_img_data[yy][xx]*DN_to_Jy*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
							map_flux[bb][yy][xx] = f0
							# flux error:
							f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*DN_to_Jy*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
							map_flux_err[bb][yy][xx] = f0

						###==> Herschel PACS:
						### image is in Jy/pixel or Jy --> this is not surface brightness unit but flux density
						elif filters[bb]=='herschel_pacs_70' or filters[bb]=='herschel_pacs_100' or filters[bb]=='herschel_pacs_160':
							# flux:
							f0 = sci_img_data[yy][xx]*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
							map_flux[bb][yy][xx] = f0
							# flux error:
							f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
							map_flux_err[bb][yy][xx] = f0

						###==> Herschel SPIRE:
						elif filters[bb]=='herschel_spire_250' or filters[bb]=='herschel_spire_350' or filters[bb]=='herschel_spire_500':
							if unit_spire == 'Jy_per_beam':
								### image is in Jy/beam -> surface brightness unit
								### Jy/pixel = Jy/beam x beam/arcsec^2 x arcsec^2/pixel  -> flux density
								# flux:
								f0 = sci_img_data[yy][xx]*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]/beam_area  ### in Jy
								f1 = f0*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
								map_flux[bb][yy][xx] = f1
								# flux error:
								f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]/beam_area  ### in Jy
								f1 = f0*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
								map_flux_err[bb][yy][xx] = f1
							### in case the data is in Mjy/sr
							elif unit_spire == 'MJy_per_sr':
								# flux:
								f0 = sci_img_data[yy][xx]*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   ## in unit of Jy
								flux0 = f0*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
								map_flux[bb][yy][xx] = flux0
								# flux error:
								f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*2.350443e-5*img_pixsizes[filters[int(idfil_align)]]*img_pixsizes[filters[int(idfil_align)]]   ## in unit of Jy
								flux0 = f0*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   ## in erg/s/cm^2/Ang.
								map_flux_err[bb][yy][xx] = flux0
							### in case the data is in Jy/pixel or Jy --> this is not surface brightness unit but flux density
							elif unit_spire == 'Jy_per_pixel':
								# flux:
								f0 = sci_img_data[yy][xx]*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   							## in erg/s/cm^2/Ang.
								map_flux[bb][yy][xx] = f0
								# flux error:
								f0 = math.sqrt(np.absolute(var_img_data[yy][xx]))*1.0e-23*2.998e+18/eff_wave[filters[bb]]/eff_wave[filters[bb]]   	## in erg/s/cm^2/Ang.
								map_flux_err[bb][yy][xx] = f0
							else:
								print ("unit of Herschel images is not recognized!")
								sys.exit()

						###==> Hubble Space Telescope (HST)
						elif PHOTFLAM != -99.0:
							# flux:
							map_flux[bb][yy][xx] = sci_img_data[yy][xx]*PHOTFLAM
							# flux error:
							map_flux_err[bb][yy][xx] = math.sqrt(np.absolute(var_img_data[yy][xx]))*PHOTFLAM

						###==> miniJPAS
						elif flag_miniJPAS_all == 1:
							# flux: 
							map_flux[bb][yy][xx] = sci_img_data[yy][xx]
							# flux error:
							map_flux_err[bb][yy][xx] = math.sqrt(np.absolute(var_img_data[yy][xx]))

						### End of if gal_region==1

			### end for bb: nbands


		# scaling the flux maps:
		print ("[scaling the flux map to a unit of %e erg/s/cm^2/Ang.]" % scale_unit)
		map_flux = map_flux/scale_unit
		map_flux_err = map_flux_err/scale_unit

		# add systematic error: assume overall the systematic error account for 10% from the flux 
		map_flux_err = map_flux_err + (sys_err_factor*map_flux)

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
		hdr['fil_sampling'] = fil_align
		hdr['pix_size'] = final_pix_size
		hdr['fil_psfmatch'] = fil_psfmatch
		hdr['psf_fwhm'] = final_psf_fwhm
		for bb in range(0,nbands):
			str_temp = 'fil%d' % int(bb)
			hdr[str_temp] = filters[int(bb)]
		primary_hdu = fits.PrimaryHDU(header=hdr)
		hdul.append(primary_hdu)
		# add galaxy_region into the HDU list:
		hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
		# add fluxes maps to the HDU list:
		hdul.append(fits.ImageHDU(map_flux, name='flux'))
		# add flux errors maps to the HDU list:
		hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
		# add one of the stamp image (the first band):
		hdul.append(fits.ImageHDU(data=stamp_img, header=stamp_hdr, name='stamp_image'))
		# write to fits file:
		if name_out_fits == None:
			name_out_fits = 'fluxmap.fits'
		hdul.writeto(name_out_fits, overwrite=True)

		#flux_map = {}
		#flux_map["galaxy_region"] = gal_region
		#flux_map["flux"] = map_flux
		#flux_map["flux_err"] = map_flux_err
		###================ (3) Store into fits file ===============####

		return name_out_fits




