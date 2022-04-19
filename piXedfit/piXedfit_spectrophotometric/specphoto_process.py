import numpy as np 
import sys, os
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.convolution import convolve_fft, Gaussian1DKernel
from reproject import reproject_exact
from photutils.psf.matching import resize_psf
from ..piXedfit_images.images_utils import get_largest_FWHM_PSF, k_lmbd_Fitz1986_LMC


global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']


__all__ = ["specphoto_califagalexsdss2masswise", "specphoto_mangagalexsdss2masswise", "match_specphoto"]


def specphoto_califagalexsdss2masswise(photo_fluxmap=None, califa_file=None, pixsize_califa=1.0, spec_sigma=2.6, name_out_fits=None):
	"""Function for matching (spatially on pixel scales) between IFS data cube from CALIFA and the multiwavelength imaging 
	data (12 bands from GALEX, SDSS, 2MASS, and WISE). 

	:param photo_fluxmap:
		Input 3D data cube of photometry. This should have the same format as the output of :func:`piXedfit.piXedfit_images.images_processing.flux_map`.

	:param califa_file:
		Input CALIFA data cube.

	:param pixsize_califa: (default: 1.0)
		Pixel size (in arcsec) of CALIFA data cube.

	:param spec_sigma: (default: 2.6)
		Spectral resolution (in Angstrom) of CALIFA IFS data.

	:param name_out_fits:
		Name of output FITS file.
	"""

	## Next update: remove arguments pixsize_califa and spec_sigma, no smoothing with spec_sigma, and remove extension spec_good_pix

	# get maps of photometric fluxes 
	hdu = fits.open(photo_fluxmap)
	header_photo_fluxmap = hdu[0].header
	photo_gal_region = hdu['GALAXY_REGION'].data
	photo_flux_map = hdu['FLUX'].data          						# structure: (band,y,x)
	photo_fluxerr_map = hdu['FLUX_ERR'].data   						# structure: (band,y,x)
	unit_photo_fluxmap = float(header_photo_fluxmap['unit'])
	# header and data of stamp image
	data_stamp_image = hdu['stamp_image'].data 
	header_stamp_image = hdu['stamp_image'].header
	# image size
	dimy_stamp_image = data_stamp_image.shape[0]
	dimx_stamp_image = data_stamp_image.shape[1]
	hdu.close()

	# number of filters
	nbands = int(header_photo_fluxmap['nfilters'])

	# get set of filters
	filters = []
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb 
		filters.append(header_photo_fluxmap[str_temp])

	# pixel size in photometric data
	pixsize_image = float(header_photo_fluxmap['pixsize'])
	filter_ref_psfmatch = header_photo_fluxmap['fpsfmtch']
	
	# open CALIFA IFS data
	cube = fits.open(califa_file)
	header_califa3D = cube[0].header
	map_flux0 = cube['PRIMARY'].data                   				# structure: (wave,y,x)
	map_var = np.square(cube['ERROR'].data)   						# variance
	map_spec_mask = cube['BADPIX'].data               				# mask
	cube.close()
	# modify header to make it has only 2D WCS keywords
	w = WCS(naxis=2)
	w.wcs.crpix = [float(header_califa3D['CRPIX1']), float(header_califa3D['CRPIX2'])]
	w.wcs.cdelt = np.array([float(header_califa3D['CDELT1']), float(header_califa3D['CDELT2'])])
	w.wcs.crval = [float(header_califa3D['CRVAL1']), float(header_califa3D['CRVAL2'])]
	w.wcs.ctype = [header_califa3D['CTYPE1'], header_califa3D['CTYPE2']]
	header_califa2D = w.to_header()
	# get wavelength grids
	min_wave = float(header_califa3D['CRVAL3'])
	del_wave = float(header_califa3D['CDELT3'])
	nwaves = int(header_califa3D['NAXIS3'])
	max_wave = min_wave + (nwaves-1)*del_wave
	wave = np.linspace(min_wave,max_wave,nwaves)
	# get flux unit
	unit_ifu = 1.0e-16    											# in erg/s/cm^2/Ang. 
	# get dimension
	dim_y = map_flux0.shape[1]
	dim_x = map_flux0.shape[2]

	#  transpose (wave,y,x) ==> (y,x,wave)
	map_spec_mask_trans = np.transpose(map_spec_mask, axes=(1, 2, 0))

	# IFU region:
	gal_region = np.zeros((dim_y,dim_x))
	for ii in range(0,nwaves):
		rows, cols = np.where(map_spec_mask[ii]==0)
		gal_region[rows,cols] = gal_region[rows,cols] + 1
	gal_region_rows, gal_region_cols = np.where(gal_region>0.4*nwaves)
	gal_region[gal_region_rows,gal_region_cols] = 1

	#=> smooting the spectra
	wave_lin = np.linspace(int(min_wave), int(max_wave), int(max_wave)-int(min_wave)+1)
	# Gaussian kernel
	spec_kernel = Gaussian1DKernel(stddev=spec_sigma)
	# transpose (wave,y,x) => (y,x,wave)
	map_flux_trans = np.transpose(map_flux0, axes=(1, 2, 0))

	map_flux1 = np.zeros((dim_y,dim_x,nwaves))
	for ii in range(0,len(gal_region_rows)):
		yy = gal_region_rows[ii]
		xx = gal_region_cols[ii]
		# linear interpolation
		spec_flux_wavelin = np.interp(wave_lin, wave, map_flux_trans[yy][xx])
		# convolution
		conv_flux = convolve_fft(spec_flux_wavelin, spec_kernel)
		# exclude bad fluxes
		idx_excld = np.where((conv_flux<=0) | (np.isnan(conv_flux)==True) | (np.isinf(conv_flux)==True))
		wave_lin_temp = np.delete(wave_lin, idx_excld[0])
		conv_flux_temp = np.delete(conv_flux, idx_excld[0])
		# return to original wavelength sampling
		map_flux1[yy][xx] = np.interp(wave, wave_lin_temp, conv_flux_temp)

	# transpose (y,x,wave) => (wave,y,x)
	map_flux = np.transpose(map_flux1, axes=(2, 0, 1))

	# get kernel for PSF matching
	# All the kernels were brought to 0.25"/pixel sampling
	dir_file = PIXEDFIT_HOME+'/data/kernels/'
	#kernel_name = 'kernel_califa_to_wise_w2.fits.gz'
	kernel_name = 'kernel_califa_to_%s.fits.gz' % filter_ref_psfmatch
	hdu = fits.open(dir_file+kernel_name)
	kernel_data0 = hdu[0].data
	hdu.close()
	# normalize
	kernel_data = kernel_data0/np.sum(kernel_data0)
	# resize/resampling kernel image to match the sampling of the image
	kernel_resize = resize_psf(kernel_data, 0.25, pixsize_califa, order=3)
	# normalize
	kernel_resize = kernel_resize/np.sum(kernel_resize)

	#========================================================#
	# each imaging layer in the IFS 3D data cube: PSF matching and alignment (spatial resampling and reprojection) to the stamp image
	map_ifu_flux_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	map_ifu_flux_err_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	map_ifu_mask_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	for ww in range(0,nwaves):
		# get imaging layer from IFS 3D data cube
		layer_ifu_flux = map_flux[ww]
		layer_ifu_var = map_var[ww]
		layer_ifu_mask = map_spec_mask[ww]

		# PSF matching
		psfmatch_layer_ifu_flux = convolve_fft(layer_ifu_flux, kernel_resize, allow_huge=True, mask=layer_ifu_mask)
		psfmatch_layer_ifu_var = convolve_fft(layer_ifu_var, kernel_resize, allow_huge=True, mask=layer_ifu_mask)

		# align to stamp image:
		data_image = psfmatch_layer_ifu_flux/pixsize_califa/pixsize_califa       		# surface brightness
		align_psfmatch_layer_ifu_flux0, footprint = reproject_exact((data_image,header_califa2D), header_stamp_image)
		align_psfmatch_layer_ifu_flux = align_psfmatch_layer_ifu_flux0*pixsize_image*pixsize_image

		data_image = psfmatch_layer_ifu_var/pixsize_califa/pixsize_califa
		align_psfmatch_layer_ifu_var0, footprint = reproject_exact((data_image,header_califa2D), header_stamp_image)
		align_psfmatch_layer_ifu_var = align_psfmatch_layer_ifu_var0*pixsize_image*pixsize_image

		align_layer_ifu_mask, footprint = reproject_exact((layer_ifu_mask,header_califa2D), header_stamp_image)
		
		map_ifu_flux_temp[ww] = align_psfmatch_layer_ifu_flux               			# in unit_ifu          
		map_ifu_flux_err_temp[ww] = np.sqrt(align_psfmatch_layer_ifu_var)   			# in unit_ifu
		map_ifu_mask_temp[ww] = align_layer_ifu_mask

		sys.stdout.write('\r')
		sys.stdout.write('Wave id: %d from %d  ==> progress: %d%%' % (ww,nwaves,(ww+1)*100/nwaves))
		sys.stdout.flush()
	sys.stdout.write('\n')
	#========================================================#

	#========================================================#
	# Construct imaging layer for galaxy's region with 0 indicating pixels belong to galaxy's region and 1e+3 otherwise
	dim_y = map_flux0.shape[1]
	dim_x = map_flux0.shape[2]

	map_mask0 = np.zeros((dim_y,dim_x))
	for ii in range(0,nwaves):
		rows, cols = np.where(map_spec_mask[ii]==0)
		map_mask0[rows,cols] = map_mask0[rows,cols] + 1
	map_mask = np.zeros((dim_y,dim_x))
	rows, cols = np.where(map_mask0<0.8*nwaves)
	map_mask[rows,cols] = 1.0e+3
	# align to the stamp image
	align_map_mask, footprint = reproject_exact((map_mask,header_califa2D), header_stamp_image)
	#========================================================#	

	#========================================================#
	# transpose (band,y,x) => (y,x,band)
	photo_flux_map_trans = np.transpose(photo_flux_map, axes=(1, 2, 0))
	photo_fluxerr_map_trans = np.transpose(photo_fluxerr_map, axes=(1, 2, 0))
	# transpose (wave,y,x) => (y,x,wave)
	map_ifu_flux_temp_trans = np.transpose(map_ifu_flux_temp, axes=(1, 2, 0))
	map_ifu_flux_err_temp_trans = np.transpose(map_ifu_flux_err_temp, axes=(1, 2, 0))
	map_ifu_mask_temp_trans = np.transpose(map_ifu_mask_temp, axes=(1, 2, 0))

	# construct spectro-photometric SEDs within the defined region
	spec_gal_region = np.zeros((dimy_stamp_image,dimx_stamp_image))
	map_specphoto_spec_flux0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves,))
	map_specphoto_spec_flux_err0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves))
	map_specphoto_spec_mask0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves))

	#if adopt_photo_region == False:
	rows, cols = np.where((align_map_mask==0) & (photo_gal_region==1))
	spec_gal_region[rows,cols] = 1
	# flux in CALIFA has been corrected for the foreground dust extinction
	corr_spec = map_ifu_flux_temp_trans[rows,cols]
	corr_spec_err = map_ifu_flux_err_temp_trans[rows,cols]
	# store in temporary arrays
	map_specphoto_spec_flux0[rows,cols] = corr_spec
	map_specphoto_spec_flux_err0[rows,cols] = corr_spec_err
	map_specphoto_spec_mask0[rows,cols] = map_ifu_mask_temp_trans[rows,cols]

	# transpose from (y,x,wave) => (wave,y,x):
	# and convert into a new flux unit which is the same as flux unit for spec+photo with MaNGA:
	unit_ifu_new = 1.0e-17          #### in erg/s/cm^2/Ang.

	map_specphoto_spec_flux = np.transpose(map_specphoto_spec_flux0, axes=(2, 0, 1))*unit_ifu/unit_ifu_new
	map_specphoto_spec_flux_err = np.transpose(map_specphoto_spec_flux_err0, axes=(2, 0, 1))*unit_ifu/unit_ifu_new
	map_specphoto_spec_mask = np.transpose(map_specphoto_spec_mask0, axes=(2, 0, 1))

	# photo SED is given to the full map as it was with photometry only
	map_specphoto_phot_flux = photo_flux_map*unit_photo_fluxmap/unit_ifu_new          			# in unit_ifu_new
	map_specphoto_phot_flux_err = photo_fluxerr_map*unit_photo_fluxmap/unit_ifu_new          	# in unit_ifu_new

	# add systematic error to the spectra: assuming 10% from the resulted flux:
	map_specphoto_spec_flux_err = map_specphoto_spec_flux_err + (0.1*map_specphoto_spec_flux)
	#========================================================#


	# Store to FITS file
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['nfilters'] = nbands
	hdr['z'] = header_photo_fluxmap['z']
	hdr['RA'] = header_photo_fluxmap['RA']
	hdr['DEC'] = header_photo_fluxmap['DEC']
	hdr['GalEBV'] = header_photo_fluxmap['GalEBV']
	hdr['unit'] = unit_ifu_new
	hdr['bunit'] = 'erg/s/cm^2/A'
	hdr['structph'] = '(band,y,x)'
	hdr['structsp'] = '(wavelength,y,x)'
	hdr['fsamp'] = header_photo_fluxmap['fsamp']
	hdr['pixsize'] = header_photo_fluxmap['pixsize']
	hdr['fpsfmtch'] = header_photo_fluxmap['fpsfmtch']
	hdr['psffwhm'] = header_photo_fluxmap['psffwhm']
	hdr['specphot'] = 1

	for bb in range(0,nbands):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = header_photo_fluxmap[str_temp]

	hdul.append(fits.ImageHDU(data=map_specphoto_phot_flux, header=hdr, name='photo_flux'))
	hdul.append(fits.ImageHDU(map_specphoto_phot_flux_err, name='photo_fluxerr'))
	hdul.append(fits.ImageHDU(wave, name='wave'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_flux, name='spec_flux'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_flux_err, name='spec_fluxerr'))
	hdul.append(fits.ImageHDU(spec_gal_region, name='spec_region'))
	hdul.append(fits.ImageHDU(photo_gal_region, name='photo_region'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_mask, name='spec_good_pix'))
	hdul.append(fits.ImageHDU(data=data_stamp_image, header=header_stamp_image, name='stamp_image'))

	if name_out_fits==None:
		name_out_fits = "specphoto_%s.fits" % califa_file
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits



def specphoto_mangagalexsdss2masswise(photo_fluxmap=None, manga_file=None, pixsize_manga=0.5, spec_sigma=3.5, name_out_fits=None):
	
	"""Function for matching (spatially on pixel scales) between IFS data cube from MaNGA and the multiwavelength imaging 
	data (12 bands from GALEX, SDSS, 2MASS, and WISE). 

	:param photo_fluxmap:
		Input 3D data cube of photometry. This should have the same format as the output of :func:`piXedfit.piXedfit_images.images_processing.flux_map`.

	:param manga_file:
		Input MaNGA data cube.

	:param pixsize_manga: (default: 0.5)
		Pixel size (in arcsec) of MaNGA data cube.

	:param spec_sigma: (default: 3.5)
		Spectral resolution (in Angstrom) of MaNGA IFS data.

	:param name_out_fits:
		Name of output FITS file.
	"""

	# get maps of photometric fluxes
	hdu = fits.open(photo_fluxmap)
	header_photo_fluxmap = hdu[0].header
	photo_gal_region = hdu['GALAXY_REGION'].data
	photo_flux_map = hdu['FLUX'].data          							# structure: (band,y,x)
	photo_fluxerr_map = hdu['FLUX_ERR'].data  		 					# structure: (band,y,x)
	unit_photo_fluxmap = float(header_photo_fluxmap['unit'])
	# header and data of stamp image
	data_stamp_image = hdu['stamp_image'].data 
	header_stamp_image = hdu['stamp_image'].header
	# dimension
	dimy_stamp_image = data_stamp_image.shape[0]
	dimx_stamp_image = data_stamp_image.shape[1]
	hdu.close()

	# number of filters
	nbands = int(header_photo_fluxmap['nfilters'])

	# get set of filters
	filters = []
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb 
		filters.append(header_photo_fluxmap[str_temp])

	# pixel size in photometric data
	pixsize_image = float(header_photo_fluxmap['pixsize'])
	filter_ref_psfmatch = header_photo_fluxmap['fpsfmtch']

	## open MaNGA IFS data
	cube = fits.open(manga_file)
	map_flux0 = cube['FLUX'].data                   					# structure: (wave,y,x)
	map_var = 1.0/cube['IVAR'].data                						# variance
	map_spec_mask = cube['MASK'].data              						# mask
	wave = cube['WAVE'].data
	nwaves = len(wave)
	# reconstructed r-band image
	rimg = cube['RIMG'].data
	header_manga2D = cube['RIMG'].header
	# E(B-V) of foreground Galactic dust attenuation
	Gal_EBV = float(cube[0].header['EBVGAL'])
	cube.close()
	unit_ifu = 1.0e-17    												# in erg/s/cm^2/Ang.
	# dimension 
	dim_y = rimg.shape[0]
	dim_x = rimg.shape[1]

	# make mask region for PSF matching process
	mask_region = np.zeros((dim_y,dim_x))
	rows, cols = np.where(rimg<=0.0)
	mask_region[rows,cols] = 1

	#=> spectral smooting
	wave_lin = np.linspace(int(min(wave)),int(max(wave)),int(max(wave))-int(min(wave))+1)
	# Gaussian kernel
	spec_kernel = Gaussian1DKernel(stddev=spec_sigma)
	# transpose (wave,y,x) => (y,x,wave)
	map_flux_trans = np.transpose(map_flux0, axes=(1, 2, 0))

	map_flux1 = np.zeros((dim_y,dim_x,nwaves))
	rows, cols = np.where(rimg>0.0)
	for ii in range(0,len(rows)):
		yy = rows[ii]
		xx = cols[ii]
		# linear interpolation
		spec_flux_wavelin = np.interp(wave_lin, wave, map_flux_trans[yy][xx])
		# convolve with gaussian kernel
		conv_flux = convolve_fft(spec_flux_wavelin, spec_kernel)
		# exclude bad fluxes
		idx_excld = np.where((conv_flux<=0) | (np.isnan(conv_flux)==True) | (np.isinf(conv_flux)==True))
		wave_lin_temp = np.delete(wave_lin, idx_excld[0])
		conv_flux_temp = np.delete(conv_flux, idx_excld[0])

		# return to original wavelength sampling
		map_flux1[yy][xx] = np.interp(wave, wave_lin_temp, conv_flux_temp)

	# transpose (y,x,wave) => (wave,y,x):
	map_flux = np.transpose(map_flux1, axes=(2, 0, 1))

	# get kernel for PSF matching
	# All kernels were brought to 0.25"/pixel sampling
	dir_file = PIXEDFIT_HOME+'/data/kernels/'
	#kernel_name = 'kernel_manga_to_wise_w2.fits.gz'              ##!!!!!!!!!!!!!!!!!!!!!!
	kernel_name = 'kernel_manga_to_%s.fits.gz' % filter_ref_psfmatch
	hdu = fits.open(dir_file+kernel_name)
	kernel_data0 = hdu[0].data
	hdu.close()
	# normalize
	kernel_data = kernel_data0/np.sum(kernel_data0)
	# resize/resampling kernel image to match the sampling of the image
	kernel_resize = resize_psf(kernel_data, 0.25, pixsize_manga, order=3)
	# normalize
	kernel_resize = kernel_resize/np.sum(kernel_resize)


	#========================================================#
	# each imaging layer in the IFS 3D data cube: PSF matching and alignment (spatial resampling and reprojection) to the stamp image
	map_ifu_flux_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	map_ifu_flux_err_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	map_ifu_mask_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	for ww in range(0,nwaves):
		# get imaging layer from IFS 3D data cube
		layer_ifu_flux = map_flux[ww]
		layer_ifu_var = map_var[ww]
		layer_ifu_mask = map_spec_mask[ww]

		# PSF matching:
		psfmatch_layer_ifu_flux = convolve_fft(layer_ifu_flux, kernel_resize, allow_huge=True, mask=mask_region)
		psfmatch_layer_ifu_var = convolve_fft(layer_ifu_var, kernel_resize, allow_huge=True, mask=mask_region)

		# align to stamp image:
		data_image = psfmatch_layer_ifu_flux/pixsize_manga/pixsize_manga
		align_psfmatch_layer_ifu_flux0, footprint = reproject_exact((data_image,header_manga2D), header_stamp_image)
		align_psfmatch_layer_ifu_flux = align_psfmatch_layer_ifu_flux0*pixsize_image*pixsize_image
		
		data_image = psfmatch_layer_ifu_var/pixsize_manga/pixsize_manga
		align_psfmatch_layer_ifu_var0, footprint = reproject_exact((data_image,header_manga2D), header_stamp_image)
		align_psfmatch_layer_ifu_var = align_psfmatch_layer_ifu_var0*pixsize_image*pixsize_image
		
		align_layer_ifu_mask, footprint = reproject_exact((layer_ifu_mask,header_manga2D), header_stamp_image)
		
		map_ifu_flux_temp[int(ww)] = align_psfmatch_layer_ifu_flux               ### in unit_ifu          
		map_ifu_flux_err_temp[int(ww)] = np.sqrt(align_psfmatch_layer_ifu_var)   ### in unit_ifu
		map_ifu_mask_temp[int(ww)] = align_layer_ifu_mask

		sys.stdout.write('\r')
		sys.stdout.write('Wave id: %d from %d  ==> progress: %d%%' % (int(ww),nwaves,(int(ww)+1)*100/nwaves))
		sys.stdout.flush()
	sys.stdout.write('\n')
	#========================================================#

	#========================================================#
	# Construct imaging layer for galaxy's region with 0 indicating pixels belong to galaxy's region and 1e+3 otherwise
	dim_y = map_flux0.shape[1]
	dim_x = map_flux0.shape[2]

	map_mask = np.zeros((dim_y,dim_x))
	rows, cols = np.where(rimg <= 0.0)
	map_mask[rows,cols] = 1.0e+3
	# align to the stamp image
	align_map_mask, footprint = reproject_exact((map_mask,header_manga2D), header_stamp_image)
	#========================================================#	

	#========================================================#
	# transpose from (band,y,x) => (y,x,band)
	photo_flux_map_trans = np.transpose(photo_flux_map, axes=(1, 2, 0))
	photo_fluxerr_map_trans = np.transpose(photo_fluxerr_map, axes=(1, 2, 0))
	# transpose from (wave,y,x) => (y,x,wave)
	map_ifu_flux_temp_trans = np.transpose(map_ifu_flux_temp, axes=(1, 2, 0))
	map_ifu_flux_err_temp_trans = np.transpose(map_ifu_flux_err_temp, axes=(1, 2, 0))
	map_ifu_mask_temp_trans = np.transpose(map_ifu_mask_temp, axes=(1, 2, 0))

	# construct spectrophotometric SEDs within the defined region:
	spec_gal_region = np.zeros((dimy_stamp_image,dimx_stamp_image))
	map_specphoto_spec_flux0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves,))
	map_specphoto_spec_flux_err0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves))
	map_specphoto_spec_mask0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves))
	
	rows, cols = np.where((align_map_mask==0) & (photo_gal_region==1))
	spec_gal_region[rows,cols] = 1

	# correct for foreground dust extinction
	corr_factor = np.power(10.0,0.4*k_lmbd_Fitz1986_LMC(wave)*Gal_EBV)
	corr_spec = map_ifu_flux_temp_trans[rows,cols]*corr_factor
	corr_spec_err = map_ifu_flux_err_temp_trans[rows,cols]*corr_factor

	# store in temporary arrays
	map_specphoto_spec_flux0[rows,cols] = corr_spec
	map_specphoto_spec_flux_err0[rows,cols] = corr_spec_err
	map_specphoto_spec_mask0[rows,cols] =map_ifu_mask_temp_trans[rows,cols]

	# transpose from (y,x,wave) => (wave,y,x)
	map_specphoto_spec_flux = np.transpose(map_specphoto_spec_flux0, axes=(2, 0, 1))
	map_specphoto_spec_flux_err = np.transpose(map_specphoto_spec_flux_err0, axes=(2, 0, 1))
	map_specphoto_spec_mask = np.transpose(map_specphoto_spec_mask0, axes=(2, 0, 1))

	# photo SED is given to the full map as it was with photometry only
	map_specphoto_phot_flux = photo_flux_map*unit_photo_fluxmap/unit_ifu                 ### in unit_ifu
	map_specphoto_phot_flux_err = photo_fluxerr_map*unit_photo_fluxmap/unit_ifu          ### in unit_ifu
	#========================================================#


	# Store into fits file 
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['nfilters'] = nbands
	hdr['z'] = header_photo_fluxmap['z']
	hdr['RA'] = header_photo_fluxmap['RA']
	hdr['DEC'] = header_photo_fluxmap['DEC']
	hdr['GalEBV'] = header_photo_fluxmap['GalEBV']
	hdr['unit'] = unit_ifu
	hdr['bunit'] = 'erg/s/cm^2/A'
	hdr['structph'] = '(band,y,x)'
	hdr['structsp'] = '(wavelength,y,x)'
	hdr['fsamp'] = header_photo_fluxmap['fsamp']
	hdr['pixsize'] = header_photo_fluxmap['pixsize']
	hdr['fpsfmtch'] = header_photo_fluxmap['fpsfmtch']
	hdr['psffwhm'] = header_photo_fluxmap['psffwhm']
	hdr['specphot'] = 1
	
	for bb in range(0,nbands):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = header_photo_fluxmap[str_temp]
	hdul.append(fits.ImageHDU(data=map_specphoto_phot_flux, header=hdr, name='photo_flux'))
	hdul.append(fits.ImageHDU(map_specphoto_phot_flux_err, name='photo_fluxerr'))
	hdul.append(fits.ImageHDU(wave, name='wave'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_flux, name='spec_flux'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_flux_err, name='spec_fluxerr'))
	hdul.append(fits.ImageHDU(spec_gal_region, name='spec_region'))
	hdul.append(fits.ImageHDU(photo_gal_region, name='photo_region'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_mask, name='spec_good_pix'))
	hdul.append(fits.ImageHDU(data=data_stamp_image, header=header_stamp_image, name='stamp_image'))

	## write to fits file:
	if name_out_fits==None:
		name_out_fits = "specphoto_%s.fits" % manga_file
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits



def match_specphoto(specphoto_file=None,spec_sigma=3.5,name_saved_randmod=None,nproc=10,
					del_wave_nebem=10.0,name_out_fits=None):

	"""Function for correcting wavelength-dependent mismatch between IFS data and the multiwavelength photometric data. 
	
	:param specphoto_file:
		Input spec+photo FITS file, which is an output from previous step: specphoto_califagalexsdss2masswise or specphoto_mangagalexsdss2masswise.

	:param spec_sigma: (default: 3.5).
		Spectral resolution of the IFS data in Angstrom.

	:param name_saved_randmod:
		FITS file that contains random model SED templates to be used for fitting photometric SED. 
		The best-fit model spectra will then be used as references for correcting wavelength-dependent mismatch 
		between the IFS spectra and the photometric SEDs.This FITS file should have the same format as 
		the output of the :func:`piXedfit.piXedfit_model.save_models` function.

	:param nproc:
		Number of cores for calculation.

	:param del_wave_nebem: (default: 15.0 Angstrom).
		The range (+/-) around emission lines in the model spectra that will be removed in producing continuum-only spectrum. 
		This will be used as reference in correcting the wavelength-dependent mismatch between the IFS spectra and photometric SEDs.    

	:param name_out_fits: (optional).
		Desired name for the output FITS file.

	"""

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_spectrophotometric/'

	if name_out_fits==None:
		name_out_fits= "corr_%s" % specphoto_file
	os.system("mpirun -n %d python %s./match_specphoto.py %s %s %lf %s %lf" % (nproc,CODE_dir,specphoto_file,name_saved_randmod,
																			spec_sigma,name_out_fits,del_wave_nebem))

	return name_out_fits





