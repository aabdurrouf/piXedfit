import numpy as np 
import math
import sys, os
import operator
from astropy.io import fits

__all__ = ["pixel_binning_specphoto"]


## Still under active development!


def pixel_binning_specphoto(specphoto_map_fits=None, fits_binmap=None, name_out_fits=None):
	"""Function for performing pixel binning on spectrophotometric data cube. 

	:param specphoto_map_fits:
		Input spectrophotometric data cube.

	:param fits_binmap:
		Input FITS file photometric data cube that has been performed with pixel binning. 
		This will be used for reference in pixel binning of the spectrophotometric data cube.

	:param name_out_fits: (defult: None)
		Desired name for the output FITS file.

	"""

	###================================###
	#### get the specphoto_map:
	cube = fits.open(specphoto_map_fits)
	header_specphoto = cube[0].header 
	spec_gal_region = cube['SPEC_REGION'].data
	wave = cube['wave'].data
	pix_spec_flux = cube['SPEC_FLUX'].data
	pix_spec_flux_err = cube['SPEC_FLUXERR'].data
	pix_photo_flux = cube['PHOTO_FLUX'].data
	pix_photo_flux_err = cube['PHOTO_FLUXERR'].data
	mod_wave = cube['MOD_WAVE'].data
	mod_spec_flux = cube['MOD_FLUX'].data
	cube.close()
	### get unit of flux in spectro-photo SEDs:
	unit_specphoto = float(header_specphoto['unit'])
	### get number of filters:
	nbands = int(header_specphoto['nfilters'])
	### number of wavelength points:
	nwaves = len(wave)
	###================================###

	### image dimension:
	dim_y = spec_gal_region.shape[0]
	dim_x = spec_gal_region.shape[1]

	###================================###
	### get the binning map:
	hdu = fits.open(fits_binmap)
	header_bin = hdu[0].header
	nbins = int(header_bin['nbins'])
	unit_temp = float(header_bin['unit'])
	map_bin_flag = hdu['bin_map'].data
	pix_bin_flux = hdu['bin_flux'].data*unit_temp/unit_specphoto
	pix_bin_flux_err = hdu['bin_fluxerr'].data*unit_temp/unit_specphoto
	hdu.close()
	###================================###

	## transpose (band,y,x) -> (y,x,band):
	photo0 = np.transpose(pix_photo_flux, axes=(1, 2, 0))
	photo_err0 = np.transpose(pix_photo_flux_err, axes=(1, 2, 0))

	## transpose (wavelength,y,x) -> (y,x,wavelength):
	spec0 = np.transpose(pix_spec_flux, axes=(1, 2, 0))
	spec_err0 = np.transpose(pix_spec_flux_err, axes=(1, 2, 0))
	mod_spec0 = np.transpose(mod_spec_flux, axes=(1, 2, 0))

	## transpose (band,y,x) -> (y,x,band):
	bin_photo0 = np.transpose(pix_bin_flux, axes=(1, 2, 0))
	bin_photo_err0 = np.transpose(pix_bin_flux_err, axes=(1, 2, 0))

	###================================###
	## pixel binning map for photo+spec: only bins that have more than 80% 
	## pixels covered in the MaNGA FoV are accounted for spec+photo binning SEDs:
	## binning flag of spec SED is set to be the same as the binning flag of photo. map:
	map_bin_flag_specphoto = np.zeros((dim_y,dim_x))
	map_bin_photo_fluxes0 = np.zeros((dim_y,dim_x,nbands))
	map_bin_photo_flux_err0 = np.zeros((dim_y,dim_x,nbands))
	map_bin_spec_fluxes0 = np.zeros((dim_y,dim_x,nwaves))
	map_bin_spec_flux_err0 = np.zeros((dim_y,dim_x,nwaves))
	map_bin_mod_spec_fluxes0 = np.zeros((dim_y,dim_x,len(mod_wave)))

	nbins_specphoto = 0
	for bb in range(0,nbins):
		#npixs_mem_ori = 0

		mem_pix_x_ori = []
		mem_pix_y_ori = []

		mem_pix_x = []
		mem_pix_y = []

		array_photo = []
		array_photo_err2 = []

		array_spec = []
		array_spec_err2 = []
		array_mod_spec = []

		for yy in range(0,dim_y):
			for xx in range(0,dim_x):
				if map_bin_flag[yy][xx] == bb+1:
					#npixs_mem_ori = npixs_mem_ori + 1
					mem_pix_x_ori.append(xx)
					mem_pix_y_ori.append(yy)

					if spec_gal_region[yy][xx] == 1:
						mem_pix_x.append(xx)
						mem_pix_y.append(yy)

						array_photo.append(photo0[yy][xx])
						array_photo_err2.append(photo_err0[yy][xx]*photo_err0[yy][xx])

						array_spec.append(spec0[yy][xx])
						array_spec_err2.append(spec_err0[yy][xx]*spec_err0[yy][xx])
						array_mod_spec.append(mod_spec0[yy][xx])

		#if len(mem_pix_x) >= 0.90*len(mem_pix_x_ori):   ### more than 80%  !!!!!!!!!!!!!!!!!!!!!!################
		if len(mem_pix_x) == len(mem_pix_x_ori): 
			temp_bin_photo_flux = np.sum(array_photo, axis=0)
			temp_bin_photo_flux_err = np.sqrt(np.sum(array_photo_err2, axis=0))

			temp_bin_spec_flux = np.sum(array_spec, axis=0)
			temp_bin_spec_flux_err = np.sqrt(np.sum(array_spec_err2, axis=0))
			temp_bin_mod_flux = np.sum(array_mod_spec, axis=0)

			nbins_specphoto = nbins_specphoto + 1
			for pp in range(0,len(mem_pix_x)):
				x0 = mem_pix_x[int(pp)]
				y0 = mem_pix_y[int(pp)]
				#map_bin_flag_specphoto[int(y0)][int(x0)] = nbins_specphoto
				map_bin_flag_specphoto[int(y0)][int(x0)] = bb+1   ####!!!!!!!!!!!

				#map_bin_photo_fluxes0[int(y0)][int(x0)] = temp_bin_photo_flux
				#map_bin_photo_flux_err0[int(y0)][int(x0)] = temp_bin_photo_flux_err

				map_bin_spec_fluxes0[int(y0)][int(x0)] = temp_bin_spec_flux
				map_bin_spec_flux_err0[int(y0)][int(x0)] = temp_bin_spec_flux_err
				map_bin_mod_spec_fluxes0[int(y0)][int(x0)] = temp_bin_mod_flux

			### modify the existing bin map flux that is not covered by spec.
			for pp in range(0,len(mem_pix_x_ori)):
				x0 = mem_pix_x_ori[int(pp)]
				y0 = mem_pix_y_ori[int(pp)]
				#map_bin_flag_specphoto[int(y0)][int(x0)] = nbins_specphoto
				map_bin_flag_specphoto[int(y0)][int(x0)] = bb+1   ####!!!!!!!!!!!

				map_bin_photo_fluxes0[int(y0)][int(x0)] = temp_bin_photo_flux
				map_bin_photo_flux_err0[int(y0)][int(x0)] = temp_bin_photo_flux_err

		elif len(mem_pix_x_ori) > 0:
			for pp in range(0,len(mem_pix_x_ori)):
				x0 = mem_pix_x_ori[int(pp)]
				y0 = mem_pix_y_ori[int(pp)]
				map_bin_photo_fluxes0[int(y0)][int(x0)] = bin_photo0[int(y0)][int(x0)]
				map_bin_photo_flux_err0[int(y0)][int(x0)] = bin_photo_err0[int(y0)][int(x0)]

	print ("Number of photometric bins: %d" % nbins)
	print ("Number of spectroscopic bins: %d" % nbins_specphoto)

	### transpose (y,x,wavelength) => (wavelength,y,x):
	map_bin_photo_fluxes = np.transpose(map_bin_photo_fluxes0, axes=(2, 0, 1))
	map_bin_photo_flux_err = np.transpose(map_bin_photo_flux_err0, axes=(2, 0, 1))

	map_bin_spec_fluxes = np.transpose(map_bin_spec_fluxes0, axes=(2, 0, 1))
	map_bin_spec_flux_err = np.transpose(map_bin_spec_flux_err0, axes=(2, 0, 1))
	map_bin_mod_spec_fluxes = np.transpose(map_bin_mod_spec_fluxes0, axes=(2, 0, 1))

	##### Store into fits file:
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['nfilters'] = header_bin['nfilters']
	hdr['refband'] = header_bin['refband']
	hdr['z'] = header_bin['z']
	hdr['nbins_photo'] = nbins
	hdr['nbins_spec'] = nbins_specphoto
	hdr['unit'] = unit_specphoto
	hdr['struct_photo'] = '(band,y,x)'
	hdr['struct_spec'] = '(wavelength,y,x)'
	for bb in range(0,int(header_bin['nfilters'])):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = header_bin[str_temp]
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)
	## add pixel bin flag:
	hdul.append(fits.ImageHDU(map_bin_flag, name='photo_bin_map'))
	## add pixel bin flag:
	hdul.append(fits.ImageHDU(map_bin_flag_specphoto, name='spec_bin_map'))
	## add map of bin flux:
	hdul.append(fits.ImageHDU(map_bin_photo_fluxes, name='bin_photo_flux'))
	## add map if bin flux error:
	hdul.append(fits.ImageHDU(map_bin_photo_flux_err, name='bin_photo_fluxerr'))
	## add wavelength grids:
	hdul.append(fits.ImageHDU(wave, name='spec_wave'))
	## add spectroscopic fluxes maps to the HDU list:
	hdul.append(fits.ImageHDU(map_bin_spec_fluxes, name='bin_spec_flux'))
	## add spectroscopic flux errors maps to the HDU list:
	hdul.append(fits.ImageHDU(map_bin_spec_flux_err, name='bin_spec_fluxerr'))
	## add spectroscopic model:
	hdul.append(fits.ImageHDU(mod_wave, name='mod_wave'))
	hdul.append(fits.ImageHDU(map_bin_mod_spec_fluxes, name='bin_mod_flux'))

	## write to fits file:
	if name_out_fits == None:
		name_out_fits = "pixbin_%s" % specphoto_map_fits
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits



def old_pixel_binning_specphoto(specphoto_map_fits=None, fits_binmap=None, SNR=None, name_out_fits=None):
	"""Function for performing pixel binning on spectrophotometric data cube. 

	:param specphoto_map_fits:
		Input spectrophotometric data cube.

	:param fits_binmap:
		Input FITS file photometric data cube that has been performed with pixel binning. 
		This will be used for reference in pixel binning of the spectrophotometric data cube.

	:param SNR: 
		S/N thresholds in multiple bands, the same as that in :func:`piXedfit.piXedfit_bin.pixel_binning_photo`.

	:param name_out_fits: (defult: None)
		Desired name for the output FITS file.

	"""

	###================================###
	#### get the specphoto_map:
	cube = fits.open(specphoto_map_fits)
	header_specphoto = cube[0].header 
	gal_region = cube['PHOTO_REGION'].data
	wave = cube['wave'].data
	pix_photo_flux = cube['PHOTO_FLUX'].data
	pix_photo_flux_err = cube['PHOTO_FLUXERR'].data
	pix_spec_flux = cube['SPEC_FLUX'].data
	pix_spec_flux_err = cube['SPEC_FLUXERR'].data
	#pix_spec_good_pix = cube['SPEC_GOOD_PIX'].data
	cube.close()
	### get unit of flux in spectro-photo SEDs:
	unit_specphoto = float(header_specphoto['unit'])
	### get number of filters:
	nbands = int(header_specphoto['nfilters'])
	### number of wavelength points:
	nwaves = len(wave)
	###================================###

	### image dimension:
	dim_y = gal_region.shape[0]
	dim_x = gal_region.shape[1]

	###================================###
	### get the binning map:
	hdu = fits.open(fits_binmap)
	header_bin = hdu[0].header
	map_bin_flag = hdu['bin_map'].data
	hdu.close()
	## number of bins:
	nbins = int(header_bin['nbins'])
	###================================###

	###================================###
	## pixel binning: only bins that have more than 80% pixels covered in the MaNGA FoV are accounted
	nbins_new = 0
	map_bin_flag_crop = np.zeros((dim_y,dim_x))
	for bb in range(0,nbins):
		npixs_mem_ori = 0
		npixs_mem = 0
		mem_pix_x = []
		mem_pix_y = []
		for yy in range(0,dim_y):
			for xx in range(0,dim_x):
				if gal_region[int(yy)][int(xx)] == 1 and map_bin_flag[int(yy)][int(xx)] == int(bb)+1:
					mem_pix_x.append(int(xx))
					mem_pix_y.append(int(yy))
					npixs_mem = npixs_mem + 1
				if map_bin_flag[int(yy)][int(xx)] == int(bb)+1:
					npixs_mem_ori = npixs_mem_ori + 1

		if npixs_mem >= 0.8*npixs_mem_ori:
			nbins_new = nbins_new + 1
			for pp in range(0,npixs_mem):
				x0 = mem_pix_x[int(pp)]
				y0 = mem_pix_y[int(pp)]
				map_bin_flag_crop[int(y0)][int(x0)] = nbins_new
	###================================###
	## get new number of bins:


	### binning the spectra based on the binning map information:
	bin_spec_fluxes = np.zeros((nbins,nwaves))
	bin_spec_flux_err = np.zeros((nbins,nwaves))
	bin_photo_fluxes = np.zeros((nbins,nbands))
	bin_photo_flux_err = np.zeros((nbins,nbands))
	nbins_new = 0
	map_bin_flag_crop = np.zeros((dim_y,dim_x))
	for bb in range(0,nbins):
		mem_pix_x = []
		mem_pix_y = []

		array_spec = []
		array_spec_err2 = []
		array_photo = []
		array_photo_err2 = []
		for yy in range(0,dim_y):
			for xx in range(0,dim_x):
				if gal_region[int(yy)][int(xx)] == 1 and map_bin_flag[int(yy)][int(xx)] == int(bb)+1:
					mem_pix_x.append(int(xx))
					mem_pix_y.append(int(yy))

					###================================###
					## transpose (wavelength,y,x) -> (y,x,wavelength):
					spec0 = np.transpose(pix_spec_flux, axes=(1, 2, 0))
					spec_err0 = np.transpose(pix_spec_flux_err, axes=(1, 2, 0))
					
					## exclude bad spectral pixel:
					#spec_wave_select = []
					#spec_flux_select = []
					#spec_flux_err_select = []
					#for ww in range(0,nwaves):
						#if pix_spec_good_pix[int(ww)][int(yy)][int(xx)] == 1:
					#	spec_wave_select.append(wave[int(ww)])
					#	spec_flux_select.append(spec0[int(yy)][int(xx)][int(ww)])
					#	spec_flux_err_select.append(spec_err0[int(yy)][int(xx)][int(ww)])
					## interpolate to fill up the wavelength points with bad pixels:
					#spec_flux_interp = np.interp(wave, spec_wave_select, spec_flux_select)
					#spec_flux_err_interp = np.interp(wave, spec_wave_select, spec_flux_err_select)
					#spec_flux_err_interp2 = np.asarray(spec_flux_err_interp)*np.asarray(spec_flux_err_interp)

					spec_flux_interp = spec0[yy][xx]
					spec_flux_err_interp = spec_err0[yy][xx]
					spec_flux_err_interp2 = np.asarray(spec_flux_err_interp)*np.asarray(spec_flux_err_interp)

					array_spec.append(spec_flux_interp)
					array_spec_err2.append(spec_flux_err_interp2)
					###================================###

					###================================###
					## transpose (band,y,x) -> (y,x,band):
					photo0 = np.transpose(pix_photo_flux, axes=(1, 2, 0))
					photo_err0 = np.transpose(pix_photo_flux_err, axes=(1, 2, 0))
					photo = photo0[int(yy)][int(xx)]
					photo_err = photo_err0[int(yy)][int(xx)]
					photo_err2 = np.asarray(photo_err)*np.asarray(photo_err)

					array_photo.append(photo)
					array_photo_err2.append(photo_err2)
					###================================###

		if len(mem_pix_x) > 0:
			bin_photo_fluxes_temp = np.sum(array_photo, axis=0)
			sum_array_photo_err2 = np.sum(array_photo_err2, axis=0)
			bin_photo_flux_err_temp = np.sqrt(sum_array_photo_err2)

			status_fil = 0
			for kk in range(0,nbands):
				if (bin_photo_fluxes_temp[int(kk)]/bin_photo_flux_err_temp[int(kk)]) > SNR[int(kk)]:
					status_fil = status_fil + 1
			
			if status_fil==nbands:
				bin_spec_fluxes[int(nbins_new)] = np.sum(array_spec, axis=0)
				sum_array_spec_err2 = np.sum(array_spec_err2, axis=0)
				bin_spec_flux_err[int(nbins_new)] = np.sqrt(sum_array_spec_err2)

				bin_photo_fluxes[int(nbins_new)] = np.sum(array_photo, axis=0)
				sum_array_photo_err2 = np.sum(array_photo_err2, axis=0)
				bin_photo_flux_err[int(nbins_new)] = np.sqrt(sum_array_photo_err2)
				nbins_new = nbins_new + 1

				for pp in range(0,len(mem_pix_x)):
					x0 = mem_pix_x[int(pp)]
					y0 = mem_pix_y[int(pp)]
					map_bin_flag_crop[int(y0)][int(x0)] = nbins_new

	########
	map_bin_photo_fluxes0 = np.zeros((dim_y,dim_x,nbands))
	map_bin_photo_flux_err0 = np.zeros((dim_y,dim_x,nbands))
	map_bin_spec_fluxes0 = np.zeros((dim_y,dim_x,nwaves))
	map_bin_spec_flux_err0 = np.zeros((dim_y,dim_x,nwaves))
	for yy in range(0,dim_y):
		for xx in range(0,dim_x): 
			if map_bin_flag_crop[int(yy)][int(xx)] > 0:
				bin_id = map_bin_flag_crop[int(yy)][int(xx)] - 1
				map_bin_spec_fluxes0[int(yy)][int(xx)] = bin_spec_fluxes[int(bin_id)]
				map_bin_spec_flux_err0[int(yy)][int(xx)] = bin_spec_flux_err[int(bin_id)]

				map_bin_photo_fluxes0[int(yy)][int(xx)] = bin_photo_fluxes[int(bin_id)]
				map_bin_photo_flux_err0[int(yy)][int(xx)] = bin_photo_flux_err[int(bin_id)]

	### transpose (y,x,band) => (band,y,x):
	map_bin_photo_fluxes = np.transpose(map_bin_photo_fluxes0, axes=(2, 0, 1))
	map_bin_photo_flux_err = np.transpose(map_bin_photo_flux_err0, axes=(2, 0, 1)) 

	### transpose (y,x,wavelength) => (wavelength,y,x):
	map_bin_spec_fluxes = np.transpose(map_bin_spec_fluxes0, axes=(2, 0, 1))
	map_bin_spec_flux_err = np.transpose(map_bin_spec_flux_err0, axes=(2, 0, 1))


	##### Store into fits file:
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['nfilters'] = header_bin['nfilters']
	hdr['refband'] = header_bin['refband']
	hdr['z'] = header_bin['z']
	hdr['nbins'] = nbins_new
	hdr['unit'] = unit_specphoto
	hdr['struct_photo'] = '(band,y,x)'
	hdr['struct_spec'] = '(wavelength,y,x)'
	for bb in range(0,int(header_bin['nfilters'])):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = header_bin[str_temp]
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)
	## add pixel bin flag:
	hdul.append(fits.ImageHDU(map_bin_flag_crop, name='bin_map'))
	## add map of bin flux:
	hdul.append(fits.ImageHDU(map_bin_photo_fluxes, name='bin_photo_flux'))
	## add map if bin flux error:
	hdul.append(fits.ImageHDU(map_bin_photo_flux_err, name='bin_photo_fluxerr'))
	## add wavelength grids:
	hdul.append(fits.ImageHDU(wave, name='spec_wave'))
	## add spectroscopic fluxes maps to the HDU list:
	hdul.append(fits.ImageHDU(map_bin_spec_fluxes, name='bin_spec_flux'))
	## add spectroscopic flux errors maps to the HDU list:
	hdul.append(fits.ImageHDU(map_bin_spec_flux_err, name='bin_spec_fluxerr'))
	## add spectroscopic masking maps to the HDU list:

	## write to fits file:
	if name_out_fits == None:
		name_out_fits = "pixbin_%s" % specphoto_map_fits
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits
