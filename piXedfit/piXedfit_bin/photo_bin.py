import numpy as np 
from math import sqrt
import sys
from operator import itemgetter
from astropy.io import fits

__all__ = ["pixel_binning"]



def pixel_binning(fits_fluxmap=None, ref_band=None, Dmin_bin=2.0, SNR=[], redc_chi2_limit=4.0, name_out_fits=None):
	"""Function for performing pixel binning, a proses of combining neighboring pixels to increase signal-to-noise ratios of the 
	spatially resolved SEDs.  

	:param fits_fluxmap:
		Input maps of multiband fluxes and fluxes uncertainties in a FITS file format.
		This FITS file should have the same format as the output of the :py:func:`flux_map` function (:py:mod:`piXedfit_images` module).

	:param ref_band: 
		Index of a band/filter that will be used as a reference in finding the brightest pixels. 
		The central pixel of a bin is the brightest pixel in this reference band.
		If ref_band=None, the ref_band is taken to be the middle band in the list of filters considered for the pixel binning.

	:param Dmin_bin: (default: 2.0 pixels)
		Minimum diameter of a bin in unit of pixel.

	:param SNR: (default: [])
		An array/list of S/N ratio thresholds in multiple bands. The length of this array should be the same as the number of bands in the fits_fluxmap. 
		S/N threshold can vary across the filters. If input SNR is empty, S/N of 5 is applied to all the filters. 

	:param redc_chi2_limit: (default: 4.0)
		A maximum of reduced chi-square for two SEDs are considered to have a similar shape. 

	:param name_out_fits: (defult: None)
		Desired name for the output FITS file.
	"""

	hdu = fits.open(fits_fluxmap)
	header = hdu[0].header
	if hdu[1].name == 'GALAXY_REGION':
		flag_specphoto = 0
		gal_region = hdu['GALAXY_REGION'].data
		map_flux = hdu['FLUX'].data
		map_flux_err = hdu['FLUX_ERR'].data 
	elif hdu[1].name == 'PHOTO_REGION':
		flag_specphoto = 1
		gal_region = hdu['PHOTO_REGION'].data
		spec_gal_region = hdu['SPEC_REGION'].data 
		map_flux = hdu['PHOTO_FLUX'].data 
		map_flux_err = hdu['PHOTO_FLUXERR'].data
		spec_wave = hdu['WAVE'].data 
		map_spec_flux = hdu['SPEC_FLUX'].data 
		map_spec_flux_err = hdu['SPEC_FLUXERR'].data
		nwaves = len(spec_wave) 
		# transpose from (wave,y,x) -> (y,x,wave)
		map_spec_flux_trans = np.transpose(map_spec_flux, axes=(1,2,0))
		map_spec_flux_err_trans = np.transpose(map_spec_flux_err, axes=(1,2,0))
	hdu.close()

	# transpose from (wave,y,x) -> (y,x,wave)
	map_flux_trans = np.transpose(map_flux, axes=(1,2,0))
	map_flux_err_trans = np.transpose(map_flux_err, axes=(1,2,0))

	nbands = int(header['nfilters'])
	if ref_band == None:
		ref_band = int((nbands-1)/2)
	else:
		ref_band = int(ref_band)

	if len(SNR)==0:
		SN_threshold = np.zeros(nbands)+5.0
	elif len(SNR) != nbands:
		print ("Number of elements in SNR should be the same as the number of filters in the fits_fluxmap!")
		sys.exit()
	else:
		SN_threshold = np.asarray(SNR) 

	dim_y = gal_region.shape[0]
	dim_x = gal_region.shape[1]

	pixbin_map = np.zeros((dim_y,dim_x))
	map_bin_flux = np.zeros((dim_y,dim_x,nbands))
	map_bin_flux_err = np.zeros((dim_y,dim_x,nbands))

	rows, cols = np.where((gal_region==1) & (pixbin_map==0))
	tot_npixs = len(rows)

	count_bin = 0
	del_r = 2
	cumul_npixs_in_bin = 0
	while len(rows)>0:
		# center pixel of a bin
		idx = np.unravel_index(map_flux[ref_band][rows,cols].argmax(), map_flux[ref_band][rows,cols].shape)
		bin_y_cent, bin_x_cent = rows[idx[0]], cols[idx[0]]

		#=> first circle
		# first, do square crop around the circle
		bin_rad = 0.5*Dmin_bin
		del_dim = bin_rad + 3
		xmin = int(bin_x_cent-del_dim)
		xmax = int(bin_x_cent+del_dim)
		ymin = int(bin_y_cent-del_dim)
		ymax = int(bin_y_cent+del_dim)

		if xmin<0:
			xmin = 0
		if xmax>=dim_x:
			xmax = dim_x-1

		if ymin<0:
			ymin = 0
		if ymax>=dim_y:
			ymax = dim_y-1 

		x = np.linspace(xmin,xmax,xmax-xmin+1)
		y = np.linspace(ymin,ymax,ymax-ymin+1)
		xx, yy = np.meshgrid(x,y)

		crop_gal_region = gal_region[ymin:ymax+1,xmin:xmax+1]
		crop_pixbin_map = pixbin_map[ymin:ymax+1,xmin:xmax+1]

		data2D_rad = np.sqrt(np.square(xx-bin_x_cent) + np.square(yy-bin_y_cent))
		rows1, cols1 = np.where((data2D_rad<=bin_rad) & (crop_gal_region==1) & (crop_pixbin_map==0))

		rows1 = rows1 + ymin
		cols1 = cols1 + xmin

		# get total fluxes
		tot_bin_flux = np.sum(map_flux_trans[rows1,cols1], axis=0)
		tot_bin_flux_err2 = np.sum(np.square(map_flux_err_trans[rows1,cols1]), axis=0)


		tot_SNR = tot_bin_flux/np.sqrt(tot_bin_flux_err2)
		idx0 = np.where(tot_SNR-SN_threshold>=0)

		if len(idx0[0]) == nbands:
			# get bin
			count_bin = count_bin + 1
			pixbin_map[rows1,cols1] = count_bin
			map_bin_flux[rows1,cols1] = tot_bin_flux
			map_bin_flux_err[rows1,cols1] = np.sqrt(tot_bin_flux_err2)
			cumul_npixs_in_bin = cumul_npixs_in_bin + len(rows1)

		#=> increase radius of the circle
		else:
			stat_increase = 1
			cumul_rows = rows1.tolist()
			cumul_cols = cols1.tolist()
			while stat_increase==1:
				rmin = bin_rad
				rmax = rmin + del_r

				#del_dim = bin_rad + 3
				del_dim = rmax + 3
				xmin = int(bin_x_cent-del_dim)
				xmax = int(bin_x_cent+del_dim)
				ymin = int(bin_y_cent-del_dim)
				ymax = int(bin_y_cent+del_dim)

				if xmin<0:
					xmin = 0
				if xmax>=dim_x:
					xmax = dim_x-1

				if ymin<0:
					ymin = 0
				if ymax>=dim_y:
					ymax = dim_y-1 

				x = np.linspace(xmin,xmax,xmax-xmin+1)
				y = np.linspace(ymin,ymax,ymax-ymin+1)
				xx, yy = np.meshgrid(x,y)

				crop_gal_region = gal_region[ymin:ymax+1,xmin:xmax+1]
				crop_pixbin_map = pixbin_map[ymin:ymax+1,xmin:xmax+1]

				data2D_rad = np.sqrt(np.square(xx-bin_x_cent) + np.square(yy-bin_y_cent))
				rows1, cols1 = np.where((data2D_rad>rmin) & (data2D_rad<=rmax) & (crop_gal_region==1) & (crop_pixbin_map==0))

				rows1 = rows1 + ymin
				cols1 = cols1 + xmin

				# check similarity of SED shape
				#cent_pix_SED_flux = map_flux_trans[bin_y_cent][bin_x_cent]
				#cent_pix_SED_flux_err = map_flux_err_trans[bin_y_cent][bin_x_cent]
				#pix_chi2 = np.zeros(len(rows1))
				#for zz in range(0,len(rows1)):
				#	top0 = np.sum(map_flux_trans[rows1[zz]][cols1[zz]]*cent_pix_SED_flux/(np.square(map_flux_err_trans[rows1[zz]][cols1[zz]])+np.square(cent_pix_SED_flux_err)))
				#	bottom0 = np.sum(np.square(cent_pix_SED_flux)/(np.square(map_flux_err_trans[rows1[zz]][cols1[zz]])+np.square(cent_pix_SED_flux_err)))
				#	norm0 = top0/bottom0
				#	pix_chi2[zz] = np.sum(np.square(map_flux_trans[rows1[zz]][cols1[zz]]-(norm0*cent_pix_SED_flux))/(np.square(map_flux_err_trans[rows1[zz]][cols1[zz]])+np.square(cent_pix_SED_flux_err)))

				cent_pix_SED_flux = np.zeros((dim_y,dim_x,nbands))
				cent_pix_SED_flux_err = np.zeros((dim_y,dim_x,nbands))
				norm0 = np.zeros((nbands,dim_y,dim_x))

				cent_pix_SED_flux[rows1,cols1] = map_flux_trans[bin_y_cent][bin_x_cent]
				cent_pix_SED_flux_err[rows1,cols1] = map_flux_err_trans[bin_y_cent][bin_x_cent]

				top0 = np.sum(map_flux_trans[rows1,cols1]*cent_pix_SED_flux[rows1,cols1]/(np.square(map_flux_err_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)
				bottom0 = np.sum(np.square(cent_pix_SED_flux[rows1,cols1])/(np.square(map_flux_err_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)
				for bb in range(0,nbands):
					norm0[bb][rows1,cols1] = top0/bottom0
				norm0_trans = np.transpose(norm0, axes=(1,2,0))
				pix_chi2 = np.sum(np.square(map_flux_trans[rows1,cols1]-(norm0_trans[rows1,cols1]*cent_pix_SED_flux[rows1,cols1]))/(np.square(map_flux_err_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)

				idx_sel = np.where((pix_chi2/nbands)<=redc_chi2_limit)

				cumul_rows = cumul_rows + rows1[idx_sel[0]].tolist()
				cumul_cols = cumul_cols + cols1[idx_sel[0]].tolist()

				# get total fluxes
				tot_bin_flux = tot_bin_flux + np.sum(map_flux_trans[rows1,cols1], axis=0)
				tot_bin_flux_err2 = tot_bin_flux_err2 + np.sum(np.square(map_flux_err_trans[rows1,cols1]), axis=0)

				tot_SNR = tot_bin_flux/np.sqrt(tot_bin_flux_err2)
				idx0 = np.where(tot_SNR-SN_threshold>=0)

				if len(idx0[0]) == nbands:
					# get bin
					count_bin = count_bin + 1
					pixbin_map[cumul_rows,cumul_cols] = count_bin
					map_bin_flux[cumul_rows,cumul_cols] = tot_bin_flux
					map_bin_flux_err[cumul_rows,cumul_cols] = np.sqrt(tot_bin_flux_err2)
					cumul_npixs_in_bin = cumul_npixs_in_bin + len(cumul_rows)

					stat_increase = 0
				
				else:
					if (len(cumul_rows)+cumul_npixs_in_bin) == tot_npixs:
						# get bin
						count_bin = count_bin + 1
						pixbin_map[cumul_rows,cumul_cols] = count_bin
						map_bin_flux[cumul_rows,cumul_cols] = tot_bin_flux
						map_bin_flux_err[cumul_rows,cumul_cols] = np.sqrt(tot_bin_flux_err2)
						cumul_npixs_in_bin = cumul_npixs_in_bin + len(cumul_rows)

						stat_increase = 0
						break
					else:
						stat_increase = 1


				bin_rad = bin_rad + del_r


		rows, cols = np.where((gal_region==1) & (pixbin_map==0))
		# end of while..

	# transpose from (y,x,wave) => (wave,y,x)
	map_bin_flux_trans = np.transpose(map_bin_flux, axes=(2,0,1))
	map_bin_flux_err_trans = np.transpose(map_bin_flux_err, axes=(2,0,1))

	if flag_specphoto == 0:
		print ("Number of bins: %d" % count_bin)

	# In case of specphoto input file: bin the IFS spectra
	elif flag_specphoto == 1:
		pixbin_map_specphoto = np.zeros((dim_y,dim_x))
		map_bin_spec_flux = np.zeros((dim_y,dim_x,nwaves))
		map_bin_spec_flux_err = np.zeros((dim_y,dim_x,nwaves))

		count_bin_specphoto = 0
		for bb in range(0,count_bin):
			rows1, cols1 = np.where(pixbin_map==bb+1)
			if np.sum(spec_gal_region[rows1,cols1])==len(rows1):
				count_bin_specphoto = count_bin_specphoto + 1
				pixbin_map_specphoto[rows1,cols1] = bb+1

				map_bin_spec_flux[rows1,cols1] = np.sum(map_spec_flux_trans[rows1,cols1], axis=0)
				map_bin_spec_flux_err[rows1,cols1] = np.sqrt(np.sum(np.square(map_spec_flux_err_trans[rows1,cols1]), axis=0))

		# transpose from (y,x,wave) -> (wave,y,x)
		map_bin_spec_flux_trans = np.transpose(map_bin_spec_flux, axes=(2,0,1))
		map_bin_spec_flux_err_trans = np.transpose(map_bin_spec_flux_err, axes=(2,0,1))

		print ("Number of bins in the photometric data cube: %d" % count_bin)
		print ("Number of bins in the spectroscopic data cube: %d" % count_bin_specphoto)


	## store into FITS file
	if flag_specphoto == 0:
		hdul = fits.HDUList()
		hdr = fits.Header()
		hdr['nfilters'] = nbands
		hdr['refband'] = ref_band
		hdr['z'] = header['z']
		hdr['nbins'] = count_bin
		hdr['unit'] = header['unit']
		hdr['bunit'] = 'erg/s/cm^2/A'
		hdr['struct'] = '(band,y,x)'
		hdr['fil_sampling'] = header['fil_sampling']
		hdr['pix_size'] = header['pix_size']
		hdr['fil_psfmatch'] = header['fil_psfmatch']
		hdr['psf_fwhm'] = header['psf_fwhm']

		for bb in range(0,nbands):
			str_temp = 'fil%d' % bb
			hdr[str_temp] = header[str_temp]

		primary_hdu = fits.PrimaryHDU(header=hdr)
		hdul.append(primary_hdu)
		hdul.append(fits.ImageHDU(pixbin_map, name='bin_map'))
		hdul.append(fits.ImageHDU(map_bin_flux_trans, name='bin_flux'))
		hdul.append(fits.ImageHDU(map_bin_flux_err_trans, name='bin_fluxerr'))

	elif flag_specphoto == 1:
		hdul = fits.HDUList()
		hdr = fits.Header()
		hdr['nfilters'] = nbands
		hdr['refband'] = ref_band
		hdr['z'] = header['z']
		hdr['nbins_photo'] = count_bin
		hdr['nbins_spec'] = count_bin_specphoto
		hdr['unit'] = header['unit']
		hdr['struct_photo'] = '(band,y,x)'
		hdr['struct_spec'] = '(wavelength,y,x)'

		for bb in range(0,nbands):
			str_temp = 'fil%d' % bb
			hdr[str_temp] = header[str_temp]

		primary_hdu = fits.PrimaryHDU(header=hdr)
		hdul.append(primary_hdu)
		hdul.append(fits.ImageHDU(pixbin_map, name='photo_bin_map'))
		hdul.append(fits.ImageHDU(pixbin_map_specphoto, name='spec_bin_map'))
		hdul.append(fits.ImageHDU(map_bin_flux_trans, name='bin_photo_flux'))
		hdul.append(fits.ImageHDU(map_bin_flux_err_trans, name='bin_photo_fluxerr'))
		hdul.append(fits.ImageHDU(spec_wave, name='spec_wave'))
		hdul.append(fits.ImageHDU(map_bin_spec_flux_trans, name='bin_spec_flux'))
		hdul.append(fits.ImageHDU(map_bin_spec_flux_err_trans, name='bin_spec_fluxerr'))


	if name_out_fits == None:
		name_out_fits = "pixbin_%s" % fits_fluxmap

	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits



