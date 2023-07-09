import numpy as np 
from math import sqrt
import sys
from operator import itemgetter
from astropy.io import fits

__all__ = ["pixel_binning", "pixel_binning_images", "open_binmap_fits", "plot_binmap", 
			"get_bins_SED_binmap", "plot_bins_SNR_radial_profile", "plot_bins_SED"]


def redchi2_two_seds(sed1_f=[], sed1_ferr=[], sed2_f=[], sed2_ferr=[]):
	top = np.sum(sed2_f*sed1_f/(np.square(sed1_ferr)+np.square(sed2_ferr)))
	bottom = np.sum(np.square(sed1_f)/(np.square(sed1_ferr)+np.square(sed2_ferr)))
	norm = top/bottom

	red_chi2 = np.sum(np.square(sed2_f-(norm*sed1_f))/(np.square(sed1_ferr)+np.square(sed2_ferr)))/len(sed1_f)

	return red_chi2


def pixel_binning(fits_fluxmap, ref_band=None, Dmin_bin=4.0, SNR=None, redc_chi2_limit=4.0, del_r=2.0, name_out_fits=None):
	"""Function for pixel binning, a proses of combining neighboring pixels to optimize the signal-to-noise ratios of the spatially resolved SEDs. 
	Input of this function is a data cube obtained from the image processing or spectrophotometric processing.  

	:param fits_fluxmap:
		Input FITS file containing the photometric or spectrophotometric data cube. The photometric data cube is obtained from the image processing with the :func:`images_processing` function, 
		while the spectrophotmetric data cube is the output of function :func:`match_imgifs_spectral`.

	:param ref_band: 
		Index of the reference band (filter) for sorting pixels based on the brightness. The central pixel of a bin is the brightest pixel in this reference band.
		If ref_band=None, the ref_band is chosen to be around the middle of the wavelength covered by the observed SEDs.

	:param Dmin_bin:
		Minimum diameter of a bin in unit of pixel.

	:param SNR:
		S/N thresholds in all bands. The length of this array should be the same as the number of bands in the fits_fluxmap. 
		S/N threshold can vary across the filters. If SNR is None, the S/N is set as 5.0 to all the filters. 

	:param redc_chi2_limit:
		A maximum reduced chi-square value for a pair of two SEDs to be considered as having a similar shape. 

	:param del_r:
		Increment of circular radius (in unit of pixel) adopted in the pixel binning process.

	:param name_out_fits: 
		Desired name for the output FITS file. If None, a default name is adopted.
	"""

	hdu = fits.open(fits_fluxmap)
	header = hdu[0].header
	if header['specphot'] == 0:
		gal_region = hdu['GALAXY_REGION'].data
		map_flux = hdu['FLUX'].data
		map_flux_err = hdu['FLUX_ERR'].data 
	elif header['specphot'] == 1:
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

	# number of filters
	nbands = int(header['nfilters'])

	# transpose from (wave,y,x) -> (y,x,wave)
	map_flux_trans = np.transpose(map_flux, axes=(1,2,0))
	map_flux_err_trans = np.transpose(map_flux_err, axes=(1,2,0))

	# modify negative fluxes in a given band with the minimum flux in that band
	# this is only used in calculating chi-square for the evaluation of the SED shape similarity  
	map_flux_corr = map_flux
	for bb in range(0,nbands):
		rows, cols = np.where((map_flux[bb]>0) & (gal_region==1))
		if len(rows) > 0:
			lowest = np.min(map_flux[bb][rows,cols])

			rows, cols = np.where((map_flux[bb]<0) & (gal_region==1))
			map_flux_corr[bb][rows,cols] = lowest

	# find systematic error factor
	rows, cols = np.where(gal_region==1)
	idx = np.unravel_index(map_flux[ref_band][rows,cols].argmax(), map_flux[ref_band][rows,cols].shape)
	yc, xc = rows[idx[0]], cols[idx[0]]
	
	status_add = 1
	factor = 0.01
	while status_add == 1:
		sed1_f = map_flux_trans[yc][xc]
		sed1_ferr = np.sqrt(np.square(map_flux_err_trans[yc][xc]) + np.square(factor*sed1_f))
		pix_chi2 = []
		for yy in range(yc-2,yc+2):
			for xx in range(xc-2,xc+2):
				if yy!=yc and xx!=xc:
					sed2_f = map_flux_trans[yy][xx]
					sed2_ferr = np.sqrt(np.square(map_flux_err_trans[yy][xx]) + np.square(factor*sed2_f))
					red_chi2 = redchi2_two_seds(sed1_f=sed1_f, sed1_ferr=sed1_ferr, sed2_f=sed2_f, sed2_ferr=sed2_ferr)
					pix_chi2.append(red_chi2)
		pix_chi2 = np.asarray(pix_chi2)
		if np.median(pix_chi2)<=2.0:
			status_add = 0

		factor = factor + 0.01
	# apply the factor
	map_flux_err_corr = np.sqrt( np.square(map_flux_err) + np.square(factor*map_flux))

	# transpose from (band,y,x) -> (y,x,band)
	map_flux_corr_trans = np.transpose(map_flux_corr, axes=(1,2,0))
	map_flux_err_corr_trans = np.transpose(map_flux_err_corr, axes=(1,2,0))

	# get reference band for pixel brightness
	if ref_band is None:
		ref_band = int((nbands-1)/2)
	else:
		ref_band = int(ref_band)

	if SNR is None:
		SN_threshold = np.zeros(nbands) + 5.0
	elif len(SNR) != nbands:
		print ("Number of elements in SNR should be the same as the number of filters in the fits_fluxmap, which is %d!" % nbands)
		sys.exit()
	else:
		SN_threshold = np.asarray(SNR)
		idx0 = np.where(SNR==0)
		SN_threshold[idx0[0]] = -1.0e+10

	dim_y = gal_region.shape[0]
	dim_x = gal_region.shape[1]

	pixbin_map = np.zeros((dim_y,dim_x))
	map_bin_flux = np.zeros((dim_y,dim_x,nbands))
	map_bin_flux_err = np.zeros((dim_y,dim_x,nbands))

	rows, cols = np.where((gal_region==1) & (pixbin_map==0))
	tot_npixs = len(rows)

	count_bin = 0
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
				cent_pix_SED_flux = np.zeros((dim_y,dim_x,nbands))
				cent_pix_SED_flux_err = np.zeros((dim_y,dim_x,nbands))
				norm0 = np.zeros((nbands,dim_y,dim_x))

				cent_pix_SED_flux[rows1,cols1] = map_flux_corr_trans[bin_y_cent][bin_x_cent]
				cent_pix_SED_flux_err[rows1,cols1] = map_flux_err_corr_trans[bin_y_cent][bin_x_cent]

				top0 = np.sum(map_flux_corr_trans[rows1,cols1]*cent_pix_SED_flux[rows1,cols1]/(np.square(map_flux_err_corr_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)
				bottom0 = np.sum(np.square(cent_pix_SED_flux[rows1,cols1])/(np.square(map_flux_err_corr_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)
				for bb in range(0,nbands):
					norm0[bb][rows1,cols1] = top0/bottom0
				# transpose from (band,y,x) -> (y,x,band)
				norm0_trans = np.transpose(norm0, axes=(1,2,0))
				pix_chi2 = np.sum(np.square(map_flux_corr_trans[rows1,cols1]-(norm0_trans[rows1,cols1]*cent_pix_SED_flux[rows1,cols1]))/(np.square(map_flux_err_corr_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)

				idx_sel = np.where((pix_chi2/nbands)<=redc_chi2_limit)

				# cut, only select pixels with similar SED shape to the central brightest pixel
				rows1_cut = rows1[idx_sel[0]]
				cols1_cut = cols1[idx_sel[0]]

				cumul_rows = cumul_rows + rows1_cut.tolist()
				cumul_cols = cumul_cols + cols1_cut.tolist()

				# get total fluxes
				tot_bin_flux = tot_bin_flux + np.sum(map_flux_trans[rows1_cut,cols1_cut], axis=0)
				tot_bin_flux_err2 = tot_bin_flux_err2 + np.sum(np.square(map_flux_err_trans[rows1_cut,cols1_cut]), axis=0)

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
					# check remaining pixels
					rows_rest, cols_rest = np.where((gal_region==1) & (pixbin_map==0))
					tflux = np.sum(map_flux_trans[rows_rest,cols_rest], axis=0)
					tflux_err2 = np.sum(np.square(map_flux_err_trans[rows_rest,cols_rest]), axis=0)
					tSNR = tflux/np.sqrt(tflux_err2)
					tidx = np.where(tSNR-SN_threshold>=0)

					if len(tidx[0]) < nbands:
						# bin all remaining pixels:
						count_bin = count_bin + 1
						pixbin_map[rows_rest,cols_rest] = count_bin
						map_bin_flux[rows_rest,cols_rest] = tflux
						map_bin_flux_err[rows_rest,cols_rest] = np.sqrt(tflux_err2)
						cumul_npixs_in_bin = cumul_npixs_in_bin + len(rows_rest)

						stat_increase = 0
						break

					elif (len(cumul_rows)+cumul_npixs_in_bin) == tot_npixs:
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

		sys.stdout.write('\r')
		sys.stdout.write('Bins: %d ==> accumulated pixels: %d/%d' % (count_bin,cumul_npixs_in_bin,tot_npixs))
		sys.stdout.flush()
	sys.stdout.write('\n')


	# transpose from (y,x,wave) => (wave,y,x)
	map_bin_flux_trans = np.transpose(map_bin_flux, axes=(2,0,1))
	map_bin_flux_err_trans = np.transpose(map_bin_flux_err, axes=(2,0,1))

	if header['specphot'] == 0:
		print ("Number of bins: %d" % count_bin)

	# In case of specphoto input file: bin the IFS spectra
	elif header['specphot'] == 1:
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
	if header['specphot'] == 0:
		hdul = fits.HDUList()
		hdr = fits.Header()
		hdr['nfilters'] = nbands
		hdr['refband'] = ref_band
		if 'RA' in header:
			hdr['RA'] = header['RA']
		if 'DEC' in header:
			hdr['DEC'] = header['DEC']
		hdr['z'] = header['z']
		if 'GalEBV' in header:
			hdr['GalEBV'] = header['GalEBV']
		hdr['nbins'] = count_bin
		hdr['unit'] = header['unit']
		hdr['bunit'] = 'erg/s/cm^2/A'
		hdr['struct'] = '(band,y,x)'
		if 'fsamp' in header:
			hdr['fsamp'] = header['fsamp']
		if 'pixsize' in header:
			hdr['pixsize'] = header['pixsize']
		if 'fpsfmtch' in header:
			hdr['fpsfmtch'] = header['fpsfmtch']
		if 'psffwhm' in header:
			hdr['psffwhm'] = header['psffwhm']
		if 'specphot' in header:
			hdr['specphot'] = header['specphot']

		for bb in range(0,nbands):
			str_temp = 'fil%d' % bb
			hdr[str_temp] = header[str_temp]

		hdul.append(fits.ImageHDU(data=pixbin_map, header=hdr, name='bin_map'))
		hdul.append(fits.ImageHDU(map_bin_flux_trans, name='bin_flux'))
		hdul.append(fits.ImageHDU(map_bin_flux_err_trans, name='bin_fluxerr'))

	elif header['specphot'] == 1:
		hdul = fits.HDUList()
		hdr = fits.Header()
		hdr['nfilters'] = nbands
		hdr['refband'] = ref_band
		if 'RA' in header:
			hdr['RA'] = header['RA']
		if 'DEC' in header:
			hdr['DEC'] = header['DEC']
		hdr['z'] = header['z']
		if 'GalEBV' in header:
			hdr['GalEBV'] = header['GalEBV']
		hdr['nbinsph'] = count_bin
		hdr['nbinssp'] = count_bin_specphoto
		hdr['unit'] = header['unit']
		hdr['bunit'] = 'erg/s/cm^2/A'
		hdr['structph'] = '(band,y,x)'
		hdr['structsp'] = '(wavelength,y,x)'
		if 'fsamp' in header:
			hdr['fsamp'] = header['fsamp']
		if 'pixsize' in header:    
			hdr['pixsize'] = header['pixsize']       
		if 'fpsfmtch' in header:
			hdr['fpsfmtch'] = header['fpsfmtch']
		if 'psffwhm' in header:
			hdr['psffwhm'] = header['psffwhm']
		if 'specphot' in header:
			hdr['specphot'] = header['specphot']

		for bb in range(0,nbands):
			str_temp = 'fil%d' % bb
			hdr[str_temp] = header[str_temp]

		hdul.append(fits.ImageHDU(data=pixbin_map, header=hdr, name='photo_bin_map'))
		hdul.append(fits.ImageHDU(spec_gal_region, name='spec_region'))
		hdul.append(fits.ImageHDU(pixbin_map_specphoto, name='spec_bin_map'))
		hdul.append(fits.ImageHDU(map_bin_flux_trans, name='bin_photo_flux'))
		hdul.append(fits.ImageHDU(map_bin_flux_err_trans, name='bin_photo_fluxerr'))
		hdul.append(fits.ImageHDU(spec_wave, name='spec_wave'))
		hdul.append(fits.ImageHDU(map_bin_spec_flux_trans, name='bin_spec_flux'))
		hdul.append(fits.ImageHDU(map_bin_spec_flux_err_trans, name='bin_spec_fluxerr'))


	if name_out_fits is None:
		name_out_fits = "pixbin_%s" % fits_fluxmap

	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits



def pixel_binning_images(images, var_images, ref_band=None, Dmin_bin=2.0, SNR=None, redc_chi2_limit=4.0, del_r=2.0, name_out_fits=None):
	"""Function for pixel binning on multiband image.  

	:param images:
		Input science images. This input should be in a list format, such as images=['image1.fits', 'image2.fits', 'image3.fits']

	:param var_images:
		Variance images in a list format. The number of variance images should be the same as that of the input science images.

	:param ref_band: 
		Index of the reference band (filter) for sorting pixels based on the brightness. The central pixel of a bin is the brightest pixel in this reference band.
		If ref_band=None, the ref_band is chosen to be around the middle of the wavelength covered by the observed SEDs.

	:param Dmin_bin:
		Minimum diameter of a bin in unit of pixel.

	:param SNR:
		S/N thresholds in all bands. The length of this array should be the same as the number of bands in the fits_fluxmap. 
		S/N threshold can vary across the filters. If SNR is None, the S/N is set as 5.0 to all the filters. 

	:param redc_chi2_limit:
		A maximum reduced chi-square value for a pair of two SEDs to be considered as having a similar shape. 

	:param del_r:
		Increment of circular radius (in unit of pixel) adopted in the pixel binning process.

	:param name_out_fits: 
		Desired name for the output FITS file. If None, a default name is adopted.
	"""

	nbands = len(images)

	# get image size
	hdu = fits.open(images[0])
	dim_y = hdu[0].data.shape[0]
	dim_x = hdu[0].data.shape[1]
	hdu.close()

	gal_region = np.zeros((dim_y,dim_x))+1

	# open the images
	map_flux = np.zeros((nbands,dim_y,dim_x))
	map_flux_err = np.zeros((nbands,dim_y,dim_x))
	for bb in range(0,nbands):
		hdu = fits.open(images[bb])
		map_flux[bb] = hdu[0].data
		hdu.close()

		hdu = fits.open(var_images[bb])
		map_flux_err[bb] = np.sqrt(hdu[0].data)
		hdu.close()

	# transpose from (wave,y,x) -> (y,x,wave)
	map_flux_trans = np.transpose(map_flux, axes=(1,2,0))
	map_flux_err_trans = np.transpose(map_flux_err, axes=(1,2,0))

	# modify negative fluxes in a given band with the minimum flux in that band
	# this is only used in calculating chi-square for the evaluation of the SED shape similarity  
	map_flux_corr = map_flux
	for bb in range(0,nbands):
		rows, cols = np.where((map_flux[bb]>0) & (gal_region==1))
		lowest = np.min(map_flux[bb][rows,cols])

		rows, cols = np.where((map_flux[bb]<0) & (gal_region==1))
		map_flux_corr[bb][rows,cols] = lowest

	# find systematic error factor
	rows, cols = np.where(gal_region==1)
	idx = np.unravel_index(map_flux[ref_band][rows,cols].argmax(), map_flux[ref_band][rows,cols].shape)
	yc, xc = rows[idx[0]], cols[idx[0]]
	
	status_add = 1
	factor = 0.01
	while status_add == 1:
		sed1_f = map_flux_trans[yc][xc]
		sed1_ferr = np.sqrt(np.square(map_flux_err_trans[yc][xc]) + np.square(factor*sed1_f))
		pix_chi2 = []
		for yy in range(yc-2,yc+2):
			for xx in range(xc-2,xc+2):
				if yy!=yc and xx!=xc:
					sed2_f = map_flux_trans[yy][xx]
					sed2_ferr = np.sqrt(np.square(map_flux_err_trans[yy][xx]) + np.square(factor*sed2_f))
					red_chi2 = redchi2_two_seds(sed1_f=sed1_f, sed1_ferr=sed1_ferr, sed2_f=sed2_f, sed2_ferr=sed2_ferr)
					pix_chi2.append(red_chi2)
		pix_chi2 = np.asarray(pix_chi2)
		if np.median(pix_chi2)<=2.0:
			status_add = 0

		factor = factor + 0.01
	# apply the factor
	map_flux_err_corr = np.sqrt( np.square(map_flux_err) + np.square(factor*map_flux))

	# transpose from (band,y,x) -> (y,x,band)
	map_flux_corr_trans = np.transpose(map_flux_corr, axes=(1,2,0))
	map_flux_err_corr_trans = np.transpose(map_flux_err_corr, axes=(1,2,0))

	# reference band for measuring pixel brightness
	if ref_band is None:
		if nbands == 1:
			ref_band = 0
		else:
			ref_band = int((nbands-1)/2)
	else:
		ref_band = int(ref_band)

	if SNR is None:
		SN_threshold = np.zeros(nbands) + 5.0
	elif len(SNR) != nbands:
		print ("Number of elements in SNR should be the same as the number of filters in the fits_fluxmap, which is %d!" % nbands)
		sys.exit()
	else:
		SN_threshold = np.asarray(SNR)
		idx0 = np.where(SNR==0)
		SN_threshold[idx0[0]] = -1.0e+10

	pixbin_map = np.zeros((dim_y,dim_x))
	map_bin_flux = np.zeros((dim_y,dim_x,nbands))
	map_bin_flux_err = np.zeros((dim_y,dim_x,nbands))

	rows, cols = np.where((gal_region==1) & (pixbin_map==0))
	tot_npixs = len(rows)

	count_bin = 0
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

				if nbands==0:
					rows1_cut = rows1
					cols1_cut = cols1
				else:
					# check similarity of SED shape
					cent_pix_SED_flux = np.zeros((dim_y,dim_x,nbands))
					cent_pix_SED_flux_err = np.zeros((dim_y,dim_x,nbands))
					norm0 = np.zeros((nbands,dim_y,dim_x))

					cent_pix_SED_flux[rows1,cols1] = map_flux_corr_trans[bin_y_cent][bin_x_cent]
					cent_pix_SED_flux_err[rows1,cols1] = map_flux_err_corr_trans[bin_y_cent][bin_x_cent]

					top0 = np.sum(map_flux_corr_trans[rows1,cols1]*cent_pix_SED_flux[rows1,cols1]/(np.square(map_flux_err_corr_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)
					bottom0 = np.sum(np.square(cent_pix_SED_flux[rows1,cols1])/(np.square(map_flux_err_corr_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)
					for bb in range(0,nbands):
						norm0[bb][rows1,cols1] = top0/bottom0
					# transpose from (band,y,x) -> (y,x,band)
					norm0_trans = np.transpose(norm0, axes=(1,2,0))
					pix_chi2 = np.sum(np.square(map_flux_corr_trans[rows1,cols1]-(norm0_trans[rows1,cols1]*cent_pix_SED_flux[rows1,cols1]))/(np.square(map_flux_err_corr_trans[rows1,cols1])+np.square(cent_pix_SED_flux_err[rows1,cols1])),axis=1)

					idx_sel = np.where((pix_chi2/nbands)<=redc_chi2_limit)

					rows1_cut = rows1[idx_sel[0]]
					cols1_cut = cols1[idx_sel[0]]

					
				cumul_rows = cumul_rows + rows1_cut.tolist()
				cumul_cols = cumul_cols + cols1_cut.tolist()

				# get total fluxes
				tot_bin_flux = tot_bin_flux + np.sum(map_flux_trans[rows1_cut,cols1_cut], axis=0)
				tot_bin_flux_err2 = tot_bin_flux_err2 + np.sum(np.square(map_flux_err_trans[rows1_cut,cols1_cut]), axis=0)

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
					# check remaining pixels
					rows_rest, cols_rest = np.where((gal_region==1) & (pixbin_map==0))
					tflux = np.sum(map_flux_trans[rows_rest,cols_rest], axis=0)
					tflux_err2 = np.sum(np.square(map_flux_err_trans[rows_rest,cols_rest]), axis=0)
					tSNR = tflux/np.sqrt(tflux_err2)
					tidx = np.where(tSNR-SN_threshold>=0)

					if len(tidx[0]) < nbands:
						# bin all remaining pixels:
						count_bin = count_bin + 1
						pixbin_map[rows_rest,cols_rest] = count_bin
						map_bin_flux[rows_rest,cols_rest] = tflux
						map_bin_flux_err[rows_rest,cols_rest] = np.sqrt(tflux_err2)
						cumul_npixs_in_bin = cumul_npixs_in_bin + len(rows_rest)

						stat_increase = 0
						break

					elif (len(cumul_rows)+cumul_npixs_in_bin) == tot_npixs:
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

	print ("Number of bins: %d" % count_bin)


	## store into FITS file
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['nfilters'] = nbands
	hdr['refband'] = ref_band
	hdr['nbins'] = count_bin

	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		hdr[str_temp] = header[str_temp]

	hdul.append(fits.ImageHDU(data=pixbin_map, header=hdr, name='bin_map'))
	hdul.append(fits.ImageHDU(map_bin_flux_trans, name='bin_flux'))
	hdul.append(fits.ImageHDU(map_bin_flux_err_trans, name='bin_fluxerr'))


	if name_out_fits is None:
		name_out_fits = "pixbin.fits"

	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def open_binmap_fits(binmap_fits):
	""" Function to get the data from binned flux maps.

	:param binmap_fits:
		Input FITS file containing binned flux maps.

	:returns flag_specphoto:
		A flag stating whether the data contains spectra (value=1) or not (value=0).

	:returns nbins_photo:
		Number of spatial bins with photometric data, which is also the total number of spatial bins.

	:returns nbins_spec:
		Number of spatial bins that have spectra.

	:returns filters:
		Set of the photometric filters.

	:returns unit_flux:
		The flux unit.

	:returns binmap_photo:
		The binning map of the photometric data.

	:returns spec_region:
		The spatial region that have spectroscopy.

	:returns binmap_spec:
		Number of spatial bins that have spectroscopy.

	:returns map_photo_flux:
		The data cube of the photometric fluxes: (nbands,ny,nx)

	:returns map_photo_flux_err:
		The data cube of the photometric flux uncertainties: (nbands,ny,nx)

	:returns spec_wave:
		Spectroscopic wavelength grids.

	:returns map_spec_flux:
		The data cube of the spectroscopic fluxes: (nwaves,ny,nx)

	:returns map_spec_flux_err:
		The data cube of the spectroscopic flux uncertainties: (nwaves,ny,nx)

	"""
	hdu = fits.open(binmap_fits)
	header = hdu[0].header
	unit_flux = float(hdu[0].header['unit'])
	flag_specphoto = hdu[0].header['specphot']
	
	if flag_specphoto == 0:
		nbins_photo = int(hdu[0].header['nbins'])
		nbins_spec = 0

		binmap_photo = hdu['BIN_MAP'].data 
		spec_region = None
		binmap_spec = None
		map_photo_flux = hdu['BIN_FLUX'].data
		map_photo_flux_err = hdu['BIN_FLUXERR'].data
		spec_wave = None 
		map_spec_flux = None 
		map_spec_flux_err = None 

	elif flag_specphoto == 1:
		nbins_photo = int(hdu[0].header['nbinsph'])
		nbins_spec = int(hdu[0].header['nbinssp'])

		binmap_photo = hdu['PHOTO_BIN_MAP'].data
		spec_region = hdu['SPEC_REGION'].data 
		binmap_spec = hdu['SPEC_BIN_MAP'].data 
		map_photo_flux = hdu['BIN_PHOTO_FLUX'].data 
		map_photo_flux_err = hdu['BIN_PHOTO_FLUXERR'].data 
		spec_wave = hdu['SPEC_WAVE'].data 
		map_spec_flux = hdu['BIN_SPEC_FLUX'].data 
		map_spec_flux_err = hdu['BIN_SPEC_FLUXERR'].data 

	filters = []
	for bb in range(int(hdu[0].header['nfilters'])):
		filters.append(hdu[0].header['fil%d' % bb])

	hdu.close()

	return flag_specphoto, nbins_photo, nbins_spec, filters, unit_flux, binmap_photo, spec_region, binmap_spec, map_photo_flux, map_photo_flux_err, spec_wave, map_spec_flux, map_spec_flux_err


def plot_binmap(binmap_fits, plot_binmap_spec=True, savefig=True, name_plot_binmap_photo=None, name_plot_binmap_spec=None):
	""" Function for plotting the binning map.

	:param binmap_fits:
		Input FITS file of the binned flux maps.

	:param plot_binmap_spec:
		Decide to plot the binning map of pixels that have spectra. This input is only relevant if the data cube has both photometry and spectroscopy.

	:param savefig:
		Decide whether to save the plot of not.

	:param name_plot_binmap_photo:
		Name of the binning map of photometry.

	:param name_plot_binmap_spec:
		Name of the binning map of spectroscopy.
	"""

	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	flag_specphoto, nbins_photo, nbins_spec, filters, unit_flux, binmap_photo, spec_region, binmap_spec, map_photo_flux, map_photo_flux_err, spec_wave, map_spec_flux, map_spec_flux_err = open_binmap_fits(binmap_fits)

	cmap = plt.get_cmap('nipy_spectral_r', nbins_photo)

	##=> plot binning map of photometry
	fig1 = plt.figure(figsize=(7,7))
	f1 = plt.subplot()
	plt.setp(f1.get_yticklabels(), fontsize=14)
	plt.setp(f1.get_xticklabels(), fontsize=14)
	plt.xlabel("[pixel]", fontsize=18)
	plt.ylabel("[pixel]", fontsize=18)

	rows, cols = np.where(binmap_photo<1)
	binmap_photo1 = binmap_photo
	binmap_photo1[rows,cols] = float('nan')

	im = plt.imshow(binmap_photo1, origin='lower', cmap=cmap, vmin=0.5, vmax=nbins_photo+0.5)

	divider = make_axes_locatable(f1)
	cax2 = divider.append_axes("top", size="7%", pad="2%")
	cb = fig1.colorbar(im, cax=cax2, orientation="horizontal")
	cax2.xaxis.set_ticks_position("top")
	cax2.xaxis.set_label_position("top")
	cb.ax.tick_params(labelsize=14)
	cb.set_label('Bin Index', fontsize=18)

	if savefig is True:
		if name_plot_binmap_photo is None:
			name_plot_binmap_photo = 'binmap_photo.png'
		plt.savefig(name_plot_binmap_photo)
	else:
		plt.show()

	##=> plot binning map of spectrophotometry
	if plot_binmap_spec==True and flag_specphoto == 1:
		fig1 = plt.figure(figsize=(7,7))
		f1 = plt.subplot()
		plt.setp(f1.get_yticklabels(), fontsize=14)
		plt.setp(f1.get_xticklabels(), fontsize=14)
		plt.xlabel("[pixel]", fontsize=18)
		plt.ylabel("[pixel]", fontsize=18)

		rows, cols = np.where(spec_region == 0)
		spec_region1 = spec_region
		spec_region1[rows,cols] = float('nan')
		plt.imshow(spec_region1, origin='lower', cmap='gray', alpha=0.5, zorder=1)

		rows, cols = np.where(binmap_spec<1)
		binmap_spec1 = binmap_spec
		binmap_spec1[rows,cols] = float('nan')
		im = plt.imshow(binmap_spec1, origin='lower', cmap=cmap, vmin=0.5, vmax=nbins_photo+0.5, zorder=2)

		divider = make_axes_locatable(f1)
		cax2 = divider.append_axes("top", size="7%", pad="2%")
		cb = fig1.colorbar(im, cax=cax2, orientation="horizontal")
		cax2.xaxis.set_ticks_position("top")
		cax2.xaxis.set_label_position("top")
		cb.ax.tick_params(labelsize=14)
		cb.set_label('Bin Index', fontsize=18)

		if savefig is True:
			if name_plot_binmap_spec is None:
				name_plot_binmap_spec = 'binmap_spec.png'
			plt.savefig(name_plot_binmap_spec)
		else:
			plt.show()


def get_bins_SED_binmap(binmap_fits):
	""" Function to extract SEDs of the spatial bins from the binned flux maps.

	:param binmap_fits:
		Input FITS file of binned flux maps.

	:returns bin_photo_flux:
		Photometric fluxes of the spatial bins: (nbins,nbands)

	:returns bin_photo_flux_err:
		Photometric flux uncertainties of the spatial bins: (nbins,nbands)

	:returns bin_spec_flux:
		Spectroscopic fluxes of the spatial bins: (nbins,nwaves)

	:returns bin_spec_flux_err:
		Spectroscopic flux uncertainties of the spatial bins: (nbins,nwaves)

	:returns bin_flag_specphoto:
		A flag stating whether a spatial bin has a spectroscopy (value=1) or not (value=0).

	:returns filters:
		The set of filters.

	:returns photo_wave:
		The central wavelength of the filters.

	:returns spec_wave: 
		The wavelength grids of the spectra.
	"""

	flag_specphoto, nbins_photo, nbins_spec, filters, unit_flux, binmap_photo, spec_region, binmap_spec, map_photo_flux, map_photo_flux_err, spec_wave, map_spec_flux, map_spec_flux_err = open_binmap_fits(binmap_fits)

	# get central wavelength of filters
	from ..utils.filtering import cwave_filters

	photo_wave = cwave_filters(filters)	

	nbands = len(filters)

	if flag_specphoto == 1:
		nwaves = len(spec_wave)
	else:
		nwaves = 1
		
	bin_photo_flux = np.zeros((nbins_photo,nbands))
	bin_photo_flux_err = np.zeros((nbins_photo,nbands))
	bin_spec_flux = np.zeros((nbins_photo,nwaves))
	bin_spec_flux_err = np.zeros((nbins_photo,nwaves))

	bin_flag_specphoto = np.zeros(nbins_photo)

	for bb in range(0,nbins_photo):
		rows, cols = np.where(binmap_photo==bb+1)
		bin_photo_flux[bb] = map_photo_flux[:,rows[0],cols[0]]*unit_flux
		bin_photo_flux_err[bb] = map_photo_flux_err[:,rows[0],cols[0]]*unit_flux

		if flag_specphoto == 1:
			rows, cols = np.where(binmap_spec==bb+1)
			if len(rows)>0:
				bin_spec_flux[bb] = map_spec_flux[:,rows[0],cols[0]]*unit_flux
				bin_spec_flux_err[bb] = map_spec_flux_err[:,rows[0],cols[0]]*unit_flux
				bin_flag_specphoto[bb] = 1

	return bin_photo_flux, bin_photo_flux_err, bin_spec_flux, bin_spec_flux_err, bin_flag_specphoto, filters, photo_wave, spec_wave


def plot_bins_SNR_radial_profile(binmap_fits, xrange=None, yrange=None, savefig=True, name_plot=None):
	""" Function for plotting the S/N ratios of spatial bins.
	
	:param binmap_fits:
		FITS file of binned flux maps produced by the pixel_binning function.

	:param xrange:
		Range in x-axis.

	:param yrange:
		Range in y-axis.

	:param savefig:
		Decide whether to save the plot or not.

	:param name_plot:
		Name of the output plot. 
	"""

	import matplotlib.pyplot as plt

	flag_specphoto, nbins_photo, nbins_spec, filters, unit_flux, binmap_photo, spec_region, binmap_spec, map_photo_flux, map_photo_flux_err, spec_wave, map_spec_flux, map_spec_flux_err = open_binmap_fits(binmap_fits)

	nbands = len(filters)
	bin_photo_flux = np.zeros((nbins_photo,nbands))
	bin_photo_flux_err = np.zeros((nbins_photo,nbands))
	for bb in range(0,nbins_photo):
		rows, cols = np.where(binmap_photo==bb+1)
		bin_photo_flux[bb] = map_photo_flux[:,rows[0],cols[0]]
		bin_photo_flux_err[bb] = map_photo_flux_err[:,rows[0],cols[0]]

	bin_SNR = np.zeros((nbands,nbins_photo))
	for bb in range(0,nbands):
		bin_SNR[bb] = bin_photo_flux[:,bb]/bin_photo_flux_err[:,bb]

	bin_ids = np.arange(1,nbins_photo+1)

	# plotting
	fig1 = plt.figure(figsize=(10,6))
	f1 = plt.subplot()

	if yrange is not None:
		plt.ylim(yrange[0],yrange[1])
 
	if xrange is not None:
		plt.xlim(xrange[0],xrange[1])

	plt.setp(f1.get_yticklabels(), fontsize=13, visible=True)
	plt.setp(f1.get_xticklabels(), fontsize=13, visible=True)
	plt.xlabel(r"Bin index", fontsize=20)
	plt.ylabel(r"log(S/N Ratio)", fontsize=20)

	cmap = plt.get_cmap('jet', nbands)

	for bb in range(0,nbands):
		plt.scatter(bin_ids, np.log10(bin_SNR[bb]), s=10, alpha=0.5, color=cmap(bb), label='%s' % filters[bb])

	plt.legend(fontsize=10, ncol=2)

	if savefig is True:
		if name_plot is None:
			name_plot = 'plot_bins_SNR.png'
		plt.savefig(name_plot)
	else:
		plt.show()


def plot_bins_SED(binmap_fits, bin_ids=None, logscale_y=True, logscale_x=True, 
	wunit='angstrom', yrange=None, xrange=None, savefig=True, name_plot=None):
	""" Function for plotting SEDs of pixels.

	:param binmap_fits:
		Input FITS file.

	:param bin_ids:
		Indexes of spatial bins which SEDs will be plotted. This index start from 0.

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

	bin_photo_flux, bin_photo_flux_err, bin_spec_flux, bin_spec_flux_err, bin_flag_specphoto, filters, photo_wave, spec_wave = get_bins_SED_binmap(binmap_fits)

	if bin_ids is None:
		bin_ids = np.arange(0,len(bin_photo_flux))

	cmap = plt.get_cmap('jet', len(bin_ids))

	# Plotting
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

	plt.ylabel(r"Flux [erg $\rm{s}^{-1}\rm{cm}^{-2}\AA^{-1}$]", fontsize=18)
	plt.setp(f1.get_xticklabels(), fontsize=12)
	plt.setp(f1.get_yticklabels(), fontsize=12)

	if yrange is not None:
		plt.ylim(yrange[0],yrange[1])
 
	if xrange is not None:
		plt.xlim(xrange[0],xrange[1])


	for ii in range(len(bin_ids)):
		bin_id = int(bin_ids[ii])

		if wunit == 'micron':
			if bin_flag_specphoto[bin_id] == 1:
				plt.fill_between(spec_wave/1e+4, bin_spec_flux[bin_id]-bin_spec_flux_err[bin_id], 
							bin_spec_flux[bin_id]+bin_spec_flux_err[bin_id], color=cmap(ii), alpha=0.3)
				plt.plot(spec_wave/1e+4, bin_spec_flux[bin_id], lw=1, color=cmap(ii))

			plt.errorbar(photo_wave/1e+4, bin_photo_flux[bin_id], yerr=bin_photo_flux_err[bin_id], fmt='-o', color=cmap(ii), alpha=0.5)

		elif wunit == 'angstrom':
			if bin_flag_specphoto[bin_id] == 1:
				plt.fill_between(spec_wave, bin_spec_flux[bin_id]-bin_spec_flux_err[bin_id], 
							bin_spec_flux[bin_id]+bin_spec_flux_err[bin_id], color=cmap(ii), alpha=0.3)
				plt.plot(spec_wave, bin_spec_flux[bin_id], lw=1, color=cmap(ii))

			plt.errorbar(photo_wave, bin_photo_flux[bin_id], yerr=bin_photo_flux_err[bin_id], fmt='-o', color=cmap(ii), alpha=0.5)

	if savefig is True:
		if name_plot is None:
			name_plot = 'plot_bins_SED.png'
		plt.savefig(name_plot)
	else:
		plt.show()



