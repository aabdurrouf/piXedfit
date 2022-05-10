import numpy as np 
import math
import os, sys
from astropy.io import fits


global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

__all__ = ["list_filters", "add_filter", "remove_filter", "change_filter_name", "get_filter_curve", "cwave_filters", "filtering", 
			"match_filters_array", "filtering_match_filters_array", "interp_filters_curves", "filtering_interp_filters"]

def list_filters():
	"""A function for listing the available filters transmission functions in piXedfit

	:returns filters:
		List of filters curves available
	"""
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	hdu = fits.open(dir_file+'filters.fits')
	nfilters = int(hdu[0].header['nfilters'])
	filters = []
	for ii in range(0,nfilters):
		str_temp = 'fil%d' % (ii+1)
		filters.append(hdu[0].header[str_temp])
	hdu.close()

	return filters

def add_filter(filter_name,filter_wave,filter_transmission,filter_cwave):
	"""A function for adding a new filter transmission function into piXedfit

	:param filter_name:
		A given name (in string) for the filter curve
	:param filter_wave:
		array of wavelength in the filter transmission function
	:param filter_transmission:
		array of transmission corresponding with the filter_wave
	:param filter_cwave:
		The central wavelength or effective wavelength of the filter
	"""

	# get the old filters.fits
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	old_filters = fits.open(dir_file+'filters.fits')
	old_header = old_filters[0].header
	# number of filters curves:
	old_nfilters = int(old_header['nfilters'])
	# get the filters:
	old_fil_wave = []
	old_fil_trans = []
	old_fil_name = []
	old_fil_cwave = np.zeros(old_nfilters)
	for bb in range(0,old_nfilters):
		old_fil_wave.append([])
		old_fil_trans.append([])

		# get name of the filter:
		str_temp = 'fil%d' % (bb+1)
		fil_name = old_header[str_temp]
		old_fil_name.append(fil_name)
		# get central wave of the filter:
		str_temp = 'cw_%s' % fil_name
		old_fil_cwave[bb] = float(old_header[str_temp])
		# get the transmission curve:
		data = old_filters[fil_name].data
		old_fil_wave[bb] = data['wave']
		old_fil_trans[bb] = data['trans']

	old_filters.close()

	# make new fits file:
	hdr = fits.Header()
	hdr['nfilters'] = old_nfilters + 1
	# add the old filters
	for bb in range(0,old_nfilters):
		temp_str = 'fil%d' % (bb+1)
		hdr[temp_str] = old_fil_name[bb]
		temp_str = 'cw_%s' % old_fil_name[bb]
		hdr[temp_str] = old_fil_cwave[bb]
	# add the new filter:
	temp_str = 'fil%d' % (old_nfilters + 1)
	hdr[temp_str] = filter_name
	temp_str = 'cw_%s' % filter_name
	hdr[temp_str] = filter_cwave

	# add to the first HDU:
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul = fits.HDUList([primary_hdu])

	# make one HDU for one filter:
	for bb in range(0,old_nfilters):
		col1_array = np.array(old_fil_wave[int(bb)])    
		col1 = fits.Column(name='wave', format='D', array=col1_array) 
		col2_array = np.array(old_fil_trans[int(bb)])     
		col2 = fits.Column(name='trans', format='D', array=col2_array) 
		cols = fits.ColDefs([col1, col2])
		hdu = fits.BinTableHDU.from_columns(cols, name=old_fil_name[bb])
		hdul.append(hdu)
	# add the new filter:
	col1_array = np.array(filter_wave)    
	col1 = fits.Column(name='wave', format='D', array=col1_array) 
	col2_array = np.array(filter_transmission)     
	col2 = fits.Column(name='trans', format='D', array=col2_array) 
	cols = fits.ColDefs([col1, col2])
	hdu = fits.BinTableHDU.from_columns(cols, name=filter_name)
	hdul.append(hdu)

	# write to fits file:
	hdul.writeto('filters.fits', overwrite=True)
	# move it to the original directory:
	os.system('mv filters.fits %s' % dir_file)


def remove_filter(filter_name):
	"""A function for removing a filter transmission function from piXedfit

	:param filter_name:
		The filter name.
	"""

	# get the old filters.fits
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	old_filters = fits.open(dir_file+'filters.fits')
	old_header = old_filters[0].header
	# number of filters curves:
	old_nfilters = int(old_header['nfilters'])
	# get the filters:
	old_fil_wave = []
	old_fil_trans = []
	old_fil_name = []
	old_fil_cwave = np.zeros(old_nfilters)
	for bb in range(0,old_nfilters):
		old_fil_wave.append([])
		old_fil_trans.append([])

		# get name of the filter:
		str_temp = 'fil%d' % (bb+1)
		fil_name = old_header[str_temp]
		old_fil_name.append(fil_name)
		# get central wave of the filter:
		str_temp = 'cw_%s' % fil_name
		old_fil_cwave[bb] = float(old_header[str_temp])
		# get the transmission curve:
		data = old_filters[fil_name].data
		old_fil_wave[bb] = data['wave']
		old_fil_trans[bb] = data['trans']

	old_filters.close()

	# make new fits file:
	hdr = fits.Header()
	hdr['nfilters'] = old_nfilters - 1
	# list the old filters and delete the selected one
	cc = 0
	for bb in range(0,old_nfilters):
		if old_fil_name[bb] != filter_name:
			temp_str = 'fil%d' % (cc+1)
			hdr[temp_str] = old_fil_name[bb]
			temp_str = 'cw_%s' % old_fil_name[bb]
			hdr[temp_str] = old_fil_cwave[bb]
			cc = cc + 1
	# add to the first HDU:
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul = fits.HDUList([primary_hdu])

	# make one HDU for one filter:
	for bb in range(0,old_nfilters):
		if old_fil_name[bb] != filter_name:
			col1_array = np.array(old_fil_wave[bb])    
			col1 = fits.Column(name='wave', format='D', array=col1_array) 
			col2_array = np.array(old_fil_trans[bb])     
			col2 = fits.Column(name='trans', format='D', array=col2_array) 
			cols = fits.ColDefs([col1, col2])
			hdu = fits.BinTableHDU.from_columns(cols, name=old_fil_name[bb])
			hdul.append(hdu)

	# write to fits file:
	hdul.writeto('filters.fits', overwrite=True)
	# move it to the original directory:
	os.system('mv filters.fits %s' % dir_file)


def change_filter_name(old_filter_name, new_filter_name):
	"""A function for changing a filter name

	:param old_filter_name:
		Old filter name.

	:param new_filter_name:
		New filter name.
	"""

	# get the old filters.fits
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	old_filters = fits.open(dir_file+'filters.fits')
	old_header = old_filters[0].header
	# number of filters curves:
	old_nfilters = int(old_header['nfilters'])
	# get the filters:
	old_fil_wave = []
	old_fil_trans = []
	old_fil_name = []
	old_fil_cwave = np.zeros(old_nfilters)
	for bb in range(0,old_nfilters):
		old_fil_wave.append([])
		old_fil_trans.append([])
		# get name of the filter:
		str_temp = 'fil%d' % (bb+1)
		fil_name = old_header[str_temp]
		old_fil_name.append(fil_name)
		# get central wave of the filter:
		str_temp = 'cw_%s' % fil_name
		old_fil_cwave[bb] = float(old_header[str_temp])
		# get the transmission curve:
		data = old_filters[fil_name].data
		old_fil_wave[bb] = data['wave']
		old_fil_trans[bb] = data['trans']
	old_filters.close()

	# make new FITS file
	hdr = fits.Header()
	# make header
	hdr['nfilters'] = old_nfilters
	for bb in range(0,old_nfilters):
		if old_fil_name[bb] == old_filter_name:
			temp_str = 'fil%d' % (bb+1)
			hdr[temp_str] = new_filter_name
			temp_str = 'cw_%s' % new_filter_name
			hdr[temp_str] = old_fil_cwave[bb]
		else:
			temp_str = 'fil%d' % (bb+1)
			hdr[temp_str] = old_fil_name[bb]
			temp_str = 'cw_%s' % old_fil_name[bb]
			hdr[temp_str] = old_fil_cwave[bb]
	# add to the first HDU:
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul = fits.HDUList([primary_hdu])

	# make one HDU for one filter:
	for bb in range(0,old_nfilters):
		if old_fil_name[bb] == old_filter_name:
			col1_array = np.array(old_fil_wave[bb])    
			col1 = fits.Column(name='wave', format='D', array=col1_array) 
			col2_array = np.array(old_fil_trans[bb])     
			col2 = fits.Column(name='trans', format='D', array=col2_array) 
			cols = fits.ColDefs([col1, col2])
			hdu = fits.BinTableHDU.from_columns(cols, name=new_filter_name)
			hdul.append(hdu)
		else:
			col1_array = np.array(old_fil_wave[bb])    
			col1 = fits.Column(name='wave', format='D', array=col1_array) 
			col2_array = np.array(old_fil_trans[bb])     
			col2 = fits.Column(name='trans', format='D', array=col2_array) 
			cols = fits.ColDefs([col1, col2])
			hdu = fits.BinTableHDU.from_columns(cols, name=old_fil_name[bb])
			hdul.append(hdu)

	# write to fits file:
	hdul.writeto('filters.fits', overwrite=True)
	# move it to the original directory:
	os.system('mv filters.fits %s' % dir_file)



def get_filter_curve(filter_name):
	"""A function to get a transmission function of a filter available in piXedfit

	:param filter_name:
		Name of the filter

	:returns wave:
		Array of wavelength

	:returns trans:
		Array of transmission values 
	"""
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	hdu = fits.open(dir_file+'filters.fits')
	wave = hdu[filter_name].data['wave']
	trans = hdu[filter_name].data['trans']

	return wave, trans 


def cwave_filters(filters):
	"""A function for retrieving central wavelengths of a set of filters

	:param filters:
		A list of filters names

	:returns cwaves:
		A list of central wavelengths of the filters
	"""
	nfilters = len(filters)
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	hdu = fits.open(dir_file+'filters.fits')
	header = hdu[0].header
	hdu.close()
	cwaves = np.zeros(nfilters)
	for ii in range(0,nfilters):
		str_temp = 'cw_%s' % filters[int(ii)]
		cwaves[int(ii)] = float(header[str_temp])

	return cwaves


def filtering(wave,spec,filters):
	"""A function for integrating a spectrum through a filter transmission function

	:param wave:
		array of wavelength of the input spectrum

	:param spec:
		array of fluxes of the input spectrum

	:param filters:
		List of filters name in array of string

	:returns fluxes:
		Array of photometric fluxes
	"""
	nbands = len(filters)
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	hdu = fits.open(dir_file+'filters.fits')

	fluxes = np.zeros(nbands)
	for bb in range(0,nbands):
		data = hdu[filters[bb]].data

		min_wave = int(min(data['wave']))
		max_wave = int(max(data['wave']))

		gwave = np.linspace(min_wave,max_wave,max_wave-min_wave+1)

		fil_trans = np.interp(gwave, data['wave'], data['trans'])
		spec_flux = np.interp(gwave, wave, spec)

		tot_u = np.sum(spec_flux*gwave*fil_trans)
		tot_l = np.sum(gwave*fil_trans)

		fluxes[bb] = tot_u/tot_l

	hdu.close()

	return fluxes


def interp_filters_curves(filters):
	nbands = len(filters)

	dir_file = PIXEDFIT_HOME+'/data/filters/'
	hdu = fits.open(dir_file+'filters.fits')

	interp_filters_waves = []
	interp_filters_trans = []
	for bb in range(0,nbands):
		data = hdu[filters[bb]].data

		min_wave = int(min(data['wave']))
		max_wave = int(max(data['wave']))

		gwave = np.linspace(min_wave,max_wave,max_wave-min_wave+1)
		interp_filters_waves.append(gwave)

		fil_trans = np.interp(gwave, data['wave'], data['trans'])
		interp_filters_trans.append(fil_trans)

	hdu.close()

	return interp_filters_waves,interp_filters_trans


def filtering_interp_filters(wave,spec,interp_filters_waves,interp_filters_trans):
	nbands = len(interp_filters_waves)

	fluxes = np.zeros(nbands)
	for bb in range(0,nbands):
		gwave = np.asarray(interp_filters_waves[bb])
		fil_trans = np.asarray(interp_filters_trans[bb])

		spec_flux = np.interp(gwave, wave, spec)

		tot_u = np.sum(spec_flux*gwave*fil_trans)
		tot_l = np.sum(gwave*fil_trans)

		fluxes[bb] = tot_u/tot_l

	return fluxes

def match_filters_array(sample_spec_wave,filters):
	"""A function for matching between wavelength in filter transmission curve to the 
	wavelength in the spectrum, based on the redshift of the spectrum
	"""

	nbands = len(filters)
	nwave = len(sample_spec_wave)
	# get filters
	dir_file = PIXEDFIT_HOME+'/data/filters/'
	hdul = fits.open(dir_file+'filters.fits')
	header = hdul[0].header
	wave_fltr = []   			### wave_fltr[idx-filter][idx-wave]
	trans_fltr = []
	for bb in range(0,nbands):
		wave_fltr.append([])
		trans_fltr.append([])
		data = hdul[filters[bb]].data
		wave_fltr[bb] = data['wave']
		trans_fltr[bb] = data['trans']

	trans_fltr_int = np.zeros((nbands,nwave))
	for bb in range(0,nbands):
		min_wave = min(wave_fltr[bb])
		max_wave = max(wave_fltr[bb])
		wave_spec_temp = []
		wave_spec_idx = []
		for ii in range(0,nwave):
			if min_wave<=sample_spec_wave[ii]<=max_wave:
				wave_spec_temp.append(sample_spec_wave[ii])
				wave_spec_idx.append(ii)

		trans_fltr_int_part = np.interp(wave_spec_temp, wave_fltr[bb], trans_fltr[bb]) 
		# store to the array
		for ii in range(0,len(wave_spec_idx)):
			idx = wave_spec_idx[ii]
			trans_fltr_int[bb][idx] = trans_fltr_int_part[ii]
	return trans_fltr_int

def filtering_match_filters_array(wave,spec,filters,trans_fltr_int):
	if np.sum(spec) == 0:
		nwaves = len(wave)
		wave = np.linspace(1,nwaves,nwaves)
		spec = np.zeros(nwaves) + 1.0e-5
	nfilters = len(filters)
	fluxes = np.zeros(nfilters)
	for bb in range(0,nfilters):
		tot_u = np.sum(spec*wave*trans_fltr_int[bb])
		tot_l = np.sum(wave*trans_fltr_int[bb])
		fluxes[bb] = tot_u/tot_l
	return fluxes





