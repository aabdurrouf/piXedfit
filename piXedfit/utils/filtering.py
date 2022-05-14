import numpy as np 
import h5py
import os, sys
from astropy.io import fits

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

__all__ = ["list_filters", "add_filter", "remove_filter", "change_filter_name", "get_filter_curve", 
			"cwave_filters", "filtering", "interp_filters_curves", "filtering_interp_filters"]

dir_file = PIXEDFIT_HOME+'/data/filters/'

def convert_fits_to_hdf5():
	# open file
	hdu = fits.open(dir_file+"filters.fits")
	header = hdu[0].header
	
	# get filters and their central wavelength
	nbands = int(header['nfilters'])
	fil_cwave = np.zeros(nbands)
	filters = []
	for bb in range(0,nbands):
		str_temp = 'fil%d' % (bb+1)
		filters0 = header[str_temp]
		filters.append(filters0)
		str_temp = 'cw_%s' % filters0
		fil_cwave[bb] = float(header[str_temp])

	# get transmission curves
	f_wave = {}
	f_trans = {}
	for bb in range(0,nbands):
		data = hdu[filters[bb]].data
		f_wave[filters[bb]] = data['wave']
		f_trans[filters[bb]] = data['trans'] 

	# store to HDF5 file
	with h5py.File('filters_w.hdf5', 'w') as f:
		for bb in range(0,nbands):
			dset = f.create_dataset(filters[bb], data=np.array(f_wave[filters[bb]]), compression="gzip")
			str_temp = 'cw_%s' % filters[bb]
			dset.attrs[str_temp] = fil_cwave[bb]

	with h5py.File('filters_t.hdf5', 'w') as f:
		for bb in range(0,nbands):
			dset = f.create_dataset(filters[bb], data=np.array(f_trans[filters[bb]]), compression="gzip")

	hdu.close()

	os.system('mv filters_w.hdf5 filters_t.hdf5 %s' % dir_file)


def get_all(name):
    filters.append(name)
    print(name)

def list_filters():
	"""A function for listing the available filters transmission functions in piXedfit

	:returns filters:
		List of filters curves available
	"""
	global filters 
	filters = []
	with h5py.File(dir_file+'filters_w.hdf5', 'r') as f:
		f.visit(get_all)

	return filters


def get_all_noprint(name):
    filters.append(name)

def list_filters_noprint():
	"""A function for listing the available filters transmission functions in piXedfit

	:returns filters:
		List of filters curves available
	"""
	global filters 
	filters = []
	with h5py.File(dir_file+'filters_w.hdf5', 'r') as f:
		f.visit(get_all_noprint)

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

	f = h5py.File(dir_file+'filters_w.hdf5', 'a')
	dset = f.create_dataset(filter_name, data=np.array(filter_wave), compression="gzip")
	str_temp = 'cw_%s' % filter_name
	dset.attrs[str_temp] = filter_cwave
	f.close()

	f = h5py.File(dir_file+'filters_t.hdf5', 'a')
	f.create_dataset(filter_name, data=np.array(filter_transmission), compression="gzip")
	f.close()

	os.system('mv filters_w.hdf5 filters_t.hdf5 %s' % dir_file)


def remove_filter(filter_name):
	"""A function for removing a filter transmission function from piXedfit

	:param filter_name:
		The filter name.
	"""

	# get list of filters
	filters = list_filters_noprint()
	nbands = len(filters)

	# get the filters data: wave
	f_wave = {}
	fil_cwave = np.zeros(nbands)
	f = h5py.File(dir_file+'filters_w.hdf5', 'r')
	for bb in range(0,nbands):
		f_wave[filters[bb]] = f[filters[bb]][:]
		str_temp = 'cw_%s' % filters[bb]
		fil_cwave[bb] = f[filters[bb]].attrs[str_temp]
	f.close()

	# get the filters data: transmission
	f_trans = {}
	f = h5py.File(dir_file+'filters_t.hdf5', 'r')
	for bb in range(0,nbands):
		f_trans[filters[bb]] = f[filters[bb]][:]
	f.close()

	# make new files
	with h5py.File('filters_w.hdf5', 'w') as f:
		for bb in range(0,nbands):
			if filters[bb] != filter_name:
				dset = f.create_dataset(filters[bb], data=np.array(f_wave[filters[bb]]), compression="gzip")
				str_temp = 'cw_%s' % filters[bb]
				dset.attrs[str_temp] = fil_cwave[bb]

	with h5py.File('filters_t.hdf5', 'w') as f:
		for bb in range(0,nbands):
			if filters[bb] != filter_name:
				dset = f.create_dataset(filters[bb], data=np.array(f_trans[filters[bb]]), compression="gzip")

	os.system('mv filters_w.hdf5 filters_t.hdf5 %s' % dir_file)


def change_filter_name(old_filter_name, new_filter_name):
	"""A function for changing a filter name

	:param old_filter_name:
		Old filter name.

	:param new_filter_name:
		New filter name.
	"""

	# get list of filters
	filters = list_filters_noprint()
	nbands = len(filters)

	# get the filters data: wave
	f_wave = {}
	fil_cwave = np.zeros(nbands)
	f = h5py.File(dir_file+'filters_w.hdf5', 'r')
	for bb in range(0,nbands):
		f_wave[filters[bb]] = f[filters[bb]][:]
		str_temp = 'cw_%s' % filters[bb]
		fil_cwave[bb] = f[filters[bb]].attrs[str_temp]
	f.close()

	# get the filters data: transmission
	f_trans = {}
	f = h5py.File(dir_file+'filters_t.hdf5', 'r')
	for bb in range(0,nbands):
		f_trans[filters[bb]] = f[filters[bb]][:]
	f.close()

	# make new files
	with h5py.File('filters_w.hdf5', 'w') as f:
		for bb in range(0,nbands):
			if filters[bb] == old_filter_name:
				dset = f.create_dataset(new_filter_name, data=np.array(f_wave[filters[bb]]), compression="gzip")
				str_temp = 'cw_%s' % new_filter_name
				dset.attrs[str_temp] = fil_cwave[bb]
			else:
				dset = f.create_dataset(filters[bb], data=np.array(f_wave[filters[bb]]), compression="gzip")
				str_temp = 'cw_%s' % filters[bb]
				dset.attrs[str_temp] = fil_cwave[bb]

	with h5py.File('filters_t.hdf5', 'w') as f:
		for bb in range(0,nbands):
			if filters[bb] == old_filter_name:
				dset = f.create_dataset(new_filter_name, data=np.array(f_trans[filters[bb]]), compression="gzip")
			else:
				dset = f.create_dataset(filters[bb], data=np.array(f_trans[filters[bb]]), compression="gzip")

	os.system('mv filters_w.hdf5 filters_t.hdf5 %s' % dir_file)


def get_filter_curve(filter_name):
	"""A function to get a transmission function of a filter available in piXedfit

	:param filter_name:
		Name of the filter

	:returns wave:
		Array of wavelength

	:returns trans:
		Array of transmission values 
	"""

	f = h5py.File(dir_file+'filters_w.hdf5', 'r')
	wave = f[filter_name][:]
	f.close()

	f = h5py.File(dir_file+'filters_t.hdf5', 'r')
	trans = f[filter_name][:]
	f.close()

	return wave, trans


def cwave_filters(filters):
	"""A function for retrieving central wavelengths of a set of filters

	:param filters:
		A list of filters names

	:returns cwaves:
		A list of central wavelengths of the filters
	"""

	f = h5py.File(dir_file+'filters_w.hdf5', 'r')
	nbands = len(filters)
	cwaves = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'cw_%s' % filters[bb]
		cwaves[bb] = f[filters[bb]].attrs[str_temp]
	f.close()

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

	w = h5py.File(dir_file+'filters_w.hdf5', 'r')
	t = h5py.File(dir_file+'filters_t.hdf5', 'r')

	nbands = len(filters)
	fluxes = np.zeros(nbands)
	for bb in range(0,nbands):
		fil_w = w[filters[bb]][:]
		fil_t = t[filters[bb]][:]

		min_wave = int(min(fil_w))
		max_wave = int(max(fil_w))

		gwave = np.linspace(min_wave,max_wave,max_wave-min_wave+1)

		fil_trans = np.interp(gwave, fil_w, fil_t)
		spec_flux = np.interp(gwave, wave, spec)

		tot_u = np.sum(spec_flux*gwave*fil_trans)
		tot_l = np.sum(gwave*fil_trans)

		fluxes[bb] = tot_u/tot_l

	w.close()
	t.close()

	return fluxes


def interp_filters_curves(filters):

	w = h5py.File(dir_file+'filters_w.hdf5', 'r')
	t = h5py.File(dir_file+'filters_t.hdf5', 'r')

	nbands = len(filters)

	interp_filters_waves = []
	interp_filters_trans = []
	for bb in range(0,nbands):
		fil_w = w[filters[bb]][:]
		fil_t = t[filters[bb]][:]

		min_wave = int(min(fil_w))
		max_wave = int(max(fil_w))

		gwave = np.linspace(min_wave,max_wave,max_wave-min_wave+1)
		interp_filters_waves.append(gwave)

		fil_trans = np.interp(gwave, fil_w, fil_t)
		interp_filters_trans.append(fil_trans)

	w.close()
	t.close()

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


