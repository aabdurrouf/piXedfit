import numpy as np
import sys, os
import fsps
import operator
from mpi4py import MPI
from astropy.io import fits

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)


from piXedfit.utils.posteriors import model_leastnorm
from piXedfit.piXedfit_model import generate_modelSED_spec, get_no_nebem_wave_fit
from piXedfit.piXedfit_spectrophotometric import spec_smoothing, match_spectra_poly_legendre_fit


"""
USAGE: mpirun -np [nproc] python ./match_specphoto.py (1)specphoto_file (2)name_saved_randmod (3)spec_sigma (4)name_out_fits
"""

if len(sys.argv)!=6:
	print ("USAGE: mpirun -np [nproc] python ./match_specphoto.py (1)specphoto_file (2)name_saved_randmod (3)spec_sigma (4)name_out_fits (5)del_wave_nebem")
	sys.exit()


global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank()


# open input FITS file
specphoto_file = str(sys.argv[1])
cube = fits.open(specphoto_file)
header = cube[0].header
photo_gal_region = cube['PHOTO_REGION'].data
spec_gal_region = cube['SPEC_REGION'].data
spec_wave = cube['WAVE'].data
photo_flux = cube['PHOTO_FLUX'].data           			# structure: (band,y,x)
photo_flux_err = cube['PHOTO_FLUXERR'].data
spec_flux = cube['SPEC_FLUX'].data             			# structure: (wavelength,y,x)
spec_flux_err = cube['SPEC_FLUXERR'].data
#spec_good_pix = cube['spec_good_pix'].data
cube.close()

# min and max of spec_wave:
nwaves = len(spec_wave)
min_spec_wave = min(spec_wave)
max_spec_wave = max(spec_wave)

# spectral resolution
spec_sigma = float(sys.argv[3])

# output name
name_out_fits = str(sys.argv[4])

# dimension
dim_y = photo_gal_region.shape[0]
dim_x = photo_gal_region.shape[1]

# set of filters
nbands = int(header['nfilters'])
filters = []
for ii in range(0,nbands):
	str_temp = 'fil%d' % ii
	filters.append(header[str_temp])

# unit of flux
unit = float(header['unit'])

# redshift
gal_z = float(header['z'])

# transpose (band,y,x) => (y,x,band)
photo_flux_trans = np.transpose(photo_flux, axes=(1,2,0))*unit
photo_flux_err_trans = np.transpose(photo_flux_err, axes=(1,2,0))*unit

# transpose (wave,y,x) => (y,x,wave)
spec_flux_trans = np.transpose(spec_flux, axes=(1,2,0))*unit 
spec_flux_err_trans = np.transpose(spec_flux_err, axes=(1,2,0))*unit 
#spec_good_pix_trans = np.transpose(spec_good_pix, axes=(1,2,0))*unit 

# select pixels
rows, cols = np.where(spec_gal_region==1)
npixs = len(rows)

# data of pre-calculated model SEDs
name_saved_randmod = str(sys.argv[2])
hdu = fits.open(name_saved_randmod)
header_randmod = hdu[0].header
data_randmod = hdu[1].data
hdu.close()

# number of model SEDs
npmod_seds0 = int(header_randmod['nrows'])
npmod_seds = int(npmod_seds0/size)*size

# basic properties models
add_neb_emission = int(header_randmod['add_neb_emission'])
#add_neb_emission = 0
gas_logu = float(header_randmod['gas_logu'])
sfh_form = header_randmod['sfh_form']
imf = int(header_randmod['imf_type'])
#duste_switch = header_randmod['duste_switch']
duste_switch = 'noduste'
dust_ext_law = header_randmod['dust_ext_law']
#add_igm_absorption = int(header_randmod['add_igm_absorption'])
add_igm_absorption = 0
#add_agn = int(header_randmod['add_agn'])
add_agn = 0

# list of parameters
global params, nparams
nparams = int(header_randmod['nparams'])
params = []
for ii in range(0,nparams):
	str_temp = 'param%d' % ii
	params.append(header_randmod[str_temp])


####====> SED fitting to each pixel
def_params_val={'dust1': 0.5, 'dust2': 0.5, 'dust_index': -0.7, 'log_age': 1.0, 'log_alpha': 0.1, 
				'log_beta': 0.1, 'log_fagn': -3.0, 'log_gamma': -2.0, 'log_mass': 0.0, 'log_qpah': 0.54, 
				'log_t0': 0.4, 'log_tau': 0.4, 'log_tauagn': 1.0, 'log_umin': 0.0, 'logzsol': 0.0, 'z': 0.001}
# call FSPS
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)


# arrays for outputs
rescaled_spec_flux = np.zeros((dim_y,dim_x,nwaves))
rescaled_spec_flux_err = np.zeros((dim_y,dim_x,nwaves))

#========================================#
# generate one model for setting up
params_val = def_params_val
params_val['z'] = gal_z
spec_SED = generate_modelSED_spec(sp=sp, imf_type=imf, duste_switch=duste_switch, 
								add_neb_emission=add_neb_emission, dust_ext_law=dust_ext_law, sfh_form=sfh_form, 
								add_agn=add_agn, add_igm_absorption=add_igm_absorption, gas_logu=gas_logu, 
								params_val=def_params_val)
# cut model spectrum to match range given by the IFS spectra
idx_mod_wave = np.where((spec_SED['wave']>min_spec_wave-50) & (spec_SED['wave']<max_spec_wave+50))
#========================================#

# get wavelength free of emission lines:
del_wave_nebem = float(sys.argv[5])
spec_wave_clean,wave_mask = get_no_nebem_wave_fit(sp,gal_z,spec_wave,del_wave_nebem)

# allocate memory for output best-fit model spectra
#map_bfit_mod_spec_wave = spec_SED['wave'][idx_mod_wave[0]]
#nwaves_bfit_spec = len(map_bfit_mod_spec_wave)
map_bfit_mod_spec_wave = spec_wave_clean
map_bfit_mod_spec_flux = np.zeros((dim_y,dim_x,len(spec_wave_clean)))


for pp in range(0,npixs):
	# obs SED
	obs_flux = photo_flux_trans[rows[pp]][cols[pp]]
	obs_flux_err = photo_flux_err_trans[rows[pp]][cols[pp]]

	# exclude negative fluxes
	idx = np.where(obs_flux>0.0)

	# set up calculation
	numDataPerRank = int(npmod_seds/size)
	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_id_temp = np.zeros(numDataPerRank)
	mod_log_mass_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# model ID:
		mod_id_temp[int(count)] = int(ii)

		# model SED
		mod_flux = np.zeros(nbands)
		for bb in range(0,nbands):
			mod_flux[bb] = data_randmod[filters[bb]][int(ii)]

		# calculate chi-square
		norm = model_leastnorm(obs_flux[idx[0]],obs_flux_err[idx[0]],mod_flux[idx[0]])
		mod_flux = mod_flux*norm
		chi2 = np.sum(np.square(mod_flux[idx[0]]-obs_flux[idx[0]])/obs_flux_err[idx[0]]/obs_flux_err[idx[0]])
		mod_chi2_temp[int(count)] = chi2

		# log_mass
		mod_log_mass_temp[int(count)] = data_randmod['log_mass'][int(ii)]+np.log10(norm)

		count = count + 1

		#sys.stdout.write('\r')
		#sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),
		#																			count*100/len(recvbuf_idx)))
		#sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_id = np.zeros(numDataPerRank*size)
	mod_log_mass = np.zeros(numDataPerRank*size)

	# gather data
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_id_temp, mod_id, root=0)
	comm.Gather(mod_log_mass_temp, mod_log_mass, root=0)

	
	if rank == 0:
		# get best-fit
		idx0, min_val = min(enumerate(mod_chi2), key=operator.itemgetter(1))

		# parameters of the best-fit model
		params_val = def_params_val
		for jj in range(0,nparams):
			params_val[params[jj]] = data_randmod[params[jj]][int(mod_id[idx0])]
		params_val['log_mass'] = mod_log_mass[idx0]
		params_val['z'] = gal_z

		# get model spectrum
		spec_SED = generate_modelSED_spec(sp=sp, imf_type=imf, duste_switch=duste_switch, 
								add_neb_emission=add_neb_emission, dust_ext_law=dust_ext_law, sfh_form=sfh_form, 
								add_agn=add_agn, add_igm_absorption=add_igm_absorption, gas_logu=gas_logu, 
								params_val=params_val)
		bfit_spec_wave = spec_SED['wave']
		bfit_spec_flux = spec_SED['flux']

		# cut model spectrum to match range given by the IFS spectra
		#idx0 = np.where((bfit_spec_wave>min_spec_wave-50) & (bfit_spec_wave<max_spec_wave+50))
		bfit_spec_wave = bfit_spec_wave[idx_mod_wave[0]]
		bfit_spec_flux = bfit_spec_flux[idx_mod_wave[0]]

		# smoothing model spectrum to meet resolution of IFS
		conv_bfit_spec_wave,conv_bfit_spec_flux = spec_smoothing(bfit_spec_wave,bfit_spec_flux,spec_sigma)

		# match scaling/normalization of IFS spectra to the best-fit model spectrum
		in_spec_flux = spec_flux_trans[rows[pp]][cols[pp]]
		final_wave,final_flux,factor,ref_spec_flux_clean = match_spectra_poly_legendre_fit(sp=sp,in_spec_wave=spec_wave,in_spec_flux=in_spec_flux,
																		ref_spec_wave=conv_bfit_spec_wave,ref_spec_flux=conv_bfit_spec_flux,
																		wave_clean=spec_wave_clean,z=gal_z,del_wave_nebem=del_wave_nebem,order=3)

		# get output re-scaled spectrum
		rescaled_spec_flux[rows[pp]][cols[pp]] = final_flux
		rescaled_spec_flux_err[rows[pp]][cols[pp]] = spec_flux_err_trans[rows[pp]][cols[pp]]*factor

		# get output best-fit model
		#map_bfit_mod_spec_flux[rows[pp]][cols[pp]] = bfit_spec_flux
		map_bfit_mod_spec_flux[rows[pp]][cols[pp]] = ref_spec_flux_clean


if rank == 0:
	# transpose (y,x,wave) => (wave,y,x) and re-normalize 
	map_rescaled_spec_flux = np.transpose(rescaled_spec_flux, axes=(2,0,1))/unit
	map_rescaled_spec_flux_err = np.transpose(rescaled_spec_flux_err, axes=(2,0,1))/unit
	map_bfit_mod_spec_flux_trans = np.transpose(map_bfit_mod_spec_flux, axes=(2,0,1))/unit


	# Store into FITS file 
	hdul = fits.HDUList()
	hdul.append(fits.ImageHDU(data=photo_flux, header=header, name='photo_flux'))
	hdul.append(fits.ImageHDU(photo_flux_err, name='photo_fluxerr'))
	hdul.append(fits.ImageHDU(spec_wave, name='wave'))
	hdul.append(fits.ImageHDU(map_rescaled_spec_flux, name='spec_flux'))
	hdul.append(fits.ImageHDU(map_rescaled_spec_flux_err , name='spec_fluxerr'))
	hdul.append(fits.ImageHDU(photo_gal_region, name='photo_region'))
	hdul.append(fits.ImageHDU(spec_gal_region, name='spec_region'))
	hdul.append(fits.ImageHDU(map_bfit_mod_spec_wave, name='mod_wave'))
	hdul.append(fits.ImageHDU(map_bfit_mod_spec_flux_trans, name='mod_flux'))
	# write to fits file
	hdul.writeto(name_out_fits, overwrite=True)




