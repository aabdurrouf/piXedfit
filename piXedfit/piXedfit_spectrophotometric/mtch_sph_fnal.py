import numpy as np
import sys, os
import h5py
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from astropy.cosmology import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.posteriors import model_leastnorm
from piXedfit.piXedfit_model import get_no_nebem_wave_fit
from piXedfit.piXedfit_spectrophotometric import spec_smoothing, match_spectra_poly_legendre_fit
from piXedfit.utils.filtering import interp_filters_curves, filtering_interp_filters
from piXedfit.utils.redshifting import cosmo_redshifting 

"""
USAGE: mpirun -np [nproc] python ./match_specphoto.py (1)configuration file
"""

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank()

temp_dir = PIXEDFIT_HOME+'/data/temp/' 
dir_mod = PIXEDFIT_HOME+'/data/mod/'

# configuration file
config_file = str(sys.argv[1])
data = np.genfromtxt(temp_dir+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]

# open input FITS file
specphoto_file = config_data['specphoto_file']
cube = fits.open(specphoto_file)
header = cube[0].header
photo_gal_region = cube['PHOTO_REGION'].data
spec_gal_region = cube['SPEC_REGION'].data
spec_wave = cube['WAVE'].data
photo_flux = cube['PHOTO_FLUX'].data           			# structure: (band,y,x)
photo_flux_err = cube['PHOTO_FLUXERR'].data
spec_flux = cube['SPEC_FLUX'].data             			# structure: (wavelength,y,x)
spec_flux_err = cube['SPEC_FLUXERR'].data
cube.close()

# min and max of spec_wave:
nwaves = len(spec_wave)
min_spec_wave = min(spec_wave)
max_spec_wave = max(spec_wave)

# spectral resolution
if header['ifssurve'] == 'manga':
	spec_sigma = 3.5
elif header['ifssurve'] == 'califa':
	spec_wave = 2.6

# output name
name_out_fits = config_data['name_out_fits']

# dimension
dim_y = photo_gal_region.shape[0]
dim_x = photo_gal_region.shape[1]

# set of filters
nbands = int(header['nfilters'])
filters = []
for ii in range(0,nbands):
	str_temp = 'fil%d' % ii
	filters.append(header[str_temp])

interp_filters_waves,interp_filters_trans = interp_filters_curves(filters)

# unit of flux
unit = float(header['unit'])

# redshift
gal_z = float(header['z'])

# cosmology
cosmo = int(config_data['cosmo'])
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])
if cosmo=='flat_LCDM' or cosmo==0:
	cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
	DL_Gpc0 = cosmo1.luminosity_distance(gal_z)      # in unit of Mpc
elif cosmo=='WMAP5' or cosmo==1:
	DL_Gpc0 = WMAP5.luminosity_distance(gal_z)
elif cosmo=='WMAP7' or cosmo==2:
	DL_Gpc0 = WMAP7.luminosity_distance(gal_z)
elif cosmo=='WMAP9' or cosmo==3:
	DL_Gpc0 = WMAP9.luminosity_distance(gal_z)
elif cosmo=='Planck13' or cosmo==4:
	DL_Gpc0 = Planck13.luminosity_distance(gal_z)
elif cosmo=='Planck15' or cosmo==5:
	DL_Gpc0 = Planck15.luminosity_distance(gal_z)
#elif cosmo=='Planck18' or cosmo==6:
#	DL_Gpc0 = Planck18.luminosity_distance(gl_z)
DL_Gpc = DL_Gpc0.value/1.0e+3 

# transpose (band,y,x) => (y,x,band)
photo_flux_trans = np.transpose(photo_flux, axes=(1,2,0))
photo_flux_err_trans = np.transpose(photo_flux_err, axes=(1,2,0))

# transpose (wave,y,x) => (y,x,wave)
spec_flux_trans = np.transpose(spec_flux, axes=(1,2,0))
spec_flux_err_trans = np.transpose(spec_flux_err, axes=(1,2,0)) 

# select pixels
rows, cols = np.where(spec_gal_region==1)
npixs = len(rows)

# data of pre-calculated model rest-frame spectra
if config_data['models_spec']=='none':
	#models_spec = dir_mod+"spec_dpl_c20_nduste_nagn_50k.hdf5"
	models_spec = dir_mod+"s_dpl_cf20_nd_na_50k.hdf5"
else:
	models_spec = config_data['models_spec']

f = h5py.File(models_spec, 'r')
# number of model SEDs
nmodels = int(f['mod'].attrs['nmodels']/size)*size

# cut model spectrum to match range given by the IFS spectra
rest_wave = f['mod/spec/wave'][:]
redsh_wave = (1.0+gal_z)*rest_wave
idx_mod_wave = np.where((redsh_wave>min_spec_wave-10) & (redsh_wave<max_spec_wave+10))

# get wavelength free of emission lines
del_wave_nebem = float(config_data['del_wave_nebem'])
spec_wave_clean,wave_mask = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)

# arrays for outputs
rescaled_spec_flux = np.zeros((dim_y,dim_x,nwaves))
rescaled_spec_flux_err = np.zeros((dim_y,dim_x,nwaves))

# allocate memory for correction factor
map_spec_corr_factor = np.zeros((dim_y,dim_x,nwaves))

# allocate memory for output best-fit model spectra
map_bfit_mod_spec_wave = spec_wave_clean
map_bfit_mod_spec_flux = np.zeros((dim_y,dim_x,len(spec_wave_clean)))

# polynomial order
poly_order = 3

for pp in range(0,npixs):
	# obs SED
	obs_flux = photo_flux_trans[rows[pp]][cols[pp]]
	obs_flux_err = photo_flux_err_trans[rows[pp]][cols[pp]]

	# exclude negative fluxes
	idx = np.where(obs_flux>0.0)

	# set up calculation
	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_id_temp = np.zeros(numDataPerRank)
	mod_norm_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# model ID:
		mod_id_temp[int(count)] = int(ii)

		# model SED
		wave0 = f['mod/spec/wave'][:]
		str_temp = 'mod/spec/f%d' % int(ii)
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,
													z=gal_z,wave=wave0,spec=extnc_spec)

		# filtering
		mod_flux = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_flux[idx[0]],obs_flux_err[idx[0]],mod_flux[idx[0]])
		mod_flux = norm*mod_flux

		chi2 = np.sum(np.square(mod_flux[idx[0]]-obs_flux[idx[0]])/obs_flux_err[idx[0]]/obs_flux_err[idx[0]])
		mod_chi2_temp[int(count)] = chi2
		mod_norm_temp[int(count)] = norm

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d: pixel %d of %d -> model %d of %d (%d%%)' % (rank,(pp+1),npixs,count,len(recvbuf_idx),
																					count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_id = np.zeros(numDataPerRank*size)
	mod_norm = np.zeros(numDataPerRank*size)

	# gather data
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_id_temp, mod_id, root=0)
	comm.Gather(mod_norm_temp, mod_norm, root=0)
	
	if rank == 0:
		# get best-fit
		idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

		# get best-fit spectrum
		wave0 = f['mod/spec/wave'][:]
		str_temp = 'mod/spec/f%d' % int(mod_id[idx0])
		extnc_spec = f[str_temp][:]
		bfit_spec_wave,bfit_spec_flux = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,
													z=gal_z,wave=wave0,spec=extnc_spec)

		# cut model spectrum to match range given by the IFS spectra
		bfit_spec_wave = bfit_spec_wave[idx_mod_wave[0]]
		bfit_spec_flux = bfit_spec_flux[idx_mod_wave[0]]*mod_norm[int(mod_id[idx0])]

		# smoothing model spectrum to meet resolution of IFS
		conv_bfit_spec_wave,conv_bfit_spec_flux = spec_smoothing(bfit_spec_wave,bfit_spec_flux,spec_sigma)

		# match scaling/normalization of IFS spectra to the best-fit model spectrum
		in_spec_flux = spec_flux_trans[rows[pp]][cols[pp]]
		final_wave,final_flux,factor,ref_spec_flux_clean = match_spectra_poly_legendre_fit(spec_wave,in_spec_flux,
																		conv_bfit_spec_wave,conv_bfit_spec_flux,
																		spec_wave_clean,gal_z,del_wave_nebem,poly_order)

		# get output re-scaled spectrum
		rescaled_spec_flux[rows[pp]][cols[pp]] = final_flux
		rescaled_spec_flux_err[rows[pp]][cols[pp]] = spec_flux_err_trans[rows[pp]][cols[pp]]*factor

		# get output best-fit model
		map_bfit_mod_spec_flux[rows[pp]][cols[pp]] = ref_spec_flux_clean

		# get correction factor
		map_spec_corr_factor[rows[pp]][cols[pp]] = factor

if rank == 0:
	# transpose (y,x,wave) => (wave,y,x) and re-normalize 
	map_rescaled_spec_flux = np.transpose(rescaled_spec_flux, axes=(2,0,1))
	map_rescaled_spec_flux_err = np.transpose(rescaled_spec_flux_err, axes=(2,0,1))
	map_bfit_mod_spec_flux_trans = np.transpose(map_bfit_mod_spec_flux, axes=(2,0,1))
	map_spec_corr_factor_trans = np.transpose(map_spec_corr_factor, axes=(2,0,1))


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
	hdul.append(fits.ImageHDU(map_spec_corr_factor_trans, name='corr_factor'))
	# write to fits file
	hdul.writeto(name_out_fits, overwrite=True)


f.close()

