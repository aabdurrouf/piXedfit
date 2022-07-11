import numpy as np 
import sys, os
from astropy.io import fits
from astropy.wcs import WCS
from mpi4py import MPI 
from astropy.convolution import convolve_fft, Gaussian1DKernel
from reproject import reproject_exact
from photutils.psf.matching import resize_psf

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

from piXedfit.piXedfit_images import k_lmbd_Fitz1986_LMC


## USAGE: mpirun -np [npros] python ./sp_spat_clf (1)name_config

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# configuration file
global config_data
config_file = str(sys.argv[1])
dir_file = PIXEDFIT_HOME+'/data/temp/'
data = np.genfromtxt(dir_file+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]

# get information
photo_fluxmap = config_data['photo_fluxmap']
manga_file = config_data['manga_file']
spec_smoothing = int(config_data['spec_smoothing'])
kernel_sigma = float(config_data['kernel_sigma'])
name_out_fits = config_data['name_out_fits']

# get maps of photometric fluxes
hdu = fits.open(photo_fluxmap)
header_photo_fluxmap = hdu[0].header
photo_gal_region = hdu['GALAXY_REGION'].data
photo_flux_map = hdu['FLUX'].data 									# structure: (band,y,x)
photo_fluxerr_map = hdu['FLUX_ERR'].data 							# structure: (band,y,x)
unit_photo_fluxmap = float(header_photo_fluxmap['unit'])
data_stamp_image = hdu['stamp_image'].data 
header_stamp_image = hdu['stamp_image'].header
dimy_stamp_image = data_stamp_image.shape[0]
dimx_stamp_image = data_stamp_image.shape[1]
hdu.close()

# pixel size in photometric data
pixsize_image = float(header_photo_fluxmap['pixsize'])
filter_ref_psfmatch = header_photo_fluxmap['fpsfmtch']

## open MaNGA IFS data
cube = fits.open(manga_file)
map_flux0 = cube['FLUX'].data 										# structure: (wave,y,x)
map_var = 1.0/cube['IVAR'].data 									# variance
wave = cube['WAVE'].data
nwaves = len(wave)
# reconstructed r-band image
rimg = cube['RIMG'].data
header_manga2D = cube['RIMG'].header
# E(B-V) of foreground Galactic dust attenuation
Gal_EBV = float(cube[0].header['EBVGAL'])
cube.close()
unit_ifu = 1.0e-17 													# in erg/s/cm^2/Ang.
# dimension 
dim_y = rimg.shape[0]
dim_x = rimg.shape[1]

pixsize_manga=0.5

# make mask region for PSF matching process
mask_region = np.zeros((dim_y,dim_x))
rows, cols = np.where(rimg<=0.0)
mask_region[rows,cols] = 1

if spec_smoothing==1:
	#=> spectral smooting
	wave_lin = np.linspace(int(min(wave)),int(max(wave)),int(max(wave))-int(min(wave))+1)
	spec_kernel = Gaussian1DKernel(stddev=kernel_sigma)

	# transpose (wave,y,x) => (y,x,wave)
	map_flux_trans = np.transpose(map_flux0, axes=(1, 2, 0))

	map_flux1 = np.zeros((dim_y,dim_x,nwaves))
	rows, cols = np.where(rimg>0.0)
	for ii in range(0,len(rows)):
		yy, xx = rows[ii], cols[ii]
		spec_flux_wavelin = np.interp(wave_lin, wave, map_flux_trans[yy][xx])
		conv_flux = convolve_fft(spec_flux_wavelin, spec_kernel)
		idx_sel = np.where((conv_flux>0) & (np.isnan(conv_flux)==False) & (np.isinf(conv_flux)==False))
		map_flux1[yy][xx] = np.interp(wave, wave_lin[idx_sel[0]], conv_flux[idx_sel[0]])

	# transpose (y,x,wave) => (wave,y,x):
	map_flux = np.transpose(map_flux1, axes=(2, 0, 1))
else:
	map_flux = map_flux0


# get kernel for PSF matching
# All kernels were brought to 0.25"/pixel sampling
dir_file = PIXEDFIT_HOME+'/data/kernels/'
kernel_name = 'kernel_manga_to_%s.fits.gz' % filter_ref_psfmatch
hdu = fits.open(dir_file+kernel_name)
kernel_data = hdu[0].data/np.sum(hdu[0].data)
hdu.close()
# resize/resampling kernel image to match the sampling of the image
kernel_resize = resize_psf(kernel_data, 0.25, pixsize_manga, order=3)
kernel_resize = kernel_resize/np.sum(kernel_resize)

# get the number of spectral pixels that will be left over after the initial calculation
nwaves_calc = int(nwaves/size)*size

numDataPerRank = int(nwaves_calc/size)
idx_mpi = np.linspace(0,nwaves_calc-1,nwaves_calc)
recvbuf_idx = np.empty(numDataPerRank, dtype='d')
comm.Scatter(idx_mpi, recvbuf_idx, root=0)

map_ifu_flux_temp0 = np.zeros((dimy_stamp_image,dimx_stamp_image,numDataPerRank))
map_ifu_flux_err_temp0 = np.zeros((dimy_stamp_image,dimx_stamp_image,numDataPerRank))

count = 0 
for ii in recvbuf_idx:
	# get imaging layer from IFS 3D data cube
	layer_ifu_flux = map_flux[int(ii)]
	layer_ifu_var = map_var[int(ii)]

	# PSF matching
	psfmatch_layer_ifu_flux = convolve_fft(layer_ifu_flux, kernel_resize, allow_huge=True, mask=mask_region)
	psfmatch_layer_ifu_var = convolve_fft(layer_ifu_var, kernel_resize, allow_huge=True, mask=mask_region)

	# align to stamp image:
	data_image = psfmatch_layer_ifu_flux/pixsize_manga/pixsize_manga				# surface brightness
	align_psfmatch_layer_ifu_flux, footprint = reproject_exact((data_image,header_manga2D), header_stamp_image)
	align_psfmatch_layer_ifu_flux = align_psfmatch_layer_ifu_flux*pixsize_image*pixsize_image

	data_image = psfmatch_layer_ifu_var/pixsize_manga/pixsize_manga
	align_psfmatch_layer_ifu_var, footprint = reproject_exact((data_image,header_manga2D), header_stamp_image)
	align_psfmatch_layer_ifu_var = align_psfmatch_layer_ifu_var*pixsize_image*pixsize_image
		
	map_ifu_flux_temp0[:,:,int(count)] = align_psfmatch_layer_ifu_flux 							# in unit_ifu
	map_ifu_flux_err_temp0[:,:,int(count)] = np.sqrt(align_psfmatch_layer_ifu_var)				# in unit_ifu

	count = count + 1

	sys.stdout.write('\r')
	sys.stdout.write('rank: %d  Calculation process: %d from %d --> %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
	sys.stdout.flush()
#sys.stdout.write('\n')

map_ifu_flux_temp1 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves_calc))
map_ifu_flux_err_temp1 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves_calc))

for yy in range(0,dimy_stamp_image):
	for xx in range(0,dimx_stamp_image):
		comm.Gather(map_ifu_flux_temp0[yy][xx], map_ifu_flux_temp1[yy][xx], root=0)
		comm.Gather(map_ifu_flux_err_temp0[yy][xx], map_ifu_flux_err_temp1[yy][xx], root=0)

if rank == 0:
	# calculate the rest of the wavelength points
	#========================================================#
	map_ifu_flux_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))
	map_ifu_flux_err_temp = np.zeros((nwaves,dimy_stamp_image,dimx_stamp_image))

	for ww in range(0,nwaves):
		if ww<nwaves_calc:
			map_ifu_flux_temp[ww] = map_ifu_flux_temp1[:,:,ww]
			map_ifu_flux_err_temp[ww] = map_ifu_flux_err_temp1[:,:,ww]
		else:
			# get imaging layer from IFS 3D data cube
			layer_ifu_flux = map_flux[ww]
			layer_ifu_var = map_var[ww]

			# PSF matching
			psfmatch_layer_ifu_flux = convolve_fft(layer_ifu_flux, kernel_resize, allow_huge=True, mask=mask_region)
			psfmatch_layer_ifu_var = convolve_fft(layer_ifu_var, kernel_resize, allow_huge=True, mask=mask_region)

			# align to stamp image:
			data_image = psfmatch_layer_ifu_flux/pixsize_manga/pixsize_manga 				# surface brightness
			align_psfmatch_layer_ifu_flux, footprint = reproject_exact((data_image,header_manga2D), header_stamp_image)
			align_psfmatch_layer_ifu_flux = align_psfmatch_layer_ifu_flux*pixsize_image*pixsize_image

			data_image = psfmatch_layer_ifu_var/pixsize_manga/pixsize_manga
			align_psfmatch_layer_ifu_var, footprint = reproject_exact((data_image,header_manga2D), header_stamp_image)
			align_psfmatch_layer_ifu_var = align_psfmatch_layer_ifu_var*pixsize_image*pixsize_image
				
			map_ifu_flux_temp[ww] = align_psfmatch_layer_ifu_flux 							# in unit_ifu
			map_ifu_flux_err_temp[ww] = np.sqrt(align_psfmatch_layer_ifu_var) 				# in unit_ifu
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

	# construct spectrophotometric SEDs within the defined region:
	spec_gal_region = np.zeros((dimy_stamp_image,dimx_stamp_image))
	map_specphoto_spec_flux0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves,))
	map_specphoto_spec_flux_err0 = np.zeros((dimy_stamp_image,dimx_stamp_image,nwaves))
	
	rows, cols = np.where((align_map_mask==0) & (photo_gal_region==1))
	spec_gal_region[rows,cols] = 1

	# correct for foreground dust extinction
	corr_factor = np.power(10.0,0.4*k_lmbd_Fitz1986_LMC(wave)*Gal_EBV)
	corr_spec = map_ifu_flux_temp_trans[rows,cols]*corr_factor
	corr_spec_err = map_ifu_flux_err_temp_trans[rows,cols]*corr_factor

	# store in temporary arrays
	map_specphoto_spec_flux0[rows,cols] = corr_spec
	map_specphoto_spec_flux_err0[rows,cols] = corr_spec_err

	# transpose from (y,x,wave) => (wave,y,x)
	map_specphoto_spec_flux = np.transpose(map_specphoto_spec_flux0, axes=(2, 0, 1))
	map_specphoto_spec_flux_err = np.transpose(map_specphoto_spec_flux_err0, axes=(2, 0, 1))

	# photo SED is given to the full map as it was with photometry only
	map_specphoto_phot_flux = photo_flux_map*unit_photo_fluxmap/unit_ifu 					### in unit_ifu
	map_specphoto_phot_flux_err = photo_fluxerr_map*unit_photo_fluxmap/unit_ifu 			### in unit_ifu
	#========================================================#

	# Store into fits file 
	hdul = fits.HDUList()
	hdr = fits.Header()
	hdr['nfilters'] = header_photo_fluxmap['nfilters']
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
	hdr['ifssurve'] = 'manga'
	
	for bb in range(0,int(header_photo_fluxmap['nfilters'])):
		str_temp = 'fil%d' % int(bb)
		hdr[str_temp] = header_photo_fluxmap[str_temp]

	hdul.append(fits.ImageHDU(data=map_specphoto_phot_flux, header=hdr, name='photo_flux'))
	hdul.append(fits.ImageHDU(map_specphoto_phot_flux_err, name='photo_fluxerr'))
	hdul.append(fits.ImageHDU(wave, name='wave'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_flux, name='spec_flux'))
	hdul.append(fits.ImageHDU(map_specphoto_spec_flux_err, name='spec_fluxerr'))
	hdul.append(fits.ImageHDU(spec_gal_region, name='spec_region'))
	hdul.append(fits.ImageHDU(photo_gal_region, name='photo_region'))
	hdul.append(fits.ImageHDU(data=data_stamp_image, header=header_stamp_image, name='stamp_image'))
	hdul.writeto(name_out_fits, overwrite=True)

	sys.stdout.write('\n')

