Available convolution kernels and PSFs
======================================

The point spread function (PSF) describes the two-dimensional distribution of light in the telescope focal plane for the astronomical point sources. 
To get reliable multiwavelength photometric SED from a set of multiband images, especially in the analysis of spatially resolved SEDs of galaxies, 
it is important to homogenize the spatial resolution (i.e., PSF size) of those images before extracting SEDs from them. This process of homogenizing 
the PSF size of multiband images is called PSF matching. The final spatial resolution to be achieved is the one that is the worst among the multiband images 
being analyzed. Commonly, PSF matching process of multiband images is done by convolving the images that have higher spatil resolution 
(i.e., smaller PSF size than the target PSF) with a set of pre-calculated convolution kerenels. The convolution kernel for matching a pair of two PSFs 
is derived from the ratio of Fourier transforms (e.g., `Gordon et al. 2008 <https://ui.adsabs.harvard.edu/abs/2008ApJ...682..336G/abstract>`_; 
`Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_). 
