Convolution kernels and PSFs
=============================

The point spread function (PSF) describes the two-dimensional distribution of light in the telescope focal plane for the astronomical point sources. 
To get reliable multiwavelength photometric SED from a set of multiband images, especially in the analysis of spatially resolved SEDs of galaxies, 
it is important to homogenize the spatial resolution (i.e., PSF size) of those images before extracting SEDs from them. This process of homogenizing 
the PSF size of multiband images is called PSF matching. The final/target spatial resolution to be achieved is the one that is the lowest (i.e., worst) among the multiband images being analyzed. Commonly, PSF matching process of multiband images is done by convolving the images that have higher spatil resolution 
(i.e., smaller PSF size than the target PSF) with a set of pre-calculated convolution kerenels. The convolution kernel for matching a pair of two PSFs 
is derived from the ratio of Fourier transforms (e.g., `Gordon et al. 2008 <https://ui.adsabs.harvard.edu/abs/2008ApJ...682..336G/abstract>`_; 
`Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_). 

For PSF matching process, :mod:`piXedfit_images` module uses convolution kernels from `Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_ that are publicly available at this `website <https://www.astro.princeton.edu/~ganiano/Kernels.html>`_. The available kernels cover various ground-based and spaced-based telescopes, including `GALEX <http://www.galex.caltech.edu/>`_, `Spitzer <http://www.spitzer.caltech.edu/>`_, `WISE <https://wise2.ipac.caltech.edu/docs/release/allsky/>`_, 
and `Herschel <https://sci.esa.int/web/herschel>`_. Besides that, `Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_ 
also provide convolution kernels for some analytical PSFs that includes Gaussian, sum of Gaussians, and Moffat. Those analytical PSFs are expected to be representative of the net (i.e., effective) PSFs of ground-based telescopes. 

To associate the PSFs of SDSS and 2MASS (which are not explicitely covered in `Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_) with the analytical PSFs of `Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_, we have constructed empirical PSFs of the 5 SDSS bands and 3 2MASS bands and compare the constructed empirical PSFs with the analytical PSFs from `Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_. We find that the empirical PSFs of SDSS :math:`u`, :math:`g`, and :math:`r` bands are best represented by double Gaussian with FWHM of 1.5'', while the other bands (:math:`i` and :math:`z`) are best represented by double Gaussian with FWHM of 1.0''. For 2MASS, all the 3 bands (:math:`J`, :math:`H`, and :math:`\text{K}_{\text{S}}`) are best represented by Gaussian with FWHM of 3.5''. Construction of these empirical PSFs is presented in Appendix A of **Abdurro'uf et al. (2020, submitted)**. The empirical PSFs are available at this `Github page <https://github.com/aabdurrouf/empPSFs_GALEXSDSS2MASS>`_. For consistency, the :mod:`piXedfit_images` use those analytical PSFs to represent the PSFs of SDSS and 2MASS and use the convolution kernels associated with them whenever needed. 

By default, when external kernels are not provided by the user, :mod:`piXedfit_images` module will use the kernels from `Aniano et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_. For flexibility, the users can also input their own kernels to :mod:`piXedfit_images`.

Figures below show a demonstration of the performance of some convolution kernels used in the :mod:`piXedfit_images` module. In the top figure, the performance of the convolution kernels used to achieve the spatial resolution of WISE/:math:`W2` is demonstrated. Different panels show different initial PSFs. In the first row from the left to right, we show the convolution results from initial PSFs of GALEX/FUV, GALEX/NUV, and SDSS/:math:`u`, respectively. The second row, from left to right, we show the result for SDSS/:math:`z`, 2MASS/:math:`J`, and 2MASS/:math:`W1`, respectively. In the bottom figure, the performance of the convolution kernels used to achieve the spatial resolution of Herschel/SPIRE350 is demonstrated. In the first row from the left to right, we show the convolution results from initial PSFs of GALEX/FUV, SDSS/:math:`u`, and 2MASS/:math:`J`, respectively. The second row, from left to right, we show the result for WISE/:math:`W1`, Spitzer/IRAC :math:`8.0\mu \text{m}`, and Spitzer/MIPS :math:`24\mu \text{m}`, respectively. The figures shows that the performance of the convolution kernels is very good, evidenced from the good matching between the shapes of the convolved PSFs and the target PSF.

.. image:: perform_kernels.png
  :width: 900
  
.. image:: perform_kernels1.png
  :width: 900

For the characteristic PSFs of the imaging data that can be analyzed with the current version of **piXedfit**, please see the description in this `page <https://pixedfit.readthedocs.io/en/latest/list_imaging_data.html>`_.
