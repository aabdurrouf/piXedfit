piXedfit
========
**piXedfit**, pixelized spectral energy distribution (SED) fitting, is a Python package that provides a *self-contained* set of tools for analyzing spatially resolved properties of galaxies using 
imaging data or a combination of imaging data and integral field spectroscopy (IFS) data. **piXedfit** has six modules which can 
handle all tasks in the analysis of spatially resolved SEDs of a galaxy, including images processing, a spatial-matching between reduced 
broad-band imaging data with IFS data, pixel binning, SED fitting, and producing visualization plots for the SED fitting results. 
**piXedfit** is a versatile tool that has been equipped with the multiprocessing module (namely message passing interface or MPI) for 
efficient analysis of the datasets of a large number of galaxies. Detailed description on **piXedfit** and its performance is presented in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.  

While this website is still under construction, people interested in knowing how **piXedfit** works can see some demonstrations in folder *examples* on the `GitHub page <https://github.com/aabdurrouf/piXedfit>`_ or some animations here: `images processing <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_img_pros.rst>`_, `pixel binning <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_pixel_binning.rst>`_ and `SED fitting <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_sed_fitting.rst>`_.   

.. image:: 3Dcube_specphoto.png
.. image:: demo_pixedfit_ngc309_new_edit.svg
.. image:: plot_maps_props_new.svg
   :width: 600

Features
--------
**piXedfit** has 6 modules that can work independent with each other. For instance, it is possible to use the SED fitting module for fitting any observed SED (either integrated 
of spaially resolved SED) without the need of using the images processing and pixel binning modules. The 6 modules and their capabilities are the following:

*  **piXedfit_images: image processing**
   
   This module can be used for spatial-matching (in resolution and spatial sampling) of multiband imaging data ranging from far-ultraviolet (FUV) to far-infrared (FIR) 
   (obtained from both the ground-based and spaced-based telescopes) and extract pixel-wise photometric SEDs within the galaxy's region of interest.

*  **piXedfit_spectrophotometric: spatial-matching of imaging data and the IFS data**

   This module can be used for spatial-matching (in spatial resolution and sampling) of a multiband imaging data 
   (that have been processed by the :mod:`piXedfit_images`) with an IFS data cube and extract pixel-wise 
   spectrophotometric SEDs within the galaxy's region of interest. For the current version of **piXedfit**, only the IFS data from 
   the `CALIFA <https://califa.caha.es/>`_ and `MaNGA <https://www.sdss.org/surveys/manga/>`_ can be analyzed by this module.   

*  **piXedfit_bin: pixel binning**

   This module is capable of performing pixel binning, which is a process of combining neighboring pixels to achieve certain set of S/N thresholds.
   The pixel binning scheme takes into account of the similarity of SED shape among the pixels. This way 
   important spatial information from pixel scale can still be preserved, while increasing the S/N of the spatially resolved SEDs. 
   The S/N threshold can be set to every band in the multiband imaging data, not only to a particular band.   

*  **piXedfit_model: generating model SEDs**

   This module can generate model SEDs of galaxies given some parameters. The SED modeling uses the `FSPS <https://github.com/cconroy20/fsps>`_ SPS model 
   with the `Python-FSPS <http://dfm.io/python-fsps/current/>`_ as the interface to the Python environment. The SED modeling incorporates four main components in the galaxy's SED: stellar emission, nebular emission, dust emission, and the AGN dusty torus emission.      

*  **piXedfit_fitting: performing SED fitting**

   This module is capable of performing SED fitting to input SED, including both spatially resolved SED and integrated SED of galaxies. The input can be 
   in the form of photometric SED or spetrophotometric SED (i.e., combination of photometry and spectroscopy).

*  **piXedfit_analysis: producing visualization plots for the SED fiting results**

   This module can make three plots for visualization of the fitting results: corner plot (i.e., plot showing 1D and 2D joint posteriors of the parameters space), SED plot, and SFH plot.


.. toctree::
   :maxdepth: 2
   :caption: User guide

   install
   manage_filters
   list_imaging_data
   list_kernels_psf
   ingredients_model
   gen_model_SEDs
   image_pros
   image_ifs_match
   pixel_binning
   fit_ing_SEDs
   fit_res_SEDs
   plot_fitres
   get_maps_prop


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials


.. toctree::
   :maxdepth: 2
   :caption: Demonstrations
   
   demos_pixel_binning
   demos_sed_fitting


.. toctree::
   :maxdepth: 2
   :caption: API reference 

   piXedfit_images
   piXedfit_spectrophotometric
   piXedfit_bin
   piXedfit_model
   piXedfit_fitting
   piXedfit_analysis
   utils



Reference
---------

A list of a few projects **piXedfit** is benefited from:

*  `Astropy <https://www.astropy.org/>`_
*  `Photutils <https://photutils.readthedocs.io/en/stable/>`_
*  `Aniano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_ who provides convolution `kernels <https://www.astro.princeton.edu/~ganiano/Kernels.html>`_ for the PSF matching
*  `FSPS <https://github.com/cconroy20/fsps>`_ and `Python-FSPS <http://dfm.io/python-fsps/current/>`_ stellar population synthesis model
*  `emcee <https://emcee.readthedocs.io/en/stable/>`_ package for the Markov Chain Monte Carlo (MCMC) Ensemble sampler
