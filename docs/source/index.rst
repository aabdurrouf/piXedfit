piXedfit
========
`**piXedfit** <https://github.com/aabdurrouf/piXedfit>`_ is a Python package that provides a complete set of tools for analyzing spatially resolved properties of galaxies using 
imaging data or a combination of imaging data and the integral field spectroscopy (IFS) data. **piXedfit** has six modules which can 
handle all tasks in the analysis of spatially resolved SEDs of a galaxy, including images processing, a spatial-matching between reduced 
broad-band images with an IFS data cube, pixel binning, performing SED fitting, and making visualization plots for the SED fitting result. 
**piXedfit** is a versatile tool that has been equipped with the multiprocessing module (namely message passing interface or MPI) for 
efficient analysis of the datasets of a large number of galaxies. Detailed description on **piXedfit** and its performance is presented in **Abdurro'uf et al. (2020, submitted)**.  

While this website is still under construction, people interested in knowing how **piXedfit** works can see a folder examples on the `GitHub page <https://github.com/aabdurrouf/piXedfit>`_.   

.. image:: 3Dcube_specphoto.png
.. image:: demo_pixedfit_ngc309_new_edit.svg
.. image:: plot_maps_props_new.svg
   :width: 600

Features
--------
**piXedfit** has 6 modules that can work independent with each other such that a user interested of using a particular module in **piXedfit** 
doesn't need to use the other modules. For instance, it is possible to use the SED fitting module for fitting any observed SED (either integrated 
of spaially resolved SED) without the need of using the image processing and pixel binning modules. The 6 modules and their usabilities are the following:

*  **piXedfit_images: image processing**
   
   This module is capable of doing spatial-matching (in resolution and spatial sampling) of multiband images ranging from the FUV to FIR 
   (from ground-based and spaced-based telescopes) and extract pixel-wise photometric SEDs within the galaxy's region of interest.

*  **piXedfit_spectrophotometric: spatial-matching of imaging data and the IFS data**

   This module is capable of doing spatial-matching (in resolution and sampling) of a multiband imaging data 
   (that have been processed by the :mod:`piXedfit_images`) with an IFS data cube (containing the same galaxy) and extract pixel-wise 
   spectrophotometric SEDs within the galaxy's region of interest.For the current version of **piXedfit**, only the IFS data from 
   the `CALIFA <https://califa.caha.es/>`_ and `MaNGA <https://www.sdss.org/surveys/manga/>`_ can can be analyzed by the 
   :mod:`piXedfit_spectrophotometric` module.   

*  **piXedfit_bin: pixel binning**

   This module is capable of performing pixel binning, which is a process of combining neighboring pixels to achieve certain S/N thresholds.
   The pixel binning scheme takes into account the similarity of SED shape among the pixels that are going to be binned together. This way 
   important spatial information from the pixel scale can still be preserved, while increasing the S/N of the spatially resolved SEDs. 
   The S/N threshold can be set to every band, not only to a particular band.   

*  **piXedfit_model: generating model SEDs**

   This module can generate model SEDs of galaxies given some parameters. The SED modeling uses the `FSPS <https://github.com/cconroy20/fsps>`_ SPS model 
   with the `Python-FSPS <http://dfm.io/python-fsps/current/>`_ as the interface to the Python environment. The SED modeling incorporates the modeling 
   of light coming from stellar emission, nebular emission, dust emission, and the AGN dusty torus emission.      

*  **piXedfit_fitting: performing SED fitting**

   This module is capable of performing SED fitting for any kind of input SED, either spatially resolved SED or integrated SED. The input can be 
   in the form of photometric SED or spetrophotometric SED (i.e., combination of photometry and spectroscopy).

*  **piXedfit_analysis: making visualization plots for the SED fiting results**

   This module can make three plots for visualizing the fitting results: corner plot (i.e., plot showing 1D and joint 2D posteriors of the parameters space), SED plot, and SFH plot.


.. toctree::
   :maxdepth: 2
   :caption: User guide

   install
   list_imaging_data
   image_pros
   image_ifs_match
   pixel_binning
   ingredients_model
   gen_model_SEDs
   fit_res_SEDs
   fit_ing_SEDs
   plot_fitres
   get_maps_prop


.. toctree::
   :maxdepth: 2
   :caption: Demonstrations and tutorials

   demos 
   tutorials


.. toctree::
   :maxdepth: 2
   :caption: API reference 

   piXedfit_images
   piXedfit_spectrophotometric
   piXedfit_bin
   piXedfit_model
   piXedfit_fitting
   piXedfit_analysis



Reference
---------

A list of a few projects **piXedfit** is benefited from:

*  `FSPS <https://github.com/cconroy20/fsps>`_ and `Python-FSPS <http://dfm.io/python-fsps/current/>`_ stellar population synthesis model
*  `emcee <https://emcee.readthedocs.io/en/stable/>`_ package for the Affine Invariant Markov Chain Monte Carlo (MCMC) Ensemble sampler
*  `Astropy <https://www.astropy.org/>`_
*  `Photutils <https://photutils.readthedocs.io/en/stable/>`_
*  `Aniano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_ who provides convolution `kernels <https://www.astro.princeton.edu/~ganiano/Kernels.html>`_ for the PSF matching
*  `SExtractor <https://www.astromatic.net/software/sextractor>`_ (`Bertin & Arnouts 1996 <https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B/abstract>`_)
*  Abdurro'uf & Akiyama (`2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2806A/abstract>`_, `2018 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5083A/abstract>`_)
