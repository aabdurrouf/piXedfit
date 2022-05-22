Installation
============

To use **piXedfit** you need to insall the following dependencies. Some dependencies are only needed for particular tasks. If you are not going to do the tasks, even if those optinal dependencies are not installed, **piXedfit** can still be used normally for doing other tasks.  

Dependencies
------------

*	**piXedfit** works with Python3
*	`NumPy <https://numpy.org/>`_, `SciPy <https://www.scipy.org/>`_, `astropy <https://docs.astropy.org/en/stable/>`_, `matplotlib <https://matplotlib.org/>`_
*	`FSPS <https://github.com/cconroy20/fsps>`_, `Python-FSPS <https://dfm.io/python-fsps/current/>`_, `mpi4py <https://mpi4py.readthedocs.io/en/stable/index.html#>`_, `h5py <https://docs.h5py.org/en/stable/index.html>`_ 
*	`photutils <https://photutils.readthedocs.io/en/stable/>`_ and `reproject <https://reproject.readthedocs.io/en/stable/>`_ if you want to use :mod:`piXedfit_images` OR :mod:`piXedfit_spectrophotometric` modules for doing images processing OR spatial matching (i.e., combining) between multiband imaging data and integral field spectroscopic (IFS) data
*	`SEP <https://sep.readthedocs.io/en/v1.0.x/index.html>`_ if you want to use sources extraction and segmentation features of SExtractor for defining galaxy's region of interest in the image processing. 
*	`emcee <https://emcee.readthedocs.io/en/stable/>`_ and `schwimmbad <https://github.com/adrn/schwimmbad>`_ if you want to use MCMC method for SED fitting with the :mod:`piXefit_fitting` module
*	`Astroquery <https://astroquery.readthedocs.io/en/latest/>`_.


How to use the current version of piXedfit
------------------------------------------

Currently **piXedfit** is still under active development (and proprietary). We haven't integrated **piXedfit** into `pip <https://pypi.org/project/pip/>`_ or `conda <https://docs.conda.io/en/latest/>`_ and shared it to public. Currently, to use **piXedfit**, you can do the following steps:

*	Download **piXedfit** and put it into your desired directory. Inside **piXedfit** directory, do `pwd` command and suppose what you get is PATH_TO_PIXEDFIT
*	Add the following text to `.bashrc` in your home directory (change PATH_TO_PIXEDFIT with what you get from `pwd` command above).  

	.. code::

		export PIXEDFIT_HOME="PATH_TO_PIXEDFIT" 

	However, often time, even after adding the above text to `.bashrc`, **piXedfit** environment still can't be tracked and causes we can't import **piXedfit** in our script. To solve this, we can type the following command every time we open a new terminal and execute a script that import **piXedfit** in it.      

	.. code::

		export PIXEDFIT_HOME="PATH_TO_PIXEDFIT" 

	This problem also present for the usage of FSPS package. To solve it, we can also do

	.. code::

		export SPS_HOME="$HOME/fsps"

	(if FSPS is installed in $HOME directory, otherwise change it with the path to FSPS directory).

*	In every script that import **piXedfit**, please put the following lines of text in the beginning of the script

	.. code-block:: python

		import os, sys

		global PIXEDFIT_HOME
		PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
		sys.path.insert(0, PIXEDFIT_HOME)  




