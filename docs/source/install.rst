Installation
============  

Dependencies
------------

* Python >= 3.7 
* `FSPS <https://github.com/cconroy20/fsps>`_ and `Python FSPS <https://dfm.io/python-fsps/current/>`_.
* `mpi4py <https://mpi4py.readthedocs.io/en/stable/index.html#>`_ for parallel processing.
* `emcee <https://emcee.readthedocs.io/en/stable/>`_ for SED fitting with MCMC method. 


Installing piXedfit
-------------------

* cd to a desired installation directory and clone **piXedfit**.

.. code-block:: shell

	cd <install_dir>
	git clone https://github.com/aabdurrouf/piXedfit.git
	cd piXedfit
	python -m pip install .

* Set an environmental variable called PIXEDFIT_HOME that point to **piXedfit** parent directory.

.. code-block:: shell

	export PIXEDFIT_HOME="$PWD"
 
* The issue with the above command is that we need to do it every time we open a new terminal. Alternatively, we can add this environmental variable to .bashrc (add export PIXEDFIT_HOME="path_to_piXedfit" to the last line in the .bashrc file). 

.. code-block:: shell

	vi ~/.bashrc

Add the above line and then do

.. code-block:: shell

	source  ~/.bashrc

* To add the environmental variable permanently, we can add the same line to .bash_profile or .profile. In case of .bash_profile:

.. code-block:: shell

	vi ~/.bash_profile

Add the line and then do

.. code-block:: shell

	source ~/.bash_profile






