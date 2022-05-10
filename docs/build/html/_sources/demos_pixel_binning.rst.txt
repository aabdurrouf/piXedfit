Pixel binning
==============

The animation below shows demonstration of how our new pixel binning scheme works. The pixel binning is done using the :mod:`piXedfit_bin` module. Basically, the pixel binning starts from a brightest pixel (in a reference band, that is selected by the user) then surrounding pixels (within a certain dimeter) that have SEDs with similar shape as the SED of the brightest pixel are binned together. If the resulted S/N in each band is still lower than a given S/N threshold, then the bin's size is grown gradually with increment radius of 2 pixels until the S/N thresholds in all bands are achieved.

In the animation below, a pixel binning process of the M51 and M81 galaxies are demonstrated. We use panchromatic imaging data in 23 bands ranging from GALEX/FUV to Herschel/SPIRE350. Before the pixel binning, the multiband images are processed (i.e., spatially-matched in resolution and sampling) using the :mod:`piXedfit_images` module. In the left panel, SDSS/:math:`r` image is shown, which is the reference band in this pixel binning process. The middle panel shows the binning map that is being constructed. The right panel shows SEDs of pixels (in colors) that are belong to a bin and the total SED of a bin (in black color).         

.. figure:: animation_pixbin_M51.gif
   :width: 800
   
.. figure:: animation_pixbin_M81.gif
   :width: 800


If the animation above doesn't run, please see the animation `here <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_pixel_binning.rst>`_.
