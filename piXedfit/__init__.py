try:
	from ._version import __version__, __githash__
except(ImportError):
	pass

from . import piXedfit_analysis
from . import piXedfit_bin
from . import piXedfit_fitting
from . import piXedfit_images
from . import piXedfit_model
from . import piXedfit_spectrophotometric
from . import utils

