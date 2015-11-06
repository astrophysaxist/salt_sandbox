#!/usr/bin/env python

import os, sys, pyfits
import numpy
from scipy.ndimage.filters import median_filter
import bottleneck
import scipy.interpolate
numpy.seterr(divide='ignore', invalid='ignore')

# Disable nasty and useless RankWarning when spline fitting
import warnings
warnings.simplefilter('ignore', numpy.RankWarning)
# also ignore some other annoying warning
warnings.simplefilter('ignore', RuntimeWarning)

import bottleneck


import scipy.spatial
import pysalt.mp_logging
import logging


import matplotlib.pyplot as pl




if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("SpecExtract")

    filename = sys.argv[1]

    from_line = int(sys.argv[2]) - 1
    to_line = int(sys.argv[3]) - 1

    hdulist = pyfits.open(filename)

    wl_data = hdulist['WAVELENGTH'].data
    obj_data = None
    try:
        obj_data = hdulist['SKYSUB.OPT'].data
        logger.info("Using optimized SCI data")
    except:
        obj_data = hdulist['SCI'].data
        logger.info("Using standard SCI data")


    var_data = None
    var_valid = False
    try:
        var_data = hdulist['VAR'].data
        var_valid = True
    except:
        pass

    #
    # Get wavelength scale
    #
    #spec_wl = numpy.average(
    spec_wl = wl_data[from_line:to_line+1, :]#, axis=1)
    spec_wl_1d = numpy.average(spec_wl, axis=0)
    print spec_wl.shape, spec_wl_1d.shape

    #
    # Also extract fluxes
    #
    spec_fluxes = obj_data[from_line:to_line+1, :]
    spec_fluxes_1d = numpy.sum(spec_fluxes, axis=0)
    print spec_fluxes_1d.shape

    #
    # Compute variance data (this is noise^2), so need to take sqrt to get real noise
    #
    var_1d = numpy.zeros((spec_wl_1d.shape[0]))
    if (var_valid):
        var_2d = var_data[from_line:to_line+1, :]
        var_1d = numpy.sqrt(numpy.sum(var_2d, axis=0))
        print "Found VAR extension"

    #
    # Now merge fluxes and wavelength scale
    #
    combined = numpy.zeros((spec_wl_1d.shape[0], 4))
    combined[:,0] = numpy.arange(combined.shape[0])+1
    combined[:,1] = spec_wl_1d[:]
    combined[:,2] = spec_fluxes_1d[:]
    combined[:,3] = var_1d[:]
    
# numpy.append(spec_wl_1d.reshape((-1,1)),
#                             spec_fluxes_1d.reshape((-1,1)),
#                             axis=1)
    print combined.shape

    numpy.savetxt(sys.argv[4], combined,
                  header="""\
Column  1: x-coordinate [pixels]
Column  2: wavelength [angstroems]
Column  3: integrated flux [ADU]
Column  4: variance of sum (=sqrt(sum_i(var_i))) [ADU]
--------------------------------------------------------\
""",
)

    pysalt.mp_logging.shutdown_logging(logger_setup)
