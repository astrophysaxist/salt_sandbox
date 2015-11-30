#!/usr/bin/env python

import os, sys, numpy, scipy, pyfits
import logging

import pysalt
import traceline
import prep_science


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")

    hdulist = pyfits.open(sys.argv[1])
    
    start_x = int(sys.argv[2])-1
    start_y = int(sys.argv[3])-1
    start = start_x,start_y

    data = hdulist['SKYSUB.OPT'].data
    pyfits.PrimaryHDU(data=data).writeto("tracespec.fits", clobber=True)

    # transpose - we can only trace up/down, not left/right
    #data = data.T

    # combined = traceline.trace_arc(
    #     data=data,
    #     start=start,
    #     direction=-1,
    #     )
    # print combined

    w = 30

    _y = start_y

    iy,ix = numpy.indices(data.shape)
    positions = numpy.empty((data.shape[1]))
    positions[:] = numpy.NaN

    # forward and backwards

    for x in range(0, data.shape[1]):
#    for x in range(1040, 1050): #0, data.shape[1]):

        # pick out peak position by computing center of moments
        slit = data[_y-w:_y+w, x]
        pos_y = iy[_y-w:_y+w, x]

        valid = numpy.isfinite(slit)
        #print x, valid

        if (valid.any()):
            weighted_y = numpy.sum((slit*pos_y)[valid]) / numpy.sum(slit[valid])
        else:
            weighted_y = numpy.NaN

        #print slit.shape
        
        print x, weighted_y

        numpy.savetxt("slit_%04d" % (x),
                      numpy.append(pos_y.reshape((-1,1)),
                                   slit.reshape((-1,1)), axis=1))

        positions[x] = weighted_y


    pos_x = numpy.arange(positions.shape[0])

    #
    # Now filter the profile to create a smoother trace profile
    #
    smoothtrace = prep_science.compute_smoothed_profile(
        data_x=pos_x,
        data_y=positions,
#        n_max_neighbors=100,
        avg_sample_width=100,
        n_sigma=3,
        )

    numpy.savetxt("tracespec", positions)
    numpy.savetxt("tracespec.smooth", smoothtrace)

    pysalt.mp_logging.shutdown_logging(logger_setup)
