#!/usr/bin/env python

import os, sys, numpy, scipy, pyfits

import scipy.ndimage

import pysalt

import logging
import bottleneck


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")

    hdulist = pyfits.open(sys.argv[1])
    
    data = hdulist['SKYSUB.OPT'].data.copy()
    wl = hdulist['WAVELENGTH'].data
    sky = hdulist['SKYSUB.IMG'].data

    y = data.shape[0]/2
    #sky1d = sky[y:y+1,:] #.reshape((-1,1))
    sky1d = sky[y,:] #.reshape((-1,1))
    print sky1d.shape

    wl_1d = wl[y,:]

    mf = scipy.ndimage.filters.median_filter(
        input=sky1d, 
        size=(75), 
        footprint=None, 
        output=None, 
        mode='reflect', 
        cval=0.0, 
        origin=0)
    
    print mf.shape

    numpy.savetxt("sky", sky1d)
    numpy.savetxt("sky2", mf)

    # pick the intensity of the lowest 10%
    max_intensity = scipy.stats.scoreatpercentile(sky1d, [10,20])
    print max_intensity

    #
    # Find emission lines
    #
    line_strength = sky1d - mf
    no_line = numpy.ones(line_strength.shape, dtype=numpy.bool)
    for i in range(3):
        med = numpy.median(line_strength[no_line])
        std = numpy.std(line_strength[no_line])

        no_line = (line_strength < med+2*std) & \
                  (line_strength > med-2*std)
        print med, std, numpy.sum(no_line)

    #
    # Select regions that do not a sky-line within N pixels
    #
    print line_strength.shape
    N = 15
    buffered = numpy.zeros(line_strength.shape[0]+2*N)
    buffered[N:-N][~no_line] = 1.0
    numpy.savetxt("lines", buffered)

    W = 15
    line_blocker = numpy.array(
        [numpy.sum(buffered[x+N-W:x+N+W]) for x in range(line_strength.shape[0])]
    )
    numpy.savetxt("lineblocker", line_blocker)

    line_contaminated = line_blocker >= 1
    line_strength[line_contaminated] = numpy.NaN
    numpy.savetxt("contsky", line_strength)
    
    #
    # Now we have a very good estimate where skylines contaminate the spectrum
    # We can now isolate regions that are clean to find sources
    #

    #
    # Look for large chunks of spectrum without skylines
    # 
    spec_blocks = []
    blockstart = 0
    in_block = False
    for x in range(1, line_contaminated.shape[0]):
        if (not line_contaminated[x]):
            # this is a good part of spectrum
            if (in_block):
                # we already started this block
                continue
            else:
                in_block = True
                blockstart = x
        else:
            # this is a region close to a skyline
            if (in_block):
                # so far we were in a good strech which is now over
                spec_blocks.append([blockstart, x-1])
                in_block = False
            else:
                continue

    spec_blocks = numpy.array(spec_blocks)
    print spec_blocks

    #
    # Now pick a block close to the center of the chip where the spectral
    # trace shows little curvature
    #
    good_blocks = (spec_blocks[:,0] > 0.35*sky1d.shape[0]) & \
                  (spec_blocks[:,1] > 0.65*sky1d.shape[0])
    central_blocks = spec_blocks[good_blocks]
    print central_blocks

    # Out of these, find the largest one
    block_size = central_blocks[:,1] - central_blocks[:,0]
    print block_size

    largest_block = numpy.argmax(block_size)

    use_block = central_blocks[largest_block]
    print "Using spectral block for source finding:", use_block
        
    wl_min = wl_1d[use_block[0]]
    wl_max = wl_1d[use_block[1]]
    print "wavelength range: %f -- %f" % (wl_min, wl_max)

    out_of_range = (wl < wl_min) | (wl > wl_max)
    data[out_of_range] = numpy.NaN

    pyfits.PrimaryHDU(data=data).writeto("xxx", clobber=True)

    #intensity_profile = bottleneck.nansum(data, axis=1)
    intensity_profile = numpy.nanmean(data, axis=1)
    print data.shape, intensity_profile.shape

    numpy.savetxt("prof", intensity_profile)

    #
    # Apply wide median filter to subtract continuum slope (if any)
    #

    pysalt.mp_logging.shutdown_logging(logger_setup)
