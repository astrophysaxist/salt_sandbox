#!/usr/bin/env python

import pyfits
import os, sys
import numpy
import scipy
import traceline
import prep_science
import pysalt.mp_logging

def trace_full_line(imgdata, x_start, y_start, window=5):

    weighted_pos = numpy.zeros((imgdata.shape[0],4))
    weighted_pos[:,:] = numpy.NaN

    x_guess_list = []
    x_guess = x_start
    for y in range(y_start, imgdata.shape[0]):
       
        # compute center of line in this row
        x_pos = numpy.arange(x_guess-window, x_guess+window+1)
        flux = imgdata[y, x_guess-window:x_guess+window+1]
        i_flux = numpy.sum(flux)
        _wp = numpy.sum(x_pos*flux) / i_flux


        x_guess_list.append(_wp)
        
        x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
        weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
        print y,_wp,x_guess

    x_guess = x_start
    x_guess_list = []
    for y in range(y_start, 0, -1):
       
        # compute center of line in this row
        x_pos = numpy.arange(x_guess-window, x_guess+window+1)
        flux = imgdata[y, x_guess-window:x_guess+window+1]
        i_flux = numpy.sum(flux)
        _wp = numpy.sum(x_pos*flux) / i_flux


        x_guess_list.append(_wp)
        
        x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
        weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
        print y,_wp,x_guess

    return weighted_pos

if __name__ == "__main__":

    logger = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]
    hdulist = pyfits.open(fn)

    # imgdata = hdulist['SCI.RAW'].data
    imgdata = hdulist['SCI.NOCRJ'].data

    skylines, continuum = prep_science.filter_isolate_skylines(data=imgdata)
    pyfits.PrimaryHDU(data=skylines).writeto("skytrace_sky.fits", clobber=True)
    pyfits.PrimaryHDU(data=continuum).writeto("skytrace_continuum.fits", clobber=True)

    # combined = traceline.trace_arc(
    #     data=skylines,
    #     start=(601,526),
    #     #start=(526,602),
    #     direction=+1,
    #     )

    # weighted_pos = numpy.zeros((imgdata.shape[0],4))
    # weighted_pos[:,:] = numpy.NaN

    # x_guess = 601
    # window = 5
    # x_guess_list = []

    # for y in range(526, imgdata.shape[0]):
       
    #     # compute center of line in this row
    #     x_pos = numpy.arange(x_guess-window, x_guess+window+1)
    #     flux = imgdata[y, x_guess-window:x_guess+window+1]
    #     i_flux = numpy.sum(flux)
    #     _wp = numpy.sum(x_pos*flux) / i_flux


    #     x_guess_list.append(_wp)
        
    #     x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
    #     weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
    #     print y,_wp,x_guess

    # x_guess = 601
    # x_guess_list = []
    # for y in range(526, 0, -1):
       
    #     # compute center of line in this row
    #     x_pos = numpy.arange(x_guess-window, x_guess+window+1)
    #     flux = imgdata[y, x_guess-window:x_guess+window+1]
    #     i_flux = numpy.sum(flux)
    #     _wp = numpy.sum(x_pos*flux) / i_flux


    #     x_guess_list.append(_wp)
        
    #     x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
    #     weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
    #     print y,_wp,x_guess

    # numpy.savetxt("skytrace_arc.txt", weighted_pos)
    # #print combined
    # #numpy.savetxt("skytrace_arc.txt", combined)

    wp2 = trace_full_line(imgdata, x_start=601, y_start=526, window=5)
    numpy.savetxt("skytrace_arc2.txt", weighted_pos)

    pysalt.mp_logging.shutdown_logging(logger)
