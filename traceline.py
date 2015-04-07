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

from PySpectrograph.Models import RSSModel

import pysalt


import scipy.spatial
import pysalt.mp_logging
import logging
import time
import math

import wlcal
import pickle


def trace_arc(data,
              start,
              direction=-1, # -1: downwards, +1: upwards
              max_window_x=5, # how far do we allow the arc to move from one row to the row
              ):

    logger = logging.getLogger("TraceArc")

    # extract x/y from tuple
    start_x, start_y = start

    # Create a list of row numbers that we need to inspect
    if (direction > 0):
        # We are going upwards
        logger.info("Moving upwards")
        row_numbers = numpy.arange(start_y+direction, data.shape[1], direction)
    elif (direction < 0):
        # downwards
        logger.info("Moving downwards")
        row_numbers = numpy.arange(start_y+direction, direction, direction)
    elif (direction == 0):
        print "bad boy, very bad boy!!!"
        return 


    current_row_idx = start_y
    current_col_idx = start_x
    y_stepsize = numpy.fabs(direction)

    # print row_numbers
    x_offsets = numpy.arange(-max_window_x, max_window_x+1)
    # print x_offsets

    #
    # Remember where in the image our center positions are
    #
    arc_center = numpy.empty((data.shape[0]))
    arc_center[:] = numpy.NaN
    arc_center[start_y] = start_x
    for next_row_idx in row_numbers:
        logger.debug("Moving from row %4d to %4d" % (current_row_idx, next_row_idx))

        # extract a bunch of pixels in the next row around the position of the 
        # current peak
        next_row = data[current_col_idx-max_window_x:current_col_idx+max_window_x+1,
                        next_row_idx]
        
        # Now compute gradients
        next_row_gradients = next_row / y_stepsize

        # pick the pixel with the largest positive gradient. 
        # this means we either move to even brighter pixels (if grad. > 0) or 
        # at least stay close to the peak without wandering off (if grad <~ 0)
        max_gradient = numpy.argmax(next_row_gradients)
        #print current_row_idx, max_gradient, x_offsets[max_gradient], current_col_idx

        # shift the new center position for the next row by the amount we 
        # just determined
        next_col_idx = current_col_idx + x_offsets[max_gradient]
        
                        
        if (math.fabs(next_row_idx-start_y) > 5000):
            logger.info("Aborting search!")
            break

        #
        # Save all positions for the next iteration
        #
        arc_center[next_row_idx] = next_col_idx

        current_row_idx = next_row_idx
        current_col_idx = next_col_idx
        
    # For debugging: save the arc position
    combined = numpy.append(numpy.arange(data.shape[0]).reshape((-1,1)),
                            arc_center.reshape((-1,1)), axis=1)
    numpy.savetxt("arcshape_%d" % (direction), combined)

    return combined



def trace_single_line(fitsdata, wls, line_idx, ds9_region_file=None):

    logger = logging.getLogger("TraceSlgLine")

    arclines = wls_data['linelist_arc']
    primeline = arclines[line_idx,:]
    print primeline
    arcpos_x = primeline[1]

    #data_around_line = fitsdata[arcpos_x-100:arcpos_x+100,:]
    #pyfits.PrimaryHDU(data=data_around_line.T).writeto("arcregion.fits", clobber=True)

    if (not ds9_region_file == None):
        ds9_region = open("ds9_arc.reg", "w")
        print >>ds9_region, """\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image\
"""

    # Now, going downwards, follow the line
    all_row_data = None

    for direction_y in [-1,+1]:
        #direction_y = -1
        lt = trace_arc(data=fitsdata,
                       start=(arcpos_x, wls_data['line']),
                       direction=direction_y,
                       max_window_x=5,
                       )
        
        valid = numpy.isfinite(lt[:,1])
        lt = lt[valid]

        if (not ds9_region_file == None): 
            for idx in range(1, lt.shape[0]):
                # print >>ds9_region, 'point(%d,%d)' % (lt[idx,1], lt[idx,0])
                print >>ds9_region, 'line(%d,%d,  %d,%d' % (lt[idx,1]+1, lt[idx,0]+1, lt[idx-1,1]+1, lt[idx-1,0]+1)

        all_row_data = lt if all_row_data == None else numpy.append(all_row_data, lt, axis=0)

    # Sort all_row_data by vertical position
    si = numpy.argsort(all_row_data[:,0])
    all_row_data = all_row_data[si]

    with open("linetrace_idx.%d" % (line_idx), "w") as lt_file:
        numpy.savetxt(lt_file, all_row_data)
        print >>lt_file, "\n\n\n\n\n"

    if (not ds9_region_file == None): ds9_region.close()


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")


    filename = sys.argv[1]
    logger.info("Tracing arcs in file %s" % (filename))
    hdulist = pyfits.open(filename)
    line = hdulist['SCI'].data.shape[0]/2

    logger.info("Attempting to find wavelength solution")
    pickle_file = "traceline.pickle"
    try:
        wls_data = pickle.load(open(pickle_file, "rb"))
        logger.info("Using pickled data - may need to delete --> %s <--" % (pickle_file))
    except:
        wls_data = wlcal.find_wavelength_solution(filename, line)
        pickle.dump(wls_data, open(pickle_file, "wb"))

    logger.info("Continuing with tracing lines!\n")
    time.sleep(0.1)

    # Now pick the strongest line from the results
    arclines = wls_data['linelist_arc']
    max_s2n = numpy.argmax(arclines[:,4])

    # For debugging, extract a data block around the position of the line
    fitsdata = hdulist['SCI'].data.T
    fitsdata[fitsdata <= 0] = numpy.NaN

    logger.info("Applying 5x0 pixel gauss filter")
    fitsdata = scipy.ndimage.filters.gaussian_filter(fitsdata, (5,0), 
                                          mode='constant', cval=0,
                                          )
    pyfits.PrimaryHDU(data=fitsdata.T).writeto("image_smooth.fits", clobber=True)

    trace_single_line(fitsdata, wls_data, max_s2n,
                      ds9_region_file="ds9_arc.reg")

    for i in range(wls_data['linelist_arc'].shape[0]):
        trace_single_line(fitsdata, wls_data, i)

    pysalt.mp_logging.shutdown_logging(logger_setup)
