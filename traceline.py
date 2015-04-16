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
              max_corner_angle=30, # in degrees
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
    arc_center = numpy.empty((data.shape[0],3))
    arc_center[:,0] = numpy.NaN
    arc_center[start_y,0] = start_x

    n_pixels_for_corner = 3
    for lines_since_start, next_row_idx in enumerate(row_numbers):
        #logger.debug("Moving from row %4d to %4d" % (current_row_idx, next_row_idx))

        # extract a bunch of pixels in the next row around the position of the 
        # current peak

        # Keep track of where the center position is

        next_row = data[current_col_idx-max_window_x:current_col_idx+max_window_x+1,
                        next_row_idx]

        # If next row contains pixels marked as NaN's stop work to avoid going 
        # off into no-mans-land
        if (numpy.sum(numpy.isnan(next_row) > 0)):
            logger.info("Found illegal pixel in next row (%d, %d), stopping here!" % (
                    current_col_idx, next_row_idx))
            break

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
        
        #
        # Add some edge-detection here:
        # If the direction of the arc changes by more than 30 degrees from the 
        # direction across the past 10 pixels, assume this is an edge and stop 
        # tracing the line at this point
        #
        #print current_row_idx
        if (lines_since_start > 2*n_pixels_for_corner):
            # compute the arc angles in the past N pixels and the N pixels 
            # before that.
            # If the two angles differ a lot ( >= max_corner_angle ) we found a 
            # corner and can stop following the arc
            dx_past = arc_center[row_numbers[lines_since_start - 2*n_pixels_for_corner],0] \
                - arc_center[row_numbers[lines_since_start - 1*n_pixels_for_corner],0]
            dx_now = arc_center[row_numbers[lines_since_start - 1*n_pixels_for_corner],0] \
                - arc_center[row_numbers[lines_since_start-1],0]
            # dx_past = arc_center[row_numbers[lines_since_start - 2*n_pixels_for_corner],0] \
            #     - arc_center[current_row_idx - 1*n_pixels_for_corner,0]
            # dx_now = arc_center[current_row_idx - 1*n_pixels_for_corner,0] \
            #     - arc_center[current_row_idx - 0*n_pixels_for_corner,0]

            angle_past = numpy.degrees(numpy.arctan2(dx_past, n_pixels_for_corner))
            angle_now = numpy.degrees(numpy.arctan2(dx_now, n_pixels_for_corner))
            #print current_row_idx, dx_past, dx_now, angle_past, angle_now 

            arc_center[next_row_idx,1] = dx_now #angle_now
            arc_center[next_row_idx,2] = dx_past #angle_past

            if (math.fabs(angle_past - angle_now) > max_corner_angle):
                # We found a corner
                # --> retroactively stop following the line at the point the line 
                #     started to deviate
                logger.info("Corner detected in line %d: %.1f then vs %.1f now" % (
                        current_row_idx, angle_past, angle_now))
                avg_past = dx_past / n_pixels_for_corner
                # for i in range(2*n_pixels_for_corners):
                break

        # corner_detected = False
        # idx_n_pixels_back = 
        # if (idx_n_pixels_back < 0 or idx_n_pixels_back > data.shape[1]):
        #     # this would put us out of range, so no check here
        #     corner_detected = False
        #     logger.info("(line %4d) No check for angle <-- out of range (%d)" % (current_row_idx, idx_n_pixels_back))
        # elif (lines_since_start < n_pixels_for_corner): #numpy.isnan(arc_center[idx_n_pixels_back,0])):
        #     # again this is a bad pixels, most likely because we are not yet 
        #     # <n_pixels_for_corner> pixels into the line
        #     corner_detected = False
        #     logger.info("(line %4d) No check for angle <-- too early" % (current_row_idx))
        # else:
        #     past_n_pixels = arc_center[idx_n_pixels_back,0] - current_col_idx
        #     past_angle = math.degrees(numpy.arctan2(past_n_pixels, n_pixels_for_corner*direction))
        #     now_angle = math.degrees(numpy.arctan2(x_offsets[max_gradient], direction))

        #     arc_center[next_row_idx,1] = now_angle
        #     arc_center[next_row_idx,2] = past_angle
        #     if (math.fabs(past_angle - now_angle) > max_corner_angle):
        #         logger.info("Identified corner at this point (%f vs %f), aborting!" % (
        #                 now_angle, past_angle))
        #         corner_detected = False
                    
        #     print current_row_idx, idx_n_pixels_back, past_n_pixels, past_angle, x_offsets[max_gradient], now_angle, corner_detected
        #     # logger.info("(line %4d) Corner detection: shift(10px)=%3d --> angle=%7.1f   || now: %3d, angle=%7.1f ==> %5s" % (
        #     #         current_row_idx, past_n_pixels, past_angle, x_offsets[max_gradient], now_angle, corner_detected))
        # if (corner_detected):
        #     break

    
        if (math.fabs(next_row_idx-start_y) > 5000):
            logger.info("Aborting search!")
            break

        #
        # Save all positions for the next iteration
        #
        arc_center[next_row_idx,0] = current_col_idx
        current_row_idx = next_row_idx
        current_col_idx = next_col_idx
        
    # For debugging: save the arc position
    # combined = numpy.append(numpy.arange(data.shape[0]).reshape((-1,1)),
    #                         arc_center.reshape((-1,1)), axis=1)
    combined = numpy.append(numpy.arange(data.shape[0]).reshape((-1,1)),
                            arc_center, axis=1)
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
        ds9_region = open(ds9_region_file, "a")
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
            print >>ds9_region, '# text(%d,%d) text={%d}' % (lt[0,1]+1, lt[0,0]+1, line_idx)

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

    #
    # Assemble the return data
    #

    # compute the wavelength of this line
    print 
    wl = numpy.polynomial.polynomial.polyval(wls_data['linelist_arc'][line_idx,1], wls_data['wl_fit_coeffs'])
    linetrace = numpy.append(all_row_data,
                             numpy.ones((all_row_data.shape[0],1))*wl, 
                             axis=1)

    return linetrace






# from http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    # x = np.random.random(numdata)
    # y = np.random.random(numdata)
    # z = x**2 + y**2 + 3*x**3 + y + np.random.random(numdata)

    # # Fit a 3rd order, 2d polynomial
    # m = polyfit2d(x,y,z)

    # # Evaluate it on a grid...
    # nx, ny = 20, 20
    # xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
    #                      np.linspace(y.min(), y.max(), ny))
    # zz = polyval2d(xx, yy, m)

    # # Plot
    # plt.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
    # plt.scatter(x, y, c=z)
    # plt.show()

import itertools
import matplotlib.pyplot as plt
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(numpy.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = numpy.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z









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
    #fitsdata[fitsdata <= 0] = numpy.NaN

    logger.info("Applying 5x0 pixel gauss filter")
    fitsdata_gf = scipy.ndimage.filters.gaussian_filter(fitsdata, (5,0), 
                                          mode='constant', cval=0,
                                          )
    fitsdata_gf[fitsdata <= 0] = numpy.NaN
    fitsdata = fitsdata_gf

    pyfits.PrimaryHDU(data=fitsdata.T).writeto("image_smooth.fits", clobber=True)

    # trace_single_line(fitsdata, wls_data, 21,
    #                   ds9_region_file="ds9_arc.reg")

    # Determine curvature with the 10 strongest lines
    # Also eliminate all lines with nearby companions that might cause problems
    sort_sn = numpy.argsort(wls_data['linelist_arc'][:,4])[::-1]
    print wls_data['linelist_arc'][sort_sn][:10,4]

    # Reset ds9_arc
    pysalt.clobberfile("ds9_arc.reg")

    traces = None
    for i in range(15):
        linetrace = trace_single_line(fitsdata, wls_data, sort_sn[i],
                           ds9_region_file="ds9_arc.reg")
        print linetrace.shape
        numpy.savetxt("LT.%d" % i, linetrace)

        traces = linetrace if traces == None else \
                 numpy.append(traces, linetrace, axis=0)
        #traces.append(linetrace)

    print traces
    traces_2d = numpy.array(traces)
    print traces_2d.shape
    numpy.savetxt("traces_2d.dmp", traces_2d)

    #
    # Now we have a full array of X, Y, and wavelength positions.
    #

    m = polyfit2d(x=traces[:,1],
                  y=traces[:,0],
                  z=traces[:,4],
                  order=2)

    plot_solution = False
    if (plot_solution):
        x=traces[:,1]
        y=traces[:,0]
        z=traces[:,4]
        nx, ny = 20, 20
        xx, yy = numpy.meshgrid(numpy.linspace(x.min(), x.max(), nx), 
                                numpy.linspace(y.min(), y.max(), ny))
        zz = polyval2d(xx, yy, m)
        plt.imshow(zz, extent=(x.min(), x.max(), y.min(), y.max()))
        plt.scatter(x, y, c=z, linewidth=0)
        plt.show()

    #
    # Go on to compute a full grid of wavelengths, with one 
    # position for each pixel in the input frame
    #

    logger.info("Computing full 2-D wavelength map for frame")
    arc_x, arc_y = numpy.indices(fitsdata.shape)

    line = wls_data['line']
    print line

    #stripwidth = 250
    #pick_strip = (arc_y > line-stripwidth) & (arc_y < line+stripwidth)

    wl_data = polyval2d(arc_x.astype(numpy.float32), arc_y.astype(numpy.float32), m)
    pyfits.PrimaryHDU(data=wl_data.T).writeto(
        "image_wavelengths.fits", clobber=True)    

    # Now merge data and wavelengths and write to file
    logger.info("dumping wavelenghts and fluxes into file")
    merged = numpy.append(wl_data.reshape((-1,1)),
                          hdulist['SCI'].data.T.reshape((-1,1)),
                          #fitsdata[pick_strip].reshape((-1,1)),
                          axis=1)
    # merged = numpy.append(wl_data[pick_strip].reshape((-1,1)),
    #                       hdulist['SCI'].data.T[pick_strip].reshape((-1,1)),
    #                       #fitsdata[pick_strip].reshape((-1,1)),
    #                       axis=1)
    si = numpy.argsort(merged[:,0])
    merged = merged[si]

    in_range = (merged[:,0] > 5930) & (merged[:,1] < 6000)
    merged = merged[in_range]

    print merged.shape
    logger.info("dumping to file")
    numpy.savetxt("wl+flux.dump", merged)

    
    # trace_single_line(fitsdata, wls_data, max_s2n,
    #                   ds9_region_file="ds9_arc.reg")

    # for i in range(wls_data['linelist_arc'].shape[0]):
    #     #skip line if S/N is too low
    #     if (wls_data['linelist_arc'][i,4] < 50):
    #         continue

    #     trace_single_line(fitsdata, wls_data, i,
    #                       ds9_region_file="ds9_arc.reg")

    pysalt.mp_logging.shutdown_logging(logger_setup)