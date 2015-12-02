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

from helpers import *
# from rk_specred import find_slit_profile

createdebugfiles = True


def trace_arc(data,
              start,
              direction=-1, # -1: downwards, +1: upwards
              max_window_x=5, # how far do we allow the arc to move from one row to the row
              max_corner_angle=60, # in degrees
              ):

    logger = logging.getLogger("TraceArc")

    # extract x/y from tuple
    start_x, start_y = start

    # Create a list of row numbers that we need to inspect
    if (direction > 0):
        # We are going upwards
        logger.debug("Moving upwards")
        row_numbers = numpy.arange(start_y+direction, data.shape[1], direction)
    elif (direction < 0):
        # downwards
        logger.debug("Moving downwards")
        row_numbers = numpy.arange(start_y+direction, direction, direction)
    elif (direction == 0):
        logger.error("going neither up nor down, bad boy, very bad boy!!!")
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

    n_pixels_for_corner = 5
    
    for lines_since_start, next_row_idx in enumerate(row_numbers):
        logger.debug("Moving from row %4d to %4d" % (current_row_idx, next_row_idx))

        # extract a bunch of pixels in the next row around the position of the 
        # current peak

        # Keep track of where the center position is

        next_row = data[current_col_idx-max_window_x:current_col_idx+max_window_x+1,
                        next_row_idx]

        # If next row contains pixels marked as NaN's stop work to avoid going 
        # off into no-mans-land
        if (numpy.sum(numpy.isnan(next_row) > 0)):
            logger.debug("Found illegal pixel in next row (%d, %d), stopping here!" % (
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
                logger.debug("Corner detected in line %d: %.1f then vs %.1f now" % (
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




def subpixel_centroid_trace(data, tracedata, width=5, dumpfile=None, return_all=False):

    logger = logging.getLogger("SubpixelCentroid")

    #
    # Grow the data by <width> pixels on either side to avoid problems with 
    # cutouts close to the edge
    #
    grow_data = numpy.empty((data.shape[0], data.shape[1]+2*width))
    # insert the image data, setting boundaries to NaN
    grow_data[:, :width] = numpy.NaN
    grow_data[:, -width:] = numpy.NaN
    grow_data[:, width:-width] = data[:,:]
    
    logger.debug("size was: %4d x %4d, now it is %4d x %4d" % (
        data.shape[1], data.shape[0], grow_data.shape[1], grow_data.shape[0]))

    #
    # Now extract only the lines for which we have data
    #
    select_y = tracedata[:,0].astype(numpy.int)
    selected_y = grow_data[select_y]

    #
    # Get indices of each pixel in the y-selected, grown data buffer
    # We'll use them to select and cutout the rough image
    #
    iy, ix = numpy.indices(selected_y.shape)
    # shift the x indiced by width to account for the NaN pixel filling
    ix -= width

    #
    # now create a boolean selection mask including only pixels in the vicinity 
    # of the traced arc
    #
    select_x = tracedata[:,1].reshape((-1,1)).astype(numpy.int)
    part_of_line = (ix >= select_x-width) & (ix <= select_x+width)

    #
    # And finally extract the line, with centers roughly aligned
    #
    data_sel = grow_data[select_y][part_of_line].reshape((-1, 2*width+1))
    line_positions = ix[part_of_line].reshape((-1, 2*width+1))

    #
    # Do some very simple background subtraction, by taking the average flux 
    # between the left and right most pixels in our little window to be the 
    # average background, then interpolate linearly between both values to find
    # the background across the slit
    #
    bg_offset = data_sel[:,0] # value at left edge
    bg_slope = (data_sel[:,-1] - data_sel[:,0]) / (data_sel.shape[1]-1)
    idx_x = numpy.arange(data_sel.shape[1], dtype=numpy.float).reshape((1,-1))
    background = idx_x * bg_slope.reshape((-1,1)) + bg_offset.reshape((-1,1))
    #print "background shape:", background.shape, data_sel.shape

    # make a copy before we do the background subtraction
    raw_data = data_sel.copy()

    # subtract off the background to avoid skewing the line center position in
    # the direction of the (potential) background slope
    data_sel -= background

    #
    # With that info, we can now create the flux-weighted center position
    #
    integrated_intensity = bottleneck.nansum(data_sel, axis=1)
    weighted_center_x = bottleneck.nansum(data_sel*line_positions, axis=1)/integrated_intensity
    #print weighted_center_x.shape

    if (not dumpfile == None and createdebugfiles):
        if (type(dumpfile) == pyfits.hdu.image.ImageHDU):
            dumpfile.data = data_sel
            dumpfile.header['OBJECT'] = "avg. line pos: %.2f px" % (numpy.median(line_positions))
        else:
            # prepare a fits file with the rough-rectified line, with column numbers
            hdulist = pyfits.HDUList([
                pyfits.PrimaryHDU(),
                pyfits.ImageHDU(data=data_sel,name='BGSUB'),
                pyfits.ImageHDU(data=line_positions,name='POS'),
                pyfits.ImageHDU(data=raw_data, name='RAW'),
            ])
            hdulist.writeto(dumpfile, clobber=True)

    if (return_all):
        return weighted_center_x, integrated_intensity, raw_data, data_sel

    return weighted_center_x, integrated_intensity


    
def trace_single_line(fitsdata, wls_data, line_idx, ds9_region_file=None,
                      fine_centroiding=False,
                      centroiding_width=5,
                      linetrace_hdulist=None):

    logger = logging.getLogger("TraceSlgLine")

    arclines = wls_data['linelist_arc']
    primeline = arclines[line_idx,:]
    # print primeline
    arcpos_x = primeline[wlcal.lineinfo_colidx['PIXELPOS']]
    logger.debug("Beginning line-trace at position X=%d, Y=%d" % (arcpos_x, wls_data['line']))
            
    #data_around_line = fitsdata[arcpos_x-100:arcpos_x+100,:]
    #pyfits.PrimaryHDU(data=data_around_line.T).writeto("arcregion.fits", clobber=True)

#     if (not ds9_region_file == None):
#         ds9_region = open(ds9_region_file, "a")
#         print >>ds9_region, """\
# # Region file format: DS9 version 4.1
# global color=black dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
# image\
# """

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

        # if (not ds9_region_file == None): 
        #     print >>ds9_region, '# text(%d,%d) text={%d}' % (lt[0,1]+1, lt[0,0]+1, line_idx)

        #     for idx in range(1, lt.shape[0]):
        #         # print >>ds9_region, 'point(%d,%d)' % (lt[idx,1], lt[idx,0])
        #         print >>ds9_region, 'line(%d,%d,  %d,%d' % (lt[idx,1]+1, lt[idx,0]+1, lt[idx-1,1]+1, lt[idx-1,0]+1)

        all_row_data = lt if all_row_data == None else numpy.append(all_row_data, lt, axis=0)

    # Sort all_row_data by vertical position
    si = numpy.argsort(all_row_data[:,0])
    all_row_data = all_row_data[si]

    # with open("linetrace_idx.%d" % (line_idx), "w") as lt_file:
    #     numpy.savetxt(lt_file, all_row_data)
    #     print >>lt_file, "\n\n\n\n\n"

    # if (not ds9_region_file == None): ds9_region.close()

    #
    # create a debug file for all line-traces combined
    #

    if (fine_centroiding):
        logger.debug("Done with tracing, starting fine centroiding")
        #print all_row_data.shape

        # cutout regions close (+/- width pixels) to line
        # traced_y_pos = all_row_data[:,0].astype(numpy.int)
        # traced_x_pos = all_row_data[:,1].astype(numpy.int)
        # x1 = traced_x_pos - centroiding_width
        # x2 = traced_x_pos + centroiding_width+1
        # print x1
        # line_cutout = fitsdata[traced_y_pos, x1:x2]
        # print line_cutout.shape

        imghdu = pyfits.ImageHDU()
        fine_pos = subpixel_centroid_trace(data=fitsdata.T, tracedata=all_row_data, width=10, 
                                           dumpfile=imghdu,
                                           )#"linetrace_%d.fits" % (line_idx))
        if (not linetrace_hdulist == None):
            linetrace_hdulist.append(imghdu)
        #pyfits.PrimaryHDU(data=rectified).writeto("linetrace_%d.fits" % (line_idx), clobber=True)
        
    else:
        fine_pos = all_row_data[:,1]

    
    if (createdebugfiles):
        with open("linetrace_idx.%d" % (line_idx), "w") as lt_file:
            logger.debug("Writing linetrace to %s" % ("linetrace_idx.%d" % (line_idx)))
            numpy.savetxt(lt_file, all_row_data)
            print >>lt_file, "\n\n\n\n\n"

            # Now replace the coarse x-position with the new, fine positions
            # all_row_data[:,1] = fine_pos[:] # XXX
            numpy.savetxt(lt_file, all_row_data)

    else:
        all_row_data[:,1] = fine_pos[:]

    # make sure all trace positions are real positions, and exclude potential problems
    good_pos = numpy.isfinite(all_row_data[:,0]) & numpy.isfinite(all_row_data[:,1])
    all_row_data = all_row_data[good_pos]


    if (not ds9_region_file == None):
        with open(ds9_region_file, "a") as ds9_region:
            print >>ds9_region, """\
# Region file format: DS9 version 4.1
global color=black dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image\
"""
            for idx in range(1, all_row_data.shape[0]):
                print >>ds9_region, 'line(%.0f,%d,  %.0f,%d) # line=0 0' % (
                    all_row_data[idx,1]+1, all_row_data[idx,0]+1, 
                    all_row_data[idx-1,1]+1, all_row_data[idx-1,0]+1)
                
            ds9_region.close()


    #
    # Assemble the return data
    #


    # compute the wavelength of this line
    # print 
    wl = numpy.polynomial.polynomial.polyval(
        wls_data['linelist_arc'][line_idx,wlcal.lineinfo_colidx['PIXELPOS']], 
        wls_data['wl_fit_coeffs'])
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
def polyfit2d(x, y, z, order=[3,2]):
    ncols = (order[0] + 1) * (order[1] + 1)
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order[0]+1), range(order[1]+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m, order

def polyval2d(x, y, m_order):
    m,order = m_order
    ij = itertools.product(range(order[0]+1), range(order[1]+1))
    z = numpy.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z



def pick_line_every_separation(
        arc_linelist,
        trace_every, min_line_separation,
        n_pixels,
        min_signal_to_noise=10,
):

    logger = logging.getLogger("PickLines")

    pickable_lines = numpy.array([])

    #
    # Convert all potentially fractional values to pixel coordinates
    #
    if (trace_every < 1):
        trace_every *= n_pixels
    if (min_line_separation < 1):
        min_line_separation *= n_pixels

    #
    # Add a unique line identifier to all lines
    #
    linelist = numpy.append(
        arc_linelist,
        numpy.arange(arc_linelist.shape[0]).reshape((-1,1)),
        axis=1)
    # numpy.savetxt("pick_input", linelist)

    # 
    # Now go through each interval, and select all lines that
    # - are strong enough to fulfill the S/N criteria
    # - or the strongest line in this interval
    # - and at least min_separation away from other lines in this interval
    #
    left_edge = 0
    right_edge = left_edge + trace_every
    while (right_edge < n_pixels):

        #
        # Find all strong lines in interval 
        # 
        in_window = \
            (linelist[:, wlcal.lineinfo_colidx['PIXELPOS']] >= left_edge) & \
            (linelist[:, wlcal.lineinfo_colidx['PIXELPOS']] <= right_edge)
        line_candidates = linelist[in_window]

        if (numpy.sum(in_window) <= 0):
            # No lines found in window
            right_edge += trace_every
            continue

        # print
        # print "Found these lines between", left_edge,"and",right_edge,":"
        # print line_candidates[:, wlcal.lineinfo_colidx['PIXELPOS']]
        # print "line-IDs:", line_candidates[:, -1]
        logger.debug("Found these lines between %d and %d:\n%s\nLine-IDs:\n%s" % (
            left_edge, right_edge,
            " ".join(["%.1f" % x for x in line_candidates[:, wlcal.lineinfo_colidx['PIXELPOS']]]),
            " ".join(["%d" % i for i in line_candidates[:, -1]]),
            ))

        print line_candidates
        strong_line =  line_candidates[:, wlcal.lineinfo_colidx['S2N']] > min_signal_to_noise
            
        if (numpy.sum(strong_line) <= 0):
            # No strong lines found
            # pick the strongest line we could find
            strong_line = [numpy.argmax(line_candidates[:, wlcal.lineinfo_colidx['S2N']])]
            logger.debug("Picking the strongest line locally: ID=%d pxX=%.1f" % (
                strong_line[0], line_candidates[strong_line, wlcal.lineinfo_colidx['PIXELPOS']]))

            
        # IDs for good lines in this part of the spectrum
        good_lines = line_candidates[strong_line,-1]
        # print "good line IDs:", good_lines
        # print "line positions:", line_candidates[strong_line, wlcal.lineinfo_colidx['PIXELPOS']]

        selected_lines = line_candidates[strong_line]
        # print "\nsel. lines:", selected_lines

        left_edge = numpy.max(selected_lines[:, wlcal.lineinfo_colidx['PIXELPOS']]) + min_line_separation
        right_edge = left_edge + trace_every
        # print "\nNew left edge:", left_edge
        #print "Searching between", left_edge,"and",right_edge

        pickable_lines = numpy.append(pickable_lines, selected_lines[:,-1])

    # print "\n=========="*5

    pickable_lines = pickable_lines.astype(numpy.int)

    # print pickable_lines
    # numpy.savetxt(sys.stdout, linelist[pickable_lines], " %8.2f")
    # numpy.savetxt("pickable", linelist[pickable_lines], " %8.2f")

    #
    # Now weed out the lines that are too close to other lines, eliminating 
    # the weaker ones first
    #
    cands = linelist[pickable_lines]
    
    final_indices = numpy.array([], dtype=numpy.int)
    while (cands.shape[0] > 0):
        xpos_tree = scipy.spatial.cKDTree(cands[:, wlcal.lineinfo_colidx['PIXELPOS']].reshape((-1,1)))
        d,i = xpos_tree.query(cands[:, wlcal.lineinfo_colidx['PIXELPOS']].reshape((-1,1)),
                              k=100,
                              distance_upper_bound=min_line_separation)

        # Select all lines with only 1 match (themselves). These are keepers
        count = numpy.sum(numpy.isfinite(d), axis=1)
        # print count.shape
        # print count

        # For all other lines, eliminate the least significant line in the 
        # close vicinity and try again

        keepers = (count == 1)
        # print cands[keepers, -1]
        final_indices = numpy.append(final_indices, cands[keepers, -1])

        
        # print "BEFORE:", cands.shape
        # numpy.delete(cands, keepers)
        cands = cands[~keepers]
        # print "AFTER:", cands.shape
        # print count[~keepers]

        if (cands.shape[0] <= 1):
            # No lines left, so nothing left to eliminate
            break

        #
        # Find the weakest line
        #
        weakest = numpy.argmin(cands[:, wlcal.lineinfo_colidx['S2N']])
        # sel_strong = numpy.ones(cands.shape[0], dtype=numpy.bool)
        # sel_strong[weakest] = False
        cands = numpy.delete(cands, weakest, axis=0)
        # print "AFTER #2:", cands.shape

        # print d.shape, d
        # print i.shape, i
        # break

    # print final_indices.shape
    # print final_indices

    logger.debug("Final line indices:\n%s" % (
        " ".join(["%d" % i for i in final_indices.astype(numpy.int)])
    ))
    # numpy.savetxt("picked_final", linelist[final_indices.astype(numpy.int)], " %8.2f")
    return final_indices.astype(numpy.int)


def compute_2d_wavelength_solution(arc_filename, 
                                   n_lines_to_trace=15, 
                                   fit_order=[3,2],
                                   output_wavelength_image=None,
                                   debug=False,
                                   arc_region_file=None,
                                   return_slitprofile=False,
                                   trace_every=None,
                                   min_line_separation=0.03):

    if (type(arc_filename) == str and os.path.isfile(arc_filename)):
        # We received a filename as parameter
        _, bn = os.path.split(arc_filename)
        logger = logging.getLogger("Comp2D-WLS(%s)" % (bn))
        logger.info("Tracing arcs in file %s" % (arc_filename))
        hdulist = pyfits.open(arc_filename)
    elif (type(arc_filename) == pyfits.hdu.hdulist.HDUList):
        # This is already a valid HDUlist, so we don't need to open anything
        hdulist = arc_filename
        logger = logging.getLogger("Comp2D-WLS(HDU)")


        
    line = hdulist['SCI'].data.shape[0]/2

    logger.info("Attempting to find wavelength solution")

    if (debug and False):
        pickle_file = "traceline.pickle"
        try:
            wls_data = pickle.load(open(pickle_file, "rb"))
            logger.info("Using pickled data - may need to delete --> %s <--" % (pickle_file))
        except:
            wls_data = wlcal.find_wavelength_solution(arc_filename, line)
            pickle.dump(wls_data, open(pickle_file, "wb"))
    else:
        wls_data = wlcal.find_wavelength_solution(arc_filename, line)


    #print wls_data

    logger.info("Continuing with tracing lines!")
    
    # Now pick the strongest line from the results
    arclines = wls_data['linelist_arc']
    max_s2n = numpy.argmax(arclines[:,wlcal.lineinfo_colidx['S2N']])
    logger.info("Strongest line detected has S/N = %8.2f" % (
            arclines[max_s2n,wlcal.lineinfo_colidx['S2N']]))

    #
    # Using routines from the spectral reduction module, flatten ARC spectrum 
    # in slit direction (vertical, along Y axis) to make arcs roughly the same 
    # brightness along their entire length
    #
    logger.info("Creating slit profile for normalization")
    slitprofile_fit, mask, slitprofile = find_slit_profile(hdulist, arc_filename, source_region=None)
    #print "SLITPROFILE:", slitprofile.shape, hdulist['SCI'].data.shape
    if (debug): numpy.savetxt("slitprofile.dump", slitprofile)

    #
    # For debugging, extract a data block around the position of the line
    #
    # also apply the slitprofile correction to minimize brightness variations 
    # along the slit
    #
    fitsdata = (hdulist['SCI'].data / slitprofile).T
    
    #truncate to cut off rough edges
    #fitsdata = fitsdata[:, 60:1985]

    if (debug): 
        pyfits.PrimaryHDU(data=fitsdata.T).writeto("image_slitflattened.fits", clobber=True)

    binx, biny = pysalt.get_binning(hdulist)

    gauss_width = 8./binx
    logger.info("Applying %.1f pixel gauss filter in spectral dir" % (gauss_width))
    fitsdata_gf = scipy.ndimage.filters.gaussian_filter(fitsdata, (gauss_width,0), 
                                          mode='constant', cval=0,
                                          )
    fitsdata_gf[fitsdata <= 0] = numpy.NaN
    fitsdata[fitsdata <= 0] = numpy.NaN

    if (debug):
        pyfits.PrimaryHDU(data=fitsdata.T).writeto("image_smooth.fits", clobber=True)


    #
    # Determine curvature with the 10 strongest lines
    # (Note: argsort sorts low to high, so need to reverse order to get high to low)
    #
    # Also eliminate all lines with nearby companions that might cause problems
    #
    #print "\n**********"*7
    #print wls_data['linelist_arc']
    #print "\n**********"*7
    #time.sleep(2)

    
    if (not trace_every == None):

        pickle.dump(wls_data['linelist_arc'], open("arclist", "wb"))
        # print "exiting"
        # sys.exit(0)
        trace_line_indices = pick_line_every_separation(
            wls_data['linelist_arc'],
            trace_every, min_line_separation,
            hdulist['SCI'].data.shape[1]
            )
    else:
        if (n_lines_to_trace == 0):
            # if 0, use all lines
            trace_line_indices = range(wls_data['linelist_arc'].shape[0])

        elif (n_lines_to_trace > 0):
            # if N positive, use the N strongest lines
            sort_sn = numpy.argsort(wls_data['linelist_arc'][:,wlcal.lineinfo_colidx['S2N']])[::-1]
            trace_line_indices = sort_sn[:n_lines_to_trace]

        else: 
            # if N negative, use the absolute value of N as S/N cutoff
            strong_enough = wls_data['linelist_arc'][:,wlcal.lineinfo_colidx['S2N']] > math.fabs(n_lines_to_trace)
            trace_line_indices = numpy.arange(wls_data['linelist_arc'].shape[0])[strong_enough]


    # Reset ds9_arc
    if (not arc_region_file == None):
        pysalt.clobberfile(arc_region_file)

    traces = None
    logger.info("Preparing to trace %d lines" % (len(trace_line_indices)))
    #print trace_line_indices
    #print "XXX"

    linetrace_hdulist = [pyfits.PrimaryHDU()]

    for i in trace_line_indices: #range(n_lines_to_trace):
        linetrace = trace_single_line(fitsdata_gf, wls_data, i,
                                      ds9_region_file=arc_region_file,
                                      fine_centroiding=True,
                                      centroiding_width=10,
                                      linetrace_hdulist=linetrace_hdulist)
        # linetrace = trace_single_line(fitsdata_gf, wls_data, sort_sn[i],
        #                    ds9_region_file=arc_region_file)
        # print linetrace.shape
        numpy.savetxt("LT.%d" % i, linetrace)

        traces = linetrace if traces == None else \
                 numpy.append(traces, linetrace, axis=0)
        #traces.append(linetrace)

    linetrace_fitsfile = "linetraces.fits"
    linetrace_hdulist = pyfits.HDUList(linetrace_hdulist)
    #clobberfile(linetrace_fitsfile)
    linetrace_hdulist.writeto(linetrace_fitsfile, clobber=True)


    traces_2d = numpy.array(traces)

    if (debug):
        #print traces
        #print traces_2d.shape
        numpy.savetxt("traces_2d.dmp", traces_2d)

    #
    # Now we have a full array of X, Y, and wavelength positions.
    # Go on and fit a full 2-D polynomial fit
    #

    #numpy.savetxt("all_traces", traces)

    m = polyfit2d(x=traces[:,1],
                  y=traces[:,0],
                  z=traces[:,4],
                  order=fit_order)

    plot_solution = False
    if (plot_solution and debug):
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
    #print line

    wl_data = polyval2d(arc_x.astype(numpy.float32), arc_y.astype(numpy.float32), m)

    if (debug):
        pyfits.PrimaryHDU(data=wl_data.T).writeto(
            "image_wavelengths.fits", clobber=True)    
        
    if (not output_wavelength_image):
        wli_hdulist = pyfits.HDUList([pyfits.PrimaryHDU(),
                                  pyfits.ImageHDU(data=wl_data.T),
                                  pyfits.ImageHDU(data=fitsdata.T)])
        wli_hdulist.writeto(output_wavelength_image, clobber=True)

    if (debug and False):
        for stripwidth in [5,25,75,150,300, 600]:
            # stripwidth = 75
            pick_strip = (arc_y > line-stripwidth) & (arc_y < line+stripwidth)

            # Now merge data and wavelengths and write to file
            logger.info("dumping wavelenghts and fluxes into file")
            # merged = numpy.append(wl_data.reshape((-1,1)),
            #                       fitsdata.T.reshape((-1,1)),
            #                       #hdulist['SCI'].data.T.reshape((-1,1)),
            #                       #fitsdata[pick_strip].reshape((-1,1)),
            #                       axis=1)
            merged = numpy.append(wl_data[pick_strip].reshape((-1,1)),
            #                       hdulist['SCI'].data.T[pick_strip].reshape((-1,1)),
                                  fitsdata[pick_strip].reshape((-1,1)),
                                  axis=1)
            si = numpy.argsort(merged[:,0])
            merged = merged[si]

            in_range = (merged[:,0] > 5930) & (merged[:,0] < 6000)
            merged = merged[in_range]

            print merged.shape
            # logger.info("dumping to file")
            # numpy.savetxt("wl+flux.dump.%d" % (stripwidth), merged)

    if (return_slitprofile):
        return wl_data.T, slitprofile

    return wl_data.T




if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")


    filename = sys.argv[1]

    n_lines = 15
    try:
        n_lines = int(sys.argv[2])
    except:
        pass

    wls_2d = compute_2d_wavelength_solution(
        arc_filename=filename, 
        n_lines_to_trace=n_lines, 
        fit_order=[3,2],
#        fit_order=[4,4],
        output_wavelength_image="wl+image.fits",
        debug=True,
        arc_region_file="ds9_arc.reg")


    
    # trace_single_line(fitsdata, wls_data, max_s2n,
    #                   ds9_region_file="ds9_arc.reg")

    # for i in range(wls_data['linelist_arc'].shape[0]):
    #     #skip line if S/N is too low
    #     if (wls_data['linelist_arc'][i,4] < 50):
    #         continue

    #     trace_single_line(fitsdata, wls_data, i,
    #                       ds9_region_file="ds9_arc.reg")

    pysalt.mp_logging.shutdown_logging(logger_setup)
