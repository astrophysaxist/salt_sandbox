#!/usr/bin/env python

import pyfits
import os, sys
import numpy
import scipy
import traceline
import prep_science
import pysalt.mp_logging
import wlcal
import bottleneck
import traceline
import logging

def filter_with_padding(data, w, fct):

    padded = numpy.empty((data.shape[0]+2*w))
    padded[:] = numpy.NaN
    padded[w:-w] = data

    fm = numpy.array([fct(padded[i-w:i+w+1]) for i in range(w,data.shape[0]+w)])
    return fm

def trace_full_line(imgdata, x_start, y_start, window=5):

    weighted_pos = numpy.zeros((imgdata.shape[0],4))
    weighted_pos[:,:] = numpy.NaN

    x_start = int(x_start)

    x_guess_list = []
    x_guess = x_start

    x_pos_all = numpy.arange(imgdata.shape[1])
    for y in range(y_start, imgdata.shape[0]):
       
        # compute center of line in this row
        if (x_guess-window < 0 or
            x_guess+window >= imgdata.shape[1]):
            continue

        select = (x_pos_all >= x_guess-window) & (x_pos_all <= x_guess+window+1)
        x_pos = x_pos_all[select] #numpy.arange(x_guess-window, x_guess+window+1)
        try:
            flux = imgdata[y, select] #x_guess-window:x_guess+window+1]
        except:
            print x_guess, window, y
            break
            continue

        #print flux.shape, x_pos.shape
        i_flux = numpy.sum(flux)
        _wp = numpy.sum(x_pos*flux) / i_flux

        x_guess_list.append(_wp)
        
        x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
        weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
        #print y,_wp,x_guess

    x_guess = x_start
    x_guess_list = []
    for y in range(y_start, 0, -1):
       
        if (x_guess-window < 0 or
            x_guess+window >= imgdata.shape[1]):
            continue

        # compute center of line in this row
        select = (x_pos_all >= x_guess-window) & (x_pos_all <= x_guess+window+1)
        x_pos = x_pos_all[select] #numpy.arange(x_guess-window, x_guess+window+1)
        try:
            flux = imgdata[y, select] #x_guess-window:x_guess+window+1]
        except:
            print x_guess, window, y
            break
            continue

        #print flux.shape, x_pos.shape
        i_flux = numpy.sum(flux)
        _wp = numpy.sum(x_pos*flux) / i_flux
        # x_pos = numpy.arange(x_guess-window, x_guess+window+1)
        # try:
        #     flux = imgdata[y, x_guess-window:x_guess+window+1]
        # except:
        #     print x_guess, window, y
        #     break
        #     continue
        # i_flux = numpy.sum(flux)
        # _wp = numpy.sum(x_pos*flux) / i_flux


        x_guess_list.append(_wp)
        
        x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
        weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
        #print y,_wp,x_guess

    return weighted_pos

def fit_with_rejection(x, y, fct, p_init):

    for iteration in range(3):
        fit_result = scipy.optimize.leastsq(
            fct,
            p_init, 
            args=(x,y),
            maxfev=500,
            full_output=1,
            )
        p_fit = fit_result[0]
        diff = fct(p_fit, x, y)
        
        percentiles = numpy.percentile(diff, [16,50,84])
        _median = percentiles[1]
        _sigma = 0.5*(percentiles[2]-percentiles[0])
        outlier = (diff > (_median+3*_sigma)) | (diff < (_median-3*_sigma))
        y[outlier] = numpy.NaN
        p_init = p_fit
        print "SCALING:", iteration, p_fit

    return p_fit

def create_wlmap_from_skylines(hdulist):

    logger = logging.getLogger("SkyTrace")

    # imgdata = hdulist['SCI.RAW'].data
    try:
        imgdata = hdulist['SCI.NOCRJ'].data
    except:
        imgdata = hdulist['SCI'].data

    logger.info("Isolating sky lines and continuum")
    skylines, continuum = prep_science.filter_isolate_skylines(data=imgdata)
    pyfits.PrimaryHDU(data=skylines).writeto("skytrace_sky.fits", clobber=True)
    pyfits.PrimaryHDU(data=continuum).writeto("skytrace_continuum.fits", clobber=True)

    # pick a region close to the center, extract block of image rows, and get 
    # line list
    sky1d = bottleneck.nanmean(imgdata[550:575, :].astype(numpy.float32), axis=0)
    print sky1d.shape
    sky_linelist = wlcal.find_list_of_lines(sky1d, avg_width=25, pre_smooth=None)
    numpy.savetxt("sky1d", sky1d)
    numpy.savetxt("skylines.all", sky_linelist)

    # select lines with good spacing
    good_lines = traceline.pick_line_every_separation(
        arc_linelist=sky_linelist,
        trace_every=0.02,
        min_line_separation=0.01,
        n_pixels=imgdata.shape[1],
        min_signal_to_noise=7,
        )
    numpy.savetxt("skylines.good", sky_linelist[good_lines])

    print "X",skylines.shape, sky_linelist.shape, good_lines.shape
    selected_lines = sky_linelist[good_lines]
    print "selected:", selected_lines.shape

    all_traces = []

    logger.info("Tracing %d lines" % (selected_lines.shape[0]))
    linetraces = open("skylines.traces", "w")
    for idx, pick_line in enumerate(selected_lines):
        #print pick_line

        wp = trace_full_line(skylines, x_start=pick_line[0], y_start=562, window=5)
        numpy.savetxt(linetraces, wp)
        print >>linetraces, "\n"*5
        all_traces.append(wp)

    numpy.savetxt("skylines.picked", selected_lines)
    for idx in range(selected_lines.shape[0]):
        pick_line = selected_lines[idx,:]
        #print pick_line

    all_traces = numpy.array(all_traces)
    print all_traces.shape

    ##########################################################################
    #
    # Now do some outlier rejection
    #
    ##########################################################################

    #
    # Compute average profile shape and mean intensity profile
    #
    logger.info("Rejecting outliers along the spatial profile")
    _cl, _cr = int(0.4*all_traces.shape[1]), int(0.6*all_traces.shape[1])
    central_position = numpy.median(all_traces[:,_cl:_cr,:], axis=1)
    numpy.savetxt("skytrace_median", central_position)
    print central_position

    # subtract central position
    all_traces[:,:,1] -= central_position[:,1:2]
    all_traces[:,:,2] -= central_position[:,2:3]

    # scale intensity by median flux
    all_traces[:,:,3] /= central_position[:,3:]

    with open("skylines.traces.norm", "w") as lt2:
        for line in range(all_traces.shape[0]):
            numpy.savetxt(lt2, all_traces[line,:,:])
            print >>lt2, "\n"*5

    #
    # Now eliminate all lines that have negative median fluxes
    #
    logger.info("eliminating all lines with median intensity < 0")
    negative_intensity = central_position[:,3] < 0
    all_traces[negative_intensity,:,:] = numpy.NaN

    #
    # Do the spatial outlier correction first
    #
    profiles = all_traces[:,:,1]
    print profiles.shape
    for iteration in range(3):
        print
        print "Iteration:", iteration
        print profiles.shape
        quantiles = numpy.array(numpy.nanpercentile(
            a=profiles, 
            q=[16,50,84],
            axis=0,
            ))
        print "new:", quantiles.shape

        # quantiles = scipy.stats.scoreatpercentile(
        #     a=profiles, 
        #     per=[16,50,84],
        #     axis=0,
        #     limit=(-1*all_traces.shape[1], 2*all_traces.shape[1])
        #     )
        # print quantiles
        # median = quantiles[1]
        # print median

        # sigma = 0.5*(quantiles[2] - quantiles[0])

        median = quantiles[1,:]
        sigma = 0.5 * (quantiles[2,:] - quantiles[0,:])

        outlier = (profiles > median+3*sigma) | (profiles < median-3*sigma)
        profiles[outlier] = numpy.NaN
        all_traces[:,:,3][outlier] = numpy.NaN

    
    with open("skylines.traces.clean", "w") as lt2:
        for line in range(all_traces.shape[0]):
            numpy.savetxt(lt2, all_traces[line,:,:])
            print >>lt2, "\n"*5

    medians = bottleneck.nanmedian(all_traces, axis=0)
    numpy.savetxt("skylines.traces.median", medians)
    print medians.shape

    stds = bottleneck.nanstd(all_traces, axis=0)
    stds[:,0] = medians[:,0]
    numpy.savetxt("skylines.traces.std", stds)

    #
    # Now reconstruct the final line traces, filling in gaps with values 
    # predicted by the median profile
    #
    logger.info("Reconstructing individual line profiles")
    if (False):
        all_median = numpy.repeat(medians.reshape((-1,1)), all_traces.shape[0], axis=1)
        print all_median.shape, all_traces[:,:,1].shape
        outlier = numpy.isnan(all_traces[:,:,1])
        print outlier.shape
        print outlier
        try:
            all_traces[:,:,1][outlier] = all_median[:,:][outlier]
        except:
            pass
        all_traces[:,:,1] += central_position[:,1:2]


        with open("skylines.traces.corrected", "w") as lt2:
            for line in range(all_traces.shape[0]):
                numpy.savetxt(lt2, all_traces[line,:,:])
                print >>lt2, "\n"*5

        with open("skylines.traces.corrected2", "w") as lt2:
            for line in range(all_traces.shape[0]):
                numpy.savetxt(lt2, all_median[:,:])
                print >>lt2, "\n"*5


    # compute average intensity profile, weighting each line profile by its 
    # median intensity
    logger.info("Computing intensity profile")
    print central_position[:,3]
    sort_intensities = numpy.argsort(central_position[:,3])
    strong_lines = sort_intensities[-10:]
    print strong_lines

    strong_line_fluxes = central_position[:,3][strong_lines]
    strong_line_traces = all_traces[strong_lines,:,:]
    print strong_line_traces.shape

    i_sum = bottleneck.nansum(strong_line_traces[:,:,3] * strong_line_fluxes.reshape((-1,1)), axis=0)
    i_count = bottleneck.nansum(strong_line_traces[:,:,3] / strong_line_traces[:,:,3] * strong_line_fluxes.reshape((-1,1)), axis=0)
    i_avg = i_sum / i_count
    print i_sum.shape
    numpy.savetxt("skylines.traces.meanflux", i_avg)

    fm = filter_with_padding(i_avg, w=50, fct=bottleneck.nanmedian)
    print fm.shape
    numpy.savetxt("skylines.traces.meanflux2", fm)


    #
    # Now fit each individual profile by scaling the median profile
    #
    scalings = []

    def arc_model(p, medianarc):
        return p[0]*medianarc + p[1]*(numpy.arange(medianarc.shape[0])-medianarc.shape[0]/2)

    def arc_error(p, arc, medianarc):
        model = arc_model(p, medianarc)
        diff = (arc-model)
        valid = numpy.isfinite(diff)
        return diff[valid] if numpy.sum(valid) > 0 else medianarc[numpy.isfinite(medianarc)]
        
    good_flux = fm > 0.5*numpy.max(fm)

    for i_arc in range(all_traces.shape[0]):
        
        if (numpy.isnan(central_position[i_arc, 1])):
            continue

        comb = numpy.empty((all_traces.shape[1], 6))
        comb[:,:4] = all_traces[i_arc, :, :]
        comb[:,4] = medians[:,1]

        # print all_traces[i_arc, :, :].shape, medians[:,1].reshape((-1,1)).shape
        # comb = numpy.append(
        #     all_traces[i_arc, :, :],
        #     medians[:,1].reshape((-1,1)), axis=1)
        ypos = int(central_position[i_arc, 1])

        p_init=[1.0, 0.0]
        fit_args=(all_traces[i_arc,:,1][good_flux],medians[:,1][good_flux])
        fit_result = scipy.optimize.leastsq(
            arc_error,
            p_init, 
            args=fit_args,
            maxfev=500,
            full_output=1,
            )
        p_bestfit = fit_result[0]
        print central_position[i_arc, 1], p_bestfit

        scaling = comb[:,4] / comb[:,1]
        scalings.append([ypos, 
                         bottleneck.nanmedian(scaling), 
                         bottleneck.nanmean(scaling), 
                         p_bestfit[0], p_bestfit[1]])
        
        med_scaling = bottleneck.nanmedian(scaling)

        comb[:,5] = arc_model(p_bestfit, medians[:,1])

        numpy.savetxt("ARC_%04d.delete" % (ypos), comb)

    numpy.savetxt("all_scalings", numpy.array(scalings))

    def model_linear(p, x):
        model = p[0]*x+p[1]
        return model
    def fit_linear(p, x, y):
        model = model_linear(p,x)
        diff = y - model
        valid = numpy.isfinite(diff)
        return diff[valid] if valid.any() else y

    fit_scalings = numpy.array(scalings)
    p_scale = fit_with_rejection(fit_scalings[:,0], fit_scalings[:,3],
                               fit_linear,
                               [0.,1.],
                               )
    fit_scalings[:,1] = model_linear(p_scale, fit_scalings[:,0])
    numpy.savetxt("all_scalings_scale", numpy.array(fit_scalings))

    fit_skew = numpy.array(scalings)
    p_skew = fit_with_rejection(fit_scalings[:,0], fit_scalings[:,4],
                                    fit_linear,
                                    [0.,0.],
                                )
    fit_skew[:,2] = model_linear(p_skew, fit_scalings[:,0])
    numpy.savetxt("all_scalings_skew", numpy.array(fit_skew))

    #
    # Now compute spline function for the median curvature profile
    #
    logger.info("Computing spline function for median curvature profile")
    mc_spline = scipy.interpolate.interp1d(
        x=numpy.arange(medians.shape[0]),
        y=medians[:,1],
        kind='linear',
        bounds_error=False,
        fill_value=0,
        )

    #
    # Compute full 2-d map of effective X positions
    #
    logger.info("Computing full 2-D x-eff map")
    y,x = numpy.indices(imgdata.shape)

    #x_eff = x + x*(p_scale[0]*mc_spline(y) + p_skew[0]*y) + p_scale[1] + p_skew[1]*y
    print p_scale
    print p_skew

    x_eff = x
    for iteration in range(3):
        x_eff = x - ((p_scale[0]*x_eff+p_scale[1])*mc_spline(y) + (p_skew[0]*x_eff+p_skew[1])*(y-imgdata.shape[0]/2))
        pyfits.PrimaryHDU(data=x_eff).writeto("x_eff_%d.fits" % (iteration+1), clobber=True)


    #
    # Convert x-eff map to wavelength 
    #
    a0 = hdulist[0].header['WLSFIT_0']
    a1 = hdulist[0].header['WLSFIT_1']
    a2 = hdulist[0].header['WLSFIT_2']
    a3 = hdulist[0].header['WLSFIT_3']

    wl_map = 0.
    for order in range(hdulist[0].header['WLSFIT_N']):
        a = hdulist[0].header['WLSFIT_%d' % (order)]
        wl_map += a * numpy.power(x_eff,order)
    pyfits.PrimaryHDU(data=wl_map).writeto("wl_map.fits", clobber=True)


    return x_eff, wl_map, medians, p_scale, p_skew, fm

    

if __name__ == "__main__":

    _logger = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]
    hdulist = pyfits.open(fn)

    (x_eff, wl_map, medians, p_scale, p_skew, fm) = create_wlmap_from_skylines(hdulist)

    numpy.savetxt("intensity_profile", medians)

    # wp2 = trace_full_line(imgdata, x_start=601, y_start=526, window=5)
    # numpy.savetxt("skytrace_arc2.txt", wp2)

    pysalt.mp_logging.shutdown_logging(_logger)



























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
