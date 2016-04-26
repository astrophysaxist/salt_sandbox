#!/usr/bin/env python

import os, sys, numpy, math
import scipy.ndimage.filters
import scipy.stats
import bottleneck
import time

import pysalt
import logging


def find_line_edges(allskies, line_sigma=None):

    logger = logging.getLogger("FastFindEdges")

    print allskies.shape

    min_l, max_l = math.floor(numpy.min(allskies[:,0])), math.ceil(numpy.max(allskies[:,0]))
    range_l = max_l - min_l

    logger.info("wavelength range: %f -- %f" % (min_l, max_l))

    resolution = 0.25
    n_bins = (max_l-min_l)/resolution

    bins = numpy.arange(n_bins+1)*resolution+min_l
    # print bins[:5], bins[-5:]

    a = time.time()
    hist_sum, edges_sum = numpy.histogram(
        allskies[:,0], 
        bins=bins,
        weights=allskies[:,1]
        )
    hist_count, edges_count = numpy.histogram(
        allskies[:,0], 
        bins=bins,
        )
    logger.info("Integrating histograms took %f seconds" % ((time.time()-a)))


    combined = numpy.empty((n_bins, 20))
    combined[:,0] = bins[:-1]+0.5*resolution
    combined[:,1] = hist_sum
    combined[:,2] = hist_count

    avg_spec = combined[:,1] / combined[:,2]
    combined[:,3] = avg_spec
    bin_center = bins[:-1]+0.5*resolution

    # smoothed_gauss = scipy.ndimage.filters.gaussian_filter(
    #     input=combined[:,3],
    #     sigma=2,
    #     order=0,
    #     mode='constant', cval=0.,
    #     )

    # smoothed_median = scipy.ndimage.filters.median_filter(
    #     input=combined[:,3],
    #     size=5,
    #     mode='constant', cval=0.,
    #     )
    
    # combined[:,4] = smoothed_gauss
    # combined[:,5] = smoothed_median

    # # now compute slopes between datapoints
    # g_slopes = numpy.diff(smoothed_gauss)
    # m_slopes = numpy.diff(smoothed_median)
    # combined[:-1,6] = g_slopes
    # combined[:-1,7] = m_slopes


    # max_m_slopes = scipy.ndimage.filters.maximum_filter(
    #     input=m_slopes,
    #     size=10,
    #     mode='constant', cval=0.,
    #     )
    # combined[:-1, 8] = max_m_slopes
    # min_m_slopes = scipy.ndimage.filters.minimum_filter(
    #     input=m_slopes,
    #     size=10,
    #     mode='constant', cval=0.,
    #     )
    # combined[:-1, 8] = max_m_slopes
    # combined[:-1, 9] = min_m_slopes

    # direct_diff = numpy.diff(combined[:,3])
    # combined[:-1, 10] = direct_diff
    
    # min_m2 = scipy.ndimage.filters.minimum_filter(
    #     input=max_m_slopes,
    #     size=10,
    #     mode='constant', cval=0.,
    #     )
    # combined[:-1, 11] = min_m2

    # min_m3 = scipy.ndimage.filters.maximum_filter(
    #     input=m_slopes,
    #     size=3,
    #     mode='constant', cval=0.,
    #     )
    # combined[:-1, 12] = min_m3

    # min_m3b = scipy.ndimage.filters.gaussian_filter(
    #     input=m_slopes, #min_m3,
    #     sigma=2,
    #     order=0,
    #     mode='constant', cval=0.,
    #     )
    # combined[:-1, 13] = min_m3b

    # #
    # # Now find maxima and minima
    # # maximum: positive slope followed by negative slope
    # #
    # s1 = numpy.diff(min_m3b[:-1])
    # s2 = numpy.diff(min_m3b[1:])
    # print s1.shape, s2.shape
    # peak = ((s1 >= 0) & (s2 < 0)) | ((s1 < 0) & (s2 >= 0))
    # xxx = numpy.array(min_m3b)
    # print xxx.shape
    # xxx[1:][~peak] = 0

    # combined[:-1,14] = xxx

    # #
    # # Now figure out which of these peaks are significant
    # #
    # abs_peak_amplitude = min_m3b[peak]
    # numpy.savetxt("peak_amplitudes", abs_peak_amplitude)

    # for iteration in range(3):
    #     #var = bottleneck.nanvar(abs_peak_amplitude)
    #     #med = bottleneck.nanmedian(abs_peak_amplitude)
    #     q = scipy.stats.scoreatpercentile(abs_peak_amplitude,[16,50,84], (-1e7,1e7))
    #     med = q[1]
    #     sigma = 0.5*(q[2] - q[0])
    #     outlier = (abs_peak_amplitude > (med+3*sigma)) | (abs_peak_amplitude < (med-3*sigma))
    #     abs_peak_amplitude[outlier] = numpy.NaN
    #     print iteration, med, sigma
        
    # three_sigma = 3*sigma

    # continuum = scipy.ndimage.filters.median_filter(
    #     input=combined[:,3],
    #     size=100/resolution,
    #     mode='mirror',
    #     )
    # combined[:,15] = continuum

    # gain = 1.3
    # noise = numpy.sqrt(avg_spec * hist_count * gain)
    # signal_to_noise = (avg_spec - continuum) / noise

    # line_edge = peak & (numpy.fabs(min_m3b[1:-1]) > 3*sigma) 
    # yyy = numpy.array(smoothed_median)
    # yyy[1:][~line_edge] = 0
    # combined[:,16] = yyy

    # line_edge = peak & (numpy.fabs(min_m3b[1:-1]) > 3*sigma) & (signal_to_noise[1:-2] > 3)
    # yyy = numpy.array(smoothed_median)
    # yyy[1:][~line_edge] = 0
    # combined[:,17] = yyy
 
    # numpy.savetxt("combined.xxx", combined)


    gain, readnoise = 1.3, 3

    # add some padding to avoid querying non-existant data
    fw = 50
    padded = numpy.empty((avg_spec.shape[0]+2*fw))
    padded[:] = numpy.NaN
    padded[fw:-fw] = avg_spec
    continuum = numpy.array([
        bottleneck.nanmedian(padded[i-fw:i+fw]) for i in range(fw, avg_spec.shape[0]+fw)])
    continuum[numpy.isnan(continuum)] = 0.

    if (line_sigma == None):
    
        #
        # find and isolate lines
        #

        spec = avg_spec
        peak = numpy.empty(spec.shape, dtype=numpy.bool)
        peak[:] = False
        peak[1:-1] = (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])

        spec_noise = numpy.sqrt(
            (avg_spec * hist_count * gain)+ (readnoise**2*hist_count)
        ) #/ hist_count / gain

        #continuum_noise = numpy.sqrt(numpy.fabs(continuum*gain)+(readnoise**2*avg_width)) / (2*avg_width)
        #numpy.savetxt("continuum_noise", continuum_noise)

        real_peak = peak & ((spec-continuum) > 3*spec_noise) #& (spec > continuum+100)

        linecomb = numpy.empty((avg_spec.shape[0],5))
        linecomb[:,0] = bin_center
        linecomb[:,1] = avg_spec
        linecomb[:,2] = spec_noise
        linecomb[:,3] = continuum
        linecomb[:,4] = avg_spec
        linecomb[:,4][~real_peak] = 0.

        numpy.savetxt("linecomb.xxx", linecomb)


        # now compute a line profile, stacking the data in the vicinity of each line
        hi_res = 0.1 * resolution

        superskies = allskies.reshape((1,-1,2)).repeat(10,axis=0)
        pixelsize=0.5
        dl = numpy.linspace(0,pixelsize,10,endpoint=False).reshape((-1,1)).repeat(allskies.shape[0], axis=1)
        print dl.shape

        print superskies[:,0,:]

        #print dl
        print allskies.shape, superskies.shape
        superskies[:,:,0] += dl
        print superskies[:,0,:]

        #superskies = superskies.reshape((-1,2))
        #print superskies[:15,:]

        print superskies.shape
        superskies = superskies.reshape((-1,2))

        good_lines = real_peak & (bin_center > min_l+0.1*range_l) & (bin_center < max_l-0.2*range_l)
        line_centers = (bins[:-1]+0.5*resolution)[real_peak]
        s2n = ((spec-continuum)/spec_noise)[good_lines]
        print s2n
        s2n_sort = numpy.argsort(s2n)[::-1]
        print s2n[s2n_sort]
        good_line_centers = line_centers[s2n_sort]

        full_count, full_sum = None, None
        line_profile_width = 20.
        n_hires_bins = (2*line_profile_width) / hi_res + 1.
        hires_bins = numpy.arange(n_hires_bins+1)*hi_res - line_profile_width
        print hires_bins

        for line in good_line_centers: #[6466.625]: #line_centers:

            #print (superskies[:10,0]-line)
            l = superskies[:,0]-line
            good_l = (l>-20) & (l<20)
            print line, numpy.sum(good_l)
            print l

            _hist_sum, edges_sum = numpy.histogram(
                l, 
                bins=hires_bins,
                weights=superskies[:,1]
            )
            _hist_count, edges_count = numpy.histogram(
                l, 
                bins=hires_bins,
            )
            print line, numpy.sum(hist_count)

            if (full_count == None):
                full_count = _hist_count
            else:
                full_count += _hist_count

            if (full_sum == None):
                full_sum = _hist_sum
            else:
                full_sum += _hist_sum

            #print line
            pass

        print full_sum.shape, full_count.shape
        hires_spec = full_sum / full_count
        hires_center = hires_bins[:-1]+0.5*hi_res
        print hires_spec.shape, hires_center.shape
        numpy.savetxt("lineprofile", 
                      numpy.append(hires_center.reshape((-1,1)), 
                                   hires_spec.reshape((-1,1)), axis=1)
        )

        min_i = numpy.min(hires_spec)
        max_i = numpy.max(hires_spec)
        halfmax = min_i + 0.5*(max_i-min_i)
        print halfmax
        left_fwhm = numpy.min(hires_center[hires_spec > halfmax])
        right_fwhm = numpy.max(hires_center[hires_spec > halfmax])
        print left_fwhm, right_fwhm, right_fwhm-left_fwhm


        # Now we have a FWHM measurement for the line profile
        # convert that into a line sigma 
        line_sigma = (right_fwhm-left_fwhm) / 2.355

    convolved_spec = scipy.ndimage.filters.gaussian_filter(
        input=avg_spec,
        sigma=line_sigma,
        order=0,
        mode='constant', cval=0.,
        )
    
    peak = numpy.empty(convolved_spec.shape, dtype=numpy.bool)
    peak[:] = False
    peak[1:-1] = (convolved_spec[1:-1] > convolved_spec[:-2]) & (convolved_spec[1:-1] > convolved_spec[2:])

    spec_noise = numpy.sqrt(
        (convolved_spec * hist_count * gain)+ (readnoise**2*hist_count)
    ) / gain

    #continuum_noise = numpy.sqrt(numpy.fabs(continuum*gain)+(readnoise**2*avg_width)) / (2*avg_width)
    #numpy.savetxt("continuum_noise", continuum_noise)

    real_peak = peak & ((convolved_spec-continuum) > 3*spec_noise) #& (spec > continuum+100)
    
    linecomb = numpy.empty((avg_spec.shape[0],5))
    linecomb[:,0] = bin_center
    linecomb[:,1] = convolved_spec
    linecomb[:,2] = spec_noise
    linecomb[:,3] = continuum
    linecomb[:,4] = convolved_spec
    linecomb[:,4][~real_peak] = 0.

    line_hwhm = line_sigma * 2.355 / 2.
    
    edges = numpy.append(
        bin_center[real_peak]-line_hwhm,
        bin_center[real_peak]+line_hwhm,
        )

    numpy.savetxt("linecomb.yyy", linecomb)

    return edges


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()

    allskies = numpy.loadtxt(sys.argv[1])
    find_line_edges(allskies, line_sigma=2.75)

    pysalt.mp_logging.shutdown_logging(logger_setup)
