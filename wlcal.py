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

import bottleneck

from PySpectrograph.Models import RSSModel




import pysalt

def extract_arc_spectrum(hdulist, line=None, avg_width=10):

    # Find central line based on the dimensions
    center = hdulist['SCI'].data.shape[0] / 2 if line == None else line

    # average over a couple of lines
    spec = hdulist['SCI'].data[center-avg_width:center+avg_width,:]
    print spec.shape

    avg_spec = numpy.average(spec, axis=0)
    print avg_spec.shape

    numpy.savetxt("arcspec.dat", avg_spec)
    return avg_spec


mm_to_A = 10e6

if __name__ == "__main__":
    filename = sys.argv[1]

    hdulist = pyfits.open(filename)
    hdulist.info()
    line = int(sys.argv[2])

    avg_width = 10
    spec = extract_arc_spectrum(hdulist, line, avg_width)

    binx, biny = pysalt.get_binning(hdulist)

    hdr = hdulist[0].header
    rss = RSSModel.RSSModel(
        grating_name=hdr['GRATING'], 
        gratang=hdr['GR-ANGLE'], #45, 
        camang=hdr['CAMANG'], #45, 
        slit=1.0, 
        xbin=binx, ybin=biny, 
        xpos=-0.30659999999999998, ypos=0.0117, wavelength=None)

    central_wl = rss.calc_centralwavelength() * mm_to_A
    print central_wl

    print "blue:", rss.calc_bluewavelength() * mm_to_A
    print "red:", rss.calc_redwavelength() * mm_to_A

    print "dispersion: A/px", (rss.calc_redwavelength()-rss.calc_bluewavelength())*mm_to_A/spec.shape[0]
    
    #print "ang.dispersion:", rss.calc_angdisp(rss.beta())
    print "ang.dispersion:", rss.calc_angdisp(-rss.beta())

    pixelsize = 15e-6
    print "lin.dispersion:", rss.calc_lindisp(rss.beta())
    print "lin.dispersion:", rss.calc_lindisp(rss.beta()) / (mm_to_A*pixelsize)

    print "resolution @ central w.l.:", rss.calc_resolution(
        w=rss.calc_centralwavelength(), 
        alpha=rss.alpha(), 
        beta=-rss.beta())
    
    print "resolution element:", rss.calc_resolelement(rss.alpha(), -rss.beta()) * mm_to_A

    #
    # Now find a list of strong lines
    #
    max_intensity = numpy.max(spec)
    x_pixels = numpy.arange(spec.shape[0]) # FITS starts counting pixels at 1

    # blockaverage spectrum
    w=5
    blkavg = numpy.array([
        numpy.average(spec[i-w:i+w]) for i in range(spec.shape[0])])
    numpy.savetxt("blkavg_spec.dat", blkavg)

    #
    # median-filter spectrum to get continuum
    #
    continuum = scipy.ndimage.filters.median_filter(spec, 25, mode='reflect')
    numpy.savetxt("continuum_scipy", continuum)

    _med, _std = 0, numpy.max(spec)/2
    for i in range(3):
        maybe_continuum = (spec > _med-2*_std) & (spec < _med+2*_std)
        _med = bottleneck.nanmedian(spec[maybe_continuum])
        _std = bottleneck.nanstd(spec[maybe_continuum])
    # flag all pixels that are likely lines
    spec_nolines = numpy.array(spec)
    spec_nolines[~maybe_continuum] = numpy.NaN
    # now median_filter over the continuum
    fw = 25
    continuum = numpy.array([
        bottleneck.nanmedian(spec_nolines[i-fw:i+fw]) for i in range(spec_nolines.shape[0])])
    continuum[numpy.isnan(continuum)] = 0.
    numpy.savetxt("continuum", continuum)


    peak = numpy.array(
        [(True if blkavg[i]>blkavg[i-1] and blkavg[i]>blkavg[i+1] else False)
         for i in range(blkavg.shape[0])])
    numpy.savetxt("peaks_yesno", peak)
    numpy.savetxt("wl_peaks", numpy.append(
        x_pixels[peak].reshape((-1,1)), spec[peak].reshape((-1,1)), axis=1))
    
    # Now reject all peaks that are not significantly over the estimated background noise
    readnoise = 2. # raw data: ron = 3.3, gain = 1.6 --> RON=2 ADU
    continuum_noise = numpy.sqrt(continuum*readnoise*2*avg_width) / (2*avg_width)
    numpy.savetxt("continuum_noise", continuum_noise)

    # require at least 3 sigma over background noise
    real_peak = peak & ((spec-continuum) > 3*continuum_noise) #& (spec > continuum+100)
    numpy.savetxt("wl_real_peaks", numpy.append(
        x_pixels[real_peak].reshape((-1,1)), spec[real_peak].reshape((-1,1)), axis=1))

    # compute full S/N for each pixels
    s2n = spec / (numpy.sqrt(spec*readnoise*2*avg_width) / (2*avg_width))
    numpy.savetxt("wl_real_peaks.sn", numpy.append(
        x_pixels[real_peak].reshape((-1,1)), s2n[real_peak].reshape((-1,1)), axis=1))
    
    ############################################################################
    #
    # Now we have a full line-list with signal-to-noise ratios as brightness
    # indicators that we can use to select bright and/or faint lines.
    #
    ############################################################################


    #print x_pixels

