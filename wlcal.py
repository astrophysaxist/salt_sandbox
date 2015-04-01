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


def match_line_catalogs(arc, ref, matching_radius):

    #
    # For each line in the ARC catalog, find the closest match in the 
    # reference line list
    #

    print arc
    print ref
    print matching_radius
    #matching_radius = 7
    kdtree = scipy.spatial.cKDTree(ref[:,0].reshape((-1,1)))
    nearest_neighbor, i = kdtree.query(x=arc[:,-1].reshape((-1,1)), 
                                       k=1, # only find 1 nearest neighbor
                                       p=1, # use linear distance
                                       distance_upper_bound=matching_radius)

    i = numpy.array(i)
    i[i>=ref.shape[0]] = 0
    print nearest_neighbor
    print i

    print "arc/ref",arc.shape, ref.shape
    print "nn/i",nearest_neighbor.shape, i.shape
    #
    # Now match both catalogs
    # 
    matched = numpy.zeros((arc.shape[0], (arc.shape[1]+ref.shape[1])))
    matched[:,:arc.shape[1]] = arc
    matched[:,arc.shape[1]:] = ref[i]
    
    #
    # Now eliminate all "matches" without a sufficiently close match
    # (i.e. where nearest_neighbordistance == inf)
    #
    print "before:",matched.shape
    good_match = numpy.isfinite(nearest_neighbor)
    matched = matched[good_match]
    print "after:",matched.shape

    return matched


    


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

def find_list_of_lines(spec, avg_width):

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


    # Combine all relevant data generated above for later use
    combined = numpy.empty((numpy.sum(real_peak), 5))
    combined[:,0] = x_pixels[real_peak]
    combined[:,1] = spec[real_peak]
    combined[:,2] = continuum[real_peak]
    combined[:,3] = continuum_noise[real_peak]
    combined[:,4] = s2n[real_peak]

    return combined

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

    blue_edge = rss.calc_bluewavelength() * mm_to_A
    red_edge = rss.calc_redwavelength() * mm_to_A
    wl_range = red_edge - blue_edge
    print "blue:", blue_edge
    print "red:", red_edge

    dispersion = (rss.calc_redwavelength()-rss.calc_bluewavelength())*mm_to_A/spec.shape[0]
    print "dispersion: A/px", dispersion
    
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
    lineinfo = find_list_of_lines(spec, avg_width)

    ############################################################################
    #
    # Now we have a full line-list with signal-to-noise ratios as brightness
    # indicators that we can use to select bright and/or faint lines.
    #
    ############################################################################

    # based on the wavelength model from RSS translate x-positions into wavelengths
    print dispersion
    print lineinfo[:,0]
    wl = lineinfo[:,0] * dispersion + blue_edge
    lineinfo = numpy.append(lineinfo, wl.reshape((-1,1)), axis=1)
    numpy.savetxt("linecal", lineinfo)

    #
    # Load linelist
    #
    lamp=hdulist[0].header['LAMPID'].strip().replace(' ', '')
    lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.txt" % lamp)
    lines = numpy.loadtxt(lampfile)
    print lines.shape
    print lines

    # Now select only lines that are in the estimated range of our ARC spectrum
    in_range = (lines[:,0] > numpy.min(wl)) & (lines[:,0] < numpy.max(wl))
    ref_lines = lines[in_range]
    print ref_lines

    #
    # Find average offset between arc lines and reference lines
    # 
    
    # only select strong lines in the ARC spectrum
    wl = wl[lineinfo[:,4] > 50]

    print ref_lines[:,0].reshape((-1,1)).T.shape
    print wl.reshape((-1,1)).shape

    numpy.savetxt("arc_lines", wl)
    numpy.savetxt("ref_lines", ref_lines[:,0])


    differences = ref_lines[:,0].reshape((-1,1)).T - wl.reshape((-1,1))
    print differences.shape
    numpy.savetxt("diffs", differences.flatten())

    # Now find the most frequently found offset
    # # Use kernel densities to avoid ambiguities between two adjacent bins

    # allow for as much as 20% shift in wavelength coverage
    # hopefully things are not THAT bad, but if: too bad for you
    max_overlap = 0.2 * wl_range 

    print wl_range

    count, bins = numpy.histogram(differences, bins=30, range=[-max_overlap,max_overlap])
    binwidth = bins[1] - bins[0]
    hist  = numpy.empty((count.shape[0],3))
    hist[:,0] = bins[:-1]
    hist[:,1] = bins[1:]
    hist[:,2] = count[:]
    numpy.savetxt("histogram", hist)

    # Now find the offset that allows to match the most lines
    hist_max = numpy.argmax(count)
    avg_shift = 0.5 * (bins[hist_max]+bins[hist_max+1])
    
    # This is the best shift to bring our line catalog in agreement 
    # with the catalog of reference lines
    print "NEED SHIFT of ~",avg_shift,"A"

    # Now improve the wavelength calibration of all found ARC lines by 
    # applying the shift we just found
    lineinfo[:,-1] += avg_shift

    #
    # Now match the two catalogs so we can derive an even better wavelength 
    # calibration
    #
    matched = match_line_catalogs(lineinfo, ref_lines, binwidth)
    numpy.savetxt("matched.lines", matched)
        

