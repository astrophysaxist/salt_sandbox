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

def match_line_catalogs(arc, ref, matching_radius, verbose=False,
                        col_ref=0, col_arc=-1):

    logger = logging.getLogger("MatchLineCat")

    #
    # For each line in the ARC catalog, find the closest match in the 
    # reference line list
    #

    logger.debug("#ARCs: %d  -- #REF: %d  --  MatchRadius: %.2f" % (
        arc.shape[0], ref.shape[0], matching_radius))
    if (verbose):
        numpy.savetxt("mlc.verbose", arc)
        
    # print arc
    # print ref
    # print matching_radius
    #matching_radius = 7
    kdtree = scipy.spatial.cKDTree(ref[:,col_ref].reshape((-1,1)))
    nearest_neighbor, i = kdtree.query(x=arc[:,col_arc].reshape((-1,1)), 
                                       k=1, # only find 1 nearest neighbor
                                       p=1, # use linear distance
                                       distance_upper_bound=matching_radius)

    i = numpy.array(i)
    i[i>=ref.shape[0]] = 0
    #print nearest_neighbor
    #print i

    #print "arc/ref",arc.shape, ref.shape
    #print "nn/i",nearest_neighbor.shape, i.shape
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
    #print "before:",matched.shape
    good_match = numpy.isfinite(nearest_neighbor)
    matched = matched[good_match]
    #print "after:",matched.shape

    logger.debug("Found %3d matched lines" % (matched.shape[0]))

    return matched


    


def extract_arc_spectrum(hdulist, line=None, avg_width=10):

    logger = logging.getLogger("ExtractSpec")

    # Find central line based on the dimensions
    logger.info("Extracting average of +/- %d lines around y = %4d" % (
            avg_width, line))
    center = hdulist['SCI'].data.shape[0] / 2 if line == None else line

    # average over a couple of lines
    spec = hdulist['SCI'].data[center-avg_width:center+avg_width,:]
    #print spec.shape

    avg_spec = numpy.average(spec, axis=0)
    #print avg_spec.shape

    logger.debug("done here!")
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
    s2n = (spec - continuum) / (numpy.sqrt(spec*readnoise*2*avg_width) / (2*avg_width))
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



def compute_wavelength_solution(matched, max_order=3):

    logger = logging.getLogger("WLS_polyfit")

    #
    # In the matched line list we have both X-column coordinates and 
    # vacuum wavelengths. Thus we can establish a polynomial connection 
    # between these two. This is what we are after
    #

    logger.debug("Running a polynomial fit to %4d data points" % (matched.shape[0]))
    numpy.savetxt("matched.for_final_fit", matched)
    ret = numpy.polynomial.polynomial.polyfit(
        x=matched[:,0],
        y=matched[:,6],
        deg=max_order,
        full=True,
        w=matched[:,4], #use S/N for weighting
        )

    coeffs, rest = ret
    residuals, rank, singular_values, rcond = rest

    logger.info("Fit coeffs: %s" % (" ".join(["%.6e" % c for c in coeffs])))
    logger.debug("residuals: %e" % (residuals))
    logger.debug("rank: %d" % (rank))
    logger.debug("singular_values: %s" % (" ".join(["%e" % sv for sv in singular_values])))
    logger.debug("rcond: %f" % (rcond))

    #print ret

    return coeffs

    


def find_matching_lines(ref_lines, lineinfo, 
                        rss,
                        dispersion, central_wavelength, reference_pixel_x,
                        matching_radius,
                        s2n_cutoff=30):
    
    #print

    logger = logging.getLogger("FindMatchingLines")
    logger.debug("Using d=%.6f A/px, central wavelength: %10.4f @ %8.2f px" % (
        dispersion, central_wavelength, reference_pixel_x))

    blue_edge = rss.calc_bluewavelength() * mm_to_A
    red_edge = rss.calc_redwavelength() * mm_to_A
    wl_range = red_edge - blue_edge
    logger.debug("Estimated wavelength range: %f -- %f A" % (blue_edge, red_edge))

    #
    # Find average offset between arc lines and reference lines
    # 
    
    # compute wavelength based on central wavelength and 
    wl = (lineinfo[:,0]-reference_pixel_x) * dispersion + central_wavelength

    # only select strong lines in the ARC spectrum
    wl = wl[lineinfo[:,4] > s2n_cutoff]

    #print ref_lines[:,0].reshape((-1,1)).T.shape
    #print wl.reshape((-1,1)).shape
    #numpy.savetxt("arc_lines", wl)
    #numpy.savetxt("ref_lines", ref_lines[:,0])


    differences = ref_lines[:,0].reshape((-1,1)).T - wl.reshape((-1,1))
    #print differences.shape
    numpy.savetxt("diffs", differences.flatten())

    # Now find the most frequently found offset
    # # Use kernel densities to avoid ambiguities between two adjacent bins

    # allow for as much as 20% shift in wavelength coverage
    # hopefully things are not THAT bad, but if: too bad for you
    max_overlap = 0.2 * wl_range 

    #print wl_range

    n_bins = 2*max_overlap/matching_radius
    logger.debug("Using %d bins, each%.2f A wide, to search for matches" % (n_bins, matching_radius))
    count, bins = numpy.histogram(differences, bins=n_bins, range=[-max_overlap,max_overlap])
    binwidth = bins[1] - bins[0]
    hist  = numpy.empty((count.shape[0],3))
    hist[:,0] = bins[:-1]
    hist[:,1] = bins[1:]
    hist[:,2] = count[:]
    numpy.savetxt("histogram__%.4f" % (dispersion), hist)

    # Now find the offset that allows to match the most lines
    hist_max = numpy.argmax(count)
    avg_shift = 0.5 * (bins[hist_max]+bins[hist_max+1])
    
    # This is the best shift to bring our line catalog in agreement 
    # with the catalog of reference lines
    logger.debug("DISPERSION %.4f --> NEED SHIFT of ~ %.2f A" % (dispersion, avg_shift))

    # Now improve the wavelength calibration of all found ARC lines by 
    # applying the shift we just found
    lineinfo[:,-1] += avg_shift

    #
    # Now match the two catalogs so we can derive an even better wavelength 
    # calibration
    #
    matched = match_line_catalogs(lineinfo, ref_lines, matching_radius)
    numpy.savetxt("matched.lines.%.4f" % (dispersion), matched)

    logger.info("Trying dispersion %8.4f A/px   ===>   shift: %8.2fA, #matches: %3d" % (
            dispersion, avg_shift, matched.shape[0]))

    return matched




def find_wavelength_solution(filename, line):

    logger = logging.getLogger("FindWLS")

    hdulist = pyfits.open(filename)
    #hdulist.info()

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
    #print central_wl

    blue_edge = rss.calc_bluewavelength() * mm_to_A
    red_edge = rss.calc_redwavelength() * mm_to_A
    wl_range = red_edge - blue_edge
    #print "blue:", blue_edge
    #print "red:", red_edge

    dispersion = (rss.calc_redwavelength()-rss.calc_bluewavelength())*mm_to_A/spec.shape[0]
    #print "dispersion: A/px", dispersion
    
    #print "ang.dispersion:", rss.calc_angdisp(rss.beta())
    #print "ang.dispersion:", rss.calc_angdisp(-rss.beta())

    pixelsize = 15e-6
    #print "lin.dispersion:", rss.calc_lindisp(rss.beta())
    #print "lin.dispersion:", rss.calc_lindisp(rss.beta()) / (mm_to_A*pixelsize)

    #print "resolution @ central w.l.:", rss.calc_resolution(
    #     w=rss.calc_centralwavelength(), 
    #     alpha=rss.alpha(), 
    #     beta=-rss.beta())
    
    # print "resolution element:", rss.calc_resolelement(rss.alpha(), -rss.beta()) * mm_to_A

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
    #print dispersion
    #print lineinfo[:,0]
    wl = lineinfo[:,0] * dispersion + blue_edge
    lineinfo = numpy.append(lineinfo, wl.reshape((-1,1)), axis=1)
    numpy.savetxt("linecal", lineinfo)

    #
    # Load linelist
    #
    lamp=hdulist[0].header['LAMPID'].strip().replace(' ', '')
    lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.txt" % lamp)
    _, fn_only = os.path.split(lampfile)
    logger.info("Reading calibration line wavelengths from data->%s" % (fn_only))
    logger.debug("Full path to lamp line list: %s" % (lampfile))
    #lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.wav" % lamp)
    #lampfile=pysalt.get_data_filename("pysalt$data/linelists/Ar.salt")
    #lampfile="Ar.lines"
    lines = numpy.loadtxt(lampfile)
    #print lines.shape
    #print lines

    # Now select only lines that are in the estimated range of our ARC spectrum
    in_range = (lines[:,0] > numpy.min(wl)) & (lines[:,0] < numpy.max(wl))
    ref_lines = lines[in_range]
    logger.debug("Found these lines for fitting:\n%s" % (
            "\n".join(["%10.4f" % l for l in ref_lines[:,0]])))
    #print ref_lines

    #
    # Match lines between ARC spectrum and reference line list, 
    # allowing for a small uncertainty in dispersion
    #

    reference_pixel_x = 0.5 * spec.shape[0]

    # dispersion was calculated above
    # central wavelength 
    central_wavelength = 0.5 * (blue_edge + red_edge)
    max_dispersion_error = 0.10 # +/- 10% should be plenty
    dispersion_search_steps = 0.01 # vary dispersion in 1% steps
    n_dispersion_steps = (max_dispersion_error / dispersion_search_steps) * 2 + 1

    # compute what dispersion factors we are trying 
    trial_dispersions = numpy.linspace(1.0-max_dispersion_error, 
                                      1.0+max_dispersion_error,
                                      n_dispersion_steps) * dispersion 
    n_matches = numpy.zeros((trial_dispersions.shape[0]))
    matched_cats = [None] * trial_dispersions.shape[0]

    for idx, _dispersion in enumerate(trial_dispersions):

        # compute dispersion including the correction
        #_dispersion = dispersion * disp_factor

        # copy the original line info so we don't accidently overwrite important data
        _lineinfo = numpy.array(lineinfo)

        # find optimal line shifts and match lines 
        # consider lines within 5A of each other matches
        # --> this most likely will depend on spectral resolution and binning
        matched_cat = find_matching_lines(ref_lines, _lineinfo, 
                                          rss,
                                          _dispersion, central_wavelength,
                                          reference_pixel_x,
                                          matching_radius = 5, 
        )

        # Save results for later picking of the best one
        n_matches[idx] = matched_cat.shape[0]
        matched_cats[idx] = matched_cat

        numpy.savetxt("dispersion_scale_%.3f" % (_dispersion), matched_cat)

    # Find the solution with the most matched lines
    n_max = numpy.argmax(n_matches)
    
    #print

    #print "most matched lines:", n_matches[n_max],
    #print "best dispersion: %f" % (trial_dispersions[n_max])
    logger.info("Choosing best solution: %4d for dispersion %8.4f A/px" % (
            n_matches[n_max], trial_dispersions[n_max]))

    numpy.savetxt("matchcount", numpy.append(trial_dispersions.reshape((-1,1)),
                                             n_matches.reshape((-1,1)),
                                             axis=1))

    matched = matched_cats[n_max]
    numpy.savetxt("matched.lines.best", matched)
    
    # print "***************************\n"*5
    # print lineinfo.shape

    logger.info("Computing an analytical wavelength calibration...")
    wls = compute_wavelength_solution(matched, max_order=3)

    # Now we have a best-match solution
    # Match lines again to see what the RMS is - use a small matching radius now
    
    _linelist = numpy.array(lineinfo)
    numpy.savetxt("lineinfo.final", _linelist)
    _linelist[:,1] = _linelist[:,0]
    _linelist[:,0] = numpy.polynomial.polynomial.polyval(_linelist[:,0], wls)

    # compute wl with polyval
    _wl = lineinfo[:,0]
    numpy.savetxt("final.1", _wl)

    numpy.savetxt("final.2", numpy.polynomial.polynomial.polyval(_wl, wls))
    numpy.savetxt("final.3", _wl*wls[1]+wls[0])

    final_match = match_line_catalogs(_linelist, ref_lines, matching_radius=5, verbose=True,
                                      col_arc=0, col_ref=0)
    numpy.savetxt("matched.cat.final", final_match)

    # Apply WLS to FITS header
    
    hdr = hdulist['SCI'].header
    hdr['CD1_1'] = wls[1]
    hdr['CRVAL1'] = wls[0]
    hdr['CRPIX1'] = 1.0
    hdr['CTYPE1'] = 'PIXEL' #'LAMBDA'

    # set the right reference line
    hdr['CRPIX2'] = 1.
    hdr['CRVAL2'] = line
    hdr['CTYPE2'] = "PIXEL"
    hdr['CD2_2'] = 1.0

    for hdrname in ['CDELT1', 'CDELT2']:
        if (hdrname in hdr): del hdr[hdrname]

    # Write a wavelength calibrated strip spectrum
    strip = numpy.repeat(spec.reshape((-1,1)), 100, axis=1)
    # print spec.shape, strip.shape
    hdulist['SCI'].data = strip.T
    if (os.path.isfile("test_out.fits")): os.remove("test_out.fits")
    hdulist.writeto("test_out.fits", clobber=True)

    # Also save the original spectrum as text file
    spec_x = numpy.polynomial.polynomial.polyval(numpy.arange(spec.shape[0]), wls).reshape((-1,1))
    spec_combined = numpy.append(spec_x, spec.reshape((-1,1)), axis=1)
    numpy.savetxt("spec.calib", spec_combined)
    

    return {
        'spec': spec,
        'spec_combined': spec_combined,
        'linelist_ref': lines,
        'linelist_arc': _linelist,
        }




if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()

    filename = sys.argv[1]
    line = int(sys.argv[2])

    wls_data = find_wavelength_solution(filename, line)

    pysalt.mp_logging.shutdown_logging(logger_setup)
