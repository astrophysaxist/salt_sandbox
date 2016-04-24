#!/usr/bin/env python

import os, sys
import numpy
from scipy.ndimage.filters import median_filter
import bottleneck
import scipy.interpolate
numpy.seterr(divide='ignore', invalid='ignore')
import itertools
import math
import matplotlib

from astropy.io import fits

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


import matplotlib.pyplot as pl

#
# Line info columns:
#
#  0: position in pixels
#  1: peak line flux
#  2: continuum flux
#  3: continuum noise
#  4: signal-to-noise ( = (peak - continuum)/continuum_noise )
#  5: position in angstroems
#
lineinfo_cols = ["PIXELPOS",
                 "FLUX",
                 "CONTINUUM",
                 "CONTINUUM_NOISE",
                 "S2N",
                 "WAVELENGTH"]
lineinfo_colidx = {}
for idx,name in enumerate(lineinfo_cols):
    lineinfo_colidx[name] = idx





def rssmodelwave(grating,grang,artic,cbin,cols):
#
# Taken from Ken Nordsieck's specpolmap.py
# https://github.com/saltastro/SALTsandbox/blob/master/polSALT/polsalt/specpolmap.py
#

    #
    # Define some constants - taken from 
    # https://github.com/saltastro/SALTsandbox/blob/master/polSALT/polsalt/data/gratings.txt
    #
    # from ArcModel.xlsx 7/10/2012, grating-alignment.xlsx/120414
    # Grating	l/mm	 Gam0      y0	  dy/arang
    # PG0300	 300.00	 0.000    2055.7	 0.307        
    # PG0900	 903.89	-0.265    2042.2	-0.156
    # PG1300	1299.76	-0.265    2054.3	 0.198
    # PG1800	1801.89	-0.265    2050.1	-0.067
    # PG2300	2302.60	-0.265    2059.5	 0.127
    # PG3000	3000.55	-0.265    2060.6	-0.151
    __rss_gratings = {
        'PG0300': (300.00 ,  0.000, 2055.7,  0.307),      
        'PG0900': (903.89 , -0.265, 2042.2, -0.156),
        'PG1300': (1299.76, -0.265, 2054.3,  0.198),
        'PG1800': (1801.89, -0.265, 2050.1, -0.067),
        'PG2300': (2302.60, -0.265, 2059.5,  0.127),
        'PG3000': (3000.55, -0.265, 2060.6, -0.151),
    }

    # more constants, from
    # https://github.com/saltastro/SALTsandbox/blob/master/polSALT/polsalt/data/spec.txt
    # fixed spectrograph constants
    Grat0  = 1.407798403  #	deg
    Home0  = -0.063025809 #deg
    ArtErr = -4.2E-05  #	(1+Err)*.75 deg
    T2Con  = -5.00	
    T3Con  = -1.00
    Fcama5 = -0.0023		
    Fcama4 = 0.0365
    Fcama3 = -0.2100
    Fcama2 = 0.5061
    Fcama1 = -0.1861
    Fcamb  = 328.697 #   mm

        
    # #   compute wavelengths from model (this can probably be done using pyraf spectrograph model)
    # spec=np.loadtxt(datadir+"spec.txt",usecols=(1,))
    # Grat0,Home0,ArtErr,T2Con,T3Con=spec[0:5]

    FCampoly = [Fcama5, Fcama4, Fcama3, Fcama2, Fcama1, Fcamb] #spec[5:11]
    
    # grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    # grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)

    lmm, grgam0, y0, dy = __rss_gratings[grating]

    # grnum = np.where(grname==grating)[0][0]
    # lmm = grlmm[grnum]
    alpha_r = numpy.radians(grang+Grat0)
    beta0_r = numpy.radians(artic*(1+ArtErr)+Home0)-alpha_r
    gam0_r = numpy.radians(grgam0)
    lam0 = 1e7*numpy.cos(gam0_r)*(numpy.sin(alpha_r) + numpy.sin(beta0_r))/lmm
    ww = lam0/1000. - 4.
    fcam = numpy.polyval(FCampoly,ww)
    disp = (1e7*numpy.cos(gam0_r)*numpy.cos(beta0_r)/lmm)/(fcam/.015)
    dfcam = 3.162*disp*numpy.polyval([FCampoly[x]*(5-x) for x in range(5)],ww)
    T2 = -0.25*(1e7*numpy.cos(gam0_r)*numpy.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*3162.*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2 
    T1 = 3162.*disp + 3*T3
    X = (numpy.array(range(cols))+1-cols/2)*cbin/3162.
    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X




def match_line_catalogs(arc, ref, matching_radius, verbose=False,
                        col_ref=0, col_arc=-1, dumpfile=None):

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

    # i is the index with the closest match
    i = numpy.array(i)
    bad_matches = (i>=ref.shape[0])
    i[bad_matches] = 0
    #print nearest_neighbor
    #print i

    #print "arc/ref",arc.shape, ref.shape
    #print "nn/i",nearest_neighbor.shape, i.shape

    #
    # Now merge both catalogs, appending the reference catalog 
    # to the ARC source catalog
    # 
    matched = numpy.zeros((arc.shape[0], (arc.shape[1]+ref.shape[1])))
    matched[:,:arc.shape[1]] = arc
    matched[:,arc.shape[1]:] = ref[i]
    #print "XXXXXXXXXXXX\n"*3,ref.shape, ref[i].shape, i.shape, bad_matches.shape
    #print bad_matches
    numpy.savetxt("matched_raw", matched)
    numpy.savetxt("matched_bad", bad_matches)
    #sys.exit(0)

    #print "XXXX\n"*3
    matched = matched[~bad_matches]

    #
    # Now eliminate all "matches" without a sufficiently close match
    # (i.e. where nearest_neighbordistance == inf)
    #

    df = None
    if (not dumpfile == None):
        df = open(dumpfile, "w")
#        with open(dumpfile, "w") as df:
        numpy.savetxt(df, matched, "%8.3f")
        
    #print "before:",matched.shape
    #good_match = numpy.isfinite(nearest_neighbor)
    #matched = matched[good_match]

    if (not df == None):
        print >>df, "\n\n\n\n\n"
        numpy.savetxt(df, matched, "%8.3f")
        print >>df, "\n\n\n\n\n"
        numpy.savetxt(df, nearest_neighbor)
        df.close()
    #print "after:",matched.shape

    logger.debug("Found %3d matched lines" % (matched.shape[0]))

    return matched


    


def extract_arc_spectrum(hdulist, line=None, avg_width=20):

    logger = logging.getLogger("ExtractSpec")

    # Find central line based on the dimensions
    logger.debug("Extracting average of +/- %d lines around y = %4d" % (
            avg_width, line))
    center = hdulist['SCI'].data.shape[0] / 2 if line == None else line

    # average over a couple of lines
    binx, biny = pysalt.get_binning(hdulist)
    binned_width = avg_width / biny

    spec = hdulist['SCI'].data[center-avg_width:center+avg_width,:]
    #print spec.shape

    avg_spec = numpy.average(spec, axis=0)
    #print avg_spec.shape

    logger.debug("done here!")
    numpy.savetxt("arcspec.dat", avg_spec)
    return avg_spec




mm_to_A = 10e6

def find_list_of_lines(spec, readnoise=2, gain=1, avg_width=1, pre_smooth=None):

    """

    Find a list of spectral emission lines for a given 1-D spectrum

    Parameters:
    -----------

    spec : numpy.ndarray, 1-d

        Input spectrum; has to be a 1-D spectrum, i.e. spec.shape needs to 
        be (N,), not (N,1).

    readnoise : float (default: 2.0)

        Readnoise of the data; necessary for properly estimating the noise in 
        the continnum and subsequently the signal-to-noise ratio of each 
        detection.

    avg_width : int

        If the input spectrum is a combination of multiple detector rows, this 
        needs to specified here.

    Returns
    -------

    linecat : numpy.array

    """

    logger = logging.getLogger("FindLineList")

    max_intensity = numpy.nanmax(spec)
    logger.debug("Found max intensity %.1f in %4d pixels" % (max_intensity, spec.shape[0]))

    x_pixels = numpy.arange(spec.shape[0]) # FITS starts counting pixels at 1

    #
    # median-filter spectrum to get continuum
    #
    logger.debug("Median-filtering spectrum to estimate continuum")
    continuum = scipy.ndimage.filters.median_filter(spec, 25, mode='reflect')
    numpy.savetxt("continuum_scipy", continuum)

    logger.debug("Estimating continuum using wide median-filter to exclude lines")
    _med, _std = 0, numpy.nanmax(spec)/2
    for i in range(3):
        maybe_continuum = (spec > _med-2*_std) & (spec < _med+2*_std)
        _med = bottleneck.nanmedian(spec[maybe_continuum])
        _std = bottleneck.nanstd(spec[maybe_continuum])
    # flag all pixels that are likely lines
    spec_nolines = numpy.array(spec)
    spec_nolines[~maybe_continuum] = numpy.NaN
    # now median_filter over the continuum
    fw = 50

    numpy.savetxt("spec_nolines", spec_nolines)

    # add some padding to avoid querying non-existant data
    padded = numpy.empty((spec_nolines.shape[0]+2*fw))
    padded[:] = numpy.NaN
    padded[fw:-fw] = spec_nolines
    continuum = numpy.array([
        bottleneck.nanmedian(padded[i-fw:i+fw]) for i in range(fw, spec_nolines.shape[0]+fw)])

    # continuum = numpy.array([
    #     bottleneck.nanmedian(spec_nolines[i-fw:i+fw]) for i in range(spec_nolines.shape[0])])
    continuum[numpy.isnan(continuum)] = 0.
    numpy.savetxt("continuum", continuum)

    #
    # Search for peaks
    # Definition of a peak: A value higher than the two neighboring values
    #
    logger.debug("Starting to search for lines")

    if (pre_smooth > 0):
        spec = scipy.ndimage.filters.gaussian_filter(
            input=spec, sigma=pre_smooth, 
            order=0, output=None, 
            mode='constant', cval=0.0, 
#            truncate=3.0,
        )
        numpy.savetxt("spec_presmoothed", spec)

    peak = numpy.empty(spec.shape, dtype=numpy.bool)
    peak[:] = False
    peak[1:-1] = (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])
    # peak = numpy.array(
    #     [(spec[i] if spec[i]>spec[i-1] and spec[i]>spec[i+1] else 0)
    #      for i in range(1,spec.shape[0]-1)])
    # peak = numpy.array(
    #     [(True if spec[i]>spec[i-1] and spec[i]>spec[i+1] else False)
    #      for i in range(spec.shape[0])])
    # print peak
    
    peak_values = numpy.array(spec)
    peak_values[~peak] = -1
    numpy.savetxt("peaks_values", peak_values)
    
    numpy.savetxt("peaks_yesno", peak)
    numpy.savetxt("wl_peaks", numpy.append(
#        x_pixels[peak].reshape((-1,1)), spec[peak].reshape((-1,1)), axis=1))
        x_pixels[peak].reshape((-1,1)), spec[peak].reshape((-1,1)), axis=1))
    
    # Now reject all peaks that are not significantly over the estimated background noise
    # number of electrons from source/sky
    continuum_noise = numpy.sqrt(
        (spec*gain*avg_width) + (readnoise**2*avg_width)
    ) / avg_width
    #continuum_noise = numpy.sqrt(numpy.fabs(continuum*gain)+(readnoise**2*avg_width)) / (2*avg_width)
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
    combined = numpy.empty((numpy.sum(real_peak), len(lineinfo_cols)))
    combined[:,0] = x_pixels[real_peak]
    combined[:,1] = spec[real_peak]
    combined[:,2] = continuum[real_peak]
    combined[:,3] = continuum_noise[real_peak]
    combined[:,4] = s2n[real_peak]
    combined[:,5] = x_pixels[real_peak]

    # numpy.savetxt("detectlines.debug", combined)

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

    logger.debug("Fit coeffs: %s" % (" ".join(["%.6e" % c for c in coeffs])))
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
                        s2n_cutoff=30,
                        use_precomputed_wavelength=False):
    
    #print

    logger = logging.getLogger("FindMatchingLines")
    logger.debug("Using d=%.6f A/px, central wavelength: %10.4f @ %8.2f px" % (
        dispersion, central_wavelength, reference_pixel_x))

    #
    # Make a copy of lineinfo to make sure we don't overwrite good information
    #
    my_lineinfo = numpy.array(lineinfo)
    #print "lineinfo.shape=",lineinfo.shape

    blue_edge = rss.calc_bluewavelength() * mm_to_A
    red_edge = rss.calc_redwavelength() * mm_to_A
    wl_range = red_edge - blue_edge
    logger.debug("Estimated wavelength range: %f -- %f A" % (blue_edge, red_edge))

    #
    # Find average offset between arc lines and reference lines
    # 
    
    # only select strong lines in the ARC spectrum
    good_s2n = my_lineinfo[:,lineinfo_colidx['S2N']] > s2n_cutoff
    #selected_lines = numpy.array(lineinfo[good_s2n])
    #print selected_lines.shape

    if (not use_precomputed_wavelength):
        # compute wavelength based on central wavelength and
        my_lineinfo[:,lineinfo_colidx['WAVELENGTH']] = \
            (my_lineinfo[:,lineinfo_colidx['PIXELPOS']]-reference_pixel_x) * dispersion + central_wavelength
        #wl = (selected_lines[:,lineinfo_colidx['PIXELPOS']]-reference_pixel_x) * dispersion + central_wavelength
        # we already have wavelength from some other source
        #wl = selected_lines[:,lineinfo_colidx['WAVELENGTH']]
    #else:


    #
    # extract only wavelengths for strong lines in the ARC spectrum
    #
    # or pick only the N strongest lines for now
    #N = 15
    #s2n_sort = numpy.argsort(my_lineinfo[:,lineinfo_colidx['S2N']])[::-1][:N]
    #my_lineinfo = my_lineinfo[s2n_sort] #good_s2n]
    #my_lineinfo = my_lineinfo[good_s2n]
    #print my_lineinfo[:,lineinfo_colidx['S2N']]
    wl = my_lineinfo[:,lineinfo_colidx['WAVELENGTH']]
#[good_s2n]

    #print ref_lines[:,0].reshape((-1,1)).T.shape
    #print wl.reshape((-1,1)).shape
    #numpy.savetxt("arc_lines", wl)
    #numpy.savetxt("ref_lines", ref_lines[:,0])

    #
    # Compute wavelength differences between each line in the ARC spectrum to 
    # every line in the lamp line catalog. Good wavelength shifts, i.e. such that 
    # make spectra overlap, will appear more frequently as the right shift is 
    # the same for a bunch of lines, whereas wrong matches differ from one line
    # to the next.
    #
    differences = ref_lines[:,0].reshape((-1,1)).T - wl.reshape((-1,1))
    numpy.savetxt("diffs", differences.flatten())

    #
    # Now find the most frequently found offset
    # # Use kernel densities to avoid ambiguities between two adjacent bins
    #

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

    #
    # Now improve the wavelength calibration of all found ARC lines by 
    # applying the shift we just found
    #
    my_lineinfo[:,lineinfo_colidx['WAVELENGTH']] += avg_shift
    numpy.savetxt("lineinfo__dispersion=%.3f" % (dispersion), my_lineinfo)

    #lineinfo[:,-1] += avg_shift

    #
    # Now match the two catalogs so we can derive an even better wavelength 
    # calibration. 
    # This also allows us to count how many lines we are able to match.
    #
    matched = match_line_catalogs(my_lineinfo, ref_lines, matching_radius,
                                  col_arc=lineinfo_colidx['WAVELENGTH'],
                                  col_ref=0)
    numpy.savetxt("matched.lines.%.4f" % (dispersion), matched)

    logger.info("Trying dispersion %8.4f A/px   ===>   shift: %8.2fA, #matches: %3d" % (
            dispersion, avg_shift, matched.shape[0]))

    return matched



def manual_loadtxt(filename, n_cols=2):

    data = []

    with open(filename, "r") as f:
        
        for line in f.readlines():
            if (len(line) <= 0 or line[0] == "#"):
                # this is a comment line
                continue

            linedata = [0] * n_cols
            for idx, item in enumerate(line.split()[:n_cols]):
                try:
                    linedata[idx] = (float(item))
                except:
                    pass
                
            data.append(linedata)

    # print data
    # print len(data)
    x = numpy.array(data)
    # print x.shape

    numpy.savetxt(sys.stdout, numpy.array(data), "%.5f")

    return numpy.array(data)


class  KenRSSModel( object ):

    def __init__(self, primhdr, ncols):

        self.logger = logging.getLogger("KenRSS")

        self.ncols = ncols

        self.primhdr = primhdr
        self.rbin,self.cbin = numpy.array(primhdr["CCDSUM"].split(" ")).astype(int)
        self.grating = primhdr['GRATING'].strip()
        self.grang = float(primhdr['GR-ANGLE'])
        self.artic = float(primhdr['CAMANG'])
        
        self.logger.info("RSS model: bin: %d,%d - grating: %s - angle: %.2f - artic: %.2f  - columns: %d" % (
            self.rbin,self.cbin,self.grating,self.grang,self.artic,self.ncols))

        self.compute()
        
    def compute(self, grating=None, grang=None, artic=None, cbin=None, ncols=None, colpos=None):

        _grating = self.grating if grating == None else grating
        _grang = self.grang if grang == None else grang
        _artic = self.artic if artic == None else artic
        _cbin = self.cbin if cbin == None else cbin
        _ncols = self.ncols if ncols == None else ncols

        self.all_wavelength = rssmodelwave(_grating,_grang,_artic,_cbin,_ncols)

        # fit a simple linear interpolation to the curve so we can look up 
        # wavelengths for a given pixel more easily
        self.wl_interpol = scipy.interpolate.interp1d(
            x=numpy.arange(self.all_wavelength.shape[0]),
            y=self.all_wavelength
        )

        if (not colpos == None):
            return self.compute_wl(colpos)

    def compute_wl(self, colpos):
        return self.wl_interpol(colpos)

    def central_wavelength(self):
        return numpy.median(self.all_wavelength)
    
    def get_wavelength_list(self):
        return self.all_wavelength


def find_wavelength_solution(filename, line):

    logger = logging.getLogger("FindWLS")

    if (type(filename) == str and os.path.isfile(filename)):
        hdulist = fits.open(filename)
    elif (type(filename) == fits.hdu.hdulist.HDUList):
        hdulist = filename
    else:
        logger.error("Invalid input, needs to be either HDUList or string, but found %s" % (str(type(filename))))
        return None

    if (line == None):
        line = hdulist['SCI'].data.shape[0] / 2
        logger.debug("Picking the central row, # = %d" % (line))

    avg_width = 10
    spec = extract_arc_spectrum(hdulist, line, avg_width)

    binx, biny = pysalt.get_binning(hdulist)
    logger.debug("Binning: %d x %d" % (binx, biny))

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
    logger.debug("From RSS model: wl-range: %.1f - %.1f [%.1f], dispersion: %.3f" % (
            blue_edge, red_edge, central_wl, dispersion))

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

    # Use Ken's RSS model data here
    logger.info("Creating dispersion model")
    cols = hdulist['SCI'].data.shape[1]
    kens_model = KenRSSModel(hdulist[0].header, cols)
    # primhdr = hdulist[0].header
    # rbin,cbin = numpy.array(primhdr["CCDSUM"].split(" ")).astype(int)
    # grating = primhdr['GRATING'].strip()
    # grang = float(primhdr['GR-ANGLE'])
    # artic = float(primhdr['CAMANG'])
    # logger.info("RSS model: bin: %d,%d - grating: %s - angle: %.2f - artic: %.2f  - columns: %d" % (
    #     rbin,cbin,grating,grang,artic,cols))
    # all_wavelength = rssmodelwave(grating,grang,artic,cbin,cols)
    # # fit a simple linear interpolation to the curve so we can look up 
    # # wavelengths for a given pixel more easily
    # wl_interpol = scipy.interpolate.interp1d(
    #     x=numpy.arange(all_wavelength.shape[0]),
    #     y=all_wavelength
    # )

    # Now use the wavelength model to translate line position in pixel coordinates
    # into wavelength positions
    wl = kens_model.compute_wl(lineinfo[:,0])
    numpy.savetxt("rss_lines", numpy.append(wl.reshape((-1,1)),
                                            lineinfo, axis=1))

    # wl = lineinfo[:,0] * dispersion + blue_edge
    #lineinfo = numpy.append(lineinfo, wl.reshape((-1,1)), axis=1)
    #numpy.savetxt("linecal", lineinfo)

    #
    # Load linelist
    #
    lamp=hdulist[0].header['LAMPID'].strip().replace(' ', '')
    lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.txt" % lamp)
    #lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.salt" % lamp)
    _, fn_only = os.path.split(lampfile)
    logger.info("Reading calibration line wavelengths from data->%s" % (fn_only))
    logger.debug("Full path to lamp line list: %s" % (lampfile))
    #lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.wav" % lamp)
    #lampfile=pysalt.get_data_filename("pysalt$data/linelists/Ar.salt")
    #lampfile="Ar.lines"
    try:
        lines = numpy.loadtxt(lampfile)
    except:
        lines = manual_loadtxt(lampfile)

    #print lines.shape
    #print lines

    #
    # Now select only lines that are in the estimated range of our ARC spectrum
    #
    in_range = (lines[:,0] > numpy.min(wl)) & (lines[:,0] < numpy.max(wl))
    ref_lines = lines #[in_range]
    logger.debug("Found these lines for fitting (range: %.2f -- %.2f):\n%s" % (
        numpy.min(wl), numpy.max(wl), 
        "\n".join(["%10.4f" % l for l in ref_lines[:,0]])))
    #print ref_lines

    ############################################################################
    #
    # Next step for wavelength calibration:
    #
    # Match lines between ARC spectrum and reference line list, 
    # allowing for a small uncertainty in dispersion
    #
    ############################################################################

    camangle = kens_model.artic
    gratingangle = kens_model.grang

    max_d_camangle = 1.
    max_d_gratingangle = 1.

    step_camangle = 0.05
    step_gratingangle = 0.1

    n_steps_camangle = int(math.ceil(2*max_d_camangle / step_camangle + 1))
    n_steps_gratingangle = int(math.ceil(2*max_d_gratingangle / step_gratingangle + 1))

    results = numpy.zeros((n_steps_camangle, n_steps_gratingangle))

    try_camangles = numpy.linspace(camangle-max_d_camangle, 
                                   camangle+max_d_camangle, 
                                   n_steps_camangle)
    try_gratingangles = numpy.linspace(gratingangle-max_d_gratingangle, 
                                       gratingangle+max_d_gratingangle, 
                                       n_steps_gratingangle)

    
    ref_kdtree = scipy.spatial.cKDTree(ref_lines[:,0].reshape((-1,1)))
    matching_radius=1.0
    for idx_camangle, idx_gratingangle in \
        itertools.product(range(n_steps_camangle), range(n_steps_gratingangle)):

        camangle = try_camangles[idx_camangle]
        gratingangle = try_gratingangles[idx_gratingangle]

        #print camangle, gratingangle


        #match_line_catalogs(arc, ref, matching_radius, verbose=False,
        #                col_ref=0, col_arc=-1, dumpfile=None):


        # compute the wavelength position for all lines using the
        # spectrograph parameters of this iteration
        wl_lines = kens_model.compute(grang=gratingangle,
                                      artic=camangle,
                                      colpos=lineinfo[:,0])

        #print wl_lines, ref_lines.shape
                                      
        nearest_neighbor, i = ref_kdtree.query(
            x=wl_lines.reshape((-1,1)), 
            k=1, # only find 1 nearest neighbor
            p=1, # use linear distance
            distance_upper_bound=matching_radius)

        # i is the index with the closest match
        # good matches have i within legal range
        good_match = i < ref_lines.shape[0]

        results[idx_camangle, idx_gratingangle] = numpy.sum(good_match)

    numpy.savetxt("results", results)

    # fig=matplotlib.pyplot.figure()
    # ax=fig.add_subplot(111)
    # ax.imshow(results, interpolation='none')
    # fig.show()
    # matplotlib.pyplot.show()

    most_matches = numpy.unravel_index(numpy.argmax(results), results.shape)

    logger.info("best results: %d matches for camangle=%.3f (%.3f), gratingangle=%.3f (%.3f)" % (
        results[most_matches], 
        try_camangles[most_matches[0]], hdulist[0].header['CAMANG'],
        try_gratingangles[most_matches[1]], hdulist[0].header['GR-ANGLE'],
    ))
    best_camangle = try_camangles[most_matches[0]]
    best_gratingangle = try_gratingangles[most_matches[1]]

    #
    # Now write the complete spectrum with the wavelength calibration
    #
    #print spec.shape

    kens_model.compute(artic=best_camangle, grang=best_gratingangle, ncols=spec.shape[0])
    spec_wl = kens_model.get_wavelength_list()
    numpy.savetxt("spec_precalib", numpy.append(spec_wl.reshape((-1,1)),
                                                spec.reshape((-1,1)), axis=1))

    #
    # Now cross-identify all lines, and do a least-sq fit to further optimize 
    # the spectrograph angles and compute the final wavelength calibration fit
    # 

    lineinfo[:,5] = kens_model.compute(
        artic=best_camangle, 
        grang=best_gratingangle,
        colpos = lineinfo[:,0],
        )
    
    numpy.savetxt("linelist.calib", lineinfo)

    # one last time, match the two line lists, using the same matching radius 
    # as above

    nearest_neighbor, i = ref_kdtree.query(
        x=lineinfo[:,5].reshape((-1,1)), 
        k=1, # only find 1 nearest neighbor
        p=1, # use linear distance
        distance_upper_bound=matching_radius)
    good_matches = i < ref_lines.shape[0]

    # print "\n-------------"*5
    # print nearest_neighbor.shape
    # print nearest_neighbor
    # print i.shape
    # print i
    # print good_matches
    # print lineinfo.shape
    # print ref_lines.shape
    # print numpy.sum(good_matches)

    n_matches = numpy.sum(good_matches)

    # matched_ref = ref_lines[:,0][good_matches]
    # matched_line_idx = i[good_matches]
    
    # print "============"
    matched_catalog = numpy.zeros((n_matches, lineinfo.shape[1]+ref_lines.shape[1]))
    # print matched_catalog.shape
    
    matched_catalog[:,:lineinfo.shape[1]] = lineinfo[good_matches]
    matched_catalog[:,lineinfo.shape[1]:] = ref_lines[i[good_matches]]

    numpy.savetxt("matched_lines.dat", matched_catalog)

    # print "xxxxxxxxxxxxx"

    def fit__wavelength(p_fit, line_x, rssmodel):
        computed_wl = rssmodel.compute(
            grang=p_fit[0],
            artic=p_fit[1],
            colpos=line_x)
        return computed_wl

    def fit__wavelength_error(p_fit, matched_lines, rssmodel):
        line_wl = fit__wavelength(p_fit, matched_lines[:,0], rssmodel)
        ref_wl = matched_lines[:,len(lineinfo_cols)]
        delta_wl = line_wl - ref_wl # add some weighting here???
        return delta_wl

    p_start = [best_gratingangle, best_camangle]
    fit_args = (matched_catalog, kens_model)
    _fit = scipy.optimize.leastsq(fit__wavelength_error,
                                  p_start, 
                                  args=fit_args,
                                  maxfev=500,
                                  full_output=1)
    p_final = _fit[0]
    logger.info("Fit results: Was: %s --> Now: %s" % (
        " ".join(["%.4f" % f for f in p_start]),
        " ".join(["%.4f" % f for f in p_final]),
        ))
    
    # print p_start, " ==> ", p_final

    lineinfo[:,lineinfo_colidx['WAVELENGTH']] = fit__wavelength(
        p_final, lineinfo[:, lineinfo_colidx["PIXELPOS"]], kens_model)
    matched_catalog[:, lineinfo_colidx['WAVELENGTH']] = fit__wavelength(
        p_final, matched_catalog[:, lineinfo_colidx["PIXELPOS"]], kens_model)

    numpy.savetxt("matched_lines_afterfit.dat", matched_catalog)

    # 
    # compute rms
    #
    rms = numpy.std(matched_catalog[:, lineinfo_colidx['WAVELENGTH']] - \
                    matched_catalog[:, len(lineinfo_cols)])
    logger.info("Found RMS: %f" % (rms))

    wls = compute_wavelength_solution(matched_catalog, max_order=3)
    logger.info("Best fit wavelength solution: L = %9.3f %+9.3f * x %+9.3e x^2 %+9.3e x^3" % (
        wls[0], wls[1], wls[2], wls[3]))

    
    # # # set the indices for bad matches to a valid value
    # # i[~good_matches] = 0

    # # matched = numpy.zeros((arc.shape[0], (arc.shape[1]+ref.shape[1])))
    # # matched[:,:arc.shape[1]] = arc
    # # matched[:,arc.shape[1]:] = ref[i]
    # # #print "XXXXXXXXXXXX\n"*3,ref.shape, ref[i].shape, i.shape, bad_matches.shape
    # # #print bad_matches
    # # numpy.savetxt("matched_raw", matched)
    # # numpy.savetxt("matched_bad", bad_matches)
    # # #sys.exit(0)

    # # #print "XXXX\n"*3
    # # matched = matched[~bad_matches]
    

    # return


    # i = numpy.array(i)
    # bad_matches = (i>=ref.shape[0])
    # i[bad_matches] = 0

        
    # return

    
    # reference_pixel_x = 0.5 * spec.shape[0]

    # # dispersion was calculated above
    # # central wavelength 
    # central_wavelength = kens_model.central_wavelength() #numpy.median(all_wavelength)  # 0.5 * (blue_edge + red_edge)
    # max_dispersion_error = 0.00 #10 # +/- 10% should be plenty
    # dispersion_search_steps = 0.01 # vary dispersion in 1% steps
    # n_dispersion_steps = (max_dispersion_error / dispersion_search_steps) * 2 + 1

    # #
    # # compute what dispersion factors we are trying 
    # #
    # trial_dispersions = numpy.linspace(1.0-max_dispersion_error, 
    #                                   1.0+max_dispersion_error,
    #                                   n_dispersion_steps) * dispersion 
    # n_matches = numpy.zeros((trial_dispersions.shape[0]))
    # matched_cats = [None] * trial_dispersions.shape[0]

    # #
    # # Now try each dispersion, one by one, and compute what wavelength offset 
    # # we would need to make the maximum number of lines match known lines from 
    # # the line catalog.
    # # 

    # #
    # # New: Assume we know line centers to within a pixel, then we can match 
    # # lines that lie within a 1-pixel radius
    # #
    # # Remember: units here are angstroem !!!
    # matching_radius = 2 * dispersion
    # logger.debug("Considering lines within %.1f A of a known ARC line as matched!" % (
    #         matching_radius))
    # # print "before loop:", lineinfo.shape

    # for idx, _dispersion in enumerate(trial_dispersions):

    #     # compute dispersion including the correction
    #     #_dispersion = dispersion * disp_factor

    #     # copy the original line info so we don't accidently overwrite important data
    #     _lineinfo = numpy.array(lineinfo)
        
    #     # find optimal line shifts and match lines 
    #     # consider lines within 5A of each other matches
    #     # --> this most likely will depend on spectral resolution and binning
    #     matched_cat = find_matching_lines(ref_lines, _lineinfo, 
    #                                       rss,
    #                                       _dispersion, central_wavelength,
    #                                       reference_pixel_x,
    #                                       matching_radius = matching_radius, 
    #     )

    #     # Save results for later picking of the best one
    #     n_matches[idx] = matched_cat.shape[0]
    #     matched_cats[idx] = matched_cat

    #     numpy.savetxt("dispersion_scale_%.3f" % (_dispersion), matched_cat)

    # #
    # # Find the solution with the most matched lines
    # #
    # n_max = numpy.argmax(n_matches)
    
    # #print

    # #print "most matched lines:", n_matches[n_max],
    # #print "best dispersion: %f" % (trial_dispersions[n_max])
    # logger.debug("Choosing best solution: %4d for dispersion %8.4f A/px" % (
    #         n_matches[n_max], trial_dispersions[n_max]))

    # numpy.savetxt("matchcount", numpy.append(trial_dispersions.reshape((-1,1)),
    #                                          n_matches.reshape((-1,1)),
    #                                          axis=1))

    # matched = matched_cats[n_max]
    # numpy.savetxt("matched.lines.best", matched)
    
    # # print "***************************\n"*5
    # # print lineinfo.shape

    # logger.info("Computing an analytical wavelength calibration...")
    # wls = compute_wavelength_solution(matched, max_order=3)

    # spec_x = numpy.polynomial.polynomial.polyval(
    #     numpy.arange(spec.shape[0]), wls).reshape((-1,1))
    # spec_combined = numpy.append(spec_x, spec.reshape((-1,1)), axis=1)
    # numpy.savetxt("spec.calib.start", spec_combined)

    # #
    # # Now we have a best-match solution
    # # Match lines again to see what the RMS is - use a small matching radius now
    # #

    # ############################################################################
    # #
    # # Since the original matching was done using a simple polynomial form, 
    # # let's now iterate the rough solution using higher order polynomials to 
    # # (hopefully) match a larger number of lines
    # #
    # # Add here: 
    # # - Keep an eye on r.m.s. of the fit, and the number of matched lines
    # # - with every step, reduce the matching radius to make sure we are matching
    # #   only with the most likely counterpart in case of close line (blends)
    # #
    # ############################################################################
    # #prev_wls = wls
    # matching_radius = 2*dispersion
    # logger.debug("Refining WLS using all %d lines" % (lineinfo.shape[0]))
    # for iteration in range(3):

    #     #sys.stdout.write("\n\n\n"); sys.stdout.flush()

    #     _linelist = numpy.array(lineinfo)

    #     # Now compute the new wavelength for each line
    #     _linelist[:,lineinfo_colidx['WAVELENGTH']] = numpy.polynomial.polynomial.polyval(
    #         _linelist[:,lineinfo_colidx['PIXELPOS']], wls)

    #     numpy.savetxt("lineinfo.iteration=%d" % (iteration+1), _linelist, "%10.4f")

    #     # compute wl with polyval
    #     # _wl = lineinfo[:,0]
    #     # numpy.savetxt("final.1", _wl)

    #     # numpy.savetxt("final.2", numpy.polynomial.polynomial.polyval(_wl, wls))
    #     # numpy.savetxt("final.3", _wl*wls[1]+wls[0])

    #     # With the refined wavelength solution, match lines between the 
    #     # ARC spectrum and the reference line catalog
    #     new_matches = match_line_catalogs(_linelist, ref_lines, 
    #                                       matching_radius=1.7, #matching_radius, # wsa 50 XXX
    #                                       verbose=False,
    #                                       col_arc=lineinfo_colidx['WAVELENGTH'], 
    #                                       col_ref=0,
    #                                       dumpfile="finalmatch.%d" % (iteration+1))

    #     logger.debug("WLS Refinement step %3d: now %3d matches!" % (
    #             iteration+1, new_matches.shape[0]))

    #     # -- for debugging --
    #     numpy.savetxt("matched.cat.iter%d" % (iteration+1), new_matches)

    #     #
    #     # Before fitting, iteratively reject obvious outliers most likely caused
    #     # by matching wrong lines
    #     #
    #     likely_outlier = numpy.isnan(new_matches[:,lineinfo_colidx['WAVELENGTH']])
    #     for reject in range(3):
    #         diff_angstroem = new_matches[:,lineinfo_colidx['WAVELENGTH']] - \
    #             new_matches[:,len(lineinfo_colidx)]
    #         med = numpy.median(diff_angstroem[~likely_outlier])
    #         stdx = numpy.std(diff_angstroem[~likely_outlier])
    #         sigma = scipy.stats.scoreatpercentile(diff_angstroem[~likely_outlier], [16,84])
    #         std = 0.5*(sigma[1]-sigma[0])
    #         likely_outlier = (diff_angstroem > med+2*std) | (diff_angstroem < med-2*std)
    #         logger.debug("Med/Std/StdX= %f / %f / %f" % (med, std, stdx))

    #     # Now we have a better idea on outliers, so reject them and work with 
    #     # what's left
    #     new_matches = new_matches[~likely_outlier]
    #     logger.debug("WLS Refinement step %3d: %3d matches left after outliers!" % (
    #             iteration+1, new_matches.shape[0]))

    #     # And with the matched line list, compute a new wavelength solution
    #     wls = compute_wavelength_solution(new_matches, max_order=4)#+iteration)

    #     # -- for debugging --
    #     numpy.savetxt("matched.outlierreject.iter%d" % (iteration+1), new_matches)
    #     #
    #     spec_x = numpy.polynomial.polynomial.polyval(
    #         numpy.arange(spec.shape[0])+1., wls).reshape((-1,1))
    #     spec_combined = numpy.append(spec_x, spec.reshape((-1,1)), axis=1)
    #     numpy.savetxt("spec.calib.iteration_%d" % (iteration+1), spec_combined)

    #     #prev_wls = wls

    # #return
    # final_match = matched


    # numpy.savetxt("matched.cat.final", final_match)


    # Also save the original spectrum as text file
    spec_x = numpy.polynomial.polynomial.polyval(numpy.arange(spec.shape[0]), wls).reshape((-1,1))
    spec_combined = numpy.append(spec_x, spec.reshape((-1,1)), axis=1)
    numpy.savetxt("spec.calib", spec_combined)
    

    # print lines
    # print _linelist
    # print spec.shape
    # print spec_combined.shape
    # print line
    # print wls

    return {
        'spec': spec,
        'spec_combined': spec_combined,
        'linelist_ref': lines,
        # 'linelist_arc': _linelist,
        'linelist_arc': lineinfo,
        'line': line,
        'wl_fit_coeffs': wls,
        }


def create_wl_calibration_plot(wls_data, hdulist, plotfile):

    fig = pl.figure()
    ax = fig.add_subplot(111)

    spec_combined = wls_data['spec_combined']

    # find wavelength range to plot
    l_min, l_max = numpy.min(spec_combined[:,0]), numpy.max(spec_combined[:,0])
    # and use this range for the plot
    ax.set_xlim((l_min, l_max))

    # also find good min and max ranges
    f_min, f_max = bottleneck.nanmin(spec_combined[:,1]), bottleneck.nanmax(spec_combined[:,1])
    # ax.set_ylim((0.9*f_min if f_min > 1 else 1, 1.1*f_max))
    ax.set_ylim((100, 1.1*f_max))
    
    # plot the actual spectrum we extracted for calibration
    ax.plot(spec_combined[:,0], spec_combined[:,1], "-g")

    # now draw vertical lines showing where we the arc lines from the catalog are
    for catline in wls_data['linelist_ref']:
        ax.axvline(x=catline[0], color='grey')

    # set the y-scale to be logarithmic
    ax.set_yscale('log')

    # add some labels
    ax.set_xlabel("Wavelength [angstroems]")
    ax.set_ylabel("flux [counts]")
    ax.set_title("name of file")

    fig.subplots_adjust(left=0.09, bottom=0.08, right=0.98, top=0.93,
                wspace=None, hspace=None)

    #fig.tight_layout(pad=0.1)

    if (not plotfile == None):
        fig.savefig(plotfile)
    else:
        fig.show()
        pl.show()




if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()

    filename = sys.argv[1]

    line = None
    try:
        line = int(sys.argv[2])
    except:
        pass

    hdulist = fits.open(filename)

    wls_data = find_wavelength_solution(hdulist, line)

    plotfile = filename[:-5]+".png"
    create_wl_calibration_plot(wls_data, hdulist, plotfile)

    hdulist.writeto("wlcal_dumpout.fits", clobber=True)

    pysalt.mp_logging.shutdown_logging(logger_setup)
