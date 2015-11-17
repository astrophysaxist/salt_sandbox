#!/usr/bin/env python


"""
SPECREDUCE

General data reduction script for SALT long slit data.

This includes step that are not yet included in the pipeline 
and can be used for extended reductions of SALT data. 

It does require the pysalt package to be installed 
and up to date.

"""

import os, sys, glob, shutil, time

import numpy
import pyfits
from scipy.ndimage.filters import median_filter
import bottleneck
import scipy.interpolate
numpy.seterr(divide='ignore', invalid='ignore')

# Disable nasty and useless RankWarning when spline fitting
import warnings
warnings.simplefilter('ignore', numpy.RankWarning)
warnings.simplefilter('ignore', pyfits.PyfitsDeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
# sys.path.insert(1, "/work/pysalt/")
# sys.path.insert(1, "/work/pysalt/plugins")
# sys.path.insert(1, "/work/pysalt/proptools")
# sys.path.insert(1, "/work/pysalt/saltfirst")
# sys.path.insert(1, "/work/pysalt/saltfp")
# sys.path.insert(1, "/work/pysalt/salthrs")
# sys.path.insert(1, "/work/pysalt/saltred")
# sys.path.insert(1, "/work/pysalt/saltspec")
# sys.path.insert(1, "/work/pysalt/slottools")
# sys.path.insert(1, "/work/pysalt/lib")

#from pyraf import iraf
#from iraf import pysalt

import pysalt
from pysalt.saltred.saltobslog import obslog
from pysalt.saltred.saltprepare import saltprepare
from pysalt.saltred.saltbias import saltbias
from pysalt.saltred.saltgain import saltgain
from pysalt.saltred.saltxtalk import saltxtalk
from pysalt.saltred.saltcrclean import saltcrclean
from pysalt.saltred.saltcombine import saltcombine
from pysalt.saltred.saltflat import saltflat
from pysalt.saltred.saltmosaic import saltmosaic
from pysalt.saltred.saltillum import saltillum

from pysalt.saltspec.specidentify import specidentify
from pysalt.saltspec.specrectify import specrectify
from pysalt.saltspec.specsky import skysubtract
from pysalt.saltspec.specextract import extract, write_extract
from pysalt.saltspec.specsens import specsens
from pysalt.saltspec.speccal import speccal

from PySpectrograph.Spectra import findobj

import pyfits
import pysalt.mp_logging
import logging
import numpy

#
# Ralf Kotulla modules
#
from helpers import *
import wlcal
import traceline
import skysub2d
import optimal_spline_basepoints as optimalskysub
import skyline_intensity
import prep_science
import podi_cython

wlmap_fitorder = [2,2]

def find_appropriate_arc(hdulist, arcfilelist, arcinfos={}):

    hdrs_to_match = [
        'CCDSUM',
        'WP-STATE', # Waveplate State Machine State
        'ET-STATE', # Etalon State Machine State
        'GR-STATE', # Grating State Machine State
        'GR-STA',   # Commanded Grating Station
        'GRATING',  # Commanded grating station
        'GRTILT',   # Commanded grating angle
        'BS-STATE', # Beamsplitter State Machine State
        'FI-STATE', # Filter State Machine State
        'AR-STATE', # Articulation State Machine State
        'AR-STA',   # Commanded Articulation Station
        'CAMANG',   # Commanded Articulation Station
        'POLCONF',  # Polarization configuration
        ]

    logger = logging.getLogger("FindGoodArc")
    logger.info("Checking the following list of ARCs:\n * %s" % ("\n * ".join(arcfilelist)))

    matching_arcs = []
    for arcfile in arcfilelist:
        if (arcfile in arcinfos):
            # use header that we extracted in an earlier run
            hdr = arcinfos[arcfile]
        else:
            # this is a new file we haven't scanned before
            arc_hdulist = pyfits.open(arcfile)
            hdr = arc_hdulist[0].header
            arcinfos[arcfile] = hdr
            arc_hdulist.close()

        #
        # Now search for files with the identical spectral setup
        #
        matches = True
        for hdrname in hdrs_to_match:
            logger.debug("Comparing header key --> %s <--" % (hdrname))

            # if we can't compare the headers we'll assume they won't match
            if (not hdrname in hdulist[0].header or
                not hdrname in hdr):
                matches = False
                logger.debug("Not found in one of the two files")
                break

            if (not hdulist[0].header[hdrname] == hdr[hdrname]):
                matches = False
                logger.debug("Found in both, but does not match!")
                break

        # if all headers exist in both files and all headers match, 
        # then this ARC file should be usable to calibrate the OBJECT frame
        if (matches):
            logger.debug("FOUND GOOD ARC")
            matching_arcs.append(arcfile)

    print "***\n"*3,matching_arcs,"\n***"*3

    return matching_arcs



def tiledata(hdulist, rssgeom):

    logger = logging.getLogger("TileData")

    out_hdus = [hdulist[0]]

    gap, xshift, yshift, rotation = rssgeom
    xshift = numpy.array(xshift)
    yshift = numpy.array(yshift)

    # Gather information about existing extensions
    sci_exts = []
    var_exts = []
    bpm_exts = []
    detsecs = []

    exts = {} #'SCI': [], 'VAR': [], 'BPM': [] }
    ext_order = ['SCI', 'BPM', 'VAR']
    for e in ext_order:
        exts[e] = []

    for i in range(1, len(hdulist)):
        if (hdulist[i].header['EXTNAME'] == 'SCI'):
            
            # Remember this function for later use
            sci_exts.append(i)
            exts['SCI'].append(i)

            # Also find out the detsec header entry so we can put all chips 
            # in the right order (blue to red) without having to rely on 
            # ordering within the file
            decsec = hdulist[i].header['DETSEC']
            detsec_startx = int(decsec[1:-1].split(":")[0])
            detsecs.append(detsec_startx)

                       
            var_ext, bpm_ext = -1, -1
            if ('VAREXT' in hdulist[i].header):
                var_ext = hdulist[i].header['VAREXT']
            if ('BPMEXT' in hdulist[i].header):
                bpm_ext = hdulist[i].header['BPMEXT']
            var_exts.append(var_ext)
            bpm_exts.append(bpm_ext)

            exts['VAR'].append(var_ext)
            exts['BPM'].append(bpm_ext)

    #print sci_exts
    #print detsecs

    #
    # Better make sure we have all 6 CCDs
    #
    # Problem: How to handle different readout windows here???
    #
    if (len(sci_exts) != 6):
        logger.critical("Could not find all 6 CCD sections!")
        return

    # convert to numpy array
    detsecs = numpy.array(detsecs)
    sci_exts = numpy.array(sci_exts)

    # sort extensions by DETSEC position
    detsec_sort = numpy.argsort(detsecs)
    sci_exts = sci_exts[detsec_sort]

    for name in exts:
        exts[name] = numpy.array(exts[name])[detsec_sort]

    #print exts

    #
    # Now we have all extensions in the right order
    #

    # Compute how big the output array should be
    width = 0
    height = -1
    amp_width = numpy.zeros_like(sci_exts)
    amp_height = numpy.zeros_like(sci_exts)
    for i, ext in enumerate(sci_exts):
        amp_width[i] = hdulist[ext].data.shape[1]
        amp_height[i] = hdulist[ext].data.shape[0]

    # Add in the widths of all gaps
    binx, biny = pysalt.get_binning(hdulist)
    logger.info("Creating tiled image using binning %d x %d" % (binx, biny))

    width = numpy.sum(amp_width) + 2*gap/binx # + numpy.sum(numpy.fabs((xshift/binx).round()))
    height = numpy.max(amp_height) #+ numpy.sum(numpy.fabs((yshift/biny).round()))

    #print width, height
    #print xshift
    #print yshift

    for name in ext_order:

        logger.info("Starting tiling for extension %s !" % (name))

        # Now create the mosaics
        data = numpy.empty((height, width))
        data[:,:] = numpy.NaN

        for i, ext in enumerate(exts[name]): #sci_exts):

            dx_gaps = int( gap * int(i/2) / binx )
            dx_shift = xshift[int(i/2)]/binx
            startx = numpy.sum(amp_width[0:i])

            # Add in gaps if applicable
            startx += dx_gaps
            # Also factor in the small corrections
            # startx -= dx_shift

            endx = startx + amp_width[i]

            logger.debug("Putting extension %d (%s) at X=%d -- %d (gaps=%d, shift=%d)" % (
                i, name, startx, endx, dx_gaps, dx_shift))
            #logger.info("input size: %d x %d" % (amp_width[i], amp_height[i]))
            #logger.info("output size: %d x %d" % (amp_width[i], height))
            data[:, startx:endx] = hdulist[ext].data[:,:amp_width[i]]

        imghdu = pyfits.ImageHDU(data=data)
        imghdu.name = name
        out_hdus.append(imghdu)
    
    logger.info("Finished tiling for all %d data products" % (len(ext_order)))

    return pyfits.HDUList(out_hdus)



            

def salt_prepdata(infile, badpixelimage=None, create_variance=False, 
                  masterbias=None, clean_cosmics=True,
                  flatfield_frame=None, mosaic=False,
                  verbose=False, *args):

    _, fb = os.path.split(infile)
    logger = logging.getLogger("PrepData(%s)" % (fb))
    logger.info("Working on file %s" % (infile))

    hdulist = pyfits.open(infile)
    
    pysalt_log = None #'pysalt.log'

    badpixel_hdu = None
    if (not badpixelimage == None):
        badpixel_hdu = pyfits.open(badpixelimage)
    
    #
    # Do some prepping
    #
    #hdulist.info()

    logger.debug("Prepare'ing")
    hdulist = pysalt.saltred.saltprepare.prepare(
        hdulist,
        createvar=create_variance, 
        badpixelstruct=badpixel_hdu)
    # Add some history headers here

    #
    # Overscan/bias
    #
    logger.debug("Subtracting bias & overscan")
    # for ext in hdulist:
    #     if (not ext.data == None): print ext.data.shape
    bias_hdu = None
    if (not masterbias == None and os.path.isfile(masterbias)):
        bias_hdu = pyfits.open(masterbias)
    hdulist = pysalt.saltred.saltbias.bias(
        hdulist, 
        subover=True, trim=True, subbias=False, 
        bstruct=bias_hdu,
        median=False, function='polynomial', order=5, rej_lo=3.0, rej_hi=5.0, 
        niter=10, plotover=False, 
        log=pysalt_log, verbose=verbose)
    logger.debug("done with bias & overscan")

    # print "--------------"
    # for ext in hdulist:
    #     if (not ext.data == None): print ext.data.shape

    # Again, add some headers here

    #
    # Gain
    #
    logger.debug("Correcting gain")
    dblist = [] #saltio.readgaindb(gaindb)
    hdulist = pysalt.saltred.saltgain.gain(hdulist,
                   mult=True, 
                   usedb=False, 
                   dblist=dblist, 
                   log=pysalt_log, verbose=verbose)
    logger.debug("done with gain")

    #
    # Xtalk
    #
    logger.debug("fixing crosstalk")
    usedb = False
    if usedb:
        xtalkfile = xtalkfile.strip()
        xdict = saltio.readxtalkcoeff(xtalkfile)
    else:
        xdict=None
    if usedb:
        obsdate=saltkey.get('DATE-OBS', struct[0])
        obsdate=int('%s%s%s' % (obsdate[0:4],obsdate[5:7], obsdate[8:]))
        xkey=numpy.array(xdict.keys())
        date=xkey[abs(xkey-obsdate).argmin()]
        xcoeff=xdict[date]
    else:
        xcoeff=[]

    hdulist = pysalt.saltred.saltxtalk.xtalk(hdulist, xcoeff, log=pysalt_log, verbose=verbose)
    logger.debug("done with crosstalk")

    #
    # crj-clean
    #
    #clean the cosmic rays
    multithread = True
    logger.debug("removing cosmics")
    if multithread and len(hdulist)>1:
        crj_function = pysalt.saltred.saltcrclean.multicrclean
    else:
        crj_function = pysalt.saltred.saltcrclean.crclean
    if (clean_cosmics):
        # hdulist = crj_function(hdulist, 
        #                        crtype='edge', thresh=5, mbox=11, bthresh=5.0,
        #                        flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
        #                        gbox=3, maxiter=5)

        gain = 1.5
        readnoise = 6

        sigclip = 5.0
        sigfrac = 0.3
        objlim = 5.0
        saturation_limit=65000

        # This is BEFORE mosaicing, therefore:
        # Loop over all SCI extensions
        for ext in hdulist:
            if (ext.name == 'SCI'):
                crj = podi_cython.lacosmics(
                    ext.data.astype(numpy.float64), 
                    gain=gain, 
                    readnoise=readnoise, 
                    niter=3,
                    sigclip=sigclip, sigfrac=sigfrac, objlim=objlim,
                    saturation_limit=saturation_limit,
                    verbose=False
                )
                cell_cleaned, cell_mask, cell_saturated = crj
                ext.data = cell_cleaned
        
    logger.debug("done with cosmics")


    #
    # Apply flat-field correction if requested
    #
    logger.info("FLAT: %s" % (str(flatfield_frame)))
    if (not flatfield_frame == None and os.path.isfile(flatfield_frame)):
        logger.debug("Applying flatfield")
        flathdu = pyfits.open(flatfield_frame)
        pysalt.specred.saltflat.flat(
            struct=hdulist, #input
            fstruct=flathdu, # flatfield
            )
        #saltflat('xgbpP*fits', '', 'f', flatimage, minflat=500, clobber=True, logfile=logfile, verbose=True)
        flathdu.close()
        logger.debug("done with flatfield")
    else:
        logger.debug("continuing without flat-field correction!")

    if (mosaic):
        logger.debug("Mosaicing all chips together")
        geomfile=pysalt.get_data_filename("pysalt$data/rss/RSSgeom.dat")
        geomfile=pysalt.get_data_filename("data/rss/RSSgeom.dat")
        logger.debug("Reading gemotry from file %s (%s)" % (geomfile, os.path.isfile(geomfile)))

        # does CCD geometry definition file exist
        if (not os.path.isfile(geomfile)):
            logger.critical("Unable to read geometry file %s!" % (geomfile))
        else:

            gap = 0
            xshift = [0, 0]
            yshift = [0, 0]
            rotation = [0, 0]
            gap, xshift, yshift, rotation, status = pysalt.lib.saltio.readccdgeom(geomfile, logfile=None, status=0)
            logger.debug("mosaicing -- GAP:%f - X-shift:%f/%f  y-shift:%f/%f  rotation:%f/%f" % (
                gap, xshift[0], xshift[1], yshift[0], yshift[1], rotation[0], rotation[1]))

            #logger.info("File structure before mosaicing:")
            #hdulist.info()

            gap = 90
            xshift = [0.0, +5.9, -2.1]
            yshift = [0.0, -2.6,  0.4]
            rotation = [0,0,0]
            hdulist = tiledata(hdulist, (gap, xshift, yshift, rotation))
            #return

            # create the mosaic
            # hdulist = pysalt.saltred.saltmosaic.make_mosaic(
            #     struct=hdulist, 
            #     gap=gap, xshift=xshift, yshift=yshift, rotation=rotation, 
            #     interp_type='linear',              
            #     #boundary='constant', constant=0, geotran=True, fill=False,
            #     boundary='constant', constant=0, geotran=False, fill=False,
            #     cleanup=True, log=None, verbose=verbose)
            logger.debug("done with mosaic")


    return hdulist




#################################################################################
#################################################################################
#################################################################################

def specred(rawdir, prodir, 
            imreduce=True, specreduce=True, 
            calfile=None, lamp='Ar', 
            automethod='Matchlines', skysection=[800,1000], 
            cleanup=True):

    print rawdir
    print prodir

    logger = logging.getLogger("SPECRED")

    #get the name of the files
    # if (type(infile) == list):
    #     infile_list = infile
    # elif (type(infile) == str and os.path.isdir(infile)):
    infile_list=glob.glob(rawdir+'*.fits')
        

    #get the current date for the files
    obsdate=os.path.basename(infile_list[0])[1:9]
    print obsdate

    #set up some files that will be needed
    logfile='spec'+obsdate+'.log'
    flatimage='FLAT%s.fits' % (obsdate)
    dbfile='spec%s.db' % obsdate

    #create the observation log
    # obs_dict=obslog(infile_list)

    # import pysalt.lib.saltsafeio as saltio
    
    print infile_list
                           
    #
    #
    # Now reduce all files, one by one
    #
    #
    work_dir = "working/"
    if (not os.path.isdir(work_dir)):
        os.mkdir(work_dir)

    # #
    # # Make sure we have all directories 
    # #
    # for rs in reduction_steps:
    #     dirname = "%s/%s" % (work_dir, rs)
    #     if (not os.path.isdir(dirname)):
    #         os.mkdir(dirname)

    #
    # Go through the list of files, find out what type of file they are
    #
    logger.info("Identifying frames and sorting by type (object/flat/arc)")
    obslog = {
        'FLAT': [],
        'ARC': [],
        'OBJECT': [],
    }

    for idx, filename in enumerate(infile_list):
        hdulist = pyfits.open(filename)
        if (not hdulist[0].header['INSTRUME'] == "RSS"):
            logger.info("Frame %s is not a valid RSS frame (instrument: %s)" % (
                filename, hdulist[0].header['INSTRUME']))
            continue
            
        obstype = hdulist[0].header['OBSTYPE']
        if (obstype in obslog):
            obslog[obstype].append(filename)
            logger.debug("Identifying %s as %s" % (filename, obstype))
        else:
            logger.info("No idea what to do with frame %s --> %s" % (filename, obstype))
            
    for type in obslog:
        if (len(obslog[type]) > 0):
            logger.info("Found the following %ss:\n -- %s" % (
                type, "\n -- ".join(obslog[type])))
        else:
            logger.info("No files of type %s found!" % (type))

    #
    # Go through the list of files, find all flat-fields, and create a master flat field
    #
    logger.info("Creating a master flat-field frame")
    flatfield_filenames = []
    flatfield_hdus = {}
    first_flat = None
    flatfield_list = {}

    for idx, filename in enumerate(obslog['FLAT']):
        hdulist = pyfits.open(filename)
        if (hdulist[0].header['OBSTYPE'].find("FLAT") >= 0 and
            hdulist[0].header['INSTRUME'] == "RSS"):
            #
            # This is a flat-field
            #

            #
            # Get some parameters so we can create flatfields for each specific
            # instrument configuration
            #
            grating = hdulist[0].header['GRATING']
            grating_angle = hdulist[0].header['GR-ANGLE']
            grating_tilt = hdulist[0].header['GRTILT']
            binning = "x".join(hdulist[0].header['CCDSUM'].split())

            if (not grating in flatfield_list):
                flatfield_list[grating] = {}
            if (not binning in flatfield_list[grating]):
                flatfield_list[grating][binning] = {}
            if (not grating_tilt in flatfield_list[grating][binning]):
                flatfield_list[grating][binning][grating_tilt] = {}    
            if (not grating_angle in flatfield_list[grating][binning][grating_tilt]):
                flatfield_list[grating][binning][grating_tilt][grating_angle] = []   

            flatfield_list[grating][binning][grating_tilt][grating_angle].append(filename)
            
    for grating in flatfield_list:
        for binning in flatfield_list[grating]:
            for grating_tilt in flatfield_list[grating][binning]:
                for grating_angle in flatfield_list[grating][binning][grating_tilt]:

                    filelist = flatfield_list[grating][binning][grating_tilt][grating_angle]
                    flatfield_hdus = {}

                    logger.info("Creating master flatfield for %s (%.3f/%.3f), %s (%d frames)" % (
                        grating, grating_angle, grating_tilt, binning, len(filelist)))

                    for filename in filelist:

                        _, fb = os.path.split(filename)
                        single_flat = "flat_%s" % (fb)

                        hdu = salt_prepdata(filename, 
                                            badpixelimage=None, 
                                            create_variance=True,
                                            clean_cosmics=False,
                                            mosaic=False,
                                            verbose=False)
                        pysalt.clobberfile(single_flat)
                        hdu.writeto(single_flat, clobber=True)
                        logger.info("Wrote single flatfield to %s" % (single_flat))

                        for extid, ext in enumerate(hdu):
                            if (ext.name == "SCI"):
                                # Only use the science extensions, leave everything else 
                                # untouched: Apply a one-dimensional median filter to take 
                                # out spectral slope. We can then divide the raw data by this 
                                # median flat to isolate pixel-by-pixel variations
                                filtered = scipy.ndimage.filters.median_filter(
                                    input=ext.data, 
                                    size=(1,25), 
                                    footprint=None, 
                                    output=None, 
                                    mode='reflect', 
                                    cval=0.0, 
                                    origin=0)
                                ext.data /= filtered

                                if (not extid in flatfield_hdus):
                                    flatfield_hdus[extid] = []
                                flatfield_hdus[extid].append(ext.data)

                        single_flat = "norm"+single_flat
                        pysalt.clobberfile(single_flat)
                        hdu.writeto(single_flat, clobber=True)
                        logger.info("Wrote normalized flatfield to %s" % (single_flat))

                        if (first_flat == None):
                            first_flat = hdulist

                    print first_flat

                    if (len(filelist) <= 0):
                        continue

                    # Combine all flat-fields into a single master-flat
                    for extid in flatfield_hdus:
                        flatstack = flatfield_hdus[extid]
                        #print "EXT",extid,"-->",flatstack
                        logger.info("Ext %d: %d flats" % (extid, len(flatstack)))
                        flatstack = numpy.array(flatstack)
                        print flatstack.shape
                        avg_flat = numpy.mean(flatstack, axis=0)
                        print "avg:", avg_flat.shape
                        
                        first_flat[extid].data = avg_flat

                    masterflat_filename = "flat__%s_%s_%.3f_%.3f.fits" % (
                        grating, binning, grating_tilt, grating_angle)
                    pysalt.clobberfile(masterflat_filename)
                    first_flat.writeto(masterflat_filename, clobber=True)

            # # hdu = salt_prepdata(filename, badpixelimage=None, create_variance=False, 
            # #                     verbose=False)
            # # flatfield_hdus.append(hdu)
        
    return

    #############################################################################
    #
    # Determine a wavelength solution from ARC frames, where available
    #
    #############################################################################

    logger.info("Searching for a wavelength calibration from the ARC files")
    skip_wavelength_cal_search = False #os.path.isfile(dbfile)
    
    # Keep track of when the ARCs were taken, so we can pick the one closest 
    # in time to the science observation for data reduction
    arc_obstimes = numpy.ones((len(obslog['ARC']))) * -999.9
    arc_mosaic_list = [None] * len(obslog['ARC'])
    arc_mef_list = [None] * len(obslog['ARC'])
    if (not skip_wavelength_cal_search):
        for idx, filename in enumerate(obslog['ARC']):
            _, fb = os.path.split(filename)
            hdulist = pyfits.open(filename)

            # Use Julian Date for simple time indexing
            arc_obstimes[idx] = hdulist[0].header['JD']

            arc_filename = "ARC_%s" % (fb)
            arc_mosaic_filename = "ARC_m_%s" % (fb)
            rect_filename = "ARC-RECT_%s" % (fb)

            logger.info("Creating MEF  for frame %s --> %s" % (fb, arc_filename))
            hdu = salt_prepdata(filename, 
                                badpixelimage=None, 
                                create_variance=True,
                                clean_cosmics=False,
                                mosaic=False,
                                verbose=False)
            pysalt.clobberfile(arc_filename)
            hdu.writeto(arc_filename, clobber=True)
            arc_mef_list[idx] = arc_filename

            logger.info("Creating mosaic for frame %s --> %s" % (fb, arc_mosaic_filename))
            hdu_mosaiced = salt_prepdata(filename, 
                                badpixelimage=None, 
                                create_variance=True,
                                clean_cosmics=False,
                                mosaic=True,
                                verbose=False)

            #
            # Now we have a HDUList of the mosaiced ARC file, so 
            # we can continue to the wavelength calibration
            #
            wls_data = wlcal.find_wavelength_solution(hdu_mosaiced, None)

            #
            # Write wavelength solution to FITS header so we can access it 
            # again if we need to at a later point
            #
            logger.info("Storing wavelength solution in ARC file (%s)" % (arc_mosaic_filename))
            hdu_mosaiced[0].header['WLSFIT_N'] = len(wls_data['wl_fit_coeffs'])
            for i in range(len(wls_data['wl_fit_coeffs'])):
                hdu_mosaiced[0].header['WLSFIT_%d' % (i)] = wls_data['wl_fit_coeffs'][i]

            #
            # Now add some plotting here just to make sure the user is happy :-)
            #
            plotfile = arc_mosaic_filename[:-5]+".png"
            wlcal.create_wl_calibration_plot(wls_data, hdu_mosaiced, plotfile)

            #
            # Simulate the ARC spectrum by extracting a 2-D ARC spectrum just 
            # like we would for the sky-subtraction in OBJECT frames
            #
            arc_region_file = "ARC_m_%s_traces.reg" % (fb[:-5])
            wls_2darc = traceline.compute_2d_wavelength_solution(
                arc_filename=hdu_mosaiced, 
                n_lines_to_trace=-15, #-50, # trace all lines with S/N > 50 
                fit_order=wlmap_fitorder,
                output_wavelength_image="wl+image.fits",
                debug=False,
                arc_region_file=arc_region_file,
                trace_every=0.05,
                )
            wl_hdu = pyfits.ImageHDU(data=wls_2darc)
            wl_hdu.name = "WAVELENGTH"
            hdu_mosaiced.append(wl_hdu)
            
            #
            # Now go ahead and extract the full 2-d sky
            #
            arc_regions = numpy.array([[0, hdu_mosaiced['SCI'].data.shape[0]]])
            arc2d = skysub2d.make_2d_skyspectrum(
                hdu_mosaiced,
                wls_2darc,
                sky_regions=arc_regions,
                oversample_factor=1.0,
                )
            simul_arc_hdu = pyfits.ImageHDU(data=arc2d)
            simul_arc_hdu.name = "SIMULATION"
            hdu_mosaiced.append(simul_arc_hdu)
            
            pysalt.clobberfile(arc_mosaic_filename)
            hdu_mosaiced.writeto(arc_mosaic_filename, clobber=True)
            arc_mosaic_list[idx] = arc_mosaic_filename


            # lamp=hdu[0].header['LAMPID'].strip().replace(' ', '')
            # lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.txt" % lamp)
            # automethod='Matchlines'
            # skysection=[800,1000]
            # logger.info("Searching for wavelength solution (lamp:%s, arc-image:%s)" % (
            #     lamp, arc_filename))
            # specidentify(arc_filename, lampfile, dbfile, guesstype='rss', 
            #              guessfile='', automethod=automethod,  function='legendre',  order=5, 
            #              rstep=100, rstart='middlerow', mdiff=10, thresh=3, niter=5, 
            #              inter=False, clobber=True, logfile=logfile, verbose=True)
            # logger.debug("Done with specidentify")

            # logger.debug("Starting specrectify")
            # specrectify(arc_filename, outimages=rect_filename, outpref='',
            #             solfile=dbfile, caltype='line', 
            #             function='legendre',  order=3, inttype='interp', 
            #             w1=None, w2=None, dw=None, nw=None,
            #             blank=0.0, clobber=True, logfile=logfile, verbose=True)

            # logger.debug("Done with specrectify")


    #return
    
    #############################################################################
    #
    # Now apply wavelength solution found above to your data frames
    #
    #############################################################################
    logger.info("\n\n\nProcessing OBJECT frames")
    arcinfos = {}
    for idx, filename in enumerate(obslog['OBJECT']):
        _, fb = os.path.split(filename)
        _fb, _ = os.path.splitext(fb)
        hdulist = pyfits.open(filename)
        logger = logging.getLogger("OBJ(%s)" % _fb)

        binx, biny = pysalt.get_binning(hdulist)
        logger.info("Using binning of %d x %d (spectral/spatial)" % (binx, biny))

        mosaic_filename = "OBJ_raw__%s" % (fb)
        out_filename = "OBJ_%s" % (fb)

        grating = hdulist[0].header['GRATING']
        grating_angle = hdulist[0].header['GR-ANGLE']
        grating_tilt = hdulist[0].header['GRTILT']
        binning = "x".join(hdulist[0].header['CCDSUM'].split())
        masterflat_filename = "flat__%s_%.3f_%.3f_%s.fits" % (
            grating, grating_angle, grating_tilt, binning)
        logger.info("FLATX: %s (%s, %f, %f, %s) = %s" % (
            masterflat_filename,
            grating, grating_angle, grating_tilt, binning, 
            filename)
        )
        if (not os.path.isfile(masterflat_filename)):
            masterflat_filename = None

        logger.info("Creating mosaic for frame %s --> %s" % (fb, mosaic_filename))
        hdu = salt_prepdata(filename, 
                            flatfield_frame = masterflat_filename,
                            badpixelimage=None, 
                            create_variance=True, 
                            clean_cosmics=True,
                            mosaic=True,
                            verbose=False,
        )
        pysalt.clobberfile(mosaic_filename)
        logger.info("Writing mosaiced OBJ file to %s" % (mosaic_filename))
        hdu.writeto(mosaic_filename, clobber=True)


        #
        # Also create the image without cosmic ray rejection, and add it to the 
        # output file
        #
        hdu_nocrj = salt_prepdata(filename, 
                            flatfield_frame = masterflat_filename,
                            badpixelimage=None, 
                            create_variance=True, 
                            clean_cosmics=False,
                            mosaic=True,
                            verbose=False,
        )
        hdu_sci_nocrj = hdu_nocrj['SCI']
        hdu_sci_nocrj.name = 'SCI.NOCRJ'
        hdu.append(hdu_sci_nocrj)



        # Make backup of the image BEFORE sky subtraction
        # make sure to copy the actual data, not just create a duplicate reference
        for source_ext in ['SCI', 'SCI.NOCRJ']:
            presub_hdu = pyfits.ImageHDU(data=numpy.array(hdu['SCI'].data),
                                         header=hdu['SCI'].header)
            presub_hdu.name = source_ext+'.RAW'
            hdu.append(presub_hdu)


        #
        # Find the ARC closest in time to this frame
        #
        # obj_jd = hdulist[0].header['JD']
        # delta_jd = numpy.fabs(arc_obstimes - obj_jd)
        # good_arc_idx = numpy.argmin(delta_jd)
        # good_arc = arc_mosaic_list[good_arc_idx]
        # logger.info("Using ARC %s for wavelength calibration" % (good_arc))
        # good_arc_list = find_appropriate_arc(hdu, obslog['ARC'], arcinfos)
        good_arc_list = find_appropriate_arc(hdu, arc_mosaic_list, arcinfos)
        logger.debug("Found these ARCs as appropriate:\n -- %s" % ("\n -- ".join(good_arc_list)))
        if (len(good_arc_list) == 0):
            logger.error("Could not find any appropriate ARCs")
            continue
        else:
            good_arc = good_arc_list[0]
            logger.info("Using ARC %s for wavelength calibration" % (good_arc))



        #
        # Use ARC to trace lines and compute a 2-D wavelength solution
        #
        logger.info("Computing 2-D wavelength map")
        arc_region_file = "OBJ_%s_traces.reg" % (fb[:-5])
        # wls_2d, slitprofile = traceline.compute_2d_wavelength_solution(
        #     arc_filename=good_arc, 
        #     n_lines_to_trace=-50, # trace all lines with S/N > 50 
        #     fit_order=wlmap_fitorder,
        #     output_wavelength_image="wl+image.fits",
        #     debug=False,
        #     arc_region_file=arc_region_file,
        #     return_slitprofile=True,
        #     trace_every=0.05)
        # print wls_2d
        # wl_hdu = pyfits.ImageHDU(data=wls_2d)
        # wl_hdu.name = "WAVELENGTH"
        # hdu.append(wl_hdu)

        arc_hdu = pyfits.open(good_arc)
        wls_2d = arc_hdu['WAVELENGTH'].data

        n_params = arc_hdu[0].header['WLSFIT_N']
        hdu[0].header["WLSFIT_N"] = arc_hdu[0].header["WLSFIT_N"]
        wls_fit = numpy.zeros(n_params)
        for i in range(n_params):
            wls_fit[i] = arc_hdu[0].header['WLSFIT_%d' % (i)]
            hdu[0].header['WLSFIT_%d' % (i)] = arc_hdu[0].header['WLSFIT_%d' % (i)]
            
        hdu.append(pyfits.ImageHDU(data=wls_2d, name='WAVELENGTH'))

        # 
        # Extract the sky-line intensity profile along the slit. Use this to 
        # correct the data. This should also improve the quality of the extracted
        # 2-D sky.
        #
        plot_filename = "%s_slitprofile.png" % (fb)
        skylines, skyline_list, intensity_profile = \
            prep_science.extract_skyline_intensity_profile(
                hdulist=hdu, 
                data=hdu['SCI.RAW'].data,
                wls=wls_fit,
                plot_filename=plot_filename,
            )
        # Flatten the science frame using the line profile
        hdu.append(
            pyfits.ImageHDU(
                data=hdu['SCI'].data, 
                header=hdu['SCI'].header, 
                name="SCI.PREFLAT"
            )
        )
        hdu.append(
            pyfits.ImageHDU(
                data=hdu['SCI'].data/intensity_profile.reshape((-1,1)), 
                header=hdu['SCI'].header, 
                name="SCI.POSTFLAT"
            )
        )

        #
        # Mask out all regions with relative intensities below 0.1x max 
        #
        stats = scipy.stats.scoreatpercentile(intensity_profile, [50, 16,84, 2.5,97.5])
        one_sigma = (stats[4] - stats[3]) / 4.
        median = stats[0]
        bad_region = intensity_profile < median-1*one_sigma
        hdu['SCI'].data[bad_region] = numpy.NaN
        intensity_profile[bad_region] = numpy.NaN


        # hdu['SCI'].data /= intensity_profile.reshape((-1,1))
        # logger.info("Slit-flattened SCI extension")

        # #
        # # Now go ahead and extract the full 2-d sky
        # #
        # logger.info("Extracting 2-D sky")
        # sky_regions = numpy.array([[0, hdu['SCI'].data.shape[0]]])
        # sky2d = skysub2d.make_2d_skyspectrum(
        #     hdu,
        #     wls_2d,
        #     sky_regions=sky_regions,
        #     oversample_factor=1.0,
        #     slitprofile=None, #slitprofile,
        #     )

        # logger.info("Performing sky subtraction")
        # sky_hdu = pyfits.ImageHDU(data=sky2d, name='SKY')
        # hdu.append(sky_hdu)

        # if (not slitprofile == None):
        #     sky_hdux = pyfits.ImageHDU(data=sky2d*slitprofile.reshape((-1,1)))
        #     sky_hdux.name = "SKY_X"
        #     hdu.append(sky_hdux)

        # # Don't forget to subtract the sky off the image
        # for source_ext in ['SCI', 'SCI.NOCRJ']:
        #     hdu[source_ext].data -= sky2d #(sky2d * slitprofile.reshape((-1,1)))

        # numpy.savetxt("OBJ_%s_slit.asc" % (fb[:-5]), slitprofile)

        #
        # Compute the optimized sky, using better-chosen spline basepoints 
        # to sample the sky-spectrum
        #
        sky_regions = numpy.array([[300,500], [1400,1700]])
        logger.info("Preparing optimized sky-subtraction")
        ia = None

        # simple_spec = optimalskysub.optimal_sky_subtraction(hdu, 
        #                                       sky_regions=sky_regions,
        #                                       N_points=1000,
        #                                       iterate=False,
        #                                       skiplength=10, 
        #                                       return_2d=False)
        # numpy.savetxt("%s.simple_spec" % (_fb), simple_spec)

        # simple_spec = hdu['VAR'].data[hdu['VAR'].data.shape[0]/2,:]
        # numpy.savetxt("%s.simple_spec_2" % (_fb), simple_spec)

        # logger.info("Searching for and analysing sky-lines")
        # skyline_list = wlcal.find_list_of_lines(simple_spec, readnoise=1, avg_width=1)
        # print skyline_list

        # logger.info("Creating spatial flatfield from sky-line intensity profiles")
        # i, ia, im = skyline_intensity.find_skyline_profiles(hdu, skyline_list)
    
        sky_2d, spline = optimalskysub.optimal_sky_subtraction(
            hdu, 
            sky_regions=sky_regions,
            N_points=10000,
            iterate=False,
            skiplength=5,
            skyline_flat=intensity_profile.reshape((-1,1)),
        )


        #
        # And finally write reduced frame back to disk
        #
        out_filename = "OBJ_%s" % (fb)
        logger.info("Saving output to %s" % (out_filename))
        pysalt.clobberfile(out_filename)
        hdu.writeto(out_filename, clobber=True)









        # #
        # # Trial: replace all 0 value pixels with NaNs
        # #
        # bpm = hdu[3].data
        # hdu[1].data[bpm == 1] = numpy.NaN

        # # for ext in hdu[1:]:
        # #     ext.data[ext.data <= 0] = numpy.NaN


        # spectrectify writes to disk, no need to do so here
        # specrectify(mosaic_filename, outimages=out_filename, outpref='', 
        #             solfile=dbfile, caltype='line', 
        #             function='legendre',  order=3, inttype='interp', 
        #             w1=None, w2=None, dw=None, nw=None,
        #             blank=0.0, clobber=True, logfile=logfile, verbose=True)
        
        # #
        # # Now we have a full 2-d spectrum, but still with emission lines
        # #
        
        # #
        # # Next, find good regions with no source contamation
        # #
        # hdu_rect = pyfits.open(out_filename)
        # hdu_rect.info()

        # src_region = [1500,2400] # Jay
        # src_region = [1850,2050] # Greg

        # #intspec = get_integrated_spectrum(hdu_rect, out_filename)
        # #slitprof, skymask = find_slit_profile(hdu_rect, out_filename) # Jay
        # slitprof, skymask = find_slit_profile(hdu_rect, out_filename, src_region)  # Greg
        # print skymask.shape[0]

        # hdu_rect['SCI'].data /= slitprof

        # rectflat_filename = "OBJ_flat_%s" % (fb)
        # pysalt.clobberfile(rectflat_filename)
        # hdu_rect.writeto(rectflat_filename, clobber=True)

        # #
        # # Block out the central region of the chip as object
        # #
        # skymask[src_region[0]/biny:src_region[1]/biny] = False
        # sky_lines = bottleneck.nanmedian(
        #     hdu_rect['SCI'].data[skymask].astype(numpy.float64),
        #     axis=0)
        # print sky_lines.shape
        
        # #
        # # Now subtract skylines
        # #
        # hdu_rect['SCI'].data -= sky_lines
        # skysub_filename = "OBJ_skysub_%s" % (fb)
        # pysalt.clobberfile(skysub_filename)
        # hdu_rect.writeto(skysub_filename, clobber=True)
        
    return

            


    verbose = False
    for idx, filename in enumerate(obslog['FLAT']):
        cur_op = 0

        dirname, filebase = os.path.split(filename)
        logger.info("basic reduction for frame %s (%s)" % (filename, filebase))

        hdu = salt_prepdata(filename, badpixelimage=None, create_variance=False, 
                            verbose=False)
        
        out_filename = "prep_"+filebase
        pysalt.clobberfile(out_filename)
        hdu.writeto(out_filename, clobber=True)

        # #
        # # 
        # #

        # #
        # # Prepare basic header stuff
        # #
        # after_prepare = "%s/%s/%s" % (work_dir, reduction_steps[cur_op], filebase)
        # print after_prepare
        # saltprepare(filename, after_prepare, '', createvar=False, 
        #             badpixelimage='', clobber=True, logfile=logfile, verbose=verbose)
        # cur_op += 1

        # #
        # # Apply bias subtraction
        # #
        # after_bias = "%s/%s/%s" % (work_dir, reduction_steps[cur_op], filebase)
        # saltbias(after_prepare, after_bias, '', subover=True, trim=True, subbias=False, masterbias='',  
        #       median=False, function='polynomial', order=5, rej_lo=3.0, rej_hi=5.0, 
        #       niter=10, plotover=False, turbo=False, 
        #          clobber=True, logfile=logfile, verbose=verbose)
        # cur_op += 1

        # #
        # # gain correct the data
        # #
        # #
        # after_gain = "%s/%s/%s" % (work_dir, reduction_steps[cur_op], filebase)
        # saltgain(after_bias, after_gain, '', 
        #          usedb=False, 
        #          mult=True, 
        #          clobber=True, 
        #          logfile=logfile, 
        #          verbose=verbose)
        # cur_op += 1

        # #
        # # cross talk correct the data
        # #
        # after_xtalk = "%s/%s/%s" % (work_dir, reduction_steps[cur_op], filebase)
        # saltxtalk(after_gain, after_xtalk, '', 
        #           xtalkfile = "", 
        #           usedb=False, 
        #           clobber=True, 
        #           logfile=logfile, 
        #           verbose=verbose)
        # cur_op += 1

        # #
        # # cosmic ray clean the data
        # # only clean the object data
        # #
        # after_crj = "%s/%s/%s" % (work_dir, reduction_steps[cur_op], filebase)
        # if obs_dict['CCDTYPE'][idx].count('OBJECT') and obs_dict['INSTRUME'][idx].count('RSS'):
        #     #img='xgbp'+os.path.basename(infile_list[i])
        #     saltcrclean(after_xtalk, after_crj, '', 
        #                 crtype='edge', thresh=5, mbox=11, bthresh=5.0,
        #                 flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
        #                 gbox=3, maxiter=5, 
        #                 multithread=True,  
        #                 clobber=True, 
        #                 logfile=logfile, 
        #                 verbose=verbose)
        # else:
        #     after_crj = after_xtalk
        # cur_op += 1

        # #
        # # flat field correct the data
        # #
        
      # flat_imgs=''
      # for i in range(len(infile_list)):
      #   if obs_dict['CCDTYPE'][i].count('FLAT'):
      #      if flat_imgs: flat_imgs += ','
      #      flat_imgs += 'xgbp'+os.path.basename(infile_list[i])

      # if len(flat_imgs)!=0:
      #    saltcombine(flat_imgs,flatimage, method='median', reject=None, mask=False,    \
      #           weight=True, blank=0, scale='average', statsec='[200:300, 600:800]', lthresh=3,    \
      #           hthresh=3, clobber=True, logfile=logfile, verbose=True)
      #    saltillum(flatimage, flatimage, '', mbox=11, clobber=True, logfile=logfile, verbose=True)

      #    saltflat('xgbpP*fits', '', 'f', flatimage, minflat=500, clobber=True, logfile=logfile, verbose=True)
      # else:
      #    flats=None
      #    imfiles=glob.glob('cxgbpP*fits')
      #    for f in imfiles:
      #        shutil.copy(f, 'f'+f)

      # #mosaic the data
      # #geomfile=iraf.osfn("pysalt$data/rss/RSSgeom.dat")
      # geomfile=pysalt.get_data_filename("pysalt$data/rss/RSSgeom.dat")
      # saltmosaic('fxgbpP*fits', '', 'm', geomfile, interp='linear', cleanup=True, geotran=True, clobber=True, logfile=logfile, verbose=True)

    return

    sys.exit(0)

    if imreduce:   
      #prepare the data
      saltprepare(infiles, '', 'p', createvar=False, badpixelimage='', clobber=True, logfile=logfile, verbose=True)

      #bias subtract the data
      saltbias('pP*fits', '', 'b', subover=True, trim=True, subbias=False, masterbias='',  
              median=False, function='polynomial', order=5, rej_lo=3.0, rej_hi=5.0, 
              niter=10, plotover=False, turbo=False, 
              clobber=True, logfile=logfile, verbose=True)

      #gain correct the data
      saltgain('bpP*fits', '', 'g', usedb=False, mult=True, clobber=True, logfile=logfile, verbose=True)

      #cross talk correct the data
      saltxtalk('gbpP*fits', '', 'x', xtalkfile = "", usedb=False, clobber=True, logfile=logfile, verbose=True)

      #cosmic ray clean the data
      #only clean the object data
      for i in range(len(infile_list)):
        if obs_dict['CCDTYPE'][i].count('OBJECT') and obs_dict['INSTRUME'][i].count('RSS'):
          img='xgbp'+os.path.basename(infile_list[i])
          saltcrclean(img, img, '', crtype='edge', thresh=5, mbox=11, bthresh=5.0,
                flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
                gbox=3, maxiter=5, multithread=True,  clobber=True, logfile=logfile, verbose=True)
 
      #flat field correct the data
      flat_imgs=''
      for i in range(len(infile_list)):
        if obs_dict['CCDTYPE'][i].count('FLAT'):
           if flat_imgs: flat_imgs += ','
           flat_imgs += 'xgbp'+os.path.basename(infile_list[i])

      if len(flat_imgs)!=0:
         saltcombine(flat_imgs,flatimage, method='median', reject=None, mask=False,    \
                weight=True, blank=0, scale='average', statsec='[200:300, 600:800]', lthresh=3,    \
                hthresh=3, clobber=True, logfile=logfile, verbose=True)
         saltillum(flatimage, flatimage, '', mbox=11, clobber=True, logfile=logfile, verbose=True)

         saltflat('xgbpP*fits', '', 'f', flatimage, minflat=500, clobber=True, logfile=logfile, verbose=True)
      else:
         flats=None
         imfiles=glob.glob('cxgbpP*fits')
         for f in imfiles:
             shutil.copy(f, 'f'+f)

      #mosaic the data
      #geomfile=iraf.osfn("pysalt$data/rss/RSSgeom.dat")
      geomfile=pysalt.get_data_filename("pysalt$data/rss/RSSgeom.dat")
      saltmosaic('fxgbpP*fits', '', 'm', geomfile, interp='linear', cleanup=True, geotran=True, clobber=True, logfile=logfile, verbose=True)

      #clean up the images
      if cleanup:
           for f in glob.glob('p*fits'): os.remove(f)
           for f in glob.glob('bp*fits'): os.remove(f)
           for f in glob.glob('gbp*fits'): os.remove(f)
           for f in glob.glob('xgbp*fits'): os.remove(f)
           for f in glob.glob('fxgbp*fits'): os.remove(f)


    #set up the name of the images
    if specreduce:
       for i in range(len(infile_list)):
           if obs_dict['OBJECT'][i].upper().strip()=='ARC':
               lamp=obs_dict['LAMPID'][i].strip().replace(' ', '')
               arcimage='mfxgbp'+os.path.basename(infile_list[i])
               lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.txt" % lamp)

               specidentify(arcimage, lampfile, dbfile, guesstype='rss', 
                  guessfile='', automethod=automethod,  function='legendre',  order=5, 
                  rstep=100, rstart='middlerow', mdiff=10, thresh=3, niter=5, 
                  inter=True, clobber=True, logfile=logfile, verbose=True)

               specrectify(arcimage, outimages='', outpref='x', solfile=dbfile, caltype='line', 
                   function='legendre',  order=3, inttype='interp', w1=None, w2=None, dw=None, nw=None,
                   blank=0.0, clobber=True, logfile=logfile, verbose=True)
     

    objimages=''
    for i in range(len(infile_list)):
       if obs_dict['CCDTYPE'][i].count('OBJECT') and obs_dict['INSTRUME'][i].count('RSS'):
          if objimages: objimages += ','
          objimages+='mfxgbp'+os.path.basename(infile_list[i])

    if specreduce:
      #run specidentify on the arc files

      specrectify(objimages, outimages='', outpref='x', solfile=dbfile, caltype='line', 
           function='legendre',  order=3, inttype='interp', w1=None, w2=None, dw=None, nw=None,
           blank=0.0, clobber=True, logfile=logfile, verbose=True)


    #create the spectra text files for all of our objects
    spec_list=[]
    for img in objimages.split(','):
       spec_list.extend(createspectra('x'+img, obsdate, smooth=False, skysection=skysection, clobber=True))
    print spec_list
 
    #determine the spectrophotometric standard
    extfile=pysalt.get_data_filename('pysalt$data/site/suth_extinct.dat')

    for spec, am, et, pc in spec_list:
        if pc=='CAL_SPST':
           stdstar=spec.split('.')[0]
           print stdstar, am, et
           stdfile=pysalt.get_data_filename('pysalt$data/standards/spectroscopic/m%s.dat' % stdstar.lower().replace('-', '_'))
           print stdfile
           ofile=spec.replace('txt', 'sens')
           calfile=ofile #assumes only one observations of a SP standard
           specsens(spec, ofile, stdfile, extfile, airmass=am, exptime=et,
                stdzp=3.68e-20, function='polynomial', order=3, thresh=3, niter=5,
                clobber=True, logfile='salt.log',verbose=True)
    

    for spec, am, et, pc in spec_list:
        if pc!='CAL_SPST':
           ofile=spec.replace('txt', 'spec')
           speccal(spec, ofile, calfile, extfile, airmass=am, exptime=et, 
                  clobber=True, logfile='salt.log',verbose=True)
           #clean up the spectra for bad pixels
           cleanspectra(ofile)


def speccombine(spec_list, obsdate):
   """Combine N spectra"""

   w1,f1, e1=numpy.loadtxt(spec_list[0], usecols=(0,1,2), unpack=True)

   w=w1
   f=1.0*f1
   e=e1**2

   for sfile in spec_list[1:]:
      w2,f2, e2=numpy.loadtxt(sfile, usecols=(0,1,2), unpack=True)
      if2=numpy.interp(w1, w2, f2)
      ie2=numpy.interp(w1, w2, e2)
      f2=f2*numpy.median(f1/if2)
      f+=if2
      e+=ie2**2

   f=f/len(spec_list)
   e=e**0.5/len(spec_list)

   sfile='%s.spec' % obsdate
   fout=open(sfile, 'w')
   for i in range(len(w)):
           fout.write('%f %e %e\n' % (w[i], f[i], e[i]))
   fout.close()


def cleanspectra(sfile, grow=6):
    """Remove possible bad pixels"""
    try:
        w,f,e=numpy.loadtxt(sfile, usecols=(0,1,2), unpack=True)
    except:
        w,f=numpy.loadtxt(sfile, usecols=(0,1), unpack=True)
        e=f*0.0+f.std()
    
    m=(f*0.0)+1
    for i in range(len(m)):
        if f[i]<=0.0:
           x1=int(i-grow)
           x2=int(i+grow)
           m[x1:x2]=0
    m[0]=0
    m[-1]=0

  
    fout=open(sfile, 'w')
    for i in range(len(w)):
        if m[i]:
           fout.write('%f %e %e\n' % (w[i], f[i], e[i]))
    fout.close()
 
def normalizespectra(sfile, compfile):
    """Normalize spectra by the comparison object"""

    #read in the spectra
    w,f,e=numpy.loadtxt(sfile, usecols=(0,1,2), unpack=True)
   
    #read in the comparison spectra
    cfile=sfile.replace('MCG-6-30-15', 'COMP')
    print cfile
    wc,fc,ec=numpy.loadtxt(cfile, usecols=(0,1,2), unpack=True)

    #read in the base star
    ws,fs,es=numpy.loadtxt(compfile, usecols=(0,1,2), unpack=True)
 
    #calcualte the normalization
    ifc=numpy.interp(ws, wc, fc) 
    norm=numpy.median(fs/ifc)
    print norm
    f=norm*f
    e=norm*e

    #write out the result
    fout=open(sfile, 'w')
    for i in range(len(w)):
        fout.write('%f %e %e\n' % (w[i], f[i], e[i]))
    fout.close()

    #copy

    
 

def createspectra(img, obsdate, minsize=5, thresh=3, skysection=[800,1000], smooth=False, maskzeros=True, clobber=True):
    """Create a list of spectra for each of the objects in the images"""
    #okay we need to identify the objects for extraction and identify the regions for sky extraction
    #first find the objects in the image
    hdu=pyfits.open(img)
    target=hdu[0].header['OBJECT']
    propcode=hdu[0].header['PROPID']
    airmass=hdu[0].header['AIRMASS']
    exptime=hdu[0].header['EXPTIME']

    if smooth:
       data=smooth_data(hdu[1].data)
    else:
       data=hdu[1].data

    #replace the zeros with the average from the frame
    if maskzeros:
       mean,std=iterstat(data[data>0])
       rdata=numpy.random.normal(mean, std, size=data.shape)
       print mean, std
       data[data<=0]=rdata[data<=0]

    #find the sections in the images
    section=findobj.findObjects(data, method='median', specaxis=1, minsize=minsize, thresh=thresh, niter=5)
    print section

    #use a region near the center to create they sky
    skysection=findskysection(section, skysection)
    print skysection
 
    #sky subtract the frames
    shdu=skysubtract(hdu, method='normal', section=skysection)
    if os.path.isfile('s'+img): os.remove('s'+img)
    shdu.writeto('s'+img)
 
    spec_list=[]
    #extract the spectra
    #extract the comparison spectrum
    section=findobj.findObjects(shdu[1].data, method='median', specaxis=1, minsize=minsize, thresh=thresh, niter=5)
    print section
    for j in range(len(section)):
        ap_list=extract(shdu, method='normal', section=[section[j]], minsize=minsize, thresh=thresh, convert=True)
        ofile='%s.%s_%i_%i.txt' % (target, obsdate, extract_number(img), j)
        write_extract(ofile, [ap_list[0]], outformat='ascii', clobber=clobber)
        spec_list.append([ofile, airmass, exptime, propcode])

    return spec_list

def smooth_data(data, mbox=25):
    mdata=median_filter(data, size=(mbox, mbox))
    return data-mdata

def find_section(section, y):
    """Find the section closest to y"""
    best_i=-1
    dist=1e5
    for i in range(len(section)):
        d=min(abs(section[i][0]-y), abs(section[i][1]-y))
        if d < dist:
           best_i=i
           dist=d
    return best_i

def extract_number(img):
    """Get the image number only"""
    img=img.split('.fits')
    nimg=int(img[0][-4:])
    return nimg

def iterstat(data, thresh=3, niter=5):
    mean=data.mean()
    std=data.std()
    for i in range(niter):
        mask=(abs(data-mean)<std*thresh)
        mean=data[mask].mean()
        std=data[mask].std()
    return mean, std

    

def findskysection(section, skysection=[800,900], skylimit=100):
    """Look through the section lists and determine a section to measure the sky in

       It should be as close as possible to the center and about 200 pixels wide
    """
    #check to make sure it doesn't overlap any existing spectra
    #and adjust if necessary
    for y1, y2 in section:
        if -30< (skysection[1]-y1)<0:
           skysection[1]=y1-30
        if 0< (skysection[0]-y2)<30:
           skysection[0]=y2+30
    if skysection[1]-skysection[0] < skylimit: print "WARNING SMALL SKY SECTION"
    return skysection
    


if __name__=='__main__':

    logger = pysalt.mp_logging.setup_logging()

    rawdir=sys.argv[1]
    prodir=os.path.curdir+'/'
    specred(rawdir, prodir)

    pysalt.mp_logging.shutdown_logging(logger)
