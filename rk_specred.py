#!/usr/bin/env python


"""
SPECREDUCE

General data reduction script for SALT long slit data.

This includes step that are not yet included in the pipeline 
and can be used for extended reductions of SALT data. 

It does require the pysalt package to be installed 
and up to date.

"""

import os, sys, glob, shutil

import numpy as np
import pyfits
from scipy.ndimage.filters import median_filter
import bottleneck
import scipy.interpolate

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
    hdulist.info()

    logger.debug("Prepare'ing")
    hdulist = pysalt.saltred.saltprepare.prepare(
        hdulist,
        createvar=create_variance, 
        badpixelstruct=badpixel_hdu)
    # Add some history headers here


    hdulist.info()

    #
    # Overscan/bias
    #
    logger.debug("Subtracting bias & overscan")
    for ext in hdulist:
        if (not ext.data == None): print ext.data.shape
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

    print "--------------"
    for ext in hdulist:
        if (not ext.data == None): print ext.data.shape

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
        xkey=np.array(xdict.keys())
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
        hdulist = crj_function(hdulist, 
                               crtype='edge', thresh=5, mbox=11, bthresh=5.0,
                               flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
                               gbox=3, maxiter=5)
    logger.debug("done with cosmics")


    #
    # Apply flat-field correction if requested
    #
    if (not flatfield_frame == None):
        logger.debug("Applying flatfield")
        #saltflat('xgbpP*fits', '', 'f', flatimage, minflat=500, clobber=True, logfile=logfile, verbose=True)
        logger.debug("done with flatfield")

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

            # create the mosaic
            hdulist = pysalt.saltred.saltmosaic.make_mosaic(
                struct=hdulist, 
                gap=gap, xshift=xshift, yshift=yshift, rotation=rotation, 
                interp_type='linear',              
                boundary='constant', constant=0, geotran=True, fill=False,
                cleanup=True, log=None, verbose=verbose)
            logger.debug("done with mosaic")


    return hdulist


#################################################################################
#################################################################################
#################################################################################
def get_integrated_spectrum(hdu_rect, filename):
    """

    This function integrates the spectrum along the spectral axis, returning a 
    1-D array of integrated intensities. 

    """
    
    _,fb = os.path.split(filename)
    logger = logging.getLogger("GetIntegratedSpec(%s)" % (fb))

    integrated_intensity = bottleneck.nansum(hdu_rect['SCI'].data.astype(numpy.float32), axis=1)
    logger.info("Integrated intensity: covers %d pixels along slit" % (integrated_intensity.shape[0]))
    # pyfits.PrimaryHDU(data=integrated_intensity).writeto()
    numpy.savetxt("1d_%s.cat" % (fb[:-5]), integrated_intensity)

    return integrated_intensity





#################################################################################
#################################################################################
#################################################################################
def find_slit_profile(integrated_intensity, filename):
    """

    Starting with the intensity profile along the slit, reject all likely 
    sources (bright things), small-scale fluctuations (more sources, cosmics, 
    etc), and finally produce a spline-smoothed profile of intensity along the
    slit. This can then be used to correct the image data with the goal to 
    improve sky-line subtraction.

    """


    _,fb = os.path.split(filename)
    logger = logging.getLogger("FindSlitProf")

    #
    # First of all, reject all pixels with zero fluxes
    #
    bad_pixels = (integrated_intensity <= 0) | (numpy.isnan(integrated_intensity))

    # Next, find average level across the profile.
    # That way, we hopefully can reject all bright targets
    bright_lim, faint_lim = 1e9, 0
    # background = (integrated_intensity <= bright_lim) | \
    #              (integrated_intensity >= faint_lim)
    background = ~bad_pixels
    likely_background_profile = numpy.array(integrated_intensity)
    likely_background_profile[~background] = numpy.NaN
    numpy.savetxt("1d_bg_%s.cat.start" % (fb[:-5]), likely_background_profile)

    for i in range(5):
        logger.info("Iteration %d: %d valid pixels considered BG" % (i+1, numpy.sum(background)))

        # compute median of all pixels considered background
        med = bottleneck.nanmedian(integrated_intensity[background])
        std = bottleneck.nanstd(integrated_intensity[background])
        logger.info("Med/Std: %f   %f" % (med, std))
        # Now set new bright and faint limits
        bright_lim, faint_lim = med+3*std, med-3*std

        # Apply new mask to intensities
        background = background & \
                     (integrated_intensity <= bright_lim) & \
                     (integrated_intensity >= faint_lim)

        likely_background_profile = numpy.array(integrated_intensity)
        likely_background_profile[~background] = numpy.NaN
        numpy.savetxt("1d_bg_%s.cat.%d" % (fb[:-5], i+1), likely_background_profile)

    skymask = ~bad_pixels & background

    #
    # Reject small outliers by median-filtering across a number of pixels
    #
    half_window_size = 25
    
    filtered_bg = numpy.array([
        bottleneck.nanmedian(
            likely_background_profile[i-half_window_size:i+half_window_size]) 
        for i in range(likely_background_profile.shape[0])])
    numpy.savetxt("1d_bg_%s.cat.filtered" % (fb[:-5]), filtered_bg)

    #
    # To smooth things out even better, fit a very low order spline, 
    # avoiding the central area where the source likely is located
    #
    x = numpy.arange(filtered_bg.shape[0])
    t = numpy.linspace(200, 1800, 50)
    avoidance = (t>700) & (t<1300)
    do_not_fit = ((x<700) | (x>1300)) & skymask
    t = t[~avoidance]
    w = numpy.ones(filtered_bg.shape[0])
    w[~skymask] = 0
    lsq_spline = scipy.interpolate.LSQUnivariateSpline(
        x=x[do_not_fit], y=filtered_bg[do_not_fit], t=t, 
        w=None, bbox=[None, None], k=2)
    numpy.savetxt("1d_bg_%s.cat.fit" % (fb[:-5]), lsq_spline(x))
    numpy.savetxt("1d_bg_%s.cat.wt" % (fb[:-5]), w)

    #
    # Also try fitting a polynomial to the function
    #
    #numpy.polyfit(

    #
    # Now normalize this line profile so we can use it to flatten out the slit image
    #
    avg_flux = bottleneck.nanmean(filtered_bg)

    #slit_flattening = filtered_bg / avg_flux
    slit_flattening = lsq_spline(x) / avg_flux
    # Fill in gaps with ones
    slit_flattening[numpy.isnan(slit_flattening)] = 1.0

    return slit_flattening.reshape((-1,1)), skymask




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

    reduction_steps = [
        '01.prepare',
        '02.bias',
        '03.gain',
        '04.xtalk',
        '05.crjclean',
        '06.flat',
        '07.',
        '08.',
    ]

    #
    # Make sure we have all directories 
    #
    for rs in reduction_steps:
        dirname = "%s/%s" % (work_dir, rs)
        if (not os.path.isdir(dirname)):
            os.mkdir(dirname)

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
    for idx, filename in enumerate(obslog['FLAT']):
        hdulist = open(filename)
        if (hdulist[0].header['OBSTYPE'].find("FLAT") >= 0 and
            hdulist[0].header['INSTRUME'] == "RSS"):
            #
            # This is a flat-field
            #
            flatfield_filenames.append(filename)
            
            hdu = salt_prepdata(filename, badpixelimage=None, create_variance=False, 
                                verbose=False)
            flatfield_hdus.append(hdu)
        
    if (len(obslog['FLAT']) > 0):
        # We have some flat-field files
        # run some combination method
        pass



    #
    # Determine a wavelength solution from ARC frames, where available
    #
    logger.info("Searching for a wavelength calibration from the ARC files")
    skip_wavelength_cal_search = os.path.isfile(dbfile)
    if (not skip_wavelength_cal_search):
        for idx, filename in enumerate(obslog['ARC']):
            _, fb = os.path.split(filename)
            hdulist = open(filename)

            arc_filename = "ARC_%s" % (fb)
            rect_filename = "ARC-RECT_%s" % (fb)
            logger.info("Creating mosaic for frame %s --> %s" % (fb, arc_filename))

            hdu = salt_prepdata(filename, 
                                badpixelimage=None, 
                                create_variance=False,
                                clean_cosmics=False,
                                mosaic=True,
                                #mosaic=False,
                                verbose=False)

            pysalt.clobberfile(arc_filename)
            hdu.writeto(arc_filename, clobber=True)

            lamp=hdu[0].header['LAMPID'].strip().replace(' ', '')
            lampfile=pysalt.get_data_filename("pysalt$data/linelists/%s.txt" % lamp)
            automethod='Matchlines'
            skysection=[800,1000]
            logger.info("Searching for wavelength solution (lamp:%s, arc-image:%s)" % (
                lamp, arc_filename))
            specidentify(arc_filename, lampfile, dbfile, guesstype='rss', 
                         guessfile='', automethod=automethod,  function='legendre',  order=5, 
                         rstep=100, rstart='middlerow', mdiff=10, thresh=3, niter=5, 
                         inter=False, clobber=True, logfile=logfile, verbose=True)
            logger.debug("Done with specidentify")

            logger.debug("Starting specrectify")
            specrectify(arc_filename, outimages=rect_filename, outpref='',
                        solfile=dbfile, caltype='line', 
                        function='legendre',  order=3, inttype='interp', 
                        w1=None, w2=None, dw=None, nw=None,
                        blank=0.0, clobber=True, logfile=logfile, verbose=True)

            logger.debug("Done with specrectify")


    #
    # Now apply wavelength solution found above to your data frames
    #
    logger.info("Applying wavelength solution to OBJECT frames")
    for idx, filename in enumerate(obslog['OBJECT']):
        _, fb = os.path.split(filename)
        hdulist = open(filename)

        mosaic_filename = "OBJ_raw__%s" % (fb)
        out_filename = "OBJ_%s" % (fb)

        logger.info("Creating mosaic for frame %s --> %s" % (fb, mosaic_filename))
        hdu = salt_prepdata(filename, 
                            badpixelimage=None, 
                            create_variance=False, 
                            clean_cosmics=True,
                            mosaic=True,
                            verbose=False)
        #
        # Trial: replace all 0 value pixels with NaNs
        #
        for ext in hdu[1:]:
            ext.data[ext.data <= 0] = numpy.NaN
        pysalt.clobberfile(mosaic_filename)
        hdu.writeto(mosaic_filename, clobber=True)

        pysalt.clobberfile(out_filename)
        # spectrectify writes to disk, no need to do so here
        specrectify(mosaic_filename, outimages=out_filename, outpref='', 
                    solfile=dbfile, caltype='line', 
                    function='legendre',  order=3, inttype='interp', 
                    w1=None, w2=None, dw=None, nw=None,
                    blank=0.0, clobber=True, logfile=logfile, verbose=True)
        
        #
        # Now we have a full 2-d spectrum, but still with emission lines
        #
        
        #
        # Next, find good regions with no source contamation
        #
        hdu_rect = pyfits.open(out_filename)
        hdu_rect.info()

        intspec = get_integrated_spectrum(hdu_rect, out_filename)
        slitprof, skymask = find_slit_profile(intspec, out_filename)
        print skymask.shape[0]

        hdu_rect['SCI'].data /= slitprof

        rectflat_filename = "OBJ_flat_%s" % (fb)
        pysalt.clobberfile(rectflat_filename)
        hdu_rect.writeto(rectflat_filename, clobber=True)

        #
        # Block out the central region of the chip as object
        #
        skymask[750:1300] = False
        sky_lines = bottleneck.nanmedian(
            hdu_rect['SCI'].data[skymask].astype(numpy.float64),
            axis=0)
        print sky_lines.shape
        
        #
        # Now subtract skylines
        #
        hdu_rect['SCI'].data -= sky_lines
        skysub_filename = "OBJ_skysub_%s" % (fb)
        pysalt.clobberfile(skysub_filename)
        hdu_rect.writeto(skysub_filename, clobber=True)
        
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

   w1,f1, e1=np.loadtxt(spec_list[0], usecols=(0,1,2), unpack=True)

   w=w1
   f=1.0*f1
   e=e1**2

   for sfile in spec_list[1:]:
      w2,f2, e2=np.loadtxt(sfile, usecols=(0,1,2), unpack=True)
      if2=np.interp(w1, w2, f2)
      ie2=np.interp(w1, w2, e2)
      f2=f2*np.median(f1/if2)
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
        w,f,e=np.loadtxt(sfile, usecols=(0,1,2), unpack=True)
    except:
        w,f=np.loadtxt(sfile, usecols=(0,1), unpack=True)
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
    w,f,e=np.loadtxt(sfile, usecols=(0,1,2), unpack=True)
   
    #read in the comparison spectra
    cfile=sfile.replace('MCG-6-30-15', 'COMP')
    print cfile
    wc,fc,ec=np.loadtxt(cfile, usecols=(0,1,2), unpack=True)

    #read in the base star
    ws,fs,es=np.loadtxt(compfile, usecols=(0,1,2), unpack=True)
 
    #calcualte the normalization
    ifc=np.interp(ws, wc, fc) 
    norm=np.median(fs/ifc)
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
       rdata=np.random.normal(mean, std, size=data.shape)
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
