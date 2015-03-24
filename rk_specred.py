#!/usr/bin/env python


"""
SPECREDUCE

Generdal data reduction script for SALT long slit data.

This includes step that are not yet included in the pipeline 
and can be used for extended reductions of SALT data. 

It does require the pysalt package to be installed 
and up to date.

"""

import os, sys, glob, shutil

import numpy as np
import pyfits
from scipy.ndimage.filters import median_filter

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

def salt_prepdata(infile, badpixelimage=None, create_variance=False, 
                  verbose=False, *args):

    logger = logging.getLogger("PrepData")
    logger.info("Working on file %s" % (infile))

    hdulist = pyfits.open(infile)
    
    pysalt_log = None #'pysalt.log'

    badpixel_hdu = None
    if (not badpixelimage == None):
        badpixel_hdu = pyfits.open(badpixelimage)
    
    #
    # Do some prepping
    #
    hdulist = pysalt.saltred.saltprepare.prepare(
        hdulist,
        createvar=create_variance, 
        badpixelstruct=badpixel_hdu)
    # Add some history headers here

    #
    # Overscan/bias
    #
    hdulist = pysalt.saltred.saltbias.bias(hdulist, *args)
    # Again, add some headers here

    #
    # Gain
    #
    dblist = [] #saltio.readgaindb(gaindb)
    hdulist = pysalt.saltred.saltgain.gain(hdulist,
                   mult=True, 
                   usedb=False, 
                   dblist=dblist, 
                   log=pysalt_log, verbose=verbose)

    #
    # Xtalk
    #
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

    #
    # crj-clean
    #
    #clean the cosmic rays
    multithread = True

    if multithread and len(hdulist)>1:
        crj_function = pysalt.saltred.saltcrclean.multicrclean
    else:
        crj_function = pysalt.saltred.saltcrclean.crclean

    hdulist = crj_function(hdulist, 
                           crtype='edge', thresh=5, mbox=11, bthresh=5.0,
                           flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
                           gbox=3, maxiter=5)

    return hdulist

    
def specred(rawdir, prodir, imreduce=True, specreduce=True, calfile=None, lamp='Ar', automethod='Matchlines', skysection=[800,1000], cleanup=True):
    print rawdir
    print prodir

    logger = logging.getLogger("SPECRED")

    #get the name of the files
    infile_list=glob.glob(rawdir+'*.fits')
    infiles=','.join(['%s' % x for x in infile_list])
    

    #get the current date for the files
    obsdate=os.path.basename(infile_list[0])[1:9]
    print obsdate

    #set up some files that will be needed
    logfile='spec'+obsdate+'.log'
    flatimage='FLAT%s.fits' % (obsdate)
    dbfile='spec%s.db' % obsdate

    #create the observation log
    obs_dict=obslog(infile_list)

    import pysalt.lib.saltsafeio as saltio
    
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
    file_type_dict = {
        'FLAT': [],
        'ARC': [],
        'OBJECT': [],
    }

    for idx, filename in enumerate(infile_list):
        hdulist = open(filename)
        if (hdulist[0].header['CCDTYPE'].find("FLAT") >= 0 and
            hdulist[0].header['INSTRUME'] == "RSS"):

            file_type_dict['FLAT'].append(filename)
        elif (hdulist[0].header['CCDTYPE']
            
    #
    # Go through the list of files, find all flat-fields, and create a master flat field
    #
    logger.info("Searching for files identified as FLAT fields")
    flatfield_filenames = []
    for idx, filename in enumerate(infile_list):
        hdulist = open(filename)
        if (hdulist[0].header['CCDTYPE'].find("FLAT") >= 0 and
            hdulist[0].header['INSTRUME'] == "RSS"):
            #
            # This is a flat-field
            #
            flatfield_filenames.append(filename)
            
            hdu = salt_prepdata(filename, badpixelimage=None, create_variance=False, 
                                verbose=False)
            flatfield_hdus.append(hdu)
        
    if (len(flatfield_filenames) > 0):
        # We have some flat-field files
        # run some combination method

    #
    # Determine a wavelength solution from ARC frames, where available
    #
    for idx, filename in enumerate(infile_list):
        hdulist = open(filename)

    verbose = False
    for idx, filename in enumerate(infile_list):
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
