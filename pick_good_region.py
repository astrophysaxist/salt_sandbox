#!/usr/bin/env python

import numpy, scipy, scipy.stats
import os, sys

i = numpy.loadtxt(sys.argv[1])

stats = scipy.stats.scoreatpercentile(i, [50, 16,84, 2.5,97.5])

one_sigma = (stats[4] - stats[3]) / 4.
median = stats[0]

bad = i < median-1*one_sigma
 
i[bad] = numpy.NaN
numpy.savetxt("output", i)
