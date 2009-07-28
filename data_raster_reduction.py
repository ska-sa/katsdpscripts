#!/usr/bin/python

import scape
import pylab as pl
import os
import sys

p = os.listdir("/var/kat/data/")
p.sort(reverse=True)
data_file = "/var/kat/data/" + p[0]

print "Reducing data file",data_file

d = scape.DataSet(data_file)
d = d.select(labelkeep="scan")
d.fit_beams_and_baselines()

scape.plot_compound_scan_on_target(d.compscans[0])
pl.show()
raw_input("Hit enter to finish.")
sys.exit()
