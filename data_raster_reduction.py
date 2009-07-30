#!/usr/bin/python
# Reduces captured data and plots using scape. Data must be local
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

pl.figure()
pl.title("Compound scan in time - pre averaging and fitting")
scape.plot_compound_scan_in_time(d.compscans[0])

d.average()
d.fit_beams_and_baselines()

pl.figure()
pl.title("Compound scan on target with fitted beam")
scape.plot_compound_scan_on_target(d.compscans[0])

pl.figure()
pl.title("Compound scan in time - post averaging and fitting")
scape.plot_compound_scan_in_time(d.compscans[0])

pl.show()
raw_input("Hit enter to finish.")
sys.exit()
