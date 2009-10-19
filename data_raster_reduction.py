#!/usr/bin/python
# Reduces captured data and plots using scape. Data must be local
import scape
import pylab as pl
import os
import sys
import ffuilib as ffui

data_file = ""
p = os.listdir(ffui.defaults.ff_directories["data"])
# p.sort(reverse=True)
while p:
    x = p.pop() # pops off the bottom of the list
    if x.endswith("001.h5"):
        data_file = ffui.defaults.ff_directories["data"] + "/" + x
        break

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
