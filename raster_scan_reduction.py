#!/usr/bin/python
# Reduces and plots raster scan data using scape.

import scape
import matplotlib.pyplot as plt
import sys
import optparse

parser = optparse.OptionParser(usage="%prog [options] <data file>",
                               description="Reduces and plots raster scan data in given file using scape.")
parser.add_option('-a', '--baseline', dest='baseline', type="string", metavar='BASELINE', default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print 'Please specify the data file to reduce'
    sys.exit(1)

# Load data set
print 'Loading baseline', opts.baseline, 'from data file', args[0]
d = scape.DataSet(args[0], baseline=opts.baseline)
d = d.select(labelkeep='scan')

plt.figure()
plt.title('Compound scan in time - pre averaging and fitting')
scape.plot_compound_scan_in_time(d.compscans[0])

d.average()
d.fit_beams_and_baselines()

plt.figure()
plt.title('Compound scan on target with fitted beam')
scape.plot_compound_scan_on_target(d.compscans[0])

plt.figure()
plt.title('Compound scan in time - post averaging and fitting')
scape.plot_compound_scan_in_time(d.compscans[0])

# Display plots - this should be called ONLY ONCE, at the VERY END of the script
# The script stops here until you close the plots...
plt.show()
