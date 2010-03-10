#!/usr/bin/python
# Reduces tipping curve data and plots tipping curve.

import sys
import optparse
import re

import numpy as np
import matplotlib.pyplot as plt

import scape
from katpoint import rad2deg

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This reduces a data file to produce a tipping curve plot.')
parser.add_option('-a', '--baseline', dest='baseline', type="string", metavar='BASELINE', default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print 'Please specify the data file to reduce'
    sys.exit(1)

# Load data set
print 'Loading baseline', opts.baseline, 'from data file', args[0]
d = scape.DataSet(args[0], baseline=opts.baseline)
# Use noise diode firings to calibrate data from raw counts to temperature
d.convert_power_to_temperature()
# Only keep main scans (discard slew and cal scans) and restrict frequency band to Fringe Finder band
d = d.select(labelkeep='scan', freqkeep=range(90, 425))
# Average all frequency channels into one band
d.average()

def plot_tipping_curve(pol, color='b'):
    """Calculate and plot tipping curve for polarisation *pol* in color *color*."""
    # First extract total power in each scan (both mean and standard deviation)
    power_stats = [scape.stats.mu_sigma(s.pol(pol)[:, 0]) for s in d.scans]
    tipping_mu, tipping_sigma = np.array([s[0] for s in power_stats]), np.array([s[1] for s in power_stats])
    # Extract elevation angle from (azel) target associated with scan, in degrees
    elevation = np.array([rad2deg(s.compscan.target.azel()[1]) for s in d.scans])
    # Sort data in the order of ascending elevation
    sort_ind = elevation.argsort()
    elevation, tipping_mu, tipping_sigma = elevation[sort_ind], tipping_mu[sort_ind], tipping_sigma[sort_ind]

    # Plot Tsys as a function of elevation, aka 'tipping curve' aka 'sky dip'
    plt.errorbar(elevation, tipping_mu, tipping_sigma, ecolor=color, color=color, capsize=6)
    plt.plot(elevation, tipping_mu, '-o', color=color, linewidth=2, label=pol)
    plt.xlim(0., 90.)
    # Large error bars look bad - improve plot appearance by extending y axis limits if needed
    min_yrange = 10 * 2 * tipping_sigma.max()
    current_ylim = plt.ylim()
    current_yrange = np.diff(current_ylim)[0]
    if current_yrange < min_yrange:
        adjust = (min_yrange - current_yrange) / 2.0
        plt.ylim(current_ylim[0] - adjust, current_ylim[1] + adjust)
    plt.xlabel('Elevation (degrees)')
    if d.data_unit == 'K':
        plt.ylabel('Temperature (K)')
    else:
        plt.ylabel('Raw power (counts)')

# Calculate and plot tipping curves
plt.figure(1)
plt.clf()
plot_tipping_curve('HH', 'b')
plot_tipping_curve('VV', 'r')
plt.title('Tipping curve for antenna %s' % (d.antenna.name,))
plt.legend()

# Display plots - this should be called ONLY ONCE, at the VERY END of the script
# The script stops here until you close the plots...
plt.show()
