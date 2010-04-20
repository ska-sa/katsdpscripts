#!/usr/bin/python
# Reduces tipping curve data and plots tipping curve.

import sys
import optparse
import re
import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

import scape
from katpoint import rad2deg

# Set up logging: logging everything (DEBUG & above), both to console and file
logger = logging.root
logger.setLevel(logging.DEBUG)

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This reduces a data file to produce a tipping curve plot.')
parser.add_option('-a', '--baseline', dest='baseline', type="string", metavar='BASELINE', default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-n", "--nd_models", dest="nd_dir", type="string", default='',
                  help="Name of optional directory containing noise diode model files")
(opts, args) = parser.parse_args()

if len(args) < 1:
    logger.error('Please specify the data file to reduce')
    sys.exit(1)

# Load data set
logger.info("Loading baseline '%s' from data file '%s'" % (opts.baseline, args[0]))
d = scape.DataSet(args[0], baseline=opts.baseline)
# Standard reduction for XDM, more hard-coded version for FF / KAT-7
# Restrict frequency band and use noise diode firings to calibrate data from raw counts to temperature
if d.antenna.name == 'XDM':
    d = d.select(freqkeep=d.channel_select)
    d.convert_power_to_temperature()
else:
    # Hard-code the FF frequency band
    d = d.select(freqkeep=range(90, 425))
    # If noise diode models are supplied, insert them into data set before converting to temperature
    if d.antenna.name[:3] == 'ant' and os.path.isdir(opts.nd_dir):
        try:
            nd_hpol_file = os.path.join(opts.nd_dir, 'T_nd_A%sH_coupler.txt' % (d.antenna.name[3],))
            nd_vpol_file = os.path.join(opts.nd_dir, 'T_nd_A%sV_coupler.txt' % (d.antenna.name[3],))
            logger.info("Loading noise diode model '%s'" % (nd_hpol_file,))
            nd_hpol = np.loadtxt(nd_hpol_file, delimiter=',')
            logger.info("Loading noise diode model '%s'" % (nd_vpol_file,))
            nd_vpol = np.loadtxt(nd_vpol_file, delimiter=',')
            nd_hpol[:, 0] /= 1e6
            nd_vpol[:, 0] /= 1e6
            d.nd_model = scape.gaincal.NoiseDiodeModel(nd_hpol, nd_vpol, std_temp=0.04)
            d.convert_power_to_temperature()
        except IOError:
            logger.warning('Could not load noise diode model files, should be named T_nd_A1H_coupler.txt etc.')
# Only keep main scans (discard slew and cal scans)
d = d.select(labelkeep='scan', copy=False)
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
