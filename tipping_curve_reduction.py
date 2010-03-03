#!/usr/bin/python
# Reduces tipping curve data. Data must be local.

# TODO: update this for the latest scape

import os
import optparse
import numpy as np
import matplotlib.pyplot as plt
import scape
from katpoint import rad2deg
import katuilib as katui

# Default data directory
data_dir = katui.defaults.kat_directories["data"]

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog <data file>',
                               description='This reduces a tipping curve in data file (or the newest one in %s).'
                                           % (data_dir,))
(options, args) = parser.parse_args()
if len(args) < 1:
    # Pick newest file in data directory, based on name (assumed to contain timestamp)
    data_file = os.path.join(data_dir, sorted(os.listdir(data_dir), reverse=True)[0])
else:
    # Use given data file
    data_file = args[0]
print 'Reducing data file', data_file

# Load data set
d = scape.DataSet(data_file)
# Use noise diode firings to calibrate data from raw counts to temperature
d.convert_power_to_temperature()
# Only keep main scans (discard slew and cal scans)
d = d.select(labelkeep='scan')
# Average all frequency channels into one band
d.average()

# Calculate tipping curve
# First extract total power in each scan (both mean and standard deviation)
compscan_power = scape.stats.ms_hstack([scape.stats.mu_sigma(s.stokes('I').squeeze()) for s in d.scans])
# Extract elevation angle of each compound scan, in radians
compscan_el = [cs.target.azel()[1] for cs in d.compscans]
# Pick lowest mean power at each unique elevation angle as Tsys estimate for that elevation
elevation = np.unique(compscan_el)
lowest_power_ind = [np.ma.masked_array(compscan_power.mu, compscan_el != el).argmin() for el in elevation]
tipping = compscan_power[lowest_power_ind]

# Plot Tsys as a function of elevation, aka 'tipping curve' aka 'sky dip'
plt.figure(1)
plt.clf()
plt.errorbar(rad2deg(elevation), tipping.mu, tipping.sigma, capsize=6)
plt.plot(rad2deg(elevation), tipping.mu, '-ob', linewidth=2)
plt.xlim(0., 90.)
# Large error bars look bad - improve plot appearance by extending y axis limits if needed
min_yrange = 10 * 2 * tipping.sigma.max()
current_ylim = plt.ylim()
current_yrange = np.diff(current_ylim)[0]
if current_yrange < min_yrange:
    adjust = (min_yrange - current_yrange) / 2.0
    plt.ylim(current_ylim[0] - adjust, current_ylim[1] + adjust)
plt.title('Tipping curve')
plt.xlabel('Elevation (degrees)')
if d.data_unit == 'K':
    plt.ylabel('Temperature (K)')
else:
    plt.ylabel('Raw power')

# Display plots - this should be called ONLY ONCE, at the VERY END of the script
plt.show()
