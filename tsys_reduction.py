#!/usr/bin/python
# Reduces Tsys scan data. Data must be local.

import os
import optparse
import matplotlib.pyplot as plt
import scape
import ffuilib as ffui

# Default data directory
data_dir = ffui.defaults.ff_directories["data"]

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog <data file>',
                               description='This reduces a Tsys scan in data file (or the newest one in %s).'
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

# Plot Tsys as a function of frequency
plt.figure(1)
plt.clf()
scape.plot_spectrum(d)
plt.title('Tsys at zenith')

# Average all frequency channels into one band
d.average()
# Get mean and standard deviation of total power in first scan, assumed to be Tsys
tsys = scape.stats.mu_sigma(d.scans[0].stokes('I').squeeze())

print 'Tsys = %f +- %f %s' % (tsys.mu, tsys.sigma, d.data_unit)

# Display plots - this should be called ONLY ONCE, at the VERY END of the script
plt.show()
