#!/usr/bin/python
# Reduces Tsys scan data. Data must be local.

import scape
import matplotlib.pyplot as plt
import os

# Pick newest file in data directory, based on name (assumed to contain timestamp)
data_dir = '/var/kat/data/'
data_file = data_dir + sorted(os.listdir(data_dir), reverse=True)[0]
print "Reducing data file", data_file

# Load data set
d = scape.DataSet(data_file)
# Use noise diode firings to calibrate data from raw counts to temperature
d.convert_power_to_temperature()
# Only keep main scans (discard slew and cal scans)
d = d.select(labelkeep="scan")

# Plot Tsys as a function of frequency
plt.figure(1)
plt.clf()
scape.plot_spectrum(d)
plt.title("Tsys at zenith")

plt.show()
