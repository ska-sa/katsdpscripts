#! /usr/bin/python
# Example script that fits gain curve to point source scan data product.
#
# First run the analyse_point_source_scans.py script to generate the data file
# that serves as input to this script.
#
# The procedure follows the description in the GBT Commissioning Memo on "Gain
# and Efficiency at S-Band" by Ghigo, Maddalena, Balser and Langston.
#
# Ludwig Schwardt
# 12 November 2009
#

import sys
import optparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import katpoint

# These fields in data file contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file>",
                               description="This fits a gain curve to the given data file.")
(options, args) = parser.parse_args()
if len(args) < 1:
    print 'Please specify the name of data file to process'
    sys.exit(1)
filename = args[0]

# Set up logging: logging everything (DEBUG & above)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.root
logger.setLevel(logging.DEBUG)

# Load data file in one shot as an array of strings
data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
# Interpret first non-comment line as header
fields = data[0].tolist()
# By default, all fields are assumed to contain floats
formats = np.tile('float32', len(fields))
# The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
# Convert to heterogeneous record array
data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fields, formats))
# Load antenna description string from first line of file and construct antenna object from it
antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])

# Make sure we only use data that had a successful noise diode calibration
# If the noise diode failed to fire, the data unit stays as 'raw' and the gain would be completely wrong
noise_diode_fired = data['data_unit'] == 'K'
data = data[noise_diode_fired]
# Also focus on the stronger sources with better signal-to-noise ratio
strong_sources = data['flux'] > 40.
# Work with data points where all three scans across source were used in beam fitting
all_scans_good = (data['refined_HH'] == 3) & (data['refined_VV'] == 3)
good = strong_sources & all_scans_good
if not np.any(good):
    print 'No good data was found (no noise diode fired, sources too weak or beam fits all bad)'
    sys.exit(1)

# Extract desired fields from data
elev = data['elevation']
targets = data['target']
# List of unique targets in data set and target index for each data point
unique_targets = np.unique(targets).tolist()
target_indices = np.array([unique_targets.index(t) for t in targets])
# Source (total) flux density [Jy]
S = data['flux']

# Dish geometric area
A = np.pi * (antenna.diameter / 2.0) ** 2
# Boltzmann constant [J / K]
boltzmann_k = 1.3806504e-23
# Radio flux density of 1 Jansky [W / m^2 / Hz]
Jy = 1e-26
# Aperture efficiency is obtained by equating received power Pant = 0.5 Ae S Jy B = k Tant B,
# with Ae the effective area in m^2, S the total flux density in Jy, Jy the above conversion factor
# to SI units, B the bandwidth in Hz, k Boltzmann's constant and Tant the antenna temperature in K.
# Therefore, the effective area Ae = 2 k Tant / S / Jy = 2 k G / Jy = eff A.
# First get theoretical gain achievable if dish is 100% efficient, about A / 2761 [K / Jy]
gain_ideal = A * Jy / (2.0 * boltzmann_k)
# Atmospheric attenuation is modelled as G = G0 exp(-tau airmass), where tau is the optical depth
# or *zenith opacity*, and airmass is assumed to be
airmass = 1.0 / np.sin(katpoint.deg2rad(elev))
# Grid on which to evaluate fitted atmospheric models
elev_grid = np.arange(3.0, 90.0, 0.5)
airmass_grid = 1.0 / np.sin(katpoint.deg2rad(elev_grid))

plt.figure(1)
plt.clf()
ylimits = np.zeros((4, 4))

for pol_ind, pol in enumerate(['HH', 'VV']):
    # Antenna temperature Tant (due to source) for each polarisation [K]
    Tant = data['beam_height_' + pol]
    # System temperature Tsys for each polarisation can be estimated from the baseline height [K]
    Tsys = data['baseline_height_' + pol]

    # Calculate gain (always defined *per receiver chain / polarisation*) for each observation
    # Gain = (antenna temperature per polarisation) / (total source flux density) [K / Jy]
    gain = Tant / S
    # Aperture efficiency is ratio of actual gain to theoretical maximum gain
    eff = gain / gain_ideal
    # Calculate system equivalent flux density (SEFD) [Jy]
    SEFD = Tsys / gain
    # Fit simple exponential atmospheric attenuation model (straight line in log domain)
    minus_tau, lng0 = np.polyfit(airmass[good], np.log(gain[good]), deg=1)
    tau, gain0 = -minus_tau, np.exp(lng0)
    gain_model = gain0 * np.exp(-tau * airmass_grid)
    # Fit simple atmospheric emission model Tsys = Trest + Tatm (1 - exp(-tau airmass)), which
    # models Tatm, the elevation-dependent atmospheric contribution to Tsys, while the remaining
    # components of Tsys are lumped together in the Trest term (mostly receiver noise?)
    # Only fit this model on low elevations, as the tipping curve rises again at high elevations
    # due to the prime-focus feed spilling over and picking up the ground
    low_elev = elev < 45.
    z = 1.0 - np.exp(-tau * airmass)
    Tatm, Trest = np.polyfit(z[good & low_elev], Tsys[good & low_elev], deg=1)
    Tsys_model = Trest + Tatm * (1.0 - np.exp(-tau * airmass_grid))
    low_grid = elev_grid <= max(elev[low_elev])

    logger.info("Polarisation %s:" % (pol,))
    logger.info("----------------")
    logger.info("theoretical gain = %g K/Jy (%g Jy/K)" % (gain_ideal, 1. / gain_ideal))
    logger.info("median gain = %g K/Jy (%g Jy/K)" % (np.median(gain[good]), 1. / np.median(gain[good])))
    logger.info("median eff = %g %%" % (100 * np.median(eff[good]), ))
    logger.info("median Tsys = %g K" % (np.median(Tsys[good]), ))
    logger.info("median SEFD = %g Jy" % (np.median(SEFD[good]), ))
    logger.info("Atmospheric model: tau = %g, G0 = %g K/Jy, Tatm = %g K, Trest = %g K\n"
                % (tau, gain0, Tatm, Trest))

    plt.figure(1)
    per_target_elev = 90. * target_indices + elev
    plt.subplot(2, 1, pol_ind + 1)
    plt.plot(per_target_elev[~good], 100 * eff[~good], color='1.0', marker='o', linestyle='none')
    plt.plot(per_target_elev[good], 100 * eff[good], 'bo')
    for target_ind in xrange(len(unique_targets)):
        plt.text(90. * target_ind + 45., 90., unique_targets[target_ind], ha='center', va='center')
        plt.axvline(90. * target_ind, linestyle='--', color='k')
    plt.xticks([])
    plt.axis((0.0, 90.0 * len(unique_targets), 0.0, 100.0))
    plt.grid()
    if pol_ind == 1:
        plt.xlabel('Elevation per target')
    plt.ylabel('Aperture efficiency (%)')
    plt.title('Aperture efficiency per target for %s polarisation' % pol)

    plt.figure(2 + pol_ind)
    plt.clf()
    ax1 = plt.subplot(411)
    plt.plot(elev[good], gain[good], 'bo')
    plt.plot(elev_grid, gain_model, 'r')
    ylimits[0, (2*pol_ind):(2*pol_ind + 2)] = ax1.get_ylim()
    ax1.set_xticklabels([])
    plt.ylabel('Gain (K/Jy)')
    plt.title('Gain and tipping curve for %s polarisation' % pol)
    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(elev[good], 100 * eff[good], 'bo')
    plt.plot(elev_grid, 100 * gain_model / gain_ideal, 'r')
    ylimits[1, (2*pol_ind):(2*pol_ind + 2)] = ax2.get_ylim()
    ax2.set_xticklabels([])
    plt.ylabel('Efficiency (%)')
    ax3 = plt.subplot(413, sharex=ax1)
    plt.plot(elev[good], Tsys[good], 'bo')
    plt.plot(elev_grid[low_grid], Tsys_model[low_grid], 'r')
    ylimits[2, (2*pol_ind):(2*pol_ind + 2)] = ax3.get_ylim()
    ax3.set_xticklabels([])
    plt.ylabel('Tsys (K)')
    ax4 = plt.subplot(414, sharex=ax1)
    plt.plot(elev[good], SEFD[good], 'bo')
    plt.plot(elev_grid[low_grid], Tsys_model[low_grid] / gain_model[low_grid], 'r')
    ylimits[3, (2*pol_ind):(2*pol_ind + 2)] = ax4.get_ylim()
    ax4.set_xlim(0, 90)
    plt.xlabel('Elevation (deg)')
    plt.ylabel('SEFD (Jy)')

# Ensure that corresponding subplots have identical axis limits
for ax_ind, ax in enumerate(plt.figure(2).axes):
    ax.set_ylim([ylimits[ax_ind, :].min(), ylimits[ax_ind, :].max()])
for ax_ind, ax in enumerate(plt.figure(3).axes):
    ax.set_ylim([ylimits[ax_ind, :].min(), ylimits[ax_ind, :].max()])

# Hand over control to the main GUI loop
# Display plots - this should be called ONLY ONCE, at the VERY END of the script
# The script stops here until you close the plots...
plt.show()
