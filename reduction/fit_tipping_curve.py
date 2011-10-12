#!/usr/bin/python
# Reduces tipping curve data and fits a tipping model to the data
# Needs: noise diode models and tipping spillover model
# The script expects to find the following files in a specified directory
# 'K7H_1200.txt', 'K7V_1200.txt', 'K7H_1600.txt', 'K7V_1600.txt', 'K7H_2000.txt', 'K7V_2000.txt'
# If the filenames or frequencies change, please modify the function interp_spillover accordingly

import sys
import optparse
import re
import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

import scape
from katpoint import rad2deg, deg2rad

# Set up logging: log everything (DEBUG & above)
logger = logging.root
logger.setLevel(logging.DEBUG)

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This reduces a data file to produce a tipping curve plot.')
parser.add_option("-a", "--baseline", default='sd',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-n", "--nd-models",
                  help="Name of optional directory containing noise diode model files")
parser.add_option("-m", "--min-nd", type="float", default=3.0,
                  help="Minimum duration of noise diode on/off segments to use for calibration, in seconds (default = %default)")
parser.add_option("-f", "--freq-chans", default='100,400',
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default %default)")
parser.add_option("-t", "--tip_models",
                  help="Name of directory containing spillover models (**required**)")
parser.add_option("-o", "--output", dest="outfilebase", default='spectra',
                  help="Base name of output files (*.png for figures), default = '%default'")

(opts, args) = parser.parse_args()

if len(args) < 1:
    logger.error('Please specify the data file to reduce')
    sys.exit(1)

if not opts.tip_models:
    logger.error('Please specify the directory containing spillover models')
    sys.exit(1)

def load_cal(filename, baseline, start_freq_channel, end_freq_channel, nd_models):
    """Load the data for the specified baseline, apply noise diode calibration and average."""
    logger.info("Loading baseline '%s' from data file '%s'" % (baseline, filename))
    d = scape.DataSet(filename, baseline=baseline, nd_models=nd_models)
    d = d.select(freqkeep=range(start_freq_channel, end_freq_channel + 1))
    d.convert_power_to_temperature(min_duration=opts.min_nd, jump_significance=10.0)
    # Only keep main scans (discard slew and cal scans)
    d = d.select(labelkeep='track',flagkeep='~nd_on', copy=False)
    # Average all frequency channels into one band
    d.average()
    return d

def interp_spillover(nu, pol):
    """Load spillover models and interpolate to centre observing frequency."""
    spillover_1200_HH_file = os.path.join(opts.tip_models, 'K7H_1200.txt')
    spillover_1200_VV_file = os.path.join(opts.tip_models, 'K7V_1200.txt')
    spillover_1600_HH_file = os.path.join(opts.tip_models, 'K7H_1600.txt')
    spillover_1600_VV_file = os.path.join(opts.tip_models, 'K7V_1600.txt')
    spillover_2000_HH_file = os.path.join(opts.tip_models, 'K7H_2000.txt')
    spillover_2000_VV_file = os.path.join(opts.tip_models, 'K7V_2000.txt')
    if pol== 'HH':
        spillover_1200 = np.loadtxt(spillover_1200_HH_file, unpack=True)
        spillover_1600 = np.loadtxt(spillover_1600_HH_file, unpack=True)
        spillover_2000 = np.loadtxt(spillover_2000_HH_file, unpack=True)
    else:
        spillover_1200 = np.loadtxt(spillover_1200_VV_file, unpack=True)
        spillover_1600 = np.loadtxt(spillover_1600_VV_file, unpack=True)
        spillover_2000 = np.loadtxt(spillover_2000_VV_file, unpack=True)

    # Extract elevations (assumes same elevation values for all models
    # Provided models are a function of zenith angle
    el = 90 - spillover_2000[0]
    sort_ind = el.argsort()
    el = el[sort_ind]
    if (nu >= 1600) and (nu <= 2000):
        spillover_nu = (nu-1600)/(2000-1600)*spillover_2000[1] + (1 - (nu-1600)/(2000-1600))*spillover_1600[1]
        return el, spillover_nu[sort_ind]
    elif (nu >= 1200) and (nu <= 1600):
        spillover_nu = (nu-1200)/(1600-1200)*spillover_1600[1] + (1 - (nu-1200)/(1600-1200))*spillover_1200[1]
        return el, spillover_nu[sort_ind]
    else:
        logger.error('Centre frequency out of valid range for spillover model.')
        sys.exit(1)

def extract_T_sys(d, nu, pol):
    """Extract tipping curve data points and surface temperature."""
    # First extract total power in each scan (both mean and standard deviation)
    power_stats = [scape.stats.mu_sigma(s.pol(pol)[:, 0]) for s in d.scans]
    tipping_mu, tipping_sigma = np.array([s[0] for s in power_stats]), np.array([s[1] for s in power_stats])
    # Extract elevation angle from (azel) target associated with scan, in degrees
    elevation = np.array([rad2deg(s.compscan.target.azel()[1]) for s in d.scans])
    # Sort data in the order of ascending elevation
    sort_ind = elevation.argsort()
    elevation, tipping_mu, tipping_sigma = elevation[sort_ind], tipping_mu[sort_ind], tipping_sigma[sort_ind]
    valid_el = (elevation >= 10)
    # Extract surface temperature from weather data
    temp_sensor = d.enviro['temperature']
    T_surface = np.mean(temp_sensor['value'])
    return  elevation[valid_el], tipping_mu[valid_el], tipping_sigma[valid_el], T_surface

def fit_tipping(elevation, T_sys, T_surface, nu, pol, spillover_el, spillover_model):
    """Fit tipping curve."""
    # T_sys(el) = T_cmb + T_gal + T_atm*(1-exp(-tau_0/sin(el))) + T_spill(el) + T_rx
    # We will fit the opacity and T_rx
    T_cmb = 2.7
    T_gal = 10 * (nu/408) ** (-2.72)
    T_atm = 1.12 * (273.15 + T_surface) - 50
    # Create a function to give the spillover at any elevation at the observing frequency
    T_spill_func = scape.fitting.PiecewisePolynomial1DFit()
    T_spill_func.fit(spillover_el, spillover_model)
    # Set up full tipping equation y = f(p, x):
    #   function input x = elevation in degrees
    #   parameter vector p = [T_rx, zenith opacity tau_0]
    #   function output y = T_sys in kelvin
    func = lambda p, x: p[0] + T_cmb + T_gal + T_spill_func(x) + T_atm * (1 - np.exp(-p[1] / np.sin(deg2rad(x))))
    # Initialise the fitter with the function and an initial guess of the parameter values
    tip = scape.fitting.NonLinearLeastSquaresFit(func, [70, 0.005])
    tip.fit(elevation, T_sys)
    logger.info('Fit results for %s polarisation:' % (pol,))
    logger.info('T_ant = %.2f K' % (tip.params[0],))
    logger.info('Zenith opacity tau_0 = %.5f' % (tip.params[1],))
    # Calculate atmospheric noise contribution at 10 degrees elevation for comparison with requirements
    T_atm_10 = (1.12 * (273.15 + T_surface) - 50) * (1 - np.exp(-tip.params[1] / np.sin(deg2rad(10))))
    logger.info('Atmospheric noise contribution at 10 degrees is: %.2f K' % (T_atm_10,))
    return tip(elevation), tip.params[0], tip.params[1]

# Frequency channels to keep
start_freq_channel = int(opts.freq_chans.split(',')[0])
end_freq_channel = int(opts.freq_chans.split(',')[1])

d = load_cal(args[0], opts.baseline, start_freq_channel, end_freq_channel, opts.nd_models)
# Centre frequency of observation
nu = d.freqs[0]  #MHz
# Fit tipping curves to the data
spillover_el_hh, spillover_model_hh = interp_spillover(nu, 'HH')
spillover_el_vv, spillover_model_vv = interp_spillover(nu, 'VV')
elevation_hh, Tsys_hh, dTsys_hh, T_surface = extract_T_sys(d, nu, 'HH')
tip_fit_hh, T_ant_hh, tau_hh = fit_tipping(elevation_hh, Tsys_hh, T_surface, nu, 'HH', spillover_el_hh, spillover_model_hh)
elevation_vv, Tsys_vv, dTsys_vv, T_surface = extract_T_sys(d, nu, 'VV')
tip_fit_vv, T_ant_vv, tau_vv = fit_tipping(elevation_vv, Tsys_vv, T_surface, nu, 'VV', spillover_el_hh, spillover_model_hh)

az = rad2deg(d.compscans[0].target.azel()[0])

# Plot the data and fits
plot_filename = opts.outfilebase + '.png'
plt.figure(10)
plt.clf()
plt.plot(elevation_hh, Tsys_hh, marker='o', color='b', label='HH', linewidth=0)
plt.errorbar(elevation_hh, Tsys_hh, dTsys_hh, ecolor='b', color='b', capsize=6, linewidth=0)
plt.plot(elevation_hh, tip_fit_hh, color='b', label="HH T_ant = %.2f K, tau_0 = %.5f" % (T_ant_hh, tau_hh))
plt.plot(elevation_vv, Tsys_vv, marker='^', color='r', label='VV', linewidth=0)
plt.errorbar(elevation_vv, Tsys_vv, dTsys_vv, ecolor='r', color='r', capsize=6, linewidth=0)
plt.plot(elevation_vv, tip_fit_vv, color='r', label="VV T_ant = %.2f K, tau_0 = %.5f" % (T_ant_vv, tau_vv))
plt.title('%s: Tipping curve for antenna %s' % (args[0], d.antenna.name))
plt.xlabel('Elevation (degrees)')
if d.data_unit == 'K':
    plt.ylabel('Temperature (K)')
else:
    plt.ylabel('Raw power (counts)')
plt.legend()
plt.savefig(plot_filename, dpi=200)

plt.show()
