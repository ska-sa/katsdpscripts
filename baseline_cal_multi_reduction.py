#! /usr/bin/python
# Baseline calibration for multiple baselines.
#
# Ludwig Schwardt
# 8 April 2010
#

import optparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import h5py
import scape
import katpoint

# Standardised baseline order used to build data matrix
baselines = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
# Offset to add to all timestamps, in seconds
time_offset = 0.0
# Array position used for fringe stopping
array_ant = katpoint.Antenna('ant0, -30:43:17.3, 21:24:38.5, 1038.0, 0.0')
# Frequency channel range to keep
channel_from, channel_to = 98, 418
# Original estimated cable lengths to 12-m container
ped_to_12m = {'ant1': 95.5, 'ant2': 108.8, 'ant3': 95.5 + 50, 'ant4': 95.5 + 70}
# Estimated Losberg cable lengths to start from, in metres
ped_to_losberg = {'ant1' : 4977.4, 'ant2' : 4988.8, 'ant3' : 5011.8, 'ant4' : 5035.4,
                  'ant5' : 5067.3, 'ant6' : 5090.9, 'ant7' : 5143.9}
# The speed of light in the fibres is lower than in vacuum
cable_lightspeed = katpoint.lightspeed / 1.4

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file>*]")
parser.add_option('-p', '--pol', type='choice', choices=['H', 'V'], default='H',
                  help="Polarisation term to use ('H' or 'V'), default is %default")
parser.add_option('-r', '--ref', help="Reference antenna, default is first antenna in file")
parser.add_option('-s', '--max-sigma', type='float', default=0.05,
                  help="Threshold on std deviation of normalised group delay, default is %default")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print 'Please specify at least one data file to reduce'
    sys.exit(1)

# Open data file
f = h5py.File(args[0])

# Extract antenna objects and input map
ant_names = f['Antennas'].listnames()
ants = [katpoint.Antenna(f['Antennas'][name].attrs['description']) for name in ant_names]
try:
    inputs = [f['Antennas'][name][opts.pol].attrs['dbe_input'] for name in ant_names]
except KeyError:
    raise KeyError("Polarisation '%s' not found for some antenna" % (opts.pol,))
reverse_input_map = dict([(v, k) for k, v in f['Correlator']['input_map'].value])
# Reference antenna and the rest
ref_ant_ind = [ant.name for ant in ants].index(opts.ref) if opts.ref is not None else 0
old_positions = np.array([ant.position_enu for ant in ants])
old_cable_lengths = np.array([ped_to_losberg[ant.name] for ant in ants])
old_receiver_delays = old_cable_lengths / cable_lightspeed
# Extract frequency information
band_center = f['Correlator'].attrs['center_frequency_hz']
channel_bw = f['Correlator'].attrs['channel_bandwidth_hz']
num_chans = f['Correlator'].attrs['num_freq_channels']
# Assume that lower-sideband downconversion has been used, which flips frequency axis
# Also subtract half a channel width to get frequencies at center of each channel
channel_freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
channel_freqs = channel_freqs[channel_from:channel_to]
dump_rate = f['Correlator'].attrs['dump_rate_hz']
sample_period = 1.0 / dump_rate
# Phase differences are associated with frequencies at midpoints between channels
mid_freqs = np.convolve(channel_freqs, [0.5, 0.5], mode='valid')
freq_diff = np.abs(np.diff(channel_freqs))

# Since the phase slope is sampled, the derived delay exhibits aliasing with a period of 1 / channel_bandwidth
# The maximum delay that can be reliably represented is therefore +- 0.5 / channel_bandwidth
max_delay = 0.5 / channel_bw
# Maximum standard deviation of delay occurs when delay samples are uniformly distributed between +- max_delay
# In this case, delay is completely random / unknown, and its estimate cannot be trusted
# The division by sqrt(N-1) converts the per-channel standard deviation to a per-snapshot deviation
max_sigma_delay = np.abs(2 * max_delay) / np.sqrt(12) / np.sqrt(len(channel_freqs) - 1)

augmented_targetdir, group_delay, sigma_delay = [], [], []
for compscan in f['Scans']:
    compscan_group = f['Scans'][compscan]
    target = katpoint.Target(compscan_group.attrs['target'])
    for scan in compscan_group:
        scan_group = compscan_group[scan]
        if scan_group.attrs['label'] != 'scan':
            continue
        # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
        # Also move correlator data timestamps from start of each sample to the middle
        timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0 + 0.5 * sample_period + time_offset
        # Obtain unit vectors pointing from array antenna to target for each timestamp in scan
        az, el = target.azel(timestamps, array_ant)
        targetdir = np.array(katpoint.azel_to_enu(az, el))
        # Invert sign of target vector, as positive dot product with baseline implies negative delay / advance
        # Augment target vector with a 1, as this allows fitting of constant (receiver) delay
        scan_augmented_targetdir = np.vstack((-targetdir, np.ones(len(el))))
        for ant_A, ant_B in baselines:
            # Extract visibility of the selected baseline
            corr_prod = str(reverse_input_map[inputs[ant_A] + inputs[ant_B]])
            vis = scan_group['data'][corr_prod][:, channel_from:channel_to]
            # Group delay is proportional to phase slope across the band - estimate this as
            # the phase difference between consecutive frequency channels calculated via np.diff.
            # Pick antenna 1 as reference antenna -> correlation product XY* means we
            # actually measure phase(antenna1) - phase(antenna2), therefore flip the sign.
            # Also divide by channel frequency difference, which correctly handles gaps in frequency coverage.
            phase_diff_per_Hz = np.diff(-np.angle(vis), axis=1) / freq_diff
            # Convert to a delay in seconds
            delay = phase_diff_per_Hz / (-2.0 * np.pi)
            # Obtain robust periodic statistics for *per-channel* phase difference
            delay_stats_mu, delay_stats_sigma = scape.stats.periodic_mu_sigma(delay, axis=1, period=2 * max_delay)
            group_delay.append(delay_stats_mu)
            # The estimated mean group delay is the average of N-1 per-channel differences. Since this is less
            # variable than the per-channel data itself, we have to divide the data sigma by sqrt(N-1).
            sigma_delay.append(delay_stats_sigma / np.sqrt(len(channel_freqs) - 1))
            # Insert augmented direction block into proper spot to build design matrix
            augm_block = np.zeros((4 * len(ants), len(timestamps)))
            # Since baseline AB = (ant B - ant A) positions, insert target dirs with appropriate sign
            # This fulfills the role of the baseline difference matrix mapping antennas to baselines
            augm_block[(4 * ant_A):(4 * ant_A + 4), :] = -scan_augmented_targetdir
            augm_block[(4 * ant_B):(4 * ant_B + 4), :] = scan_augmented_targetdir
            augmented_targetdir.append(augm_block)
# Concatenate per-scan arrays into a single array for data set
augmented_targetdir = np.hstack(augmented_targetdir)
group_delay = np.hstack(group_delay)
sigma_delay = np.hstack(sigma_delay)
# Sanitise the uncertainties (can't be too certain...)
sigma_delay[sigma_delay < 1e-5 * max_sigma_delay] = 1e-5 * max_sigma_delay

# Construct design matrix, containing weighted basis functions
A = augmented_targetdir / sigma_delay
# Throw out reference antenna columns, as we can't solve reference antenna parameters (assumed zero)
A = np.vstack((A[:(4 * ref_ant_ind), :], A[(4 * ref_ant_ind + 4):, :]))
# Measurement vector, containing weighted observed delays
b = group_delay / sigma_delay
# Throw out data points with standard deviations above the given threshold
A = A[:, sigma_delay < opts.max_sigma * max_sigma_delay]
b = b[sigma_delay < opts.max_sigma * max_sigma_delay]
print 'Fitting %d parameters to %d data points (discarded %d)...' % (A.shape[0], len(b), len(sigma_delay) - len(b))
if len(b) == 0:
    raise ValueError('No solution possible, as all data points were discarded')
# Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
U, s, Vt = np.linalg.svd(A.transpose(), full_matrices=False)
params = np.dot(Vt.T, np.dot(U.T, b) / s)
# Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
sigma_params = np.sqrt(np.sum((Vt.T / s[np.newaxis, :]) ** 2, axis=1))

# Reshape parameters to be per antenna, also inserting a row of zeros for reference antenna
ant_params = params.reshape(-1, 4)
ant_params = np.vstack((ant_params[:ref_ant_ind, :], np.zeros(4), ant_params[ref_ant_ind:, :]))
ant_sigma_params = sigma_params.reshape(-1, 4)
# Assign half the uncertainty of each antenna offset to the reference antenna itself, which is assumed
# to be known perfectly, but obviously isn't (and should have similar uncertainty to the rest)
ref_sigma = 0.5 * ant_sigma_params.mean(axis=0)
ant_sigma_params = np.vstack((ant_sigma_params[:ref_ant_ind, :], np.zeros(4), ant_sigma_params[ref_ant_ind:, :]))
ant_sigma_params -= ref_sigma[np.newaxis, :]
ant_sigma_params[ref_ant_ind] = ref_sigma
# Convert to useful output (antenna positions and cable lengths in metres)
positions = ant_params[:, :3] * katpoint.lightspeed + old_positions[ref_ant_ind, :]
sigma_positions = ant_sigma_params[:, :3] * katpoint.lightspeed
cable_lengths = ant_params[:, 3] * cable_lightspeed + old_cable_lengths[ref_ant_ind]
sigma_cable_lengths = ant_sigma_params[:, 3] * cable_lightspeed
receiver_delays = cable_lengths / cable_lightspeed
sigma_receiver_delays = sigma_cable_lengths / cable_lightspeed

# Output results
for n, ant in enumerate(ants):
    print "Antenna", ant.name, ' (*REFERENCE*)' if n == ref_ant_ind else ''
    print "------------"
    print "E (m):               %7.3f +- %.5f (was %7.3f)%s" % \
          (positions[n, 0], sigma_positions[n, 0], old_positions[n, 0],
           ' *' if np.abs(positions[n, 0] - old_positions[n, 0]) > 3 * sigma_positions[n, 0] else '')
    print "N (m):               %7.3f +- %.5f (was %7.3f)%s" % \
          (positions[n, 1], sigma_positions[n, 1], old_positions[n, 1],
           ' *' if np.abs(positions[n, 1] - old_positions[n, 1]) > 3 * sigma_positions[n, 1] else '')
    print "U (m):               %7.3f +- %.5f (was %7.3f)%s" % \
          (positions[n, 2], sigma_positions[n, 2], old_positions[n, 2],
           ' *' if np.abs(positions[n, 2] - old_positions[n, 2]) > 3 * sigma_positions[n, 2] else '')
    print "Cable length (m):    %7.3f +- %.5f (was %7.3f)%s" % \
          (cable_lengths[n], sigma_cable_lengths[n], old_cable_lengths[n],
           ' *' if np.abs(cable_lengths[n] - old_cable_lengths[n]) > 3 * sigma_cable_lengths[n] else '')
    print "Receiver delay (ns): %7.3f +- %.3f   (was %7.3f)%s" % \
          (receiver_delays[n] * 1e9, sigma_receiver_delays[n] * 1e9, old_receiver_delays[n] * 1e9,
           ' *' if np.abs(receiver_delays[n] - old_receiver_delays[n]) > 3 * sigma_receiver_delays[n] else '')
    print
