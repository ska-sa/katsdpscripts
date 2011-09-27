#! /usr/bin/python
#
# Baseline calibration for multiple baselines using HDF5 format version 1 and 2 files.
#
# Ludwig Schwardt
# 5 April 2011
#

import optparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import katfile
import scape
import katpoint

# Array position used for fringe stopping
array_ant = katpoint.Antenna('ant0, -30:43:17.3, 21:24:38.5, 1038.0, 0.0')
# Original estimated cable lengths to 12-m container
ped_to_12m = {'ant1': 95.5, 'ant2': 108.8, 'ant3': 95.5 + 50, 'ant4': 95.5 + 70}
# Estimated Losberg cable lengths to start from, in metres
ped_to_losberg = {'ant1' : 4977.4, 'ant2' : 4988.8, 'ant3' : 5011.8, 'ant4' : 5035.4,
                  'ant5' : 5067.3, 'ant6' : 5090.9, 'ant7' : 5143.9}
# The speed of light in the fibres is lower than in vacuum
cable_lightspeed = katpoint.lightspeed / 1.4

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file> ...]")
parser.add_option('-a', '--ants',
                  help="Comma-separated subset of antennas to use in fit (e.g. 'ant1,ant2'), default is all antennas")
parser.add_option("-f", "--freq-chans", default='98,417',
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default %default)")
parser.add_option('-p', '--pol', type='choice', choices=['H', 'V'], default='H',
                  help="Polarisation term to use ('H' or 'V'), default is %default")
parser.add_option('-r', '--ref', help="Reference antenna, default is first antenna in file")
parser.add_option('-s', '--max-sigma', type='float', default=0.2,
                  help="Threshold on std deviation of normalised group delay, default is %default")
parser.add_option("-t", "--time-offset", type='float', default=0.0,
                  help="Time offset to add to DBE timestamps, in seconds (default = %default)")
parser.add_option('-x', '--exclude', default='', help="Comma-separated list of sources to exclude from fit")
(opts, args) = parser.parse_args()

if len(args) < 1:
    raise RuntimeError('Please specify HDF5 data file(s) to use as arguments of the script')
channel_range = [int(chan) for chan in opts.freq_chans.split(',')]

print "\nLoading and processing data...\n"

data = katfile.open(args, opts.ref, channel_range, opts.time_offset)

# Filter available antennas via script --ants option, if provided
ants = [ant for ant in data.ants if opts.ants is None or ant.name in opts.ants]
ref_ant_ind = [ant.name for ant in ants].index(data.ref_ant)
# Form desired signal paths and obtain all baselines connecting them
signals = [ant.name + opts.pol for ant in ants]
baselines = data.all_corr_products(signals)
# Throw out autocorrelations
baselines = [(antA, antB) for antA, antB in baselines if antA != antB]
baseline_names = ['%s - %s' % (ants[antA].name, ants[antB].name) for antA, antB in baselines]
num_bls = len(baselines)
if num_bls == 0:
    raise RuntimeError('No baselines based on the requested antennas and polarisation found in data set')

# Reference antenna and excluded sources
excluded_targets = opts.exclude.split(',')
old_positions = np.array([ant.position_enu for ant in ants])
old_cable_lengths = np.array([ped_to_losberg[ant.name] for ant in ants])
old_receiver_delays = old_cable_lengths / cable_lightspeed

# Phase differences are associated with frequencies at midpoints between channels
mid_freqs = np.convolve(data.channel_freqs, [0.5, 0.5], mode='valid')
freq_diff = np.abs(np.diff(data.channel_freqs))
num_chans, sample_period = len(data.channel_freqs), 1. / data.dump_rate

# Since phase slope is sampled in frequency, the derived delay exhibits aliasing with period of 1 / channel_bandwidth
# The maximum delay that can be reliably represented is therefore +- 0.5 delay_period
delay_period = 1. / data.channel_bw
# Maximum standard deviation of delay occurs when delay samples are uniformly distributed between +- max_delay
# In this case, delay is completely random / unknown, and its estimate cannot be trusted
# The division by sqrt(N-1) converts the per-channel standard deviation to a per-snapshot deviation
max_sigma_delay = delay_period / np.sqrt(12) / np.sqrt(num_chans - 1)

ant_list = ', '.join([(ant.name + ' (*ref*)' if ind == ref_ant_ind else ant.name) for ind, ant in enumerate(ants)])
print 'antennas (%d): %s [pol %s]' % (len(ants), ant_list, opts.pol)
print 'baselines (%d): %s' % (num_bls, ' '.join([('%d-%d' % (antA, antB)) for antA, antB in baselines]))

# Iterate through scans (as indicated by activity sensor)
augmented_targetdir, group_delay, sigma_delay = [], [], []
scan_targets, scan_mid_az, scan_mid_el, scan_timestamps, scan_phase = [], [], [], [], []
for scan_ind, compscan_ind, state, target in data.scans():
    ts = data.timestamps()
    if state != 'track':
        print "scan %3d (%4d samples) skipped '%s'" % (scan_ind, len(ts), state)
        continue
    if len(ts) < 2:
        print "scan %3d (%4d samples) skipped - too short" % (scan_ind, len(ts))
        continue
    if target.name in excluded_targets:
        print "scan %3d (%4d samples) skipped - excluded '%s'" % (scan_ind, len(ts), target.name)
        continue
    # Obtain unit vectors pointing from array antenna to target for each timestamp in scan
    az, el = target.azel(ts, array_ant)
    targetdir = np.array(katpoint.azel_to_enu(az, el))
    # Invert sign of target vector, as positive dot product with baseline implies negative delay / advance
    # Augment target vector with a 1, as this allows fitting of constant (receiver) delay
    scan_augmented_targetdir = np.vstack((-targetdir, np.ones(len(el))))
    scan_rel_sigma_delay, bl_phase = [], []
    for antA, antB in baselines:
        # Extract visibility of the selected baseline
        vis = data.vis((signals[antA], signals[antB]))
        bl_phase.append(np.angle(vis).T)
        # Group delay is proportional to phase slope across the band - estimate this as
        # the phase difference between consecutive frequency channels calculated via np.diff.
        # Pick antenna 1 as reference antenna -> correlation product XY* means we
        # actually measure phase(antenna1) - phase(antenna2), therefore flip the sign.
        # Also divide by channel frequency difference, which correctly handles gaps in frequency coverage.
        phase_diff_per_Hz = np.diff(-np.angle(vis), axis=1) / freq_diff
        # Convert to a delay in seconds
        delay = phase_diff_per_Hz / (-2.0 * np.pi)
        # Obtain robust periodic statistics for *per-channel* phase difference
        delay_stats_mu, delay_stats_sigma = scape.stats.periodic_mu_sigma(delay, axis=1, period=delay_period)
        # The estimated mean group delay is the average of N-1 per-channel differences. Since this is less
        # variable than the per-channel data itself, we have to divide the data sigma by sqrt(N-1).
        delay_stats_sigma /= np.sqrt(num_chans - 1)
        group_delay.append(delay_stats_mu)
        sigma_delay.append(delay_stats_sigma)
        # Insert augmented direction block into proper spot to build design matrix
        augm_block = np.zeros((4 * len(ants), len(ts)))
        # Since baseline AB = (ant B - ant A) positions, insert target dirs with appropriate sign
        # This fulfills the role of the baseline difference matrix mapping antennas to baselines
        augm_block[(4 * antA):(4 * antA + 4), :] = -scan_augmented_targetdir
        augm_block[(4 * antB):(4 * antB + 4), :] = scan_augmented_targetdir
        augmented_targetdir.append(augm_block)
        scan_rel_sigma_delay.append(delay_stats_sigma.mean() / max_sigma_delay)
    scan_targets.append(target.name)
    scan_mid_az.append(np.median(az))
    scan_mid_el.append(np.median(el))
    scan_timestamps.append(ts - data.start_time)
    scan_phase.append(np.vstack(bl_phase))
    print "scan %3d (%4d samples) %s '%s'" % \
          (scan_ind, len(ts), ' '.join([('%.3f' % rel_sigma) for rel_sigma in scan_rel_sigma_delay]), target.name)
if not augmented_targetdir:
    raise RuntimeError('No usable scans found (are you tracking any targets?)')
# Concatenate per-baseline arrays into a single array for data set
augmented_targetdir = np.hstack(augmented_targetdir)
group_delay = np.hstack(group_delay)
sigma_delay = np.hstack(sigma_delay)
# Sanitise the uncertainties (can't be too certain...)
sigma_delay[sigma_delay < 1e-5 * max_sigma_delay] = 1e-5 * max_sigma_delay

# Assume that delay errors are within +-0.5 delay_period, based on current delay model
# Then unwrap the measured group delay to be within this range of the predicted delays, to avoid delay wrapping issues
old_delay_model = np.c_[old_positions / katpoint.lightspeed, old_receiver_delays].ravel()
old_predicted_delay = np.dot(old_delay_model, augmented_targetdir)
norm_residual_delay = (group_delay - old_predicted_delay) / delay_period
unwrapped_group_delay = delay_period * (norm_residual_delay - np.round(norm_residual_delay)) + old_predicted_delay
old_resid = unwrapped_group_delay - old_predicted_delay

# Construct design matrix, containing weighted basis functions
A = augmented_targetdir / sigma_delay
# Throw out reference antenna columns, as we can't solve reference antenna parameters (assumed zero)
A = np.vstack((A[:(4 * ref_ant_ind), :], A[(4 * ref_ant_ind + 4):, :]))
# Measurement vector, containing weighted observed delays
b = unwrapped_group_delay / sigma_delay
# Throw out data points with standard deviations above the given threshold
good = sigma_delay < opts.max_sigma * max_sigma_delay
A = A[:, good]
b = b[good]
print '\nFitting %d parameters to %d data points (discarded %d)...' % (A.shape[0], len(b), len(sigma_delay) - len(b))
if len(b) == 0:
    raise ValueError('No solution possible, as all data points were discarded')
# Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
U, s, Vrt = np.linalg.svd(A.transpose(), full_matrices=False)
params = np.dot(Vrt.T, np.dot(U.T, b) / s)
# Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
sigma_params = np.sqrt(np.sum((Vrt.T / s[np.newaxis, :]) ** 2, axis=1))
print 'Condition number = %.3f' % (s[0] / s[-1],)

# Reshape parameters to be per antenna, also inserting a row of zeros for reference antenna
ant_params = params.reshape(-1, 4)
ant_params = np.vstack((ant_params[:ref_ant_ind, :], np.zeros(4), ant_params[ref_ant_ind:, :]))
ant_sigma_params = sigma_params.reshape(-1, 4)
# Assign half the uncertainty of each antenna offset to the reference antenna itself, which is assumed
# to be known perfectly, but obviously isn't (and should have similar uncertainty to the rest)
## ref_sigma = 0.5 * ant_sigma_params.min(axis=0)
ant_sigma_params = np.vstack((ant_sigma_params[:ref_ant_ind, :], np.zeros(4), ant_sigma_params[ref_ant_ind:, :]))
## ant_sigma_params -= ref_sigma[np.newaxis, :]
## ant_sigma_params[ref_ant_ind] = ref_sigma
# Convert to useful output (antenna positions and cable lengths in metres)
positions = ant_params[:, :3] * katpoint.lightspeed + old_positions[ref_ant_ind, :]
sigma_positions = ant_sigma_params[:, :3] * katpoint.lightspeed
cable_lengths = ant_params[:, 3] * cable_lightspeed + old_cable_lengths[ref_ant_ind]
sigma_cable_lengths = ant_sigma_params[:, 3] * cable_lightspeed
receiver_delays = cable_lengths / cable_lightspeed
sigma_receiver_delays = sigma_cable_lengths / cable_lightspeed
# Obtain new predictions
new_delay_model = np.c_[positions / katpoint.lightspeed, receiver_delays].ravel()
new_predicted_delay = np.dot(new_delay_model, augmented_targetdir)
new_resid = unwrapped_group_delay - new_predicted_delay

# Output results
for n, ant in enumerate(ants):
    print "\nAntenna", ant.name, ' (*REFERENCE*)' if n == ref_ant_ind else ''
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

scan_lengths = [len(ts) for ts in scan_timestamps]
scan_bl_starts = num_bls * np.cumsum([0] + scan_lengths)[:-1]
def extract_bl_segments(x, n, scale=1., offset=0.):
    """Given sequence *x*, extract the bits pertaining to baseline *n* as list of per-scan data."""
    return [scale * x[(scan_start + n * scan_len):(scan_start + (n+1)*scan_len)] + offset
            for scan_start, scan_len in zip(scan_bl_starts, scan_lengths)]
def extract_scan_segments(x):
    """Given sequence *x*, extract the bits pertaining to scan *n*."""
    return [x[scan_start:(scan_start + num_bls*scan_len)]
            for scan_start, scan_len in zip(scan_bl_starts, scan_lengths)]

plt.figure(1)
plt.clf()
scan_freqinds = [np.arange(num_bls * num_chans)] * len(scan_timestamps)
scape.plots_basic.plot_segments(scan_timestamps, scan_freqinds, scan_phase, labels=scan_targets)
plt.xlabel('Time (s), since %s' % (katpoint.Timestamp(data.start_time).local(),))
plt.yticks(np.arange(num_chans // 2, num_bls * num_chans, num_chans), baseline_names)
for yval in range(0, num_bls * num_chans, num_chans):
    plt.axhline(yval, color='k', lw=2)
plt.title('Raw visibility phase per baseline')

plt.figure(2)
plt.clf()
resid_ind = [np.arange(scan_start, scan_start + num_bls*scan_len)
             for scan_start, scan_len in zip(scan_bl_starts, scan_lengths)]
scape.plots_basic.plot_segments(resid_ind, extract_scan_segments(old_resid / 1e-9),
                                labels=scan_targets, width=sample_period, color='b')
scape.plots_basic.plot_segments(resid_ind, extract_scan_segments(new_resid / 1e-9), labels=[],
                                width=sample_period, color='r')
plt.axhline(-0.5 * delay_period / 1e-9, color='k', linestyle='--')
plt.axhline(0.5 * delay_period / 1e-9, color='k', linestyle='--')
plt.xticks([])
plt.xlabel('Measurements')
plt.ylabel('Delay error (ns)')
plt.title('Residual delay errors (blue = old model and red = new model)')

plt.figure(3)
plt.clf()
for n in range(num_bls):
    # Pick 25th or 75th percentile of each residual, whichever is larger
    resids = np.sort(old_resid)
    resid_excursion = [resids[int(0.25 * len(resids))], resids[int(0.75 * len(resids))]]
    resids = np.sort(new_resid)
    resid_excursion += [resids[int(0.25 * len(resids))], resids[int(0.75 * len(resids))]]
    scale = 0.125 * delay_period / np.max(np.abs(resid_excursion))
    bl_old_resid = extract_bl_segments(old_resid, n, scale, n * delay_period)
    bl_new_resid = extract_bl_segments(new_resid, n, scale, n * delay_period)
    resid_sigma = extract_bl_segments(sigma_delay, n, scale)
    old_resid_range = [np.c_[resid - sigma, resid + sigma] for resid, sigma in zip(bl_old_resid, resid_sigma)]
    plt.axhline(n * delay_period, color='k')
    plt.axhline((n + 0.5) * delay_period, color='k', linestyle='--')
    scape.plots_basic.plot_segments(scan_timestamps, old_resid_range, labels=[], width=sample_period,
                                    add_breaks=False, color='b', alpha=0.5)
    scape.plots_basic.plot_segments(scan_timestamps, bl_old_resid, labels=scan_targets,
                                    width=sample_period, color='b')
    scape.plots_basic.plot_segments(scan_timestamps, bl_new_resid, labels=[], width=sample_period,
                                    add_breaks=False, color='r', lw=2)
plt.ylim(-0.5 * delay_period, (num_bls - 0.5) * delay_period)
plt.yticks(np.arange(num_bls) * delay_period, baseline_names)
plt.xlabel('Time (s), since %s' % (katpoint.Timestamp(data.start_time).local(),))
plt.title('Residual delay errors per baseline (blue = old model and red = new model)')

plt.figure(4)
plt.clf()
ax = plt.axes(polar=True)
eastnorth_radius = np.sqrt(old_positions[:, 0] ** 2 + old_positions[:, 1] ** 2)
eastnorth_angle = np.arctan2(old_positions[:, 0], old_positions[:, 1])
for ant, theta, r in zip(ants, eastnorth_angle, eastnorth_radius):
    ax.text(np.pi/2. - theta, r * 0.9 * np.pi/2. / eastnorth_radius.max(), ant.name,
            ha='center', va='center').set_bbox(dict(facecolor='b', lw=1, alpha=0.3))
# Quality of delays obtained from source, with 0 worst and 1 best
quality = np.hstack([q.mean(axis=0) for q in extract_scan_segments(1.0 - sigma_delay / max_sigma_delay)])
ax.scatter(np.pi/2 - np.array(scan_mid_az), np.pi/2 - np.array(scan_mid_el), 100*quality, 'k',
           edgecolors=None, linewidths=0, alpha=0.5)
for name, az, el in zip(scan_targets, scan_mid_az, scan_mid_el):
    ax.text(np.pi/2. - az, np.pi/2. - el, name, ha='center', va='top')
ax.set_xticks(katpoint.deg2rad(np.arange(0., 360., 90.)))
ax.set_xticklabels(['E', 'N', 'W', 'S'])
ax.set_ylim(0., np.pi / 2.)
ax.set_yticks(katpoint.deg2rad(np.arange(0., 90., 10.)))
ax.set_yticklabels([])
plt.title('Antenna positions and source directions')
