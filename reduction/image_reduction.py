#
# Script that produced first KAT-7 image.
#
# Ludwig Schwardt
# 18 July 2011
#

import os
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage

import katfile
import katpoint
import scape

parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file> ...]",
                               description='Produce image from HDF5 data file(s). Please identify '
                                           'the image target and calibrators via the options.')
parser.add_option('-i', '--image-target', help="Imaging target name (**required**)")
parser.add_option('-b', '--bandpass-cal', help="Bandpass calibrator name (**required**)")
parser.add_option('-g', '--gain-cal', help="Gain calibrator name (**required**)")
parser.add_option('-a', '--ants',
                  help="Comma-separated subset of antennas to use (e.g. 'ant1,ant2'), default is all antennas")
parser.add_option("-f", "--freq-chans", default='200,800',
                  help="Range of frequency channels to keep (zero-based 'start,end', default = %default)")
parser.add_option("--chan-avg", type='int', default=60,
                  help="Number of adjacent frequency channels to average together in MFS imaging (default = %default)")
parser.add_option("--time-avg", type='int', default=90,
                  help="Number of consecutive dumps to average together for imaging (default %default)")
parser.add_option('-p', '--pol', type='choice', choices=['H', 'V'], default='H',
                  help="Polarisation term to use ('H' or 'V'), default is %default")
parser.add_option('-r', '--ref', dest='ref_ant', help="Reference antenna, default is first antenna in file")
parser.add_option("-t", "--time-offset", type='float', default=0.0,
                  help="Time offset to add to DBE timestamps, in seconds (default = %default)")
parser.add_option("--time-slice", type='int', default=30,
                  help="Index of sample relative to start of each scan where vis "
                       "is plotted as function of frequency (default = %default)")
parser.add_option("--freq-slice", type='int', default=250,
                  help="Frequency channel index for which vis is plotted as a function of time (default = %default)")
(opts, args) = parser.parse_args()

if opts.image_target is None:
    raise ValueError('Please specify the name of the target to image via the -i option')
if opts.bandpass_cal is None:
    raise ValueError('Please specify the name of the bandpass calibrator via the -b option')
if opts.gain_cal is None:
    raise ValueError('Please specify the name of the gain calibrator via the -g option')

# Quick way to set options for use with cut-and-pasting of script bits
# opts = optparse.Values()
# opts.image_target = 'Cen A'
# opts.bandpass_cal = '3C 273'
# opts.gain_cal = 'PKS 1421-490'
# opts.ants = None
# opts.freq_chans = '200,800'
# opts.chan_avg = 60
# opts.time_avg = 90
# opts.pol = 'H'
# opts.ref_ant = 'ant2'
# opts.time_offset = 0.0
# opts.time_slice = 30
# opts.freq_slice = 250
# import glob
# args = sorted(glob.glob('*.h5'))
# args = ['1313238698.h5', '1313240388.h5']

# Frequency channel range to keep, and number of channels to average together into band
freq_chans = [int(chan_str) for chan_str in opts.freq_chans.split(',')]
first_chan, one_past_last_chan = freq_chans[0], freq_chans[1]
channels_per_band, dumps_per_vis = opts.chan_avg, opts.time_avg
# Slices for plotting
time_slice = opts.time_slice
freq_slice = opts.freq_slice - first_chan

# Latest KAT-7 antenna positions and H / V cable delays via recent baseline cal (1313748602 dataset, not joint yet)
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22', 23220.506e-9, 23228.551e-9),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22', 23283.799e-9, 23286.823e-9),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22', 23407.970e-9, 23400.221e-9),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22', 23514.801e-9, 23514.801e-9),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22', 23676.033e-9, 23668.223e-9),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.854e-9, 23782.150e-9),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22', 24047.672e-9, 24039.237e-9),
}

katpoint.logger.setLevel(30)

################################ LOAD DATA #####################################

print "Opening data file(s)..."

# Open data files
data = katfile.open(args, ref_ant=opts.ref_ant, channel_range=(first_chan, one_past_last_chan - 1),
                    time_offset=opts.time_offset)

# Create antenna objects with latest positions for antennas used in experiment, and list of inputs and cable delays
ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
inputs, delays = [], {}
for ant in sorted(ants):
    if ant + opts.pol in data.inputs:
        inputs.append(ant + opts.pol)
        delays[ant + opts.pol] = new_ants[ant][1 if opts.pol == 'H' else 2]
# Extract available cross-correlation products, as pairs of indices into input list
crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]

# Extract frequency information
center_freqs = data.channel_freqs
wavelengths = katpoint.lightspeed / center_freqs

# Create catalogue of targets found in data
targets = katpoint.Catalogue()
for scan_ind, cs_ind, state, target in data.scans():
    if state == 'track' and target.name not in targets:
        targets.add(target)

image_target = targets[opts.image_target]
if image_target is None:
    raise KeyError("Unknown image target '%s' - data contains targets '%s'" %
                   (opts.image_target, "', '".join([tgt.name for tgt in targets])))
bandpass_cal = targets[opts.bandpass_cal]
if bandpass_cal is None:
    raise KeyError("Unknown bandpass calibrator '%s' - data contains targets '%s'" %
                   (opts.bandpass_cal, "', '".join([tgt.name for tgt in targets])))
gain_cal = targets[opts.gain_cal]
if gain_cal is None:
    raise KeyError("Unknown gain calibrator '%s' - data contains targets '%s'" %
                   (opts.gain_cal, "', '".join([tgt.name for tgt in targets])))

############################## STOP FRINGES ####################################

print "Assembling bandpass calibrator data and checking fringe stopping..."

# Assemble fringe-stopped visibility data for main (bandpass) calibrator
orig_cal_vis_samples, cal_vis_samples, cal_timestamps = [], [], []
for scan_ind, cs_ind, state, target in data.scans():
    if state != 'track' or target != bandpass_cal:
        continue
    timestamps = data.timestamps()
    if len(timestamps) < 2:
        continue
    vis_pre = np.zeros((len(timestamps), len(wavelengths), len(crosscorr)), dtype=np.complex64)
    vis_post = np.zeros((len(timestamps), len(wavelengths), len(crosscorr)), dtype=np.complex64)
    # Iterate through baselines and assemble visibilities
    for n, (indexA, indexB) in enumerate(crosscorr):
        inputA, inputB = inputs[indexA], inputs[indexB]
        antA, antB = inputA[:-1], inputB[:-1]
        vis = data.vis((inputA, inputB))
        vis_pre[:, :, n] = vis
        # Get uvw coordinates of A->B baseline
        u, v, w = target.uvw(ants[antB], timestamps, ants[antA])
        # Number of turns of phase that signal B is behind signal A due to geometric delay
        geom_delay_turns = - w[:, np.newaxis] / wavelengths
        # Number of turns of phase that signal B is behind signal A due to cable / receiver delay
        cable_delay_turns = (delays[inputB] - delays[inputA]) * center_freqs
        # Visibility <A, B*> has phase (A - B), therefore add (B - A) phase to stop fringes (i.e. do delay tracking)
        vis *= np.exp(2j * np.pi * (geom_delay_turns + cable_delay_turns))
        vis_post[:, :, n] = vis
    orig_cal_vis_samples.append(vis_pre)
    cal_vis_samples.append(vis_post)
    cal_timestamps.append(timestamps)

def plot_vis_crosshairs(fig, vis_data, title, upper=True, units='', **kwargs):
    """Create phasor plot (upper or lower triangle of baseline matrix)."""
    fig.subplots_adjust(wspace=0., hspace=0.)
    data_lim = np.max([np.abs(vis).max() for vis in vis_data])
    ax_lim = 1.05 * data_lim
    for n, (indexA, indexB) in enumerate(crosscorr):
        subplot_index = (len(ants) * indexA + indexB + 1) if upper else (indexA + len(ants) * indexB + 1)
        ax = fig.add_subplot(len(ants), len(ants), subplot_index)
        for vis in vis_data:
            ax.plot(vis[:, n].real, vis[:, n].imag, **kwargs)
        ax.axhline(0, lw=0.5, color='k')
        ax.axvline(0, lw=0.5, color='k')
        ax.add_patch(mpl.patches.Circle((0., 0.), data_lim, facecolor='none', edgecolor='k', lw=0.5))
        ax.add_patch(mpl.patches.Circle((0., 0.), 0.5 * data_lim, facecolor='none', edgecolor='k', lw=0.5))
        ax.axis('image')
        ax.axis([-ax_lim, ax_lim, -ax_lim, ax_lim])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        if upper:
            if indexA == 0:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel(inputs[indexB][3:])
            if indexB == len(ants) - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(inputs[indexA][3:], rotation='horizontal')
        else:
            if indexA == 0:
                ax.set_ylabel(inputs[indexB][3:], rotation='horizontal')
            if indexB == len(ants) - 1:
                ax.set_xlabel(inputs[indexA][3:])
    fig.text(0.5, 0.95 if upper else 0.05, title, ha='center', va='bottom' if upper else 'top')
    fig.text(0.95 if upper else 0.05, 0.5, 'Outer radius = %g %s' % (data_lim, units), va='center', rotation='vertical')

fig = plt.figure(1)
fig.clear()
plot_vis_crosshairs(fig, [vis[:, freq_slice, :] for vis in orig_cal_vis_samples],
                    "'%s' raw vis for channel %d (all times)" % (bandpass_cal.name, freq_slice + first_chan,),
                    upper=True, units='counts', marker='.', linestyle='-')
plot_vis_crosshairs(fig, [vis[time_slice, :, :] for vis in orig_cal_vis_samples],
                    "'%s' raw vis for scan time sample %d (all channels)" % (bandpass_cal.name, time_slice),
                    upper=False, units='counts', marker='.', linestyle='')

fig = plt.figure(2)
fig.clear()
plot_vis_crosshairs(fig, [vis[:, freq_slice, :] for vis in cal_vis_samples],
                    "'%s' stopped vis for channel %d (all times)" % (bandpass_cal.name, freq_slice + first_chan,),
                    upper=True, units='counts', marker='.', linestyle='-')
plot_vis_crosshairs(fig, [vis[time_slice, :, :] for vis in cal_vis_samples],
                    "'%s' stopped vis for scan time sample %d (all channels)" % (bandpass_cal.name, time_slice),
                    upper=False, units='counts', marker='.', linestyle='')

############################## BANDPASS CAL ####################################

print "Performing bandpass calibration on '%s'..." % (bandpass_cal.name,)

# Vector that contains real and imaginary gain components for all signal paths
full_params = np.zeros(2 * len(inputs))
# Indices of gain parameters that will be optimised
params_to_fit = range(len(full_params))
ref_input_index = inputs.index(data.ref_ant + opts.pol)
# Don't fit the imaginary component of the gain on the reference signal path (this is assumed to be zero)
params_to_fit.pop(2 * ref_input_index + 1)
initial_gains = np.tile([1., 0.], len(inputs))[params_to_fit]

def apply_gains(params, input_pairs, model_vis=1.0):
    """Apply relevant antenna gains to model visibility to estimate measurements.

    This corrupts the ideal model visibilities by applying a set of complex
    antenna gains to them.

    Parameters
    ----------
    params : array of float, shape (2 * N - 1,)
        Array of gain parameters with 2 parameters per signal path (real and
        imaginary components of complex gain), except for phase reference input
        which has a single real component
    input_pairs : array of int, shape (2, M)
        Input pair(s) for which visibilities will be calculated. The inputs are
        specified by integer indices into the main input list.
    model_vis : complex or array of complex, shape (M,), optional
        The modelled (ideal) source visibilities on each specified baseline.
        The default model is that of a point source with unit flux density.

    Returns
    -------
    estm_vis : array of float, shape (2,) or (2, M)
        Estimated visibilities specified by their real and imaginary components

    """
    full_params[params_to_fit] = params
    antA, antB = input_pairs[0], input_pairs[1]
    reA, imA, reB, imB = full_params[2 * antA], full_params[2 * antA + 1], full_params[2 * antB], full_params[2 * antB + 1]
    # Calculate gain product (g_A g_B*)
    reAB, imAB = reA * reB + imA * imB, imA * reB - reA * imB
    re_model, im_model = np.real(model_vis), np.imag(model_vis)
    return np.vstack((reAB * re_model - imAB * im_model, reAB * im_model + imAB * re_model)).squeeze()

# Vector that contains gain phase components for all signal paths
phase_params = np.zeros(len(inputs))
# Indices of phase parameters that will be optimised
phase_params_to_fit = range(len(phase_params))
# Don't fit the phase on the reference signal path (this is assumed to be zero)
phase_params_to_fit.pop(ref_input_index)
initial_phases = np.zeros(len(inputs))[phase_params_to_fit]

def apply_phases(params, input_pairs, model_vis=1.0):
    """Apply relevant antenna phases to model visibility to estimate measurements.

    This corrupts the ideal model visibilities by adding a set of per-antenna
    phases to them.

    Parameters
    ----------
    params : array of float, shape (N - 1,)
        Array of gain parameters with 1 parameter per signal path (phase
        component of complex gain), except for phase reference input which has
        zero phase
    input_pairs : array of int, shape (2, M)
        Input pair(s) for which visibilities will be calculated. The inputs are
        specified by integer indices into the main input list.
    model_vis : complex or array of complex, shape (M,), optional
        The modelled (ideal) source visibilities on each specified baseline.
        The default model is that of a point source with unit flux density.

    Returns
    -------
    estm_vis : array of float, shape (2,) or (2, M)
        Estimated visibilities specified by their real and imaginary components

    """
    phase_params[phase_params_to_fit] = params
    phaseA, phaseB = phase_params[input_pairs[0]], phase_params[input_pairs[1]]
    # Calculate gain product (g_A g_B*) where each gain has unit magnitude
    estm_vis = np.exp(1j * (phaseA - phaseB)) * model_vis
    return np.vstack((np.real(estm_vis), np.imag(estm_vis))).squeeze()

# Solve for antenna bandpass gains
bandpass_gainsols = []
bp_source_vis = bandpass_cal.flux_density(center_freqs / 1e6)
# Iterate over solution intervals
for solint_vis in cal_vis_samples:
    gainsol = np.zeros((len(inputs), solint_vis.shape[1]), dtype=np.complex64)
    input_pairs = np.tile(np.array(crosscorr).T, solint_vis.shape[0])
    # Iterate over frequency channels
    for n in xrange(solint_vis.shape[1]):
        vis, model_vis = solint_vis[:, n, :].ravel(), bp_source_vis[n]
        fitter = scape.fitting.NonLinearLeastSquaresFit(lambda p, x: apply_gains(p, x, model_vis), initial_gains)
        fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)))
        full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2 * ref_input_index])
        gainsol[:, n] = full_params.view(np.complex128)
    bandpass_gainsols.append(gainsol)

# Combine bandpass gain solutions into a single solution by removing drifts from the first one and averaging
# orig_bandpass_gainsols = bandpass_gainsols[:]
# for bp_gain in bandpass_gainsols[1:]:
#     amp_drift = np.exp((np.log(np.abs(bp_gain)) - np.log(np.abs(bandpass_gainsols[0]))).mean(axis=1))
#     angle_diff = np.angle(bp_gain) - np.angle(bandpass_gainsols[0])
#     # Calculate a "safe" mean angle on the unit circle
#     phase_drift = np.arctan2(np.sin(angle_diff).mean(axis=1), np.cos(angle_diff).mean(axis=1))
#     gain_drift = amp_drift * np.exp(1.0j * phase_drift)
#     bp_gain /= gain_drift[:, np.newaxis]
bandpass_gains = np.dstack(bandpass_gainsols).mean(axis=2)

# Apply bandpass gain calibration to cal source visibilities
bp_cal_vis_samples = [vis.copy() for vis in cal_vis_samples]
for vis in bp_cal_vis_samples:
    for n, (indexA, indexB) in enumerate(crosscorr):
        vis[:, :, n] /= (bandpass_gains[indexA, :] * bandpass_gains[indexB, :].conjugate())

fig = plt.figure(3)
fig.clear()
ax = fig.add_subplot(211)
for n in range(len(inputs)):
    ax.plot(center_freqs / 1e6, np.abs(bandpass_gains[n]), label=inputs[n][3:])
ax.axis([center_freqs.min() / 1e6, center_freqs.max() / 1e6, 0., ax.get_ylim()[1]])
ax.set_xticklabels([])
ax.set_ylabel('Amplitude (linear)')
ax.set_title('Bandpass gain solutions')
ax.legend(loc='lower right', numpoints=1)
ax = fig.add_subplot(212)
ax.plot(center_freqs / 1e6, katpoint.rad2deg(np.angle(bandpass_gains.T)))
ax.axis([center_freqs.min() / 1e6, center_freqs.max() / 1e6, -180, 180])
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Phase (degrees)')

fig = plt.figure(4)
fig.clear()
plot_vis_crosshairs(fig, [vis[:, freq_slice, :] for vis in bp_cal_vis_samples],
                    "'%s' bp-corrected vis for channel %d (all times)" % (bandpass_cal.name, freq_slice + first_chan,),
                    upper=True, units='Jy', marker='.', linestyle='-')
plot_vis_crosshairs(fig, [vis[time_slice, :, :] for vis in bp_cal_vis_samples],
                    "'%s' bp-corrected vis for scan time sample %d (all channels)" % (bandpass_cal.name, time_slice),
                    upper=False, units='Jy', marker='.', linestyle='')

############################## BAND AVERAGE ####################################

print "Averaging all calibrator data into single frequency band..."

# Assemble visibility data for all calibrators, and average it to a single frequency band
all_cal_vis_samples, cal_source, all_cal_times = [], [], []
# Add phase calibrator data
for scan_ind, cs_ind, state, target in data.scans():
    if state != 'track' or target != gain_cal:
        continue
    timestamps = data.timestamps()
    if len(timestamps) < 2:
        continue
    # Extract visibilities and uvw coordinates
    vis_cross = np.zeros((len(timestamps), len(wavelengths), len(crosscorr)), dtype=np.complex64)
    # Iterate through baselines and assemble visibilities
    for n, (indexA, indexB) in enumerate(crosscorr):
        inputA, inputB = inputs[indexA], inputs[indexB]
        antA, antB = inputA[:-1], inputB[:-1]
        vis = data.vis((inputA, inputB))
        # Get uvw coordinates of A->B baseline
        u, v, w = target.uvw(ants[antB], timestamps, ants[antA])
        # Number of turns of phase that signal B is behind signal A due to geometric delay
        geom_delay_turns = - w[:, np.newaxis] / wavelengths
        # Number of turns of phase that signal B is behind signal A due to cable / receiver delay
        cable_delay_turns = (delays[inputB] - delays[inputA]) * center_freqs
        # Visibility <A, B*> has phase (A - B), therefore add (B - A) phase to stop fringes (i.e. do delay tracking)
        vis *= np.exp(2j * np.pi * (geom_delay_turns + cable_delay_turns))
        # Now apply bandpass calibration
        vis /= (bandpass_gains[indexA, :] * bandpass_gains[indexB, :].conjugate())
        vis_cross[:, :, n] = vis
    # It should be safe to average all channels into a single band now, for the purpose of gain calibration
    all_cal_vis_samples.append(vis_cross.mean(axis=1))
    cal_source.append(target)
    all_cal_times.append(timestamps[len(timestamps) // 2])
# Add bandpass calibrator data in a similar vein (if not there already)
for vis_cross, timestamps in zip(bp_cal_vis_samples, cal_timestamps):
    if timestamps[len(timestamps) // 2] not in all_cal_times:
        all_cal_vis_samples.append(vis_cross.mean(axis=1))
        cal_source.append(bandpass_cal)
        all_cal_times.append(timestamps[len(timestamps) // 2])

# Sort the data chronologically
time_index = np.argsort(all_cal_times)
all_cal_vis_samples = [all_cal_vis_samples[n] for n in time_index]
cal_source = np.array([cal_source[n] for n in time_index])
all_cal_times = np.array([all_cal_times[n] for n in time_index])

fig = plt.figure(5)
fig.clear()
plot_vis_crosshairs(fig, [vis for n, vis in enumerate(all_cal_vis_samples) if cal_source[n] == bandpass_cal],
                    "Frequency-averaged visibilities of '%s' (all times)" % (bandpass_cal.name,),
                    upper=True, units='Jy', marker='.', linestyle='-')
plot_vis_crosshairs(fig, [vis for n, vis in enumerate(all_cal_vis_samples) if cal_source[n] == gain_cal],
                    "Frequency-averaged visibilities of '%s' (all times)" % (gain_cal.name,),
                    upper=False, units='Jy', marker='.', linestyle='-')

################################ GAIN CAL ######################################

print "Performing gain calibration on '%s'..." % (gain_cal.name,)

# Average each solution interval as well, to get rid of secondary sources causing bubbles in visibility tracks
gain_cal_vis = np.array([vis.mean(axis=0) for vis in all_cal_vis_samples])
# Obtain average flux of calibrators
# Far cal has known flux, but the other calibrator has to be bootstrapped off the average data in a SETJY procedure
bandpass_cal_source = np.array([target == bandpass_cal for target in cal_source])
gain_cal_source = np.array([target == gain_cal for target in cal_source])
bandpass_cal_flux = bp_source_vis.mean()
gain_cal_flux = np.abs(gain_cal_vis[gain_cal_source, :]).mean() * \
                bandpass_cal_flux / np.abs(gain_cal_vis[bandpass_cal_source, :]).mean()
cal_source_vis = np.zeros(gain_cal_vis.shape[0])
cal_source_vis[gain_cal_source] = gain_cal_flux
cal_source_vis[bandpass_cal_source] = bandpass_cal_flux
# Use phase calibrator for standard gain calibration
gain_cal_vis, cal_source_vis = gain_cal_vis[gain_cal_source, :], cal_source_vis[gain_cal_source]
gain_times = all_cal_times[gain_cal_source]

# Solve for time-varying antenna gains
ant_gains = []
input_pairs = np.array(crosscorr).T
# Iterate over solution intervals
for vis, flux in zip(gain_cal_vis, cal_source_vis):
    fitter = scape.fitting.NonLinearLeastSquaresFit(lambda p, x: apply_gains(p, x, flux), initial_gains)
    fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)))
    full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2 * ref_input_index])
    gainsol = full_params.view(np.complex128).astype(np.complex64)
    ant_gains.append(gainsol)
ant_gains = np.array(ant_gains).transpose()

# Interpolate gain as a function of time
amp_interps, phase_interps = [], []
for n in range(len(inputs)):
    amp_interp = scape.fitting.PiecewisePolynomial1DFit()
    amp_interp.fit(gain_times, np.abs(ant_gains[n]))
    amp_interps.append(amp_interp)
    phase_interp = scape.fitting.PiecewisePolynomial1DFit()
    angle = np.angle(ant_gains[n])
    # Do a quick and dirty angle unwrapping
    angle_diff = np.diff(angle)
    angle_diff[angle_diff > np.pi] -= 2 * np.pi
    angle_diff[angle_diff < -np.pi] += 2 * np.pi
    angle[1:] = angle[0] + np.cumsum(angle_diff)
    phase_interp.fit(gain_times, angle)
    phase_interps.append(phase_interp)

fig = plt.figure(6)
fig.clear()
fig.subplots_adjust(right=0.8)
ax = fig.add_subplot(121)
plot_times = np.arange(gain_times[0] - 1000, gain_times[-1] + 1000, 100.)
for n in range(len(inputs)):
    ax.plot(plot_times - gain_times[0], amp_interps[n](plot_times), 'k')
    ax.plot(gain_times - gain_times[0], np.abs(ant_gains[n]), 'o', label=inputs[n][3:])
ax.set_xlabel('Time since start (seconds)')
ax.set_title('Gain amplitude')
ax = fig.add_subplot(122)
for n in range(len(inputs)):
    ax.plot(plot_times - gain_times[0], katpoint.rad2deg(scape.stats.angle_wrap(phase_interps[n](plot_times))), 'k')
    ax.plot(gain_times - gain_times[0], katpoint.rad2deg(np.angle(ant_gains[n])), 'o', label=inputs[n][3:])
ax.set_xlabel('Time since start (seconds)')
ax.set_title('Gain phase (degrees)')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0., numpoints=1)

# Apply both bandpass and gain calibration to bandpass calibrator visibilities, as a check of the procedure
corrected_bpcal_vis_samples = [vis.copy() for vis in cal_vis_samples]
for vis, timestamps in zip(corrected_bpcal_vis_samples, cal_timestamps):
    # Interpolate antenna gains to timestamps of visibilities
    interp_ant_gains = np.zeros((len(inputs), len(timestamps)), dtype=np.complex64)
    for n in range(len(inputs)):
        interp_ant_gains[n, :] = amp_interps[n](timestamps) * np.exp(1.0j * phase_interps[n](timestamps))
    for n, (indexA, indexB) in enumerate(crosscorr):
        vis[:, :, n] /= (bandpass_gains[indexA, :] * bandpass_gains[indexB, :].conjugate())
        vis[:, :, n] /= (interp_ant_gains[indexA, :] * interp_ant_gains[indexB, :].conjugate())[:, np.newaxis]

fig = plt.figure(7)
fig.clear()
plot_vis_crosshairs(fig, [vis[:, freq_slice, :] for vis in bp_cal_vis_samples],
                    "'%s' gain-corrected vis for channel %d (all times)" % (bandpass_cal.name, freq_slice + first_chan,),
                    upper=True, units='Jy', marker='.', linestyle='-')
plot_vis_crosshairs(fig, [vis[time_slice, :, :] for vis in bp_cal_vis_samples],
                    "'%s' gain-corrected vis for scan time sample %d (all channels)" % (bandpass_cal.name, time_slice),
                    upper=False, units='Jy', marker='.', linestyle='')

######################### TRANSFER CAL TO TARGET ###############################

print "Applying calibration to imaging target..."

# Assemble visibility data and uvw coordinates for imaging target
vis_samples_per_scan, uvw_samples_per_scan = [], []
start_chans = np.arange(0, (len(wavelengths) // channels_per_band) * channels_per_band, channels_per_band)
for scan_ind, cs_ind, state, target in data.scans():
    if state != 'track' or target != image_target:
        continue
    timestamps = data.timestamps()
    if len(timestamps) < 2:
        continue
    # Extract visibilities and uvw coordinates
    vis_cross = np.zeros((len(timestamps), len(wavelengths), len(crosscorr)), dtype=np.complex64)
    uvw = np.zeros((3, len(timestamps), len(wavelengths), len(crosscorr)))
    # Interpolate antenna gains to timestamps of visibilities
    interp_ant_gains = np.zeros((len(inputs), len(timestamps)), dtype=np.complex64)
    for n in range(len(inputs)):
        interp_ant_gains[n, :] = amp_interps[n](timestamps) * np.exp(1.0j * phase_interps[n](timestamps))
    # Iterate through baselines and assemble visibilities
    for n, (indexA, indexB) in enumerate(crosscorr):
        inputA, inputB = inputs[indexA], inputs[indexB]
        antA, antB = inputA[:-1], inputB[:-1]
        vis = data.vis((inputA, inputB))
        # Get uvw coordinates of A->B baseline as multiples of the channel wavelength
        uvw[:, :, :, n] = np.array(target.uvw(ants[antB], timestamps, ants[antA]))[:, :, np.newaxis] / wavelengths
        # Number of turns of phase that signal B is behind signal A due to cable / receiver delay
        cable_delay_turns = (delays[inputB] - delays[inputA]) * center_freqs
        # Visibility <A, B*> has phase (A - B), therefore add (B - A) phase to stop fringes (i.e. do delay tracking)
        vis *= np.exp(2j * np.pi * (- uvw[2, :, :, n] + cable_delay_turns))
        # Now also apply results of bandpass and gain calibration
        vis /= (bandpass_gains[indexA, :] * bandpass_gains[indexB, :].conjugate())
        vis /= (interp_ant_gains[indexA, :] * interp_ant_gains[indexB, :].conjugate())[:, np.newaxis]
        vis_cross[:, :, n] = vis
    # Average over adjacent time and frequency bins to create coarser bins, which reduces processing load
    start_times = np.arange(0, (len(timestamps) // dumps_per_vis) * dumps_per_vis, dumps_per_vis)
    averaged_vis = np.zeros((len(start_times), len(start_chans), len(crosscorr)), dtype=np.complex64)
    averaged_uvw = np.zeros((3, len(start_times), len(start_chans), len(crosscorr)))
    for m, start_time in enumerate(start_times):
        for n, start_chan in enumerate(start_chans):
            averaged_vis[m, n, :] = vis_cross[start_time:(start_time + dumps_per_vis),
                                              start_chan:(start_chan + channels_per_band), :].mean(axis=0).mean(axis=0)
            averaged_uvw[:, m, n, :] = uvw[:, start_time:(start_time + dumps_per_vis),
                                              start_chan:(start_chan + channels_per_band), :].mean(axis=1).mean(axis=1)
    vis_samples_per_scan.append(averaged_vis)
    uvw_samples_per_scan.append(averaged_uvw)
vis_samples = np.vstack(vis_samples_per_scan).ravel()
uvw_samples = np.hstack(uvw_samples_per_scan)
u_samples, v_samples, w_samples = uvw_samples[0].ravel(), uvw_samples[1].ravel(), uvw_samples[2].ravel()
uvdist = np.sqrt(u_samples * u_samples + v_samples * v_samples)

fig = plt.figure(8)
fig.clear()
plot_vis_crosshairs(fig, [vis.reshape(-1, vis.shape[-1]) for vis in vis_samples_per_scan],
                    "Calibrated '%s' visibilities averaged in coarse bins" % (image_target.name,),
                    upper=True, units='Jy', marker='.', linestyle='-')

fig = plt.figure(9)
fig.clear()
ax = fig.add_subplot(111)
ax.plot(uvdist, np.abs(vis_samples), 'o')
ax.set_title("Calibrated '%s' visibilities averaged in coarse bins" % (image_target.name))
ax.set_xlabel('UV distance (lambda)')
ax.set_ylabel('Visibility amplitude (Jy)')

################################## IMAGE #######################################

print "Producing dirty image of '%s'..." % (image_target.name,)

# Set up image grid coordinates (in radians)
# First get some basic data parameters together (center freq in Hz, primary beamwidth in rads)
band_center = center_freqs[len(center_freqs) // 2]
ref_ant = ants[data.ref_ant]
primary_beam_width = ref_ant.beamwidth * katpoint.lightspeed / band_center / ref_ant.diameter
# The pixel size is a fixed fraction of the synthesised beam width
image_grid_step = 0.1 / uvdist.max()
# The number of pixels is determined by the primary beam width
# (and kept a power of two for compatibility with other packages that use the FFT instead of DFT)
image_size = 2 ** int(np.log2(primary_beam_width / image_grid_step))
num_pixels = image_size * image_size
# Create image pixel (l,m) coordinates similar to CASA (in radians)
m_range = (np.arange(image_size) - image_size // 2) * image_grid_step
l_range = np.flipud(-m_range)
l_image, m_image = np.meshgrid(l_range, m_range)
n_image = np.sqrt(1 - l_image*l_image - m_image*m_image)
lm_positions = np.array([l_image.ravel(), m_image.ravel()]).transpose()

# Direct Fourier imaging (DFT) of dirty beam and image
dirty_beam = np.zeros((image_size, image_size), dtype='double')
dirty_image = np.zeros((image_size, image_size), dtype='double')
for u, v, vis in zip(u_samples, v_samples, vis_samples):
    arg = 2*np.pi*(u*l_image + v*m_image)
    dirty_beam += np.cos(arg)
    dirty_image += np.abs(vis) * np.cos(arg - np.angle(vis))
dirty_beam *= n_image / len(vis_samples)
dirty_image *= n_image / len(vis_samples)

# Plot ranges for casapy
arcmins = 60 * 180 / np.pi
l_plot = l_range * arcmins
m_plot = m_range * arcmins

fig = plt.figure(10)
fig.clear()
ax = fig.add_subplot(111)
ax.plot(u_samples, v_samples, '.', markersize=2)
ax.plot(-u_samples, -v_samples, '.r', markersize=2)
ax.set_xlabel('u (lambda)')
ax.set_ylabel('v (lambda)')
ax.set_title("UV coverage for '%s' target" % (image_target.name,))
uvmax = max(np.abs(ax.axis()))
ax.axis('image')
ax.axis((-uvmax, uvmax, -uvmax, uvmax))

fig = plt.figure(11)
fig.clear()
ax = fig.add_subplot(111)
ax.imshow(dirty_beam, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
ax.set_xlabel('l (arcmins)')
ax.set_ylabel('m (arcmins)')
ax.set_title('Dirty beam')
ax.axis('image')
ax.set_xlim(ax.get_xlim()[::-1])

fig = plt.figure(12)
fig.clear()
ax = fig.add_subplot(111)
ax.imshow(dirty_image, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
ax.set_xlabel('l (arcmins)')
ax.set_ylabel('m (arcmins)')
ax.set_title("Dirty image of '%s' at %.0f MHz" % (image_target.name, band_center / 1e6))
ax.axis('image')
ax.set_xlim(ax.get_xlim()[::-1])
dirty_image_clim = ax.images[0].get_clim()

################################## CLEAN #######################################

print "CLEANing the image..."

# The CLEAN variant that will be used
def omp_plus(A, y, S, At_times=None, A_column=None, N=None, printEveryIter=1, resThresh=0.0):
    """Positive Orthogonal Matching Pursuit.

    This approximately solves the linear system A x = y for sparse positive real x,
    where A is an MxN matrix with M << N. This is very similar to the NNLS algorithm of
    Lawson & Hanson (Solving Least Squares Problems, 1974, Chapter 23).

    Parameters
    ----------
    A : array, shape (M, N)
        The measurement matrix of compressed sensing (None for implicit functions)
    y : array, shape (M,)
        A vector of measurements
    S : integer
        Maximum number of sparse components to find (sparsity level), between 1 and M.
    At_times : function
        Function that calculates At_times(x) = A' * x implicitly. It takes
        an array of shape (M,) as argument and returns an array of shape (N,).
        Default is None.
    A_column : function
        Function that returns the n-th column of A as A_column(n) = A[:, n]. It takes
        an integer as argument and returns an array of shape (M,). Default is None.
    N : integer
        Number of columns in matrix A (dictionary size in MP-speak). Default is None,
        which means it is automatically determined from A.
    printEveryIter : integer
        A progress line is printed every 'printEveryIter' iterations (0 for no
        progress report). The default is a progress report after every iteration.
    resThresh : real
        Stop iterating if the residual l2-norm (relative to the l2-norm of the
        measurements y) falls below this threshold. Default is 0.0 (no threshold).

    Returns
    -------
    x : array of same type as y, shape (N,)
        The approximate sparse positive solution to A x = y.

    """
    # Convert explicit A matrix to functional form (or use provided functions)
    if At_times is None:
        At_times = lambda x: np.dot(A.conjugate().transpose(), x)
        A_column = lambda n: A[:, n]
        N = A.shape[1]
    M = len(y)
    # Initialization
    residual = y
    resSize = 1.0
    atoms = np.zeros((M, S), dtype=y.dtype)
    atomIndex = -np.ones(S, dtype='int32')
    atomWeights = np.zeros(S, dtype='float64')
    atomHistory = set()
    numAtoms = 0
    iterCount = 0
    # Maximum iteration count suggested by Lawson & Hanson
    iterMax = 3 * N

    try:
        # MAIN LOOP (a la NNLS) to find all atoms
        # Loop until the desired number of components / atoms are found, or the
        # residual size drops below the threshold (an earlier exit is also possible)
        while (numAtoms < S) and (resSize >= resThresh):
            # Form the real part of the dirty image residual A' r
            # This happens to be the negative gradient of 0.5 || y - A x ||_2^2,
            # the associated l2-norm (least-squares) objective, and also the dual vector
            dual = At_times(residual).real
            # Ensure that no existing atoms will be selected again
            dual[atomIndex[:numAtoms]] = 0.0
            # Loop until a new atom with positive weight is found, or die trying
            while True:
                newAtom = dual.argmax()
                # Stop if atom is already in active set, or gradient is non-positive
                if dual[newAtom] <= 0.0:
                    break
                # Tentatively add new atom to active set
                atomIndex[numAtoms] = newAtom
                atoms[:, numAtoms] = A_column(newAtom)
                activeAtoms = atoms[:, :numAtoms+1]
                # Solve unconstrained least-squares problem (Gram-Schmidt orthogonalisation step)
                newWeights = np.linalg.lstsq(activeAtoms, y)[0].real
                # If weight of new atom is non-positive, discard it and go for next best atom
                if newWeights[-1] <= 0.0:
                    dual[newAtom] = 0.0
                else:
                    break
            if dual[newAtom] <= 0.0:
                break
            # If search has been in this state before, it will get stuck in endless loop
            # until iterMax is reached, which is pointless
            # TODO: check the effort involved in this check (maybe we don't need it if the
            # endless loop is due to a bug somewhere else?)
            atomState = tuple(sorted(atomIndex[:numAtoms+1]))
            if atomState in atomHistory:
                print "endless loop detected, terminating"
                break
            else:
                atomHistory.add(atomState)
            numAtoms += 1
            # SECONDARY LOOP (a la NNLS) to get all atom weights to be positive simultaneously
            # Forced to terminate if it takes too long
            while iterCount <= iterMax:
                iterCount += 1
                # Check for non-positive weights
                nonPos = [n for n in xrange(len(newWeights)) if newWeights[n] <= 0.0]
                if len(nonPos) == 0:
                    break
                # Interpolate between old and new weights so that at least one atom
                # with negative weight now has zero weight, and can therefore be discarded
                oldWeights = atomWeights[:numAtoms]
                alpha = oldWeights[nonPos] / (oldWeights[nonPos] - newWeights[nonPos])
                worst = alpha.argmin()
                oldWeights += alpha[worst] * (newWeights - oldWeights)
                # Make sure the selected atom really has 0 weight (round-off could change it)
                oldWeights[nonPos[worst]] = 0.0
                # Only keep the atoms with positive weights (could be more efficient...)
                goodAtoms = [n for n in xrange(len(oldWeights)) if oldWeights[n] > 0.0]
                numAtoms = len(goodAtoms)
                print "iter %d : best atom = %d, found negative weights, worst at %d, reduced atoms to %d" % \
                      (iterCount, newAtom, atomIndex[nonPos[worst]], numAtoms)
                atomIndex[:numAtoms] = atomIndex[goodAtoms].copy()
                atomIndex[numAtoms:] = -1
                activeAtoms = atoms[:, goodAtoms].copy()
                atoms[:, :numAtoms] = activeAtoms
                atoms[:, numAtoms:] = 0.0
                atomWeights[:numAtoms] = atomWeights[goodAtoms].copy()
                atomWeights[numAtoms:] = 0.0
                # Solve least-squares problem again to get new proposed atom weights
                newWeights = np.linalg.lstsq(activeAtoms, y)[0].real
            if iterCount > iterMax:
                break
            # Accept new weights, update residual and continue with main loop
            atomWeights[:numAtoms] = newWeights
            residual = y - np.dot(activeAtoms, newWeights)
            resSize = np.linalg.norm(residual) / np.linalg.norm(y)
            if printEveryIter and (iterCount % printEveryIter == 0):
                print "iter %d : best atom = %d, dual = %.3e, atoms = %d, residual l2 = %.3e" % \
                      (iterCount, newAtom, dual[newAtom], numAtoms, resSize)

    # Return last results on Ctrl-C, for the impatient ones
    except KeyboardInterrupt:
        # Create sparse solution vector
        x = np.zeros(N, dtype='float64')
        x[atomIndex[:numAtoms]] = atomWeights[:numAtoms]
        if printEveryIter:
            print 'omp: atoms = %d, residual = %.3e (interrupted)' % (sum(x != 0.0), resSize)

    else:
        # Create sparse solution vector
        x = np.zeros(N, dtype='float64')
        x[atomIndex[:numAtoms]] = atomWeights[:numAtoms]
        if printEveryIter:
            print 'omp: atoms = %d, residual = %.3e' % (sum(x != 0.0), resSize)

    return x

# Set up CLEAN boxes around main peaks in dirty image
# Original simplistic attempt at auto-boxing
# mask = (dirty_image > 0.3 * dirty_image.max()).ravel()
## First try to pick a decent threshold based on a knee shape in sorted amplitudes
sorted_dirty = np.sort(dirty_image.ravel())
norm_sd_x = np.linspace(0, 1, len(sorted_dirty))
norm_sd_y = sorted_dirty / sorted_dirty[-1]
# Break graph into coarse steps, in order to get less noisy slope estimates
norm_sd_coarse_steps = norm_sd_y.searchsorted(np.arange(0., 1., 0.05))
norm_sd_coarse_x = norm_sd_coarse_steps / float(len(sorted_dirty))
norm_sd_coarse_y = norm_sd_y[norm_sd_coarse_steps]
norm_sd_coarse_slope = np.diff(norm_sd_coarse_y) / np.diff(norm_sd_coarse_x)
# Look for rightmost point in graph with a tangent slope of around 1
knee = norm_sd_coarse_steps[norm_sd_coarse_slope.searchsorted(2., side='right') + 1]
# Look for closest point in graph to lower right corner of plot
# knee = np.sqrt((norm_sd_x - 1) ** 2 + norm_sd_y ** 2).argmin()
mask = (dirty_image > sorted_dirty[knee]).ravel()
mask_image = mask.reshape(image_size, image_size)
# Create measurement matrix (potentially *very* big - use smaller mask to reduce it)
masked_phi = np.exp(2j * np.pi * np.dot(np.c_[u_samples, v_samples], lm_positions.T[:, mask]))
# Desired number of pixels (the sparsity level m of the signal)
num_components = 20
vis_snr_dB = 20
# Pick a more sensible threshold in the case of noiseless data
effective_snr_dB = min(vis_snr_dB, 40.0)
res_thresh = 1.0 / np.sqrt(1.0 + 10 ** (effective_snr_dB / 10.0))

# Clean the image
masked_comps = omp_plus(A=masked_phi, y=vis_samples, S=num_components, resThresh=res_thresh)
model_vis_samples = np.dot(masked_phi, masked_comps)
clean_components = np.zeros(image_size * image_size)
clean_components[mask] = masked_comps
clean_components = clean_components.reshape(image_size, image_size)

# Create residual image
residual_vis = vis_samples - model_vis_samples
residual_image = np.zeros((image_size, image_size), dtype='double')
for u, v, vis in zip(u_samples, v_samples, residual_vis):
    arg = 2*np.pi*(u*l_image + v*m_image)
    residual_image += np.abs(vis) * np.cos(arg - np.angle(vis))
residual_image *= n_image / len(residual_vis)

# Create restoring beam from inner part of dirty beam
# Threshold the dirty beam image and identify blobs
blob_image, blob_count = ndimage.label(dirty_beam > 0.2)
# Pick the centre blob and enlarge it slightly to make up for the aggressive thresholding in the previous step
centre_blob = ndimage.binary_dilation(blob_image == blob_image[image_size // 2, image_size // 2])
# Fit Gaussian beam to central part of dirty beam
beam_weights = centre_blob * dirty_beam
lm = np.vstack((l_image.ravel(), m_image.ravel()))
beam_cov = np.dot(lm * beam_weights.ravel(), lm.T) / beam_weights.sum()
restoring_beam = np.exp(-0.5 * np.sum(lm * np.dot(np.linalg.inv(beam_cov), lm), axis=0)).reshape(image_size, image_size)
# Create clean image by restoring with clean beam
clean_image = np.zeros((image_size, image_size), dtype='double')
comps_row, comps_col = clean_components.nonzero()
origin = (image_size // 2, image_size // 2 - 1)
for comp_row, comp_col in zip(comps_row, comps_col):
    flux = clean_components[comp_row, comp_col]
    clean_image += ndimage.shift(flux * restoring_beam, (comp_row - origin[0], comp_col - origin[1]))
# Get final image and corresponding DR estimate
final_image = clean_image + residual_image

print "Estimated dynamic range = ", final_image.max() / residual_image.std()

fig = plt.figure(13)
fig.clear()
ax = fig.add_subplot(111)
ax.imshow(0.2 * mask_image, interpolation='nearest', origin='lower',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]], cmap=mpl.cm.gray_r, vmin=0., vmax=1.)
ax.imshow(np.ma.masked_array(clean_components, clean_components == 0), interpolation='nearest', origin='lower',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
ax.set_xlabel('l (arcmins)')
ax.set_ylabel('m (arcmins)')
ax.set_title('Clean components')
ax.axis('image')
ax.set_xlim(ax.get_xlim()[::-1])

fig = plt.figure(14)
fig.clear()
ax = fig.add_subplot(111)
ax.imshow(residual_image, origin='lower', interpolation='bicubic',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
ax.imshow(mask_image, interpolation='nearest', origin='lower',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]], cmap=mpl.cm.gray_r, alpha=0.5)
ax.set_xlabel('l (arcmins)')
ax.set_ylabel('m (arcmins)')
ax.set_title("Residual image")
ax.axis('image')
ax.set_xlim(ax.get_xlim()[::-1])

fig = plt.figure(15)
fig.clear()
ax = fig.add_subplot(111)
ax.imshow(final_image, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
ax.set_xlabel('l (arcmins)')
ax.set_ylabel('m (arcmins)')
ax.set_title("Clean image of '%s' at %.0f MHz" % (image_target.name, band_center / 1e6))
ax.axis('image')
ax.set_xlim(ax.get_xlim()[::-1])
ax.images[0].set_cmap(mpl.cm.gist_heat)

# Save final image to FITS
ra0, dec0 = image_target.radec()
fits_filename = '%s_%.0fMHz.fits' % (image_target.name.replace(' ', ''), band_center / 1e6)
# The PyFITS package has a problem with the clobber flag, therefore remove any existing file or face a crash
if os.path.isfile(fits_filename):
    print "Overwriting existing file '%s'" % (fits_filename,)
    os.remove(fits_filename)
# Normalise image by the beam volume to get to Jy/beam units
scape.plots_basic.save_fits_image(fits_filename,
                                  katpoint.rad2deg(ra0 + l_range[::-1]), katpoint.rad2deg(dec0 + m_range),
                                  np.fliplr(final_image / restoring_beam.sum()), target_name=image_target.name,
                                  coord_system='radec', projection_type='SIN', data_unit='Jy/beam',
                                  observe_date=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(data.start_time)),
                                  create_date=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                                  telescope='KAT-7', observer=data.observer)
