#
# Attempt to image Centaurus A.
#
# Ludwig Schwardt
# 7 April 2010
#

import re

import numpy as np
import matplotlib.pyplot as plt

import h5py
import katpoint
import scape

# Correlator baseline format
baselines = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 2), (1, 3), (0, 3), (1, 2), (2, 2), (3, 3), (2, 3), (3, 2)]
autocorr = [0, 1, 8, 9]
crosscorr = [2, 4, 5, 6, 7, 10]
# Frequency channel range to keep, and number of channels to average together into band
first_chan, one_past_last_chan, channels_per_band = 98, 418, 32
dumps_per_vis = 90

# Load and fit noise diode models
nd = []
std_temp = lambda freq, temp: np.tile(0.04, len(temp))
a1v = np.loadtxt('T_nd_A1V_coupler.txt', delimiter=',')
nd.append(scape.fitting.Spline1DFit(std_y=std_temp))
nd[0].fit(a1v[:, 0], a1v[:, 1])
a2v = np.loadtxt('T_nd_A2V_coupler.txt', delimiter=',')
nd.append(scape.fitting.Spline1DFit(std_y=std_temp))
nd[1].fit(a2v[:, 0], a2v[:, 1])
a3v = np.loadtxt('T_nd_A3V_coupler.txt', delimiter=',')
nd.append(scape.fitting.Spline1DFit(std_y=std_temp))
nd[2].fit(a3v[:, 0], a3v[:, 1])
a4v = np.loadtxt('T_nd_A4V_coupler.txt', delimiter=',')
nd.append(scape.fitting.Spline1DFit(std_y=std_temp))
nd[3].fit(a4v[:, 0], a4v[:, 1])

# Open data file
f = h5py.File('1270070695.h5', 'r')

# Updated antennas after baseline calibration
new_ants = {'Antenna1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.119 -8.944 0.083, , 1.22', 478.041e-9),
            'Antenna2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.315 26.648 -0.067, , 1.22', 545.235e-9),
            'Antenna3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.989 26.925 -0.006, , 1.22', 669.900e-9),
            'Antenna4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22', 772.868e-9)}

# Create antenna objects based on updated baseline cal (swapping antennas 1 and 2 relative to the data)
ant_names = ['Antenna2', 'Antenna1', 'Antenna3', 'Antenna4']
#ants = [katpoint.Antenna(f['Antennas'][name].attrs['description']) for name in ant_names]
ants = [katpoint.Antenna(new_ants[name][0]) for name in ant_names]
delays = [new_ants[name][1] for name in ant_names]

# Extract frequency information
band_center = f['Correlator'].attrs['center_frequency_hz']
channel_bw = f['Correlator'].attrs['channel_bandwidth_hz']
num_chans = f['Correlator'].attrs['num_freq_channels']
# Assume that lower-sideband downconversion has been used, which flips frequency axis
# Also subtract half a channel width to get frequencies at center of each channel
center_freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
# Select channel range to work on at an early stage
center_freqs = center_freqs[first_chan:one_past_last_chan]
dump_rate = f['Correlator'].attrs['dump_rate_hz']
sample_period = 1.0 / dump_rate
wavelengths = katpoint.lightspeed / center_freqs

# First extract noise diode firings and deduce gain (in counts/K)
nd_gains = []
for ant_index, ant in enumerate(f['Antennas']):
    nd_jump_power_mu = []
    for compscan in f['Scans']:
        if 'Scan2' not in f['Scans'][compscan]:
            continue
        # Get 'cal' scan
        scan_group = f['Scans'][compscan]['Scan2']
        # Extract autocorr data of given antenna
        data = scan_group['data'][str(autocorr[ant_index])].real[:, first_chan:one_past_last_chan]
        # Segments are hard-coded to ignore transitions and time offset - noise diode sensor not used, as Antenna 2 has a broken one
        off_segment, on_segment = [0] + range(8, min(data.shape[0], 30)), [3, 4, 5]
        # Calculate mean and standard deviation of the *averaged* power data in the two segments.
        # Since the estimated mean of data is less variable than the data itself, we have to divide the data sigma by sqrt(N).
        nd_off_mu, nd_off_sigma = data[off_segment, :].mean(axis=0), data[off_segment, :].std(axis=0)
        nd_off_sigma /= np.sqrt(len(off_segment))
        nd_on_mu, nd_on_sigma = data[on_segment, :].mean(axis=0), data[on_segment, :].std(axis=0)
        nd_on_sigma /= np.sqrt(len(on_segment))
        # Obtain mean and standard deviation of difference between averaged power in the segments
        nd_delta_mu, nd_delta_sigma = nd_on_mu - nd_off_mu, np.sqrt(nd_on_sigma ** 2 + nd_off_sigma ** 2)
        # Only keep jumps with significant *increase* in power (focus on the positive HH/VV)
        # This discards segments where noise diode did not fire as expected
        norm_jump = nd_delta_mu / nd_delta_sigma
        if np.mean(norm_jump, axis=0).max() > 10.0:
            nd_jump_power_mu.append(nd_delta_mu)
    nd_jump_power_mu = np.vstack(nd_jump_power_mu)
    temp_nd = nd[ant_index](center_freqs)
    # Keep the square root of gain, to simplify normalisation of cross-correlation visibilities
    nd_gains.append(np.sqrt(nd_jump_power_mu.mean(axis=0) / temp_nd))

# Hard-code the expected target objects for now
close_cal = katpoint.Target('1218-460, radec J2000, 12:18:06.25, -46:00:29.0')
far_cal = katpoint.Target('3C273 | J1229+0203, radec J2000, 12:29:06.70, 2:03:08.6, (1410.0 8400.0 10.0465955419 -4.90623613415 0.709890678937)')
cenA = katpoint.Target('radec, 13:25:27.60, -43:01:09.0')
cenA_N = katpoint.Target('radec, 13:25:27.60, -42:32:23.8')
cenA_S = katpoint.Target('radec, 13:25:27.60, -43:29:54.2')

# Assemble visibility data and uvw coordinates for main (bandpass) calibrator
orig_cal_vis_samples, cal_vis_samples, cal_timestamps = [], [], []
for compscan in f['Scans']:
    compscan_group = f['Scans'][compscan]
    target = katpoint.Target(compscan_group.attrs['target'])
    if target.name != far_cal.name:
        continue
    # Get 'scan' scan
    scan_group = compscan_group['Scan1']
    # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
    # Also move correlator data timestamps from start of each sample to the middle
    timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0 + 0.5 * sample_period
    # Extract visibilities and uvw coordinates
    vis_auto = np.array([scan_group['data'].value[str(index)][:, first_chan:one_past_last_chan] for index in autocorr]).real
    vis_cross = np.array([scan_group['data'].value[str(index)][:, first_chan:one_past_last_chan] for index in crosscorr]).transpose(1, 2, 0)
    orig_cal_vis_samples.append(vis_cross.copy())
    uvw = np.zeros((3, len(timestamps), len(wavelengths), len(crosscorr)))
    for n, index in enumerate(crosscorr):
        antA, antB = baselines[index]
        # Get uvw coordinates as multiples of the channel wavelength
        uvw[:, :, :, n] = np.array(target.uvw(ants[antB], timestamps, ants[antA]))[:, :, np.newaxis] / wavelengths
        # Get cable delay difference as multiples of the channel wavelength
        cable_delay_diff = (delays[antB] - delays[antA]) * center_freqs
        # Normalise cross-corr by autocorr to get correlation coefficients in range [-1, 1]
#        vis_cross[:, :, n] /= np.sqrt(vis_auto[antA] * vis_auto[antB])
        # Normalise cross-corr by gain derived from noise diode firings
        vis_cross[:, :, n] /= (nd_gains[antA] * nd_gains[antB])
        # Also stop fringes (do delay tracking) based on w coordinate
        vis_cross[:, :, n] *= np.exp(-2j * np.pi * (uvw[2, :, :, n] + cable_delay_diff))
    cal_vis_samples.append(vis_cross)
    cal_timestamps.append(timestamps)

plt.figure(1)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in orig_cal_vis_samples:
        plt.plot(vis[:, 50, n].real, vis[:, 50, n].imag, '-o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-100, 100, -100, 100])

plt.figure(2)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in orig_cal_vis_samples:
        plt.plot(vis[30, :, n].real, vis[30, :, n].imag, 'o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-100, 100, -100, 100])

plt.figure(3)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in cal_vis_samples:
        plt.plot(vis[:, 50, n].real, vis[:, 50, n].imag, '-o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-2, 2, -2, 2])

plt.figure(4)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in cal_vis_samples:
        plt.plot(vis[30, :, n].real, vis[30, :, n].imag, 'o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-2, 2, -2, 2])

# Assume the last antenna (antenna 4) is the reference antenna
full_params = np.zeros(8)
def gains_to_vis(params, x):
    """Estimate visibility of point source from relevant antenna gains."""
    full_params[:7] = params
    antA, antB = x[0], x[1]
    reA, imA, reB, imB = full_params[2 * antA], full_params[2 * antA + 1], full_params[2 * antB], full_params[2 * antB + 1]
    return np.vstack((reA * reB + imA * imB, imA * reB - reA * imB)).squeeze()

# Solve for antenna bandpass gains
initial_gains = np.array([1., 0., 1., 0., 1., 0., 1.])
bandpass_gainsols = []
bp_source_vis = far_cal.flux_density(center_freqs / 1e6)
# Iterate over solution intervals
for vis in cal_vis_samples:
    gainsol = np.zeros((4, vis.shape[1]), dtype=np.complex64)
    # Iterate over frequency channels
    for n in xrange(vis.shape[1]):
        fitter = scape.fitting.NonLinearLeastSquaresFit(gains_to_vis, initial_gains)
        x = np.tile(np.array(baselines)[crosscorr].transpose(), vis.shape[0])
        gain_product = vis[:, n, :].ravel() / bp_source_vis[n]
        y = np.vstack((gain_product.real, gain_product.imag))
        fitter.fit(x, y)
        p = fitter.params * np.sign(fitter.params[6])
        gainsol[0, n] = p[0] + 1.0j * p[1]
        gainsol[1, n] = p[2] + 1.0j * p[3]
        gainsol[2, n] = p[4] + 1.0j * p[5]
        gainsol[3, n] = p[6]
    bandpass_gainsols.append(gainsol)

# Combine bandpass gain solutions into a single solution by removing drifts from the first one and averaging
for bp_gain in bandpass_gainsols[1:]:
    amp_drift = np.exp((np.log(np.abs(bp_gain)) - np.log(np.abs(bandpass_gainsols[0]))).mean(axis=1))
    angle_diff = np.angle(bp_gain) - np.angle(bandpass_gainsols[0])
    # Calculate a "safe" mean angle on the unit circle
    phase_drift = np.arctan2(np.sin(angle_diff).mean(axis=1), np.cos(angle_diff).mean(axis=1))
    gain_drift = amp_drift * np.exp(1.0j * phase_drift)
    bp_gain /= gain_drift[:, np.newaxis]
bandpass_gains = np.dstack(bandpass_gainsols).mean(axis=2)

# Apply bandpass gain calibration to cal source visibilities
bp_cal_vis_samples = [vis.copy() for vis in cal_vis_samples]
for vis in bp_cal_vis_samples:
    for n, index in enumerate(crosscorr):
        antA, antB = baselines[index]
        vis[:, :, n] /= (bandpass_gains[antA, :] * bandpass_gains[antB, :].conjugate())

plt.figure(5)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in bp_cal_vis_samples:
        plt.plot(vis[:, 50, n].real, vis[:, 50, n].imag, '-o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-80, 80, -80, 80])

plt.figure(6)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in bp_cal_vis_samples:
        plt.plot(vis[30, :, n].real, vis[30, :, n].imag, 'o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-80, 80, -80, 80])

# Assemble visibility data and uvw coordinates for all calibrators, and average it to a single frequency band
all_cal_vis_samples, cal_source, all_cal_times = [], [], []
for compscan in f['Scans']:
    compscan_group = f['Scans'][compscan]
    target = katpoint.Target(compscan_group.attrs['target'])
    if target.name != far_cal.name and target.name != close_cal.name:
        continue
    # Get 'scan' scan
    scan_group = compscan_group['Scan1']
    # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
    # Also move correlator data timestamps from start of each sample to the middle
    timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0 + 0.5 * sample_period
    # Extract visibilities and uvw coordinates
    vis_cross = np.array([scan_group['data'].value[str(index)][:, first_chan:one_past_last_chan] for index in crosscorr]).transpose(1, 2, 0)
    uvw = np.zeros((3, len(timestamps), len(wavelengths), len(crosscorr)))
    for n, index in enumerate(crosscorr):
        antA, antB = baselines[index]
        # Get uvw coordinates as multiples of the channel wavelength
        uvw[:, :, :, n] = np.array(target.uvw(ants[antB], timestamps, ants[antA]))[:, :, np.newaxis] / wavelengths
        # Get cable delay difference as multiples of the channel wavelength
        cable_delay_diff = (delays[antB] - delays[antA]) * center_freqs
        # Normalise cross-corr by gain derived from noise diode firings
        vis_cross[:, :, n] /= (nd_gains[antA] * nd_gains[antB])
        # Also stop fringes (do delay tracking) based on w coordinate
        vis_cross[:, :, n] *= np.exp(-2j * np.pi * (uvw[2, :, :, n] + cable_delay_diff))
        # Now apply bandpass calibration
        vis_cross[:, :, n] /= (bandpass_gains[antA, :] * bandpass_gains[antB, :].conjugate())
    # It should be safe to average all channels into a single band now, for the purpose of gain calibration
    all_cal_vis_samples.append(vis_cross.mean(axis=1))
    cal_source.append(target)
    all_cal_times.append(timestamps[30])

# Sort the data chronologically
time_index = np.argsort(all_cal_times)
all_cal_vis_samples = [all_cal_vis_samples[n] for n in time_index]
cal_source = np.array([cal_source[n] for n in time_index])
all_cal_times = np.array([all_cal_times[n] for n in time_index])

plt.figure(7)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in all_cal_vis_samples:
        plt.plot(vis[:, n].real, vis[:, n].imag, '-o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-60, 60, -60, 60])

# Average each solution interval as well, to get rid of the bubbles associated with antenna 1 in the previous figure
gain_cal_vis = np.array([vis.mean(axis=0) for vis in all_cal_vis_samples])
# Obtain average flux of calibrators
# 3C273 has known flux, but the other calibrator has to be bootstrapped off the average data in a SETJY procedure
close_cal_source = np.array([target.name == close_cal.name for target in cal_source])
far_cal_flux = bp_source_vis.mean()
close_cal_flux = far_cal_flux / np.abs(gain_cal_vis[~close_cal_source, :]).mean() * np.abs(gain_cal_vis[close_cal_source, :]).mean()
cal_source_vis = np.zeros(gain_cal_vis.shape[0])
cal_source_vis[close_cal_source] = close_cal_flux
cal_source_vis[~close_cal_source] = far_cal_flux
# On second thought, throw out the weak calibrator, as it might not play well together with 3C273
gain_cal_vis, cal_source_vis, gain_times = gain_cal_vis[close_cal_source, :], cal_source_vis[close_cal_source], all_cal_times[close_cal_source]

# Solve for time-varying antenna gains
initial_gains = np.array([1., 0., 1., 0., 1., 0., 1.])
ant_gains = []
# Iterate over solution intervals
for vis, flux in zip(gain_cal_vis, cal_source_vis):
    gainsol = np.zeros(4, dtype=np.complex64)
    fitter = scape.fitting.NonLinearLeastSquaresFit(gains_to_vis, initial_gains)
    x = np.array(baselines)[crosscorr].transpose()
    gain_product = vis / flux
    y = np.vstack((gain_product.real, gain_product.imag))
    fitter.fit(x, y)
    p = fitter.params * np.sign(fitter.params[6])
    gainsol[0] = p[0] + 1.0j * p[1]
    gainsol[1] = p[2] + 1.0j * p[3]
    gainsol[2] = p[4] + 1.0j * p[5]
    gainsol[3] = p[6]
    ant_gains.append(gainsol)
ant_gains = np.array(ant_gains).transpose()

# Interpolate gain as a function of time
amp_interps, phase_interps = [], []
for n in range(4):
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

plt.figure(8)
plt.clf()
plt.subplot(121)
plot_times = np.arange(gain_times[0] - 1000, gain_times[-1] + 1000, 100.)
for n in range(4):
    plt.plot(plot_times - gain_times[0], amp_interps[n](plot_times), 'k')
    plt.plot(gain_times - gain_times[0], np.abs(ant_gains[n]), 'o', label='ant%d' % (n + 1))
plt.xlabel('Time since start (seconds)')
plt.ylabel('Gain amplitude')
plt.legend(loc='upper left')
plt.subplot(122)
for n in range(4):
    plt.plot(plot_times - gain_times[0], katpoint.rad2deg(scape.stats.angle_wrap(phase_interps[n](plot_times))), 'k')
    plt.plot(gain_times - gain_times[0], katpoint.rad2deg(np.angle(ant_gains[n])), 'o', label='ant%d' % (n + 1))
plt.xlabel('Time since start (seconds)')
plt.ylabel('Gain phase (degrees)')
plt.legend(loc='lower left')

# Apply both bandpass and gain calibration to cal source visibilities
final_cal_vis_samples = [vis.copy() for vis in cal_vis_samples]
for vis, timestamps in zip(final_cal_vis_samples, cal_timestamps):
    # Interpolate antenna gains to timestamps of visibilities
    interp_ant_gains = np.zeros((4, len(timestamps)), dtype=np.complex64)
    for n in range(4):
        interp_ant_gains[n, :] = amp_interps[n](timestamps) * np.exp(1.0j * phase_interps[n](timestamps))
    for n, index in enumerate(crosscorr):
        antA, antB = baselines[index]
        vis[:, :, n] /= (bandpass_gains[antA, :] * bandpass_gains[antB, :].conjugate())
        vis[:, :, n] /= (interp_ant_gains[antA, :] * interp_ant_gains[antB, :].conjugate())[:, np.newaxis]

plt.figure(9)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in final_cal_vis_samples:
        plt.plot(vis[:, 50, n].real, vis[:, 50, n].imag, '-o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-80, 80, -80, 80])

plt.figure(10)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in final_cal_vis_samples:
        plt.plot(vis[30, :, n].real, vis[30, :, n].imag, 'o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-80, 80, -80, 80])

# Assemble visibility data and uvw coordinates for imaging target
vis_samples_per_scan, uvw_samples_per_scan = [], []
for compscan in f['Scans']:
    # Discard tracks on CenA that are not straddled by calibrator sources (the last ones)
    if int(compscan[12:]) >= 30:
        continue
    compscan_group = f['Scans'][compscan]
    target = katpoint.Target(compscan_group.attrs['target'])
    if target.name != cenA_N.name:
        continue
    # Get 'scan' scan
    scan_group = compscan_group['Scan1']
    # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
    # Also move correlator data timestamps from start of each sample to the middle
    timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0 + 0.5 * sample_period
    # Extract visibilities and uvw coordinates
    vis_cross = np.array([scan_group['data'].value[str(index)][:, first_chan:one_past_last_chan] for index in crosscorr]).transpose(1, 2, 0)
    uvw = np.zeros((3, len(timestamps), len(wavelengths), len(crosscorr)))
    # Interpolate antenna gains to timestamps of visibilities
    interp_ant_gains = np.zeros((4, len(timestamps)), dtype=np.complex64)
    for n in range(4):
        interp_ant_gains[n, :] = amp_interps[n](timestamps) * np.exp(1.0j * phase_interps[n](timestamps))
    for n, index in enumerate(crosscorr):
        antA, antB = baselines[index]
        # Get uvw coordinates as multiples of the channel wavelength
        uvw[:, :, :, n] = np.array(target.uvw(ants[antB], timestamps, ants[antA]))[:, :, np.newaxis] / wavelengths
        # Get cable delay difference as multiples of the channel wavelength
        cable_delay_diff = (delays[antB] - delays[antA]) * center_freqs
        # Normalise cross-corr by gain derived from noise diode firings
        vis_cross[:, :, n] /= (nd_gains[antA] * nd_gains[antB])
        # Stop fringes (do delay tracking) based on w coordinate
        vis_cross[:, :, n] *= np.exp(-2j * np.pi * (uvw[2, :, :, n] + cable_delay_diff))
        # Now also apply results of bandpass and gain calibration
        vis_cross[:, :, n] /= (bandpass_gains[antA, :] * bandpass_gains[antB, :].conjugate())
        vis_cross[:, :, n] /= (interp_ant_gains[antA, :] * interp_ant_gains[antB, :].conjugate())[:, np.newaxis]
    # Average over adjacent time and frequency bins to create coarser bins, which reduces processing load
    start_times = np.arange(0, len(timestamps), dumps_per_vis)
    start_chans = np.arange(0, len(wavelengths), channels_per_band)
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

plt.figure(11)
plt.clf()
for n, index in enumerate(crosscorr):
    antA, antB = baselines[index]
    plt.subplot(2, 3, n + 1)
    for vis in vis_samples_per_scan:
        plt.plot(vis[:, :, n].ravel().real, vis[:, :, n].ravel().imag, 'o')
    plt.title('%d-%d' % (antA + 1, antB + 1))
    plt.axis('image')
    plt.axis([-200, 200, -200, 200])

plt.figure(12)
plt.clf()
plt.plot(uvdist, np.abs(vis_samples), 'o')
plt.xlabel('UV distance (lambda)')
plt.ylabel('Visibility amplitude (Jy)')

# Set up image grid coordinates (in radians)
coarse_factor = 0.5
image_size = int(32 / coarse_factor)
num_pixels = image_size * image_size
image_grid_step = 0.2 / uvdist.max() * coarse_factor
# Create image pixel (l,m) coordinates similar to CASA (in radians)
m_range = (np.arange(image_size) - image_size // 2) * image_grid_step
l_range = np.flipud(-m_range)
l_image, m_image = np.meshgrid(l_range, m_range)
n_image = np.sqrt(1 - l_image*l_image - m_image*m_image)
lm_positions = np.array([l_image.ravel(), m_image.ravel()]).transpose()

# Direct Fourier imaging of dirty beam and image
dirty_beam = np.zeros((image_size, image_size), dtype='double')
dirty_image = np.zeros((image_size, image_size), dtype='double')
for u, v, vis in zip(u_samples, v_samples, vis_samples):
    arg = 2*np.pi*(u*l_image + v*m_image)
    dirty_beam += np.cos(arg)
    dirty_image += np.abs(vis) * np.cos(arg - np.angle(vis))
dirty_beam *= n_image / len(vis_samples)
dirty_image *= n_image / len(vis_samples)

# Clean the image...
# Create measurement matrix (potentially *very* big)
phi = np.exp(1j * 2 * np.pi * np.dot(np.column_stack([u_samples, v_samples]), lm_positions.transpose()))
#phi_angle = 2 * np.pi * np.dot(np.column_stack([u_samples, v_samples]), lm_positions.transpose())
#phi_real = np.vstack((np.cos(phi_angle), np.sin(phi_angle)))
#del phi_angle
# Desired number of pixels (the sparsity level m of the signal)
num_components = 20
vis_snr_dB = 10
# Pick a more sensible threshold in the case of noiseless data
effective_snr_dB = min(vis_snr_dB, 40.0)
res_thresh = 1.0 / np.sqrt(1.0 + 10 ** (effective_snr_dB / 10.0))

# Clean the image
from compsense.greedy_pos import omp
omp_sources = omp(A=phi, y=vis_samples, S=num_components, resThresh=res_thresh)
#from compsense.minl1_pos_customkkt import regls_qp
#bp_sources = regls_qp(A=phi_real, y=np.hstack([vis_samples.real.astype(np.float64), vis_samples.imag.astype(np.float64)]),
#                      gamma=1.0)
clean_components = omp_sources.reshape(image_size, image_size)

# Create restoring beam from inner part of dirty beam (very beam-specific for middle cenA target!)
restoring_beam = dirty_beam[20:45, 25:38].copy()
restoring_beam[restoring_beam < 0.0] = 0.0
restoring_beam[:,0] *= 0.25
restoring_beam[:,-1] *= 0.25
# Create clean image by restoring with clean beam
clean_image = np.zeros((image_size, image_size), dtype='double')
comps_row, comps_col = clean_components.nonzero()
for comp_row, comp_col in zip(comps_row, comps_col):
    flux = clean_components[comp_row, comp_col]
    top, bottom = min(comp_row, 12), min(image_size - comp_row, 13)
    left, right = min(comp_col, 6), min(image_size - comp_col, 7)
    clean_image[(comp_row - top):(comp_row + bottom), (comp_col - left):(comp_col + right)] += \
        flux * restoring_beam[(12 - top):(12 + bottom), (6 - left):(6 + right)]

# Plot ranges for casapy
arcmins = 60 * 180 / np.pi
l_plot = l_range * arcmins
m_plot = m_range * arcmins

plt.figure(13)
plt.clf()
plt.plot(u_samples, v_samples, '.', markersize=1)
plt.xlabel('u (lambda)')
plt.ylabel('v (lambda)')
plt.title('UV coverage')
plt.axis('equal')

plt.figure(14)
plt.clf()
plt.imshow(dirty_beam, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
plt.xlabel('l (arcmins)')
plt.ylabel('m (arcmins)')
plt.title('Dirty beam')
plt.axis('image')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])

plt.figure(15)
plt.clf()
plt.imshow(dirty_image, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
plt.xlabel('l (arcmins)')
plt.ylabel('m (arcmins)')
plt.title('Dirty image of Cen A at 1820 MHz')
plt.axis('image')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])

plt.figure(16)
plt.clf()
plt.imshow(clean_components, interpolation='nearest', origin='lower', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
plt.xlabel('l (arcmins)')
plt.ylabel('m (arcmins)')
plt.title('Clean components')
plt.axis('image')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])

plt.figure(17)
plt.clf()
plt.imshow(clean_image, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
plt.xlabel('l (arcmins)')
plt.ylabel('m (arcmins)')
plt.title('Clean image of Cen A at 1820 MHz')
plt.axis('image')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
