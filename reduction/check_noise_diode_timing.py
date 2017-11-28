#!/usr/bin/python
#
# This examines all noise diode firings in an HDF5 file and compares the timestamp
# of each firing with one derived from the rise in recorded power.
#
# Ludwig Schwardt
# 8 February 2011
#

import time
import optparse
import sys
import pickle

import numpy as np
import katdal

################################################### Helper routines ###################################################

def contiguous_cliques(x, step=1):
    """Partition *x* into contiguous cliques, where elements in clique differ by *step*."""
    if len(x) == 0:
        return []
    cliques, current, prev = [], [x[0]], x[0]
    for n in x[1:]:
        if n == prev + step:
            current.append(n)
        else:
            cliques.append(current)
            current = [n]
        prev = n
    cliques.append(current)
    return cliques

def ratio_stats(mean_num, std_num, mean_den, std_den, corrcoef=0):
    """Approximate second-order statistics of ratio of correlated normal variables."""
    # Transform num/den to standard form (a + x) / (b + y), with x and y uncorrelated standard normal vars
    # Then num/den is distributed as 1/r (a + x) / (b + y) + s
    # Therefore determine parameters a and b, scale r and translation s
    s = corrcoef * std_num / std_den
    a, b = mean_num - s * mean_den, mean_den / std_den
    # Pick the sign of h so that a and b have the same sign
    sign_h = 2 * (a >= 0) * (b >= 0) - 1
    h = sign_h * std_num * np.sqrt(1. - corrcoef ** 2)
    a, r = a / h, std_den / h
    # Calculate the approximate mean and standard deviation of (a + x) / (b + y) a la F-distribution
    mean_axby = a * b / (b**2 - 1)
    std_axby = np.abs(b) / (b**2 - 1) * np.sqrt((a**2 + b**2 - 1) / (b**2 - 2))
    # Translate by s and scale by r
    return s + mean_axby / r, std_axby / np.abs(r)

def find_jumps(timestamps, power, std_power, margin_factor, jump_significance, max_onoff_segment_duration, **kwargs):
    """Find significant jumps in power and estimate the time instant of each jump."""
    # Margin within which power is deemed to be constant (i.e. expected variation of power)
    # This works around potential biases in power levels, such as small linear slopes across jump
    margin = margin_factor * std_power
    upper, lower = power + margin, power - margin
    delta_power = np.r_[power[1:], power[-1]] - np.r_[power[0], power[:-1]]
    # Shifted versions of the upper and lower power bounds
    previous_upper, next_upper = np.r_[upper[0], upper[:-1]], np.r_[upper[1:], upper[-1]]
    previous_lower, next_lower = np.r_[lower[0], lower[:-1]], np.r_[lower[1:], lower[-1]]
    # True for each power value that is considered to be the same as the one on its left
    same_as_previous = ((power > previous_lower) & (power < previous_upper)).tolist()
    same_as_previous[0] = False
    # True for each power value that is considered to be the same as the one on its right
    same_as_next = ((power > next_lower) & (power < next_upper)).tolist()
    same_as_next[-1] = False

    jumps = []
    # Look for large instantaneous rises in power (spanning at most 2 dumps)
    # Then pick midpoint of jump (or end of jump if no clear midpoint), and add it to candidate jumps
    rise = np.where((power > previous_upper) & (power < next_upper))[0].tolist()
    jumps += [clique[0] for clique in contiguous_cliques(rise)
              if (len(clique) == 1) or ((len(clique) == 2) and (delta_power[clique[0]] > delta_power[clique[1]]))]
    # Look for large instantaneous drops in power (spanning at most 2 dumps)
    # Then pick midpoint of jump (or end of jump if no clear midpoint), and add it to candidate jumps
    drop = np.where((power < previous_lower) & (power > next_lower))[0].tolist()
    jumps += [clique[0] for clique in contiguous_cliques(drop)
              if (len(clique) == 1) or ((len(clique) == 2) and (delta_power[clique[0]] < delta_power[clique[1]]))]
    # Throw out jumps on the very edges of time series (we need a dump before and after the jump)
    jumps = [jump for jump in jumps if jump not in (0, len(power) - 1)]

    # Investigate each jump and determine accurate timestamp with corresponding uncertainty
    jump_time, jump_std_time, jump_size = [], [], []
    for jump in jumps:
        # Margin / uncertainty associated with power difference between samples adjacent to jump
        jump_margin = np.sqrt(margin[jump - 1] ** 2 + margin[jump + 1] ** 2)
        # Throw out insignificant jumps
        if np.abs(delta_power[jump]) / jump_margin <= jump_significance:
            continue
        # Limit the range of points around jump to use in estimation of jump instant (to ensure stationarity)
        segment_range = np.abs(timestamps - timestamps[jump]) <= max_onoff_segment_duration
        # Determine earliest (and last) sample to use to estimate the mean power before (and after) the jump
        before = min(np.where(segment_range)[0][0], jump - 1)
        after = max(len(segment_range) - 1 - np.where(segment_range[::-1])[0][0], jump + 1)
        before = max(jump - 1 - same_as_previous[jump - 1::-1].index(False), before)
        after = min(same_as_next.index(False, jump + 1), after)
        # Estimate power before and after jump, with corresponding uncertainty
        mean_power_before, mean_power_after = power[before:jump].mean(), power[jump + 1:after + 1].mean()
        std_power_before = np.sqrt(np.sum(std_power[before:jump] ** 2)) / (jump - before)
        std_power_after = np.sqrt(np.sum(std_power[jump + 1:after + 1] ** 2)) / (after - jump)
        # Use ratio of power differences (at - before) / (after - before) to estimate where in dump the jump happened
        mean_num, mean_den = power[jump] - mean_power_before, mean_power_after - mean_power_before
        std_num = np.sqrt(std_power[jump] ** 2 + std_power_before ** 2)
        std_den = np.sqrt(std_power_after ** 2 + std_power_before ** 2)
        # Since "before" power appears in both numerator and denominator, they are (slightly) correlated.
        # NOTE: The complementary ratio (after - at) / (after - before) can also be used, and at first glance
        # it appears to be better if std_power_after < std_power_before, which will result in smaller std_num
        # for the same std_den. The correlation coefficient will change in such a way to cancel out this advantage,
        # however, resulting in the same stats. We therefore do not need to consider the complementary ratio.
        corrcoef = std_power_before ** 2 / std_num / std_den
        mean_subdump, std_subdump = ratio_stats(mean_num, std_num, mean_den, std_den, corrcoef)
        # Estimate instant of jump with corresponding uncertainty (assumes timestamps are accurately known)
        jump_time.append(mean_subdump * timestamps[jump] + (1. - mean_subdump) * timestamps[jump + 1])
        jump_std_time.append(std_subdump * (timestamps[jump + 1] - timestamps[jump]))
        # Refined estimate of the significance of the jump, using averaged data instead of single dumps
        jump_size.append(mean_den / std_den / margin_factor)
    return jump_time, jump_std_time, jump_size

#################################################### Main function ####################################################

parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description="This checks the noise diode timing in the given HDF5 file.")
parser.add_option('-f', '--freq-chans',
                  help="Range of frequency channels to use "
                       "(zero-based, specified as 'start,end', default is [0.25*num_chans, 0.75*num_chans])")
parser.add_option('-o', '--max-offset', type='float', default=2.,
                  help="Maximum allowed offset between CAM and DBE timestamps, in accumulations (default %default)")
parser.add_option('-d', '--max-duration', type='float', dest='max_onoff_segment_duration', default=0.,
                  help="Maximum duration of segments around jump used to estimate instant, in seconds (default 1 dump)")
parser.add_option('-m', '--margin', type='float', dest='margin_factor', default=24.,
                  help="Allowed variation in power, as multiple of theoretical standard deviation (default %default)")
parser.add_option('-s', '--significance', type='float', dest='jump_significance', default=10.,
                  help="Keep jumps that are bigger than margin by this factor (default %default)")
parser.add_option("-c", "--channel-mask", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle',
                  help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")

(opts, args) = parser.parse_args()
if len(args) < 1:
    raise RunTimeError('Please specify an HDF5 file to check')

data = katdal.open(args[0])

n_chan = np.shape(data.channels)[0]
if not opts.freq_chans is None :
    start_freq_channel = int(opts.freq_chans.split(',')[0])
    end_freq_channel = int(opts.freq_chans.split(',')[1])
    edge = np.tile(True, n_chan)
    edge[slice(start_freq_channel, end_freq_channel)] = False
else :
    edge = np.tile(True, n_chan)
    edge[slice(data.shape[1] // 4, 3 * data.shape[1] // 4)] = False
#load static flags if pickle file is given
channel_mask = opts.channel_mask
if len(channel_mask)>0:
    pickle_file = open(channel_mask)
    rfi_static_flags = pickle.load(pickle_file)
    pickle_file.close()
    if n_chan > rfi_static_flags.shape[0] :
        rfi_static_flags = rfi_static_flags.repeat(8) # 32k mode
else:
    rfi_static_flags = np.tile(False, n_chan)
static_flags = np.logical_or(edge,rfi_static_flags)
data.select(channels=~static_flags)

# Number of real normal variables squared and added together
dof = 2 * data.shape[1] * data.channel_width * data.dump_period
corrprod_to_index = dict([(tuple(cp), ind) for cp, ind in zip(data.corr_products, range(len(data.corr_products)))])

offset_stats = {}
for ant in data.ants:
    print 'Individual firings: timestamp | offset +/- uncertainty (magnitude of jump)'
    print '--------------------------------------------------------------------------'
    hh_index = corrprod_to_index.get((ant.name + 'h', ant.name + 'h'))
    vv_index = corrprod_to_index.get((ant.name + 'v', ant.name + 'v'))
    for diode_name in ('pin', 'coupler'):
        # Ignore missing sensors or sensors with one entry (which serves as an initial value instead of real event)
        try:
            sensor = data.sensor.get('Antennas/%s/nd_%s' % (ant.name, diode_name), extract=False)
        except KeyError:
            continue
        if len(sensor[:]) <= 1:
            continue
        # Collect all expected noise diode firings
        print "Diode:", ant.name, diode_name
        print "Timestamp (UTC)     | offset in (ms)  +/- error ms (magnitude of jump )"
        nd_timestamps = sensor['timestamp']
        nd_state = np.array(sensor['value'], dtype=np.int)
        for scan_index, state, target in data.scans():
            # Extract averaged power data time series and DBE timestamps (at start of each dump)
            dbe_timestamps = data.timestamps[:] - 0.5 * data.dump_period
            if len(dbe_timestamps) < 3:
                continue
            power = data.vis[:, :, hh_index].real.mean(axis=1) if hh_index is not None \
                    else np.zeros(data.shape[0], dtype=np.float32)
            power += data.vis[:, :, vv_index].real.mean(axis=1) if vv_index is not None \
                     else np.zeros(data.shape[0], dtype=np.float32)
            # Since I = HH + VV and not the average of HH and VV, the dof actually halves instead of doubling
            power_dof = dof / 2 if (hh_index is not None and vv_index is not None) else dof
            jump_time, jump_std_time, jump_size = find_jumps(dbe_timestamps, power, power * np.sqrt(2. / power_dof),
                                                             **vars(opts))
            # Focus on noise diode events within this scan (and not on the edges of scan either)
            firings_in_scan = (nd_timestamps > dbe_timestamps[1]) & (nd_timestamps < dbe_timestamps[-1])
            for n, firing in enumerate(nd_timestamps[firings_in_scan]):
                # Obtain closest time offset between expected firing and power jump
                offsets = np.array(jump_time) - firing
                # Ensure that jump is in the expected direction (up or down)
                same_direction = (2 * nd_state[firings_in_scan][n] - 1) * np.sign(jump_size) > 0
                if same_direction.any():
                    same_direction = np.where(same_direction)[0]
                    closest_jump = same_direction[np.argmin(np.abs(offsets[same_direction]))]
                    offset = offsets[closest_jump]
                    # Only match the jump if it is within a certain window of the expected firing
                    if np.abs(offset) < data.dump_period*opts.max_offset:
                        std_offset, jump = jump_std_time[closest_jump], jump_size[closest_jump]
                        stats_key = ant.name + ' ' + diode_name
                        # For each diode, collect the offsets and their uncertainties
                        stats = offset_stats.get(stats_key, [])
                        offset_stats[stats_key] = stats + [(offset, std_offset)]
                        print '%s | offset %8.2f +/- %5.2f ms (magnitude of %+.0f margins)' % \
                              (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(firing)),
                               1000 * offset, 1000 * std_offset, jump)
                    else:
                        num = data.dumps[0] + np.argmin(np.abs(data.timestamps-firing)) 
                        print '%s | not found at location %i' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(firing)),num,)
if offset_stats :
    print
    print 'Summary of offsets (DBE - CAM) per diode'
    print '----------------------------------------'
    for key, val in offset_stats.iteritems():
        # Change unit to milliseconds, and from an array from list
        offset_ms, std_offset_ms = 1000 * np.asarray(val).T
        mean_offset = offset_ms.mean()
        # Variation of final mean offset due to uncertainty of each measurement (influenced by integration time,
        # bandwidth, magnitude of power jump)
        std1 = np.sqrt(np.sum(std_offset_ms ** 2)) / len(std_offset_ms)
        # Variation of final mean due to offsets in individual measurements - this can be much bigger than std1 and
        # is typically due to changes in background power while noise diode is firing, resulting in measurement bias
        std2 = offset_ms.std() / np.sqrt(len(offset_ms))
        std_mean_offset = np.sqrt(std1 ** 2 + std2 ** 2)
        min_offset, max_offset = np.argmin(offset_ms), np.argmax(offset_ms)
        print '%s diode: mean %.2f +/- %.2f ms [%.3f +/- %.3f dumps], min %.2f +/- %.2f ms, max %.2f +/- %.2f ms' % \
              (key, mean_offset, std_mean_offset,
               mean_offset / data.dump_period / 1e3, std_mean_offset / data.dump_period / 1e3,
               offset_ms[min_offset], std_offset_ms[min_offset], offset_ms[max_offset], std_offset_ms[max_offset])
else:
    print ("No valid noisediode values found in file")