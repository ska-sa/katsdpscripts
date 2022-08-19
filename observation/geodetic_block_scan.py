#!/usr/bin/env python
#
# Track many (e.g. 10-30) well-spaced targets in a short time (e.g. 30 minutes),
# which is useful for delay/baseline calibration because it ensures reasonably
# constant atmospheric effects which may be fitted and removed from the delays.
#
# This is known as a "geodetic block" observation in VLBI, hence the name.
#
# Ludwig Schwardt
# 18 August 2022
#
# [1] M.J. Reid and M. Honma, "Microarcsecond Radio Astrometry,"
#     Annual Review of Astronomy and Astrophysics, vol. 52:1, pp. 339--372, 2014.
#

import time
import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)
from katpoint import rad2deg, wrap_angle


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


def target_coordinates(targets, timestamps, horizon, min_az=-185, max_az=275):
    """Get azimuth and elevation for all `targets` at `timestamps`."""
    az = []
    el = []
    for target in targets:
        a, e = target.azel(timestamps)
        az.append(a)
        el.append(e)
    az = rad2deg(np.array(az))
    el = rad2deg(np.array(el))

    # Identify sources above horizon
    az[el < horizon] = np.nan
    el[el < horizon] = np.nan
    visible_targets = np.nonzero((el >= horizon).any(axis=-1))[0]
    az = az[visible_targets]
    el = el[visible_targets]

    az[az > max_az] -= 360  # az range now -85 -> 275
    # Duplicate sources in azimuth wrap region (175 -> 275)
    # to have both azimuth versions
    wrap_region = az >= min_az + 360
    targets_in_wrap = wrap_region.sum(axis=-1) > 0
    wrapped_az = az[targets_in_wrap]
    wrapped_az[wrap_region[targets_in_wrap]] -= 360
    add_wrap = np.r_[
        np.arange(len(visible_targets), dtype=int),
        np.nonzero(targets_in_wrap)[0]
    ]
    az = np.r_[az, wrapped_az]
    el = el[add_wrap]
    target_labels = visible_targets[add_wrap]
    return az, el, target_labels


def optimal_bitonic_tour(edge_lengths):
    """The optimal bitonic tour on distance matrix `edge_lengths`.

    https://metacpan.org/pod/Algorithm::TravelingSalesman::BitonicTour
    """
    n_points = edge_lengths.shape[0]
    # Dynamic programming data structures:
    # - A cache of the lengths of the optimal open bitonic tours, shape (N, N)
    # - Pointers remembering the N-2 choices (i.e. argmins) made, shape (N,)
    open_tour_lengths = np.zeros_like(edge_lengths)
    back_pointers = np.zeros(n_points, dtype=int)
    # Initialise the base case
    open_tour_lengths[0, 1] = edge_lengths[0, 1]
    # Extend optimal open tours from left to right
    for j in range(2, n_points):
        # Recurrence relation #1: straightforward extension of longer leg
        i = slice(0, j - 1)
        open_tour_lengths[i, j] = open_tour_lengths[i, j - 1] + edge_lengths[j - 1, j]
        # Recurrence relation #2: shorter leg overtakes longer leg (remember from where)
        i = j - 1
        k = slice(0, i)
        candidates = open_tour_lengths[k, i] + edge_lengths[k, j]
        best_k = candidates.argmin()
        open_tour_lengths[i, j] = candidates[best_k]
        back_pointers[i] = best_k
    # Close the tour by connecting the shorter leg to endpoint (remember from where)
    rightmost = n_points - 1
    k = slice(0, rightmost)
    candidates = open_tour_lengths[k, rightmost] + edge_lengths[k, rightmost]
    best_k = candidates.argmin()
    score = candidates[best_k]
    back_pointers[rightmost] = best_k

    # Backtrack to obtain sequence of partial open tours that are locally optimal
    open_tour_sequence = [(rightmost, rightmost)]
    i = rightmost
    for j in reversed(range(n_points)):
        assert j >= i, 'whoops'
        # When longer leg shrinks back to shorter leg, shorter leg hops back again
        if j == i:
            i = back_pointers[i]
        # The new partial tour has at least i or j in common with
        # the previous partial tour, so line up those two indices.
        last_open_tour = open_tour_sequence[-1]
        flipped = (last_open_tour[0] - i) * (last_open_tour[1] - j)
        open_tour_sequence.append((j, i) if flipped else (i, j))
    # Turn open tours into a closed tour starting on the left, running up j and down i
    open_tour_sequence = np.array(open_tour_sequence)
    closed_tour = np.r_[open_tour_sequence[::-1, 1], open_tour_sequence[:, 0]]
    # Get rid of duplicate points
    last_of_duplicates = closed_tour[:-1] != closed_tour[1:]
    optimal_tour = closed_tour[:-1][last_of_duplicates]
    return optimal_tour, score


def optimal_target_order(az, el, az_deg_per_sec=2.0, el_deg_per_sec=1.0):
    # Operate on the midpoint of each target trajectory to simplify matters
    az_mid = np.nanmedian(az, axis=-1)
    el_mid = np.nanmedian(el, axis=-1)
    # Sort the azimuth / elongated coordinate from left to right
    left_to_right = np.argsort(az_mid)
    az_mid = az_mid[left_to_right]
    el_mid = el_mid[left_to_right]
    # The distance between sources is the time it takes to slew between them
    az_diff = np.abs(az_mid[:, np.newaxis] - az_mid[np.newaxis, :])
    el_diff = np.abs(el_mid[:, np.newaxis] - el_mid[np.newaxis, :])
    slew_distance = np.maximum(az_diff / az_deg_per_sec, el_diff / el_deg_per_sec)
    # Find sequence of targets that minimises slew time
    tour, score = optimal_bitonic_tour(slew_distance)
    # Undo the sorting afterwards instead of sorting everything else
    tour = left_to_right[tour]
    return tour


def simulator(schedule, timestamps, target_labels, target_az, target_el,
              track_duration, initial_az, initial_el, min_az=-185, max_az=275):
    """Very basic antenna simulator that tracks a list of targets."""
    slots_left = list(schedule)
    targets_done = []
    array_az = np.full_like(timestamps, np.nan)
    array_el = np.full_like(timestamps, np.nan)
    tracking = np.zeros_like(timestamps, dtype=bool)
    time_steps = np.diff(timestamps)
    current_slot = slots_left.pop(0)
    array_az[0] = initial_az
    array_el[0] = initial_el
    tracking[0] = False
    track_accum = 0.0

    def next_slot():
        if not slots_left:
            return None
        slot = slots_left.pop(0)
        while target_labels[slot] in targets_done:
            if not slots_left:
                return None
            slot = slots_left.pop(0)
        return slot

    for n, time_step in enumerate(time_steps):
        if tracking[n]:
            if track_accum < track_duration:
                track_accum += time_step
            else:
                targets_done.append(target_labels[current_slot])
                current_slot = next_slot()
                if current_slot is None:
                    break
                tracking[n + 1] = False
                track_accum = 0.0

        requested_az = target_az[current_slot, n + 1]
        requested_el = target_el[current_slot, n + 1]
        delta_az = wrap_angle(requested_az - array_az[n], period=360)
        delta_el = requested_el - array_el[n]
        az_step = np.clip(delta_az, -time_step * 2, time_step * 2)
        el_step = np.clip(delta_el, -time_step, time_step)
        proposed_az = array_az[n] + az_step
        if min_az <= proposed_az <= max_az:
            array_az[n + 1] = proposed_az
            array_el[n + 1] = array_el[n] + el_step
            if az_step == delta_az and el_step == delta_el:
                tracking[n + 1] = True
        else:
            array_az[n + 1] = array_az[n]
            array_el[n + 1] = array_el[n]
            current_slot = next_slot()
            if current_slot is None:
                break
            tracking[n + 1] = False
            track_accum = 0.0

    return array_az, array_el, tracking, targets_done


def evaluate_tour_start(tour, timestamps, target_labels, az, el,
                        track_duration, initial_az, initial_el):
    target_lists = []
    num_targets = np.zeros_like(tour)
    lowness = np.full_like(tour, 90.0, dtype=float)
    for starting_slot in range(len(tour)):
        schedule = np.roll(tour, -starting_slot)
        array_az, array_el, tracking, targets_done = simulator(
            schedule, timestamps, target_labels, az, el,
            track_duration, initial_az, initial_el
        )
        target_lists.append(targets_done)
        num_targets[starting_slot] = len(targets_done)
        if np.isnan(array_el[0]):
            continue
        lowness[starting_slot] = np.percentile(array_el[tracking], 25)
    most_targets = np.nonzero(num_targets == num_targets.max())[0]
    best = most_targets[lowness[most_targets].argmin()]
    return np.roll(tour, -best), target_lists[best], lowness[best]


def optimal_target_sequence(timestamps, target_labels, az, el,
                            track_duration, initial_az, initial_el):
    # First get the most efficient target order as a bitonic tour
    tour = optimal_target_order(az, el)
    # Pick the best starting point and direction to go around the tour
    tour1, targets1, lowness1 = evaluate_tour_start(
        tour, timestamps, target_labels, az, el,
        track_duration, initial_az, initial_el
    )
    tour2, targets2, lowness2 = evaluate_tour_start(
        tour[::-1], timestamps, target_labels, az, el,
        track_duration, initial_az, initial_el
    )
    if len(targets2) > len(targets1):
        schedule = tour2
    elif len(targets2) == len(targets1):
        schedule = tour2 if lowness2 <= lowness1 else tour1
    else:
        schedule = tour1
    # Re-evaluate the best target sequence
    array_az, array_el, tracking, targets_done = simulator(
        schedule, timestamps, target_labels, az, el,
        track_duration, initial_az, initial_el
    )
    obs_steps = len(np.isfinite(array_el))
    track_steps = tracking.sum()
    slew_steps = obs_steps - track_steps
    scan_time_step = timestamps[1] - timestamps[0]
    user_logger.info('Bitonic scheduler results:')
    user_logger.info('Visit %d of %d visible targets',
                     len(targets_done), len(set(target_labels)))
    user_logger.info('Lowest elevation: %.1f degrees', np.nanmin(array_el))
    user_logger.info('Track for %.1f minutes',
                     track_steps * scan_time_step / 60.0)
    user_logger.info('Slew for %.1f minutes',
                     slew_steps * scan_time_step / 60.0)
    user_logger.info('Total observation time: %.1f minutes',
                     obs_steps * scan_time_step / 60.0)
    user_logger.info('Efficiency: %.1f%%', 100.0 * track_steps / obs_steps)
    return targets_done


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Perform geodetic block scan on given catalogue. At least one ' \
              'target must be specified. Note also some **required** options below.'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=32.0,
                  help='Length of time to track each source, in seconds '
                       '(default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=2000.0,
                  help='Maximum duration of the script in seconds, after which '
                       'script will end as soon as the current track finishes '
                       '(no limit by default)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Geodetic block scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(targets.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        session.capture_start()

        # Figure out where most of the antennas are pointing at the start
        initial_az = np.median([ant.sensor.pos_actual_scan_azim.get_value()
                                for ant in session.ants])
        initial_el = np.median([ant.sensor.pos_actual_scan_elev.get_value()
                                for ant in session.ants])
        user_logger.info('Initial az: %.1f el: %.1f', initial_az, initial_el)

        # Optimise the target schedule
        start_time = time.time()
        timestamps = start_time + np.arange(0, opts.max_duration, 2.0)
        az, el, target_labels = target_coordinates(
            targets, timestamps, opts.horizon
        )
        schedule = optimal_target_sequence(
            timestamps, target_labels, az, el,
            opts.track_duration, initial_az, initial_el
        )

        for target_index in schedule:
            target = targets.targets[target_index]
            session.label('track')
            session.track(target, duration=opts.track_duration)
