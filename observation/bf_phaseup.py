#!/usr/bin/env python
#
# Track calibrator target for a specified time.
# Obtain calibrated gains and apply them to the F-engine afterwards.

import numpy as np
import scipy.ndimage
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                SessionSDP)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


def clean_bandpass(bp_gains, cal_channel_freqs, max_gap_Hz):
    """Clean up bandpass gains by linear interpolation across narrow flagged regions."""
    clean_gains = {}
    # Linearly interpolate across flagged regions as long as they are not too large
    for inp, bp in bp_gains.items():
        flagged = np.isnan(bp)
        if flagged.all():
            clean_gains[inp] = bp
            continue
        chans = np.arange(len(bp))
        interp_bp = np.interp(chans, chans[~flagged], bp[~flagged])
        # Identify flagged regions and tag each with unique integer label
        gaps, n_gaps = scipy.ndimage.label(flagged)
        for n in range(n_gaps):
            gap = np.nonzero(gaps == n + 1)[0]
            gap_freqs = cal_channel_freqs[gap]
            lower = gap_freqs.min()
            upper = gap_freqs.max()
            if upper - lower > max_gap_Hz:
                interp_bp[gap] = np.nan
        clean_gains[inp] = interp_bp
    return clean_gains


def calculate_corrections(G_gains, B_gains, delays, cal_channel_freqs,
                          random_phase, flatten_bandpass,
                          target_average_correction):
    """Turn cal pipeline products into corrections to be passed to F-engine."""
    average_gain = {}
    gain_corrections = {}
    # First find relative corrections per input with arbitrary global average
    for inp in G_gains:
        # Combine all calibration products for input into single array of gains
        K_gains = np.exp(-2j * np.pi * delays[inp] * cal_channel_freqs)
        gains = K_gains * B_gains[inp] * G_gains[inp]
        if np.isnan(gains).all():
            average_gain[inp] = gain_corrections[inp] = 0.0
            continue
        abs_gains = np.abs(gains)
        # Track the average gain to fix overall power level (and as diagnostic)
        average_gain[inp] = np.nanmedian(abs_gains)
        corrections = 1.0 / gains
        if not flatten_bandpass:
            # Let corrections have constant magnitude equal to 1 / (avg gain),
            # which ensures that power levels are still equalised between inputs
            corrections *= abs_gains / average_gain[inp]
        if random_phase:
            corrections *= np.exp(2j * np.pi * np.random.rand(len(corrections)))
        gain_corrections[inp] = np.nan_to_num(corrections)
    # All invalid gains (NaNs) have now been turned into zeros
    valid_average_gains = [g for g in average_gain.values() if g > 0]
    if not valid_average_gains:
        raise ValueError("All gains invalid and beamformer output will be zero!")
    global_average_gain = np.median(valid_average_gains)

    # Iterate over inputs again and fix average values of corrections
    for inp in sorted(G_gains):
        relative_gain = average_gain[inp] / global_average_gain
        if relative_gain == 0.0:
            user_logger.warning("%s has no valid gains and will be zeroed", inp)
            continue
        # This ensures that input at the global average gets target correction
        gain_corrections[inp] *= target_average_correction * global_average_gain
        safe_relative_gain = np.clip(relative_gain, 0.5, 2.0)
        if relative_gain == safe_relative_gain:
            user_logger.info("%s: average gain relative to global average = %5.2f",
                             inp, relative_gain)
        else:
            user_logger.warning("%s: average gain relative to global average "
                                "= %5.2f out of range, clipped to %.1f",
                                inp, relative_gain, safe_relative_gain)
            gain_corrections[inp] *= relative_gain / safe_relative_gain
    return gain_corrections


# Default F-engine gain as a function of number of channels
DEFAULT_GAIN = {1024: 116, 4096: 70, 32768: 360}


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track the source with the largest flux density which is above ' \
              'the horizon and calibrate gains based on it. At least one ' \
              'target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=64.0,
                  help='Length of time to track the source for calibration, '
                       'in seconds (default=%default)')
parser.add_option('--verify-duration', type='float', default=64.0,
                  help='Length of time to revisit the source for verification, '
                       'in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int', default=0,
                  help='Override correlator F-engine gain (average magnitude), '
                       'using the default gain value for the mode if 0')
parser.add_option('--flatten-bandpass', action='store_true', default=False,
                  help='Applies magnitude bandpass correction in addition to phase correction')
parser.add_option('--random-phase', action='store_true', default=False,
                  help='Applies random phases in F-engines')
parser.add_option('--fft-shift', type='int',
                  help='Set correlator F-engine FFT shift (default=leave as is)')
parser.add_option('--reconfigure-sdp', action="store_true", default=False,
                  help='Reconfigure SDP subsystem at the start to clear crashed containers')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Phase-up observation that sets the F-engine weights')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('J1939-6342'), description ('radec, 19:39, -63:42') or "
                     "catalogue file name ('three_calib.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    if opts.reconfigure_sdp:
        user_logger.info("Reconfiguring SDP subsystem")
        sdp = SessionSDP(kat)
        sdp.req.product_reconfigure()
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        if opts.fft_shift is not None:
            session.cbf.fengine.req.fft_shift(opts.fft_shift)
        if opts.fengine_gain <= 0:
            num_channels = session.cbf.fengine.sensor.n_chans.get_value()
            try:
                opts.fengine_gain = DEFAULT_GAIN[num_channels]
            except KeyError:
                raise KeyError("No default gain available for F-engine with "
                               "%i channels - please specify --fengine-gain"
                               % (num_channels,))
        user_logger.info("Resetting F-engine gains to %g to allow phasing up",
                         opts.fengine_gain)
        gains = {inp: opts.fengine_gain for inp in session.cbf.fengine.inputs}
        session.set_fengine_gains(gains)
        # Quit early if there are no sources to observe or not enough antennas
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        observation_sources = collect_targets(kat, args)
        sources_above_horizon = observation_sources.filter(el_limit_deg=opts.horizon)
        if not sources_above_horizon:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Pick the first source that is up (this assumes that the sources in
        # the catalogue are ordered from highest to lowest priority)
        target = sources_above_horizon.targets[0]
        target.add_tags('bfcal single_accumulation')
        user_logger.info("Target to be observed: %s", target.description)
        session.capture_init()
        session.cbf.correlator.req.capture_start()
        session.label('un_corrected')
        user_logger.info("Initiating %g-second track on target '%s'",
                         opts.track_duration, target.name)
        session.track(target, duration=opts.track_duration, announce=False)
        # Attempt to jiggle cal pipeline to drop its gains
        session.stop_antennas()
        user_logger.info("Waiting for gains to materialise in cal pipeline")
        # Wait for the last bfcal product from the pipeline
        gains = session.get_cal_solutions('G', timeout=opts.track_duration)
        bp_gains = session.get_cal_solutions('B')
        delays = session.get_cal_solutions('K')
        cal_channel_freqs = session.get_cal_channel_freqs()
        bp_gains = clean_bandpass(bp_gains, cal_channel_freqs, max_gap_Hz=64e6)

        if opts.random_phase:
            user_logger.info("Setting F-engine gains with random phases")
        else:
            user_logger.info("Setting F-engine gains to phase up antennas")
        if not kat.dry_run:
            corrections = calculate_corrections(gains, bp_gains, delays,
                                                cal_channel_freqs, opts.random_phase,
                                                opts.flatten_bandpass, opts.fengine_gain)
            session.set_fengine_gains(corrections)
        if opts.verify_duration > 0:
            user_logger.info("Revisiting target %r for %g seconds to verify phase-up",
                             target.name, opts.verify_duration)
            session.label('corrected')
            session.track(target, duration=opts.verify_duration, announce=False)

        if not opts.random_phase:
            # Set last-phaseup script sensor on the subarray.
            session.sub.req.set_script_param('script-last-phaseup', kat.sb_id_code)
