#!/usr/bin/env python
#
# Observe a bandpass calibrator to establish some basic health
# properties of the MeerKAT telescope.

import numpy as np
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Observe a bandpass calibrator to establish some ' \
              'basic health properties of the MeerKAT system.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('--verify-duration', type='float', default=64.0,
                  help='Length of time to revisit source for verification, '
                       'in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int_or_default', default='default',
                  help='Set correlator F-engine gain (average magnitude)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='basic_health', nd_params='off',
                    project_id='MKAIV-308', reduction_label='MKAIV-308',
                    description='Basic health test of the system.',
                    horizon=25, track_duration=64)
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('J1939-6342'), description ('radec, 19:39, -63:42') or "
                     "catalogue file name ('three_calib.csv')")

# ND states
nd_off = {'diode': 'coupler', 'on': 0., 'off': 0., 'period': -1.}
nd_on = {'diode': 'coupler', 'on': opts.track_duration, 'off': 0., 'period': 0.}

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    user_logger.info(observation_sources.visibility_list())
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        # Reset F-engine to a known good state first
        fengine_gain = session.set_fengine_gains(opts.fengine_gain)
        # Quit if there are no sources to observe or not enough antennas for cal
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        sources_above_horizon = observation_sources.filter(el_limit_deg=opts.horizon)
        if not sources_above_horizon:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Pick the first source that is up (this assumes that the sources in
        # the catalogue are ordered from highest to lowest priority)
        target = sources_above_horizon.targets[0]
        target.add_tags('bfcal single_accumulation')
        session.capture_start()
        # Calibration tests
        user_logger.info("Performing calibration tests")
        session.label('calibration')
        user_logger.info("Initiating %g-second track on target '%s'",
                         opts.track_duration, target.name)
        session.track(target, duration=opts.track_duration, announce=False)
        # Attempt to jiggle cal pipeline to drop its gains
        session.stop_antennas()
        user_logger.info("Waiting for gains to materialise in cal pipeline")
        # Wait for the last bfcal product from the pipeline
        gains = session.get_cal_solutions('G', timeout=300.)
        bp_gains = session.get_cal_solutions('B')
        delays = session.get_cal_solutions('K')
        cal_channel_freqs = session.get_cal_channel_freqs()
        user_logger.info("Setting F-engine gains to phase up antennas")
        new_weights = {}
        for inp in gains:
            orig_weights = gains[inp]
            bp = bp_gains[inp]
            valid = ~np.isnan(bp)
            if valid.any():  # not all flagged
                chans = np.arange(len(bp))
                bp = np.interp(chans, chans[valid], bp[valid])
                orig_weights *= bp
                delay_weights = np.exp(-2j * np.pi * delays[inp] * cal_channel_freqs)
                orig_weights *= delay_weights  # unwrap the delays
                amp_weights = np.abs(orig_weights)
                phase_weights = orig_weights / amp_weights
                new_weights[inp] = fengine_gain * phase_weights.conj()
        session.set_fengine_gains(new_weights)
        if opts.verify_duration > 0:
            user_logger.info("Revisiting target %r for %g seconds to verify phase-up",
                             target.name, opts.verify_duration)
            session.track(target, duration=opts.verify_duration, announce=False)

        # interferometric pointing
        user_logger.info("Performing interferometric pointing tests")
        session.label('interferometric_pointing')
        session.track(target, duration=opts.track_duration, announce=False)
        for direction in {'x', 'y'}:
            for offset in np.linspace(-1, 1, 10 // 2):
                if direction == 'x':
                    offset_target = [offset, 0.0]
                else:
                    offset_target = [0.0, offset]
                user_logger.info("Initiating %g-second track on target '%s'",
                                 opts.track_duration, target.name)
                user_logger.info("Offset of (%f, %f) degrees", *offset_target)
                session.set_target(target)
                if not kat.dry_run:
                    session.ants.req.offset_fixed(offset_target[0], offset_target[1],
                                                  opts.projection)
                nd_params = session.nd_params
                session.track(target, duration=opts.track_duration, announce=False)
        session.ants.req.offset_fixed(0, 0, opts.projection)  # reset any dangling offsets
        # Tsys and averaging
        user_logger.info("Performing Tsys and averaging tests")
        session.nd_params = nd_off
        # 10 second track so that the antenna profiler does not run away
        session.track(target, duration=10)
        user_logger.info("Now capturing data - diode %s on", nd_on['diode'])
        session.label('%s' % (nd_on['diode'],))
        if not session.fire_noise_diode(announce=True, **nd_on):
            user_logger.error("Noise diode %s did not fire", nd_on['diode'])
        session.nd_params = nd_off
        user_logger.info("Now capturing data - noise diode off")
        session.track(target, duration=320)  # get 5 mins of data to test averaging

        # Single dish pointing ... to compare with interferometric
        user_logger.info("Performing single dish pointing tests")
        session.label('raster')
        user_logger.info("Doing scan of '%s' with current azel (%s, %s)",
                         target.description, *target.azel())
        # Do different raster scan on strong and weak targets
        session.raster_scan(target, num_scans=5, scan_duration=80,
                            scan_extent=6.0, scan_spacing=0.25,
                            scan_in_azimuth=True, projection=opts.projection)
        # reset the gains always
        session.set_fengine_gains(opts.fengine_gain)
