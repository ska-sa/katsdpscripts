#!/usr/bin/env python
#
# Observe either 1934-638, 0408-65 or 3C286 to establish some basic health
# properties of the MeerKAT AR1 system.

import time

import numpy as np
import katpoint
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                start_session, user_logger)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Default F-engine gain as a function of number of channels
DEFAULT_GAIN = {1024: 116, 4096: 70, 32768: 360}


# Set up standard script options
usage = "%prog"
description = 'Observe either 1934-638, 0408-65 or 3C286 to establish some ' \
              'basic health properties of the MeerKAT system.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('--verify-duration', type='float', default=64.0,
                  help='Length of time to revisit source for verification, '
                       'in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int', default=0,
                  help='Correlator F-engine gain (average magnitude), '
                       'automatically determined if 0 (the default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='basic_health', nd_params='off',
                    project_id='MKAIV-308', reduction_label='MKAIV-308',
                    description='Basic health test of the system.',
                    horizon=25, track_duration=64)
# Parse the command line
opts, args = parser.parse_args()

# Set of targets with flux models
J1934 = 'PKS1934-638, radec, 19:39:25.03, -63:42:45.7, (200.0 10000.0 -30.7667 26.4908 -7.0977 0.605334)'
J0408 = 'J0408-6545, radec, 04:08:20.3788, -65:45:09.08, (300.0 50000.0 0.4288422 1.9395659 -0.66243187 0.03926736)'
J1331 = '3C286, radec, 13:31:08.29, +30:30:33.0, (300.0 50000.0 0.1823 1.4757 -0.4739 0.0336)'

# ND states
nd_off = {'diode': 'coupler', 'on': 0., 'off': 0., 'period': -1.}
nd_on = {'diode': 'coupler', 'on': opts.track_duration, 'off': 0., 'period': 0.}

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    observation_sources.add(J1934)
    observation_sources.add(J0408)
    observation_sources.add(J1331)
    user_logger.info(observation_sources.visibility_list())
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe or not enough antennas
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        # Pick source with the highest elevation as our target
        target = observation_sources.sort('el').targets[-1]
        target.add_tags('bfcal single_accumulation')
        # Calibration tests
        user_logger.info("Performing calibration tests")
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
        session.capture_init()
        session.cbf.correlator.req.capture_start()
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
                new_weights[inp] = opts.fengine_gain * phase_weights.conj()
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
                    session.ants.req.offset_fixed(offset_target[0], offset_target[1], opts.projection)
                nd_params = session.nd_params
                session.fire_noise_diode(announce=True, **nd_params)
                if kat.dry_run:
                    session.track(target, duration=opts.track_duration, announce=False)
                else:
                    time.sleep(opts.track_duration)  # Snooze
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
        session.raster_scan(target, num_scans=5, scan_duration=80, scan_extent=6.0,
                            scan_spacing=0.25, scan_in_azimuth=True,
                            projection=opts.projection)
        # reset the gains always
        user_logger.info("Resetting F-engine gains to %g", opts.fengine_gain)
        gains = {inp: opts.fengine_gain for inp in gains}
        session.set_fengine_gains(gains)
