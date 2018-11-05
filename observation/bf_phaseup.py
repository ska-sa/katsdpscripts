#!/usr/bin/env python
#
# Track calibrator target for a specified time.
# Obtain calibrated gains and apply them to the F-engine afterwards.

import numpy as np
import katpoint
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                SessionSDP)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Default F-engine gain as a function of number of channels
DEFAULT_GAIN = {4096: 200, 32768: 4000}


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track the source with the highest elevation and calibrate ' \
              'gains based on it. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=64.0,
                  help='Length of time to track the source for calibration, '
                       'in seconds (default=%default)')
parser.add_option('--verify-duration', type='float', default=64.0,
                  help='Length of time to revisit the source for verification, '
                       'in seconds (default=%default)')
parser.add_option('--reset', action='store_true', default=False,
                  help='Reset the gains to the default value then exit')
parser.add_option('--default-gain', type='int', default=0,
                  help='Default correlator F-engine gain, '
                       'automatically set if 0 (default=%default)')
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

# Set of targets with flux models
J1934 = 'PKS1934-638, radec, 19:39:25.03, -63:42:45.7, (200.0 10000.0 -30.7667 26.4908 -7.0977 0.605334)'
J0408 = 'J0408-6545, radec, 04:08:20.3788, -65:45:09.08, (300.0 50000.0 0.4288422 1.9395659 -0.66243187 0.03926736)'
J1331 = '3C286, radec, 13:31:08.29, +30:30:33.0, (300.0 50000.0 0.1823 1.4757 -0.4739 0.0336)'

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    if len(args) == 0:
        observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
        observation_sources.add(J1934)
        observation_sources.add(J0408)
        observation_sources.add(J1331)
    else:
        observation_sources = collect_targets(kat, args)
    if opts.reconfigure_sdp:
        user_logger.info("Reconfiguring SDP subsystem")
        sdp = SessionSDP(kat)
        sdp.req.product_reconfigure()
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
        session.capture_init()
        if opts.fft_shift is not None:
            session.cbf.fengine.req.fft_shift(opts.fft_shift)
        session.cbf.correlator.req.capture_start()
        gains = {}
        # Pick source with the highest elevation as our target
        target = observation_sources.sort('el').targets[-1]
        target.add_tags('bfcal single_accumulation')
        if not opts.default_gain:
            channels = 32768 if session.product.endswith('32k') else 4096
            opts.default_gain = DEFAULT_GAIN[channels]
        user_logger.info("Target to be observed: %s", target.description)
        user_logger.info("Resetting F-engine gains to %g to allow phasing up",
                         opts.default_gain)
        for inp in session.cbf.fengine.inputs:
            gains[inp] = opts.default_gain
        session.set_fengine_gains(gains)
        if not opts.reset:
            
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

            if opts.random_phase:
                user_logger.info("Setting F-engine gains with random phases")
            else:
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
                    if opts.random_phase:
                        phase_weights *= np.exp(2j * np.pi * np.random.random_sample(size=len(bp)))
                    new_weights[inp] = opts.default_gain * phase_weights.conj()
                    if opts.flatten_bandpass:
                        new_weights[inp] /= amp_weights
            session.set_fengine_gains(new_weights)
            if opts.verify_duration > 0:
                user_logger.info("Revisiting target %r for %g seconds to verify phase-up",
                                 target.name, opts.verify_duration)
                session.label('corrected')
                session.track(target, duration=opts.verify_duration, announce=False)
