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
description = 'Track one or more sources for a specified time and calibrate ' \
              'gains based on them. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=32.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('--reset', action='store_true', default=False,
                  help='Reset the gains to the default value afterwards')
parser.add_option('--default-gain', type='int', default=0,
                  help='Default correlator F-engine gain, '
                       'automatically set if 0 (default=%default)')
parser.add_option('--flatten-bandpass', action='store_true', default=False,
                  help='Applies magnitude bandpass correction in addition to phase correction')
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
J1934 = 'PKS 1934-63 | J1939-6342, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231)'
J0408 = 'PKS 0408-65 | J0408-6545, radec, 4:08:20.38, -65:45:09.1, (800.0 8400.0 -3.708 3.807 -0.7202)'
J1331 = '3C286      | J1331+3030, radec, 13:31:08.29, +30:30:33.0,(800.0 43200.0 0.956 0.584 -0.1644)'


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
        sdp.req.data_product_reconfigure()
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        if opts.fft_shift is not None:
            session.cbf.fengine.req.fft_shift(opts.fft_shift)
        session.cbf.correlator.req.capture_start()

        for target in [observation_sources.sort('el').targets[-1]]:
            target.add_tags('bfcal single_accumulation')
            if not opts.default_gain:
                channels = 32768 if session.product.endswith('32k') else 4096
                opts.default_gain = DEFAULT_GAIN[channels]
            user_logger.info("Target to be observed: %s", target.description)
            if target.flux_model is None:
                user_logger.warning("Target has no flux model (katsdpcal will need it in future)")
            user_logger.info("Resetting F-engine gains to %g to allow phasing up",
                             opts.default_gain)
            for inp in session.cbf.fengine.inputs:
                session.cbf.fengine.req.gain(inp, opts.default_gain)
            session.label('un_corrected')
            user_logger.info("Initiating %g-second track on target '%s'",
                             opts.track_duration, target.name)
            session.track(target, duration=opts.track_duration, announce=False)
            # Attempt to jiggle cal pipeline to drop its gains
            session.ants.req.target('')
            session.label('corrected')
            for inp in set(session.cbf.fengine.inputs):
                # Correct the phase and optionally the amplitude as well
                
                phase_weights = (2*np.pi) * np.random.random_sample(size=channels) 
                new_weights = opts.default_gain * phase_weights.conj()
                weights_str = [('%+5.3f%+5.3fj' % (w.real, w.imag)) for w in new_weights]
                session.cbf.fengine.req.gain(inp, *weights_str)
            user_logger.info("Revisiting target %r for %g seconds to see if phasing worked",
                             target.name, opts.track_duration)
            session.track(target, duration=opts.track_duration, announce=False)
        if opts.reset:
            user_logger.info("Resetting F-engine gains to %g", opts.default_gain)
            for inp in session.cbf.fengine.inputs:
                session.cbf.fengine.req.gain(inp, opts.default_gain)
