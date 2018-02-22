#!/usr/bin/env python
#
# Track delay calibrator target for a specified time.
# Obtain delay solutions and apply them to the delay tracker in the CBF proxy.
#
# Ludwig Schwardt
# 5 April 2017
#

import json

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)
from katsdptelstate import TimeoutError


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Default F-engine gain as a function of number of channels
DEFAULT_GAIN = {4096: 200, 32768: 4000}


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track the source with the highest elevation and calibrate ' \
              'delays based on it. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=32.0,
                  help='Length of time to track the source, in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int', default=0,
                  help='Correlator F-engine gain, automatically set if 0 (the '
                       'default) and left alone if negative')
parser.add_option('--reset-delays', action='store_true', default=False,
                  help='Zero the delay adjustments afterwards')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Delay calibration observation')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")


# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Pick source with the highest elevation as our target
        target = observation_sources.sort('el').targets[-1]
        target.add_tags('bfcal single_accumulation')
        session.standard_setup(**vars(opts))
        session.cbf.correlator.req.capture_start()
        if opts.fengine_gain <= 0 :
                num_channels = session.cbf.fengine.sensor.n_chans.get_value()
                opts.fengine_gain = DEFAULT_GAIN.get(num_channels, -1)
        gains = {}
        for inp in  session.get_cal_inputs():
            gains[inp] = opts.fengine_gain
        session.set_fengine_gains(gains)
        user_logger.info("Zeroing all delay adjustments for starters")
        delays = {}
        for inp in  session.get_cal_inputs():
            delays[inp] = 0.0
        session.set_delays(delays)

        session.capture_start()
        user_logger.info("Initiating %g-second track on target %r",
                         opts.track_duration, target.description)
        session.label('un_corrected')
        session.track(target, duration=opts.track_duration, announce=False)
        # Attempt to jiggle cal pipeline to drop its delay solutions
        session.ants.req.target('')

        user_logger.info("Waiting for delays to materialise in cal pipeline")
        sample_rate = 0.0
        delays = session.get_delaycal_solutions(timeout=90.)
        session.set_delays(delays)

        user_logger.info("Revisiting target %r for %g seconds to see if "
                         "delays are fixed", target.name, opts.track_duration)
        session.label('corrected')
        session.track(target, duration=opts.track_duration, announce=False)
        if opts.reset_delays:
            user_logger.info("Zeroing all delay adjustments on CBF proxy")
            for inp in  session.get_cal_inputs():
                delays[inp] = 0.0
            session.set_delays(delays)
