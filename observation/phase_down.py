#!/usr/bin/env python
#
# Reset the delay adjustments and CBF F-engine gains to zero.

import time

import numpy as np
import katpoint
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                SessionSDP)

class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""

# Set up standard script options
usage = "%prog"
description = 'Reset all the delays and phases to default.'
parser = standard_script_options(usage, description)
# Add experiment-specific options

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Phase-up observation that sets the F-engine weights')
# Parse the command line
opts, args = parser.parse_args()

# Set of targets with flux models
J1934 = 'PKS 1934-63 | J1939-6342, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231)'
J0408 = 'PKS 0408-65 | J0408-6545, radec, 4:08:20.38, -65:45:09.1, (800.0 8400.0 -3.708 3.807 -0.7202)'
J1331 = '3C286      | J1331+3030, radec, 13:31:08.29, +30:30:33.0,(800.0 43200.0 0.956 0.584 -0.1644)'

observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
observation_sources.add(J1934)
observation_sources.add(J0408)
observation_sources.add(J1331)

if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
    raise NoTargetsUpError("No targets are currently visible - please re-run the script later")
        
with verify_and_connect(opts) as kat:
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.cbf.correlator.req.capture_start()
        channels = 32768 if session.product.endswith('32k') else 4096
        if channels == 4096:
            default_gain = 200
        elif channels == 32768:
            default_gain = 4000
        user_logger.info("Resetting F-engine gains to %g", default_gain)
        for inp in session.cbf.fengine.inputs:
            session.cbf.fengine.req.gain(inp, default_gain)
        user_logger.info("Resetting delay adjustments to zero")
        session.cbf.req.adjust_all_delays()
        
        track_duration = 24
        for target in [observation_sources.sort('el').targets[-1]]:
            user_logger.info("Initiating %g-second track on target '%s' to validate reset",
                              track_duration, target.name)
            session.track(target, duration=track_duration, announce=False)
