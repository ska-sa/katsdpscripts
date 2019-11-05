#!/usr/bin/env python
# Track target(s) for a specified time.

import time
import numpy as np
from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track one or more sources for a specified time. At least one ' \
              'target must be specified. Note also some **required** options below.'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=30.0,
                  help='Length of time to track each source, in seconds '
                       '(default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which '
                       'script will end as soon as the current track finishes '
                       '(no limit by default)')
parser.add_option('--gain', default='10,5000,500',
                  help='Values of the correlator F-engine gain '
                       'in the form "start,stop,number of steps" '
                       '(default=%default)')
parser.add_option('--fft-shift', type='int_or_default',
                  help='Override correlator F-engine FFT shift')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track', nd_params='coupler,30,0,-1')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    args.append('SCP, radec, 0, -90')

g_start, g_end, g_step = np.array(opts.gain.split(',')).astype(float)
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
        if opts.fft_shift is not None:
            session.set_fengine_fft_shift(opts.fft_shift)
        session.capture_start()

        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            keep_going = opts.max_duration is not None
            targets_before_loop = len(targets_observed)
            # Iterate through source list, picking the next one that is up
            for gain in np.logspace(np.log10(g_start), np.log10(g_end), g_step):
                for target in targets.iterfilter(el_limit_deg=opts.horizon):
                    # Cut the track short if time ran out
                    duration = opts.track_duration
                    if opts.max_duration is not None:
                        time_left = opts.max_duration - (time.time() - start_time)
                        if time_left <= 0.:
                            user_logger.warning("Maximum duration of %g seconds "
                                                "has elapsed - stopping script",
                                                opts.max_duration)
                            keep_going = False
                            break
                        duration = min(duration, time_left)
                    # Set the gain to a single non complex number if needed
                    session.label('track_gain,%g,%gi' % (gain.real, gain.imag))
                    session.set_fengine_gains(gain)
                    if session.track(target, duration=duration):
                        targets_observed.append(target.description)
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - "
                                        "stopping script instead of hanging around")
                    keep_going = False
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
