#!/usr/bin/env python
# Track target(s) for a specified time.

import time

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
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds '
                       '(default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which '
                       'script will end as soon as the current track finishes '
                       '(no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum '
                       'duration (which must be set for this)')
parser.add_option('--reset-gain', type='int', default=None,
                  help='Value for the reset of the correlator F-engine gain '
                       '(default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(targets.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Set the gain to a single non complex number if needed
        if opts.reset_gain is not None:
            if not session.cbf.fengine.inputs:
                raise RuntimeError("Failed to get correlator input labels, "
                                   "cannot set the F-engine gains")
            for inp in session.cbf.fengine.inputs:
                session.cbf.fengine.req.gain(inp, opts.reset_gain)
                user_logger.info("F-engine %s gain set to %g",
                                 inp, opts.reset_gain)

        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            keep_going = (opts.max_duration is not None) and opts.repeat
            targets_before_loop = len(targets_observed)
            # Iterate through source list, picking the next one that is up
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
                session.label('track')
                if session.track(target, duration=duration):
                    targets_observed.append(target.name)
            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - "
                                    "stopping script instead of hanging around")
                keep_going = False
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
