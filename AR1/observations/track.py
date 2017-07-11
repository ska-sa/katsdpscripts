#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum duration (which must be set for this)')
parser.add_option('--reset-gain', type='int', default=None,
                  help='Value for the reset of the correlator F-engine gain (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            # Set the gain to a single non complex number if needed
            if opts.reset_gain is not None:
                if not session.cbf.fengine.inputs:
                    user_logger.error("Failed to get correlator input labels")
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
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    session.label('track')
                    user_logger.info("Initiating %g-second track on target %r",
                                     opts.track_duration, target.name)
                    # Split the total track on one target into segments lasting as long as the noise diode period
                    # This ensures the maximum number of noise diode firings
                    total_track_time = 0.
                    while total_track_time < opts.track_duration:
                        next_track = opts.track_duration - total_track_time
                        # Cut the track short if time ran out
                        if opts.max_duration is not None:
                            next_track = min(next_track, opts.max_duration - (time.time() - start_time))
                        if opts.nd_params['period'] > 0:
                            next_track = min(next_track, opts.nd_params['period'])
                        if next_track <= 0 or not session.track(target, duration=next_track, announce=False):
                            break
                        total_track_time += next_track
                    if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script",
                                            opts.max_duration)
                        keep_going = False
                        break
                    targets_observed.append(target.name)
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - "
                                        "stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)",
                             len(targets_observed), len(set(targets_observed)))
