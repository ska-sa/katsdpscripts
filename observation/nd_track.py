#!/usr/bin/env python
# Track target(s) for a specified time while noise diode is activated continously.

import time
import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track sources in sequence that are above the horizon while firing' \
              'noise diode continuously. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track the source while firing noise diode, '
                       'in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which '
                       'script will end as soon as the current track finishes '
                       '(no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum '
                       'duration (which must be set for this)')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('J1939-6342'), description ('radec, 19:39, -63:42') or "
                     "catalogue file name ('three_calib.csv')")

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(targets.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        target_total_duration = [0.0] * len(targets)
        keep_going = True
        while keep_going:
            keep_going = (opts.max_duration is not None) and opts.repeat
            targets_before_loop = len(targets_observed)
            # Iterate through source list, picking the next one that is up
            for n, target in enumerate(targets):
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
                session.label('nd_track')
                # Get onto the source
                if session.track(target, duration=0):
                    # Fire noise diode during track
                    session.fire_noise_diode(on=opts.track_duration, off=0)
                    targets_observed.append(target.description)
                    target_total_duration[n] += duration
            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - "
                                    "stopping script instead of hanging around")
                keep_going = False
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
        # print out a sorted list of target durations
        ind = np.argsort(target_total_duration)
        for i in reversed(ind):
            user_logger.info('Source %s observed for %.2f hrs',
                             targets.targets[i].description, target_total_duration[i] / 3600.0)