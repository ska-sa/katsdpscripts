#!/usr/bin/env python
# Observe target(s) for a specified time at a sequence of offsets that are suitable for deriving antenna pointing offsets.

import time

import numpy as np
from katcorelib import collect_targets, standard_script_options, verify_and_connect, start_session, user_logger
import katpoint


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Observe one or more targets for a specified time at a sequence of offsets '
                                             'that are suitable for deriving antenna pointing offsets. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=20.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option('--max-offset', type='float', default=1.0,
                  help='Maximum radial offset of pointings from target in degrees (default=%default)')
parser.add_option('--number-of-steps', type='int', default=5,
                  help='Number of pointings to measure at, one on bore sight, the rest off (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Inferometric Pointing offset track', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
# Ensure that the lowest offset pointing will also be up if target is up
opts.horizon += opts.max_offset

# Set up the pointing pattern excluding bore sight
num_off = opts.number_of_steps - 1
angles = 2 * np.pi * np.arange(num_off) / num_off
offsets = opts.max_offset * np.c_[np.cos(angles), np.sin(angles)]

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    args_target_list = []
    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    try:
        observation_sources.add_tle(file(args[0]))
    except (IOError, ValueError):  # If the file failed to load assume it is a target string
        args_target_obj = collect_targets(kat, args)
        observation_sources.add(args_target_obj)
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []
        try: # Keep going until the time is up or interrupted
            keep_going = True
            while keep_going:
                keep_going = (opts.max_duration is not None)
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    # Check if all offset pointings in compound scan will be up
                    # Add some extra time for slews between pointings
                    step_duration = opts.track_duration + 4.
                    compound_duration = opts.number_of_steps * step_duration
                    if not session.target_visible(target, duration=compound_duration):
                        continue
                    session.label('interferometric_pointing')
                    session.ants.req.offset_fixed(0, 0, opts.projection) # In case it's the same target as previous cycle
                    session.track(target, duration=opts.track_duration, announce=False)
                    target.tags = target.tags[:1] # this is to avoid overloading the cal pipeline
                    session.track(target, duration=0, announce=False) # Offsets are relative to THIS target (altered in previous line)
                    for offset_target in offsets:
                        user_logger.info("Offset of (%f, %f) degrees", *offset_target)
                        if not kat.dry_run:
                            session.ants.req.offset_fixed(offset_target[0], offset_target[1], opts.projection)
                        session.track(target, duration=opts.track_duration, announce=True)
                    targets_observed.append(target.name)
                    if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script",
                                            opts.max_duration)
                        keep_going = False
                        break

                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - "
                                        "stopping script instead of hanging around")
                    keep_going = False
        finally:
            user_logger.info("Targets observed : %d (%d unique)",
                             len(targets_observed), len(set(targets_observed)))
