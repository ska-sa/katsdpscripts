#! /usr/bin/env python
#
# Check antenna pointing on a series of sources.
#
# Do a pointing scan and fit pointing offsets on one or more targets.
# Repeat this for a specified duration, which will correct and verify
# the pointing when done on the same or nearby sources.
#
# Ludwig Schwardt
# 17 September 2024
#

import time

from katcorelib.observe import (
    standard_script_options,
    verify_and_connect,
    collect_targets,
    start_session,
    user_logger
)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = ('Perform offset pointings on each source and obtain pointing offsets '
               'based on interferometric gains. At least one target must be specified.')
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=24.0,
                  help='Duration of each offset pointing, in seconds (default=%default)')
parser.add_option('-m', '--min-time', type="float", default=-1.0,
                  help="Minimum duration of observation, in seconds "
                       "(default=one loop through sources)")
parser.add_option('--max-extent', type='float', default=1.0,
                  help='Maximum distance of pointing offset from target, in degrees')
parser.add_option('--pointings', type='int', default=9,
                  help='Number of offset pointings per pointing scan')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Pointing check', nd_params='off')
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
        # Quit early if there are no sources to observe or not enough antennas
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        session.capture_start()
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            targets_before_loop = len(targets_observed)
            # Iterate through pointing sources that are up
            for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                target.add_tags('pointingcal')
                session.label('interferometric_pointing')
                session.reference_pointing_scan(
                    target, opts.track_duration, opts.max_extent, opts.pointings
                )
                targets_observed.append(target.name)
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break
            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - "
                                    "stopping script instead of hanging around")
                keep_going = False
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
