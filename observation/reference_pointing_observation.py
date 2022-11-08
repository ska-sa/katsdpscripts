# TODO: @mamkhari

import time

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                CalSolutionsUnavailable)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""

# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Perform offset pointings on the first source and obtain ' \
              'pointing offsets based on interferometric gains. At least ' \
              'one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=16.0,
                  help='Duration of each offset pointing, in seconds (default=%default)')
parser.add_option('--max-extent', type='float', default=1.0,
                  help='Maximum distance of offset from target, in degrees')
parser.add_option('--pointings', type='int', default=10,
                  help='Number of offset pointings')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Reference pointing', nd_params='off')
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
            raise ValueError(
                "Not enough receptors to do calibration - you "
                "need 4 and you have %d" % (len(session.ants),)
            )
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError(
                "No targets are currently visible - " "please re-run the script later"
            )
        session.standard_setup(**vars(opts))
        session.capture_start()

        # XXX Eventually pick closest source as our target, now take first one
        target = observation_sources.targets[0]
        target.add_tags("bfcal single_accumulation")
        session.reference_pointing_scan(
            target,
            duration=opts.track_duration,
            extent=opts.max_extent,
            num_pointings=opts.pointings,
        )
