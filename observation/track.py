#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement
import time
from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target 1'> [<'target 2'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='If set, this would be the max duration of the track script. After max-duration seconds the script will end after the current track. (default=%default)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track')
# Parse the command line
opts, args = parser.parse_args()
if  opts.max_duration is not  None :
    opts.track_duration = opts.max_duration
# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) == 0:
    raise ValueError("Please specify at least one target argument "
                     "(via name, e.g. 'Cygnus A' or description, e.g. 'azel, 20, 30')")
start_time = time.time()
with verify_and_connect(opts) as kat:

    targets = lookup_targets(kat, args)

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        for target in targets:
            target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
            user_logger.info("Initiating %g-second track on target '%s'" % (opts.track_duration, target.name))
            # Split the total track on one target into segments lasting as long as the noise diode period
            # This ensures the maximum number of noise diode firings
            total_track_time = 0.
            while  (opts.max_duration is None and total_track_time < opts.track_duration ) or ( opts.max_duration is not  None and  ( time.time() - start_time)< opts.max_duration):
                if  opts.max_duration is None :
                    next_track = opts.track_duration - total_track_time
                else  :
                    next_track =  opts.max_duration - (time.time() - start_time)
                if opts.nd_params['period'] > 0:
                    next_track = min(next_track, opts.nd_params['period'])
                session.track(target, duration=next_track, label='', announce=False)
                total_track_time += next_track

