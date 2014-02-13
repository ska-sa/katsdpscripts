#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
#import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'>  [--cold-target=<'target/catalogue'> ...]",
                                 description='Track 2 sources , one strong source which is bracketed by the cold sky scource' 
                                             'for a specified time. The strong target must be specified.'
                                             'The first valid source in the catalogue give will be used')
# Add experiment-specific options
parser.add_option('--project-id',
                  help='Project ID code the observation (**required**) This is a required option')
parser.add_option('-t', '--track-duration', type='float', default=7200.0,
                  help='Length of time to track the Main source in seconds (default=%default)')
parser.add_option('--cold-duration', type='float', default=900.0,
                  help='Length of time to track the cold sky source in seconds when bracketing the obsevation (default=%default)')
parser.add_option('--cold-target', type='string', default="SCP,radec,0,-90",
                  help='The target/catalogue of the cold sky source to use when bracketing the obsevation (default=%default)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Strong Sources track',dump_rate=0.1)
# Parse the command line
opts, args = parser.parse_args()

if not hasattr(opts, 'project_id') or opts.project_id is None:
    raise ValueError('Please specify the Project id code via the --project_id option '
                     '(yes, this is a non-optional option...)')
if len(args) == 0:
    raise ValueError("Please specify at target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    strong_sources = collect_targets(kat, args)
    cold_sources = collect_targets(kat, opts.cold_target)
    # Quit early if there are no sources to observe
    valid_targets = True
    if len(strong_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No strong source targets are currently visible - please re-run the script later")
        valid_targets = False
    if len(cold_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No cold source targets are currently visible - please re-run the script later")
        valid_targets = False
    if valid_targets:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            if not opts.no_delays and not kat.dry_run :
                if session.dbe.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
                if session.dbe.req.auto_delay('off'):
                    user_logger.info("Turning off delay tracking.")
                else:
                    user_logger.error('Unable to turn off delay tracking.')
                if session.dbe.req.zero_delay():
                    user_logger.info("Zeroed the delay values.")
                else:
                    user_logger.error('Unable to zero delay values.')

            session.standard_setup(**vars(opts))
            session.capture_start()
            target_list = []
            target_list.append((cold_sources,opts.cold_duration)) 
            target_list.append((strong_sources,opts.track_duration))
            target_list.append((cold_sources,opts.cold_duration))
            for observation_sources,track_duration in target_list:    
                # Iterate through source list, picking the first one that is up
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    session.label('track')
                    user_logger.info("Initiating %g-second track on target '%s'" % (opts.track_duration, target.name,))
                    # Split the total track on one target into segments lasting as long as the noise diode period
                    # This ensures the maximum number of noise diode firings
                    total_track_time = 0.
                    start_time = time.time()
                    while total_track_time < track_duration:
                        next_track = track_duration - total_track_time
                        # Cut the track short if time ran out
                        if opts.nd_params['period'] > 0:
                            next_track = min(next_track, opts.nd_params['period'])
                        if next_track <= 0 or not session.track(target, duration=next_track, announce=False):
                            break
                        total_track_time += next_track
