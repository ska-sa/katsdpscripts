#!/usr/bin/python
# Track target(s) for a specified time.


import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

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

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track',dump_rate=1)
# Parse the command line
opts, args = parser.parse_args()

#if len(args) == 0:
#    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
#                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
#    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
#    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
#        user_logger.warning("No targets are currently visible - please re-run the script later")
#    else:
        # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []
            # Keep going until the time is up
   
   
#   General: 4 Hz dumps, full speed movement.
#   Whole Azimuth scan: Elevation 15, Az from 0 to 360. Elevation 45, Az from 360 to 0. Elevation to 16.
        
        session.label('scan')
        target1 = katpoint.Target('scan1 - Whole Az = 0 to -180 El=16, azel, 90, 16')    
        user_logger.info("Initiating '%s'" % (target1.name))	
        session.scan(target1, duration=100, start=(-90, 0), end=(90, 0), projection='plate-carree')

        session.label('scan')
        target1 = katpoint.Target('scan2 - Whole Az = -180 to 180 El=45, azel, 0, 45')    
        user_logger.info("Initiating '%s'" % (target1.name))	
        session.scan(target1, duration=190, start=(180, 0), end=(-180, 0), projection='plate-carree')
        
        session.label('scan')
        target1 = katpoint.Target('scan2 - Whole Az = 180 to 0 El=16, azel, -90, 16')    
        user_logger.info("Initiating '%s'" % (target1.name))	
        session.scan(target1, duration=100, start=(-90, 0), end=(90, 0), projection='plate-carree')

#                keep_going = (opts.max_duration is not None) and opts.repeat
#                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that:wis up
#                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
#                    session.label('track')
#                    user_logger.info("Initiating %g-second track on target '%s'" % (opts.track_duration, target.name,))
                    # Split the total track on one target into segments lasting as long as the noise diode period
                    # This ensures the maximum number of noise diode firings
#                    total_track_time = 0.
#                    while total_track_time < opts.track_duration:
#                        next_track = opts.track_duration - total_track_time
                        # Cut the track short if time ran out
#                        if opts.max_duration is not None:
#                            next_track = min(next_track, opts.max_duration - (time.time() - start_time))
#                        if opts.nd_params['period'] > 0:
#                            next_track = min(next_track, opts.nd_params['period'])
#                        if next_track <= 0 or not session.track(target, duration=next_track, announce=False):
#                            break
#                        total_track_time += next_track
#                    if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
#                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script" %
#                                            (opts.max_duration,))
#                        keep_going = False
#                        break
#                    targets_observed.append(target.name)
#                if keep_going and len(targets_observed) == targets_before_loop:
#                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
           # keep_going = False
#            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
