#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# # Add experiment-specific options
# parser.add_option('-t', '--track-duration', type='float', default=60.0,
#                   help='Length of time to track each source, in seconds (default=%default)')
# parser.add_option('-m', '--max-duration', type='float', default=None,
#                   help='Maximum duration of the script in seconds, after which script will end '
#                        'as soon as the current track finishes (no limit by default)')
# parser.add_option('--repeat', action="store_true", default=False,
#                   help='Repeatedly loop through the targets until maximum duration (which must be set for this)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track',dump_rate=1)
# Parse the command line
opts, args = parser.parse_args()

#if len(args) == 0:
#    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
#                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
         user_logger.error("Unable to set Antenna mode to 'STOP'.")

#    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
#    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
#        user_logger.warning("No targets are currently visible - please re-run the script later")
#    else:
        # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        if not opts.no_delays and not kat.dry_run :
            if session.data.req.auto_delay('on'):
                user_logger.info("Turning on delay tracking.")
            else:
                user_logger.error('Unable to turn on delay tracking.')
        elif opts.no_delays and not kat.dry_run:
            if session.data.req.auto_delay('off'):
                user_logger.info("Turning off delay tracking.")
            else:
                user_logger.error('Unable to turn off delay tracking.')
            if session.data.req.zero_delay():
                user_logger.info("Zeroed the delay values.")
            else:
                user_logger.error('Unable to zero delay values.')

        session.standard_setup(**vars(opts))
#         session.capture_start()
# 
#         start_time = time.time()
#         targets_observed = []
#             # Keep going until the time is up

#   General: 4 Hz dumps, full speed movement.
#   Test elevation scan: Az=0, El=15 to 60. Stop at 70 for 10 seconds. El from 70 to 15.
	continue_test=True
	if not kat.dry_run:
            # User position sensors to make sure the system is in a safe place before starting movement
            current_az = session.ants[0].sensor.pos_actual_scan_azim.get_value()
            current_el = session.ants[0].sensor.pos_actual_scan_elev.get_value()
            if current_az is None:
                user_logger.warning("Sensor kat.%s.sensor.pos_actual_scan_azim failed - using default azimuth" %
                                    (session.ants[0].name))
                continue_test=False
            if current_el is None:
                user_logger.warning("Sensor kat.%s.sensor.pos_actual_scan_elev failed - using default elevation" %
                                    (session.ants[0].name))
                continue_test=False
 	    elif current_el < 20.: 
                continue_test=False

	    if continue_test: #only continue if the antenna is in a safe place to move
                session.label('0.5 deg/sec')
                kat.ants.req.mode('STOP')
                time.sleep(5)
                kat.ants.req.ap_rate(0.5,0)
                time.sleep(40)
            
                session.label('1 deg/sec')
                kat.ants.req.ap_rate(1,0)
                time.sleep(40)
                
                session.label('1.5 deg/sec')
                kat.ants.req.ap_rate(-1.5,0)
                time.sleep(40)
            
                kat.ants.req.mode('STOP')
                time.sleep(5)
#                 target1 = katpoint.Target('slew - back to origin Az=0 El=16, azel, 0, 16')    
#                 user_logger.info("Initiating '%s'" % (target1.name))	
#                 session.track(target1, duration=0)
#                 kat.ants.req.mode('STOP')
        
