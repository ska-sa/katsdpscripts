#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description='Warming up the antennas if they are to cold to move.')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track',dump_rate=1)
# Parse the command line
opts, args = parser.parse_args()

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))

# General: 4 Hz dumps, full speed movement.
# Test elevation scan: Az=0, El=15 to 60. Stop at 70 for 10 seconds. El from 70 to 15.
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
                target1 = katpoint.Target('slew - back to origin Az=0 El=16, azel, 0, 16')    
                user_logger.info("Initiating '%s'" % (target1.name))	
                session.track(target1, duration=0)
                kat.ants.req.mode('STOP')

# -fin-      
