#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time, string
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

def stop_ants(kat):
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
	cntr = 60
	for ant in kat.ants:
	    while ant.sensor.mode.get_value() not in ['STOP']:
                time.sleep(1)
		cntr -= 1
		if cntr < 0:
		    break
    else:
         user_logger.error("Unable to set Antenna mode to 'STOP'.")

def mv_idx(kat, band):
    stop_ants(kat)
    user_logger.info("Moving Receiver Indexer to position %s" % string.upper(band))
    try:
         if not kat.dry_run:
             kat.ants.req.ap_set_indexer_position(string.lower(band))
             time.sleep(60)
    except: raise RuntimeError('Unknown indexer %s' % string.upper(band))

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description='Panel Gap Test 4 - Full azimuth scan.')
# Add experiment-specific options
parser.add_option('--rip', type='string' ,default='u',
                  help='Receiver indexer position (default=%default)')
## RvR 20151206 -- AR1 no delay tracking
# parser.add_option('--no-delays', action="store_true", default=False,
#                   help='Do not use delay tracking, and zero delays')
## RvR 20151206 -- AR1 no delay tracking

# Set default value for any option (both standard and experiment-specific options)
## RvR 20151206 -- DUMP-RATE=4 IN INSTRUCTION SET -- DEFAULT DUMP-RATE OF 1 FORCED
# parser.set_defaults(description='Panel Gap Test',dump_rate=1)
parser.set_defaults(description='Panel Gap Test') # not setting dump-rate
## RvR 20151206 -- DUMP-RATE=4 IN INSTRUCTION SET -- DEFAULT DUMP-RATE OF 1 FORCED

# Parse the command line
opts, args = parser.parse_args()

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

## RvR 20151206 -- RTS antenna to stop mode (need to check this for AR1)
    stop_ants(kat)
## RvR 20151206 -- RTS antenna to stop mode (need to check this for AR1)
    with start_session(kat, **vars(opts)) as session:
## RvR 20151206 -- AR1 no delay tracking
#         if not opts.no_delays and not kat.dry_run :
#             if session.dbe.req.auto_delay('on'):
#                 user_logger.info("Turning on delay tracking.")
#             else:
#                 user_logger.error('Unable to turn on delay tracking.')
#         elif opts.no_delays and not kat.dry_run:
#             if session.dbe.req.auto_delay('off'):
#                 user_logger.info("Turning off delay tracking.")
#             else:
#                 user_logger.error('Unable to turn off delay tracking.')
#             if session.dbe.req.zero_delay():
#                 user_logger.info("Zeroed the delay values.")
#             else:
#                 user_logger.error('Unable to zero delay values.')
## RvR 20151206 -- AR1 no delay tracking

        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []

#   General: 4 Hz dumps, full speed movement.
#   Whole Azimuth scan: Elevation 15, Az from 0 to 360. Elevation 45, Az from 360 to 0. Elevation to 16.

## RvR 20151208 -- Set indexer position command seem not reliable, removing temporarity
# ## RvR 20151207 -- Indexer can only be moved at low elevation
#         target1 = katpoint.Target('slew - back to origin Az=0 El=16, azel, 0, 16')
#         user_logger.info("Initiating '%s'" % (target1.name))
#         session.track(target1, duration=0)
# ## RvR 20151207 -- Indexer can only be moved at low elevation
# ## RvR 20151207 -- Default receiver indexer position
#         mv_idx(kat, opts.rip)
# ## RvR 20151207 -- Default receiver indexer position
## RvR 20151208 -- Set indexer position command seem not reliable, removing temporarity

        session.label('scan')
        target1 = katpoint.Target('scan1 - Whole Az = 0 to -180 El=16, azel, 90, 16')
        user_logger.info("Initiating '%s'" % (target1.name))
        session.scan(target1, duration=500, start=(-90, 0), end=(90, 0), projection='plate-carree')

        session.label('scan')
        target1 = katpoint.Target('scan2 - Whole Az = -180 to 180 El=45, azel, 0, 45')
        user_logger.info("Initiating '%s'" % (target1.name))
        session.scan(target1, duration=1000, start=(180, 0), end=(-180, 0), projection='plate-carree')

        session.label('scan')
        target1 = katpoint.Target('scan3 - Whole Az = 180 to 0 El=16, azel, -90, 16')
        user_logger.info("Initiating '%s'" % (target1.name))
        session.scan(target1, duration=500, start=(-90, 0), end=(90, 0), projection='plate-carree')

## RvR 20151208 -- Nag user to ensure L-band before returning system and leaving
	if string.lower(opts.rip) != 'l':
	    user_logger.info('Receiver Indexer currently on \'%s\', please return to \'l\' before leaving' % string.lower(opts.rip))
## RvR 20151208 -- Nag user to ensure L-band before returning system and leaving

# -fin-
