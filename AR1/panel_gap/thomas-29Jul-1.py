#!/usr/bin/python
# Track target(s) for a specified time.


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
                                 description='Panel Gap Test 1 - Az=0 EL=15 to 86 and back.')
# Add experiment-specific options
parser.add_option('--rip', type='string' ,default='u',
                  help='Receiver indexer position (default=%default)')

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
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []

#   General: 4 Hz dumps, full speed movement.
#   Test elevation scan: Az=0, El=15 to 60. Stop at 70 for 10 seconds. El from 70 to 15.

## RvR 20151208 -- Set indexer position command seem not reliable, removing temporarity
# ## RvR 20151207 -- Indexer can only be moved at low elevation
#         target1 = katpoint.Target('slew - back to origin Az=0 El=16, azel, 0, 16')
#         user_logger.info("Initiating '%s'" % (target1.name))
#         session.track(target1, duration=0)
# ## RvR 20151207 -- Indexer can only be moved at low elevation
# ## RvR 20151207 -- Default receiver indexer position
#         mv_idx(kat, opts.rip)
## RvR 20151207 -- Default receiver indexer position
## RvR 20151208 -- Set indexer position command seem not reliable, removing temporarity

        session.label('scan')
        target1 = katpoint.Target('scan1 - Az=0 El=16 to 86, azel, 0, 51')
        user_logger.info("Initiating '%s'" % (target1.name))
        session.scan(target1, duration=80, start=(0, -35), end=(0, 35))

        target1 = katpoint.Target('scan2 - Az=0 El=86 Hold for 10 sec, azel, 0, 86')
        user_logger.info("Initiating '%s'" % (target1.name))
        session.track(target1, duration=10)

        target1 = katpoint.Target('scan3 - Az=0 El=86 to 16, azel, 0, 51')
        user_logger.info("Initiating '%s'" % (target1.name))
        session.scan(target1, duration=80, start=(0, 35), end=(0, -35))

## RvR 20151208 -- Nag user to ensure L-band before returning system and leaving
	if string.lower(opts.rip) != 'l':
	    user_logger.info('Receiver Indexer currently on \'%s\', please return to \'l\' before leaving' % string.lower(opts.rip))
## RvR 20151208 -- Nag user to ensure L-band before returning system and leaving

# -fin-
