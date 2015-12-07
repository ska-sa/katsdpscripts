#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description='Panel Gap Test 1 - Az=0 EL=15 to 86 and back.')
# Add experiment-specific options
## RvR 20151206 -- AR1 no delay tracking
# parser.add_option('--no-delays', action="store_true", default=False,
#                   help='Do not use delay tracking, and zero delays')
## RvR 20151206 -- AR1 no delay tracking

# Set default value for any option (both standard and experiment-specific options)
## RvR 20151206 -- DUMP-RATE=4 IN INSTRUCTION SET -- DEFAULT DUMP-RATE OF 1 FORCED
parser.set_defaults(description='Target track',dump_rate=1)
## RvR 20151206 -- DUMP-RATE=4 IN INSTRUCTION SET -- DEFAULT DUMP-RATE OF 1 FORCED

# Parse the command line
opts, args = parser.parse_args()

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

## RvR 20151206 -- RTS antenna to stop mode (need to check this for AR1)
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
         time.sleep(10)
    else:
         user_logger.error("Unable to set Antenna mode to 'STOP'.")
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
#   Test elevation scan: Az=0, El=15 to 60. Stop at 70 for 10 seconds. El from 70 to 15.

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

# -fin-
