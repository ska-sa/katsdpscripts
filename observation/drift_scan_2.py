#!/usr/bin/python
# Track target briefly, then advance in Hour angle by half the specified duration at which
# point the drift scan is observed. Ends off with another track on target.
# The noise diode is fired during both the initial and final tracks on target.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint
import math

# Set up standard script options
parser = standard_script_options(usage="%prog [options] 'target'",description=
                                             "Track target briefly, then advance in Hour angle by half the specified duration"
                                             "at which point the drift scan is observed. Ends off with another track on target."
                                             "The noise diode is fired during both the initial and final tracks on target."
                                             "'target' must be specified. Note also some **required** options below.")
# Add experiment-specific options
parser.add_option('-m', type='float', default=0, help='Ignored, option is kept for backwards compatibility')
parser.add_option('-t', '--drift-duration', type='float', default=300,
                  help='Total duration of drift scan, in seconds (default=%default)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track',nd_params="off") # Prevent session.track() from firing the noise diode
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "or description ('azel, 20, 30')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(5)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")

    observation_sources = collect_targets(kat, args)
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        with start_session(kat, **vars(opts)) as session:

            session.standard_setup(**vars(opts))
            session.capture_start()
            target = observation_sources.filter(el_limit_deg=opts.horizon).targets[0]

            session.track(target, duration=1) # APH added this 12/2018 to avoid first ND cycle whiletarget is drifting off bore sight
            time.sleep(4)
            start_time = time.time()

            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)

            session.label('track')
            user_logger.info("Tracking %s for 30 seconds" % (target.name))
            session.track(target, duration=30)

            session.label('drift')
            start_time = time.time()
            az,el = target.azel(start_time + (opts.drift_duration / 2))
            if (az*180/math.pi > 275.0):
                az = az - (360/180 * math.pi)
            new_targ = katpoint.Target('Drift scan_duration of %s, azel, %10.8f, %10.8f' % (target.name, az*180/math.pi ,el*180/math.pi))
            user_logger.info("Initiating drift scan of %s" % (target.name))
            az,el = target.azel(start_time + (opts.drift_duration / 2))
            session.track(new_targ, duration=opts.drift_duration)

            session.label('track')
            user_logger.info("Tracking %s for 30 seconds" % (target.name))
            session.track(target, duration=30)

            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)

