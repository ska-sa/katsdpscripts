#! /usr/bin/python

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os
import numpy as np
import time

from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Perfrom a test track on a target + 8*0.25deg offset points.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of the drift scan for each source, in seconds (default=%default)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Track Test')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
         user_logger.error("Dry Run: Unable to set Antenna mode to 'STOP'.")

    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    args_target_obj = collect_targets(kat,args)
    observation_sources.add(args_target_obj)

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
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
            session.capture_start()

            # Iterate through source list, picking the next one that is up
            for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                user_logger.info(target)
                [ra,dec]=target.radec()
                (tra,tdec) = (katpoint.rad2deg(float(ra)), katpoint.rad2deg(float(dec)))
                session.label('track')
                user_logger.info("Initiating %g-second track on target (%.2f, %.2f)" % (opts.track_duration, tra, tdec,))
                session.set_target(target) # Set the target
                session.track(target, duration=opts.track_duration, announce=False) # Set the target & mode = point
                for dra in [-1,0,1]:
                    for ddec in [-1,0,1]:
                        (ra,dec) = (tra+0.25*dra, tdec+0.25*ddec)
                        target = katpoint.construct_radec_target(katpoint.deg2rad(ra), katpoint.deg2rad(dec))
                        session.label('track')
                        user_logger.info("Initiating %g-second track on target (%.2f, %.2f)" % (opts.track_duration, ra, dec,))
                        session.set_target(target) # Set the target
                        session.track(target, duration=opts.track_duration, announce=False) # Set the target & mode = point


    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
         user_logger.error("Dry Run: Unable to set Antenna mode to 'STOP'.")

# -fin-
