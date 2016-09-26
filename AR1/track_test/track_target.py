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
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Track Test')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    args_target_obj = collect_targets(kat,args)
    observation_sources.add(args_target_obj)

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
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
                        [ra,dec]=target.radec()
                        (tra,tdec) = (katpoint.rad2deg(float(ra)), katpoint.rad2deg(float(dec)))
#                         (ra,dec) = (tra+0.25*dra, tdec+0.25*ddec)
                        (ra,dec) = (tra+0.5*dra, tdec+0.5*ddec)
#                         (ra,dec) = (tra+1*dra, tdec+1*ddec)
                        newtarget = katpoint.construct_radec_target(katpoint.deg2rad(ra), katpoint.deg2rad(dec))
                        session.label('track')
                        user_logger.info("Initiating %g-second track on target (%.2f, %.2f)" % (opts.track_duration, ra, dec,))
                        session.set_target(newtarget) # Set the target
                        session.track(newtarget, duration=opts.track_duration, announce=False) # Set the target & mode = point
# -fin-
