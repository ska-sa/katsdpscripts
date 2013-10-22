#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
# Track various point sources as specified in a catalogue for the purpose of baseline calibration.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time

from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] [<'target/catalogue'> ...]",
                                 description='Track various point sources specified by name, string or catalogue, or '
                                             'use the default catalogue if none are specified. This is useful for '
                                             'baseline (antenna location) calibration. Remember to specify the '
                                             'observer and antenna options, as these are **required**.')
# Add experiment-specific options
parser.add_option('-m', '--min-time', type='float', default=-1.0,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Baseline calibration', nd_params='pin,0,0,-1')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    # Create baseline calibrator catalogue
    if len(args) > 0:
        # Load catalogue files or targets if given
        baseline_sources = collect_targets(kat, args)
    else:
        # Prune the standard catalogue to only contain sources that are good for baseline calibration
        great_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A', 'Pictor A']
        good_sources =  ['3C48', '3C84', 'J0408-6545', 'J0522-3627', '3C161', 'J1819-6345', 'J1939-6342', '3C433', 'J2253+1608']
        baseline_sources = katpoint.Catalogue([kat.sources[src] for src in great_sources + good_sources],
                                              antenna=kat.sources.antenna)
        user_logger.info("No targets specified, loaded default catalogue with %d targets" % (len(baseline_sources),))

    with start_session(kat, **vars(opts)) as session:
        if not kat.dry_run:
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
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            # Iterate through baseline sources that are up
            for target in baseline_sources.iterfilter(el_limit_deg=5):
                session.label('track')
                session.track(target, duration=120.0)
                targets_observed.append(target.name)
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break
        user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
