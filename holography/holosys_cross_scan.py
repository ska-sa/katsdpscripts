#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
# Perform mini (Zorro) raster scans across the holography system's satellite of choice, EUTELSAT W2M.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time

from katcorelib import standard_script_options, verify_and_connect, user_logger, CaptureSession

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                               description="Perform mini (Zorro) raster scans across the holography sources \
                                            Some options are **required**.")
# Add experiment-specific options
parser.add_option('-e', '--scan_in_elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth, (default=%default)")
parser.add_option('-m', '--min_time', type="float", default=-1.0,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Multiple raster scans on holography target')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    # Source to scan across
    pointing_sources = [kat.sources['EUTELSAT W2M']]

    start_time = time.time()
    targets_observed = []

    # The real experiment: Create a data capturing session with the selected sub-array of antennas
    with CaptureSession(kat, **vars(opts)) as session:
        # HACK DBE to accept target(target) and do nothing with it.
        kat.dbe.req.target = lambda target: None
        session.ants.req.sensor_sampling("lock", "event")
        session.ants.req.sensor_sampling("scan.status", "event")

        # Keep going until the time is up
        keep_going = True
        while keep_going:
            # Iterate through source list, picking the next one that is up
            for target in pointing_sources:
                session.raster_scan(target, num_scans=7, scan_duration=10, scan_extent=0.5,
                                    scan_spacing=0.1, scan_in_azimuth=not opts.scan_in_elevation)
                targets_observed.append(target.name)
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break

    user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
