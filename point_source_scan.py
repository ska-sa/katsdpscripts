#!/usr/bin/python
# Perform mini (Zorro) raster scans across (point) sources from a catalogue for pointing model fits and gain curve calculation.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time

from katuilib.observe import standard_script_options, verify_and_connect, CaptureSession, TimeSession
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] [<catalogue files>]",
                                 description="Perform mini (Zorro) raster scans across (point) sources for pointing \
                                              model fits and gain curve calculation. Use the specified catalogue(s) \
                                              or the default. This script is aimed at fast scans across a large range \
                                              of sources. Some options are **required**.")
# Add experiment-specific options
parser.add_option('-e', '--scan_in_elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth (default=%default)")
parser.add_option('-m', '--min_time', type="float", default=-1.0,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source scan')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    # Load pointing calibrator catalogues
    if len(args) > 0:
        pointing_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
        for catfile in args:
            pointing_sources.add(file(catfile))
    else:
        # Default catalogue contains the radec sources in the standard kat database
        pointing_sources = kat.sources.filter(tags='radec')

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            # Iterate through source list, picking the next one that is up
            for target in pointing_sources.iterfilter(el_limit_deg=5):
                # Do different raster scan on strong and weak targets
                if target.flux_density(opts.centre_freq) > 25.0:
                    session.raster_scan(target, num_scans=5, scan_duration=30, scan_extent=6.0,
                                        scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation)
                else:
                    session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=4.0,
                                        scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation)
                targets_observed.append(target.name)
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break
        print "Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed)))
