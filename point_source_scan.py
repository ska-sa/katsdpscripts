#!/usr/bin/python
# Perform mini (Zorro) raster scans across (point) sources from a catalogue for pointing model fits and gain curve calculation.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import datetime
import os
import time
from cStringIO import StringIO

from katuilib.observe import standard_script_options, verify_and_connect, CaptureSession, TimeSession, user_logger
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
parser.add_option('-z', '--skip-catalogue', dest='skip_catalogue', type="string", default=None,
                  help="Name of file containing catalogue of targets to skip (default is not to skip any targets).")
parser.add_option('--source-strength', dest='source_strength', type="choice", default="auto",
                  choices=['strong', 'weak', 'auto'],
                  help="Select scanning strategy based on strength of sources. Options are 'strong' or 'weak' or 'auto'. Auto is based "
                       "on flux density specified in catalogue")
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

    skip_sources = katpoint.Catalogue(add_specials=False, antenna=kat.sources.antenna)
    if opts.skip_catalogue is not None and os.path.exists(opts.skip_catalogue):
        skip_sources.add(file(opts.skip_catalogue))

    if opts.skip_catalogue is not None and not opts.dry_run:
        skip_file = file(opts.skip_catalogue, "a")
    else:
        skip_file = StringIO()

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        skip_file.write("# Record of targets observed on %s by %s\n" % (datetime.datetime.now(), opts.observer))
        while keep_going:
            # Iterate through source list, picking the next one that is up
            for target in pointing_sources.iterfilter(el_limit_deg=5):
                # Do different raster scan on strong and weak targets
                if target.name in skip_sources:
                    continue
                if (opts.source_strength == 'strong' or
                    (opts.source_strength == 'auto' and target.flux_density(opts.centre_freq) > 25.0)):
                    session.raster_scan(target, num_scans=5, scan_duration=30, scan_extent=6.0,
                                        scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation)
                else:
                    session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=4.0,
                                        scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation)
                targets_observed.append(target.name)
                skip_file.write(target.description + "\n")
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break
        user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
        skip_file.close()
