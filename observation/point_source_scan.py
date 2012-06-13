#!/usr/bin/python
# Perform mini (Zorro) raster scans across (point) sources from a catalogue for pointing model fits and gain curve calculation.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os.path
from cStringIO import StringIO
import datetime
import time

from katcorelib import collect_targets, standard_script_options, verify_and_connect, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] [<'target/catalogue'> ...]",
                                 description="Perform mini (Zorro) raster scans across (point) sources for pointing "
                                             "model fits and gain curve calculation. Use the specified target(s) and "
                                             "catalogue(s) or the default. This script is aimed at fast scans across "
                                             "a large range of sources. Some options are **required**.")
# Add experiment-specific options
parser.add_option('-e', '--scan-in-elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth (default=%default)")
parser.add_option('-m', '--min-time', type="float", default=-1.0,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.add_option('-z', '--skip-catalogue',
                  help="Name of file containing catalogue of targets to skip (updated with observed targets). "
                       "The default is not to skip any targets.")
parser.add_option('--source-strength', type='choice', default='auto', choices=('strong', 'weak', 'auto'),
                  help="Scanning strategy based on source strength, one of 'strong', 'weak' or 'auto' (default). "
                       "Auto is based on flux density specified in catalogue.")
parser.add_option( '--quick', action="store_true" , default=False,
                  help='Do a quick "Zorro" type scan, 3 scans of 15 seconds over 5 degrees spaced 0.5 apart at 2Hz sample rate.')
parser.set_defaults(description='Point source scan')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    if len(args) > 0:
        # Load pointing calibrator catalogues and command line targets
        pointing_sources = collect_targets(kat, args)
    else:
        # Default catalogue contains the radec sources in the standard kat database
        pointing_sources = kat.sources.filter(tags='radec')
        user_logger.info("No valid targets specified, loaded default catalogue with %d targets" %
                         (len(pointing_sources),))

    # Remove sources in skip catalogue file, if provided
    if opts.skip_catalogue is not None and os.path.exists(opts.skip_catalogue):
        skip_sources = katpoint.Catalogue(file(opts.skip_catalogue))
        for target in skip_sources:
            pointing_sources.remove(target.name)
        user_logger.info("After skipping, %d targets are left" % (len(pointing_sources.targets),))

    # Quit early if there are no sources to observe
    if len(pointing_sources.targets) == 0:
        user_logger.warning("Empty point source catalogue or all targets are skipped")
    elif len(pointing_sources.filter(el_limit_deg=opts.horizon).targets) == 0:
        user_logger.warning("No targets in catalogue are currently visible - please re-run the script later")
    else:
        # Observed targets will be written back to catalogue file, or into the void
        skip_file = file(opts.skip_catalogue, "a") if opts.skip_catalogue is not None and not opts.dry_run else StringIO()
        if opts.quick :opts.dump_rate = 2.0
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            skip_file.write("# Record of targets observed on %s by %s\n" % (datetime.datetime.now(), opts.observer))
            while keep_going:
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in pointing_sources.iterfilter(el_limit_deg=opts.horizon):
                    session.label('raster')
                    # Do different raster scan on strong and weak targets
                    if not opts.quick :
                        if (opts.source_strength == 'strong' or (opts.source_strength == 'auto' and target.flux_density(opts.centre_freq) > 10.0)):
                            session.raster_scan(target, num_scans=5, scan_duration=30, scan_extent=6.0,
                                                scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation,
                                                projection=opts.projection)
                        else :
                            session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=4.0,
                                                scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation,
                                                projection=opts.projection)
                    else :
                        session.raster_scan(target, num_scans=3, scan_duration=15, scan_extent=5.0,
                                            scan_spacing=0.5, scan_in_azimuth=not opts.scan_in_elevation,
                                            projection=opts.projection)

                    targets_observed.append(target.name)
                    skip_file.write(target.description + "\n")
                    # The default is to do only one iteration through source list
                    if opts.min_time <= 0.0:
                        keep_going = False
                    # If the time is up, stop immediately
                    elif time.time() - start_time >= opts.min_time:
                        keep_going = False
                        break
                if len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))

        skip_file.close()
