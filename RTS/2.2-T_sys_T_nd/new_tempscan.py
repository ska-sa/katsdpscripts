#!/usr/bin/python
# Perform mini (Zorro) raster scans across (point) sources from a catalogue
# for pointing model fits and gain curve calculation.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os.path
from cStringIO import StringIO
import datetime
import time

from katcorelib import collect_targets, standard_script_options, verify_and_connect, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(
    usage="%prog [options] [<'target/catalogue'> ...]",
    description="Perform mini (Zorro) raster scans across (point) sources for pointing "
    "model fits and gain curve calculation. Use the specified target(s) and "
    "catalogue(s) or the default. This script is aimed at fast scans across "
    "a large range of sources. This script does 2 scans of the target one normal "
    "and the other with the nd on Some options are **required**.")
# Add experiment-specific options
parser.add_option('-e', '--scan-in-elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth (default=%default)")
parser.add_option('-m', '--min-time', type="float", default=-1.0,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.add_option('-z', '--skip-catalogue',
                  help="Name of file containing catalogue of targets to skip (updated with observed targets). "
                       "The default is not to skip any targets.")
parser.add_option( '--quick', action="store_true" , default=False,
                  help='Do a quick "Zorro" type scan, which is 3 5-degree scans lasting 15 seconds each and '
                       'spaced 0.5 degrees apart with 2 Hz dump rate.')

parser.add_option('--project-id',
                    help='Project ID code the observation (**required**) This is a required option')

parser.add_option('--no-delays', action="store_true", default=False,
                    help='Do not use delay tracking, and zero delays')

parser.set_defaults(description='Point source scan')
# Parse the command line
opts, args = parser.parse_args()
if opts.quick:
    opts.dump_rate = 2.0
diode = 'coupler'
if not hasattr(opts, 'project_id') or opts.project_id is None:
    raise ValueError('Please specify the Project id code via the --project_id option '
                     '(yes, this is a non-optional option...)')

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
        user_logger.info("After skipping, %d targets are left" % (len(pointing_sources),))

    # Quit early if there are no sources to observe
    if len(pointing_sources) == 0:
        user_logger.warning("Empty point source catalogue or all targets are skipped")
    elif len(pointing_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Observed targets will be written back to catalogue file, or into the void
        skip_file = file(opts.skip_catalogue, "a") \
                    if opts.skip_catalogue is not None and not kat.dry_run else StringIO()
        with start_session(kat, **vars(opts)) as session:
            if not opts.no_delays and not kat.dry_run :
                if session.dbe.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
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
            session.nd_params =  {'diode' : 'coupler', 'on' : 0., 'off' : 0., 'period' : -1.}
            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            skip_file.write("# Record of targets observed on %s by %s\n" % (datetime.datetime.now(), opts.observer))
            def  rscan(target):
                session.label('raster')
                if not opts.quick:
                    session.raster_scan(target, num_scans=5, scan_duration=30, scan_extent=6.0,
                                        scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation,
                                        projection=opts.projection)
                else:
                    session.raster_scan(target, num_scans=3, scan_duration=15, scan_extent=5.0,
                                        scan_spacing=0.5, scan_in_azimuth=not opts.scan_in_elevation,
                                        projection=opts.projection)

            while keep_going:
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in pointing_sources.iterfilter(el_limit_deg=opts.horizon):
                    if not np.isnan(target.flux_density(opts.centre_freq)): # check flux model
                        ants.req.rfe3_rfe15_noise_source_on(diode, 1, time.time()+2.0, 0)
                        time.time.sleep(2)
                        rscan(target)
                        ants.req.rfe3_rfe15_noise_source_on(diode, 0, time.time()+2.0, 0)
                        time.sleep(2)
                        rscan(target)

                        targets_observed.append(target.name)
                        skip_file.write(target.description + "\n")
                    # The default is to do only one iteration through source list
                    if opts.min_time <= 0.0:
                        keep_going = False
                    # If the time is up, stop immediately
                    elif time.time() - start_time >= opts.min_time:
                        keep_going = False
                        break
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))

        skip_file.close()
