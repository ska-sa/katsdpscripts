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

# temporary hack to ensure antenna does not timeout for the moment
def bad_ar1_alt_hack(target, duration, limit=88.):
    import numpy
    [az, el] = target.azel()
    delta_transit = duration*(15./3600.)
    if (numpy.rad2deg(float(el))+delta_transit+delta_transit) > limit: return True
    return False

# Set up standard script options
parser = standard_script_options(
    usage="%prog [options] [<'target/catalogue'> ...]",
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
                  help='Do a quick "Zorro" type scan, which is 3 5-degree scans lasting 15 seconds each and '
                       'spaced 0.5 degrees apart with 2 Hz dump rate.')
parser.add_option( '--fine', action="store_true" , default=False,
                  help='Do a fine grained pointscan with an extent of 1 degree and a duration of 60 seconds.'
                  'The intention of this is for use in Ku-band obsevations where the beam is 8 arc-min .')

parser.add_option( '--search-fine', action="store_true" , default=False,
                  help='Do a fine grained pointscan with an extent of 2 degree and a duration of 60 seconds.'
                  'The intention of this is for use in Ku-band obsevations where the beam is 8 arc-min .')

parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')

parser.set_defaults(description='Point source scan')
# Parse the command line
opts, args = parser.parse_args()

if opts.quick:
    opts.dump_rate = 2.0

with verify_and_connect(opts) as kat:
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")

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

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            skip_file.write("# Record of targets observed on %s by %s\n" % (datetime.datetime.now(), opts.observer))
            while keep_going:
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in pointing_sources.iterfilter(el_limit_deg=opts.horizon+7.0):
# RvR -- Very bad hack to keep from tracking above 89deg until AR1 AP can handle out of range values better
		    if bad_ar1_alt_hack(target, 60.): continue
# RvR -- Very bad hack to keep from tracking above 89deg until AR1 AP can handle out of range values better

                    session.label('raster')
                    # Do different raster scan on strong and weak targets
                    if not opts.quick and not opts.fine:
                        if opts.source_strength == 'strong' or \
                           (opts.source_strength == 'auto' and target.flux_density(opts.centre_freq) > 10.0):
                            user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
                            session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=6.0,
                                                scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation,
                                                projection=opts.projection)
                        else:
                            session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=4.0,
                                                scan_spacing=0.25, scan_in_azimuth=not opts.scan_in_elevation,
                                                projection=opts.projection)
                            user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
                    else:  # The branch for Quick and Fine scans
                        if opts.quick:
                            user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
                            session.raster_scan(target, num_scans=3, scan_duration=30, scan_extent=5.0,
                                            scan_spacing=0.5, scan_in_azimuth=not opts.scan_in_elevation,
                                            projection=opts.projection)
                        if opts.fine:
                            user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
                            session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=1.0,
                                            scan_spacing=4./60., scan_in_azimuth=not opts.scan_in_elevation,
                                            projection=opts.projection)
                        else: #if opts.search_fine:
                            user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
                            session.raster_scan(target, num_scans=9, scan_duration=60, scan_extent=2.0,
                                            scan_spacing=5./60., scan_in_azimuth=not opts.scan_in_elevation,
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
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))

        skip_file.close()
