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


# Raster scan styles
styles = {
  # Sources at L-band, default
  'strong': dict(dump_rate=1.0, num_scans=5, scan_duration=30, scan_extent=6.0, scan_spacing=0.25),
  # Weaker sources at L-band, requiring more time on source
  'weak': dict(dump_rate=1.0, num_scans=5, scan_duration=60, scan_extent=4.0, scan_spacing=0.25),
  # Either 'strong' or 'weak', only evaluated once the source flux is known
  'auto': dict(),
  # Sources at L-band, quick check
  'quick': dict(dump_rate=2.0, num_scans=3, scan_duration=15, scan_extent=5.0, scan_spacing=0.5),
  # Sources at Ku-band
  'fine': dict(dump_rate=1.0, num_scans=5, scan_duration=60, scan_extent=1.0, scan_spacing=4./60.),
  # Sources at Ku-band, wider search area
  'search-fine': dict(dump_rate=4.0, num_scans=25, scan_duration=30, scan_extent=2.0, scan_spacing=5./60.),
}

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
parser.add_option('-z', '--skip-catalogue',
                  help="Name of file containing catalogue of targets to skip (updated with observed targets). "
                       "The default is not to skip any targets.")
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
parser.add_option('--style', type='choice', default='auto', choices=styles.keys(),
                  help="Raster scan style determining number of scans, scan duration, scan extent, "
                       "scan spacing and default dump rate. Most options are tuned for L-band, while "
                       "the 'fine' options are meant for Ku-band. The 'auto' option is either 'strong' "
                       "or 'weak', depending on the source flux. The available styles are: %s" % (styles,))
# Deprecated options
parser.add_option('--source-strength', type='choice', default='none', choices=('strong', 'weak', 'auto', 'none'),
                  help="Scanning strategy based on source strength, one of 'strong', 'weak' or 'auto' (default). "
                       "Auto is based on flux density specified in catalogue. *DEPRECATED*")
parser.add_option( '--quick', action="store_true" , default=False,
                  help='Do a quick "Zorro" type scan, which is 3 5-degree scans lasting 15 seconds each and '
                       'spaced 0.5 degrees apart with 2 Hz dump rate. *DEPRECATED*')
parser.add_option( '--fine', action="store_true" , default=False,
                  help='Do a fine-grained point scan with an extent of 1 degree and a duration of 60 seconds.'
                  'This is for Ku-band observations where the beam is 8 arcmin. *DEPRECATED*')
parser.add_option( '--search-fine', action="store_true" , default=False,
                  help='Do a fine-grained point scan with an extent of 2 degrees and a duration of 30 seconds.'
                  'This is for Ku-band observations where the beam is 8 arcmin. *DEPRECATED*')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source scan', dump_rate=None)
# Parse the command line
opts, args = parser.parse_args()

# Handle deprecated options
if opts.quick:
    user_logger.warning("The --quick option is deprecated, use --style=quick instead")
    opts.style == 'quick'
elif opts.fine:
    user_logger.warning("The --fine option is deprecated, use --style=fine instead")
    opts.style == 'fine'
elif opts.search_fine:
    user_logger.warning("The --search-fine option is deprecated, use --style=search-fine instead")
    opts.style == 'search-fine'
elif opts.source_strength != 'none':
    user_logger.warning("The --source-strength=blah option is deprecated, use --style=blah instead")
    opts.style = opts.source_strength
# If dump rate is not specified, take it from the raster scan style
if not opts.dump_rate:
    opts.dump_rate = styles[opts.style].pop('dump_rate', 1.0)

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
                if session.dbe.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
                if session.dbe.req.auto_delay('off'):
                    user_logger.info("Turning off delay tracking.")
                else:
                    user_logger.error('Unable to turn off delay tracking.')
                #if session.dbe.req.zero_delay():
                #    user_logger.info("Zeroed the delay values.")
                #else:
                #    user_logger.error('Unable to zero delay values.')
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
                    session.label('raster')
                    az, el = target.azel()
                    user_logger.info("Doing scan of target %r with current azel (%s, %s)" % (target.description, az, el))
                    if opts.style == 'auto':
                        style = 'strong' if target.flux_density(opts.centre_freq) > 10.0 else 'weak'
                    else:
                        style = opts.style
                    kwargs = styles[style]
                    # Get rid of keyword arguments that are not meant for session.raster_scan
                    kwargs.pop('dump_rate', None)
                    session.raster_scan(target, scan_in_azimuth=not opts.scan_in_elevation,
                                        projection=opts.projection, **kwargs)
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
