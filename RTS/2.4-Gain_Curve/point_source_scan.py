#!/usr/bin/python
# Perform mini (Zorro) raster scans across (point) sources from a catalogue
# for pointing model fits and gain curve calculation.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os.path
from cStringIO import StringIO
import datetime
import time
import numpy as np

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
    'search-fine': dict(dump_rate=1.0, num_scans=16, scan_duration=16, scan_extent=32./60., scan_spacing=2./60.),
    # Standard for MeerKAT single dish pointing, covering null-to-null at centre frequency (+-1.3*HPBW), resolution ~HPBW/3
    'uhf': dict(dump_rate=1.0, num_scans=9, scan_duration=60, scan_extent=5.5, scan_spacing=5.5/8),
    'l': dict(dump_rate=1.0, num_scans=9, scan_duration=40, scan_extent=3.5, scan_spacing=3.5/8),
    's': dict(dump_rate=1.0, num_scans=9, scan_duration=30, scan_extent=2.0, scan_spacing=2.0/8),
    'ku': dict(dump_rate=1.0, num_scans=9, scan_duration=20, scan_extent=0.5, scan_spacing=0.5/8),
    # Standard for SKA Dish
    'skab1': dict(dump_rate=1.0, num_scans=9, scan_duration=60, scan_extent=6.0, scan_spacing=6.0/8),
    'skaku': dict(dump_rate=1.0, num_scans=9, scan_duration=24, scan_extent=0.36, scan_spacing=0.36/8),
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
# Set default value for any option (both standard and experiment-specific options)
parser.add_option('-z', '--skip-catalogue',
                  help="Name of file containing catalogue of targets to skip (updated with observed targets). "
                       "The default is not to skip any targets.")
parser.add_option('--style', type='choice', default='auto', choices=styles.keys(),
                  help="Raster scan style determining number of scans, scan duration, scan extent, "
                       "scan spacing and default dump rate. Most options are tuned for L-band, while "
                       "the 'fine' options are meant for Ku-band. The 'auto' option is either 'strong' "
                       "or 'weak', depending on the source flux. The available styles are: %s" % (styles,))
# Deprecated options
parser.add_option('--source-strength', type='choice', default='none', choices=('strong', 'weak', 'auto', 'none'),
                  help="Scanning strategy based on source strength, one of 'strong', 'weak' or 'auto' (default). "
                       "Auto is based on flux density specified in catalogue. *DEPRECATED*")
parser.add_option('--quick', action="store_true", default=False,
                  help='Do a quick "Zorro" type scan, which is 3 5-degree scans lasting 15 seconds each and '
                       'spaced 0.5 degrees apart with 2 Hz dump rate. *DEPRECATED*')
parser.add_option('--fine', action="store_true", default=False,
                  help='Do a fine-grained point scan with an extent of 1 degree and a duration of 60 seconds.'
                  'This is for Ku-band observations where the beam is 8 arcmin. *DEPRECATED*')
parser.add_option('--search-fine', action="store_true", default=False,
                  help='Do a fine grained pointscan with an extent of 2 degree and a duration of 60 seconds.'
                  'The intention of this is for use in Ku-band obsevations where the beam is 8 arc-min .*DEPRECATED*')

parser.set_defaults(description='Point source scan',dump_rate=None)


def filter_separation(catalogue, T_observed, antenna=None, separation_deg=1, sunmoon_separation_deg=10):
    """ Removes targets from the supplied catalogue which are within the specified distance from others or either the Sun or Moon.

        @param catalogue: [katpoint.Catalogue]
        @param T_observed: UTC timestamp, seconds since epoch [sec].
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str]
        @param separation_deg: eliminate targets closer together than this (default 1) [deg]
        @param sunmoon_separation_deg: omit targets that are closer than this distance from Sun & Moon (default 10) [deg]
        @return: katpoint.Catalogue (a filtered copy of input catalogue)
    """
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    targets = list(catalogue.targets)
    avoid_sol = [katpoint.Target('%s, special'%n) for n in ['Sun','Moon']] if (sunmoon_separation_deg>0) else []

    separation_rad = separation_deg*np.pi/180.
    sunmoon_separation_rad = sunmoon_separation_deg*np.pi/180.

    # Remove targets that are too close together (unfortunately also duplicated pairs)
    overlap = np.zeros(len(targets), float)
    for i in range(len(targets)-1):
        t_i = targets[i]
        sep = [(t_i.separation(targets[j], T_observed, antenna) < separation_rad) for j in range(i+1, len(targets))]
        sep = np.r_[np.any(sep), sep] # Flag t_j too, if overlapped
        overlap[i:] += np.asarray(sep, int)
        # Check for t_i overlapping with solar system bodies
        sep = [(t_i.separation(j, T_observed, antenna) < sunmoon_separation_rad) for j in avoid_sol]
        if np.any(sep):
            user_logger.info("%s appears within %g deg from %s"%(t_i, sunmoon_separation_deg, np.compress(sep,avoid_sol)))
            overlap[i] += 1
    if np.any(overlap > 0):
        user_logger.info("Planning drops the following due to being within %g deg away from other targets:\n%s"%(separation_deg, np.compress(overlap>0,targets)))
        targets = list(np.compress(overlap==0, targets))

    filtered = katpoint.Catalogue(targets, antenna=antenna)
    return filtered

def plan_targets(catalogue, T_start, t_observe, dAdt=1.8, antenna=None, el_limit_deg=20):
    """ Generates a "nearest-neighbour" sequence of targets to observe, starting at the specified time.
        This does not consider behaviour around the azimuth wrap zone.

        @param catalogue: [katpoint.Catalogue]
        @param T_start: UTC timestamp, seconds since epoch [sec].
        @param t_observe: duration of an observation per target [sec]
        @param dAdt: angular rate when slewing (default 1.8) [deg/sec]
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str or antenna proxy]
        @param el_limit_deg: observation elevation limit (default 20) [deg]
        @return: [list of Targets], expected duration in seconds
    """
    # If it's an "antenna proxy, use current coordinates as starting point
    try:
        az0, el0 = antenna.sensor.pos_actual_scan_azim.get_value(), antenna.sensor.pos_actual_scan_elev.get_value()
        antenna = antenna.sensor.observer.value
    except: # No "live" coordinates so start from zenith
        az0, el0 = 0, 90
    start_pos = katpoint.construct_azel_target(az0*np.pi/180., el0*np.pi/180.)

    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna

    todo = list(catalogue.targets)
    done = []
    T = T_start # Absolute time
    available = catalogue.filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
    next_tgt = available.closest_to(start_pos, T, antenna)[0] if (len(available.targets) > 0) else None
    while (next_tgt is not None):
        # Observe
        next_tgt.antenna = antenna
        done.append(next_tgt)
        todo.pop(todo.index(next_tgt))
        T += t_observe
        # Find next visible target
        available = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
        next_tgt, dGC = available.closest_to(done[-1], T, antenna)
        # Slew to next
        if next_tgt:
            T += dGC * dAdt
    return done, (T-T_start)


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
    if len(args) > 0:
        # Load pointing calibrator catalogues and command line targets
        pointing_sources = collect_targets(kat, args)
    else:
        # Default catalogue contains the radec sources in the standard kat database
        pointing_sources = kat.sources.filter(tags='radec')
        user_logger.info("No valid targets specified, loaded default catalogue with %d targets" %
                         (len(pointing_sources),))

    # Remove sources that crowd too closely
    pointing_sources = filter_separation(pointing_sources, time.time(), kat.sources.antenna, separation_deg=1, sunmoon_separation_deg=10)

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
            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            skip_file.write("# Record of targets observed on %s by %s\n" % (datetime.datetime.now(), opts.observer))
            while keep_going:
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one from the nearest neighbour plan
                raster_params = styles[opts.style] # TODO: Fails for 'auto'!
                raster_duration = raster_params["num_scans"] * (raster_params["scan_duration"]+5) # Incl. buffer between scans. TODO: Ignoring ND
                for target in plan_targets(pointing_sources, start_time, t_observe=raster_duration,
                                           antenna=kat.ants[0], el_limit_deg=opts.horizon+5.0)[0]:
                    session.label('raster')
                    az, el = target.azel()
                    user_logger.info("Scanning target %r with current azel (%s, %s)" % (target.description, az, el))
                    if opts.style == 'auto':
                        style = 'strong' if target.flux_density(session.get_centre_freq()) > 10.0 else 'weak'
                    else:
                        style = opts.style
                    kwargs = styles[style]
                    # Get rid of keyword arguments that are not meant for session.raster_scan
                    kwargs.pop('dump_rate', None)
                    if session.raster_scan(target, scan_in_azimuth=not opts.scan_in_elevation,
                                           projection=opts.projection, **kwargs):
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
