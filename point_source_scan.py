#!/usr/bin/python
# Perform mini (Zorro) raster scans across (point) sources from a catalogue for pointing model fits and gain curve calculation.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, ant_array, CaptureSession
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
parser.add_option('-p', '--print_only', action="store_true", default=False,
                  help="Do not actually observe, but display which sources will be scanned, "+
                       "plus predicted end time (default=%default)")
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

    start_time = katpoint.Timestamp()
    targets_observed = []

    if opts.print_only:
        current_time = katpoint.Timestamp(start_time)
        # Find out where first antenna is currently pointing (assume all antennas point there)
        ants = ant_array(kat, opts.ants)
        az = ants.devs[0].sensor.pos_actual_scan_azim.get_value()
        el = ants.devs[0].sensor.pos_actual_scan_elev.get_value()
        prev_target = katpoint.construct_azel_target(az, el)
        # Keep going until the time is up
        keep_going, compscan = True, 0
        while keep_going:
            # Iterate through baseline sources that are up
            for target in pointing_sources.iterfilter(el_limit_deg=5, timestamp=current_time):
                print "At about %s, antennas will start slewing to '%s'" % (current_time.local(), target.name)
                # Assume 1 deg/s slew rate on average -> add time to slew from previous target to new one
                current_time += 1.0 * katpoint.rad2deg(target.separation(prev_target))
                print "At about %s, point source scan (compound scan %d) will start on '%s'" % \
                      (current_time.local(), compscan, target.name)
                # Standard raster scan is 5 scans of 30 seconds each, with 4 slews of about 2 seconds in between scans,
                # followed by 20 seconds of noise diode on/off. Also allow one second of overhead per scan.
                if target.flux_density(opts.centre_freq) > 25.0:
                    current_time += 5 * 30.0 + 4 * 2.0 + 20.0 + 10 * 1.0
                else:
                    # Weaker targets get longer-duration scans
                    current_time += 5 * 60.0 + 4 * 2.0 + 20.0 + 10 * 1.0
                targets_observed.append(target.name)
                prev_target = target
                compscan += 1
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif current_time - start_time >= opts.min_time:
                    keep_going = False
                    break
        print "Experiment to finish at about", current_time.local()

    else:
        # The real experiment: Create a data capturing session with the selected sub-array of antennas
        with CaptureSession(kat, **vars(opts)) as session:
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
                    elif katpoint.Timestamp() - start_time >= opts.min_time:
                        keep_going = False
                        break

    print "Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed)))
