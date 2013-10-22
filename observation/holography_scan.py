#!/usr/bin/python
# Template script

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

# Import script helper functions from observe.py
from katcorelib import standard_script_options, verify_and_connect, collect_targets, \
                             start_session, user_logger, ant_array
import numpy as np

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='This script performs a holography scan on one or more targets. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a raster scan on the target. Note also some '
                                             '**required** options below.')
# Add experiment-specific options
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('-k', '--num-scans', type='int', default=1,
                  help='Number of scans across target, usually an odd number (default=%default) ')
parser.add_option('-t', '--scan-duration', type='float', default=30.0,
                  help='Minimum duration of each scan across target, in seconds (default=%default) ')
parser.add_option('-l', '--scan-extent', type='float', default=5.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('-m', '--scan-spacing', type='float', default=0.0,
                  help='Separation between scans, in degrees (default=%default)')
parser.add_option('-e', '--scan-in-elevation', action='store_true', default=False,
                  help='Scan in elevation rather than in azimuth (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Holography scan')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Create start and end positions of each scan, based on scan parameters
scan_levels = np.arange(-(opts.num_scans // 2), opts.num_scans // 2 + 1)
scanning_coord = (opts.scan_extent / 2.0) * (-1) ** scan_levels
stepping_coord = opts.scan_spacing * scan_levels
# Flip sign of elevation offsets to ensure that the first scan always starts at the top left of target
scan_starts = zip(stepping_coord, -scanning_coord) if opts.scan_in_elevation else zip(scanning_coord, -stepping_coord)
scan_ends = zip(stepping_coord, scanning_coord) if opts.scan_in_elevation else zip(-scanning_coord, -stepping_coord)

# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)

    # Initialise a capturing session (which typically opens an HDF5 file)
    with start_session(kat, **vars(opts)) as session:
        # Use the command-line options to set up the system
        session.standard_setup(**vars(opts))
        all_ants = session.ants
        # Form scan antenna subarray (or pick the first antenna as the default scanning antenna)
        scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else session.ants[0], 'scan_ants')
        # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
        nd_params = session.nd_params
        session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
        session.capture_start()
        user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in session.ants]),))

        for target in targets:
            # The entire sequence of commands on the same target forms a single compound scan
            session.label('holo')
            user_logger.info("Initiating holography scan (%d %g-second scans extending %g degrees) on target '%s'"
                             % (opts.num_scans, opts.scan_duration, opts.scan_extent, target.name))
            # Slew all antennas onto the target (don't spend any more time on it though)
            session.track(target, duration=0, announce=False)
            # Provide opportunity for noise diode to fire on all antennas
            session.fire_noise_diode(announce=False, **nd_params)
            # Perform multiple scans across the target with the scan antennas only
            for scan_index, (start, end) in enumerate(zip(scan_starts, scan_ends)):
                session.ants = scan_ants
                user_logger.info("Using scan antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                session.scan(target, duration=opts.scan_duration, start=start, end=end,
                             index=scan_index, projection=opts.projection)
                # Provide opportunity for noise diode to fire on all antennas
                session.ants = all_ants
                user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                session.fire_noise_diode(announce=False, **nd_params)
