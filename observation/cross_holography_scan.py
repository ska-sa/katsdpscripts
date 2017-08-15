#!/usr/bin/env python
# Perform radial holography scan on specified target(s). Mostly used for beam pattern mapping.

import time

# Import script helper functions from observe.py
from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger, ant_array)
import numpy as np


# Set up standard script options
description = 'This script performs a holography scan on one or more targets. ' \
              'All the antennas initially track the target, whereafter a ' \
              'subset of the antennas (the "scan antennas" specified by the ' \
              '--scan-ants option) perform a radial raster scan on the '\
              'target. Note also some **required** options below.'
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description=description)
# Add experiment-specific options
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('-k', '--num-scans', type='int', default=3,
                  help='Number of scans across target (default=%default)')
parser.add_option('-t', '--scan-duration', type='float', default=20.0,
                  help='Minimum duration of each scan across target, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=2.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=3600,
                  help='Minimum duration of obsevation (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Radial holography scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()


if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)
    # Initialise a capturing session (which typically opens an HDF5 file)
    with start_session(kat, **vars(opts)) as session:
        # Use the command-line options to set up the system
        session.standard_setup(**vars(opts))

        all_ants = session.ants
        # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
        scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else session.ants[0], 'scan_ants')
        # Assign rest of antennas to tracking antenna subarray
        # track_ants = ant_array(kat, [ant for ant in all_ants if ant not in scan_ants], 'track_ants')
        # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
        nd_params = session.nd_params
        session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
        session.capture_start()
        start_time = time.time()
        targets_observed = []
        keep_going = True
        while keep_going:
            targets_before_loop = len(targets_observed)

            for target in targets.iterfilter(el_limit_deg=opts.horizon + opts.scan_extent / 2.):
                # The entire sequence of commands on the same target forms a single compound scan
                session.label('holo')
                user_logger.info("Initiating holography scan (%d %g-second "
                                 "scans extending %g degrees) on target '%s'",
                                 opts.num_scans, opts.scan_duration,
                                 opts.scan_extent, target.name)
                user_logger.info("Using all antennas: %s",
                                 ' '.join([ant.name for ant in session.ants]))
                # Slew all antennas onto the target (don't spend any more time on it though)
                session.track(target, duration=0, announce=False)
                # Provide opportunity for noise diode to fire on all antennas
                session.fire_noise_diode(announce=False, **nd_params)
                # Perform multiple scans across the target at various angles with the scan antennas only
                for scan_index, angle in enumerate(np.arange(0., np.pi, np.pi / opts.num_scans)):
                    session.ants = scan_ants
                    user_logger.info("Using scan antennas: %s",
                                     ' '.join([ant.name for ant in session.ants]))
                    # Perform radial scan at specified angle across target
                    offset = np.array((np.cos(angle), -np.sin(angle))) * opts.scan_extent / 2. * (-1) ** scan_index
                    session.scan(target, duration=opts.scan_duration,
                                 start=-offset, end=offset, index=scan_index,
                                 projection=opts.projection, announce=False)
                    # Ensure that tracking antennas are still on target (i.e. collect antennas that strayed)
                    session.ants = all_ants
                    user_logger.info("Using track antennas: %s",
                                     ' '.join([ant.name for ant in session.ants]))
                    session.track(target, duration=0, announce=False)
                    # Provide opportunity for noise diode to fire on all antennas
                    session.ants = all_ants
                    user_logger.info("Using all antennas: %s",
                                     ' '.join([ant.name for ant in session.ants]))
                    session.fire_noise_diode(announce=False, **nd_params)
                if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
                    user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script",
                                        opts.max_duration)
                    keep_going = False
                    break
                targets_observed.append(target.name)
            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - "
                                    "stopping script instead of hanging around")
                keep_going = False

        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
