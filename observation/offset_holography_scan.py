#!/usr/bin/env python
# Perform raster holography scans on specified target(s) where there is a
# significant difference between scanning region and target.

# Import script helper functions from observe.py
from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger, ant_array)
import numpy as np


# Set up standard script options
description = 'This script performs a holography scan on one or more targets. ' \
              'All the antennas initially track the target, whereafter a ' \
              'subset of the antennas (the "scan antennas" specified by the ' \
              '--scan-ants option) perform a radial raster scan on the ' \
              'target. Note also some **required** options below.'
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description=description)
# Add experiment-specific options
parser.add_option('-b', '--scan-ants',
                  help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('-k', '--num-scans', type='int', default=3,
                  help='Number of scans across target (default=%default)')
parser.add_option('--num-grouped-scans', type='int', default=4,
                  help='Number of scans grouped before slewing to target (default=%default)')
parser.add_option('-t', '--scan-duration', type='float', default=20.0,
                  help='Minimum duration of each scan across target, in seconds (default=%default)')
parser.add_option('--tracktime', type='float', default=0.0,
                  help='Scanning antenna tracks target this long when passing '
                       'over target, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=2.0,
                  help='Length of each scan, azimuthal, in degrees (default=%default)')
parser.add_option('--scan-el-extent', type='float', default=2.0,
                  help='Length of each scan, elevation, in degrees (default=%default)')
parser.add_option('--scan-el-offset', type='float', default=70.0,
                  help='Elevation offset from target to scanning area center, in degrees (default=%default)')
parser.add_option('--scan-az-offset', type='float', default=0.0,
                  help='Azimuthal offset from target to scanning area center, in degrees (default=%default)')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of beam measurement cycles to complete (default=%default)')
parser.add_option('--elevation-scan', action="store_true", default=False,
                  help='Perform elevation rather than azimuthal scan')
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
        track_ants = ant_array(kat, [ant for ant in all_ants if ant not in scan_ants], 'track_ants')
        # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
        nd_params = session.nd_params
        session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
        session.capture_start()

        targets_observed = []
        for cycle in range(opts.num_cycles):
            for target in targets.iterfilter(el_limit_deg=opts.horizon + opts.scan_extent / 2.):
                # The entire sequence of commands on the same target forms a single compound scan
                session.label('holo')
                user_logger.info("Initiating holography cycle %d of %d "
                                 "(%d %g-second scans extending %g degrees) on target '%s'",
                                 cycle + 1, opts.num_cycles, opts.num_scans,
                                 opts.scan_duration, opts.scan_extent, target.name)
                user_logger.info("Using all antennas: %s",
                                 ' '.join([ant.name for ant in session.ants]))
                # Slew all antennas onto the target
                # spend extra 3 seconds in beginning
                session.track(target, duration=3.0 + opts.tracktime, announce=False)
                # Provide opportunity for noise diode to fire on all antennas
                session.fire_noise_diode(announce=False, **nd_params)
                # Perform multiple scans across the target at various angles with the scan antennas only
                if (opts.elevation_scan):
                    for scan_index, azangle in enumerate(np.linspace(-opts.scan_extent / 2.,
                                                                     opts.scan_extent / 2.0, opts.num_scans)):
                        session.ants = scan_ants
                        user_logger.info("Using scan antennas: %s",
                                         ' '.join([ant.name for ant in session.ants]))
                        # Perform radial scan at specified angle across target
                        offsetaz= azangle + opts.scan_az_offset
                        offsetel0=(-1)**((scan_index)%2)*opts.scan_extent / 2 + opts.scan_el_offset
                        offsetel1=(-1)**((scan_index+1)%2)*opts.scan_extent / 2 + opts.scan_el_offset
                        session.scan(target, duration=(opts.scan_duration-opts.tracktime)/2.0, start=[offsetaz,offsetel0], end=[offsetaz,offsetel1], index=scan_index,
                                         projection=opts.projection, announce=False)

                        if (scan_index%opts.num_grouped_scans==opts.num_grouped_scans-1):
                            user_logger.info("Slewing scan antennas to target: %s",
                                             ' '.join([ant.name for ant in session.ants]))
                            session.scan(target, duration=opts.tracktime, start=[0.0,0.0], end=[0.0,0.0], index=scan_index,
                                     projection=opts.projection, announce=False)
                            session.track(target, duration=opts.tracktime, announce=False)                        
                        # Ensure that tracking antennas are still on target (i.e. collect antennas that strayed)
                        session.ants = track_ants
                        user_logger.info("Using track antennas: %s",
                                         ' '.join([ant.name for ant in session.ants]))
                        session.track(target, duration=0, announce=False)
                        # Provide opportunity for noise diode to fire on all antennas
                        session.ants = all_ants
                        user_logger.info("Using all antennas: %s",
                                         ' '.join([ant.name for ant in session.ants]))
                        session.fire_noise_diode(announce=False, **nd_params)
                else:#azimuthal scan
                    for scan_index, elangle in enumerate(np.linspace(-opts.scan_el_extent / 2., opts.scan_el_extent / 2.0, opts.num_scans)):
                        session.ants = scan_ants
                        user_logger.info("Using scan antennas: %s",
                                         ' '.join([ant.name for ant in session.ants]))
                        # Perform radial scan at specified angle across target
                        offsetaz0= (-1)**(scan_index%2)*opts.scan_extent / 2 + opts.scan_az_offset
                        offsetaz1= (-1)**((scan_index+1)%2)*opts.scan_extent / 2 + opts.scan_az_offset
                        offsetel=elangle+opts.scan_el_offset
                        session.scan(target, duration=(opts.scan_duration-opts.tracktime)/2.0, start=[offsetaz0,offsetel], end=[offsetaz1,offsetel], index=scan_index,
                                         projection=opts.projection, announce=False)

                        if (scan_index%opts.num_grouped_scans==opts.num_grouped_scans-1):
                            user_logger.info("Slewing scan antennas to target: %s",
                                             ' '.join([ant.name for ant in session.ants]))
                            session.scan(target, duration=opts.tracktime, start=[0.0,0.0], end=[0.0,0.0], index=scan_index,
                                     projection=opts.projection, announce=False)
                            session.track(target, duration=opts.tracktime, announce=False)                        
                        # Ensure that tracking antennas are still on target (i.e. collect antennas that strayed)
                        session.ants = track_ants
                        user_logger.info("Using track antennas: %s",
                                         ' '.join([ant.name for ant in session.ants]))
                        session.track(target, duration=0, announce=False)
                        # Provide opportunity for noise diode to fire on all antennas
                        session.ants = all_ants
                        user_logger.info("Using all antennas: %s",
                                         ' '.join([ant.name for ant in session.ants]))
                        session.fire_noise_diode(announce=False, **nd_params)
                
                session.ants = all_ants
                session.track(target, duration=3, announce=False)#spend extra 3 seconds at end
                
                targets_observed.append(target.name)
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
