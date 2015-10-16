#!/usr/bin/python
# Perform radial holography scan on specified target(s). Mostly used for beam pattern mapping.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time

# Import script helper functions from observe.py
from katcorelib import standard_script_options, verify_and_connect, collect_targets, \
                       start_session, user_logger, ant_array
import numpy as np

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='This script performs a holography scan on one or more targets. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a radial raster scan on the target. Note also some '
                                             '**required** options below.')
# Add experiment-specific options
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('-k', '--num-scans', type='int', default=3,
                  help='Number of scans across target (default=%default)')
parser.add_option('-t', '--scan-duration', type='float', default=20.0,
                  help='Minimum duration of each scan across target, in seconds (default=%default)')
parser.add_option('--tracktime', type='float', default=0.0,
                  help='Scanning antenna tracks target this long when passing over target, in seconds. (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=2.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of beam measurement cycles to complete (default=%default)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
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

        all_ants = session.ants

        session.ants.req.mode('STOP')#necessary hack for now
        time.sleep(10)
        
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
            for target in targets.iterfilter(el_limit_deg=opts.horizon+(opts.scan_extent/2.0)):
                # The entire sequence of commands on the same target forms a single compound scan
                session.label('holo')
                user_logger.info("Initiating holography cycle %d of %d (%d %g-second scans extending %g degrees) on target '%s'"
                                 % (cycle,opts.num_cycles,opts.num_scans, opts.scan_duration, opts.scan_extent, target.name))
                user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                # Slew all antennas onto the target
                session.track(target, duration=3.0+opts.tracktime, announce=False)#spend extra 3 seconds in beginning
                
                # Provide opportunity for noise diode to fire on all antennas
                session.fire_noise_diode(announce=False, **nd_params)
                # Perform multiple scans across the target at various angles with the scan antennas only
                for scan_index, angle in enumerate(np.arange(0., 2.0*np.pi, 2.0*np.pi / opts.num_scans)):
                    session.ants = scan_ants
                    user_logger.info("Using scan antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                    # Perform radial scan at specified angle across target
                    dangle=2.0*np.pi/(opts.num_scans*2.0)
                    offset1 = np.array((np.cos(angle), -np.sin(angle))) * opts.scan_extent / 2.
                    offset2 = np.array((np.cos(angle+dangle), -np.sin(angle+dangle))) * opts.scan_extent / 2.
                    session.scan(target, duration=(opts.scan_duration-opts.tracktime)/2.0, start=[0,0], end=offset1, index=scan_index,
                                     projection=opts.projection, announce=False)
                    session.scan(target, duration=(opts.scan_duration-opts.tracktime)/2.0, start=offset2, end=[0,0], index=scan_index,
                                     projection=opts.projection, announce=False)
                    session.track(target, duration=opts.tracktime, announce=False)
                        
                    # Ensure that tracking antennas are still on target (i.e. collect antennas that strayed)
                    session.ants = track_ants
                    user_logger.info("Using track antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                    session.track(target, duration=0, announce=False)
                    # Provide opportunity for noise diode to fire on all antennas
                    session.ants = all_ants
                    user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                    session.fire_noise_diode(announce=False, **nd_params)
                    
                session.ants = all_ants
                session.track(target, duration=3, announce=False)#spend extra 3 seconds at end
                
                targets_observed.append(target.name)
        user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
        session.ants.req.mode('STOP')#necessary hack for now
