#!/usr/bin/env python
"""Test observation for Mario Santos for noise diode alignment check.

Set the noise diode period to a multiple of the correlator integration time.
Then scan over a target.

"""

import time

import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)


# Set up standard script options
description = ('This script performs a scan on one or more targets. '
               'The noise diode can be turned on before the scans. ')
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> "
                                       "[<'target/catalogue'> ...]",
                                 description=description)
# Add experiment-specific options
parser.add_option('-k', '--num-scans', type='int', default=1,
                  help='Number of scans across target, usually an '
                       'odd number (default=%default) ')
parser.add_option('-t', '--scan-duration', type='float', default=30.0,
                  help='Minimum duration of each scan across target, '
                       'in seconds (default=%default) ')
parser.add_option('-l', '--scan-extent', type='float', default=5.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('-m', '--scan-spacing', type='float', default=0.0,
                  help='Separation between scans, in degrees (default=%default)')
parser.add_option('-e', '--scan-in-elevation', action='store_true', default=False,
                  help='Scan in elevation rather than in azimuth (default=%default)')
parser.add_option('--noise-source-cycle-length', type='float', default=0.0,
                  help="Approximate period, in seconds, for the noise diode pattern "
                       "on all antennas (0 to disable) (default=%default)")
parser.add_option('--noise-source-on-time', type='float', default=0.0,
                  help="Approximate on-time, in seconds, for the noise diode pattern "
                       "on all antennas (0 to disable) (default=%default)")

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description="Noise diode test scan")
# Parse the command line
opts, args = parser.parse_args()
# Disable noise diode in session (to prevent it firing on scan antennas)
opts.nd_params = 'off'

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via "
                     "name ('Cygnus A'), description ('azel, 20, 30') or "
                     "catalogue file name ('sources.csv')")

# Check basic command-line options and obtain a kat object connected to the
# appropriate system
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)
    # Initialise a capturing session
    with start_session(kat, **vars(opts)) as session:
        # Use the command-line options to set up the system
        session.standard_setup(**vars(opts))

        # Manually enable noise diode
        if opts.noise_source_cycle_length and opts.noise_source_on_time:
            # Calculate best parameters that are a multiple of the correlator
            # dump period (aka integration time)
            dump_period = session.cbf.correlator.sensor.int_time.get_value()
            cycle_length = (round(opts.noise_source_cycle_length / dump_period)
                            * dump_period)
            on_time = round(opts.noise_source_on_time / dump_period) * dump_period
            on_fraction = on_time / cycle_length
            user_logger.info('Setting noise-source pattern to %.15g s on '
                             'with %.15g s period, (%.15g fraction on)',
                             on_time, cycle_length, on_fraction)
            # Try to trigger noise diodes on all antennas in array simultaneously.
            # - use integer second boundary as that is most likely be an exact
            #   time that DMC can execute at, and also fit a unix epoch time
            #   into a double precision float accurately
            # - add a 2 second lead time so enough time for all digitisers
            #   to be set up
            timestamp = np.ceil(time.time()) + 2.0
            user_logger.info('Set all noise diodes with timestamp %s (%s)',
                             timestamp, time.ctime(timestamp))
            replies = kat.ants.req.dig_noise_source(timestamp, on_fraction,
                                                    cycle_length)
            if not kat.dry_run:
                user_logger.info('Actual settings from digitisers:')
                user_logger.info('  %-4s %-16s %-16s %-16s',
                                 'ant', 'timestamp', 'on-fraction', 'cycle-length')
                for ant in sorted(replies):
                    reply, informs = replies[ant]
                    actual_time, actual_on_frac, actual_cycle = reply[1:4]
                    user_logger.info('  %-4s %16.4f %16.14f %16.14f',
                                     ant, actual_time, actual_on_frac, actual_cycle)
        else:
            user_logger.info('The noise-source is disabled')

        session.capture_start()
        for target in targets:
            # Perform multiple scans across the target
            session.label('raster')
            session.raster_scan(target, num_scans=opts.num_scans,
                                scan_duration=opts.scan_duration,
                                scan_extent=opts.scan_extent,
                                scan_spacing=opts.scan_spacing,
                                scan_in_azimuth=not opts.scan_in_elevation,
                                projection=opts.projection)

        # switch noise-source pattern off
        if opts.noise_source_cycle_length and opts.noise_source_on_time:
            user_logger.info('Ending noise source pattern')
            kat.ants.req.dig_noise_source('now', 0)
