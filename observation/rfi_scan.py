#!/usr/bin/env python
# run a simple scan script to derive the horizon mask for KAT-7
# scan over constant elevation range but loop over azimuth

import time

from katcorelib import (standard_script_options, verify_and_connect,
                        start_session, user_logger)


# Set up standard script options
description = 'Perform a rfi scan with the MeerKAT. Scan over constant ' \
              'elevation with 3 scans at 15.1,21.1,27.1 degrees. This takes ' \
              'the form of 2x180 raster scans in opposite directions, with ' \
              '180 seconds per scan. There are non-optional options.(Antennas)'
parser = standard_script_options(usage="%prog [options]",
                                 description=description)
# Add experiment-specific options
parser.add_option('-m', '--min-duration', type="float", default=None,
                  help="The The minimum time to repeat the rfi scan over (default=%default)")
parser.add_option('--long', type="float", default=None,
                  help="Run the long version of the scan with length (default=%default) seconds")

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Basic RFI Scan', no_delays=True,horizon=15.)
# Parse the command line
opts, args = parser.parse_args()

opts.description = "Basic RFI Scan: %s" % (opts.description,) \
    if opts.description != "Basic RFI Scan" else opts.description

el_start = 15.1  # 3.1
el_end = 27.1  # 15.1
scan_spacing = 5.0 # 6.0
num_scans = 3
scan_duration = 180.
if opts.long is not None:
    scan_duration = float(opts.long)
scan_extent = 180.
opts.dump_rate = 1.

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        start_time = time.time()
        nd_params = session.nd_params.copy()
        nd_params['period'] = 0
        end_loop_now = False
        while (opts.min_duration is None or (time.time() - start_time) < opts.min_duration) and not end_loop_now:
            if opts.min_duration is None:
                end_loop_now = True
            if (time.time() - start_time) < opts.min_duration or opts.min_duration is None:
                session.fire_noise_diode(announce=False, **nd_params)
                # First Half
                scan_time = time.time()
                azimuth_angle = abs(-90.0 - 270.0) / 4.  # should be 90 deg.
                target1 = 'azel, %f, %f' % (-90. + azimuth_angle, (el_end + el_start) / 2.)
                session.label('raster')
                session.raster_scan(target1, num_scans=num_scans,
                                    scan_duration=scan_duration,
                                    scan_extent=scan_extent,
                                    scan_spacing=scan_spacing,
                                    scan_in_azimuth=True,
                                    projection='plate-carree')
                user_logger.info("Observed horizon part 1/2 for %d seconds",
                                 time.time() - scan_time)
                # Second Half
                half_time = time.time()
                target2 = 'azel, %f, %f' % (-90. + azimuth_angle * 3., (el_end + el_start) / 2.)
                session.label('raster')
                session.raster_scan(target2, num_scans=num_scans,
                                    scan_duration=scan_duration,
                                    scan_extent=scan_extent,
                                    scan_spacing=scan_spacing,
                                    scan_in_azimuth=True,
                                    projection='plate-carree')
                user_logger.info("Observed horizon part 2/2 for %d Seconds (%d Seconds in Total)",
                                 time.time() - half_time, time.time() - start_time)
