#!/usr/bin/env python
# run a simple scan script to derive the horizon mask for MeerKAT
# scan over constant elevation range but loop over azimuth
import time
from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger

# Set up standard script options
description = ("Derive the horizon mask for a MeerKAT dish. Scan over constant elevation "
               "range but loop over azimuth. This takes the form of 2x179 raster scans "
               "in opposite directions, with 180*2 seconds per scan. "
               "There are non-optional options.")
parser = standard_script_options(usage="%prog [options]", description=description)
# Add experiment-specific options
parser.add_option('--elevation-range', dest='elevation_range', type="float", default=1.0,
                  help="The range in elevation to cover starting from 3.1 degrees "
                       "elevation (default=%default)")
parser.add_option('--scan-spacing', dest='scan_spacing', type="float", default=1.0,
                  help="The spacing of the scan lines in the experiment, in degrees (default=%default)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Horizon mask')
# Parse the command line
opts, args = parser.parse_args()

el_start = 16.
el_end = el_start + opts.elevation_range
scan_spacing = opts.scan_spacing
num_scans = int((el_end - el_start) / scan_spacing)
scan_duration = 179.*2.
scan_extent = 179.

with verify_and_connect(opts) as kat:
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        # First Half
        start_time = time.time()
        azimuth_angle = abs(-90.0 - 270.0) / 4.  # should be 90 deg.
        target1 = 'azel, %f, %f' % (-90. + azimuth_angle, (el_end + el_start) / 2.)
        session.label('raster')
        session.raster_scan(target1, num_scans=num_scans, scan_duration=scan_duration,
                            scan_extent=scan_extent, scan_spacing=scan_spacing,
                            scan_in_azimuth=True, projection='plate-carree')
        user_logger.info("Observed horizon part 1/2 for %d seconds" % (time.time() - start_time))
        # Second Half
        half_time = time.time()
        target2 = 'azel, %f, %f' % (-90. + azimuth_angle * 3., (el_end + el_start) / 2.)
        session.label('raster')
        session.raster_scan(target2, num_scans=num_scans, scan_duration=scan_duration,
                            scan_extent=scan_extent, scan_spacing=scan_spacing,
                            scan_in_azimuth=True, projection='plate-carree')
        user_logger.info("Observed horizon part 2/2 for %d Seconds (%d Seconds in Total)" %
                         ((time.time() - half_time), (time.time() - start_time)))
