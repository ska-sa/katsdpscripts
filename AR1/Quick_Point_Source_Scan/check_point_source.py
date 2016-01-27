#! /usr/bin/python
# Do mini-raster scan on nearest strong point source and reduce data to check system performance.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os
import numpy as np
import time

from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] ['target']",
                                 description='Perform mini raster scan across (nearest) point source to check system performance.')
# Add experiment-specific options
# parser.add_option('-c', '--channels', default='100,400',
#                   help="Frequency channel range to keep (zero-based, specified as 'start,end', default='%default')")
## Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source check', observer='check')#, dump_rate=1.0)
# parser.set_defaults(description='Point source check', observer='check', nd_params='coupler,0,0,-1', dump_rate=2.0)
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
         user_logger.error("Unable to set Antenna mode to 'STOP'.")

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
# RvR -- temporarily added to allow manual start of metadata
        user_logger.info('Start metadata capture')
        time.sleep(2) # give time to issue cmd: cam.data_1.req.cbf_capture_meta('c856M4k')
# RvR -- temporarily added to allow manual start of metadata
        # Pick a target, either explicitly or the closest strong one
        if len(args) > 0:
            target = collect_targets(kat, [args[0]]).targets[0]
        else:
            # Get current position of first antenna in the list (assume the rest are the same or close)
            if kat.dry_run:
                current_az, current_el = session._fake_ants[0][2:]
            else:
                current_az = session.ants[0].sensor.pos_actual_scan_azim.get_value()
                current_el = session.ants[0].sensor.pos_actual_scan_elev.get_value()
                if current_az is None:
                    user_logger.warning("Sensor kat.%s.sensor.pos_actual_scan_azim failed - using default azimuth" %
                                        (session.ants[0].name))
                    current_az = 0.
                if current_el is None:
                    user_logger.warning("Sensor kat.%s.sensor.pos_actual_scan_elev failed - using default elevation" %
                                        (session.ants[0].name))
                    current_el = 30.
            current_pos = katpoint.construct_azel_target(katpoint.deg2rad(current_az), katpoint.deg2rad(current_el))
            # Get closest strong source that is up
            strong_sources = kat.sources.filter(el_limit_deg=[20, 75], flux_limit_Jy=100, flux_freq_MHz=opts.centre_freq)
            if len(strong_sources) == 0:
                user_logger.warning("Empty point source catalogue or no targets currently visible")
            target = strong_sources.targets[np.argmin([t.separation(current_pos) for t in strong_sources])]
            user_logger.info("No target specified, picked the closest strong source")

        session.label('raster')
#         session.fire_noise_diode('coupler', on=4, off=4)
#         session.raster_scan(target, num_scans=3, scan_duration=15, scan_extent=5.0, scan_spacing=0.5, projection=opts.projection)
# Longer test but better beam shape for fitting
#         session.raster_scan(target, num_scans=3, scan_duration=59, scan_extent=5.0, scan_spacing=0.5, projection=opts.projection)
# Quick test but wide scan if you do not know your pointing
        session.raster_scan(target, num_scans=7, scan_duration=15, scan_extent=7.0, scan_spacing=1, projection=opts.projection)
#         session.raster_scan(target, num_scans=7, scan_duration=30, scan_extent=5.0, scan_spacing=1, projection=opts.projection)

# -fin-
