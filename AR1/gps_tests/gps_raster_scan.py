#! /usr/bin/python
# Do mini-raster scan on nearest strong point source and reduce data to check system performance.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os
import numpy as np
import time

from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katconf
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] ['target']",
                                 description='Perform mini raster scan across (nearest) point source to check system performance.')
# Add experiment-specific options
## Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source check', observer='check')#, dump_rate=1.0)
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    try:
        # load file from head node via katconf server (e.g. gps-ops.txt)
        file_path = 'katexternaldata/catalogues/%s' % (args[0])
        user_logger.info('Adding TLE from file: %s', file_path)
        lines = katconf.resource_string(file_path).split('\n')
        lines = [line + '\r\n' for line in lines if len(line) > 0]
        observation_sources.add_tle(lines)
    except (IOError, ValueError):#IOError or ValueError : # If the file failed to load assume it is a target string
        args_target_obj = collect_targets(kat,args)
        observation_sources.add(args_target_obj)

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            session.capture_start()
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
            # Get the first satellite that is up
            gps_sats = observation_sources.filter(el_limit_deg=[20, 75])
            if len(gps_sats) == 0:
                user_logger.warning("Empty gps catalogue or no targets currently visible")
            target = gps_sats.targets[np.argmin([t.separation(current_pos) for t in gps_sats])]
            user_logger.info("No target specified, picked the closest strong source")

            session.label('raster')
	    session.fire_noise_diode('coupler', on=10, off=10)
	    time.sleep(10)
            session.raster_scan(target, num_scans=3, scan_duration=60, scan_extent=5.0, scan_spacing=0.1, projection=opts.projection)

# -fin-
