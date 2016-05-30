#!/usr/bin/python
# Do track, mini-raster and drift on nearest strong point source and reduce data to check power levels and verify antenna tracking 

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint
import math

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Special verification observation that will do a track, scan and drift obs.')
# Add experiment-specific options

parser.add_option('--max-elevation', type='float', default=90.0,
                  help="Maximum elevation angle, in degrees (default=%default)")
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
parser.add_option('--drift', action="store_true", default=False,
                  help='Do a drift scan observation')
parser.add_option('--drift-duration', type='float', default=300,
                  help='Total duration of drift scan')
parser.add_option('--scan', action="store_true", default=False,
                  help='Do a raster scan observation')
parser.add_option('--scan-duration', type='float', default=60,
                  help='Total duration of raster scan')
parser.add_option('--track', action="store_true", default=False,
                  help='Track the target for some time')
parser.add_option('--track-duration', type='float', default=30,
                  help='Total duration of target track')

# Set default value for any option (both standard and experiment-specific options)
# RvR -- Cannot set dump rate for AR1
# parser.set_defaults(description='Target track',dump_rate=1)
parser.set_defaults(description='Target track/scan/drift')
# RvR -- Cannot set dump rate for AR1
# Parse the command line
opts, args = parser.parse_args()

# #if len(args) == 0:
# #    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
# #                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(5)
    else:
        user_logger.error("Dry Run: Unable to set Antenna mode to 'STOP'.")

    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    try:
        observation_sources.add_tle(file(args[0]))
    except (IOError, ValueError):#IOError or ValueError : # If the file failed to load assume it is a target string
        args_target_obj = collect_targets(kat,args)
        observation_sources.add(args_target_obj)

#     observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=[opts.horizon,opts.max_elevation])) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:

            if not opts.no_delays and not kat.dry_run :
                if session.data.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
                if session.data.req.auto_delay('off'):
                    user_logger.info("Turning off delay tracking.")
                else:
                    user_logger.error('Unable to turn off delay tracking.')
                if session.data.req.zero_delay():
                    user_logger.info("Zeroed the delay values.")
                else:
                    user_logger.error('Unable to zero delay values.')

            session.standard_setup(**vars(opts))
            session.capture_start()

            time.sleep(5)
            start_time = time.time()
            target =observation_sources.filter(el_limit_deg=[opts.horizon, opts.max_elevation]).targets[0]

            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)

            if opts.track:
                session.label('track')
                session.set_target(target) # Set the target
                user_logger.info("Tracking %s for %d seconds" % (target.name, opts.track_duration))
                session.track(target, duration=opts.track_duration)
            time.sleep(10)

            if opts.scan:
                session.label('raster')
                session.set_target(target) # Set the target
                user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
                session.raster_scan(target, num_scans=3, scan_duration=opts.scan_duration, scan_extent=3.0,
                                            scan_spacing=0.1, scan_in_azimuth=True,
                                            projection=opts.projection)
            time.sleep(10)

            if opts.drift:
                session.label('drift')
                session.set_target(target) # Set the target
                start_time = time.time()
                az,el = target.azel(start_time + (opts.drift_duration / 2))
                if (az*180/math.pi > 275.0):
                    az = az - (360/180 * math.pi)
                new_targ = katpoint.Target('Drift scan_duration of %s, azel, %10.8f, %10.8f' % (target.name, az*180/math.pi ,el*180/math.pi))
                user_logger.info("Initiating drift scan of %s" % (target.name))
                az,el = target.azel(start_time + (opts.drift_duration / 2))
                session.track(new_targ, duration=opts.drift_duration)
            time.sleep(10)

            if opts.track:
                session.label('track')
                az,el = target.azel(start_time + (2 / 2))
                new_targ = katpoint.Target('Dummy target to force session to slew to target %s, azel, %10.8f, %10.8f' % (target.name, az*180/math.pi ,el*180/math.pi))
                session.track(new_targ, duration=0)
                session.set_target(target) # Set the target
                user_logger.info("Tracking %s for %d seconds" % (target.name, opts.track_duration))
                session.track(target, duration=opts.track_duration)
            time.sleep(10)

            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)

    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(5)
    else:
        user_logger.error("Dry Run: Unable to set Antenna mode to 'STOP'.")
