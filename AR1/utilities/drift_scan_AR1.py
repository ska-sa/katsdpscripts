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

# parser.add_option('-m', '--max-duration', type='float', default=None,
#                   help='Maximum duration of the script in seconds, after which script will end '
#                        'as soon as the current track finishes (no limit by default)')
# parser.add_option('--repeat', action="store_true", default=False,
#                   help='Repeatedly loop through the targets until maximum duration (which must be set for this)')

parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
parser.add_option('--drift-duration', type='float', default=300,
                  help='Total duration of drift scan')                  

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
        user_logger.error("Unable to set Antenna mode to 'STOP'.")
        
    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
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
            target =observation_sources.filter(el_limit_deg=opts.horizon).targets[0]
            
            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)
            
            session.label('track')
            user_logger.info("Tracking %s for 30 seconds" % (target.name))	
            session.track(target, duration=30)

            session.label('raster')
            user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
            session.raster_scan(target, num_scans=5, scan_duration=50, scan_extent=2.0,
                                        scan_spacing=0.4, scan_in_azimuth=True,
                                        projection=opts.projection)

            session.label('drift')
            start_time = time.time()
            az,el = target.azel(start_time + (opts.drift_duration / 2))
            if (az*180/math.pi > 275.0):
                az = az - (360/180 * math.pi)
            new_targ = katpoint.Target('Drift scan_duration of %s, azel, %10.8f, %10.8f' % (target.name, az*180/math.pi ,el*180/math.pi))
            user_logger.info("Initiating drift scan of %s" % (target.name))	
            az,el = target.azel(start_time + (opts.drift_duration / 2))            
            session.track(new_targ, duration=opts.drift_duration)

            session.label('track')
            user_logger.info("Tracking %s for 30 seconds" % (target.name))	
            session.track(target, duration=30)

            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)


