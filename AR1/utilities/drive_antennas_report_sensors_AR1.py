#!/usr/bin/python
# Track sources all around the sky for a few seconds each without recording data (mostly to keep tourists or antennas amused).

import time
from katmisc.utils.ansi import get_sensor_colour, col
from katmisc.utils.utils import get_time_str, escape_name
from katcorelib import (standard_script_options, verify_and_connect,
                        user_logger, start_session)
from katpoint import rad2deg

def report_sensors(kat, filter, status):
    user_logger.info("Requesting sensor values... (filter={}, status={})"
                     "".format(filter, status))
    sensors = kat.list_sensors(filter=filter)
    for s in sensors:
        try:
            sensor_obj = getattr(kat.sensor, escape_name(s[1]))
            sensor_obj.get_value()
        except:
            user_logger.error('Could not get the sensor value for {}'.format(s.name))
    user_logger.info("Waiting...")
    for i in range(0, len(sensors) / 20 + 1):
        time.sleep(0.1)

    sensors = kat.list_sensors(filter=filter, status=status)
    for s in sensors:
        name = s.name
        reading = s.reading
        val = str(reading.value)
        valTime = reading.received_timestamp
        updateTime = reading.timestamp
        stat = reading.status
        colour = get_sensor_colour(stat)
        # Print status with stratchar prefix - indicates strategy has been set
        val = val if len(val) <= 45 else val[:42] + "..."  # truncate value to first 45 character
        val = r"\n".join(val.splitlines())
        user_logger.info("{} {} {} {} {}"
                         "".format(col(colour) + name.ljust(45),
                                   str(stat).ljust(15),
                                   get_time_str(valTime).ljust(15),
                                   get_time_str(updateTime).ljust(15),
                                   str(val).ljust(45) + col('normal')))

def track(ants, target, duration=10):
    # send this target to the antenna.
    ants.req.target(target.description)
    ants.req.mode("POINT")
    user_logger.info("Slewing to target : %s" % target.name)
    # wait for antennas to lock onto target
    locks = 0
    for ant_x in ants:
        ant_x.set_sampling_strategy("lock", ("event",))
        if ant_x.wait(sensor_name="lock", condition_or_value=True, timeout=300):
            locks += 1
    if len(ants) == locks:
        user_logger.info("Tracking Target : %s for %s seconds" % (target.name, str(duration)))
        time.sleep(duration)
        user_logger.info("Target tracked : %s " % (target.name,))
        return True
    else:
        user_logger.warning("Unable to track Targe : %s " % (target.name,))
        return False


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(
    usage="usage: %prog [options]",
    description=("Track sources all around the sky for a few seconds each without recording data\n"
                 "(mostly to keep tourists or antennas amused). Uses the standard catalogue,\n"
                 "but excludes the extremely strong sources (Sun, Afristar). Some options\n"
                 "are **required**."))
parser.add_option(
    '-m', '--max-duration', type='float', default=60.0,
    help="Maximum time to run experiment, in seconds (default=%default)")
parser.add_option(
    '-t', '--target-duration', type='float', default=10.0,
    help="Time to spend on a target in seconds, in seconds (default=%default)")
parser.add_option(
    '-v', '--verbose', action="store_true", default=False,
    help='List all sensors in error')
parser.add_option(
    '--drive-antenna', action="store_true", default=False,
    help='Drive and point the antenna to verify lock and track')
parser.add_option(
    '--max-elevation', type='float', default=80.0,
    help="Maximum elevation angle, in degrees (default=%default)")
parser.set_defaults(nd_params='off')
(opts, args) = parser.parse_args()
user_logger.info("drive_antennas.py: start")
# Try to build the  KAT configuration
# This connects to all the proxies and devices and queries their commands and sensors
with verify_and_connect(opts) as kat:

    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Dry-run: Unable to set Antenna mode to 'STOP'.")

    if opts.verbose or opts.drive_antennas:
        user_logger.info('{}{}{}'.format('-'*40, 'Non-nominal Sensors before track', '-'*40))
        report_sensors(kat, '', 'unknown|warn|error|failure|unreachable')
        user_logger.info('-'*120)

    if opts.drive_antenna:
         if not opts.dry_run:
             time.sleep(2)
             cat = kat.sources
             # remove some very strong sources so as not to saturate equipment deliberately.
             cat.remove('Sun')
             cat.remove('AFRISTAR')

             start_time = time.time()
             targets_observed = []
             # Keep going until the time is up
             keep_going = True
             while keep_going:
                 targets_before_loop = len(targets_observed)
                 for target in cat.iterfilter(el_limit_deg=[opts.horizon, opts.max_elevation]):
                     if not track(kat.ants, target, duration=opts.target_duration):
                         break
                     else:
                         targets_observed.append(target.name)
                     if (time.time() - start_time >= opts.max_duration):
                             user_logger.warning(
                                 "Maximum duration of %g seconds has elapsed - stopping script" %
                                 (opts.max_duration,))
                             keep_going = False
                             break
                 if keep_going and len(targets_observed) == targets_before_loop:
                     user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                     keep_going = False
             user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
         else:
             with start_session(kat, **vars(opts)) as session:  # Fake session to make dry-run happy
                 session.standard_setup(**vars(opts))
                 session.capture_start()
                 cat = kat.sources
                 # remove some very strong sources so as not to saturate equipment deliberately.
                 cat.remove('Sun')
                 cat.remove('AFRISTAR')
                 opts.target_duration = 10
                 start_time = time.time()
                 targets_observed = []
                 # Keep going until the time is up
                 keep_going = True
                 while keep_going:
                     targets_before_loop = len(targets_observed)
                     for target in cat.iterfilter(el_limit_deg=[opts.horizon, opts.max_elevation]):
		         user_logger.info('Selected target %s' % (target))
		         user_logger.info('Az/El coordinates (%.2f,%.2f)' % (rad2deg(float(target.azel()[0])),rad2deg(float(target.azel()[1]))))
                         if not session.track(target, duration=opts.target_duration, announce=False):
                             break
                         else:
                             targets_observed.append(target.name)
                         if (time.time() - start_time >= opts.max_duration):
                                 user_logger.warning(
                                     "Maximum duration of %g seconds has elapsed - stopping script" %
                                     (opts.max_duration,))
                                 keep_going = False
                                 break
                     if keep_going and len(targets_observed) == targets_before_loop:
                         user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                         keep_going = False
                 user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))

         user_logger.info('{}{}{}'.format('-'*40, 'Non-nominal Sensors after track', '-'*40))
         report_sensors(kat, '', 'unknown|warn|error|failure|unreachable')
         user_logger.info('-'*120)

user_logger.info("drive_antennas_report_sensors.py: stop")
