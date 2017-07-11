#!/usr/bin/python
# Track sources all around the sky for a few seconds each without recording data (mostly to keep tourists or antennas amused).

import time
from katcorelib import standard_script_options, verify_and_connect, user_logger ,start_session

def track(ants,target,duration=10):
    # send this target to the antenna.
    ants.req.target(target)
    ants.req.mode("POINT")
    user_logger.info("Slewing to target : %s"%target.name)
    # wait for antennas to lock onto target
    locks = 0
    for ant_x in ants:
        if ant_x.wait("lock", True, 300): locks += 1
    if len(ants) == locks:
        user_logger.info("Tracking Target : %s for %s seconds"%(target.name,str(duration)))
        time.sleep(duration)
        user_logger.info("Target tracked : %s "%(target.name,))
        return True
    else:
        user_logger.warning("Unable to track Targe : %s "%(target.name,))
        return False


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="Track sources all around the sky for a few seconds each without recording data\n"+
                            "(mostly to keep tourists or antennas amused). Uses the standard catalogue,\n"+
                            "but excludes the extremely strong sources (Sun, Afristar). Some options\n"+
                            "are **required**.")
parser.add_option('-m', '--max-duration', type='float', default=600.0,
    help="Maximum time to run experiment, in seconds (default=%default)")
parser.add_option('-t', '--target-duration', type='float', default=10.0,
    help="Time to spend on a target in seconds, in seconds (default=%default)")

on_target_duration = 10
parser.set_defaults(nd_params='off')
(opts, args) = parser.parse_args()
user_logger.info("drive_antennas.py: start")
on_target_duration = opts.target_duration
# Try to build the  KAT configuration
# This connects to all the proxies and devices and queries their commands and sensors
with verify_and_connect(opts) as kat:
    if not opts.dry_run :
        kat.ants.req.sensor_sampling("lock","event")
        cat = kat.sources
        # remove some very strong sources so as not to saturate equipment deliberately.
        cat.remove('Sun')
        cat.remove('AFRISTAR')

        #on_target_duration = 10
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            targets_before_loop = len(targets_observed)
            for target in cat.iterfilter(el_limit_deg=[opts.horizon,89]):
                if  not track(kat.ants,target, duration= on_target_duration):
                    break
                else :
                    targets_observed.append(target.name)
                if (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script" %
                            (opts.max_duration,))
                        keep_going = False
                        break
            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                keep_going = False
        user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
    else:
        with start_session(kat, **vars(opts)) as session: # Fake session to make dry-run happy
            session.standard_setup(**vars(opts))
            session.capture_start()
            cat = kat.sources
            # remove some very strong sources so as not to saturate equipment deliberately.
            cat.remove('Sun')
            cat.remove('AFRISTAR')
            on_target_duration = 10
            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                targets_before_loop = len(targets_observed)
                for target in cat.iterfilter(el_limit_deg=[opts.horizon,89]):
                    if  not session.track(target, duration= on_target_duration,announce=False):
                        break
                    else :
                        targets_observed.append(target.name)
                    if (time.time() - start_time >= opts.max_duration):
                            user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script" %
                                (opts.max_duration,))
                            keep_going = False
                            break
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
user_logger.info("drive_antennas.py: stop")
