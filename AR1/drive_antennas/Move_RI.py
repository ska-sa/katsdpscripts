#!/usr/bin/python
# Exercise the indexer at various elevations.

import time
import random 
import katpoint

from katcorelib import standard_script_options, verify_and_connect, collect_targets, user_logger


def wait_until_sensor_equals(timeout, sensorname, value, sensortype=str, places=5, pollperiod=0.5):

    stoptime = time.time() + timeout
    success = False

    if sensortype == float:
        cmpfun = lambda got, exp: abs(got - exp) < 10 ** -places
    else:
        cmpfun = lambda got, exp: got == exp

    lastval = None
    while time.time() < stoptime:
        # TODO: support selectable AP, not just m036
        lastval= kat.m036.sensors.get(sensorname,None).get_value()

        if cmpfun(lastval, value):
	    success = True
	    break
        time.sleep(pollperiod)
    return (success, lastval) 


def track(ants, target, ridx_position='l', duration=10, dry_run=False):

    # TODO: make the indexer timeout configurable parameter
    indexer_timeout = 120
    # send this target to the antenna.
    ants.req.target(target.description)
    user_logger.info("Target description: '%s' " % target.description)
    ant_names = ','.join([ant.name for ant in ants])
    ants.req.mode("POINT")
    # TODO: more meaningful indication of the target (where no name is specified
    user_logger.info("Slewing %s to target : %s" % (ant_names, target.name))
    # Wait for antenna to lock onto target
    locks=0
    if not dry_run:
        for ant_x in ants:
            if ant_x.wait('lock', True, timeout=300): 
                user_logger.info("%s on target..." % ant_x.name)
                locks +=1
            user_logger.info("%s out of %s antennas on target" % (locks, len(ants)))
    else:
        locks = len(ants)
    if len(ants)==locks:
        user_logger.info("Target reached. Wait for %d seconds before starting indexer cycle" % duration)
        if not dry_run :
            time.sleep(duration)
    else:
        user_logger.error("Antennas failed to reach target.")
        
    ants.req.mode('STOP')
    if not dry_run:
        # Added sleep to wait for AP brakes to engage 
        time.sleep(2)

    user_logger.info("Setting initial RI on %s to position : %s " % (ant_names, ridx_position.upper()))
    ants.req.ap_set_indexer_position(ridx_position)
    if not dry_run:
        result = wait_until_sensor_equals(indexer_timeout, 'ap_indexer_position', ridx_position)
    else:
        result = (True, ridx_position)

    if result[0]==True:
        pass
    else:
        # TODO: support selectable AP, not just m036
        ridx_position_raw = kat.m036.sensors.get('ap_indexer_position_raw',None).get_value()
        user_logger.error("Timed out while waiting %s seconds for indexer to reach '%s' position. "
                          "Last position reading was %s degrees." %
                          (indexer_timeout ,ridx_position.upper(), ridx_position_raw))

    count = 0
    # TODO: make this sequence easier to configure
    ridx_sequence = ['u','l','x','s','l','u','s','x','u','l','s','x','l','u','x','s','u','x','l','s','l','s']

    #Setting indexer positions per cycles 
    indexing_times = []
    if not dry_run:
        for pos in ridx_sequence:
            # TODO: support selectable AP, not just m036
            ridx_last_position = kat.m036.sensors.get('ap_indexer_position',None).get_value()
            count = time.time()
            user_logger.info("--- Moving RI to position: '%s' ---" % pos.upper())
            ants.req.ap_set_indexer_position(pos)

            result = wait_until_sensor_equals(indexer_timeout, 'ap_indexer_position', pos)
            user_logger.debug("Request result: '%s', last sensor reading: '%s'" % (result[0], result[1]))

            if result[0]==True:
                pass
            else:
                #  TODO: support selectable AP, not just m036
                ridx_position_raw = kat.m036.sensors.get('ap_indexer_position_raw',None).get_value()
                ridx_brakes_released = kat.m036.sensors.get('ap_ridx_brakes_released',None).get_value()
                user_logger.error("Timed out while waiting %s seconds for indexer to reach '%s' position."
                                  "Last position reading was %s degrees."
                                  "Brakes released: '%s'. " %
                                  (indexer_timeout ,pos.upper(), ridx_position_raw, ridx_brakes_released))

            # TODO: support selectable AP, not just m036
            ridx_current_position = kat.m036.sensors.get('ap_indexer_position',None).get_value()
            time_to_index = time.time() - count
            indexing_times.append((ridx_last_position + ' to ' + ridx_current_position, time_to_index))
            if ridx_current_position in ['undefined']:
                user_logger.warning("Movement ended in undefined position")
            else:
                user_logger.info("RIDX from '%s' to '%s' position took '%s' seconds." %
                                 (ridx_last_position.upper(), ridx_current_position.upper(), time_to_index))

            # 60 seconds comes from the antenna specification
            if (time_to_index > 60.0):
                user_logger.warning("Indexer took longer than 60 seconds!")

            user_logger.info("Dwell for %s seconds before going to next position." % duration)
            time.sleep(duration)

        user_logger.info("Pattern complete. Heading to next sky target.")


# Set up standard script options

parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Exercise the receiver indexer drive at different elevations.")

# Set default value for any option (both standard and experiment-specific optionsi)
parser.add_option('--dwell-time',type='float',
                  default=5.0,
                  help='Time between changing indexer positions in seconds (default=%default)')
parser.add_option('--num-repeat', type='int',
                  default=1,
                  help='The number of times to repeat the sequence (once by by default)')

# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

if len(args) == 0:
    raise RuntimeError("No targets and indexer position provided to the script")


targetlist = []
for argstr in args:
    temp,taz,tel,band= argstr.split(',')
    print "azimuth : %r elevation : %r indexer : %s" % (taz,tel,band)
    targetlist.append([taz,tel,band])

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    # Set sensor strategies"
    kat.ants.set_sampling_strategy("lock", "event")
    kat.ants.set_sampling_strategy("ap.indexer-position","event")

    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(2)
    else:
        if not kat.dry_run :
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    for taz,tel,band in targetlist:
        for i in range(int(opts.num_repeat)):
            target = katpoint.Target('Name,azel, %s,%s' % (taz,tel))
            track(kat.ants, target, ridx_position=band, duration=opts.dwell_time, dry_run=kat.dry_run)






























































