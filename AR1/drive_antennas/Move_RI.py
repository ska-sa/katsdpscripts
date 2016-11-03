#!/usr/bin/env python
# Exercise the indexer at various elevations.

import time
import katpoint

from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)


def wait_until_sensor_equals(timeout, sensorname, value,
                             sensortype=str, places=5, pollperiod=0.5):

    stoptime = time.time() + timeout
    success = False

    if sensortype == float:
        cmpfun = lambda got, exp: abs(got - exp) < 10 ** -places
    else:
        cmpfun = lambda got, exp: got == exp

    lastval = None
    while time.time() < stoptime:
        lastval = kat.sensors.get(sensorname, None).get_value()

        if cmpfun(lastval, value):
            success = True
            break
        time.sleep(pollperiod)
    return (success, lastval)


def track(ant, target, ridx_position='l', duration=10, dry_run=False):

    # TODO: make the indexer timeout configurable parameter
    indexer_timeout = 120
    # send this target to the antenna.
    ant.req.target(target.description)
    user_logger.info("Target description: '%s' " % target.description)
    ant.req.mode("POINT")

    if not dry_run:
        # Added sleep to wait for AP brakes to disengage
        time.sleep(8)

    # TODO: more meaningful indication of the target
    #       (where no name is specified)
    user_logger.info("Slewing %s to target: %s" % (ant.name, target.name))
    # Wait for antenna to lock onto target
    if not dry_run:
        if ant.wait('lock', True, timeout=300):
            user_logger.info("%s on target... "
                             "Wait for %d seconds "
                             "before starting indexer cycle" %
                             (ant.name, duration))
            time.sleep(duration)
        else:
            user_logger.error("Antennas failed to reach target.")

    ant.req.mode('STOP')
    if not dry_run:
        # Added sleep to wait for AP brakes to engage
        time.sleep(2)
        result = wait_until_sensor_equals(5.0, ant.name + '_mode', 'STOP')
        if result[0] == False:
            user_logger.error("Failed to set AP to 'STOP' mode. "
                              "Indexer commands will not be processed.")
            return

    user_logger.info("Setting initial RI on %s to position : %s " %
                     (ant.name, ridx_position.upper()))

    ant.req.ap_set_indexer_position(ridx_position)
    if not dry_run:
        result = wait_until_sensor_equals(indexer_timeout,
                                          ant.name + '_ap_indexer_position',
                                          ridx_position)
    else:
        result = (True, ridx_position)

    if result[0] == False:
        ridx_position_raw = kat.sensors.get(ant.name + '_ap_indexer_position_raw', None).get_value()
        user_logger.error("Timed out while waiting %s seconds for "
                          "indexer to reach '%s' position. "
                          "Last position reading was %s degrees." %
                          (indexer_timeout,
                           ridx_position.upper(),
                           ridx_position_raw))

    # TODO: make this sequence easier to configure
    ridx_sequence = ['u', 'l', 'x', 's', 'l', 'u', 's', 'x',
                     'u', 'l', 's', 'x', 'l', 'u', 'x', 's',
                     'u', 'x', 'l', 's', 'l', 's']

    # Cycling indexer positions
    if not dry_run:
        for pos in ridx_sequence:
            ridx_last_position = kat.sensors.get(ant.name + '_ap_indexer_position', None).get_value()
            ridx_movement_start_time = time.time()
            user_logger.info("--- Moving RI to position: '%s' ---" %
                             pos.upper())
            ant.req.ap_set_indexer_position(pos)

            result = wait_until_sensor_equals(indexer_timeout,
                                              ant.name + '_ap_indexer_position',
                                              pos)
            user_logger.debug("Request result: '%s', "
                              "last sensor reading: '%s'" %
                              (result[0], result[1]))

            if result[0] == False:
                ridx_position_raw = kat.sensors.get(ant.name + '_ap_indexer_position_raw', None).get_value()
                ridx_brakes_released = kat.sensors.get(ant.name + '_ap_ridx_brakes_released', None).get_value()

                user_logger.error("Timed out while waiting %s seconds "
                                  "for indexer to reach '%s' position. "
                                  "Last position reading was %s degrees. "
                                  "Brakes released: '%s'. " %
                                  (indexer_timeout, pos.upper(),
                                   ridx_position_raw,
                                   ridx_brakes_released))

            ridx_current_position = kat.sensors.get(ant.name + '_ap_indexer_position', None).get_value()
            time_to_index = time.time() - ridx_movement_start_time
            if ridx_current_position in ['undefined']:
                user_logger.warning("Movement ended in undefined position")
            else:
                user_logger.info("RIDX from '%s' to '%s' position "
                                 "took '%s' seconds." %
                                 (ridx_last_position.upper(),
                                  ridx_current_position.upper(),
                                  time_to_index))

            # 60 seconds comes from the antenna specification
            if (time_to_index > 60.0):
                user_logger.warning("Indexer took longer than 60 seconds!")

            user_logger.info("Dwell for %s seconds before "
                             "going to next position." % duration)
            time.sleep(duration)

        user_logger.info("Pattern complete. Heading to next sky target.")


# Set up standard script options

parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Exercise the receiver indexer drive at different elevations.")

parser.add_option('--dwell-time', type='float',
                  default=5.0,
                  help='Time between changing indexer positions in seconds (default=%default)')
parser.add_option('--num-repeat', type='int',
                  default=1,
                  help='The number of times to repeat the sequence (once by by default)')
parser.add_option('--ap', type='str',                                   
                  default="m036",                                                    
                  help='Receptor under test (default is m036)')

# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

if len(args) == 0:
    raise RuntimeError("No targets and indexer positions"
                       "provided to the script")

receptor = None
targetlist = []

for argstr in args:
    temp, taz, tel, band = argstr.split(',')
    print "azimuth : %r elevation : %r indexer : %s" % (taz, tel, band)
    targetlist.append([taz, tel, band])

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    
    for ant in kat.ants:
        if opts.ap in [ant.name]:
            receptor = ant

    if receptor is None:
        raise RuntimeError("Receptor under test is not in controlled array")

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("lock", "event")
    kat.ants.set_sampling_strategy("ap.indexer-position", "event")

    if not kat.dry_run and receptor.req.mode('STOP'):
        user_logger.info("Setting Antenna Mode to 'STOP', "
                         "Powering on Antenna Drives.")
        time.sleep(2)
    else:
        if not kat.dry_run:
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    for taz, tel, band in targetlist:
        for i in range(int(opts.num_repeat)):
            target = katpoint.Target('Name, azel, %s, %s' % (taz, tel))
            track(receptor, target, ridx_position=band,
                  duration=opts.dwell_time, dry_run=kat.dry_run)
            user_logger.info("Elevation: '%s' "
                             "Patterns attempted: '%s' "
                             "Contractual cycles: '%s'." %
                             (tel, i+1, 11*(i+1)))
