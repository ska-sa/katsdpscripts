#!/usr/bin/python
# Track sources all around the sky for a few seconds each without recording data (mostly to keep tourists or antennas amused).

from optparse import OptionParser
import sys
import time

import katcorelib
from katcorelib.observe import standard_script_options, verify_and_connect
from katcorelib.defaults import activity_logger, user_logger

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="Track sources all around the sky for a few seconds each without recording data\n"+
                            "(mostly to keep tourists or antennas amused). Uses the standard catalogue,\n"+
                            "but excludes the extremely strong sources (Sun, Afristar). Some options\n"+
                            "are **required**.")
(opts, args) = parser.parse_args()

activity_logger.info("drive_antennas.py : start")
user_logger.info("drive_antennas.py: start")
# Try to build the  KAT configuration
# This connects to all the proxies and devices and queries their commands and sensors
try:
    kat = verify_and_connect(opts)
except ValueError, err:
    activity_logger.info("drive_antennas.py : could not build host for sb-id-code %s (%s) " % (opts.sb_id_code, err))
    user_logger.info("drive_antennas.py : could not build host for sb-id-code %s (%s) " % (opts.sb_id_code, err))
    raise ValueError("Could not build host for sb-id-code %s (%s)" % (opts.sb_id_code, err))
print "Using KAT connection with configuration: %s" % (kat.system,)

user_logger.info("drive_antennas.py: built %s" % kat.system)

# Create a list of the specified antenna devices, and complain if they are not found
if opts.ants.strip() == 'all':
    ants = kat.ants
else:
    try:
        ants = katcorelib.Array('ants', [getattr(kat, ant_x.strip()) for ant_x in opts.ants.split(",")])
    except AttributeError:
        raise ValueError("Antenna '%s' not found" % ant_x)


#Setup strategies for the sensors we are interested in
kat.ants.req.sensor_sampling("lock","event")

# time to stay on each target (secs)
on_target_duration = 10

# set the drive strategy for how antenna moves between targets
# (options are: "longest-track", the default, or "shortest-slew")
ants.req.drive_strategy("shortest-slew")

# get sources from catalogue that are in specified elevation range. Antenna will get
# as close as possible to targets which are out of drivable range.
cat = kat.sources
# remove some very strong sources so as not to saturate equipment deliberately.
cat.remove('Sun')
cat.remove('AFRISTAR')

# Get an iterator which recalculates the el limits each time a new object is requested
up_sources = cat.iterfilter(el_limit_deg=[3, 89])

total_target_count = 0
targets_tracked = 0
start_time = time.time()

try:
    for source in up_sources:
        print "Target to track: ",source.name

        # send this target to the antenna.
        ants.req.target(source.description)
        ants.req.mode("POINT")

        # wait for antennas to lock onto target
        locks = 0
        for ant_x in ants:
            if ant_x.wait("lock", True, 300): locks += 1

        if len(ants) == locks:
            targets_tracked += 1
            # continue tracking the target for specified time
            time.sleep(on_target_duration)

        total_target_count += 1

finally:
    # still good to get the stats even if script interrupted
    end_time = time.time()
    print '\nelapsed_time: %.2f mins' % ((end_time - start_time) / 60.0,)
    print 'targets completed: ', total_target_count
    print 'target lock achieved: ', targets_tracked, '\n'
    user_logger.info('elapsed_time: %.2f mins' % ((end_time - start_time) / 60.0,))
    user_logger.info('targets completed: %d' % (total_target_count,))
    user_logger.info('target lock achieved: %d' % (targets_tracked,))

    # exit
    print "setting drive-strategy back to the longest-track"
    ants.req.drive_strategy("longest-track") # set back to the default
    user_logger.info("drive_antennas.py: stop")
    activity_logger.info("drive_antennas.py: stop")
