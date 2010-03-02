#!/usr/bin/python
# Track sources all around the sky for a few seconds each without recording data (mostly to keep tourists or antennas amused).

import katuilib
from optparse import OptionParser
import sys
import time

# Parse command-line options that allow the defaults to be overridden
# Default KAT configuration is *local*, to prevent inadvertent use of the real hardware
parser = OptionParser(usage="usage: %prog [options]\n\n"+
                            "Track sources all around the sky for a few seconds each without recording data\n"+
                            "(mostly to keep tourists or antennas amused). Uses the standard catalogue.\n"+
                            "Excludes the extremely strong sources (Sun, Afristar).")
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                  help='Telescope configuration file to use in conf directory (default="%default")')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                  help='Selected configuration to use (default="%default")')
parser.add_option('-a', '--ants', dest='ants', type="string", metavar='ANTS',
                  help="Comma-separated list of antennas to include in scan (e.g. 'ant1,ant2')," +
                       " or 'all' for all antennas - this MUST be specified (safety reasons)")
(opts, args) = parser.parse_args()

# Force antennas to be specified to sensitise the user to what will physically move
if opts.ants is None:
    print 'Please specify the antennas to use via -a option (yes, this is a non-optional option..., -h for help)'
    sys.exit(1)

# Build KAT configuration, as specified in user-facing config file
# This connects to all the proxies and devices and queries their commands and sensors
kat = katuilib.tbuild(opts.ini_file, opts.selected_config)

# Create a list of the specified antenna devices, and complain if they are not found
if opts.ants.strip() == 'all':
    ants = kat.ants
else:
    try:
        ants = katuilib.Array('ants', [getattr(ff, ant_x.strip()) for ant_x in opts.ants.split(",")])
    except AttributeError:
        raise ValueError("Antenna '%s' not found" % ant_x)

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

# Get an interator which recalculates the el limits each time a new object is requested
up_sources = cat.iterfilter(el_limit_deg=[3,89])

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
        for ant_x in ants.devs:
            if ant_x.wait("lock", True, 300): locks += 1

        if len(ants.devs) == locks:
            targets_tracked += 1
            # continue tracking the target for specified time
            time.sleep(on_target_duration)

        total_target_count += 1
except Exception, e:
    print "Exception: ", e
    print 'Exception caught: attempting to exit cleanly...'
finally:
    # still good to get the stats even if script interrupted
    end_time = time.time()
    print '\nelapsed_time: %.2f mins' %((end_time - start_time)/60.0)
    print 'targets attempted: ', total_target_count
    print 'target lock achieved: ', targets_tracked, '\n'

    # exit
    print "setting drive-strategy back to the default"
    ants.req.drive_strategy("longest-track") # set back to the default
    kat.disconnect()
