#!/usr/bin/python
# Track sources all around the sky

import ffuilib as ffui
import time

# creates connection to two antennas Reads in default configuration and build targets and observatory lists
ff = ffui.tbuild("cfg-user.ini", "local_ant_only")

# time to stay on each target (secs)
on_target_duration = 10

# set the drive strategy for how antenna moves between targets
# (options are: "longest-track", the default, or "shortest-slew")
ff.ant1.req_drive_strategy("shortest-slew")

# get sources from catalog that are between 0 and 90 degrees. Antenna will get as close as possible
# to targets wich are out of drivable range.
# Note: This returns an interator which recalculates the el limits each time a new object is requested
up_sources = ff.sources.iterfilter(el_limit_deg=[0,90])

for source in up_sources:
    print "Target to track: ",source.name

    # send this target to the antenna.
    ff.ant1.req_target(source.get_description())
    ff.ant1.req_mode("POINT")

    # wait for lock
    ff.ant1.wait("lock","1",150)

    # continue tracking the target for specified time
    time.sleep(on_target_duration)

# exit
ff.disconnect()

