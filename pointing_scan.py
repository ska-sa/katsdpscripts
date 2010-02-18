#!/usr/bin/python
# Perform pointing scan measurements for a range of sources and produce data

# Startup in different terminals:
# start C dbe simulator: ~/svnDS/code/ffinder/trunk/src/simulators/dbe/dbe_server 8000
# start k7 writer: cd /var/kat/data; ~/svnDS/code/ffinder/trunk/src/streaming/k7writer/k7w_server 8001
# start the system: kat-launch.py
# start the central monitor: ~/svnDS/code/ffinder/trunk/src/services/monitoring/central_monitor.py
# start the central logger: ~/svnDS/code/ffinder/trunk/src/services/monitoring/central_logger.py
# and signal display server: ~/svnDS/code/ffinder/trunk/src/streaming/sdisp/ffsocket.py


import katuilib
import time
import numpy as np
import math

import katpoint

# Build Fringe Finder configuration, as specified in user-facing config file
# The specific configuration is one that runs locally with DBE simulator included
# This connects to all the proxies and devices and queries their commands and sensors
# TODO: check that the pointing sources are configured!!
kat = katuilib.tbuild('cfg-local.ini', 'local_ff')

# Clean up any existing experiment
kat.dbe.req.dbe_capture_stop()
kat.ant1.req.mode('STOP')
time.sleep(0.5)

# max measurements
max_compound_scans = 5

# time (secs)for each (sub)scan
scan_duration = 20

# scan parameters
scans = [ (-2,0.5) , (2,0) , (-2,-0.5) ] # azimuth raster scan

# min and max elevation for targets
elmin = 2.0
elmax = 88.0

# Pointing calibrator catalogue (create manually here for now with initial tests - load from file later)
ant = katpoint.construct_antenna('KAT-7, -30:43:16.71, 21:24:35.86, 1055, 12.0')
cat = katpoint.Catalogue(add_specials=False,antenna=ant)
cat.add('Takreem-17+70,azel,-17,70.0')
cat.add('Takreem+10+20,azel,10,20')
cat.add('Takreem+20+30,azel,20,30')
cat.add('Takreem+160+05,azel,160,5')

# set the drive strategy for how antenna moves between targets
# (options are: "longest-track", the default, or "shortest-slew")
kat.ant1.req.drive_strategy("shortest-slew")

# Let the data collector know about data file location and format
kat.k7w.req.output_directory(katuilib.defaults.kat_directories["data"])
kat.k7w.req.write_hdf5(1)
# Set the target description string for the first compound scan in the output file
kat.k7w.req.target('azel,5,5') # TODO: This is a bit arb - need to address!!
# First scan will be a slew to the target - mark it as such before k7w starts
kat.k7w.req.scan_tag('slew')
# Do this BEFORE starting the DBE, otherwise no data will be captured
kat.k7w.req.capture_start()

# Correlator dump rate set to 1 Hz
kat.dbe.req.dbe_dump_rate(1)
# Stream 15 minutes of data (900 dumps) or until stop issued
kat.dbe.req.dbe_packet_count(900)
# Create a new data source labelled "stream", and send data to port 7010 (default k7w data port)
kat.dbe.req.dbe_capture_destination('stream', '127.0.0.1:7010')
# Now start emitting data on stream "stream"
kat.dbe.req.dbe_capture_start('stream')

# stream target updates to simulator
kat.ant1.sensor.pos_actual_scan_azim.register_listener(kat.dbesim.req.pointing_az, 0.5)
kat.ant1.sensor.pos_actual_scan_elev.register_listener(kat.dbesim.req.pointing_el, 0.5)

compound_scan_id = 1
distinct_targets_tracked = {}
start_time = time.time()

try:
    while compound_scan_id < max_compound_scans+1:

        # get sources from catalogue that are in specified elevation range. Antenna will get
        # as close as possible to targets which are out of drivable range.
        # Note: This returns an interator which recalculates the el limits each time a new object is requested
        up_sources = cat.iterfilter(el_limit_deg=[elmin,elmax])

        for source in up_sources:

            print '\n------------------------------------------'
            print 'Starting compound scan ', compound_scan_id, ' of ', max_compound_scans
            print 'Script elapsed_time: %.2f mins' %((time.time() - start_time)/60.0)
            print "Target to scan: ",source.name
            print '------------------------------------------'

            # Start a new compound scan, which involves a new target description and 'slew' scan label
            # This part is actually slewing to target position
            kat.k7w.req.scan_tag('slew')
            kat.k7w.req.target(source.description)
            kat.k7w.req.compound_scan_id(compound_scan_id)

            # tell the DBE simulator about the target so that we get some signal coming through (temp, for testing)
            kat.dbe.req.dbe_test_target(source.azel()[0]*180.0/math.pi,source.azel()[1]*180.0/math.pi)
            print 'dbe test target: ', source.azel()[0]*180.0/math.pi, source.azel()[1]*180.0/math.pi

            # send this target to the antenna and wait for lock
            kat.ant1.req.target(source.description)
            kat.ant1.req.mode("POINT")
            target_locked = kat.ant1.wait("lock","1",200)

            # perform raster scan on target
            if target_locked:
                # keep note of which distinct targets have been tracked and how many times
                if distinct_targets_tracked.has_key(source.name):
                    distinct_targets_tracked[source.name] += 1
                else:
                    distinct_targets_tracked[source.name] = 1

                for scan_count, scan in enumerate(scans):
                    print "Compound scan progress:",int((float(scan_count) / len(scans))*100),"%"
                    kat.k7w.req.scan_id(2*scan_count, 'slew')
                    kat.ant1.req.scan_asym(-scan[0],scan[1],scan[0],scan[1],scan_duration)
                    kat.ant1.wait("lock",True,300)
                    # we are on the target at the start of the scan, start a new scan labelled 'scan'
                    kat.k7w.req.scan_id(2*scan_count+1, 'scan')
                    kat.ant1.req.mode("SCAN")
                    kat.ant1.wait("scan_status","after",300)

            compound_scan_id += 1
            if compound_scan_id >= max_compound_scans + 1: break

except BaseException, e:
    print "Exception: ", e
    print 'Exception caught: attempting to exit cleanly...'
finally:
    # still good to get the stats even if script interrupted
    end_time = time.time()
    print '\nelapsed_time: %.2f mins' %((end_time - start_time)/60.0)
    print 'targets tracked: ', distinct_targets_tracked ,'\n'

    # Find out which files have been created
    files = kat.k7w.req.get_current_files(tuple=True)[1][2]
    print 'Data captured to ', files

    # Stop recording and shut down the experiment
    kat.dbe.req.dbe_capture_stop('stream')
    kat.k7w.req.capture_stop()

    print "setting drive-strategy back to the default"
    kat.ant1.req.drive_strategy("longest-track") # good practice

    kat.disconnect()

