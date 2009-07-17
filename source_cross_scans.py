#!/usr/bin/python
# Perform cross hair scans on all targets within a specified elevation range

import ffuilib as ffui

ff = ffui.tbuild("cfg-telescope.ini", "local_ant_only")
 # creates connection to ANT, DBE and RFE. Reads in default configuration and build targets and observatory lists

scans = [ (-5,0) , (0,-5) ]
 # cross hair scan of the target from -5 to 5 degrees in x and y

scan_duration = 30
 # take 30s per leg of the crosshair

up_sources = ff.sources.iterfilter(tags="CAL",el_limit_deg=[20,55])
 # get all calibrator sources from the built in catalog that are between 20 and 55 degrees
 # Note: This returns an interator which recalculates the el limits each time a new object is requested

for source in up_sources:
    print "Pointing Scan: ",source.name
    ff.ant1.req_target(source.get_description())
     # send this target to the antenna. No time offset
    ff.ant1.req_mode("POINT")
     # set mode to point
    ff.ant1.wait("lock","1",300)
     # wait for lock
    scan_count = 0
    
    for scan in scans:
        print "Scan Progress:",int((float(scan_count) / len(scans))*100),"%"
        ff.ant1.req_scan_sym(scan[0],scan[1],scan_duration)
        ff.ant1.wait("lock","1",300)
         # ready to start scan
        ff.ant1.req_mode("SCAN")
        ff.ant1.wait("scan_status","after",300)
         # wait for the scan to complete
        scan_count += 1
    print "Scan complete."

ff.disconnect()
 # exit

