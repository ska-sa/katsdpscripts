#!/usr/bin/python

import ffuilib as ffui

ff = ffui.cbuild("ffuilib.ant_only.rc")
 # creates connection to ANT, DBE and RFE. Reads in default configuration and build targets and observatory lists

scans = [ (-5,0) , (0,-5) ]
 # cross hair scan of the target from -5 to 5 degrees in x and y

scan_duration = 30
 # take 30s per leg of the crosshair

cal_sources = ff.sources.filter(tag='CALIBRATOR')
 # get calibrator sources from the built in catalog

source = cal_sources.filterpop(el_min=20,el_max=55)
 # find the first available source that is above 20 and below 55 degrees elevation

while source is not None:
    source_name = source.name.rfind(" ") == -1 and source.name or source.name[:source.name.rfind(" ")+1]
    print "Pointing Scan: ",source_name
    ff.ant2.req_target_named(source_name)
     # send this target to the antenna. No time offset
    ff.ant2.req_mode("POINT")
     # set mode to point
    ff.ant2.wait("lock","1",300)
     # wait for lock
    scan_count = 0
    
    for scan in scans:
        print "Scan Progress:",int((float(scan_count) / len(scans))*100),"%"
        ff.ant2.req_scan_sym(scan[0],scan[1],scan_duration)
        ff.ant2.wait("lock","1",300)
         # ready to start scan
        ff.ant2.req_mode("SCAN")
        ff.ant2.wait("scan_status","after",300)
         # wait for the scan to complete
        scan_count += 1
    
    print "Scan complete."
    source = cal_sources.filterpop(el_min=20, el_max=55)

ff.disconnect()
 # exit

