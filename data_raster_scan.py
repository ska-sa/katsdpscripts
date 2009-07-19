#!/usr/bin/python
# Create a DBE data stream, capture it and send the data to the signal displays.
# Needs the C dbe simulator running (./dbe_server2 8000) and k7w which should be running (./k7w_server 8001)

import ffuilib as ffui
import time

ff = ffui.tbuild("cfg-user.ini", "local_ff_client_sim")
 # make fringe fingder connections

tgt = 'Takreem,azel,20,30'
 # the virtual target that the dbe simulator contains

ff.dbe.req_dbe_capture_stop()
ff.ant1.req_mode("STOP")
time.sleep(0.5)
 # cleanup any existing experiment

ff.k7w.req_target(tgt)
 # let the data collector know the current target
ff.k7w.req_output_directory("/var/kat/data/")
ff.k7w.req_write_hdf5(1)
ff.k7w.req_capture_start()
ff.k7w.req_scan_id(1,"slew")
 # make note that our first scan is a slew

ff.dbe.req_dbe_packet_count(600)
 # stream 10 minutes of data or until stop issued
ff.dbesim.req_dump_rate(1)
 # correlator dump rate set to 1 Hz
ff.dbe.req_dbe_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbe.req_dbe_capture_start("stream")
 # start emitting data on stream "stream"

ff.ant1.sensor_pos_actual_scan_azim.register_listener(ff.dbesim.req_pointing_az, 0.5)
ff.ant1.sensor_pos_actual_scan_elev.register_listener(ff.dbesim.req_pointing_el, 0.5)
 # when the sensor value changes send an update to the listening function. Rate limited to 0.5 second updates.

scans = [ (-1,0.5) , (1,0) , (-1,-0.5) ]
 # raster scan
scan_duration = 120

ff.ant1.req_target(tgt)
  # send this target to the antenna. No time offset
ff.ant1.req_mode("POINT")
 # set mode to point
ff.ant1.wait("lock",True,300)
ff.k7w.req_compound_scan_id(1)
 # once we are on the target begin a new compound scan
scan_count = 1

for scan in scans:
    print "Scan Progress:",int((float(scan_count) / len(scans))*100),"%"
    ff.ant1.req_scan_asym(-scan[0],scan[1],scan[0],scan[1],scan_duration)
    ff.ant1.wait("lock",True,300)
    ff.k7w.req_scan_id(scan_count,"scan")
     # mark this section as valid scan data
    ff.ant1.req_mode("SCAN")
    ff.ant1.wait("scan_status","after",300)
    ff.k7w.req_scan_id(scan_count+1,"slew")
     # slewing to next raster pointg
    scan_count += 2
print "Scan complete."

ff.dbe.req_dbe_capture_stop("stream")
ff.k7w.req_capture_stop()
ff.disconnect()
