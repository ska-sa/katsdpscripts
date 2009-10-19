#!/usr/bin/python
# Raster scan across a simulated target producing scan data for signal displays and loading into scape

# Ensure that the integration environment is running as described in the integration test script file

import ffuilib as ffui
import time

ff = ffui.tbuild("cfg-user.ini","integration_ff_client")
 # make fringe fingder connections

tgt = 'Takreem,azel,20,30'
 # the virtual target that the dbe simulator contains

ff.dbe.req.dbe_capture_stop()
ff.ant1.req.mode("STOP")
time.sleep(0.5)
 # cleanup any existing experiment

ff.k7w.req.target(tgt)
 # let the data collector know the current target
ff.dbe.req.target(tgt)
 # tell the dbe where to point
#ff.k7w.req.baseline_mask("1","2","3")
ff.k7w.req.output_directory(ffui.defaults.ff_directories["data"])
ff.k7w.req.scan_tag("cal")
 # first scan has noise diode firing
ff.k7w.req.write_hdf5(1)
ff.k7w.req.capture_start()

ff.dbe.req.dbe_packet_count(900)
 # stream 10 minutes of data or until stop issued
ff.dbe.req.dbe_dump_rate(1)
 # correlator dump rate set to 1 Hz
ff.dbe.req.dbe_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbe.req.dbe_capture_start("stream")
 # start emitting data on stream "stream"

ff.ant1.sensor.pos_actual_scan_azim.register_listener(ff.dbesim.req.pointing_az, 0.5)
ff.ant1.sensor.pos_actual_scan_elev.register_listener(ff.dbesim.req.pointing_el, 0.5)
 # when the sensor value changes send an update to the listening function. Rate limited to 0.5 second updates.

scans = [ (-2,0.5) , (2,0) , (-2,-0.5) ]
 # raster scan
scan_duration = 60

ff.rfe.req.rfe3_rf15_noise_source_on("rfe31","pin",1,time.time(),0)
ff.dbe.req.dbe_noise(300)
time.sleep(5)
ff.rfe.req.rfe3_rf15_noise_source_on("rfe31","pin",0,time.time(),0)
ff.dbe.req.dbe_noise(0)
time.sleep(5)

ff.k7w.req.scan_id(1,"slew")
 # new scan, we are now slewing

ff.ant1.req.target(tgt)
  # send this target to the antenna. No time offset
ff.ant1.req.mode("POINT")
 # set mode to point
ff.ant1.wait("lock",True,300)
 # once we are on the target begin a new compound scan
 # (compound scan 0 will be the slew to the target, default scan tag is "slew")
scan_count = 2

for ix, scan in enumerate(scans):
    print "Scan Progress:",int((float(scan_count - 2) / (len(scans)*2))*100),"%"
    ff.ant1.req.scan_asym(-scan[0],scan[1],scan[0],scan[1],scan_duration)
    ff.ant1.wait("lock",True,300)
    ff.k7w.req.scan_id(scan_count,"scan")
     # mark this section as valid scan data
    ff.ant1.req.mode("SCAN")
    ff.ant1.wait("scan_status","after",300)
    if ix < len(scans):
        ff.k7w.req.scan_id(scan_count+1,"slew")
         # slewing to next raster pointg
    scan_count += 2
print "Scan complete."

ff.k7w.req.scan_id(scan_count-1,"cal")

ff.rfe.req.rfe3_rf15_noise_source_on("rfe31","pin",1,time.time(),0)
ff.dbe.req.dbe_noise(300)
time.sleep(5)
ff.rfe.req.rfe3_rf15_noise_source_on("rfe31","pin",0,time.time(),0)
ff.dbe.req.dbe_noise(0)
time.sleep(5)

files = ff.k7w.req.get_current_files(tuple=True)[1][2]
print "Data captured to",files
time.sleep(2)
ff.dbe.req.dbe_capture_stop("stream")
ff.k7w.req.capture_stop()
ff.disconnect()

