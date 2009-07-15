#!/usr/bin/python
# Raster scan across a simulated target

# Startup: 
# start C dbe simulator: ~/svnDS/code/ffinder/trunk/src/simulators/dbe/dbe_server2 8000
# start k7 writer: ~/svnDS/code/ffinder/trunk/src/streaming/k7writer/k7w_server 8001
# start the system: kat-launch.py
# and signal display server: ~/svnDS/code/ffinder/trunk/src/streaming/sdisp/ffsocket.py
# (kat-launch.py and ffsocket.py use the defaults: -i cfg-telescope.ini -s local-simulated-ff)

import ffuilib as ffui
import time

ff = ffui.tbuild("cfg-user.ini", "local_ff_client_sim")
 # make fringe finder connections

ff.dbesim.req_capture_stop()

time.sleep(2)

ff.k7w.req_capture_start()
 # startup the k7 capture process

ff.dbe.req_dbe_packet_count(50000)
 # stream 5000 packets of data
ff.dbe.req_dbe_rate(300)
 # stream data at 300 kbps. Approx 1s per integration
ff.dbe.req_dbe_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbe.req_dbe_capture_start("stream")
 # start emitting data on stream "stream"

ff.ant1.sensor_pos_actual_scan_azim.register_listener(ff.dbesim.req_pointing_az, 0.5)
ff.ant1.sensor_pos_actual_scan_elev.register_listener(ff.dbesim.req_pointing_el, 0.5)
 # when the sensor value changes send an update to the listening function. Rate limited to 0.5 second updates.

scans = [ (2,2) , (2,1) , (2,0) , (2,-1) , (2,-2)]
 # raster scan
scan_duration = 60

ff.ant1.req_target_azel(20,30)
  # send this target to the antenna. No time offset (hardcoded target in DBE sim at 20 az,30 el)
ff.ant1.req_mode("POINT")
 # set mode to point
ff.ant1.wait("lock",True,300)
scan_count =  0

for scan in scans:
    print "Scan Progress:",int((float(scan_count) / len(scans))*100),"%"
    ff.ant1.req_scan_asym(-scan[0],scan[1],scan[0],scan[1],scan_duration)
    ff.ant1.wait("lock",True,300)
     # ready to start scan
    ff.ant1.req_mode("SCAN")
    ff.ant1.wait("scan_status","after",300)
     # wait for the scan to complete
    scan_count += 1
print "Scan complete."

ff.disconnect()


