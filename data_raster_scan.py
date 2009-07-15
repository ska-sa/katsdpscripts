#!/usr/bin/python
# Raster scan across a simulated target producing scan data

# Startup: 
# start C dbe simulator: ~/svnDS/code/ffinder/trunk/src/simulators/dbe/dbe_server2 8000
# start k7 writer: ~/svnDS/code/ffinder/trunk/src/streaming/k7writer/k7w_server 8001
# start the system: kat-launch.py
# and signal display server: ~/svnDS/code/ffinder/trunk/src/streaming/sdisp/ffsocket.py
# (kat-launch.py and ffsocket.py use the defaults: -i cfg-telescope.ini -s local-simulated-ff)

import ffuilib as ffui
import time

# make fringe finder connections
ff = ffui.tbuild("cfg-user.ini", "local_ff_client_sim")

ff.dbesim.req_capture_stop()

time.sleep(2)

# write data into hdf5 file
ff.k7w.req_write_hdf5(1)
#ff.k7w.req_output_directory("~") # this does not work at present (defaults to current dir from which k7w is launched)

# startup the k7 capture process
ff.k7w.req_capture_start()
ff.k7w.req_compound_scan_id(0)
ff.k7w.req_compound_scan_tag("Raster scan of hard-coded dbe sim target at 20,30 az/el")

# stream n packets of data
ff.dbe.req_dbe_packet_count(50000)

# stream data at 300 kbps. Approx 1s per integration
ff.dbe.req_dbe_rate(300)

# create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbe.req_dbe_capture_destination("stream","127.0.0.1:7010")

# start emitting data on stream "stream"
ff.dbe.req_dbe_capture_start("stream")

# when the sensor value changes send an update to the listening function. Rate limited to 0.5 second updates.
ff.ant1.sensor_pos_actual_scan_azim.register_listener(ff.dbesim.req_pointing_az, 0.5)
ff.ant1.sensor_pos_actual_scan_elev.register_listener(ff.dbesim.req_pointing_el, 0.5)

scans = [ (2,2) , (-2,1) , (2,0) , (-2,-1) , (2,-2)]
#scans = [ (2,2) ]

# raster scan

scan_duration = 60
#scan_duration = 20

# send this target to the antenna. No time offset (hardcoded target in DBE sim at 20 az,30 el)
ff.ant1.req_target_azel(20,30)

# set mode to point
ff.ant1.req_mode("POINT")
ff.ant1.wait("lock",True,300)
scan_count =  0

for scan in scans:
    print "Scan Progress:",int((float(scan_count) / len(scans))*100),"%"
    ff.k7w.req_scan_id(scan_count)
    ff.k7w.req_scan_tag("scan ", str(scan_count), " of raster scan")
    ff.ant1.req_scan_asym(-scan[0],scan[1],scan[0],scan[1],scan_duration)

    # ready to start scan
    ff.ant1.wait("lock",True,300)
    ff.ant1.req_mode("SCAN")

    # wait for the scan to complete
    ff.ant1.wait("scan_status","after",300)
    scan_count += 1
print "Scan complete."

#stop capture
ff.dbesim.req_dbe_capture_stop()
ff.k7w.req_capture_stop()            

ff.disconnect()

# now augment the hdf5 file with metadata (TODO) 
# ~/svnDS/code/ffinder/trunk/src/streaming/k7augment/augment.py -h

# load into scape (TODO)....

