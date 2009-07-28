#!/usr/bin/python
# Raster scan across a simulated target producing scan data for signal displays and loading into scape

# Once-off setup:
# For the augment part to work, need some webserver setup from cmd line:
#   cd /Library/WebServer/Documents/
#   sudo ln -s /var/kat/central_monitoring/ central_monitoring
#   sudo apachectl start
# ensure hat you have a data dir e.g. /var/kat/data below

# Startup in different terminals:
# start C dbe simulator: ~/svnDS/code/ffinder/trunk/src/simulators/dbe/dbe_server 8000
# start k7 writer: cd /var/kat/data; ~/svnDS/code/ffinder/trunk/src/streaming/k7writer/k7w_server 8001
# start the system: kat-launch.py
# start the central monitor: ~/svnDS/code/ffinder/trunk/src/services/monitoring/central_monitor.py
# and signal display server: ~/svnDS/code/ffinder/trunk/src/streaming/sdisp/ffsocket.py
# (kat-launch.py and ffsocket.py use the defaults: -i cfg-telescope.ini -s local-simulated-ff)

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

ff.dbe.req_dbe_packet_count(900)
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

scans = [ (-2,0.5) , (2,0) , (-2,-0.5) ]
 # raster scan
scan_duration = 240

ff.ant1.req_target(tgt)
  # send this target to the antenna. No time offset
ff.ant1.req_mode("POINT")
 # set mode to point
ff.ant1.wait("lock",True,300)
ff.k7w.req_compound_scan_id(1)
 # once we are on the target begin a new compound scan
 # (compound scan 0 will be the slew to the target, default scan tag is "slew")
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

# now augment the hdf5 file with metadata (pointing info etc):
#   ~/svnDS/code/ffinder/trunk/src/streaming/k7augment/augment.py -d /var/kat/data -f [xxx.h5]

# load into scape from within python:
#   import scape
#   import pylab as pl
#   d = scape.DataSet("[xxx.h5]") # load data from file into dataset
#   print d
#   scape.plot_compound_scan_on_target(d.compscans[1])
#   pl.show()
#   d = d.select(labelkeep="scan") # get rid of the slew data from the dataset
#   print d
#   print d.compscans[0]
#   scape.plot_compound_scan_on_target(d.compscans[0])
