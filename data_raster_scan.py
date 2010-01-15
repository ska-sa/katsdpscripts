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
# (kat-launch.py and ffsocket.py use the defaults: -i cfg-local.ini -s local_ff)

import ffuilib as ffui
import time
from optparse import OptionParser



if __name__ == "__main__":

    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)

    parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                      help='Telescope configuration file to use in conf directory (default="%default")')
    parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                      help='Selected configuration to use (default="%default")')
    parser.add_option('-a', '--ants', dest='ants', type="string", default="ant1,ant2", metavar='ANTS',
                      help='Comma seperated list of antennas to include in scan (default="%default")')
    (opts, args) = parser.parse_args()


    ff = ffui.tbuild(opts.ini_file, opts.selected_config)
    # make fringe fingder connections

    tgt = 'Takreem,azel,20,30'
    # the virtual target that the dbe simulator contains

    ants = []
    print "====================Including antennas - ",
    for ant_x in opts.ants.split(","):
        ant = eval("ff."+ant_x)
        ants.append(ant)
        print ant.name,
    print

    ff.dbe.req.dbe_capture_stop()
    for ant_x in ants:
        ant_x.req.mode("STOP")
    time.sleep(0.5)
    # cleanup any existing experiment

    ff.dbe.req.k7w_target(tgt)
    ff.dbe.req.target(tgt)
    # let the data collector know the current target
    ff.dbe.req.k7w_output_directory(ffui.defaults.ff_directories["data"])
    ff.dbe.req.k7w_write_hdf5(1)
    ff.dbe.req.k7w_capture_start()

    ff.dbe.req.dbe_packet_count(900)
    # stream 10 minutes of data or until stop issued
    ff.dbe.req.dbe_dump_rate(1)
    # correlator dump rate set to 1 Hz
    ff.dbe.req.dbe_capture_destination("stream","127.0.0.1:7010")
    # create a new data source labelled "stream". Send data to localhost on port 7010
    ff.dbe.req.dbe_capture_start("stream")
    # start emitting data on stream "stream"

    for ant_x in ants:
        ant_x.sensor.pos_actual_scan_azim.register_listener(ff.dbe.req.dbe_pointing_az, 0.5)
        ant_x.sensor.pos_actual_scan_elev.register_listener(ff.dbe.req.dbe_pointing_el, 0.5)
        # when the sensor value changes send an update to the listening function. Rate limited to 0.5 second updates.

    scans = [ (-2,0.5) , (2,0) , (-2,-0.5) ]
    # raster scan
    scan_duration = 240

    for ant_x in ants:
        ant_x.req.target(tgt)
        # send this target to the antenna. No time offset
        ant_x.req.mode("POINT")
        # set mode to point

    for ant_x in ants:
        ant_x.wait("lock",True,300)

    ff.dbe.req.k7w_compound_scan_id(1)
    # once we are on the target begin a new compound scan
    # (compound scan 0 will be the slew to the target, default scan tag is "slew")
    scan_count = 1

    for scan in scans:
        print "Scan Progress:",int((float(scan_count) / len(scans)*2)*100),"%"
        for ant_x in ants:
            ant_x.req.scan_asym(-scan[0],scan[1],scan[0],scan[1],scan_duration)
        for ant_x in ants:
            ant_x.wait("lock",True,300)

        ff.dbe.req.k7w_scan_id(scan_count,"scan")
        # mark this section as valid scan data
        for ant_x in ants:
            ant_x.req.mode("SCAN")
        for ant_x in ants:
            ant_x.wait("scan_status","after",300)
        ff.dbe.req.k7w_scan_id(scan_count+1,"slew")
        # slewing to next raster pointg
        scan_count += 2
    print "Scan complete."

    files = ff.dbe.req.k7w_get_current_files(tuple=True)[1][2]
    print "Data captured to",files
    time.sleep(2)
    ff.dbe.req.dbe_capture_stop("stream")
    ff.dbe.req.k7w_capture_stop()
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
