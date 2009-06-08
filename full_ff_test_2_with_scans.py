import ffuilib as ffui
# A dummy script that will form the basis of a functional pointing model test
 
#ticket = ffui.get_ticket(devices="dbe,k7w,ant1,ant2",level="control",tag="Full FF test stript ...")
 # requests control level access. Blocks until available.

ff = ffui.cbuild("ffuilib.katff.rc")
 # creates connection to ANT, DBE and RFE. Reads in default configuration and build targets and observatory lists

scans = [ (3.5, -6.0, 5.0, 10.0), ("-5", "2") ]
 # cross hair scan of the target from -5 to 5 degrees in x and y

scan_duration = 30
 # time (in sec) to take per leg of the crosshair

cal_sources = ff.sources.filter()  #tag='CALIBRATOR')
 # get calibrator sources from the built in catalog

ff.new_experiment(tag="Full FF Test 2 - with scans")
 # cleans environment and prepares for new experiment. Optional descriptive tag supplied

ff.dbe.configure(defaults=True, cfreq=1420.1)
 # configure the dbe using standard default apart from specifying the centre frequencys

ff.dbe.req_dbe_capture_destination("127.0.0.1:7010")
# instruct the simulator to send data to localhost

ffui.print_defaults()
 # print the current fringe finder configuration and status to the screen

scan_count = 0

# specify and drive to a fixed target
ff.ant1.req_target_azel(12,35)
#setup scan
ff.ant1.req_scan_asym(scans[0][0], scans[0][1], scans[0][2], scans[0][3], scan_duration)
#point and wait for lock
ff.ant1.req_mode("POINT")
ff.ant1.wait("lock","1",300)

# ready to start scan
# start data capture
ff.k7w.req_capture_start()
ff.dbe.req_dbe_capture_start("127.0.0.1:7010")
#do the scan
ff.ant1.req_mode("SCAN")
# wait for the scan to complete
ff.ant1.wait("mode","POINT",300)
#stop capture
ff.dbe.req_dbe_capture_stop()
ff.k7w.req_capture_stop()
scan_count += 1

#drive to next target and do a new scan
ff.ant1.req_target_azel(5,40)
#setup scan
ff.ant1.req_scan_sym(scans[1][0],scans[1][1],scan_duration)
#point and wait for lock
ff.ant1.req_mode("POINT")
ff.ant1.wait("lock","1",300)

 # ready to start scan
 # start data capture
ff.k7w.req_capture_start()
ff.dbe.req_dbe_capture_start("127.0.0.1:7010")
#do the scan
ff.ant1.req_mode("SCAN")
# wait for the scan to complete
ff.ant1.wait("mode","POINT",300)
#stop capture
ff.dbe.req_dbe_capture_stop()
ff.k7w.req_capture_stop()
scan_count += 1

#ffui.release_ticket(ticket)
# we are done so clean up

ff.disconnect()
# exit

