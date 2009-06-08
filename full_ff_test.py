import ffuilib as ffui
# A dummy script that will form the basis of a functional pointing model test
 
#ticket = ffui.get_ticket(devices="dbe,k7w,ant1,ant2",level="control",tag="Full FF test stript ...")
 # requests control level access. Blocks until available.

ff = ffui.cbuild("ffuilib.katff.rc")
 # creates connection to ANT, DBE and RFE. Reads in default configuration and build targets and observatory lists

scans = [ ("-5","0") , ("0","-5") ]
 # cross hair scan of the target from -5 to 5 degrees in x and y

scan_duration = 30
 # take 30s per leg of the crosshair

cal_sources = ff.sources.filter()  #tag='CALIBRATOR')
 # get calibrator sources from the built in catalog

ff.new_experiment(tag="Full FF Test")
 # cleans environment and prepares for new experiment. Optional descriptive tag supplied

ff.dbe.configure(defaults=True, cfreq=1420.1)
 # configure the dbe using standard default apart from specifying the centre frequencys

ff.dbe.req_dbe_capture_destination("127.0.0.1:7010")
# instruct the simulator to send data to localhost

ffui.print_defaults()
 # print the current fringe finder configuration and status to the screen


# specify and drive to a fixed target
ff.ant1.req_target_azel(12,35)
ff.ant1.req_mode("POINT")
ff.ant1.wait("lock","1",300)

 # ready to start scan
scan_count = 0
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

## drive to new target, noting start time
#time.sleep(4.0) # track start az-el position for a few seconds


#drive to next target and do a new scan
ff.ant1.req_target_azel(-5,40)
ff.ant1.wait("lock","1",300)

 # ready to start scan
scan_count += 1
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

#
#ra = ff.ant1.sensor_pos_request_scan_ra.value
#dec = ff.ant1.sensor_pos_request_scan_dec.value
#ff.ant1.req_target_radec(ra,dec)
#time.sleep(40.0) # track ra-dec target for a while



#source = cal_sources.filterpop(20)
# # find the first available source that is above elevation limit 70
# 
# 
#while source is not None:
#    print "Full FF test - pointing source: ",source.name
#    ff.ant1.req_mode("STOP")
#    ff.ant1.req_target_named(source.name)
#     # send this target to the antenna. No time offset
#    ff.ant1.req_mode("POINT")
#     # set mode to point
#    ff.ant1.wait("lock","1",300)
#     # wait for lock
#    scan_count = 0
#    
#    for scan in scans:
#        print "Scan Progress:",int((float(scan_count) / len(scans))*100),"%"
#        ff.ant1.req_offset(scan[0],scan[1],scan_duration)
#        ff.ant1.wait("lock","1",300)
#         # ready to start scan
#        ff.k7w.req_capture_start()
#        ff.dbe.req_capture_start()
#         # start data capture
#        ff.ant1.req_mode("SCAN")
#        ff.ant1.wait("mode","POINT",300)
#         # wait for the scan to complete
#        ff.dbe.req_capture_stop()
#        ff.k7w.req_capture_stop()
#        scan_count += 1
#    
#    print "Scan complete."
#    source = cal_sources.filterpop(70)

#ffui.release_ticket(ticket)
 # we are done so clean up

ff.disconnect()
 # exit

#ff.dp.process(ff.dbe.data[-1], alg="pointing_model_gen", meta=ff.meta)
 # process the most recent data capture block. Using point model algorithm (blocking)

#print "Scan completed. Data captured:",ff.dbe.data
 # prints a listing of the data captured:
 #
 # Data for experiment 079483B3-CA5B-4BC0-A77F-E121BEA7F408
 # ID  Start Time      Duration  Data Size(GB)  Tag
 # 1   12:21 04/02/09  65        43.23          Source: Vela, Scan: (-5,-5),(5,-5)
 # 2   12:23 04/02/09  65        43.23          Source: Vela, Scan: (-5,0),(5,0)
 # etc...

#print "Results:", ff.dp.results
 # prints a listing of the data processing output
 #
 # Results for experiment 079483B3-CA5B-4BC0-A77F-E121BEA7F408
 # ID Algorithm           Data Size(GB)  OffsetX        OffsetY
 # 1  pointing_model_gen  0.54           0.32 +/- 0.03  0.11 +/- 0.005
 # etc...

