import ffuilib as ffui
# A dummy script that will form the basis of a functional pointing model test
 
#ticket = ffui.get_ticket(devices="all",level="control",tag="RF Proxy ffuilib tests")
 # requests control level access to the telescope. Blocks until available.

ff = ffui.tbuild("cfg-telescope.ini", "local_rf_only")
 # creates connection to ANT, DBE and RFE. Reads in default configuration and build targets and observatory lists

scans = [ ("-5","0") , ("0","-5") ]
 # cross hair scan of the target from -5 to 5 degrees in x and y

scan_duration = 30
 # take 30s per leg of the crosshair

cal_sources = ff.sources.filter(tags='CALIBRATOR')
 # get calibrator sources from the built in catalog

ff.new_experiment(tag="RF Proxy ffuilib tests")
 # cleans environment and prepares for new experiment. Optional descriptive tag supplied

ff.dbe.configure(defaults=True, cfreq=1420.1)
 # configure the dbe using standard default apart from specifying the centre frequency

ff.dbe.req_destination_ip("192.168.6.40")
 # instruct the simulator to send data to dss-dp1

ffui.print_defaults()
 # print the current fringe finder configuration and status to the screen


########################
#Add rf proxy tests here
########################



#ffui.release_ticket(ticket)
# we are done so clean up

ff.disconnect()
 # exit

