import ffuilib as ff

ff.access(devices=all,level=control,tag="Simon Ratcliffe - Engineering Tests...")
 # requests control level access to the telescope. Blocks until available.

ff.build()
 # creates connection to ANT, DBE and RFE. Reads in default configuration and build targets and observatory lists

scans = [ [(-5,-5),(5,-5)] , [(-5,0),(5,0)] , [(-5,5),(5,5)] ]
 # do three scan across the target from -5 to 5 in x for y -5, 0, 5

available_sources = ff.sources(elevation_limit=20, keyword=CALIBRATOR).sort(keyword=elevation)
 # get a list of calibrator sources the system knows about currently above 20 degrees

ff.new_experiment(tag="Pointing Model Scan")
 # cleans environment and prepares for new experiment. Optional descriptive tag supplied

ff.dbe.configure(defaults=True, cfreq=1420.1)
 # configure the dbe using standard default apart from specifying the centre frequency

ff.print_config()
 # print the current fringe finder configuration and status to the screen

for source in available_sources:
    ff.point_wait(source,antenna=1)
     # points antenna 1 to source and waits for lock
	for scan in scans:
		ff.scan(scan[0],scan[1],duration_s=60,antenna=1,start=False)
		 # scan antenna 1 across source in scan_duration seconds. Do not start automatically just slew to start position
		ff.dbe.capture_start(tag="Source: "+source+", Scan: "+str(scan[0])+","+str(scan[1]))

		ff.dp.connect(channel=1,averaging=1,baseline=1,product=corr_phase,name=f1a1corrp)
		
		ff.scan_start_wait()
		 # start scan and block until scan is complete
		ff.dbe.capture.stop(min_elapsed=60)
		 # halts the capture after a minimum of 60s have elapsed (blocking)
		ff.dp.process(ff.dbe.data[-1], alg="pointing_model_gen", meta=ff.meta)
		 # process the most recent data capture block. Using point model algorithm (blocking)
		
ff.ant.req_sensor_sampling("actual_az","period","100")

pylAB.plot(ff.meta.ant.actual_az)
print "Scan completed. Data captured:",ff.dbe.data
 # prints a listing of the data captured:
 #
 # Data for experiment 079483B3-CA5B-4BC0-A77F-E121BEA7F408
 # ID  Start Time      Duration  Data Size(GB)  Tag
 # 1   12:21 04/02/09  65        43.23          Source: Vela, Scan: (-5,-5),(5,-5)
 # 2   12:23 04/02/09  65        43.23          Source: Vela, Scan: (-5,0),(5,0)
 # etc...

print "Results:", ff.dp.results
 # prints a listing of the data processing output
 #
 # Results for experiment 079483B3-CA5B-4BC0-A77F-E121BEA7F408
 # ID Algorithm           Data Size(GB)  OffsetX        OffsetY
 # 1  pointing_model_gen  0.54           0.32 +/- 0.03  0.11 +/- 0.005
 # etc...

