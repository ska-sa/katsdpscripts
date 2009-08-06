import ffuilib as ffui
# An example script for accessing and controlling RF devices: Cryo, RFE3, RFE5, RFE7

ff = ffui.tbuild("cfg-telescope.ini", "local_rf_only")

########################
#Add rf proxy tests here
########################

rfe  = ff.__dict__["rfe"] # Lookup rfe key in ff dictionary


print "".ljust(50,"=")
print "ACCESS THROUGH ffuilib - SENSORS"
print "".ljust(50,"=")

#List specific sensor
rfe.list_sensors("rfe71.cryo1.ambient.temperature")
#List sensors with filter
rfe.list_sensors("rfe31")
#Only those sensors with strategies
rfe.list_sensors("rfe31",strategy=True)
#Return result in a tuple for processing
sens_list = rfe.list_sensors(filter="rfe71.cryo2", tuple=True)


print "".ljust(50,"=")
print "ACCESS THROUGH ffuilib - REQUESTS"
print "".ljust(50,"=")

# List all
rfe.list_requests()
#Requests filtered, and returned in tuple
rfe.list_requests("noise")
#Return results in a tuple for processing
req_list = rfe.list_requests(filter="lna", tuple=True)

print "".ljust(50,"=")
print "ACCESS THROUGH ffuilib - STRATEGIES"
print "".ljust(50,"=")

#Set all cryo1 sensors to periodic 2000
rfe.set_sensor_strategy("cryo1","period","2000")
#Specific sensor - differential
rfe.set_sensor_strategy("rfe71.cryo1.coldfinger.temperature", strategy="differential",param="1000", override=True)
#Specific sensor - remove strategy
rfe.set_sensor_strategy("rfe71.cryo1.coldfinger.temperature", strategy="none", override=True)
#Specific sensor - event
rfe.set_sensor_strategy("rfe71.cryo1.coolingfans.error",strategy="event", override=True)
#Specific sensor - event with rate limit
rfe.set_sensor_strategy("rfe71.cryo1.ambient.temperature", strategy="event", param="1000", override=True)

print "".ljust(50,"=")
print "katcp - HELP"
print "".ljust(50,"=")

# katcp - request help
rfe.req_help()
rfe.req_help("sensor-value")

print "".ljust(50,"=")
print "katcp - SENSOR LIST"
print "".ljust(50,"=")

# katcp - sensor list all
rfe.req_sensor_list()
# katcp - sensor list with pattern
rfe.req_sensor_list("/temperature/")

print "".ljust(50,"=")
print "katcp - SENSOR VALUES"
print "".ljust(50,"=")

#katcp -sensor value - Specific sensor
rfe.req_sensor_value("rfe71.psu.cam5.volt")
#katcp - senosr value - Sensors with a pattern - start and end with /
rfe.req_sensor_value("/noise/")

print "".ljust(50,"=")
print "katcp - SENSOR SAMPLING"
print "".ljust(50,"=")

# katcp - sensor sampling
rfe.req_sensor_sampling("rfe71.cryo1.ambient.temperature","event", "1000")
rfe.req_sensor_sampling("rfe71.cryo1.ambient.temperature","period", "1500")

print "".ljust(50,"=")
print "katcp - GROUPED COMMANDS"
print "".ljust(50,"=")

#Switch all on
rfe.req_rfe3_psu_on("all",1)
#Switch specific instance off
rfe.req_rfe3_psu_on("rfe71.rfe32",0)

print "".ljust(50,"=")
print "katcp - COMMON REQUESTS"
print "".ljust(50,"=")

#katcp - common requests
rfe.req_client_list()
rfe.req_device_list()
rfe.req_watchdog() #katcp - watchdog "ping"
rfe.req_scheduler_mode()
#katcp - Lifecycle commands
#rfe.req_halt()
#rfe.req_restart()

print "".ljust(50,"=")
print "katcp - LOG LEVEL"
print "".ljust(50,"=")

#Log level:  {'all', 'trace', 'debug', 'info', 'warn', 'error', 'fatal', 'off'}, optional
#katcp - Proxy log level
rfe.req_log_level() #Get the current log level
rfe.req_log_level("debug")
#katcp - Log level for specific device
rfe.req_log_level("rfe71.cryo1")
rfe.req_log_level("rfe71.cryo1", "warn")


# exit
ff.disconnect()

