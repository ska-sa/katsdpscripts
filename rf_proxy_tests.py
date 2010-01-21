import ffuilib as ffui
# An example script for accessing and controlling RF device and pedX devices: Cryo, RFE3, RFE5, RFE7

ff = ffui.tbuild("cfg-local.ini", "local_ff")

##################################
#Add rf and pedX proxy tests here
##################################

rfe7 = ff.__dict__["rfe7"] # Lookup rfe7 key in ff dictionary
ped1 = ff.__dict__["ped1"] # Lookup ped1 key in ff dictionary
ped2 = ff.__dict__["ped2"] # Lookup ped2 key in ff dictionary

print "".ljust(50,"=")
print "ACCESS THROUGH ffuilib - SENSORS"
print "".ljust(50,"=")

# List specific sensor
ped1.list_sensors("cryo.ambient.temperature")
# List sensors with filter
ped1.list_sensors("rfe3")
# Only those sensors with strategies
ped1.list_sensors("rfe3",strategy=True)
# Return result in a tuple for processing
sens_list = ped2.list_sensors(filter="cryo", tuple=True)

print "".ljust(50,"=")
print "ACCESS THROUGH ffuilib - REQUESTS"
print "".ljust(50,"=")

# List all
ff.ped1.req.sensor_list()
# ff.peds.req.sensor_list()

# Requests filtered, and returned in tuple
ped1.list_requests("noise")
# rfe.list_requests("noise")

# Return results in a tuple for processing
ped1.list_requests(filter="lna", tuple=True)
# req_list = rfe.list_requests(filter="lna", tuple=True)

print "".ljust(50,"=")
print "ACCESS THROUGH ffuilib - STRATEGIES"
print "".ljust(50,"=")

# Set all cryo1 sensors to periodic 2000
ped1.set_sensor_strategies("cryo","period","2000")
# Specific sensor - differential
ped1.set_sensor_strategies("cryo.coldfinger.temperature", strategy="differential",param="1000", override=True)
# Specific sensor - remove strategy
ped1.set_sensor_strategies("cryo.coldfinger.temperature", strategy="none", override=True)
# Specific sensor - event
ped1.set_sensor_strategies("cryo.coolingfans.error",strategy="event", override=True)
# Specific sensor - event with rate limit
ped1.set_sensor_strategies("cryo.ambient.temperature", strategy="event", param="1000", override=True)

print "".ljust(50,"=")
print "katcp - HELP"
print "".ljust(50,"=")

# katcp - request help
ped1.req.help()
# ff.peds.req.help()
ped1.req.help("sensor-value")
# ff.peds.req.help("sensor-value")

print "".ljust(50,"=")
print "katcp - SENSOR LIST"
print "".ljust(50,"=")

# katcp - sensor list all
# rfe7.req.sensor_list()
# katcp - sensor list with pattern
# rfe7.list_sensors("temp")
rfe7.req.sensor_list("/temp/")

print "".ljust(50,"=")
print "katcp - SENSOR VALUES"
print "".ljust(50,"=")

# katcp -sensor value - Specific sensor
rfe7.req.sensor_value("rfe7.psu.cam5.volt")
# katcp - sensor value - Sensors with a pattern - start and end with /
rfe7.req.sensor_value("/noise/")

print "".ljust(50,"=")
print "katcp - SENSOR SAMPLING"
print "".ljust(50,"=")

# katcp - sensor sampling
ped1.req.sensor_sampling("cryo.ambient.temperature","event", "1000")
ped1.req.sensor_sampling("cryo.ambient.temperature","period", "1500")

print "".ljust(50,"=")
print "katcp - GROUPED COMMANDS"
print "".ljust(50,"=")

# Switch all on
ped1.req.rfe3_psu_on(1)
# ff.peds.req.rfe3_psu_on(1) # To Check
# rfe.req.rfe3_psu_on("all",1)

# Switch specific instance off
ped2.req.rfe3_psu_on(0)
# rfe.req.rfe3_psu_on("rfe32",0)

print "".ljust(50,"=")
print "katcp - COMMON REQUESTS"
print "".ljust(50,"=")

# katcp - common requests
rfe7.req.client_list()
rfe7.req.device_list()
rfe7.req.watchdog() # katcp - watchdog "ping"
rfe7.req.scheduler_mode()
# katcp - Lifecycle commands
#rfe7.req.halt()
#rfe7.req.restart()

print "".ljust(50,"=")
print "katcp - LOG LEVEL"
print "".ljust(50,"=")

# Log level:  {'all', 'trace', 'debug', 'info', 'warn', 'error', 'fatal', 'off'}, optional
# katcp - Proxy log level
rfe7.req.log_level() #Get the current log level
rfe7.req.log_level("debug")
#katcp - Log level for specific device
ped1.req.log_level("cryo")
ped1.req.log_level("cryo", "warn")

# exit
ff.disconnect()

