#!/usr/bin/python
# Drive antenna to two targets and then make a pointing error plot

import katuilib as katui
import numpy as np
import pylab

kat = katui.tbuild("cfg-local.ini","local_ant_only")
 # make fringe fingder connections

kat.ant2.req.target_azel(20.31,30.45)
 # send an az/el target to antenna 2

kat.ant2.req.mode("POINT")
 # switch to mode point

kat.ant2.wait("lock","1",120)
 # wait for lock to be achieved (timeout=120 seconds)

kat.ant2.req.target_azel(40.2,60.32)
 # send a new az/el target

kat.ant2.wait("lock","1",120)
 # wait for lock again

 # produce custom pointing error plot
 # each sensor has local history

req_az = kat.ant2.sensor.pos_request_scan_azim.get_cached_history()
req_el = kat.ant2.sensor.pos_request_scan_elev.get_cached_history()
actual_az = kat.ant2.sensor.pos_actual_scan_azim.get_cached_history()
actual_el = kat.ant2.sensor.pos_actual_scan_elev.get_cached_history()

az_error = np.array(actual_az[1]) - np.array(req_az[1][:len(actual_az[1])])
el_error = np.array(actual_el[1]) - np.array(req_el[1][:len(actual_el[1])])

pylab.plot(actual_az[0], az_error)
pylab.plot(actual_el[0], el_error)
pylab.show()

raw_input("Hit enter to terminate...")
kat.disconnect()
