#!/usr/bin/python

import ffuilib as ffui
import time
import sys

ff = ffui.cbuild("ffuilib.ant_only.rc")
state = ["|","/","-","\\"]
period_count = 0
print "\n"
try:
    while True:
        lock = ff.ant1.sensor_lock.get_value() == '1' and 'True' or 'False'
        mode = ff.ant1.sensor_mode.get_value()
        desired_az = float(ff.ant1.sensor_pos_request_tgt_azim.get_value())
        desired_el = float(ff.ant1.sensor_pos_request_tgt_elev.get_value())
        actual_az = float(ff.ant1.sensor_pos_actual_tgt_azim.get_value())
        actual_el = float(ff.ant1.sensor_pos_actual_tgt_elev.get_value())
        error_az = abs(actual_az - desired_az)
        error_el = abs(actual_el - desired_el)
        status = "\r%s Time:%s  Mode:%s  Lock:%s  Requested[Az:\033[32m%.2F\033[0m El:\033[32m%.2F\033[0m]  Actual[Az:\033[33m%.2F\033[0m El:\033[33m%.2F\033[0m]  Error[Az:\033[31m%.2F\033[0m El:\033[31m%.2F\033[0m]" % (state[period_count % 4], time.ctime().split(" ")[3], mode, lock, desired_az, desired_el, actual_az, actual_el, error_az, error_el)

        sys.stdout.write(status)
        sys.stdout.flush()
        period_count += 1
        time.sleep(0.5)
except Exception,err:
    print "Error: Disconnecting... (",err,")"
    ff.disconnect()
except KeyboardInterrupt:
    print "\nDisconnecting..."
    ff.disconnect()
print "Done."
