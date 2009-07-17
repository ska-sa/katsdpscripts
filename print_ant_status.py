#!/usr/bin/python
# Print out the status (mode, lock, pointing etc...) for the specified antenna 

import ffuilib as ffui
import time
import sys
from optparse import OptionParser

if __name__ == "__main__":

    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)

    parser.add_option('-a', '--antenna', dest='ant', type="string", default="ant1", metavar='ANTENNA',
                      help='antenna proxy to attach to (default="%default") as per the rc file')
    parser.add_option('-i', '--ini', dest='ini_file', type="string", default="cfg-telescope.ini", metavar='INI-FILE',
                      help='load system description from INI-FILE (name is relative to CONF folder; default=%default)')
    parser.add_option('-s', '--selected', dest='selected_config', type="string", default="local_ant_only", metavar='SELECTED-CONFIG',
                      help='selected configuration from INI-FILE (default=%default)')
    (opts, args) = parser.parse_args()

    ff = ffui.tbuild(opts.ini_file,opts.selected_config)
    ant  = ff.__dict__[opts.ant] # some Simon magic
    
    state = ["|","/","-","\\"]
    period_count = 0
    print "\n"
    try:
        while True:
            lock = ant.sensor_lock.value == '1' and 'True' or 'False'
            mode = ant.sensor_mode.value
            scan = ant.sensor_scan_status.value
            request_az = float(ant.sensor_pos_request_scan_azim.value)
            request_el = float(ant.sensor_pos_request_scan_elev.value)
            request_ra = float(ant.sensor_pos_request_scan_ra.value)
            request_dec = float(ant.sensor_pos_request_scan_dec.value)
            actual_az = float(ant.sensor_pos_actual_scan_azim.value)
            actual_el = float(ant.sensor_pos_actual_scan_elev.value)
            error_az = abs(actual_az - request_az)
            error_el = abs(actual_el - request_el)
            status = "\r%s: %s %s Mode:\033[34m%s\033[0m Scan:\033[34m%s\033[0m Lock:\033[34m%s\033[0m  Req[Ra:\033[32m%.2F\033[0m Dec:\033[32m%.2F\033[0m] Req[Az:\033[32m%.2F\033[0m El:\033[32m%.2F\033[0m]  Act[Az:\033[34m%.2F\033[0m El:\033[34m%.2F\033[0m]  Err[Az:\033[31m%.2F\033[0m El:\033[31m%.2F\033[0m]" % (opts.ant, state[period_count % 4], time.ctime().replace("  "," ").split(" ")[3], mode, scan, lock, request_ra, request_dec, request_az, request_el, actual_az, actual_el, error_az, error_el)

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
