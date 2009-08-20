#!/usr/bin/python
# Print out the status (mode, lock, pointing etc...) for the filtered sensors on a specified proxy
# If no filter is given, all sensors with strategies are printed
# If a filter is given, all sensors in the filter without strategies have a periodic strategy set to the period param
#
# e.g. to print out all gps sensors on ancillary proxy
# run print_sensors -x anc -f gps
#
# e.g. to print out all lna sensors on rfe proxy
# python print_sensors.py -x rfe -f lna


import ffuilib as ffui
import time
import sys
from optparse import OptionParser
import StringIO
from ansi import gotoxy, clrscr, fg, bg, prnt, reset, underline, col

redirected = False
savestdout = sys.stdout
fstdout = None
x = StringIO.StringIO()


######## Methods #########

def stdout_redirect():
    global savestdout
    global fstdout
    global redirected
    savestdout = sys.stdout
    sys.stdout = StringIO.StringIO()
    redirected = True
    return savestdout

def stdout_restore():
    global savestdout
    global fstdout
    global redirected
    sOut = ""
    if redirected == True:
        sOut = sys.stdout.getvalue()
        sys.stdout = savestdout
        redirected = False
    return sOut

def get_time_str(time_f):
    if (str(time_f) == "") or (str(time_f) == "0.0"):
        return ""
    else:
        stime = time.ctime(float(time_f))
        i = stime.find(":")
        return stime[i-2:i+6] + "." + str(time_f).split(".")[1]


if __name__ == "__main__":

    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)

    parser.add_option('-x', '--proxy', dest='proxy', type="string", default="anc", metavar='PROXY',
                      help='Name of proxy to attach to (default="%default") as per the configuration file')
    parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-telescope.ini", metavar='INI',
                      help='Telescope configuration file to use in /var/kat/conf (default="%default")')
    parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_simulated_ff", metavar='SELECTED',
                      help='Selected configuration to use (default="%default")')
    parser.add_option('-f', '--filter', dest='filter', type='string', default='rfe31', metavar='FILTER',
                      help='Filter on sensors to print (default="%default")')
    parser.add_option('-p', '--period', dest='period', type='float', default='500', metavar='PERIOD',
                      help='Refresh period in milliseconds (default="%default")')
    parser.add_option('-o', '--override', dest='override', type='string', default='0', metavar='OVERRIDE',
                      help='If true existing sensor strategies will be overridden (default="%default")')
    (opts, args) = parser.parse_args()


    ff = ffui.tbuild(opts.ini_file, opts.selected_config)
    proxy  = ff.__dict__[opts.proxy] # Lookup rfe key in ff dictionary

    if opts.filter.startswith("all"):
        #Do not set any additional strategies as only sensors with strategies will be reported
        pass
    else:
        if opts.override.startswith("1"):
            #Set strategy on all sensors
            proxy.set_sensor_strategies(opts.filter,"period",str(opts.period).split(".")[0], override=True)
        else:
            #Only sets strategies on sensors without strategies
            proxy.set_sensor_strategies(opts.filter,"period",str(opts.period).split(".")[0])

    state = ["|","/","-","\\"]
    period_count = 0
    print "\n"
    try:
        clrscr()
        stdout_redirect()
        try:
            proxy.req_device_list()
        finally:
            s = stdout_restore()
            sys.stdout.write("Number of devices "+s)

        while True:
            gotoxy(6,1)
            print "Print filtered sensors from %s: %s %s %s" % (opts.proxy, opts.filter, state[period_count % 4], col("red")+time.ctime().replace("  "," ").split(" ")[3])+col("normal")
            print "%s %s %s %s %s" % ("Name".ljust(45), "Value".ljust(15), "Unit".ljust(7), "Value time".ljust(25), "Update time".ljust(25))
            if opts.filter.startswith("all"):
                sens = proxy.list_sensors(tuple=True, strategy=True)
            else:
                sens = proxy.list_sensors(opts.filter,tuple=True)

            for s in sens:
                name = s[0]
                val = s[1]
                valTime = s[2]
                type = s[3]
                units = s[4]
                updateTime = s[5]
                print "%s %s %s %s %s" % (name.ljust(45), str(val).ljust(15), str(units).ljust(7), get_time_str(valTime).ljust(25), get_time_str(updateTime).ljust(25) )
                sys.stdout.flush()

            #Wait, then do it all again
            time.sleep(opts.period/1000.0)
            period_count += 1

    except Exception,err:
        stdout_restore()
        print "\nError: Disconnecting... (",err,")"
        ff.disconnect()
    except KeyboardInterrupt:
        stdout_restore()
        print "\nDisconnecting..."
        ff.disconnect()
    print "Done."
