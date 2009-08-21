#!/usr/bin/python
# Print out the info for sensors in the specified filter on a specified proxy
# If a filter is given, those sensors in the filter without strategies have a periodic strategy set to the period param.
# But when the override flag is set on the command line (-o 1), then _all_ sensors have periodic strategy set,
# not only those without strategies.
# If filter is set to "all", those sensors that have strategies defined are printed
# But when the override flag is set _all_ sensors  the command line (-o 1), then _all_ sensors have periodic strategy set,
# not only those without strategies.
#
# e.g. to print out all gps sensors on ancillary proxy
# run print_sensors -x anc -f gps -o 1
#
# e.g. to print out all lna sensors on rfe proxy
# python print_sensors.py -x rfe -f lna


import ffuilib as ffui
import time
import sys
import select
from optparse import OptionParser
import StringIO
from ansi import gotoxy, clrscr, fg, bg, prnt, reset, underline, col

redirected = False
savestdout = sys.stdout
fstdout = None
x = StringIO.StringIO()


import termios,os,string



#    import sys
#    import select
#    import tty
#    import termios
#
#    def isData():
#            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
#
#    old_settings = termios.tcgetattr(sys.stdin)
#    try:
#            tty.setcbreak(sys.stdin.fileno())
#
#            i = 0
#            while 1:
#                    print i
#                    i += 1
#
#                    if isData():
#                            c = sys.stdin.read(1)
#                            if c == '\x1b':         # x1b is ESC
#                                    break
#
#    finally:
#            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

######## Methods #########

def getKeyIf():
    #Non-blocking key - returns 0 if no key available
    oldattr = termios.tcgetattr(0)
    try:
        attr = termios.tcgetattr(0)
        attr[2] = (attr[2] & ~termios.NLDLY) | termios.NL0
        attr[3] = attr[3] & ~(termios.ICANON|termios.ECHO)
        termios.tcsetattr(0,termios.TCSANOW,attr)
        return os.read(0,1)
    finally:
        termios.tcsetattr(0,termios.TCSANOW,oldattr)
    return 0

def getKey():
    #Get a key from stdin - block until key is available
    c = raw_input()
    return c

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
    parser.add_option('-p', '--period', dest='period', type='float', default='1000', metavar='PERIOD',
                      help='Refresh period in milliseconds (default="%default")')
    parser.add_option('-o', '--override', dest='override', type='string', default='0', metavar='OVERRIDE',
                      help='If true existing sensor strategies will be overridden (default="%default")')
    (opts, args) = parser.parse_args()


    ff = ffui.tbuild(opts.ini_file, opts.selected_config)
    proxy  = ff.__dict__[opts.proxy] # Lookup rfe key in ff dictionary

    if opts.filter.startswith("all"):
        if opts.override.startswith("1"):
            #Set strategy on all sensors
            proxy.set_sensor_strategies("","period",str(opts.period).split(".")[0], override=True)
        else:
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
        page = 0
        perpage = 40

        while True:
            clrscr()
            gotoxy(1,1)
            if opts.filter.startswith("all"):
                sens = proxy.list_sensors(tuple=True, strategy=True)
            else:
                sens = proxy.list_sensors(opts.filter,tuple=True)
            numpages = len(sens)/perpage
            print "Print filtered sensors from %s: %s %s %s   Page %d of %d (%d)" % \
                (opts.proxy, opts.filter, state[period_count % 4], col("red")+time.ctime().replace("  "," ").split(" ")[3]+col("normal"), page+1, numpages, perpage)
            print "%s %s %s %s %s" % ("Name".ljust(45), "Value".ljust(25), "Unit".ljust(15), "Value time".ljust(25), "Update time".ljust(25))

            for s in sens[page*perpage:page*perpage+perpage]:
                name = s[0]
                val = s[1]
                valTime = s[2]
                type = s[3]
                units = s[4]
                updateTime = s[5]
                print "%s %s %s %s %s" % (name.ljust(45), str(val).ljust(25), str(units).ljust(15), get_time_str(valTime).ljust(25), get_time_str(updateTime).ljust(25) )
                sys.stdout.flush()

            #Wait, then do it all again
            time.sleep(opts.period/1000.0)
            period_count += 1

            #Get user input for display control
            c = getKeyIf()
            if c == '<':
                page = (page - 1) % numpages
            elif c == '>':
                page = (page + 1) % numpages
            elif c == '-':
                perpage = max(perpage - 1, 5)
            elif c == '+':
                perpage = min(perpage + 1, 60)
            elif c == 0 or c == '':
                pass


    except Exception,err:
        stdout_restore()
        print "\nError: Disconnecting... (",err,")"
        ff.disconnect()
    except KeyboardInterrupt:
        stdout_restore()
        print "\nDisconnecting..."
        ff.disconnect()
    print "Done."
