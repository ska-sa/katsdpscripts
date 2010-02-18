#!/usr/bin/python
# Print out the status (mode, lock, pointing etc...) for the specified antenna

import katuilib as katui
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
    #fstdout = open('test_out.log', 'w')
    sys.stdout = StringIO.StringIO()
    redirected = True
    return savestdout

def stdout_restore():
    global savestdout
    global fstdout
    global redirected
    #fstdout.close()
    #fstdout = open('test_out.log', 'r')
    #Out = fstdout.read()
    #fstdout.close()
    sOut = ""
    if redirected == True:
        sOut = sys.stdout.getvalue()
        sys.stdout = savestdout
        redirected = False
    return sOut




if __name__ == "__main__":

#    #savestdout = sys.stdout
#    #sys.stdout = StringIO.StringIO()
#    stdout_redirect()
#    print "foo", "bar", "baz"
#    s = stdout_restore()
#    #sys.stdout = savestdout
#
#    print "Printing" + s

    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)

    parser.add_option('-r', '--rfe7', dest='rfe7', type="string", default="rfe7", metavar='RFE7',
                      help='Name of RFE proxy to attach to (default="%default") as per the configuration file')
    parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                      help='Telescope configuration file to use in conf directory (default="%default")')
    parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                      help='Selected configuration to use (default="%default")')
    parser.add_option('-f', '--filter', dest='filter', type='string', default='rfe7', metavar='FILTER',
                      help='Filter on sensors to print (default="%default")')
    parser.add_option('-p', '--period', dest='period', type='float', default='500', metavar='PERIOD',
                      help='Refresh period in milliseconds (default="%default")')
    (opts, args) = parser.parse_args()


    kat = katui.tbuild(opts.ini_file, opts.selected_config)
    rfe7  = kat.__dict__[opts.rfe7] # Lookup rfe key in kat dictionary

    state = ["|","/","-","\\"]
    period_count = 0
    print "\n"
    try:
        clrscr()
        stdout_redirect()
        try:
            rfe7.req.device_list()
        finally:
            s = stdout_restore()
            sys.stdout.write("Number of devices "+s)

        while True:
            gotoxy(5,1)
            s = "Print filtered sensors: %s %s %s" % (opts.filter, state[period_count % 4], col("red")+time.ctime().replace("  "," ").split(" ")[3])+col("normal")
            print s
            print "Name".ljust(45),"Value".ljust(15)
            if opts.filter.startswith("all"):
                sens = rfe7.list_sensors(tuple=True)
            else:
                sens = rfe7.list_sensors(opts.filter,tuple=True)
            #Set strategies for these or do a get_value ?
            for s in sens:
                name = s[0]
                val = s[1]
                print "%s %s " % (name.ljust(45), str(val).ljust(15))
                sys.stdout.flush()

            #Wait, then do it all again
            time.sleep(opts.period/1000.0)
            period_count += 1

    except Exception,err:
        stdout_restore()
        print "\nError: Disconnecting... (",err,")"
        kat.disconnect()
    except KeyboardInterrupt:
        stdout_restore()
        print "\nDisconnecting..."
        kat.disconnect()
    print "Done."
