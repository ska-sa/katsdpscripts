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
import tty

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


class ConsoleNB(object):
    """Class to manage non blocking reads from console.

    Opens non blocking read file descriptor on console
    Use instance method close to close file descriptor
    Use instance methods getline & put to read & write to console
    Needs os module
    """

    def __init__(self, canonical = True):
        """Initialization method for instance.

        opens fd on terminal console in non blocking mode
        os.ctermid() returns path name of console usually '/dev/tty'
        os.O_NONBLOCK makes non blocking io
        os.O_RDWR allows both read and write.
        Don't use print as same time since it could mess up non blocking reads.
        Default is canonical mode so no characters available until newline
        """
        #need to add code to enable  non canonical mode

        self.fd = os.open(os.ctermid(),os.O_NONBLOCK | os.O_RDWR)

    def close(self):
        """Closes fd. Should use  in try finally block.

        """
        os.close(self.fd)

    def getline(self,bs = 80):
        """Gets nonblocking line from console up to bs characters including newline.

          Returns None if no characters available else returns line.
          In canonical mode no chars available until newline is entered.
        """
        line = None
        try:
            line = os.read(self.fd, bs)
        except OSError, ex1:  #if no chars available generates exception
            try: #need to catch correct exception
                errno = ex1.args[0] #if args not sequence get TypeError
                if errno == 35:
                    pass #No characters available
                else:
                    raise #re raise exception ex1
            except TypeError, ex2:  #catch args[0] mismatch above
                raise ex1 #ignore TypeError, re-raise exception ex1

        return line

    def put(self, data = '\n'):
        """Writes data string to console.

        """
        os.write(self.fd, data)





######## Methods #########

def getKeyIf(which):
    if which == 1:
        # On MAC this requires a key to continue
        x = os.read(0,1)

        if len(x):
            # ok, some key got pressed
            return x[:1]
        else:
            return 0

    elif which == 2:
        #MAC - is blocking, but reads key without enter
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

    elif which == 3:
        #This does not block on MAC but keypress never registers
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if isData():
                c = sys.stdin.read(1)
                print "....",c
            else:
                c = 0
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return c

    elif which == 4:
        #This works on MAC - updates without keypress - but lenny needs keypress
        #Non-blocking key - returns 0 if no key available
        c = 0
        oldattr = termios.tcgetattr(0)
        try:
            attr = termios.tcgetattr(0)
            attr[2] = (attr[2] & ~termios.NLDLY) | termios.NL0
            attr[3] = attr[3] & ~(termios.ICANON|termios.ECHO)
            termios.tcsetattr(0,termios.TCSANOW,attr)
            c = os.read(0,1)
        finally:
            termios.tcsetattr(0,termios.TCSANOW,oldattr)
        return c

    elif which == 5:
        fd = os.open(os.ctermid(),os.O_NONBLOCK | os.O_RDWR)
        c = 0
        try:
            c = os.read(fd, 1)
        except OSError, ex1:  #if no chars available generates exception
            try: #need to catch correct exception
                errno = ex1.args[0] #if args not sequence get TypeError
                if errno == 35 or errno == 11:  # Make provision for MAC and UNIX
                    pass #No characters available
                else:
                    raise #re raise exception ex1
            except TypeError, ex2:  #catch args[0] mismatch above
                raise ex1 #ignore TypeError, re-raise exception ex1
        os.close(fd)
        return c

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
    parser.add_option('-p', '--period', dest='period', type='float', default='500', metavar='PERIOD',
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
            numpages,rest = divmod(len(sens),perpage)
            numpages +=1
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
            c = getKeyIf(5)
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
            elif c == 'q' or c == 'Q':
                stdout_restore()
                print "\nDisconnecting..."
                ff.disconnect()
                exit()


    except Exception,err:
        stdout_restore()
        print "\nError: Disconnecting... (",err,")"
        ff.disconnect()
    except KeyboardInterrupt:
        stdout_restore()
        print "\nDisconnecting..."
        ff.disconnect()
    print "Done."
