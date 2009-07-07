#!/usr/bin/python
# Print out the status (mode, lock, pointing etc...) for the specified antenna 

import ffuilib as ffui
import time
import sys
from optparse import OptionParser
import StringIO

savestdout = sys.stdout 
fstdout = None
x = StringIO.StringIO()


######## Methods #########

def stdout_redirect():
    global savestdout
    global fstdout
    savestdout = sys.stdout                                     
    #fstdout = open('test_out.log', 'w')                             
    sys.stdout = StringIO.StringIO()                                     
    return savestdout

def stdout_restore():
    global savestdout
    global fstdout                                
    #fstdout.close()
    #fstdout = open('test_out.log', 'r')     
    #Out = fstdout.read()           
    #fstdout.close() 
    sOut = sys.stdout.getvalue()
    sys.stdout = savestdout  
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

    parser.add_option('-r', '--rfe', dest='rfe', type="string", default="rfe", metavar='RFE',
                      help='RFE proxy to attach to (default="%default") as per the configuration file')
    parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-telescope.ini", metavar='INI',
                      help='Telescope configuration file to use in /var/kat/conf (default="%default")')
    parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local-rf-only", metavar='SELECTED',
                      help='Selected configuration to use (default="%default")')
    (opts, args) = parser.parse_args()


    ff = ffui.tbuild(opts.ini_file, opts.selected_config)
    rfe  = ff.__dict__[opts.rfe] # Lookup rfe key in ff dictionary
    
    state = ["|","/","-","\\"]
    period_count = 0
    print "\n"
    try:
        while True:
            sys.stdout.write("\rPrint all modes sensors")
            sys.stdout.flush()
            modes = rfe.list_sensors("mode",tuple=True)
            for m in modes:
               name = m[0]
               val = m[1]
               out = "\r%s: %s Name:%s Val:%s" % (opts.rfe, time.ctime().split(" ")[3], name, val)
               sys.stdout.write(out)
               sys.stdout.flush()
            
            
            sys.stdout.write("\rTest grouped commands\r")
            sys.stdout.flush()
            
            stdout_redirect()
            rfe.req_device_list()
            s = stdout_restore()
            sys.stdout.write("\r Number of devices "+s)
#
#In [120]: ff.rfe.req_device_list()
#!device-list ok 11
##device-list rfe72
##device-list rfe71
##device-list rfe71.rfe52
##device-list rfe72.rfe31
##device-list rfe72.cryo1
##device-list rfe71.cryo1
##device-list rfe71.cryo2
##device-list rfe72.rfe51
##device-list rfe71.rfe31
##device-list rfe71.rfe32
##device-list rfe71.rfe51         
            
            
            stdout_redirect()
            rfe.req_rfe3_set_lna_psu("all","on")
            s = stdout_restore()
            sys.stdout.write(s)
            sys.stdout.flush()
            
            #Wait for 2 seconds, then do it all again
            time.sleep(2.0)
            
            #For now
            
    except Exception,err:
        stdout_restore()
        print "\nError: Disconnecting... (",err,")"
        ff.disconnect()
    except KeyboardInterrupt:
        stdout_restore()
        print "\nDisconnecting..."
        ff.disconnect()
    print "Done."
