#!/usr/bin/python
# Silly script to check the LNAs of an antenna
#
# Initial script
# Quick check: run exercise_lna.py -o ruby --ant m062
# Note do this for each antenna separately, script is not optimised for multiple antennas


from __future__ import with_statement
import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger, start_session
from katcorelib import cambuild, katconf

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_sensors(inpt, sensor_list):
    for atr in sensor_list:
	msg = '    Value: %s is %s (%s)' % (atr,getattr(inpt, atr).get_value(),getattr(inpt, atr).get_status())
	if getattr(inpt, atr).get_status()=='nominal':
	    print bcolors.OKGREEN + msg + bcolors.ENDC
	elif getattr(inpt, atr).get_status()=='warn':
	    print bcolors.WARNING + msg + bcolors.ENDC
	else:
	    print bcolors.FAIL + msg + bcolors.ENDC

def lna_sensors(ant):
    print "Current status of LNA" 
    if ant.sensor.rsc_rxl_lna_h_power_enabled.get_value():
	print '  L-band LNA H is ON'
    else:
	print '  L-band LNA H is OFF'
    hpol=[
	 'dig_l_band_rfcu_hpol_rf_power_in',
	 'dig_l_band_adc_hpol_rf_power_in']
    check_sensors(ant.sensor, hpol)
    if ant.sensor.rsc_rxl_lna_v_power_enabled.get_value():
	print '  L-band LNA V is ON'
    else:
	print '  L-band LNA V is OFF'
    vpol=[
	 'dig_l_band_rfcu_vpol_rf_power_in',
	 'dig_l_band_adc_vpol_rf_power_in']
    check_sensors(ant.sensor, vpol)


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="AR1 antenna quick check")
parser.add_option('--ant', type=str, default=None,
                  help='Antennas to check in the format m0xx. (default="%default")')
parser.add_option('--timeout', type=int, default=60,
                  help='Number of seconds to sleep between LNA off and on cycle. (default="%default")')
parser.add_option('--repeat', type=int, default=1,
                  help='Number of seconds to repeat LNA off and on cycle. (default="%default")')
# assume basic options passed from instruction_set
parser.set_defaults(description = 'AR1 AP Check LNAs')
(opts, args) = parser.parse_args()

with verify_and_connect(opts) as kat:
    try:
        cam = None
        done = False
        count = 1

	print('Building CAM object')
        cam = cambuild(password="camcam", full_control="all")
	time.sleep(5)

        ant_list = opts.ant.strip().split(',')
        for ant in cam.ants:
            if ant.name in ant_list:
		print '\nAntenna %s' % ant.name
		lna_sensors(ant)
		for nr in range(opts.repeat):
		    print 'Cycle %d of %d' % (nr+1, opts.repeat)
		    print 'Switching L-band LNAs off'
 		    ant.req.rsc_rxl_lna_h_power('disable')
 		    ant.req.rsc_rxl_lna_v_power('disable')
		    time.sleep(5)
		    print 'Sleeping for %d seconds' % opts.timeout
		    time.sleep(opts.timeout)
		    lna_sensors(ant)
		    print 'Switching L-band LNAs on'
 		    ant.req.rsc_rxl_lna_h_power('enable')
 		    ant.req.rsc_rxl_lna_v_power('enable')
		    time.sleep(5)
		    print 'Sleeping for %d seconds' % opts.timeout
		    time.sleep(opts.timeout)
		    lna_sensors(ant)

    finally:
        if cam:
	    print("Cleaning up cam object")
            cam.disconnect()

# -fin-

