#!/usr/bin/python
# check the system against the expecte default values and optionally reset to these defaults.

import ffuilib as ffui
from optparse import OptionParser
from ansi import col
import time

defaults = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("ff.rfe.sensor.rfe31_psu_on.get_value()", 1, "ff.rfe.req.rfe3_psu_on('rfe31',1)"),
("ff.rfe.sensor.rfe31_rfe15_rfe1_lna_psu_on.get_value()", 1, "ff.rfe.req.rfe3_rfe15_rfe1_lna_psu_on('rfe31',1)"),
("ff.rfe.sensor.rfe51_attenuator_horizontal.get_value()", 5.0, "ff.rfe.req.rfe5_attenuation('rfe51','h',5.0)"),
("ff.rfe.sensor.rfe51_attenuator_vertical.get_value()", 5.0, "ff.rfe.req.rfe5_attenuation('rfe51','v',5.0)"),
("ff.rfe.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 5.5, "ff.rfe.req.rfe7_downconverter_attenuation('1','h',5.5)" ),
("ff.rfe.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 5.5, "ff.rfe.req.rfe7_downconverter_attenuation('1','v',5.5)" ),
("ff.rfe.req.log_level('cryo1',tuple=True)[0][2][1]", "fatal", "ff.rfe.req.log_level('cryo1', 'fatal')"),
("ff.rfe.sensor.rfe7_lo1_frequency.get_value()", 5700000000.0, "ff.rfe.req.rfe7_lo1_frequency(5.7,'GHz')"),
("ff.rfe.sensor.rfe31_rfe15_noise_pin_on.get_value()", 0, "ff.rfe.req.rfe3_rfe15_noise_source_on('rfe31','pin',0,'now',0)"),
("ff.rfe.sensor.rfe31_rfe15_noise_coupler_on.get_value()", 0, "ff.rfe.req.rfe3_rfe15_noise_source_on('rfe31','coupler',0,'now',0)"),
("ff.rfe.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "ff.rfe.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("ff.rfe.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "ff.rfe.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("ff.rfe.sensor.rfe7_orx1_powerswitch.get_value()", 1, "ff.rfe.req.rfe7_orx_powerswitch(1,1)"),
]

def check_sensors(ff):
    # check current system setting and compare with defaults as specified above
    print "%s %s %s" % ("Sensor".ljust(65), "Current Value".ljust(25),"Default Value".ljust(25))
    current_vals = []
    for i in range(len(defaults)):
        current_vals.append(str(eval(defaults[i][0])))
        if current_vals[i] <> str(defaults[i][1]):
            print "%s %s %s" % (col("red")+str(defaults[i][0]).ljust(65),str(current_vals[i]).ljust(25),str(defaults[i][1]).ljust(25)+col("normal"))
        else:
            print "%s %s %s" % (str(defaults[i][0]).ljust(65),str(current_vals[i]).ljust(25),str(defaults[i][1]).ljust(25))

    return

def reset_defaults(ff):
    # reset system to default setting as specified by commands above
    for i in range(len(defaults)):
        eval(defaults[i][2])
    return

if __name__ == "__main__":

    usage = "usage: %prog [options]"
    description = "check the system against the expecte default values and optionally reset to these defaults."
    parser = OptionParser(usage=usage)

    parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-lab.ini", metavar='INI',
                      help='Telescope configuration file to use in conf directory (default="%default")')
    parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="lab_ff", metavar='SELECTED',
                      help='Selected configuration to use (default="%default")')
    parser.add_option('-r', '--reset', dest='reset', action='store_true',default=False,
                      help='Reset system to default values, if include this switch (default="%default")')
    (opts, args) = parser.parse_args()

try:
    ff = ffui.tbuild(opts.ini_file, opts.selected_config)

    print "Checking current settings....."
    check_sensors(ff)

    if opts.reset:
        print "\nResetting to default settings..."
        reset_defaults(ff)
        print "\nRechecking settings...."
        time.sleep(1.5) # wait a little time for sensor to update
        check_sensors(ff)

except Exception, e:
    print "Exception: ", e
    print 'Exception caught: attempting to exit cleanly...'
finally:
    ff.disconnect()


