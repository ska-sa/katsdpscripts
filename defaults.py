#!/usr/bin/python
# Check the system against the expected default values and optionally reset to these defaults.

from optparse import OptionParser
import time
import sys

import katuilib
from katuilib.ansi import col

# Default settings logically grouped in lists

ped1 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped1.req.log_level('cryo')[1]", "fatal", "kat.ped1.req.log_level('cryo', 'fatal')"),
("kat.ped1.sensor.rfe3_psu_on.get_value()", 1, "kat.ped1.req.rfe3_psu_on(1)"),
("kat.ped1.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped1.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped1.sensor.rfe5_attenuator_horizontal.get_value()", 7.0, "kat.ped1.req.rfe5_attenuation('h',7.0)"),
("kat.ped1.sensor.rfe5_attenuator_vertical.get_value()", 6.5, "kat.ped1.req.rfe5_attenuation('v',6.5)"),
("kat.ped1.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped1.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped1.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ped2 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped2.req.log_level('cryo')[1]", "fatal", "kat.ped2.req.log_level('cryo', 'fatal')"),
("kat.ped2.sensor.rfe3_psu_on.get_value()", 1, "kat.ped2.req.rfe3_psu_on(1)"),
("kat.ped2.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped2.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped2.sensor.rfe5_attenuator_horizontal.get_value()", 7.0, "kat.ped2.req.rfe5_attenuation('h',7.0)"),
("kat.ped2.sensor.rfe5_attenuator_vertical.get_value()", 7.0, "kat.ped2.req.rfe5_attenuation('v',7.0)"),
("kat.ped2.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped2.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped2.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped2.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ped3 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped3.req.log_level('cryo')[1]", "fatal", "kat.ped3.req.log_level('cryo', 'fatal')"),
("kat.ped3.sensor.rfe3_psu_on.get_value()", 1, "kat.ped3.req.rfe3_psu_on(1)"),
("kat.ped3.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped3.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped3.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ped3.req.rfe5_attenuation('h',5.0)"),
("kat.ped3.sensor.rfe5_attenuator_vertical.get_value()", 6.0, "kat.ped3.req.rfe5_attenuation('v',6.0)"),
("kat.ped3.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped3.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped3.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped3.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ped4 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped4.req.log_level('cryo')[1]", "fatal", "kat.ped4.req.log_level('cryo', 'fatal')"),
("kat.ped4.sensor.rfe3_psu_on.get_value()", 1, "kat.ped4.req.rfe3_psu_on(1)"),
("kat.ped4.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped4.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped4.sensor.rfe5_attenuator_horizontal.get_value()", 4.0, "kat.ped4.req.rfe5_attenuation('h',4.0)"),
("kat.ped4.sensor.rfe5_attenuator_vertical.get_value()", 3.5, "kat.ped4.req.rfe5_attenuation('v',3.5)"),
("kat.ped4.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped4.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped4.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped4.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

rfe7 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 6022000000.0, "kat.rfe7.req.rfe7_lo1_frequency(6.022,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 6.0, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',6.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 6.0, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',6.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_attenuation.get_value()", 6.0, "kat.rfe7.req.rfe7_downconverter_attenuation('2','h',6.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_attenuation.get_value()", 6.0, "kat.rfe7.req.rfe7_downconverter_attenuation('2','v',6.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'v',1)"),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_orx_powerswitch(1,1)"),
]

lab_rfe7 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 6022000000.0, "kat.rfe7.req.rfe7_lo1_frequency(6.022,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 5.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',5.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 5.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',5.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_orx_powerswitch(1,1)"),
]

# Dictionary containing multiple sets of default settings, identified by name (user selects these by name at runtime)
defaults_set = {
'karoo' : ped1 + ped2 + ped3 + ped4 + rfe7,
'karoo1' : ped1 + rfe7,
'karoo2' : ped2 + rfe7,
'karoo3' : ped3 + rfe7,
'karoo4' : ped4 + rfe7,
'karoo_rfe7' : rfe7,
'lab' : ped1 + lab_rfe7,
}

def check_sensors(kat, defaults):
    # check current system setting and compare with defaults as specified above
    print "%s %s %s" % ("Sensor".ljust(65), "Current Value".ljust(25),"Default Value".ljust(25))
    current_vals = []
    for i in range(len(defaults)):
        try:
            current_vals.append(str(eval(defaults[i][0])))
            if current_vals[i] <> str(defaults[i][1]):
                print "%s %s %s" % (col("red")+str(defaults[i][0]).ljust(65),str(current_vals[i]).ljust(25),str(defaults[i][1]).ljust(25)+col("normal"))
            else:
                print "%s %s %s" % (str(defaults[i][0]).ljust(65),str(current_vals[i]).ljust(25),str(defaults[i][1]).ljust(25))
        except:
            print "Could not check", str(defaults[i])

def reset_defaults(kat, defaults):
    # reset system to default setting as specified by commands above
    for i in range(len(defaults)):
        try:
            eval(defaults[i][2])
        except Exception, err:
            print "Cannot set - ",str(defaults[1][2]),"ERROR :",str(err)

if __name__ == "__main__":

    parser = OptionParser(usage="%prog [options]",
                          description="Check the system against the expected default values and optionally reset to these defaults.")
    parser.add_option('-s', '--system', help='System configuration file to use, relative to conf directory ' +
                      '(default reuses existing connection, or falls back to systems/local.conf)')
    parser.add_option('-d', '--defaults_set', default="karoo", metavar='DEFAULTS',
                      help='Selected defaults set to use, ' + '|'.join(defaults_set.keys()) + ' (default="%default")')
    parser.add_option('-r', '--reset', action='store_true', default=False,
                      help='Reset system to default values, if this switch is included (default="%default")')
    (opts, args) = parser.parse_args()

    try:
        defaults = defaults_set[opts.defaults_set]
    except KeyError:
        print "Unknown defaults set '%s', expected one of %s" % (opts.defaults_set, defaults_set.keys())
        sys.exit()

    # Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
    # This connects to all the proxies and devices and queries their commands and sensors
    try:
        kat = katuilib.tbuild(opts.system)
    # Fall back to *local* configuration to prevent inadvertent use of the real hardware
    except ValueError:
        kat = katuilib.tbuild('systems/local.conf')
    print "Using KAT connection with configuration: %s" % (kat.config_file,)

    print "Checking current settings....."
    check_sensors(kat,defaults)

    if opts.reset:
        print "\nResetting to default settings..."
        reset_defaults(kat, defaults)
        print "\nRechecking settings...."
        time.sleep(1.5) # wait a little time for sensor to update
        check_sensors(kat, defaults)
