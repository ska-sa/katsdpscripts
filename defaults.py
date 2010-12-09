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

ped5 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped5.req.log_level('cryo')[1]", "fatal", "kat.ped5.req.log_level('cryo', 'fatal')"),
("kat.ped5.sensor.rfe3_psu_on.get_value()", 1, "kat.ped5.req.rfe3_psu_on(1)"),
("kat.ped5.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped5.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped5.sensor.rfe5_attenuator_horizontal.get_value()", 4.0, "kat.ped5.req.rfe5_attenuation('h',4.0)"),
("kat.ped5.sensor.rfe5_attenuator_vertical.get_value()", 3.5, "kat.ped5.req.rfe5_attenuation('v',3.5)"),
("kat.ped5.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped5.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped5.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped5.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ped6 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped6.req.log_level('cryo')[1]", "fatal", "kat.ped6.req.log_level('cryo', 'fatal')"),
("kat.ped6.sensor.rfe3_psu_on.get_value()", 1, "kat.ped6.req.rfe3_psu_on(1)"),
("kat.ped6.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped6.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped6.sensor.rfe5_attenuator_horizontal.get_value()", 4.0, "kat.ped6.req.rfe5_attenuation('h',4.0)"),
("kat.ped6.sensor.rfe5_attenuator_vertical.get_value()", 3.5, "kat.ped6.req.rfe5_attenuation('v',3.5)"),
("kat.ped6.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped6.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped6.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped6.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ped7 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped7.req.log_level('cryo')[1]", "fatal", "kat.ped7.req.log_level('cryo', 'fatal')"),
("kat.ped7.sensor.rfe3_psu_on.get_value()", 1, "kat.ped7.req.rfe3_psu_on(1)"),
("kat.ped7.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped7.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped7.sensor.rfe5_attenuator_horizontal.get_value()", 4.0, "kat.ped7.req.rfe5_attenuation('h',4.0)"),
("kat.ped7.sensor.rfe5_attenuator_vertical.get_value()", 3.5, "kat.ped7.req.rfe5_attenuation('v',3.5)"),
("kat.ped7.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped7.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped7.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ped7.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

rfe7 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 6022000000.0, "kat.rfe7.req.rfe7_lo1_frequency(6.022,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 3.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',3.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 4.0, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',4.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_attenuation.get_value()", 0.5, "kat.rfe7.req.rfe7_downconverter_attenuation('2','h',0.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_attenuation.get_value()", 1.5, "kat.rfe7.req.rfe7_downconverter_attenuation('2','v',1.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant3_h_attenuation.get_value()", 3.0, "kat.rfe7.req.rfe7_downconverter_attenuation('3','h',3.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant3_v_attenuation.get_value()", 4.5, "kat.rfe7.req.rfe7_downconverter_attenuation('3','v',4.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant4_h_attenuation.get_value()", 2.5, "kat.rfe7.req.rfe7_downconverter_attenuation('4','h',2.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant4_v_attenuation.get_value()", 3.5, "kat.rfe7.req.rfe7_downconverter_attenuation('4','v',3.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant5_h_attenuation.get_value()", 0.0, "kat.rfe7.req.rfe7_downconverter_attenuation('5','h',0.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant5_v_attenuation.get_value()", 1.5, "kat.rfe7.req.rfe7_downconverter_attenuation('5','v',1.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant6_h_attenuation.get_value()", 2.0, "kat.rfe7.req.rfe7_downconverter_attenuation('6','h',2.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant6_v_attenuation.get_value()", 1.5, "kat.rfe7.req.rfe7_downconverter_attenuation('6','v',1.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant7_h_attenuation.get_value()", 3.5, "kat.rfe7.req.rfe7_downconverter_attenuation('7','h',3.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant7_v_attenuation.get_value()", 4.0, "kat.rfe7.req.rfe7_downconverter_attenuation('7','v',4.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant3_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(3,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant3_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(3,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant4_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(4,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant4_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(4,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant5_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(5,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant5_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(5,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant6_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(6,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant6_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(6,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant7_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(7,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant7_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(7,'v',1)"),
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
'karoo' : ped1 + ped2 + ped3 + ped4 + ped5 + ped6 + ped7 + rfe7,
'karoo1' : ped1 + rfe7,
'karoo2' : ped2 + rfe7,
'karoo3' : ped3 + rfe7,
'karoo4' : ped4 + rfe7,
'karoo5' : ped5 + rfe7,
'karoo6' : ped6 + rfe7,
'karoo7' : ped7 + rfe7,
'karoo_rfe7' : rfe7,
'lab' : ped1 + lab_rfe7,
}

def check_sensors(kat, defaults):
    # check current system setting and compare with defaults as specified above
    print "%s %s %s" % ("Sensor".ljust(65), "Current Value".ljust(25),"Default Value".ljust(25))
    for checker, default, _setter in defaults:
        try:
            default = str(default)
            current_val = str(eval(checker))
            if current_val != default:
                print "%s %s %s" % (col("red") + checker.ljust(65), current_val.ljust(25), default.ljust(25) + col("normal"))
            else:
                print "%s %s %s" % (checker.ljust(65), current_val.ljust(25), default.ljust(25))
        except:
            print "Could not check", checker, "[expected value: %r]" % (default,)

def reset_defaults(kat, defaults):
    # reset system to default setting as specified by commands above
    for _checker, _default, setter in defaults:
        try:
            eval(setter)
        except Exception, err:
            print "Cannot set - ", setter, "ERROR :", err

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
