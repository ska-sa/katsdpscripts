#!/usr/bin/python
# Check the system against the expected default values and optionally reset to these defaults.

from optparse import OptionParser
import time
import sys

import katcorelib
from katcorelib.observe import standard_script_options, verify_and_connect, ant_array, collect_targets, user_logger

from katmisc.utils.ansi import col

# Default settings logically grouped in lists
ant1 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant1.req.log_level('cryo')[1]", "fatal", "kat.ant1.req.log_level('cryo', 'fatal')"),
("kat.ant1.sensor.rfe3_psu_on.get_value()", 1, "kat.ant1.req.rfe3_psu_on(1)"),
("kat.ant1.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant1.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant1.sensor.rfe5_attenuator_horizontal.get_value()", 7.0, "kat.ant1.req.rfe5_attenuation('h',7.0)"),
("kat.ant1.sensor.rfe5_attenuator_vertical.get_value()", 6.5, "kat.ant1.req.rfe5_attenuation('v',6.5)"),
("kat.ant1.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant1.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant1.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant1.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ant2 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant2.req.log_level('cryo')[1]", "fatal", "kat.ant2.req.log_level('cryo', 'fatal')"),
("kat.ant2.sensor.rfe3_psu_on.get_value()", 1, "kat.ant2.req.rfe3_psu_on(1)"),
("kat.ant2.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant2.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant2.sensor.rfe5_attenuator_horizontal.get_value()", 7.0, "kat.ant2.req.rfe5_attenuation('h',7.0)"),
("kat.ant2.sensor.rfe5_attenuator_vertical.get_value()", 7.0, "kat.ant2.req.rfe5_attenuation('v',7.0)"),
("kat.ant2.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant2.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant2.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant2.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ant3 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant3.req.log_level('cryo')[1]", "fatal", "kat.ant3.req.log_level('cryo', 'fatal')"),
("kat.ant3.sensor.rfe3_psu_on.get_value()", 1, "kat.ant3.req.rfe3_psu_on(1)"),
("kat.ant3.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant3.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant3.sensor.rfe5_attenuator_horizontal.get_value()", 6.0, "kat.ant3.req.rfe5_attenuation('h',6.0)"),
("kat.ant3.sensor.rfe5_attenuator_vertical.get_value()", 5.5, "kat.ant3.req.rfe5_attenuation('v',5.5)"),
("kat.ant3.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant3.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant3.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant3.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ant4 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant4.req.log_level('cryo')[1]", "fatal", "kat.ant4.req.log_level('cryo', 'fatal')"),
("kat.ant4.sensor.rfe3_psu_on.get_value()", 1, "kat.ant4.req.rfe3_psu_on(1)"),
("kat.ant4.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant4.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant4.sensor.rfe5_attenuator_horizontal.get_value()", 4.0, "kat.ant4.req.rfe5_attenuation('h',4.0)"),
("kat.ant4.sensor.rfe5_attenuator_vertical.get_value()", 3.5, "kat.ant4.req.rfe5_attenuation('v',3.5)"),
("kat.ant4.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant4.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant4.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant4.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ant5 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant5.req.log_level('cryo')[1]", "fatal", "kat.ant5.req.log_level('cryo', 'fatal')"),
("kat.ant5.sensor.rfe3_psu_on.get_value()", 1, "kat.ant5.req.rfe3_psu_on(1)"),
("kat.ant5.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant5.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant5.sensor.rfe5_attenuator_horizontal.get_value()", 8.0, "kat.ant5.req.rfe5_attenuation('h',8.0)"),
("kat.ant5.sensor.rfe5_attenuator_vertical.get_value()", 6.0, "kat.ant5.req.rfe5_attenuation('v',6.0)"),
("kat.ant5.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant5.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant5.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant5.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ant6 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant6.req.log_level('cryo')[1]", "fatal", "kat.ant6.req.log_level('cryo', 'fatal')"),
("kat.ant6.sensor.rfe3_psu_on.get_value()", 1, "kat.ant6.req.rfe3_psu_on(1)"),
("kat.ant6.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant6.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant6.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ant6.req.rfe5_attenuation('h',5.0)"),
("kat.ant6.sensor.rfe5_attenuator_vertical.get_value()", 6.5, "kat.ant6.req.rfe5_attenuation('v',6.5)"),
("kat.ant6.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant6.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant6.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant6.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

ant7 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ant7.req.log_level('cryo')[1]", "fatal", "kat.ant7.req.log_level('cryo', 'fatal')"),
("kat.ant7.sensor.rfe3_psu_on.get_value()", 1, "kat.ant7.req.rfe3_psu_on(1)"),
("kat.ant7.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ant7.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ant7.sensor.rfe5_attenuator_horizontal.get_value()", 4.0, "kat.ant7.req.rfe5_attenuation('h',4.0)"),
("kat.ant7.sensor.rfe5_attenuator_vertical.get_value()", 3.5, "kat.ant7.req.rfe5_attenuation('v',3.5)"),
("kat.ant7.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ant7.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ant7.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,"kat.ant7.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
]

rfe7 = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 6022000000.0, "kat.rfe7.req.rfe7_lo1_frequency(6.022,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 3.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',3.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 4.0, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',4.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_attenuation.get_value()", 0.0, "kat.rfe7.req.rfe7_downconverter_attenuation('2','h',0.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_attenuation.get_value()", 1.0, "kat.rfe7.req.rfe7_downconverter_attenuation('2','v',1.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant3_h_attenuation.get_value()", 3.0, "kat.rfe7.req.rfe7_downconverter_attenuation('3','h',3.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant3_v_attenuation.get_value()", 4.0, "kat.rfe7.req.rfe7_downconverter_attenuation('3','v',4.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant4_h_attenuation.get_value()", 2.5, "kat.rfe7.req.rfe7_downconverter_attenuation('4','h',2.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant4_v_attenuation.get_value()", 2.5, "kat.rfe7.req.rfe7_downconverter_attenuation('4','v',2.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant5_h_attenuation.get_value()", 2.0, "kat.rfe7.req.rfe7_downconverter_attenuation('5','h',2.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant5_v_attenuation.get_value()", 1.5, "kat.rfe7.req.rfe7_downconverter_attenuation('5','v',1.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant6_h_attenuation.get_value()", 3.5, "kat.rfe7.req.rfe7_downconverter_attenuation('6','h',3.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant6_v_attenuation.get_value()", 1.5, "kat.rfe7.req.rfe7_downconverter_attenuation('6','v',1.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant7_h_attenuation.get_value()", 2.5, "kat.rfe7.req.rfe7_downconverter_attenuation('7','h',2.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant7_v_attenuation.get_value()", 4.5, "kat.rfe7.req.rfe7_downconverter_attenuation('7','v',4.5)" ),
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
'karoo' : ant1 + ant2 + ant3 + ant4 + ant5 + ant6 + ant7 + rfe7,
'karoo1' : ant1 + rfe7,
'karoo2' : ant2 + rfe7,
'karoo3' : ant3 + rfe7,
'karoo4' : ant4 + rfe7,
'karoo5' : ant5 + rfe7,
'karoo6' : ant6 + rfe7,
'karoo7' : ant7 + rfe7,
'karoo_rfe7' : rfe7,
'lab' : ant1 + lab_rfe7,
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

    parser = standard_script_options(usage="%prog [options]",
                          description="Check the system against the expected default values and optionally reset to these defaults.")
    parser.add_option('--defaults_set', default="karoo", metavar='DEFAULTS',
                      help='Selected defaults set to use, ' + '|'.join(defaults_set.keys()) + ' (default="%default")')
    parser.add_option('--reset', action='store_true', default=False,
                      help='Reset system to default values, if this switch is included (default="%default")')
    (opts, args) = parser.parse_args()

    try:
        defaults = defaults_set[opts.defaults_set]
    except KeyError:
        print "Unknown defaults set '%s', expected one of %s" % (opts.defaults_set, defaults_set.keys())
        sys.exit()

    if opts.reset and not opts.sb_id_code:
        raise ValueError("To reset system to defaults you need to specify the schedule block: use --sb-id-code, or run without --reset")

    # Try to build the  KAT configuration
    # This connects to all the proxies and devices and queries their commands and sensors
    try:
        kat = verify_and_connect(opts)
    except ValueError, err:
        raise ValueError("Could not build host for sb-id-code %s (%s)" % (opts.sb_id_code, err))
    print "Using KAT connection with configuration: %s" % (kat.system,)

    user_logger.info("defaults.py: start")
    smsg = "Checking current settings....."
    user_logger.info(smsg)
    print smsg

    check_sensors(kat,defaults)

    if opts.reset:
        smsg = "Resetting to default settings..."
        user_logger.info(smsg)
        print "\n"+smsg
        reset_defaults(kat, defaults)
        smsg = "Rechecking settings...."
        user_logger.info(smsg)
        print "\n"+smsg
        time.sleep(1.5) # wait a little time for sensor to update
        check_sensors(kat, defaults)

    user_logger.info("defaults.py: stop")
