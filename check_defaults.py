#!/usr/bin/python
# Check the system against the expected default values and optionally reset to these defaults.

from optparse import OptionParser
import time
import sys

import katuilib
from katuilib.ansi import col


karoo_default_set = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped1.req.log_level('cryo',tuple=True)[0][2][1]", "fatal", "kat.ped1.req.log_level('cryo', 'fatal')"),
("kat.ped2.req.log_level('cryo',tuple=True)[0][2][1]", "fatal", "kat.ped2.req.log_level('cryo', 'fatal')"),
("kat.ped1.sensor.rfe3_psu_on.get_value()", 1, "kat.ped1.req.rfe3_psu_on(1)"),
("kat.ped2.sensor.rfe3_psu_on.get_value()", 1, "kat.ped2.req.rfe3_psu_on(1)"),
("kat.ped1.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped1.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped2.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped2.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped1.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ped1.req.rfe5_attenuation('h',5.0)"),
("kat.ped1.sensor.rfe5_attenuator_vertical.get_value()", 5.0, "kat.ped1.req.rfe5_attenuation('v',5.0)"),
("kat.ped2.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ped2.req.rfe5_attenuation('h',5.0)"),
("kat.ped2.sensor.rfe5_attenuator_vertical.get_value()", 5.0, "kat.ped2.req.rfe5_attenuation('v',5.0)"),
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 5775000000.0, "kat.rfe7.req.rfe7_lo1_frequency(5.775,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 5.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',5.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 4.0, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',4.0)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_attenuation.get_value()", 6.5, "kat.rfe7.req.rfe7_downconverter_attenuation('2','h',6.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_attenuation.get_value()", 6.5, "kat.rfe7.req.rfe7_downconverter_attenuation('2','v',6.5)" ),
("kat.ped1.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped1.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
("kat.ped2.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped2.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped2.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0, "kat.ped2.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'v',1)"),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_orx_powerswitch(1,1)"),
]

karoo1_default_set = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped1.req.log_level('cryo',tuple=True)[0][2][1]", "fatal", "kat.ped1.req.log_level('cryo', 'fatal')"),
("kat.ped1.sensor.rfe3_psu_on.get_value()", 1, "kat.ped1.req.rfe3_psu_on(1)"),
("kat.ped1.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped1.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped1.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ped1.req.rfe5_attenuation('h',5.0)"),
("kat.ped1.sensor.rfe5_attenuator_vertical.get_value()", 5.0, "kat.ped1.req.rfe5_attenuation('v',5.0)"),
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 5775000000.0, "kat.rfe7.req.rfe7_lo1_frequency(5.775,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 5.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',5.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 4.0, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',4.0)" ),
("kat.ped1.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped1.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_orx_powerswitch(1,1)"),
]

karoo2_default_set = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped2.req.log_level('cryo',tuple=True)[0][2][1]", "fatal", "kat.ped2.req.log_level('cryo', 'fatal')"),
("kat.ped2.sensor.rfe3_psu_on.get_value()", 1, "kat.ped2.req.rfe3_psu_on(1)"),
("kat.ped2.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped2.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped2.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ped2.req.rfe5_attenuation('h',5.0)"),
("kat.ped2.sensor.rfe5_attenuator_vertical.get_value()", 5.0, "kat.ped2.req.rfe5_attenuation('v',5.0)"),
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 5775000000.0, "kat.rfe7.req.rfe7_lo1_frequency(5.775,'GHz')"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_attenuation.get_value()", 6.5, "kat.rfe7.req.rfe7_downconverter_attenuation('2','h',6.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_attenuation.get_value()", 6.5, "kat.rfe7.req.rfe7_downconverter_attenuation('2','v',6.5)" ),
("kat.ped2.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped2.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped2.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0, "kat.ped2.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant2_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(2,'v',1)"),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_orx_powerswitch(1,1)"),
]

lab_default_set = [ # structure is list of tuples with (command to access sensor value, default value, command to set default)
("kat.ped1.req.log_level('cryo',tuple=True)[0][2][1]", "fatal", "kat.ped1.req.log_level('cryo', 'fatal')"),
("kat.ped1.sensor.rfe3_psu_on.get_value()", 1, "kat.ped1.req.rfe3_psu_on(1)"),
("kat.ped1.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1, "kat.ped1.req.rfe3_rfe15_rfe1_lna_psu_on(1)"),
("kat.ped1.sensor.rfe5_attenuator_horizontal.get_value()", 5.0, "kat.ped1.req.rfe5_attenuation('h',5.0)"),
("kat.ped1.sensor.rfe5_attenuator_vertical.get_value()", 5.0, "kat.ped1.req.rfe5_attenuation('v',5.0)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_attenuation.get_value()", 5.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','h',5.5)" ),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_attenuation.get_value()", 5.5, "kat.rfe7.req.rfe7_downconverter_attenuation('1','v',5.5)" ),
("kat.rfe7.sensor.rfe7_lo1_frequency.get_value()", 5775000000.0, "kat.rfe7.req.rfe7_lo1_frequency(5.775,'GHz')"),
("kat.ped1.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('pin',0,'now',0)"),
("kat.ped1.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0, "kat.ped1.req.rfe3_rfe15_noise_source_on('coupler',0,'now',0)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'h',1)"),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_downconverter_powerswitch(1,'v',1)"),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1, "kat.rfe7.req.rfe7_orx_powerswitch(1,1)"),
]


def check_sensors(kat,defaults):
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

    return

def reset_defaults(kat,defaults):
    # reset system to default setting as specified by commands above
    for i in range(len(defaults)):
        eval(defaults[i][2])
    return

if __name__ == "__main__":

    usage = "usage: %prog [options]"
    description = "Check the system against the expected default values and optionally reset to these defaults."
    parser = OptionParser(usage=usage)

    parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                      help='Telescope configuration file to use in conf directory (default="%default")')
    parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                      help='Selected configuration to use (default="%default")')
    parser.add_option('-d', '--defaults_set', dest='defaults_set', type="string", default="karoo", metavar='DEFAULTS',
                      help='Selected defaults set config to use - karoo|karoo1|karoo2|lab (default="%default")')
    parser.add_option('-r', '--reset', dest='reset', action='store_true',default=False,
                      help='Reset system to default values, if include this switch (default="%default")')
    (opts, args) = parser.parse_args()

    built_kat = False

    try:
        if opts.ini_file == "" or opts.selected_config == "":
            print "Please specify ini file and selected config (-h for help)."
            sys.exit()

        if opts.defaults_set == "karoo":
            defaults = karoo_default_set
        elif opts.defaults_set == "karoo1":
            defaults = karoo1_default_set
        elif opts.defaults_set == "karoo2":
            defaults = karoo2_default_set
        elif opts.defaults_set == 'lab':
            defaults = lab_default_set
        else:
            print 'Unknown defaults set specified', opt.defaults
            sys.exit()

        kat = katuilib.tbuild(opts.ini_file, opts.selected_config)
        built_kat = True

        print "Checking current settings....."
        check_sensors(kat,defaults)

        if opts.reset:
            print "\nResetting to default settings..."
            reset_defaults(kat,defaults)
            print "\nRechecking settings...."
            time.sleep(1.5) # wait a little time for sensor to update
            check_sensors(kat,defaults)

    except Exception, e:
        print "Exception: ", e
        print 'Exception caught: attempting to exit cleanly...'
    finally:
        if built_kat: kat.disconnect()


