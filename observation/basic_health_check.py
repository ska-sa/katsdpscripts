#!/usr/bin/python
# Check for basic system health (modified from defaults.py script).
# Intended for the use of an observer.
# Code: Jasper Horrell (jasper@ska.ac.za)

## TODO:
# 1) Might be better to pull max and min values from the system config in some
# way rather than specify them again separately here. The ones used here are aimed
# at what an observer should look out for (rather than perhaps an engineer) and so
# necessarily may not be in sync with the value used for alarms etc. although one
# would hope that there would not be major differences. The script does currently
# display the sensor status along with the current values which, over time, will help
# to iron out any discrepancies between the two sources of expected sensor ranges.

from optparse import OptionParser
import time
import sys

import katuilib
from katuilib.ansi import col

# Sensor groups

# programmatically generate various sensor groups e.g. ant1, ant2, ... ant7
ant_template = [ # structure is list of tuples with (command to access sensor value, min value, max value)
("kat.ped#.sensor.cryo_lna_temperature.get_value()", 60.0,80.0),
("kat.ped#.sensor.bms_chiller_flow_present.get_value()", 1,1),
("kat.ped#.sensor.rfe3_psu_on.get_value()", 1,1),
("kat.ped#.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1,1),
("kat.ped#.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0,0),
("kat.ped#.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,0),
("kat.ant#.sensor.mode.get_value()",["POINT","STOP","STOW","SCAN"],''), # command, list of string options, blank string
("kat.ant#.sensor.windstow_active.get_value()",0,0),
("kat.ant#.sensor.pos_actual_scan_azim.get_value()",-185.0,275.0),
("kat.ant#.sensor.pos_actual_scan_elev.get_value()",2.0,95.0),
]

rfe7_downconverter_template = [
("kat.rfe7.sensor.rfe7_downconverter_ant#_h_powerswitch.get_value()", 1,1), # do we actually need these to be checked?
("kat.rfe7.sensor.rfe7_downconverter_ant#_v_powerswitch.get_value()", 1,1),
]

dbe7_adc_power_template = [
("kat.dbe7.sensor.dbe_ant#h_adc_power.get_value()",-27.0,-25.0),
("kat.dbe7.sensor.dbe_ant#v_adc_power.get_value()",-27.0,-25.0),
]

for i in range(1,8):
    antvar = 'ant' + str(i)
    vars()[antvar] = []
    for sensor in ant_template:
        vars()[antvar].append((sensor[0].replace("#",str(i)),sensor[1],sensor[2]))
    for sensor in rfe7_downconverter_template:
        vars()[antvar].append((sensor[0].replace("#",str(i)),sensor[1],sensor[2]))
    for sensor in dbe7_adc_power_template:
        vars()[antvar].append((sensor[0].replace("#",str(i)),sensor[1],sensor[2]))
    vars()[antvar].append(("","",""))

rfe7 = [ # structure is list of tuples with (command to access sensor value, min value, max value)
("kat.mon_kat_proxy.sensor.agg_rfe7_psu_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_orx1_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_orx2_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_orx3_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_osc_states_ok.get_value()", 1,1),
("","",""), # creates a blank line
]

dc = [# structure is list of tuples with (command to access sensor value, min value, max value)
("kat.dbe7.sensor.dbe_mode.get_value()",['wbc','wbc8k'],''), # command, list of string options, blank string
("kat.dbe7.sensor.k7w_status.get_value()",['init','idle','capturing','complete'],''),
("kat.nm_kat_dc1.sensor.k7capture_running.get_value()",1,1),
("kat.nm_kat_dc1.sensor.k7aug_running.get_value()",1,1),
("kat.nm_kat_dc1.sensor.k7arch_running.get_value()",1,1),
("","",""), # creates a blank line
]

tfr = [# structure is list of tuples with (command to access sensor value, min value, max value)
("kat.mon_kat_proxy.sensor.agg_anc_tfr_time_synced.get_value()",1,1),
("kat.mon_kat_proxy.sensor.agg_anc_css_ntp_synch.get_value()",1,1), # does this include kat-dc1?
("kat.mon_kat_proxy.sensor.agg_anc_css_ut1_current.get_value()",1,1),
("kat.mon_kat_proxy.sensor.agg_anc_css_tle_current.get_value()",1,1),
("kat.ant1.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.ant2.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.ant3.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.ant4.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.ant5.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.ant6.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.ant7.sensor.antenna_acu_ntp_time.get_value()",1,1),
("kat.dbe7.sensor.dbe_ntp_synchronised.get_value()",1,1),
("","",""), # creates a blank line
]

anc = [# structure is list of tuples with (command to access sensor value, min value, max value)
("kat.anc.sensor.asc_asc_air_temperature.get_value()", 0.0,32.0),
("kat.anc.sensor.asc_chiller_water_temperature.get_value()", 6.0,22.0),
("kat.anc.sensor.cc_cc_air_temperature.get_value()", 0.0,30.0),
("kat.anc.sensor.cc_chiller_water_temperature.get_value()", 6.0,18.0),
("kat.anc.sensor.asc_wind_speed.get_value()", 0.0,15.2),
("kat.anc.sensor.asc_fire_ok.get_value()", 1,1), # these sensors really should be something like "(not) on fire"
("kat.anc.sensor.cc_fire_ok.get_value()", 1,1),
("kat.anc.sensor.cmc_fire_ok.get_value()", 1,1),
("kat.anc.sensor.asc_ups_battery_not_discharging.get_value()", 1,1),
("kat.anc.sensor.asc_ups_ok.get_value()", 1,1),
("kat.anc.sensor.cc_ups_battery_not_discharging.get_value()", 1,1),
("kat.anc.sensor.cc_ups_fault.get_value()", 0,0),
("","",""), # creates a blank line
]

lab_rfe7 = [ # structure is list of tuples with (command to access sensor value, default value, tolerance)
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1,1),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1,1),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1,1),
("","",""), # creates a blank line
]

# Dictionary containing multiple sensor groups, identified by name (user selects one of these by name at runtime)
sensor_group = {
'karoo' : ant1 + ant2 + ant3 + ant4 + ant5 + ant6 + ant7 + rfe7 + dc + tfr + anc,
'ant1' : ant1,
'ant2' : ant2,
'ant3' : ant3,
'ant4' : ant4,
'ant5' : ant5,
'ant6' : ant6,
'ant7' : ant7,
'ants' : ant1 + ant2 + ant3 + ant4 + ant5 + ant6 + ant7,
'rfe7' : rfe7,
'dc' : dc,
'tfr' : tfr,
'anc' : anc,
'lab_rfe7' : lab_rfe7,
'lab' : ant1 + lab_rfe7,
}

sensor_status_errors = ['failure','error','unknown'] # other possibilities are warn, nominal

def check_sensors(kat, selected_sensors, show_only_errors):
    # Check current system setting and compare with expected range as specified above.
    # Things appear red if out of expected range or if sensor status is one of the error states as defined above.
    print "%s %s %s %s" % ("Sensor".ljust(65), "Value (sensor status)".ljust(25),"Min Expected".ljust(25), "Max Expected".ljust(25))
    for checker, min_val, max_val in selected_sensors:
        if checker.strip() == '':
            if not show_only_errors: print "" # print a blank line, but skip this if only showing errors
        else:
            try:
                current_val = str(eval(checker))
                sensor_status = str(eval(checker.split(".get_value()")[0] + ".status"))

                if type(min_val) is list:
                    if current_val in min_val and sensor_status not in sensor_status_errors:
                        if not show_only_errors:
                            if sensor_status == 'warn':
                                print "%s %s %s %s" % (col("brown") + checker.ljust(65),\
                                (current_val+" (" + sensor_status + ")").ljust(25),str(min_val).ljust(25), '' + col("normal"))
                            else:
                                print "%s %s %s %s" % (col("green") + checker.ljust(65),current_val.ljust(25),\
                                 str(min_val).ljust(25), '' + col("normal"))
                    else:
                        print "%s %s %s %s" % (col("red") + checker.ljust(65), (current_val+" (" + sensor_status + ")").ljust(25),\
                        str(min_val).ljust(25),'' + col("normal"))
                else:
                    if (min_val <= float(current_val) and float(current_val) <=  max_val) and sensor_status not in sensor_status_errors:
                        if not show_only_errors:
                            if sensor_status == 'warn':
                                print "%s %s %s %s" % (col("brown") + checker.ljust(65),\
                                (current_val+" (" + sensor_status + ")").ljust(25), str(min_val).ljust(25),\
                                str(max_val).ljust(25) + col("normal"))
                            else:
                                print "%s %s %s %s" % (col("green") + checker.ljust(65),\
                                 current_val.ljust(25), str(min_val).ljust(25), str(max_val).ljust(25) + col("normal"))
                    else:
                        print "%s %s %s %s" % (col("red") + checker.ljust(65), (current_val+" (" + sensor_status + ")").ljust(25),\
                        str(min_val).ljust(25), str(max_val).ljust(25) + col("normal"))
            except Exception, e:
                print "Could not check", checker, "[expected range: %r , %r]" % (min_val,max_val)
                print e


if __name__ == "__main__":

    parser = OptionParser(usage="%prog [options]",
                          description="Perform basic status (blue = busy) and health check of the system for observers. " +
                          "Can be run at any time without affecting current settings/observation. " +
                          "Default is to show all sensors checked, but the -e option to show errors only may prove popular.",
                          epilog = 'Examples: "basic_health_check.py -e", or "basic_health_check.py -g ants"')
    parser.add_option('-s', '--system', help='System configuration file to use, relative to conf directory ' +
                      '(default reuses existing connection, or falls back to systems/local.conf)')
    sensor_group_keys = sensor_group.keys()
    sensor_group_keys.sort(key=str.lower) # for some reason python does not like to do this in one line
    parser.add_option('-g', '--sensor_group', default="karoo",
                      help='Selected sensor group to use: ' + '|'.join(sensor_group_keys) + ' (default="%default")')
    parser.add_option('-e', '--errors_only', action='store_true', default=False,
                      help='Show only values in error, if this switch is included (default="%default")')

    (opts, args) = parser.parse_args()

    try:
        selected_sensors = sensor_group[opts.sensor_group]
    except KeyError:
        print "Unknown sensor group '%s', expected one of %s" % (opts.sensor_group, sensor_group_keys)
        sys.exit()

    # Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
    # This connects to all the proxies and devices and queries their commands and sensors
    try:
        kat = katuilib.tbuild(opts.system)
    # Fall back to *local* configuration to prevent inadvertent use of the real hardware
    except ValueError:
        kat = katuilib.tbuild('systems/local.conf')
    print "Using KAT connection with configuration: %s" % (kat.config_file,)
    print "Current local time: " + time.ctime()

    try:
        if kat.dbe7.sensor.k7w_script_status.get_value() == 'busy':
            print 'Current script running: %s' \
                  % ( col('blue') + kat.dbe7.sensor.k7w_script_name.get_value() + \
                  ' (' + kat.dbe7.sensor.k7w_script_description.get_value() + ') with status: "' +  \
                  " ".join(kat.dbe7.sensor.k7w_script_log.get_value().split()) + '"' + col('normal') )
            print 'Current observer: %s' % (col('blue') + kat.dbe7.sensor.k7w_script_observer.get_value() + col('normal'))
        else:
            print 'Current script running: none'
        dbe_mode = kat.dbe7.sensor.dbe_mode.get_value()
        if dbe_mode == 'wbc':
            print 'Current DBE7 mode: %s' % str(dbe_mode)
        elif dbe_mode == 'wbc8k':
            print 'Current DBE7 mode: %s' % (col('brown') + str(dbe_mode) + col('normal'))
        else:
            print 'Current DBE7 mode: %s' % (col('red') + 'unknown' + col('normal'))
        print 'Current centre frequency: %s MHz' % (str(kat.rfe7.sensor.rfe7_lo1_frequency.get_value() / 1e6 - 4200.))


        # Some fancy footwork to list antennas by target after retrieving target per antenna.
        # There may be a neater/more compact way to do this, but a dict with target strings as keys
        # and an expanding list of antennas corresponding to each target as values did not work. Hence
        # the more explicit approach here.
        ants = katuilib.observe.ant_array(kat,'all')
        tgt_index = {} # target strings as keys with values as a zero-based index to ant_list list of lists
        ant_list = [] # list of lists of antennas per target
        locks = []
        modes = []
        for ant in ants.devs:
            tgt = ant.sensor.target.get_value()
            if tgt == '' : tgt = 'None'
            if not tgt_index.has_key(tgt):
                tgt_index[tgt] = len(tgt_index)
                ant_list.append([ant.name])
            else:
                ant_list[tgt_index[tgt]].append(ant.name)
            locks.append(ant.sensor.lock.get_value())
            modes.append(ant.sensor.mode.get_value())
        print '\nCurrent targets for antennas (green = locked):'
        tgt_index_keys = tgt_index.keys()
        tgt_index_keys.sort(key=str.lower) # order targets alphabetically

        # print list of targets with corresponding antennas (locked ones in green)
        for key in tgt_index_keys:
            ant_list_str = '['
            for ant in ant_list[tgt_index[key]]:
                if locks[int(ant.split('ant')[1])-1] == '1':
                    ant_list_str = ant_list_str + col("green") + str(ant) + col("normal") + ','
                else:
                    ant_list_str = ant_list_str + str(ant) + ','
            if str(key) is not 'None':
                print '  ' + col('blue') + str(key) + col('normal') +' : ' + ant_list_str[0:len(ant_list_str)-1] + ']' # remove extra trailing comma
            else:
                print '  ' + str(key) +' : ' + ant_list_str[0:len(ant_list_str)-1] + ']' # remove extra trailing comma
        print 'Antenna lock: ' + str(locks) # also useful to show locks in this fashion (single glance)
        ant_mode_str = '['
        for mode in modes:
            if mode == 'POINT':
                ant_mode_str = ant_mode_str + col('blue') + str(mode) + col('normal') + ', '
            else:
                ant_mode_str = ant_mode_str + str(mode) + ', '
        print 'Antenna mode:' + ant_mode_str[0:len(ant_mode_str)-2] + ']'
    except Exception, e:
        print "Error: could not retrieve centre frequency or antenna target/lock info..."
        print '(' + str(e) + ')'
        
    print "\nChecking basic health (red => potential problem) ..."
    check_sensors(kat,selected_sensors,opts.errors_only)

