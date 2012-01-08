#!/usr/bin/python
# Check for basic system health. Intended for the use of an observer.
# Code: Jasper Horrell (jasper@ska.ac.za)

# Basic philosophy is for a light-weight, non-intrusive set of checks of the overall
# and sensors that an observer is likely to be interested in (i.e. not geared for full
# engineering diagnosis of the system). Intended to be especially useful for (very)
# remote observations where the command line is being used, but will also be more
# generally useful in standard operations (a GUI/browser version of this is likely
# to be made available).
#
# The max and min values specified in this script are aimed at what an observer
# should look out for (rather than perhaps an engineer) and so necessarily may not
# be in sync with the value used for alarms etc. although there should not be major
# differences. The script does currently check and display (for error statuses) the
# sensor status along with the current sensor values. In time, this will help
# to iron out any discrepancies between the two sources of expected sensor ranges.

from optparse import OptionParser
import time
import sys

import katuilib
from katuilib.ansi import col

# Sensor groups (and templates for creating sensor groups).
# Structure is list of tuples with either (command to access sensor value, min value, max value)
# or (command, list of string options, blank string)

ant_template = [ # the rfe7_template and dbe7_template sensors get added to this
("kat.ped#.sensor.cryo_lna_temperature.get_value()", 60.0,80.0),
("kat.ped#.sensor.bms_chiller_flow_present.get_value()", 1,1),
("kat.ped#.sensor.rfe3_psu_on.get_value()", 1,1),
("kat.ped#.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1,1),
("kat.ped#.sensor.rfe3_rfe15_noise_pin_on.get_value()", 0,0),
("kat.ped#.sensor.rfe3_rfe15_noise_coupler_on.get_value()", 0,0),
("kat.ant#.sensor.mode.get_value()",["POINT","STOP","STOW","SCAN"],''),
("kat.ant#.sensor.windstow_active.get_value()",0,0),
("kat.ant#.sensor.pos_actual_scan_azim.get_value()",-185.0,275.0),
("kat.ant#.sensor.pos_actual_scan_elev.get_value()",2.0,95.0),
]

rfe7_template = [
("kat.rfe7.sensor.rfe7_downconverter_ant#_h_powerswitch.get_value()", 1,1),
("kat.rfe7.sensor.rfe7_downconverter_ant#_v_powerswitch.get_value()", 1,1),
]

rfe7_base_group = [
("kat.mon_kat_proxy.sensor.agg_rfe7_psu_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_orx1_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_orx2_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_orx3_states_ok.get_value()", 1,1),
("kat.mon_kat_proxy.sensor.agg_rfe7_osc_states_ok.get_value()", 1,1),
("","",""), # creates a blank line
]

dbe7_template = [
("kat.dbe7.sensor.dbe_ant#h_adc_power.get_value()",-27.0,-20.0),
("kat.dbe7.sensor.dbe_ant#v_adc_power.get_value()",-27.0,-20.0),
]

dbe7_base_group = [
("kat.dbe7.sensor.dbe_mode.get_value()",['wbc','wbc8k'],''),
("","",""), # creates a blank line
]

dc_group = [
("kat.dbe7.sensor.k7w_status.get_value()",['init','idle','capturing','complete'],''),
("kat.nm_kat_dc1.sensor.k7capture_running.get_value()",1,1),
("kat.nm_kat_dc1.sensor.k7aug_running.get_value()",1,1),
("kat.nm_kat_dc1.sensor.k7arch_running.get_value()",1,1),
("","",""), # creates a blank line
]

tfr_template = [
("kat.ant#.sensor.antenna_acu_ntp_time.get_value()",1,1),
]

tfr_base_group = [
("kat.mon_kat_proxy.sensor.agg_anc_tfr_time_synced.get_value()",1,1),
("kat.mon_kat_proxy.sensor.agg_anc_css_ntp_synch.get_value()",1,1), # does this include kat-dc1?
("kat.mon_kat_proxy.sensor.agg_anc_css_ut1_current.get_value()",1,1),
("kat.mon_kat_proxy.sensor.agg_anc_css_tle_current.get_value()",1,1),
("kat.dbe7.sensor.dbe_ntp_synchronised.get_value()",1,1),
("","",""), # creates a blank line
]

anc_group = [
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

lab_rfe7_group = [
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1,1),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1,1),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1,1),
("","",""), # creates a blank line
]


# Dictionary containing selectable sensor groups, identified by name (user selects one of these at runtime).
# Where there are empty lists, entries get added by the generate_sensor_groups() method at run-time.
# Had thoughts of generating the per antenna entries progammatically here based on selected antennas only,
# but these need to exist for the basic script help i.e. before the kat object is interrogated.
sensor_group_dict = {
'karoo' : [] + rfe7_base_group + dbe7_base_group + dc_group + tfr_base_group + anc_group, # antenna groups get added to this
'ant1' : [],'ant2' : [],'ant3' : [],'ant4' : [],'ant5' : [],'ant6' : [],'ant7' : [],
'ants' : [],
'rfe7' : [], # per antenna sensors get added
'dbe7' : [], # per antenna sensors get added
'dc' : dc_group,
'tfr' : [], #per antenna sensors get added
'anc' : anc_group,
'lab_rfe7' : lab_rfe7_group,
'lab' : [], # per antenna sensors get added
}

def generate_sensor_groups(kat,selected_ants,sensor_groups):

    """Create the per antenna sensor groups programmatically based on the selected antennas
    """
    dbe7_ants_group = [] # per antenna dbe7 sensors
    rfe7_ants_group = [] # per antenna rfe7 sensors
    tfr_ants_group = [] # per antenna tfr sensors

    ants = katuilib.observe.ant_array(kat,selected_ants)
    for ant in ants.devs:
        i = ant.name.split('ant')[1]
        for sensor in ant_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
        for sensor in rfe7_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
            rfe7_ants_group.append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
        for sensor in dbe7_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
            dbe7_ants_group.append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
        for sensor in tfr_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
            tfr_ants_group.append((sensor[0].replace('#',str(i)),sensor[1],sensor[2]))
        sensor_groups[ant.name].append(("","","")) # add a blank line to the per antenna group
        sensor_groups['ants'] =  sensor_groups['ants'] + sensor_groups[ant.name]
    sensor_groups['karoo'] = sensor_groups['ants'] + sensor_groups['karoo']
    sensor_groups['dbe7'] = dbe7_ants_group + dbe7_base_group
    sensor_groups['rfe7'] = rfe7_ants_group + rfe7_base_group
    sensor_groups['tfr'] = tfr_ants_group +  tfr_base_group
    sensor_groups['lab'] = sensor_groups['ant1'] + lab_rfe7_group


sensor_status_errors = ['failure','error','unknown'] # other possibilities are warn, nominal

def check_sensors(kat, selected_sensors, opts):
    """ Check current system setting and compare with expected range as specified above.
    Things appear colour-coded according to whether in expected range and if sensor status.
    """

    print '%s %s %s %s' % ('Sensor'.ljust(65), 'Value (sensor status)'.ljust(25),'Min Expected'.ljust(25), 'Max Expected'.ljust(25))
    potential_problems = False
    
    if len(selected_sensors) == 0:
        potential_problems = True
        print col('red') + 'No sensors to check! Are you sure this device is connected?' + col('normal')
    
    for checker, min_val, max_val in selected_sensors:
        if checker.strip() == '':
            if opts.show_full: print '' # print a blank line, but skip this if only showing errors
        else:
            try:
                current_val = str(eval(checker))
                sensor_status = str(eval(checker.split('.get_value()')[0] + '.status'))

                if current_val == 'None':
                    potential_problems = True
                    print '%s %s %s %s' % (col('red') + checker.ljust(65), ('<no value> (' + sensor_status + ')').ljust(25),\
                    str(min_val).ljust(25), str(max_val).ljust(25) + col('normal'))
                elif type(min_val) is list:
                    if current_val in min_val and sensor_status not in sensor_status_errors:
                        if opts.show_full:
                            if sensor_status == 'warn':
                                print '%s %s %s %s' % (col('brown') + checker.ljust(65),\
                                (current_val+' (' + sensor_status + ')').ljust(25),str(min_val).ljust(25), '' + col('normal'))
                            else:
                                print '%s %s %s %s' % (col('green') + checker.ljust(65),current_val.ljust(25),\
                                 str(min_val).ljust(25), '' + col('normal'))
                    else:
                        potential_problems = True
                        print '%s %s %s %s' % (col('red') + checker.ljust(65), (current_val+' (' + sensor_status + ')').ljust(25),\
                        str(min_val).ljust(25),'' + col('normal'))
                else:
                    if (min_val <= float(current_val) and float(current_val) <=  max_val) and sensor_status not in sensor_status_errors:
                        if opts.show_full:
                            if sensor_status == 'warn':
                                print '%s %s %s %s' % (col('brown') + checker.ljust(65),\
                                (current_val+' (' + sensor_status + ')').ljust(25), str(min_val).ljust(25),\
                                str(max_val).ljust(25) + col('normal'))
                            else:
                                print '%s %s %s %s' % (col('green') + checker.ljust(65),\
                                 current_val.ljust(25), str(min_val).ljust(25), str(max_val).ljust(25) + col('normal'))
                    else:
                        potential_problems = True
                        print '%s %s %s %s' % (col('red') + checker.ljust(65), (current_val+' (' + sensor_status + ')').ljust(25),\
                        str(min_val).ljust(25), str(max_val).ljust(25) + col('normal'))
            except Exception, e:
                potential_problems = True
                print col('red') + 'Could not check ',checker, ' [expected range: %r , %r]' % (min_val,max_val)
                print str(e) + col('normal')

    if potential_problems:
        print col('red') + 'Some potential problems found' + col('normal')
    else:
        print col('green') + 'All seems well :)' + col('normal')

def run(kat, opts, selected_sensors):
    try:
        if kat.dbe7.sensor.k7w_script_status.get_value() == 'busy':
            print 'Current script running: %s' \
                  % ( col('blue') + kat.dbe7.sensor.k7w_script_name.get_value() + \
                  ' (' + kat.dbe7.sensor.k7w_script_description.get_value() + ') with status: "' +  \
                  ' '.join(kat.dbe7.sensor.k7w_script_log.get_value().split()) + '"' + col('normal') )
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
        ants = katuilib.observe.ant_array(kat,opts.ants)
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
                    ant_list_str = ant_list_str + col('green') + str(ant) + col('normal') + ','
                else:
                    ant_list_str = ant_list_str + str(ant) + ','
            if str(key) is not 'None':
                print '  ' + col('blue') + str(key) + col('normal') +' : ' + ant_list_str[0:len(ant_list_str)-1] + ']' # remove extra comma
            else:
                print '  ' + str(key) +' : ' + ant_list_str[0:len(ant_list_str)-1] + ']' # remove extra trailing comma
        print 'Antenna lock: ' + str(locks) # also useful to show single-glance locks
        ant_mode_str = '['
        for mode in modes:
            if mode == 'POINT':
                ant_mode_str = ant_mode_str + col('blue') + str(mode) + col('normal') + ', '
            else:
                ant_mode_str = ant_mode_str + str(mode) + ', '
        print 'Antenna mode:' + ant_mode_str[0:len(ant_mode_str)-2] + ']'
    except Exception, e:
        print col('red') + '\nERROR: could not retrieve status info... ' + col('normal')
        print col('red') + '(' + str(e) + ')' + col('normal')
    
    if not opts.busy_only:
        print '\nChecking basic health (red => potential problem) ...'
        check_sensors(kat,selected_sensors,opts)
    else:
        print 'No checking of sensors performed (as per user selection).'

if __name__ == '__main__':

    parser = OptionParser(usage='%prog [options]',
                          description='Perform basic status (blue = busy) and health check of the system for observers. ' +
                          'Can be run at any time without affecting current settings/observation.',
                          epilog = 'Examples: "basic_health_check.py -e", or "basic_health_check.py -g ants"')
    parser.add_option('-s', '--system', help='System configuration file to use, relative to conf directory ' +
                      '(default reuses existing connection, or falls back to systems/local.conf)')
    sensor_group_dict_keys = sensor_group_dict.keys()
    sensor_group_dict_keys.sort(key=str.lower) # for some reason python does not like to do this in one line
    parser.add_option('-g', '--sensor_group', default='karoo',
                      help='Selected sensor group to use: ' + '|'.join(sensor_group_dict_keys) + ' (default="%default")')
    parser.add_option('-a', '--ants', default='all',
                      help="Comma-separated list of antennas to include (e.g. 'ant1,ant2'), "+
                      "or 'all' for all antennas (default='%default').")
    parser.add_option('-f', '--show_full', action='store_true', default=False,
                      help='Show all sensors that are checked. Default is to only show values in error. (default="%default")')
    parser.add_option('-b', '--busy_only', action='store_true', default=False,
                    help='Show only header (busy) info. Skip error checks. (default="%default")')
    parser.add_option('-r', '--refresh', default=0,
                      help='Re-run every r secs where min non-zero value is 1 sec (default="%default")')
    parser.add_option('-m', '--max_duration', default=-1,
                    help='Stop refreshing after specified secs. Works with the -r option e.g. set to length of ' +
                    'observation run. Default is to continue indefinitely if -r option is used. (default="%default")')

    (opts, args) = parser.parse_args()

    # Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
    # This connects to all the proxies and devices and queries their commands and sensors
    try:
        kat = katuilib.tbuild(opts.system)
    # Fall back to *local* configuration to prevent inadvertent use of the real hardware
    except ValueError:
        kat = katuilib.tbuild('systems/local.conf')
    print 'Using KAT connection with configuration: %s' % (kat.config_file,)

    # construct the per antenna sensor groups (restricting to those that were selected)
    generate_sensor_groups(kat,opts.ants,sensor_group_dict)

    try:
        selected_sensors = sensor_group_dict[opts.sensor_group]
    except KeyError:
        print 'Unknown sensor group "%s", expected one of %s' % (opts.sensor_group, sensor_group_dict_keys)
        sys.exit()

    if opts.refresh > 0:
        if opts.max_duration > 0: end_time = time.time() + float(opts.max_duration)
        ended = False
        while (not ended):
            print '\nCurrent local time: ' + time.ctime()
            run(kat,opts,selected_sensors)
            if opts.max_duration > opts.refresh:
                if time.time() > end_time: ended = True
                print 'Checking every ' + opts.refresh + ' secs (ends in ' + str(int(end_time-time.time())) + ' secs or Control-c to break)'
            else:
                print 'Checking every ' + opts.refresh + ' secs (Control-c to break)'
            time.sleep(max(1.0,float(opts.refresh))) # don't try go faster than 1 sec
    else:
        print '\nCurrent local time: ' + time.ctime()
        run(kat,opts,selected_sensors)
