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

# TODO:
# - option to load sensors of interest from file ("custom" group).
# - option to write output to file as well as display on screen?
# - think about some way to optionally auto select to the script ants (else at least
#   display them) in the status - a bit tricky since requires refresh to change

from optparse import OptionParser
import sys, math, time

import katcorelib, katpoint
from katcorelib.defaults import activity_logger, user_logger
from katmisc.utils.ansi import col, getKeyIf

# Some globals
busy_colour = 'blue'
ok_colour = 'green'
critical_colour = 'blinkred'
critical_colour2 = 'blinkyellow'
error_colour = 'red'
warn_colour = 'brown'
normal_colour = 'normal'
sensor_status_errors = ['failure','error','unknown'] # other possibilities are warn, nominal
K7 = katpoint.Antenna('K7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0') # array centre position
quiet_check_refresh = 5 # time in secs between sensor checks in quiet mode (under the hood)


# Sensor groups (and templates for creating sensor groups).

# Structure is a list of tuples with either:
#   (command to access sensor value, min value, max value, m, n) or
#   (command, list of string options, blank string, m, n)
# where:
#   m and n specify an "m from n" check for quiet mode
#   (error is raised if sensor falls in error range in m out of n consecutive checks)

ant_template = [ # the rfe7_template and dbe7_template sensors get added to this
("kat.ant#.sensor.cryo_lna_temperature.get_value()", 67.0,87.0,1,1),
("kat.ant#.sensor.bms_chiller_flow_present.get_value()", 1,1,1,1),
("kat.ant#.sensor.rfe3_psu_on.get_value()", 1,1,1,1),
("kat.ant#.sensor.rfe3_psu_ok.get_value()", 1,1,1,1),
("kat.ant#.sensor.rfe3_rfe15_rfe1_lna_psu_on.get_value()", 1,1,1,1),
("kat.ant#.sensor.rfe3_rfe15_rfe1_lna_psu_ok.get_value()", 1,1,1,1),
("kat.ant#.sensor.rfe3_rfe15_noise_pin_on.get_value()", ['0','1'],'',1,1),
("kat.ant#.sensor.rfe3_rfe15_noise_coupler_on.get_value()", ['0','1'],'',1,1),
("kat.ant#.sensor.mode.get_value()",["POINT","STOP","STOW","SCAN"],'',1,1),
("kat.ant#.sensor.activity.get_value()",["track","slew","scan_ready","scan","scan_complete","stop","stow"],'',1,1), # wind_stow will show as error
("kat.ant#.sensor.pos_actual_scan_azim.get_value()",-185.0,275.0,1,1),
("kat.ant#.sensor.pos_actual_scan_elev.get_value()",2.0,95.0,1,1),
]

rfe7_template = [ # used to check powerswitch sensors here.
]

rfe7_base_group = [
("kat.sensors.agg_rfe7_psu_states_ok.get_value()", 1,1,1,1),
("kat.sensors.agg_rfe7_orx1_states_ok.get_value()", 1,1,1,1),
("kat.sensors.agg_rfe7_orx2_states_ok.get_value()", 1,1,1,1),
("kat.sensors.agg_rfe7_orx3_states_ok.get_value()", 1,1,1,1),
("kat.sensors.agg_rfe7_osc_states_ok.get_value()", 1,1,1,1),
("","","","",""), # creates a blank line
]

dbe7_template = [
("kat.dbe7.sensor.dbe_ant#h_adc_power.get_value()",-28.0,-24.0,4,6),
("kat.dbe7.sensor.dbe_ant#v_adc_power.get_value()",-28.0,-24.0,4,6),
("kat.dbe7.sensor.dbe_ant#h_fft_overrange.get_value()", 0,0,1,1),
("kat.dbe7.sensor.dbe_ant#v_fft_overrange.get_value()", 0,0,1,1),
("kat.dbe7.sensor.dbe_ant#h_adc_overrange.get_value()", 0,0,2,5),
("kat.dbe7.sensor.dbe_ant#v_adc_overrange.get_value()", 0,0,2,5),
("kat.dbe7.sensor.dbe_ant#h_adc_terminated.get_value()", 0,0,1,1),
("kat.dbe7.sensor.dbe_ant#v_adc_terminated.get_value()", 0,0,1,1),
]

dbe7_base_group = [
("kat.dbe7.sensor.dbe_corr_lru_available.get_value()", 1,1,1,1),
("kat.dbe7.sensor.dbe_mode.get_value()",['wbc','wbc8k'],'',1,1),
("","","","",""), # creates a blank line
]

dc_group = [
("kat.dbe7.sensor.k7w_status.get_value()",['init','idle','capturing','complete'],'',1,1),
("kat.nm_kat_dc1.sensor.k7capture_running.get_value()",1,1,1,1),
("kat.nm_kat_dc1.sensor.k7aug_running.get_value()",1,1,1,1),
("kat.nm_kat_dc1.sensor.k7arch_running.get_value()",1,1,1,1),
("","","","",""), # creates a blank line
]

tfr_template = [
("kat.ant#.sensor.antenna_acu_ntp_time.get_value()",1,1,1,1),
]

tfr_base_group = [
("kat.sensors.agg_anc_tfr_time_synced.get_value()",1,1,1,1),
("kat.sensors.agg_anc_css_ntp_synch.get_value()",1,1,1,1), # does this include kat-dc1?
("kat.sensors.agg_anc_css_ut1_current.get_value()",1,1,1,1),
("kat.sensors.agg_anc_css_tle_current.get_value()",1,1,1,1),
("kat.dbe7.sensor.dbe_ntp_synchronised.get_value()",1,1,1,1),
("","","","",""), # creates a blank line
]

anc_group = [
("kat.anc.sensor.asc_asc_air_temperature.get_value()", 0.0,32.0,4,6),
("kat.anc.sensor.asc_chiller_water_temperature.get_value()", 6.0,22.0,1,1),
("kat.anc.sensor.cc_cc_air_temperature.get_value()", 0.0,30.0,4,6),
("kat.anc.sensor.cc_chiller_water_temperature.get_value()", 6.0,18.0,1,1),
("kat.anc.sensor.asc_wind_speed.get_value()", -0.5,12.5,10,20), # the occasional small negative windspeeds are 'interesting'
("kat.anc.sensor.asc_fire_ok.get_value()", 1,1,1,1), # these sensors really should be something like "(not) on fire"
("kat.anc.sensor.cc_fire_ok.get_value()", 1,1,1,1),
("kat.anc.sensor.cmc_fire_ok.get_value()", 1,1,1,1),
("kat.anc.sensor.asc_ups_battery_not_discharging.get_value()", 1,1,1,1),
("kat.anc.sensor.asc_ups_ok.get_value()", 1,1,1,1),
("kat.anc.sensor.cc_ups_battery_not_discharging.get_value()", 1,1,1,1),
("kat.anc.sensor.cc_ups_fault.get_value()", 0,0,1,1),
("","","","",""), # creates a blank line
]

lab_rfe7_group = [
("kat.rfe7.sensor.rfe7_downconverter_ant1_h_powerswitch.get_value()", 1,1,1,1),
("kat.rfe7.sensor.rfe7_downconverter_ant1_v_powerswitch.get_value()", 1,1,1,1),
("kat.rfe7.sensor.rfe7_orx1_powerswitch.get_value()", 1,1,1,1),
("","","","",""), # creates a blank line
]

# Dictionary containing selectable sensor groups, identified by name (user selects one of these at runtime).
# Where there are empty lists, (usually per antenna) entries get added by the generate_sensor_groups()
# method at run-time. Had thoughts of generating the per antenna entries progammatically here based on
# selected antennas only, but these need to exist for the basic script help i.e. before the kat object is
#interrogated.
sensor_group_dict = {
'karoo' : [] + rfe7_base_group + dbe7_base_group + dc_group + tfr_base_group + anc_group,
'ant1' : [],'ant2' : [],'ant3' : [],'ant4' : [],'ant5' : [],'ant6' : [],'ant7' : [],
'ants' : [],
'rfe7' : [],
'dbe7' : [],
'dc' : dc_group,
'tfr' : [],
'anc' : anc_group,
'lab_rfe7' : lab_rfe7_group,
'lab' : [],
}

def generate_sensor_groups(kat,selected_ants,sensor_groups):

    """Create the per antenna sensor groups programmatically based on the selected antennas.
    The way this set up, a sensor may be shared between multiple sensor groups. This is handy
    for example if one wants to see all the dbe7 power levels together (-g dbe7) and also see
    the per antenna dbe7 power levels with each antenna (-g ants)
    """
    dbe7_ants_group = [] # per antenna dbe7 sensors
    rfe7_ants_group = [] # per antenna rfe7 sensors
    tfr_ants_group = [] # per antenna tfr sensors

    ants = katcorelib.observe.ant_array(kat,selected_ants)
    for ant in ants:
        i = ant.name.split('ant')[1]
        for sensor in ant_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
        for sensor in rfe7_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
            rfe7_ants_group.append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
        for sensor in dbe7_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
            dbe7_ants_group.append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
        for sensor in tfr_template:
            sensor_groups[ant.name].append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
            tfr_ants_group.append((sensor[0].replace('#',str(i)),sensor[1],sensor[2],sensor[3],sensor[4]))
        sensor_groups[ant.name].append(("","","","","")) # add a blank line to the per antenna group
        sensor_groups['ants'] =  sensor_groups['ants'] + sensor_groups[ant.name]
    sensor_groups['karoo'] = sensor_groups['ants'] + sensor_groups['karoo']
    sensor_groups['dbe7'] = dbe7_ants_group + dbe7_base_group
    sensor_groups['rfe7'] = rfe7_ants_group + rfe7_base_group
    sensor_groups['tfr'] = tfr_ants_group +  tfr_base_group
    sensor_groups['lab'] = sensor_groups['ant1'] + lab_rfe7_group


def print_sensor(colour,sensor,val,min_val,max_val,m,n,quiet,action=''):
    if quiet and sensor != '' and action == 'trigger':
        print '%s - %s : (%s, %s, %s, %s/%s)%s' % (col(colour)+time.ctime()+' (trigger)',sensor, val,\
        str(min_val),str(max_val),str(m),str(n),col(normal_colour))
    elif quiet and sensor != '' and action == 'clear':
        print '%s - %s : (%s, %s, %s, %s/%s )%s' % (col(colour)+time.ctime()+' (cleared)',sensor, val,\
        str(min_val),str(max_val),str(m),str(n),col(normal_colour))
    else:
        print '%s %s %s %s' % (col(colour)+sensor.ljust(65),\
          (val).ljust(25),str(min_val).ljust(25),str(max_val).ljust(25)+col(normal_colour))


def update_alarm(alarms,sensor,m,n,action):
    updated = False
    if m==1 and n==1: # if not m from n case
        if action == 'trigger':
            if not alarms.has_key(sensor):
                alarms[sensor] = 1 # the 1 is arb and not used currently
                updated = True
        elif action == 'clear':
            if alarms.has_key(sensor):
                alarms.pop(sensor) # remove sensor from alarm
                updated = True
    else: # for an m from n case:
        if not alarms.has_key(sensor):
            if action == 'trigger':
                alarms[sensor] = [1,1] # 1 from 1
            else: # action must be 'clear'
                alarms[sensor] = [0,1] # 0 from 1
        else:
            if alarms[sensor][1] < n:
                alarms[sensor][1] = alarms[sensor][1] + 1
                if action == 'trigger':
                    alarms[sensor][0] = alarms[sensor][0] + 1
                    if alarms[sensor][0] == m:
                        updated = True
            else: # n values have already been checked
                if action == 'trigger' and alarms[sensor][0] < n:
                    alarms[sensor][0] = alarms[sensor][0] + 1
                    if alarms[sensor][0] == m: updated = True
                else: # action must be 'clear'
                    if alarms[sensor][0] == m: updated = True
                    alarms[sensor][0] = max(alarms[sensor][0] - 1,0)
    return updated


def check_sensors(kat, opts, selected_sensors, quiet=False, alarms={}):
    """ Check current system setting and compare with expected range as specified above.
    Things appear colour-coded according to whether in expected range and sensor status.
    """
    
    potential_problems = False
    
    if len(selected_sensors) == 0:
        potential_problems = True
        print col(error_colour) + 'No sensors to check! Are you sure this device is connected?' + col(normal_colour)
    
    for checker, min_val, max_val, m, n in selected_sensors:
        if checker.strip() == '':
            if opts.verbose: print '' # print a blank line, but skip this if only showing errors
        else:
            try:
                current_val = str(eval(checker))
                sensor_status = str(eval(checker.split('.get_value()')[0] + '.status'))
                sensor_ok = False
                quiet_warning = False

                if current_val == 'None':
                    potential_problems = True
                    if not quiet or (quiet and update_alarm(alarms,checker,m,n,'trigger')):
                        print_sensor(error_colour,checker,'<no value>',min_val,max_val,m,n,quiet,'trigger')
                    return potential_problems
                
                if type(min_val) is list:
                    if current_val in min_val and sensor_status not in sensor_status_errors:
                        sensor_ok = True
                elif (min_val <= float(current_val) and float(current_val) <=  max_val) and sensor_status not in sensor_status_errors:
                    sensor_ok = True
                
                if quiet and opts.warn and sensor_status == 'warn' and sensor_ok == True:
                    sensor_ok = False
                    quiet_warning = True

                if sensor_ok:
                    if opts.verbose: # won't be verbose in quiet mode (check is performed earlier)
                        if sensor_status == 'warn':
                            print_sensor(warn_colour,checker,str(current_val)+' (' + sensor_status + ')',min_val,max_val,m,n,quiet)
                        else:
                            print_sensor(ok_colour,checker,current_val,min_val,max_val,m,n,quiet,'clear')
                    elif quiet and update_alarm(alarms,checker,m,n,'clear'): # quiet mode where sensor alarm toggles to 'clear'
                        print_sensor(ok_colour,checker,current_val,min_val,max_val,m,n,quiet,'clear')
                else:
                    potential_problems = True
                    if not quiet or (quiet and update_alarm(alarms,checker,m,n,'trigger')):
                        if quiet_warning:
                            print_sensor(warn_colour,checker,str(current_val)+' (' + sensor_status + ')',min_val,max_val,m,n,quiet,'trigger')
                        else:
                            print_sensor(error_colour,checker,str(current_val)+' (' + sensor_status + ')',min_val,max_val,m,n,quiet,'trigger')
                            
            except Exception, e:
                potential_problems = True
                print col(error_colour) + 'Could not check ',checker, ' [expected range: %r , %r]' % (min_val,max_val)
                print str(e) + col(normal_colour)

    return potential_problems

def show_status_header(kat, opts, selected_sensors):
    # Might want to split out presentation from logic soon...
    try:
        dbe_mode = kat.dbe7.sensor.dbe_mode.get_value()
        if dbe_mode == 'wbc': dbe_mode_colour = normal_colour
        elif dbe_mode == 'wbc8k': dbe_mode_colour = warn_colour
        else: dbe_mode = 'unknown'; dbe_mode_color = error_colour
        system_centre_freq = kat.rfe7.sensor.rfe7_lo1_frequency.get_value() / 1e6 - 4200. # most reliable place to get this

        # Retrieve weather info
        air_pressure = kat.anc.sensor.asc_air_pressure.get_value()
        air_humidity = kat.anc.sensor.asc_air_relative_humidity.get_value()
        air_temperature= kat.anc.sensor.asc_air_temperature.get_value()
        wind_direction = kat.anc.sensor.asc_wind_direction.get_value()
        wind_speed = kat.anc.sensor.asc_wind_speed.get_value()
        lights = kat.anc.sensor.asccombo_floodlights_on.get_value()
        lights_str = '**ON**' if lights == '1' else 'Off'
        lights_colour = normal_colour if lights == '1' else busy_colour

        if kat.dbe7.sensor.k7w_script_status.get_value() == 'busy':

            # Retrieve some script relevant info
            script_name = kat.dbe7.sensor.k7w_script_name.get_value()
            script_description = kat.dbe7.sensor.k7w_script_description.get_value()
            observer = kat.dbe7.sensor.k7w_script_observer.get_value()
            start_time =  float(kat.dbe7.sensor.k7w_script_starttime.get_value())
            script_arguments = kat.dbe7.sensor.k7w_script_arguments.get_value()

            dump_rate_str = kat.dbe7.sensor.k7w_script_rf_params.get_value().split('Dump rate=')[1] # will produce string e.g. '1 Hz'
            packets_captured = kat.dbe7.sensor.k7w_packets_captured.get_value()
            data_file = ''
            data_file_req = kat.dbe7.req.k7w_get_current_file()
            if len(data_file_req) == 2: data_file = data_file_req[1].split('/')[-1] # should produce e.g. '1326186470.writing.h5'

            max_duration_ok = False
            if script_arguments.find(" -m ") != -1:
                max_duration = float(script_arguments.split(' -m ')[1].split()[0])
                max_duration_ok = True

            if dbe_mode_colour == normal_colour: dbe_mode_colour = busy_colour

            print '# Script running: %s' % ( col(busy_colour)+script_name+' - "'+script_description+'" by '+observer+col(normal_colour) )
            if max_duration_ok:
                print '# Run times (local time): %s -> %s (%.2f hours)%s' \
                  % (col(busy_colour) + time.ctime(start_time),time.ctime(start_time+max_duration),max_duration/3600.0,col(normal_colour))
            else:
                print '# Start time (localtime): %s' % (col(busy_colour) + time.ctime(start_time) + col(normal_colour))

            print '# Data file: %s %s' % (col(busy_colour)+data_file, '('+str(packets_captured)+' packets captured)'+col(normal_colour) )
            print '# DBE7 mode & dump rate: %s (%s)%s' \
              % (col(dbe_mode_colour)+dbe_mode+col(normal_colour),col(busy_colour)+dump_rate_str,col(normal_colour))
            print '# RF centre freq: %s' %(col(busy_colour)+str(system_centre_freq)+' MHz'+col(normal_colour))
            print '# Air pressure: %s' %(col(busy_colour)+str(air_pressure)+' mbar'+col(normal_colour))
            print '# Air humidity: %s' %(col(busy_colour)+str(air_humidity)+' percent'+col(normal_colour))
            print '# Air temperature: %s' %(col(busy_colour)+str(air_temperature)+' degC'+col(normal_colour))
            print '# Wind direction: %s' %(col(busy_colour)+str(wind_direction)+' deg'+col(normal_colour))
            print '# Wind speed: %s' %(col(busy_colour)+str(wind_speed)+' m/s'+col(normal_colour))
            print '# Floodlights are: %s' %(col(lights_colour)+lights_str+col(normal_colour))

        else:
            print '# Script running: none'
            print '# DBE7 mode and RF centre freq: %s @ %s MHz' % (col(dbe_mode_colour)+dbe_mode+col(normal_colour),system_centre_freq)
            print '# Air pressure: %s' %(col(busy_colour)+str(air_pressure)+' mbar'+col(normal_colour))
            print '# Air humidity: %s' %(col(busy_colour)+str(air_humidity)+' percent'+col(normal_colour))
            print '# Air temperature: %s' %(col(busy_colour)+str(air_temperature)+' degC'+col(normal_colour))
            print '# Wind direction: %s' %(col(busy_colour)+str(wind_direction)+' deg'+col(normal_colour))
            print '# Wind speed: %s' %(col(busy_colour)+str(wind_speed)+' m/s'+col(normal_colour))
            print '# Floodlights are: %s' %(col(lights_colour)+lights_str+col(normal_colour))

        # Some fancy footwork to list antennas by target after retrieving target per antenna.
        # There may be a neater/more compact way to do this, but a dict with target strings as keys
        # and an expanding list of antennas corresponding to each target as values did not work. Hence
        # the more explicit approach here.
        ants = katcorelib.observe.ant_array(kat,opts.ants)
        tgt_index = {} # target strings as keys with values as a zero-based index to ant_list list of lists
        ant_list = [] # list of lists of antennas per target
        locks, modes, activity = {}, {}, {}
        for ant in ants:
            tgt = ant.sensor.target.get_value()
            if tgt == '' or tgt == None: tgt = 'None'
            if not tgt_index.has_key(tgt):
                tgt_index[tgt] = len(tgt_index)
                ant_list.append([ant.name])
            else:
                ant_list[tgt_index[tgt]].append(ant.name)
            locks[ant.name] = ant.sensor.lock.get_value()
            modes[ant.name] = ant.sensor.mode.get_value()
            activity[ant.name] = ant.sensor.activity.get_value()

        # print antenna modes
        all_ants = modes.keys()
        all_ants.sort(key=str.lower) # sort alphabetically
        ant_mode_str = '['
        all_ants_ok = True
        for ant in all_ants:
            if modes[ant] == 'POINT' or modes[ant] == 'SCAN':
                ant_mode_str = ant_mode_str + col(busy_colour) + str(ant) +':' + str(modes[ant]) + col(normal_colour) + ', '
            else:
                if modes[ant] == 'ERROR': # This is bad
                    ant_mode_str = ant_mode_str + col(critical_colour) + str(ant) +':' + str(modes[ant]) + col(normal_colour) + ', '
                    all_ants_ok = False # This is not ok
                else: # Not so bad
                    ant_mode_str = ant_mode_str + col(warn_colour) + str(ant) +':' + str(modes[ant]) + col(normal_colour) + ', '
        if not all_ants_ok:
            print col(critical_colour2)+"\n============================================================"+col(normal_colour)
            print "======================"+col(critical_colour)+"Antenna Error!!!"+ col(normal_colour)+"======================"
            print col(critical_colour2)+"============================================================"+col(normal_colour)
        print '\n# Ant modes: ' + ant_mode_str[0:len(ant_mode_str)-2] + ']'
        if not all_ants_ok:
            print col(critical_colour2)+"\n============================================================"+col(normal_colour)
            print "======================"+col(critical_colour)+"Antenna Error!!!"+ col(normal_colour)+"======================"
            print col(critical_colour2)+"============================================================"+col(normal_colour)

        # print list of targets with corresponding antennas (locked ones in green)
        print '# Targets & antennas (orange => not locked):'
        tgt_index_keys = tgt_index.keys()
        tgt_index_keys.sort(key=str.lower) # order targets alphabetically
        for key in tgt_index_keys:
            ant_list_str = '['
            for ant in ant_list[tgt_index[key]]:
                if locks[ant] == '1':
                    ant_list_str = ant_list_str+col(busy_colour)+str(ant)+':'+str(activity[ant])+col(normal_colour)+', '
                else:
                    ant_list_str = ant_list_str+col(warn_colour)+str(ant)+':'+str(activity[ant])+col(normal_colour)+', '
            if str(key) is not 'None':
                tgt = katpoint.Target(key)
                az = tgt.azel(antenna = K7)[0]*180.0/math.pi
                el = tgt.azel(antenna = K7)[1]*180.0/math.pi
                print '  %s : %s] (az=%.2f,el=%.2f)' % (col(busy_colour)+str(key)+ col(normal_colour),ant_list_str[0:len(ant_list_str)-2],az,el) # remove extra comma
            else:
                print '  ' + str(key) +' : ' + ant_list_str[0:len(ant_list_str)-2] + ']' # remove extra trailing comma

    except Exception, e:
        print col(error_colour) + '\nERROR: could not retrieve status info... ' + col(normal_colour)
        print col(error_colour) + '(' + str(e) + ')' + col(normal_colour)


def print_checks_header(quiet=False):
    if quiet:
        print '\nTimestamp - %s : (%s,%s,%s)' % ('Sensor (red => problem, green => clear)','Value & sensor status',\
        'Min Expected','Max Expected')
    else:
        print '\n%s %s %s %s' % ('Sensor (red => potential problem)'.ljust(65), 'Value (sensor status)'.ljust(25),\
          'Min Expected'.ljust(25), 'Max Expected'.ljust(25))

def print_check_result(potential_problems):
    if potential_problems:
        print '\n' + col(error_colour) + 'Potential problems found' + col(normal_colour)
    else:
        print '\n' + col(ok_colour) + 'All seems well :)' + col(normal_colour)


def print_msg(quiet, header_only, refresh, duration, end_time):

    if duration >= 1.0:
        time_left = end_time - time.time()

    # Cater for the following possibilities:
    if not quiet and header_only and refresh is None: # - no alarm: status header only, no refresh
        print "Single pass status only, no sensor checks"
    elif not quiet and header_only and refresh >= 1.0 and duration is None: # - no alarm: status header only, refresh, no max duration
        print "Status only, updated every %s secs - cntrl-c to exit (r to refresh)" % (refresh)
    elif not quiet and header_only and refresh >= 1.0 and duration >= 1.0: # - no alarm: status header only, refresh, max duration
        print "Status only, updated every % secs, ending in %.3f hours - cntrl-c to exit (r to refresh)" % (refresh, time_left/3600.0)
    elif not quiet and refresh is None: # - no alarm: no refresh (=> one-shot, no max_duration allowed)
        print "Single pass health and status check"
    elif not quiet and refresh >= 1.0 and duration is None: # - no alarm: refresh, no max duration
        print "Health and status check every %s secs - ctrl-c to exit (r to refresh)" %(refresh)
    elif not quiet and refresh >= 1.0 and duration >= 1.0: # - no alarm: refresh, max duration
        print "Health and status check every %s secs, ending in %.3f hours - cntrl-c to exit (r to refresh)" %(refresh,time_left/3600.0)
    elif quiet and refresh is None and duration is None: # - alarm: no refresh, no max duration
        print "\nQuiet mode (%s sec sensor checks) - ctrl-c to exit (r to refresh)" %(quiet_check_refresh)
    elif quiet and refresh is None and duration >= 1.0: # - alarm: no refresh, max duration
        print "\nQuiet mode (%s sec sensor checks) - ending in %.3f hours - ctrl-c to exit (r to refresh)" \
         %(quiet_check_refresh,time_left/3600.0)
    elif quiet and refresh >= 1.0 and duration is None: # - alarm: refresh, no max duration
        print  "\nQuiet mode (%s sec sensor checks) - full refresh every %s secs - cntrl-c to exit (r to refresh)" \
         %(quiet_check_refresh,refresh)
    elif quiet and refresh >= 1.0 and duration >= 1.0: # - alarm: refresh, max duration
        print  "\nQuiet mode (%s sec sensor checks) - full refresh every %s secs, ending in %.3f hours - cntl-c to exit (r to refresh)" \
        %(quiet_check_refresh,refresh, time_left/3600.0)
    else:
        print "Unknown/unsupported combination of options in print_msg()"
        sys.exit()

USAGE="""
Usage examples:
    - basic_health_check.py (show status and errors only in a single pass, then quit)
    - basic_health_check.py -q (quiet mode: show status at start and errors when they first occur or are cleared, continue indefinitely)
    - basic_health_check.py -q -w (quiet mode with warnings: run indefinitely in quiet mode and also show warnings)
    - basic_health_check.py -q -r 600 (run in quiet mode with full refesh every 600 secs)
    - basic_health_check.py -q -m 1000 (run in quiet mode for 1000 secs, then quit)
    - basic_health_check.py -q -r 200 -m 1000 (run in quiet mode for 1000 secs, with full refresh every 200 secs, then quit)
    - basic_health_check.py -v (verbose: show status and all sensors checked in a single pass - not only errors)
    - basic_health_check.py -v -r 20 (refresh verbose listing every 20 secs)
    - basic_health_check.py -v -r 20 -m 100 (refresh verbose listing every 20 secs for 100 secs, then quit)
    - basic_health_check.py -b (show only the status info - no sensor checks, then quit)
    - basic_health_check.py -b -r 10 (show only the status info and refresh every 10 secs, continuing indefinitely)
    - basic_health_check.py -r 10 (show status and errors every 10 secs, continuing indefinitely)
    - basic_health_check.py -r 10 -m 50 (show status and errors every 10 secs for 50 secs, then quit)
    - basic_health_check.py -g 'ants' (single pass check showing only the 'ants' sensor group)
    - basic_health_check.py -a 'ant1,ant7' (limit status and checks to antennas 1 and 7)
    - basic_health_check.py -a 'ant1,ant7' -g ant1 (limit status to antennas 1 and 7 and checks to ant1 only)
    - basic_health_check.py -a 'ant1,ant7' -g ant4 (limit status to antennas 1 and 7 and checks to ant4 only -> no checks done)
    - basic_health_check.py -a 'ant1,ant7' -g anc -v (limit status to antennas 1 and 7 and verbose checks to 'anc' sensor group)
    - basic_health_check.py -a 'ant2,ant4,ant5' -q -r 600 -m 8000 (quiet mode status and errors for ant2,4,5 with 600 sec refresh, quit in 8000 secs)
    - etc. (script can handle most combinations of switches)
"""


if __name__ == '__main__':

    parser = OptionParser(usage='%prog [options]',
                          description='Perform basic status (blue = busy) and health check of the system for observers. ' +
                          'Can be run at any time without affecting current settings/observation.')
    sensor_group_dict_keys = sensor_group_dict.keys()
    sensor_group_dict_keys.sort(key=str.lower) # for some reason python does not like to do this in one line
    parser.add_option('-g', '--sensor_group', default='karoo',
                      help='Selected sensor group to use. Options are: ' + '|'.join(sensor_group_dict_keys) + '. (default="%default")')
    parser.add_option('-a', '--ants', default='all',
                      help="Comma-separated list of antennas to include (e.g. 'ant1,ant2'), "+
                      "or 'all' for all antennas (default='%default')")
    parser.add_option('-v', '--verbose', action='store_true', default=False,
                    help='Verbose. Show all sensors that are checked. Default is to only show values in error. (default="%default")')
    parser.add_option('-b', '--header_only', action='store_true', default=False,
                    help='Show only status header (busy) info. Skip error checks. (default="%default")')
    parser.add_option('-r', '--refresh', type='float',
                      help='Refresh display of header and sensors every specified secs where min non-zero value is 1 sec. '+
                      '(default="%default" -> no refresh)')
    parser.add_option('-q', '--quiet', action='store_true', default=False,
                  help='Quiet mode. Only update sensor check output when new error occurs or error clears. Error if selected with ' +
                  '-v and -b options. Check sensors at 5 secs intervals under-the-hood and also prints a time every 30 mins ' +
                  'to indicate that it is still alive. (default="%default")')
    parser.add_option('-w', '--warn', action='store_true', default=False,
                help='Include warnings in quiet mode. (default="%default")')
    parser.add_option('-m', '--max_duration', type='float',
                    help='Quit programme after specified secs. Works with the -r and/or -q options e.g. set to length of ' +
                    'observation run. Default is to continue indefinitely if refresh is specified or in quiet mode. Error if used ' +
                    'outside of refresh or quiet modes. (default="%default")')
    parser.add_option('-u', '--show_usage_examples', action='store_true', default=False,
                    help='Show usage examples and exit. (default="%default")')

    (opts, args) = parser.parse_args()

    if opts.show_usage_examples:
        print USAGE
        sys.exit()

    # some option checks
    if opts.refresh and opts.refresh < 1.0:
        print "Error: Min refresh is 1 sec. Exiting."
        sys.exit()
    if opts.max_duration and opts.max_duration < 1.0:
        print "Error: Max duration must be >= 1 sec"
        sys.exit()
    if opts.max_duration >= 1.0 and opts.refresh is None and not opts.quiet:
        print "Error: Max duration only applies to refresh and quiet modes"
        sys.exit()
    if opts.max_duration >= 1.0 and opts.refresh > opts.max_duration:
        print "Error: Refresh time must be < max duration"
        sys.exit()
    if opts.quiet and opts.verbose:
        print "Error: Cannot select verbose mode and quiet mode together - see help"
        sys.exit()
    if opts.quiet and opts.header_only:
        print "Error: Cannot select header only and quiet mode together"
        sys.exit()
    if opts.quiet and opts.refresh and opts.refresh < quiet_check_refresh:
        # does not make sense to have full refresh (incl header) at shorter period than quiet refresh in quiet mode
        print "Error: Min refresh in quiet mode is %s secs" %(quiet_check_refresh)
        sys.exit()
    if opts.warn and not opts.quiet:
        print "Error: Cannot select warn option when quiet mode not selected. Warns are shown by default with verbose mode."
        sys.exit()

    # Try to build the KAT configuration
    # This connects to all the proxies and devices and queries their commands and sensors
    site, system = katcorelib.conf.get_system_configuration()
    try:
        kat = katcorelib.tbuild(system=system)
    except ValueError:
        raise ValueError("Could not build KAT connection for %s" % (system,))
    print 'Using KAT connection with configuration: %s' % (kat.system,)

    # construct the per antenna sensor groups (restricting to those that were selected)
    generate_sensor_groups(kat,opts.ants,sensor_group_dict)

    try:
        selected_sensors = sensor_group_dict[opts.sensor_group]
    except KeyError:
        print 'Unknown sensor group "%s", expected one of %s' % (opts.sensor_group, sensor_group_dict_keys)
        sys.exit()

    activity_logger.info("basic_health_check.py: start")
    user_logger.info("basic_health_check.py: start")
    # end_time is the end time for the whole programme
    if opts.max_duration >= 1.0:
        end_time = time.time() + float(opts.max_duration)
    else:
        end_time = time.time() + 10*365*24*3600.0 # set to some far future time (10 years i.e. don't end)

    try:
        ended = False
        while (not ended):
            refresh_cycle_start =  time.time()
            user_logger.info("basic_health_check.py: refresh @ %s " % time.ctime(refresh_cycle_start))
            activity_logger.info("basic_health_check.py: refresh @ %s " % time.ctime(refresh_cycle_start))
            refresh_forced = False # keyboard input forced refresh
            print '\nCurrent local time: %s' % (time.ctime(refresh_cycle_start))
            alarms = {} # reset any alarms when full refresh
            show_status_header(kat,opts,selected_sensors)
            if not opts.header_only:
                if not opts.quiet:
                    print_checks_header()
                    potential_problems = check_sensors(kat,opts,selected_sensors)
                    print_check_result(potential_problems)
                    print_msg(opts.quiet,opts.header_only,opts.refresh,opts.max_duration,end_time)
                else:
                    print_msg(opts.quiet,opts.header_only,opts.refresh,opts.max_duration,end_time) # message before sensor check loop
                    print_checks_header(opts.quiet)
                    quiet_cycles_ended = False
                    quiet_cycles = 0
                    quiet_cycles_start = time.time()
                    while not quiet_cycles_ended and not ended:
                        this_cycle_start = time.time()
                        if (quiet_cycles % 360) == 0: # 360 -> 30 mins for 5 sec quiet_cycle_refresh
                            print 'In quiet loop: Current local time: ' + time.ctime()
                        check_sensors(kat,opts,selected_sensors,opts.quiet,alarms)
                        time_now = time.time()
                        if opts.refresh > 1.0:
                            if time_now > quiet_cycles_start + opts.refresh: quiet_cycles_ended = True
                        if getKeyIf(6) == 'r': # forced refresh
                            quiet_cycles_ended = True
                            refresh_forced  = True
                        if opts.max_duration > 1.0:
                            if time_now > end_time: ended = True
                        if not ended and not quiet_cycles_ended:
                            time.sleep(max(0.0,this_cycle_start + quiet_check_refresh - time_now))
                            quiet_cycles = quiet_cycles + 1
            else:
                print_msg(opts.quiet,opts.header_only,opts.refresh,opts.max_duration,end_time)

            if (time.time() >= end_time) or (not opts.quiet and opts.refresh is None):
                ended = True

            if not ended and not refresh_forced and (refresh_cycle_start + opts.refresh - time.time() > 0):
                in_refresh_sleep = True
                while in_refresh_sleep:
                    time_now = time.time()
                    if time_now < end_time and getKeyIf(6) != 'r' and (refresh_cycle_start + opts.refresh - time_now > 0):
                        time.sleep(0.5)
                    else:
                        in_refresh_sleep = False

            if time.time() >= end_time: # do another check since the refresh sleep loop takes up time
                ended = True
    except KeyboardInterrupt:
        print '\nKeyboard Interrupt detected... exiting gracefully :)'
        user_logger.info("basic_health_check.py: KeyboardInterrupt")
  
    user_logger.info("basic_health_check.py: stop")
    activity_logger.info("basic_health_check.py: stop")

