#! /usr/bin/python
#

from __future__ import with_statement

import time
import sys
import optparse
import numpy as np

from katcorelib.observe import standard_script_options, verify_and_connect, ant_array, collect_targets, user_logger
from katcorelib import colors
import katpoint

wait_secs = 2.5 # time to wait in secs to allow power levels to settle after changing attenuators

def ant_pedestal(kat, ant_name):
    """Pedestal device associated with antenna device."""
    return getattr(kat, ant_name)

###################### RFE Stage 5 getters and setters ########################

# RFE5 minimum attenuation, maximum attenuation, finest resolution to which attenuation can be set, all in dB
rfe5_min_att, rfe5_max_att, rfe5_att_step = 0.0, 63.5, 0.5
rfe5_power_range = 2.
rfe5_out_max_meas_power = -40 # dBm - power sensor cannot measure larger signals than this


def get_rfe5_attenuation(kat, inputs):
    sensor_val = []
    for ant_name,pol in inputs:
        ped = ant_pedestal(kat, ant_name)
        sensor = getattr(ped.sensor, 'rfe5_attenuator_%s' % ('horizontal' if pol == 'H' else 'vertical',))
        sensor_val.append(sensor.get_value())
    return np.array(sensor_val)

def set_rfe5_attenuation(kat, inputs, inputs_attenuation):
    sensor_val = []
    for (ant_name,pol),value in zip(inputs,inputs_attenuation):
        ped = ant_pedestal(kat, ant_name)
        ped.req.rfe5_attenuation(pol.lower(), value)


###################### RFE Stage 7 getters and setters ########################

# RFE7 minimum attenuation, maximum attenuation, finest resolution to which attenuation can be set, all in dB
rfe7_min_att, rfe7_max_att, rfe7_att_step = 0.0, 31.5, 0.5

def get_rfe7_attenuation(kat, inputs):
    sensor_val = []
    for ant_name,pol in inputs:
        sensor = getattr(kat.rfe7.sensor, 'rfe7_downconverter_%s_%s_attenuation' % (ant_name, pol.lower()))
        sensor_val.append(sensor.get_value())
    return np.array(sensor_val)

def set_rfe7_attenuation(kat, inputs, inputs_attenuation):
    sensor_val = []
    for (ant_name,pol),value in zip(inputs,inputs_attenuation):
        # This assumes that antenna names have the format 'antx', where x is an integer (the antenna number)
        ant_num = int(ant_name.strip()[3:])
        kat.rfe7.req.rfe7_downconverter_attenuation(str(ant_num), pol, value)

################################ DBE getters ##################################

connected_antpols = {}



############################### Main script ###################################

# Parse command-line opts and arguments
parser = standard_script_options(usage="%prog [opts]",
                               description="This automatically max's out  RFE5 and RFE7 attenuator settings "
                                           "to achieve safe levels for work on the antenna. Some options are **required**.")

# Parse the command line
opts, args = parser.parse_args()
# The standard option verification routine below requires this option (otherwise ignored)
opts.observer = 'Max Attenuator'

with verify_and_connect(opts) as kat:

    # Create device array of antennas, based on specification string
    ants = kat.ants #ant_array(kat, opts.ants)
    user_logger.info('Using antennas: %s' % (' '.join([ant.name for ant in ants]),))

    inputs = []
    for ant in ants:
        for pol in ('h', 'v'):
            inputs.append([ant.name, pol.upper()])


    # Get stage 5 attenuation Set it to max and check it
    rfe5_att = get_rfe5_attenuation(kat, inputs)
    rfe5_att = np.clip(rfe5_att, rfe5_max_att, rfe5_max_att)
    set_rfe5_attenuation(kat, inputs, rfe5_att)
    #rfe5_att = get_rfe5_attenuation(kat, inputs)

    # Get stage 7 attenuation Set it to max and check it
    rfe7_att = get_rfe7_attenuation(kat, inputs)
    rfe7_att = np.clip(rfe7_att, rfe7_max_att, rfe7_max_att) # keep array structure
    set_rfe7_attenuation(kat,inputs,rfe7_att)
    #rfe7_att = get_rfe7_attenuation(kat, inputs)
    time.sleep(wait_secs)
    rfe5_att = get_rfe5_attenuation(kat, inputs)
    set_rfe7_attenuation(kat,inputs,rfe7_att)
    user_logger.info('Attenuation Levels: rfe5 Max | rfe5 Set | rfe7 Max | rfe7 Set' )
    for key,data in enumerate(inputs):
        user_logger.info('%s %s            : %-4.1f     | %s%-4.1f%s     | %4.1f     | %s%4.1f%s' %
                     (data[0],data[1],rfe5_max_att,
                      colors.Green if rfe5_max_att == rfe5_att[key] else colors.Red, rfe5_att[key], colors.Normal, rfe7_max_att,
                      colors.Green if rfe7_max_att == rfe7_att[key] else colors.Red, rfe7_att[key], colors.Normal))

    for key,data in enumerate(inputs):
        if not rfe5_max_att == rfe5_att[key]:
            user_logger.error('%s Failed to set the max attenuation on stage 5 %s %s %s' % (colors.Red,data[0],data[1],colors.Normal))
        if not rfe7_max_att == rfe7_att[key]:
            user_logger.error('%s Failed to set the max attenuation on stage 7 %s %s %s' % (colors.Red,data[0],data[1],colors.Normal))





