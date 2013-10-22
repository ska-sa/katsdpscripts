###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
#! /usr/bin/python
#
# Ensure correlator setup with basic default mode and gain settings. Set attenuators on RFE stages 5 and 7
# to ensure the optimal input power to the ADCs while tracking a radio source of interest.
#
# Ludwig Schwardt
# 16 February 2011
##
# 06 June 2011 - (SimonR) Hacked a little
# 30 June 2011 - (JasperH) Added correlator mode check and intial setting of correlator gains.
#                 Removed the dreaded "eval" in the process. Fixed bug re setting RFE5 initial
#                 att. Added some debug print statements. Updated FF power calc using method as
#                 suggested in Jason Manley email 30/06/2011.

from __future__ import with_statement

import time
#import sys
#import optparse
import numpy as np

from katcorelib.observe import standard_script_options, verify_and_connect, collect_targets, user_logger,start_session
from katcorelib import colors
#import katpoint

wait_secs = 0.5 # time to wait in secs to allow power levels to settle after changing attenuators

def ant_pedestal(kat, ant_name):
    """Pedestal device associated with antenna device."""
    return getattr(kat, ant_name)
    #ant_num = int(ant_name.strip()[3:])
    #return getattr(kat, 'ped%d' % (ant_num,))

##################### DBE7 GAIN setter ####################################
# This is due to a CAM bug which on sends 1024 values to the DBE.
def set_k7_gains(kat,value):
    """ This sets digital gains in the dbe7
    this takes two parmeters. The kat object and the value
    of the gain
    Usage:
    set_k7_gains(kat,55)"""
    kat.dbe7.req.dbe_k7_gain('0x',int(value))
    kat.dbe7.req.dbe_k7_gain('0y',int(value))
    for pol in ['h','v']:
        for ant in range(1,8):
            kat.dbe7.req.dbe_k7_gain('ant%i%s'%(ant,pol), int(value))



###################### RFE Stage 5 getters and setters ########################

# RFE5 minimum attenuation, maximum attenuation, finest resolution to which attenuation can be set, all in dB
rfe5_min_att, rfe5_max_att, rfe5_att_step = 0.0, 63.5, 0.5
rfe5_power_range = 2.
rfe5_out_max_meas_power = -40 # dBm - power sensor cannot measure larger signals than this

def get_rfe5_input_power(kat, inputs):
    power = np.zeros((len(inputs),100))
    sensor_list = []
    for key,(ant_name,pol) in enumerate(inputs):
        ped = ant_pedestal(kat, ant_name)
        sensor_list.append(getattr(ped.sensor, 'rfe5_%s_power_in' % ('horizontal' if pol == 'H' else 'vertical',)))
    for t in range(100): # Read sensor 100 times in one second and return median to obtain a more stable value
        for key,sensor in  enumerate(sensor_list):
            power[key,t] = sensor.get_value()
        time.sleep(0.01) # second sample
    return np.median(power,axis=1)

def get_rfe5_attenuation(kat, inputs):
    sensor_val = []
    for ant_name,pol in inputs:
        ped = ant_pedestal(kat, ant_name)
        sensor = getattr(ped.sensor, 'rfe5_attenuator_%s' % ('horizontal' if pol == 'H' else 'vertical',))
        sensor_val.append(sensor.get_value())
    return np.array(sensor_val)

def set_rfe5_attenuation(kat, inputs, inputs_attenuation):
    #sensor_val = []
    for (ant_name,pol),value in zip(inputs,inputs_attenuation):
        ped = ant_pedestal(kat, ant_name)
        ped.req.rfe5_attenuation(pol.lower(), value)

def get_rfe5_output_power(kat, inputs):
    power = np.zeros((len(inputs),100))
    sensor_list = []
    for key,(ant_name,pol) in enumerate(inputs):
        ped = ant_pedestal(kat, ant_name)
        sensor_list.append(getattr(ped.sensor, 'rfe5_%s_power_out' % ('horizontal' if pol == 'H' else 'vertical',)))
    for t in range(100): # Read sensor 100 times in one second and return median to obtain a more stable value
        for key,sensor in  enumerate(sensor_list):
            power[key,t] = sensor.get_value()
        time.sleep(0.01) # second sample
    return np.median(power,axis=1)

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
    #sensor_val = []
    for (ant_name,pol),value in zip(inputs,inputs_attenuation):
        # This assumes that antenna names have the format 'antx', where x is an integer (the antenna number)
        ant_num = int(ant_name.strip()[3:])
        kat.rfe7.req.rfe7_downconverter_attenuation(str(ant_num), pol, value)

################################ DBE getters ##################################

connected_antpols = {}
dbe_power_range = 2. # dBm - Jason reckons we need to get within 1 dBm of the target DBE input power level

def get_dbe_input_power(kat, inputs, dbe):
    sensor_val = []
    for ant_name,pol in inputs:
        if dbe == 'dbe7':
            dbe_input = ant_name + pol.lower()
            dbe_device = getattr(kat, dbe)
            power_sensor = getattr(dbe_device.sensor, "dbe_%s_adc_power" % dbe_input)
            if power_sensor.get_value() is None :
                raise ValueError("The sensor dbe_%s_adc_power on %s is not returning a value." % (dbe_input,dbe))
            sensor_val.append( power_sensor.get_value())
        else:
            raise ValueError("Unknown dbe device (%s) specified. Expecting either 'dbe' or 'dbe7'" % (dbe))
    return np.array(sensor_val)

####################### Generic iterative adjustment ##########################

def adjust(kat, inputs, get_att, set_att, dbe, desired_power, min_att, max_att, att_step, wait=wait_secs):
    """Iteratively adjust attenuator to move power towards desired value."""
    # This should converge in a few iterations, else stop anyway
    for n in range(5):
        att = get_att(kat, inputs)
        power = get_dbe_input_power(kat, inputs, dbe)
        # The difference between actual and desired power is roughly the extra attenuation needed
        new_att = att + power - desired_power
        # Round desired attenuation to the nearest allowed one
        new_att = np.round((new_att - min_att) / att_step) * att_step + min_att
        # Force attenuation to stay within allowed range
        new_att = np.clip(new_att, min_att, max_att)
        # Stop once attenuator value has settled
        #if new_att == att:
        #    break
        set_att(kat, inputs, new_att)
        # Wait until power has stabilised before making the next measurement
        time.sleep(wait)
        # Obtain latest measurements on the last iteration
        if n == 4:
            att = get_att(kat, inputs)
            power = get_dbe_input_power(kat, inputs, dbe)
    return att, power

############################### Main script ###################################

# Set up standard script options
parser = standard_script_options(usage="%prog [opts]",
                               description="This automatically adjusts RFE5 and RFE7 attenuator settings "
                                           "to achieve optimal power levels. Some options are **required**.")

# Add extra command-line opts and arguments
parser.add_option('-t', '--target', default='',
                  help="Radio source on which to calibrate the attenuators (default='%default'). "+
                  "Won't drive antennas if not set.")
parser.add_option('--rfe5-desired', type='float', dest='rfe5_desired_power', default=-47.0,
                  help='Desired RFE5 output power, in dBm (default=%default).')
parser.add_option('--dbe-desired', type='float', dest='dbe_desired_power', default=-26.0,
                  help='Desired DBE input power, in dBm (default=%default). Success will be within 1 dBm of this value.')
#parser.remove_option('-d')
parser.set_defaults(observer='Otto Attenuate')
parser.set_defaults(description='Auto Attenuate data')
parser.set_defaults(nd_params='off')
# Parse the command line
opts, args = parser.parse_args()
opts.description='Auto atten'

with verify_and_connect(opts) as kat:
    try:
        # In the past there was a command line option to select dbe
        # (fringe finder) / dbe7, but since fringe finder has been
        # decommisioned that option has been removed
        selected_dbe = kat.dbe7
    except NameError:
        raise RuntimeError("Unknown dbe device (%s) specified. Typically it should be either 'dbe' or 'dbe7'")
    with start_session(kat, **vars(opts)) as session:
        # If centre frequency is specified, set it accordingly
        user_logger.info('Current centre frequency: %s MHz' % (session.get_centre_freq(),))
        if not kat.dry_run and opts.centre_freq and not session.get_centre_freq() == opts.centre_freq:
            session.set_centre_freq(opts.centre_freq)
            time.sleep(1.0)
            user_logger.info('Updated centre frequency: %s MHz' % (session.get_centre_freq(),))
            if np.abs(session.get_centre_freq() - opts.centre_freq) > 2e-5:  # we have 10 HZ resolution
                user_logger.warning('Failed to updated centre frequency to %s MHz, it is currently set to %s MHz waning is due to the difference between actual & spcified frequency is larger that 10 Hz' % (opts.centre_freq, session.get_centre_freq(),))

        session.standard_setup(**vars(opts))
        session.capture_start()
        if True: # Dry run will now exec this branch, in the past this branch was exculded from the dry checking
            # check that the selected dbe is set to the correct mode
            dbe_mode = kat.dbe7.sensor.dbe_mode.get_value()
            dbe7_mode_dict =  {'bc16n400M1k':160,'c16n400M1k':160,'c16n400M8k':160,'wbc':160, 'wbc8k':160,'c16n7M4k':31,'c16n2M4k':59,'c16n25M4k':17,'c16n13M4k':23}
            if dbe_mode in dbe7_mode_dict.keys() :
                user_logger.info("dbe7 mode is '%s', as expected :)" % dbe_mode)
                gain =dbe7_mode_dict[dbe_mode]
                set_k7_gains(kat,gain)
                user_logger.info("Set digital gain on selected DBE to %d." % gain)
            else:
                user_logger.error("dbe7 mode is '%s' and not in the list of valid modes. Could not set appropriate gain." % (dbe_mode))

            # Populate lookup table that maps ant+pol to DBE input
            for dbe_input_sensor in [sensor for sensor in vars(selected_dbe.sensor) if sensor.startswith('input_mappings_')]:
                ant_pol = getattr(selected_dbe.sensor, dbe_input_sensor).get_value()
                connected_antpols[ant_pol] = dbe_input_sensor[15:]

            # Create device array of antennas, based on specification string
            ants = kat.ants
            user_logger.info('Using antennas: %s' % (' '.join([ant.name for ant in ants]),))

            # Move all antennas onto calibration source and wait for lock
            try:
                targets = collect_targets(kat, [opts.target]).targets
            except ValueError:
                user_logger.info("No valid targets specified. Antenna will not be moved.")
            else:
                session.track(targets[0], duration=1, announce=False)
            # Warn if requesting an RFE5 desired output power larger than max measurable power of the RFE5 output power sensor
            if opts.rfe5_desired_power > rfe5_out_max_meas_power:
                user_logger.warn("Requested RFE5 output power %-4.1f larger than max measurable power of %-4.1f dBm. Could cause problems..." % ( opts.rfe5_desired_power, rfe5_out_max_meas_power))

            user_logger.info('Input: --dBm->| RFE5 |--dBm->| RFE7 |--dBm->| DBE |')
            user_logger.info('Desired:      | RFE5 | %-4.1f | RFE7 | %-4.1f | DBE |' %
                             (opts.rfe5_desired_power, opts.dbe_desired_power))
            inputs = []
            for ant in ants:
                for pol in ('h', 'v'):
                    if '%s, %s' % (ant.name, pol) not in connected_antpols:
                        user_logger.info('%s %s: not connected to DBE' % (ant.name, pol.upper()))
                        continue
                    inputs.append([ant.name, pol.upper()])

            # Adjust RFE stage 5 attenuation to give desired output power
            rfe5_att = get_rfe5_attenuation(kat, inputs)
            rfe5_in = get_rfe5_input_power(kat, inputs)
            rfe5_out = get_rfe5_output_power(kat, inputs)
            for key,data in enumerate(inputs):
                user_logger.info("%s %s: Start RFE5 input power | atten | output power = %-4.1f | %-4.1f | %-4.1f" % (data[0],data[1],rfe5_in[key], rfe5_att[key], rfe5_out[key]))

            # The difference between actual and desired power is roughly the extra attenuation needed
            rfe5_att = rfe5_att + rfe5_out - opts.rfe5_desired_power
            # Round desired attenuation to the nearest allowed one
            rfe5_att = np.round((rfe5_att - rfe5_min_att) / rfe5_att_step) * rfe5_att_step + rfe5_min_att
            # Force attenuation to stay within allowed range
            rfe5_att = np.clip(rfe5_att, rfe5_min_att, rfe5_max_att )

            for key,data in enumerate(inputs):
                user_logger.info("%s %s: Setting updated RFE5 attenuation of %-4.1f dB" %  (data[0],data[1],rfe5_att[key]))
            set_rfe5_attenuation(kat, inputs, rfe5_att)
            time.sleep(wait_secs) # add a small sleep to allow change to propagate

            # Get the newly-set rfe5 attenuation value as a check and new
            # rfe5 input and output power value (input should be approx same)
            rfe5_att = get_rfe5_attenuation(kat, inputs)
            rfe5_in = get_rfe5_input_power(kat, inputs)
            rfe5_out = get_rfe5_output_power(kat, inputs)
            for key,data in enumerate(inputs):
                user_logger.info("%s %s: Updated RFE5 input power | atten | output power = %-4.1f | %-4.1f | %-4.1f" % (data[0],data[1],rfe5_in[key], rfe5_att[key], rfe5_out[key]))

            # Adjust RFE stage 7 attenuation to give desired DBE input power
            rfe7_att, dbe_in = adjust(kat, inputs, get_rfe7_attenuation, set_rfe7_attenuation,
                                      'dbe7', opts.dbe_desired_power,
                                      rfe7_min_att, rfe7_max_att, rfe7_att_step)
            # If RFE7 hits minimum attenuation, go back to RFE5 to try and reach desired DBE input power
            rfe5_att, dbe_in = adjust(kat, inputs, get_rfe5_attenuation, set_rfe5_attenuation,
                                      'dbe7', opts.dbe_desired_power,
                                      rfe5_min_att, rfe5_max_att, rfe5_att_step)
            rfe5_out = get_rfe5_output_power(kat, inputs)
            # Check whether final power levels are within expected bounds
            rfe5_success = np.abs(rfe5_out - opts.rfe5_desired_power) <= rfe5_power_range / 2

            dbe_success = np.abs(dbe_in - opts.dbe_desired_power) <= dbe_power_range / 2.
            for key,data in enumerate(inputs):
                user_logger.info('%s %s: %-4.1f | %4.1f | %s%-4.1f%s | %4.1f | %s%-4.1f%s' %
                             (data[0],data[1], rfe5_in[key], rfe5_att[key],
                              colors.Green if rfe5_success[key] else colors.Red, rfe5_out[key], colors.Normal, rfe7_att[key],
                              colors.Green if dbe_success[key] else colors.Red, dbe_in[key], colors.Normal))
