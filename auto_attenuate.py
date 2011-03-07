#! /usr/bin/python
#
# Set attenuators on RFE stages 5 and 7 to ensure the optimal input power to the
# ADCs while tracking a radio source of interest.
#
# Ludwig Schwardt
# 16 February 2011
#

from __future__ import with_statement

import time
import optparse
import numpy as np

from katuilib.observe import verify_and_connect, ant_array, lookup_targets, user_logger
from katuilib import colors

###################### RFE Stage 5 getters and setters ########################

# RFE5 minimum attenuation, maximum attenuation, finest resolution to which attenuation can be set, all in dB
rfe5_min_att, rfe5_max_att, rfe5_att_step = 0.0, 63.5, 0.5
rfe5_power_range = 2.

def get_rfe5_input_power(kat, ant_num, pol):
    pedestal = getattr(kat, 'ped%d' % (ant_num,))
    sensor = getattr(pedestal.sensor, 'rfe5_%s_power_in' % ('horizontal' if pol == 'h' else 'vertical',))
    return sensor.get_value()

def get_rfe5_attenuation(kat, ant_num, pol):
    pedestal = getattr(kat, 'ped%d' % (ant_num,))
    sensor = getattr(pedestal.sensor, 'rfe5_attenuator_%s' % ('horizontal' if pol == 'h' else 'vertical',))
    return sensor.get_value()

def set_rfe5_attenuation(kat, ant_num, pol, value):
    pedestal = getattr(kat, 'ped%d' % (ant_num,))
    pedestal.req.rfe5_attenuation(pol, value)

def get_rfe5_output_power(kat, ant_num, pol):
    pedestal = getattr(kat, 'ped%d' % (ant_num,))
    sensor = getattr(pedestal.sensor, 'rfe5_%s_power_out' % ('horizontal' if pol == 'h' else 'vertical',))
    return sensor.get_value()

################### RFE Stage 7 / DBE getters and setters #####################

# RFE7 minimum attenuation, maximum attenuation, finest resolution to which attenuation can be set, all in dB
rfe7_min_att, rfe7_max_att, rfe7_att_step = 0.0, 31.5, 0.5
snapshot_length = 8192
connected_antpols = {}

def get_rfe7_attenuation(kat, ant_num, pol):
    sensor = getattr(kat.rfe7.sensor, 'rfe7_downconverter_ant%d_%s_attenuation' % (ant_num, pol))
    return sensor.get_value()

def set_rfe7_attenuation(kat, ant_num, pol, value):
    kat.rfe7.req.rfe7_downconverter_attenuation(str(ant_num), pol, value)

def get_dbe_input_power(kat, ant_num, pol):
    dbe_input = connected_antpols['ant%d, %s' % (ant_num, pol)]
    PinFS_dBm, nbits = 0, 8 # 0 dBm for noise into iADC (J. Manley), may be different for KATADC
    fullscale = 2 ** (nbits - 1)
    gainADC_dB = 20 * np.log10(fullscale) - PinFS_dBm # Scale factor from dBm to numbers
    voltage_samples = kat.dh.get_snapshot('adc', dbe_input)
    return 10 * np.log10(np.var(voltage_samples)) - gainADC_dB

####################### Generic iterative adjustment ##########################

def adjust(kat, ant_num, pol, get_att, set_att, get_power, desired_power, min_att, max_att, att_step, wait=0.):
    """Iteratively adjust attenuator to move power towards desired value."""
    # This should converge in a few iterations, else stop anyway
    for n in range(5):
        att = get_att(kat, ant_num, pol)
        power = get_power(kat, ant_num, pol)
        # The difference between actual and desired power is roughly the extra attenuation needed
        new_att = att + power - desired_power
        # Round desired attenuation to the nearest allowed one
        new_att = np.round((new_att - min_att) / att_step) * att_step + min_att
        # Force attenuation to stay within allowed range
        new_att = np.clip(new_att, min_att, max_att)
        # Stop once attenuator value has settled
        if new_att == att:
            break
        set_att(kat, ant_num, pol, new_att)
        # Optionally wait until power has stabilised before making the next measurement
        time.sleep(wait)
        # Obtain latest measurements on the last iteration
        if n == 4:
            att = get_att(kat, ant_num, pol)
            power = get_power(kat, ant_num, pol)
    return att, power

############################### Main script ###################################

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts]",
                               description="This automatically adjusts RFE5 and RFE7 attenuator settings "
                                           "to achieve optimal power levels. Some options are **required**.")
parser.add_option('-s', '--system', help='System configuration file to use, relative to conf directory ' +
                  '(default reuses existing connection, or falls back to systems/local.conf)')
parser.add_option('-a', '--ants', help="Comma-separated list of antennas to include " +
                  "(e.g. 'ant1,ant2'), or 'all' for all antennas (**required** - safety reasons)")
parser.add_option('-f', '--centre-freq', type='float', help='Centre frequency, in MHz (ignored by default)')
parser.add_option('-t', '--target', help='Radio source on which to calibrate the attenuators (ignored by default)')
parser.add_option('--rfe5-desired', type='float', dest='rfe5_desired_power', default=-47.0,
                  help='Desired RFE5 output power, in dBm (default=%default)')
parser.add_option('-d', '--dbe-desired', type='float', dest='dbe_desired_power', default=-27.0,
                  help='Desired DBE input power, in dBm (default=%default)')
# Parse the command line
opts, args = parser.parse_args()
# The standard option verification routine below requires this option (otherwise ignored)
opts.observer = 'Otto Attenuator'

with verify_and_connect(opts) as kat:

    # Populate lookup table that maps ant+pol to DBE input for FF correlator
    for dbe_input in ['0x', '0y', '1x', '1y']:
        ant_pol = getattr(kat.dbe.sensor, 'input_mappings_%s' % (dbe_input,)).get_value()
        connected_antpols[ant_pol] = dbe_input

    # Create device array of antennas, based on specification string
    ants = ant_array(kat, opts.ants)

    # If centre frequency is specified, set it accordingly
    if opts.centre_freq:
        kat.rfe7.req.rfe7_lo1_frequency(4200.0 + opts.centre_freq, 'MHz')

    # If a calibration source is provided and known to the system, move all antennas onto it and wait for lock
    if opts.target:
        targets = lookup_targets(kat, [opts.target])
        if len(targets) > 0:
            ants.req.target(targets[0])
            ants.req.mode('POINT')
            ants.req.sensor_sampling('lock', 'event')
            ants.wait('lock', True, 300)

    user_logger.info('Input: --dBm->| RFE5 |--dBm->| RFE7 |--dBm->| DBE |')
    user_logger.info('Desired:      | RFE5 | %-4.1f | RFE7 | %-4.1f | DBE |' %
                     (opts.rfe5_desired_power, opts.dbe_desired_power))
    for ant in ants.devs:
        # This assumes that antenna names have the format 'antx', where x is an integer (the antenna number)
        ant_num = int(ant.name.strip()[3:])
        for pol in ('h', 'v'):
            if 'ant%d, %s' % (ant_num, pol) not in connected_antpols:
                user_logger.info('ant%d %s: not connected to FF DBE' % (ant_num, pol.upper()))
                continue
            rfe5_in = get_rfe5_input_power(kat, ant_num, pol)
            # Adjust RFE stage 5 attenuation to give desired output power (short waits required to stabilise power)
            rfe5_att, rfe5_out = adjust(kat, ant_num, pol, get_rfe5_attenuation, set_rfe5_attenuation,
                                        get_rfe5_output_power, opts.rfe5_desired_power,
                                        rfe5_min_att, rfe5_max_att, rfe5_att_step, 0.1)
            # Adjust RFE stage 7 attenuation to give desired DBE input power (no waits required, as DBE lookup is slow)
            rfe7_att, dbe_in = adjust(kat, ant_num, pol, get_rfe7_attenuation, set_rfe7_attenuation,
                                      get_dbe_input_power, opts.dbe_desired_power,
                                      rfe7_min_att, rfe7_max_att, rfe7_att_step)
            # If RFE7 hits minimum attenuation, go back to RFE5 to try and reach desired DBE input power
            if rfe7_att == rfe7_min_att:
                rfe5_att, dbe_in = adjust(kat, ant_num, pol, get_rfe5_attenuation, set_rfe5_attenuation,
                                          get_dbe_input_power, opts.dbe_desired_power,
                                          rfe5_min_att, rfe5_max_att, rfe5_att_step)
                rfe5_out = get_rfe5_output_power(kat, ant_num, pol)
            # Check whether final power levels are within expected bounds
            rfe5_success = np.abs(rfe5_out - opts.rfe5_desired_power) <= rfe5_power_range / 2
            # 95% confidence interval of DBE power, assuming large snapshot length and normally distributed samples
            dbe_conf_interval = 2 * 10. * np.log10(1. + 2. * np.sqrt(2. / snapshot_length))
            dbe_success = np.abs(dbe_in - opts.dbe_desired_power) <= (rfe7_att_step + dbe_conf_interval) / 2
            user_logger.info('ant%d %s: %-4.1f | %4.1f | %s%-4.1f%s | %4.1f | %s%-4.1f%s' %
                             (ant_num, pol.upper(), rfe5_in, rfe5_att,
                              colors.Green if rfe5_success else colors.Red, rfe5_out, colors.Normal, rfe7_att,
                              colors.Green if dbe_success else colors.Red, dbe_in, colors.Normal))
            if rfe5_att == rfe5_min_att and not dbe_success:
                user_logger.warning('RFE5 attenuation of less than %g dB required on ant%d %s - no input signal?'
                                    % (rfe5_min_att, ant_num, pol.upper()))
