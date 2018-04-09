#!/usr/bin/env python
# Track target(s) for a specified time.

import time

import numpy
from katcorelib import (
    user_logger, standard_script_options, verify_and_connect, colors)


def get_update(sensors):
    # This samples the senors and obtains a new value
    # this combats the https://xkcd.com/221/ situation.
    for sensor in sensors:
        value = kat.sensor.get(sensor).value  # value updated by cam
        if value is None or value == []:
            value = kat.sensor.get(sensor).get_value()  # Kick the system
        if len(sensors[sensor]) == 0 or sensors[sensor][-1] != value:
            sensors[sensor].append(value)


def get_last_value(sensors):
    # This samples the sensors and appends a new value
    last_value = {}
    for sensor in sensors:
        if len(sensors[sensor]) == 0:
            last_value[sensor] = None
        else:
            last_value[sensor] = sensors[sensor][-1]
    return last_value


def get_mean_value(sensors):
    # This samples the senors and obtains a new value
    last_value = {}
    for sensor in sensors:
        if len(sensors[sensor]) == 0:
            last_value[sensor] = None
        else:
            last_value[sensor] = numpy.mean(sensors[sensor])
    return last_value


def color_code(value, warn, error):
    "Return color code based warn,error levels"
    code_color = colors.Green
    if value <= warn:
        code_color = colors.Yellow
    if value <= error:
        code_color = colors.Red
    return code_color


def point(ants, target, timeout=300):
    # send this target to the antenna.
    ants.req.target(target)
    ants.req.mode("POINT")
    user_logger.info("Slewing to target : %s" % (target,))
    # wait for antennas to lock onto target
    success = ants.wait("lock", True, timeout)
    if success:
        user_logger.info("Tracking Target : %s " % (target,))
    else:
        failed = [client for client in success if not success[client]]
        msg = "Waiting for sensor 'lock' == True "
        # Report failure to user (including list of clients that failed)
        msg += "reached timeout of %.1f seconds. " % (timeout,)
        msg += "Clients %r failed." % (sorted(failed),)
        user_logger.error(msg)
    return success


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> "
description = 'Calculate the predicted attenuation needed and Set the attenuation to appropriate levels'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option('--rfcu-in', type='float', default=-40.0,
                  help='The target power level for the rfcu (default=%default)')
parser.add_option('--adc-in', type='float', default=-30.0,
                  help='The target power level for the adc (default=%default)')
parser.add_option('-t', '--track-duration', type='float', default=600.0,
                  help='Length of time to track the source, in seconds (default=%default)')
parser.add_option('-b', '--band', default='l',
                  help='The band of the receiver  (default=%default)')
parser.add_option('--change-attenuation', action="store_true", default=False,
                  help='Change the attenuation to the predicted levels. ')


# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Auto Attenuate', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

# adc  : WARNING at -31dBm and ERROR at -37dBm
# rfcu : WARNING at -41dBm and ERROR at -47dBm
# Check options and build KAT configuration, connecting to proxies and devices
band = opts.band
track_duration = opts.track_duration
rfcu_in, adc_in = opts.rfcu_in, opts.adc_in
change_attenuation = opts.change_attenuation
with verify_and_connect(opts) as kat:
    rfcu_power = {}
    adc_power = {}
    attenuation = {}
    user_logger.info('Input: --dBm->| RFCU |--dBm->| ADC  |--dBm->|')
    user_logger.info('Desired:      | RFCU | %-4.1f | ADC  | %-4.1f |' %
                     (rfcu_in, adc_in))
    lookup = {}
    new_atten = {}
    if not kat.dry_run:
        for pol in {'h', 'v'}:
            kat.ants.set_sampling_strategy("dig_%s_band_rfcu_%spol_rf_power_in" %
                                           (band, pol), "period 1.0")
            kat.ants.set_sampling_strategy("dig_%s_band_adc_%spol_rf_power_in" %
                                           (band, pol), "period 1.0")
            kat.ants.set_sampling_strategy("dig_%s_band_adc_%spol_attenuation" %
                                           (band, pol), "period 1.0")
            kat.ants.req.sensor_sampling("lock", "event")
            for ant in kat.ants:
                sensor_list = []
                rfcu_power['%s_dig_%s_band_rfcu_%spol_rf_power_in' %
                           (ant.name, band, pol)] = []
                sensor_list.append(
                    '%s_dig_%s_band_rfcu_%spol_rf_power_in' % (ant.name, band, pol))
                adc_power['%s_dig_%s_band_adc_%spol_rf_power_in' %
                          (ant.name, band, pol)] = []
                sensor_list.append('%s_dig_%s_band_adc_%spol_rf_power_in' %
                                   (ant.name, band, pol))
                attenuation['%s_dig_%s_band_rfcu_%spol_attenuation' %
                            (ant.name, band, pol)] = []
                sensor_list.append('%s_dig_%s_band_rfcu_%spol_attenuation' %
                                   (ant.name, band, pol))
                lookup["%s,%s" % (ant.name, pol)] = sensor_list
                kat.sensor.get("%s_lock" % (ant.name)).set_strategy('event')
        point(kat.ants, "SCP,radec,0,-90", timeout=300)
        start_time = time.time()
        while time.time()-start_time < track_duration:
            get_update(rfcu_power)
            get_update(adc_power)
            time.sleep(10)
        get_update(attenuation)
        rfcu_power_v = get_mean_value(rfcu_power)
        adc_power_v = get_mean_value(adc_power)
        attenuation_v = get_last_value(attenuation)
        for ant_pol in sorted(lookup.keys()):
            ant, pol = ant_pol.split(',')
            rfcu_st = rfcu_power_v[lookup[ant_pol][0]]
            adc_st = adc_power_v[lookup[ant_pol][1]]
            atten = attenuation_v[lookup[ant_pol][2]]
            # rfcu : WARNING at -41dBm and ERROR at -47dBm
            rfcu_color = color_code(rfcu_st, -41, -47)
            # adc  : WARNING at -31dBm and ERROR at -37dBm
            adc_color = color_code(adc_st, -31, -37)
            attenuation_change = numpy.floor(
                numpy.min([(adc_st-adc_in) + atten, (rfcu_st-rfcu_in) + atten])).astype(int)
            atten_color = color_code(attenuation_change, 0, 0)
            new_atten[lookup[ant_pol][2]] = attenuation_change
            user_logger.info("%s %s: Start input power | atten | output power = %s%-4.1f %s| %-4.1f | %s%-4.1f %s    Attenuation needed %s %i %s" %
                             (ant, pol, rfcu_color, rfcu_st, colors.Normal, atten, adc_color, adc_st, colors.Normal, atten_color, attenuation_change, colors.Normal))
        if change_attenuation:
            for ant_pol in sorted(lookup.keys()):
                ant, pol = ant_pol.split(',')
                key = lookup[ant_pol][2]
                if new_atten[key] < 0:
                    user_logger.error(
                        "%s %s: input power detected is too low to correct, setting to 0dB attenuation" % (ant, pol))
                    new_atten[key] = 0
                if new_atten[key] >= 0 and attenuation_v[key] != new_atten[key]:
                    user_logger.info("%s %s: Changing attenuation from %idB to %idB " % (
                        ant, pol, attenuation_v[key], new_atten[key]))
                    kat.get(ant).req.get("dig_attenuation")(pol, new_atten[key])
                else:
                    user_logger.warning("%s %s: Will not try change attenuation from %idB to %idB " % (
                        ant, pol, attenuation_v[key], new_atten[key]))
