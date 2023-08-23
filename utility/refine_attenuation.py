#!/usr/bin/env python
# Track SCP and and determine attenuation values

import time
from contextlib import closing
import smtplib
from email.mime.text import MIMEText

import katconf

import numpy as np
from katcorelib import (user_logger, standard_script_options,
                        verify_and_connect, colors)

import StringIO


def send_email(email_to, lines, subject, messagefrom='operators@ska.ac.za'):
    if not isinstance(email_to, list):
        emailto = email_to.replace(';', ',').split(',')
    emailto = ','.join(map(str.strip, emailto))
    msg = MIMEText('\n'.join(lines))
    msg['Subject'] = subject
    msg['From'] = messagefrom
    msg['To'] = emailto
    with closing(smtplib.SMTP('smtp.kat.ac.za')) as smtp_server:
        smtp_server.sendmail(messagefrom, emailto, msg.as_string())


def color_code(value, warn, error):
    "Return color code based warn,error levels"
    code_color = colors.Green
    if value <= warn:
        code_color = colors.Yellow
    if value <= error:
        code_color = colors.Red
    return code_color


def color_code_eq(value, test, errorv=0.01):
    """
        This function returns the color code string bassed on if the values are within a range
        Example:
                $ color_code_eq(1., 2.,errorv=0.01)
            returns yellow color code

                $ color_code_eq(1., 1.0005,errorv=0.01)
            returns green color code

            value, test,errorv are floating point numbers
            value and test are the 2 values tested
            and errorv is the equality range.
    """
    code_color = colors.Green
    if value >= test + errorv or value <= test - errorv:
        code_color = colors.Yellow
    return code_color


def measure_atten(ant, pol, band='l'):
    """ This function returns the attenuation of an antenna.
    Example:
            $ measure_atten('m064', 'h',band='x')
        returns 4
        ant is an katcp antenna object
        pol is a string
        value and test are the antenna name and the polorisation
    """

    sensor = "dig_%s_band_rfcu_%spol_attenuation" % (band, pol)
    atten = ant.sensor[sensor].get_value()
    return atten


def get_ant_band(ant):
    """ This function returns the selected band of an antenna
    Example:
            $ get_ant_band('m064')
        returns 'x'
          ant is an katcp antenna object
    """
    sensor = "dig_selected_band"
    band = ant.sensor[sensor].get_value()
    return band


def point(ants, target, timeout=300):
    # send this target to the antenna.
    ants.req.target(target)
    ants.req.mode("POINT")
    user_logger.info("Slewing to target : %s" % (target,))
    # wait for antennas to lock onto target
    ants.set_sampling_strategy("lock", "period 1.0")
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


def plus_minus(num):
    return (np.mod(np.arange(num), 2) * 2 - 1)


def sample_bits_fake(ant, pol, band):  # fake
    tmp_data = np.zeros((5, 4096, 4))
    data = np.random.normal(loc=1.6, scale=20.89, size=tmp_data.flatten(
    ).shape[0]).astype(int) * plus_minus(tmp_data.flatten().shape[0])
    std = (data * plus_minus(data.shape[0])).std()
    color_d = color_code(std, 12, 8)
    sensor = 'dig_%s_band_rfcu_%spol_attenuation' % (band, pol)
    atten = ant.sensor[sensor].get_value()
    windowed_data = data.reshape(-1, 256)
    # 67:89  corresponds to 1300 to 1450 MHz
    voltage = np.abs(np.fft.fft(windowed_data)[:, 67:89]).mean()
    # channels 67:89  correspond to a RFI free section of band (1300 to 1450
    # MHz).
    string = "fake values %s ADC rms %s: %s%-4.1f %s  vlevel: %-4.1f  Attenuation : %-2i  " % (
        ant.name, pol, color_d, std, colors.Normal, voltage, atten)
    user_logger.info(string)
    return std, atten, voltage


def sample_bits(ant, pol, band):
    tmp_data = np.zeros((5, 4096, 4))
    for i in range(5):
        snap = ant.req.dig_adc_snap_shot(pol)
        tmp_data[i, :, :] = np.array(
            [snip.arguments[1:] for snip in snap.messages[1:]]).astype('float')
    data = tmp_data.flatten()
    std = (data * plus_minus(data.shape[0])).std()
    color_d = color_code(std, 12, 8)
    sensor = 'dig_%s_band_rfcu_%spol_attenuation' % (band, pol)
    atten = ant.sensor[sensor].get_value()
    windowed_data = data.reshape(-1, 256)
    # 67:89  corresponds to 1300 to 1450 MHz
    voltage = np.abs(np.fft.fft(windowed_data)[:, 67:89]).mean()
    # channels 67:89  correspond to a RFI free section of band (1300 to 1450
    # MHz).
    string = "%s ADC rms %s: %s%-4.1f %s  vlevel: %-4.1f  Attenuation : %-2i  " % (
        ant.name, pol, color_d, std, colors.Normal, voltage, atten)
    user_logger.info(string)
    return std, atten, voltage


def init_data_dict():
    tmp = {}
    tmp['ant_name'] = []
    tmp['time'] = []
    tmp['attenuation'] = []
    tmp['std'] = []
    tmp['vol'] = []
    tmp['atten_change'] = [0]
    tmp['std_change'] = [0]
    tmp['vol_change'] = [0]
    tmp['dB_change'] = [0]
    tmp['katconfig_values'] = []
    tmp['counts'] = 0
    return tmp


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> "
description = 'Calculate the attenuation needed and Set the attenuation to appropriate levels'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options

parser.add_option('--adc-std-in', type='float', default=12.0,
                  help='The target adc rms level  (default=%default)')
parser.add_option('--adc-volt', type='float', default=190.0,
                  help='The target power level for the adc (default=%default)')
parser.add_option('--adc-volt-range', type='float', default=20.0,
                  help='The power level range for the adc (default=%default)')

parser.add_option(
    '--email-to',
    type='str',
    default='operators@ska.ac.za,meerkat-attenuation@sarao.ac.za',
    help='Comma separated email list of people to send report to (default=%default)')

# Set default value for any option (both standard and experiment-specific
# options)
parser.set_defaults(description='Auto Attenuate', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if not len(args) == 0:
    raise RuntimeError(
        'This script no longer takes in an attenuation file. Please raise an issue if you need this ')


# adc  : WARNING at -31dBm and ERROR at -37dBm
# rfcu : WARNING at -41dBm and ERROR at -47dBm
# Check options and build KAT configuration, connecting to proxies and devices

adc_volt = opts.adc_volt
adc_volt_range = opts.adc_volt_range
adc_std_in = opts.adc_std_in
bandlist = ['l', 'u']  # ,'s','x'   # Read in the bands

# starting emails lines
lines2 = []

dictionary_data = {}
atten_ref = {}


with verify_and_connect(opts) as kat:
    user_logger.info('Number of available antennas : %s%-4.1f %s ' %(colors.Blue,len(kat.ants),colors.Normal))
    if not kat.dry_run:
        # Point the antennas to cold sky
        point(kat.ants, 'SCP,radec,0,-90', timeout=300)
    if not kat.dry_run:
        for ant in kat.ants:  # Main antenna loop
            dictionary_data[str(ant.name) + 'h'] = init_data_dict()
            dictionary_data[str(ant.name) + 'v'] = init_data_dict()
            band = get_ant_band(ant)

            if band == 'u':
                adc_volt = opts.adc_volt * (np.sqrt(2))
                adc_volt_range = opts.adc_volt_range * (np.sqrt(2))
            else:
                adc_volt = opts.adc_volt
                adc_volt_range = opts.adc_volt_range

            for pol in {'h', 'v'}:  # Set up the sampling stratergy
                kat.ants.set_sampling_strategy(
                    "dig_%s_band_adc_%spol_attenuation" %
                    (band, pol), "period 1.0")
            if band not in bandlist:
                user_logger.error(
                    "'%s' band %s is not in the list of valid bands " %
                    (band, ant.name))
            elif band in bandlist:
                user_logger.info(
                    "This script used values found in katconf/katconfig")
                user_logger.info(
                    "Reading file katconf:'katconfig/user/attenuation/mkat/dig_attenuation_%s.csv'" %
                    (band))
                file_string = katconf.resource_string(
                    'katconfig/user/attenuation/mkat/dig_attenuation_%s.csv' % (band))
                tmp_data = np.loadtxt(
                    StringIO.StringIO(file_string),
                    dtype=np.str,
                    delimiter=',')
                # Calculating stored values
                for ant_, value_h, value_v in tmp_data:
                    try:
                        atten_ref['%s%s' % (ant_, 'h')] = np.int(value_h)
                        atten_ref['%s%s' % (ant_, 'v')] = np.int(value_v)
                    except ValueError:
                        user_logger.warning(
                            "'%s' band  %s: attenuation value '%s','%s' is not an integer " %
                            (band, ant_, value_h, value_v))
                user_logger.info(' ')
                user_logger.info('Evaluating attenna : '+str(ant.name))

                # initalize pol_update here
                pol_update = np.ones((2)).astype(bool)
                count = 0
                while pol_update.sum() > 0 and count < 20 and not kat.dry_run:
                    count = count + 1
                    #time.sleep(30) # Only need to sleep on counts >1  also
                    # only need to wait 30seconds from the command to change
                    # attenuation. Reading the power takes ~30 secs
                    user_logger.info("Loop number : %s%-4.1f %s " %(colors.Purple,(count),colors.Normal))

                    for i, pol in enumerate({'h', 'v'}):
                        key = str(ant.name) + pol
                        if pol_update[i]:
                            pol_update[i] = False
                            std, atten, voltage = sample_bits(
                                ant, pol, band=band)
                            dictionary_data[key]['time'].append(
                                time.strftime('%d/%m/%Y %H:%M:%S'))
                            dictionary_data[key]['attenuation'].append(atten)
                            dictionary_data[key]['std'].append(std)
                            dictionary_data[key]['vol'].append(voltage)
                            dictionary_data[key]['ant_name'].append(key)

                            if atten < 32 and (
                                    voltage > adc_volt + adc_volt_range):  # Attenuation was increased
                                user_logger.info(
                                    "'%s' band %s %s: Changing attenuation from %idB to %idB " %
                                    (band, ant.name, pol, atten, atten + 1))
                                ant.req.dig_attenuation(pol, atten + 1)
                                pol_update[i] = True
                                dictionary_data[key]['dB_change'].append(
                                    "'%s' band %s %s: Changing attenuation from %idB to %idB " %
                                    (band, ant.name, pol, atten, atten + 1))
                            # Attenuation was reduced
                            elif atten > 0 and (voltage < adc_volt or std < adc_std_in):
                                user_logger.info(
                                    "'%s' band %s %s: Changing attenuation from %idB to %idB " %
                                    (band, ant.name, pol, atten, atten - 1))
                                ant.req.dig_attenuation(pol, atten - 1)
                                pol_update[i] = True
                                dictionary_data[key]['dB_change'].append(
                                    "'%s' band %s %s: Changing attenuation from %idB to %idB " %
                                    (band, ant.name, pol, atten, atten - 1))
                            else:  # Attenuation is fine  .............
                                std, atten, voltage = sample_bits(
                                    ant, pol, band=band)
                                dictionary_data[key]['time'].append(
                                    time.strftime('%d/%m/%Y %H:%M:%S'))
                                dictionary_data[key]['attenuation'].append(
                                    atten)
                                dictionary_data[key]['std'].append(std)
                                dictionary_data[key]['vol'].append(voltage)
                                dictionary_data[key]['ant_name'].append(key)
                            dictionary_data[key]['counts'] += 1

        for key in dictionary_data.keys():  # Sorting for email writing
            attenuation_change_list = np.array(
                np.diff(dictionary_data[key]['attenuation']))
            std_change_list = np.array(np.diff(dictionary_data[key]['std']))
            vol_change_list = np.array(np.diff(dictionary_data[key]['vol']))
            for i in range(len(std_change_list)):
                dictionary_data[key]['atten_change'].append(
                    attenuation_change_list[i])
                dictionary_data[key]['std_change'].append(std_change_list[i])
                dictionary_data[key]['vol_change'].append(vol_change_list[i])

        summary = []
        summary.append('The refine attenuation report summary')
        summary.append(' ')
        summary.append("Antenna   Band    H-pol    V-pol ")
        summary.append('-------------------------------------------')
        summary.append('   ')

        summary_notes = []
        summary_notes.append(' ')
        summary_notes.append(' Antennas that changed: ')
        summary_notes.append('---------------------------------------')

        changed_antennas = []
        changed_antennas.append(' ')
        changed_antennas.append(
            ' Changed antennas updates (ant, H-pol, V-pol): ')
        changed_antennas.append('---------------------------------------')

        for ant in kat.ants:  # Reporting antenna loop

            band = get_ant_band(ant)

            pol_summary = []
            for pol in {'h', 'v'}:
                key = str(ant.name) + pol
                lines2.append(' ')
                lines2.append(' Refine attenuation report for: ' +
                              key +
                              ',Band = ' +
                              band +
                              ', No. of evaluations  = ' +
                              str(dictionary_data[key]['counts']))
                lines2.append('--------------------------------')
                lines2.append(
                    "Attenuation      Std       Voltage      Delta atten   Delta Std    Delta Voltage")

                attenuation_lst = np.array(dictionary_data[key]['attenuation'])
                std_lst = np.array(dictionary_data[key]['std'])
                vol_lst = np.array(dictionary_data[key]['vol'])

                chg_att_lst = np.array(dictionary_data[key]['atten_change'])
                chg_std_lst = np.array(dictionary_data[key]['std_change'])
                chg_vol_lst = np.array(dictionary_data[key]['vol_change'])
                changed_attenuation = np.array(
                    dictionary_data[key]['dB_change'])

                for i in range(len(attenuation_lst)):
                    lines2.append(
                        "{:>10.3f}     {:>10.3f}      {:>10.3f}      {:>10.3f}   {:>10.3f}      {:>10.3f}".format(
                            round(
                                attenuation_lst[i], 3), round(
                                std_lst[i], 3), round(
                                vol_lst[i], 3), chg_att_lst[i], chg_std_lst[i], chg_vol_lst[i]))

                if int(attenuation_lst[-1]) != int(atten_ref[key]):
                    summary_notes.append(changed_attenuation[-1])
                    if str(ant.name) not in changed_antennas:
                        changed_antennas.append(str(ant.name) + ', ' + str(dictionary_data[str(
                            ant.name) + 'h']['attenuation'][-1]) + ', ' + str(dictionary_data[str(ant.name) + 'v']['attenuation'][-1]))
                # append the last value in the list, where the loop ended
                pol_summary.append(int(attenuation_lst[-1]))
            summary.append(str(ant.name) + '        ' + str(band) + \
                           "{:>10.3f}      {:>10.3f}   ".format(int(pol_summary[0]), int(pol_summary[1])))

        summary = np.concatenate((summary, summary_notes, changed_antennas))
        send_email(
            opts.email_to,
            summary,
            'Summary:Changing attenuation %s' %
            (time.strftime('%d/%m/%Y %H:%M:%S')),
            messagefrom='operators@ska.ac.za')
        send_email(
            'meerkat-attenuation@sarao.ac.za',
            lines2,
            'Details Changing attenuation %s' %
            (time.strftime('%d/%m/%Y %H:%M:%S')),
            messagefrom='operators@ska.ac.za')
