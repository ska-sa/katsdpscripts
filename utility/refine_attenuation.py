#!/usr/bin/env python
# Track SCP and and determine attenuation values

import time
import numpy as np
from katcorelib import (
    user_logger, standard_script_options, verify_and_connect, colors)
import smtplib
from email.mime.text import MIMEText

def send_email(email_to,lines, subject, messagefrom='operators@ska.ac.za'):
    body = '\n'.join(lines)
    body = string.replace(body, '\n', '<br>\n')
    html = """\
    <html>
        <body>
          <p>
              """ + body + """\
          </p>
        </body>
    </html>
    """
    if type(opts.email_to) is list:
        messageto = ', '.join((opts.email_to).replace(' ', ''))
    else:
        messageto = (email_to).replace(' ', '')
    msg = MIMEText(html, 'html')
    msg['Subject'] = subject
    msg['From'] = messagefrom
    msg['To'] = messageto
    if type(email_to) is list:
        sendto = (email_to).replace(' ', '')
    elif (email_to).find(',') >= 0:
        sendto = ((email_to).replace(' ', '')).split(',')
    elif (opts.email_to).find(';') >= 0:
        sendto = ((email_to).replace(' ', '')).split(';')
    else:
        sendto = (email_to).replace(' ', '')
    smtp_server = smtplib.SMTP('smtp.kat.ac.za')
    smtp_server.sendmail(messagefrom, sendto, msg.as_string())
    smtp_server.quit()


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
    return (np.mod(np.arange(num), 2)*2-1)


def sample_bits(ant, pol, band='l'):
    tmp_data = np.zeros((5, 4096, 4))
    for i in range(5):
        snap = ant.req.dig_adc_snap_shot(pol)
        tmp_data[i, :, :] = np.array(
            [snip.arguments[1:] for snip in snap.messages[1:]]).astype('float')
    data = tmp_data.flatten()
    std = (data*plus_minus(data.shape[0])).std()
    color_d = color_code(std, 12, 8)
    sensor = 'dig_%s_band_rfcu_%spol_attenuation' % (band, pol)
    atten = ant.sensor[sensor].get_value()
    windowed_data = data.reshape(-1, 256)
    voltage = np.abs(np.fft.fft(windowed_data)[:, 67:89]).mean() # 67:89  corresponds to 1300 to 1450 MHz
    # channels 67:89  correspond to a RFI free section of band (1300 to 1450 MHz).
    string = "%s ADC rms %s: %s%-4.1f %s  vlevel: %-4.1f  Attenuation : %-2i  " % (
        ant.name, pol, color_d, std, colors.Normal, voltage, atten)
    user_logger.info(string)
    return std, atten, voltage



# Set up standard script options
usage = "%prog [options] <'target/catalogue'> "
description = 'Calculate the attenuation needed and Set the attenuation to appropriate levels'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options

parser.add_option('--adc-std-in', type='float', default=12.0,
                  help='The target adc rms level  (default=%default)')
parser.add_option('--adc-volt', type='float', default=190.0,
                  help='The target power level for the adc (default=%default)')
parser.add_option('--email-to', type='str',
    default='sean@ska.ac.za,operators@ska.ac.za,cgumede@ska.ac.za',
    help='Comma separated email list of people to send report to (default=%default)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Auto Attenuate', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

# adc  : WARNING at -31dBm and ERROR at -37dBm
# rfcu : WARNING at -41dBm and ERROR at -47dBm
# Check options and build KAT configuration, connecting to proxies and devices
adc_volt = opts.adc_volt
adc_std_in = opts.adc_std_in
bandlist = ['l', 'u'] # ,'s','x'   # Read in the bands
with verify_and_connect(opts) as kat:
    for band in bandlist:  # ,'s','x'   # Read in the bands
        for pol in {'h', 'v'}:
            kat.ants.set_sampling_strategy("dig_%s_band_adc_%spol_attenuation" %
                                       (band, pol), "period 1.0")
    if not kat.dry_run:
        point(kat.ants, 'SCP,radec,0,-90', timeout=300)
        ant_update = np.ones((len(kat.ants)*2)).astype(bool)
        count = 0
        while ant_update.sum() > 0 and count < 20:
            i = -1
            count = count + 1
            time.sleep(5)
            print("New loop")
            for pol in {'h', 'v'}:
                for ant in kat.ants:
                    band = get_ant_band(ant)
                    if band in bandlist:
                        i = i + 1
                        if ant_update[i]:
                            ant_update[i] = False
                            std, atten, voltage = sample_bits(ant, pol ,band=band)
                            if atten < 32 and (voltage > adc_volt + 20):  # Up
                                user_logger.info("'%s' band %s %s: Changing attenuation from %idB to %idB " % (
                                    band,ant.name, pol, atten, atten+1))
                                ant.req.dig_attenuation(pol, atten+1)
                                ant_update[i] = True
                            if atten > 0 and (voltage < adc_volt or std < adc_std_in):
                                user_logger.info("'%s' band %s %s: Changing attenuation from %idB to %idB " % (
                                    band,ant.name, pol, atten, atten-1))
                                ant.req.dig_attenuation(pol, atten-1)
                                ant_update[i] = True
                    else :
                        user_logger.error("'%s' band is not in the list of valid bands " % (band))
        lines = []
        summary = []
        atten_ref = {}
        ant_list = []
        lines.append('Changing attenuation , report of refine_attenuation.py')
        for ant in kat.ants:
            band = get_ant_band(ant)
            if band in bandlist:
                ant_list.append(ant.name)
                for pol in {'h', 'v'}:
                    std, atten, voltage = sample_bits(ant, pol,band=band)
                    lines.append("'%s' band %s %s: ,%i #  std:%f   vol:%f"%(band,ant.name, pol,atten,std,voltage))
                    atten_ref['%s_%s' % (ant.name, pol)] = [measure_atten(ant=ant, pol=pol,band=band),band]
            else :
                user_logger.error("'%s' band is not in the list of valid bands " % (band))
            user_logger.info("Reading Back set Attenuations ")
            user_logger.info("# band Antenna Name, H-pol , V-pol " )
            summary.append("# band Antenna Name, H-pol , V-pol " )
            for ant in ant_list.sort():
                string =  (" '%s' band : %s, %i, %i "%(
                atten_ref['%s_%s'%(ant,'h')][1] ,ant, atten_ref['%s_%s'%(ant,'h')][0] ,atten_ref['%s_%s'%(ant,'h')][0] ) )
                user_logger.info(string)
                summary.append(string)
            lines = summary.append(lines)
        #try:
        #    send_email(opts.email_to,lines, 'Changing attenuation %s'%(time.strftime('%d/%m/%Y %H:%M:%S')), messagefrom='operators@ska.ac.za')
