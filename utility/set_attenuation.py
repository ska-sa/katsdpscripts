#!/usr/bin/env python
# This script sets the attenuation values based on a csv file

import numpy as np
import katconf
import time
import StringIO
from katcorelib import (
    user_logger, standard_script_options, verify_and_connect, colors)


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


def measure_atten(ant, pol, atten_ref=None, band='l'):
    """ This function returns the attenuation of an antenna and colors the 
        logging if this number differs from the reference value
    Example:
            $ measure_atten('m064', 'h',atten_ref=5)
        returns 4
        with log message:
        <<date time>> band l: m064 h  Attenuation : <yellow> 4 <default color> "

        ant is an katcp antenna object
        pol is a string
        value and test are the antenna name and the polorisation
        and atten_ref is the expected values.
    """

    sensor = "dig_%s_band_rfcu_%spol_attenuation" % (band, pol)
    atten = ant.sensor[sensor].get_value()
    color_d = color_code_eq(atten, atten_ref)
    string = "'%s' band: %s %s  Attenuation : %s %-2i %s " % (
        band, ant.name, pol, color_d, atten, colors.Normal)
    print string
    user_logger.info(string)
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


# Set up standard script options
usage = "%prog   "  # <atten_ref.csv>
description = 'Sets the attenuation according to a attenuation reference file  '
parser = standard_script_options(usage=usage, description=description)
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Set Attenuate', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    atten_ref = {}
    for band in ['l', 'u']:  # ,'s','x'   # Read in the bands
        if not len(args) == 0:
            raise RuntimeError(
                'This script no longer takes in an attenuation file. Please raise an issue if you need this ')
        user_logger.info("This script used values found in katconf/katconfig")
        user_logger.info("Reading file katconf:'katconfig/user/attenuation/mkat/dig_attenuation_%s.csv'" % (band))
        file_string = katconf.resource_string(
            'katconfig/user/attenuation/mkat/dig_attenuation_%s.csv' % (band))
        tmp_data =  np.loadtxt(StringIO.StringIO(file_string),dtype=np.str,delimiter=',')
        for ant, value_h, value_v in tmp_data:
            try:
                atten_ref['%s_%s_%s' % (band, ant, 'h')] = np.int(value_h)
                atten_ref['%s_%s_%s' % (band, ant, 'v')] = np.int(value_v)
            except ValueError:
                user_logger.warning(
                    "'%s' band  %s: attenuation value '%s','%s' is not an integer " % (band, ant,  value_h, value_v))
    if not kat.dry_run:
        for pol in {'h', 'v'}:
            for ant in kat.ants:  # note ant is an katcp antenna object
                band = get_ant_band(ant)
                key_lookup = '%s_%s_%s' % (band, ant.name, pol)
                if not key_lookup in atten_ref:
                    user_logger.error("'%s' band %s %s: Has no attenuation value in the file " % (
                        band, ant.name, pol))
                else:
                    atten = measure_atten(
                        ant, pol, atten_ref=atten_ref[key_lookup], band=band)
                    if atten != atten_ref[key_lookup]:
                        user_logger.info("'%s' band %s %s: Changing attenuation from %idB to %idB " % (
                            band, ant.name, pol, atten, atten_ref[key_lookup]))
                        print "%s band %s %s: Changing attenuation from %idB to %idB " % (
                            band,ant.name, pol, atten, atten_ref[key_lookup])
                        ant.req.dig_attenuation(
                            pol, atten_ref[key_lookup])
            user_logger.info("Sleeping for 30 seconds ")
            time.sleep(30)
