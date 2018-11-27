#!/usr/bin/env python
# This script sets the attenuation values based on a csv file

import numpy as np
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
        <<date time>> m064 h  Attenuation : <yellow> 4 <default color> "

        ant, pol are strings point numbers
        value and test are the antenna name and the polorisation
        and atten_ref is the expected values.
    """

    sensor = "dig_%s_band_rfcu_%spol_attenuation" % (band, pol)
    atten = ant.sensor[sensor].get_value()
    color_d = color_code_eq(atten, atten_ref)
    string = "%s %s  Attenuation : %s %-2i %s " % (
        ant.name, pol, color_d, atten, colors.Normal)
    user_logger.info(string)
    return atten


# Set up standard script options
usage = "%prog  <atten_ref.csv> "
description = 'Sets the attenuation according to a attenuation reference file  '

parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Set Attenuate', nd_params='off')
parser.add_option('--force', action="store_True", dest="verbose", default=False,  
                  help='Forces the attenuation change (default=%default)')

# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise IOError('No file passed to script.')
with verify_and_connect(opts) as kat:
    band = 'l'
    for pol in {'h', 'v'}:
        kat.ants.set_sampling_strategy("dig_%s_band_adc_%spol_attenuation" %
                                       (band, pol), "period 1.0")
    tmp_data = np.loadtxt(args[0], dtype=np.str,delimiter=',')
    atten_ref = {}
    for ant, pol, value in tmp_data:
        try:
            val = np.int(value)
            atten_ref['%s%s' % (ant, pol)] = val
        except ValueError:
            user_logger.warning(
                "%s %s: attenuation value '%s' is not an integer " % (ant, pol, value))
            #print ("%s %s: attenuation value '%s' is not an integer  " % (ant, pol, value))
    if not kat.dry_run:
        for ant in kat.ants:
            for pol in {'h', 'v'}:
                if '%s%s' % (ant.name, pol) in atten_ref:
                    atten = measure_atten(
                        ant, pol, atten_ref=atten_ref['%s%s' % (ant.name, pol)], band='l')
                    if atten != atten_ref['%s%s' % (ant.name, pol)] or opts.force :
                        user_logger.info("%s %s: Changing attenuation from %idB to %idB " % (
                            ant.name, pol, atten, atten_ref['%s%s' % (ant.name, pol)]))
                        ant.req.dig_attenuation(
                            pol, atten_ref['%s%s' % (ant.name, pol)])
                else :
                    user_logger.error("%s %s: Has no attenuation value in the file " % (ant.name, pol) )
