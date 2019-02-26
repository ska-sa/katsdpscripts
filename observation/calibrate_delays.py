#!/usr/bin/env python
#
# Track delay calibrator target for a specified time.
# Obtain delay solutions and apply them to the delay tracker in the CBF proxy.
#
# Ludwig Schwardt
# 5 April 2017
#

import numpy as np
import katconf
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                CalSolutionsUnavailable,colors)

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


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Default F-engine gain as a function of number of channels
DEFAULT_GAIN = {1024: 116, 4096: 70, 32768: 360}

# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track the source with the highest elevation and calibrate ' \
              'delays based on it. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=32.0,
                  help='Length of time to track the source for calibration, '
                       'in seconds (default=%default)')
parser.add_option('--verify-duration', type='float', default=64.0,
                  help='Length of time to revisit source for verification, '
                       'in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int', default=0,
                  help='Override correlator F-engine gain, using the default '
                       'gain value for the mode if 0')
parser.add_option('--fft-shift', type='int',
                  help='Set correlator F-engine FFT shift (default=leave as is)')
parser.add_option('--reset-delays', action='store_true', default=False,
                  help='Zero the delay adjustments afterwards')
parser.add_option('--set-attenuation', action='store_true', default=False,
                  help='Set the attenuation of the system.')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Delay calibration observation')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    if opts.set_attenuation :
        atten_ref = {}
        for band in ['l', 'u']:  # ,'s','x'   # Read in the bands
            user_logger.info("Reading file katconf:'katconfig/user/attenuation/mkat/dig_attenuation_%s.csv'" % (band))
            file_string = katconf.resource_string(
                'katconfig/user/attenuation/mkat/dig_attenuation_%s.csv' % (band))
            tmp_data = [a.split(',') for a in file_string.split('\n')]
            for ant, value_h, value_v in tmp_data:
                if not ant[0] == '#':
                    try:
                        atten_ref['%s_%s_%s' % (band, ant, 'h')] = np.int(value_h)
                        atten_ref['%s_%s_%s' % (band, ant, 'v')] = np.int(value_v)
                    except ValueError:
                        user_logger.warning(
                            "'%s' band  %s: attenuation value '%s','%s' is not an integer " % (band, ant,  value_h, value_v))
        if not kat.dry_run:
            for ant in kat.ants:  # note ant is an katcp antenna object
                band = get_ant_band(ant)
                for pol in {'h', 'v'}:
                    if '%s_%s_%s' % (band, ant.name, pol) in atten_ref:
                        atten = measure_atten(
                            ant, pol, atten_ref=atten_ref['%s_%s_%s' % (band, ant.name, pol)], band=band)
                        if atten != atten_ref['%s_%s_%s' % (band, ant.name, pol)]:
                            user_logger.info("'%s' band %s %s: Changing attenuation from %idB to %idB " % (
                                band, ant.name, pol, atten, atten_ref['%s_%s_%s' % (band, ant.name, pol)]))
                            # print "%s band %s %s: Changing attenuation from %idB to %idB " % (
                            #    band,ant.name, pol, atten, atten_ref['%s_%s_%s' % (band,ant.name, pol)])
                            ant.req.dig_attenuation(
                                pol, atten_ref['%s_%s_%s' % (band, ant.name, pol)])
                    else:
                        user_logger.error("'%s' band %s %s: Has no attenuation value in the file " % (
                            band, ant.name, pol))
    observation_sources = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe or not enough antennas
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Pick source with the highest elevation as our target
        target = observation_sources.sort('el').targets[-1]
        target.add_tags('bfcal single_accumulation')
        session.standard_setup(**vars(opts))
        if opts.fft_shift is not None:
            session.cbf.fengine.req.fft_shift(opts.fft_shift)
        if opts.fengine_gain <= 0:
            num_channels = session.cbf.fengine.sensor.n_chans.get_value()
            try:
                opts.fengine_gain = DEFAULT_GAIN[num_channels]
            except KeyError:
                raise KeyError("No default gain available for F-engine with "
                               "%i channels - please specify --fengine-gain"
                               % (num_channels,))
        cal_inputs = session.get_cal_inputs()
        gains = {inp: opts.fengine_gain for inp in cal_inputs}
        delays = {inp: 0.0 for inp in cal_inputs}
        session.set_fengine_gains(gains)
        user_logger.info("Zeroing all delay adjustments for starters")
        session.set_delays(delays)
        session.capture_init()
        user_logger.info("Only calling capture_start on correlator stream directly")
        session.cbf.correlator.req.capture_start()
        user_logger.info("Initiating %g-second track on target %r",
                         opts.track_duration, target.description)
        session.label('un_corrected')
        session.track(target, duration=0)  # get onto the source
        # Fire noise diode during track
        session.fire_noise_diode(on=opts.track_duration, off=0)
        # Attempt to jiggle cal pipeline to drop its delay solutions
        session.stop_antennas()
        user_logger.info("Waiting for delays to materialise in cal pipeline")
        hv_delays = session.get_cal_solutions('KCROSS_DIODE', timeout=300.)
        delays = session.get_cal_solutions('K')
        # Add hv_delay to total delay
        for inp in sorted(delays):
            delays[inp] += hv_delays[inp]
            if np.isnan(delays[inp]):
                user_logger.warning("Delay fit failed on input %s (all its "
                                    "data probably flagged)", inp)
        # XXX Remove any NaNs due to failed fits (move this into set_delays)
        delays = {inp: delay for inp, delay in delays.items()
                  if not np.isnan(delay)}
        if not delays and not kat.dry_run:
            raise CalSolutionsUnavailable("No valid delay fits found "
                                          "(is everything flagged?)")
        session.set_delays(delays)
        if opts.verify_duration > 0:
            user_logger.info("Revisiting target %r for %g seconds "
                             "to see if delays are fixed",
                             target.name, opts.verify_duration)
            session.label('corrected')
            session.track(target, duration=0)  # get onto the source
            # Fire noise diode during track
            session.fire_noise_diode(on=opts.verify_duration, off=0)
        if opts.reset_delays:
            user_logger.info("Zeroing all delay adjustments on CBF proxy")
            delays = {inp: 0.0 for inp in delays}
            session.set_delays(delays)
        else:
            # Set last-delay-calibration script sensor on the subarray.
            session.sub.req.set_script_param('script-last-delay-calibration', kat.sb_id_code)
