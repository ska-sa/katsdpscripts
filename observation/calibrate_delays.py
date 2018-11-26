#!/usr/bin/env python
#
# Track delay calibrator target for a specified time.
# Obtain delay solutions and apply them to the delay tracker in the CBF proxy.
#
# Ludwig Schwardt
# 5 April 2017
#

import numpy as np
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                CalSolutionsUnavailable)


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
parser.add_option('--reset-delays', action='store_true', default=False,
                  help='Zero the delay adjustments afterwards')
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
