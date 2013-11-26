#!/usr/bin/python
# Track target and possibly calibrator for beamforming.

# The *with* keyword is standard in Python 2.6, but has to be explicitly
# imported in Python 2.5
from __future__ import with_statement

import time
import StringIO

import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger, ant_array)


def bf_inputs(cbf, bf):
    """Input labels associated with specified beamformer."""
    reply = cbf.req.dbe_label_input()
    return [] if not reply.succeeded else \
           [m.arguments[0] for m in reply.messages[1:] if m.arguments[3] == bf]


def select_ant(cbf, ant, bf='bf0'):
    """Only use one antenna in specified beamformer."""
    for inp in bf_inputs(cbf, bf):
        weight = '1' if inp == ant else '0'
        cbf.req.dbe_k7_beam_weights(bf, inp, *(1024 * [weight]))


def get_weights(cbf):
    weights = {}
    for sensor_name in vars(cbf.sensor):
        if sensor_name.endswith('_gain_correction_per_channel'):
            sensor = getattr(cbf.sensor, sensor_name)
            weights[sensor_name.split('_')[1]] = sensor.get_value()
    return weights


def phase_up(cbf, weights, ants=None, bf='bf0', phase_only=True, scramble=False):
    """Phase up a group of antennas using latest gain corrections."""
    for inp in bf_inputs(cbf, bf):
        status = 'beamformer input ' + inp + ':'
        if (ants is None or inp in ants) and inp in weights and weights[inp]:
            weights_str = weights[inp]
            if phase_only:
                f = StringIO.StringIO(weights_str)
                weights_arr = np.loadtxt(f, dtype=np.complex, delimiter=' ')
                norm_weights = weights_arr / np.abs(weights_arr)
                status += ' normed'
                if scramble:
                    norm_weights *= np.exp(2j * np.pi *
                                           np.random.rand(len(norm_weights)))
                    status += ' scrambled'
                weights_str = ' '.join([('%+5.3f%+5.3fj' % (w.real, w.imag))
                                        for w in norm_weights])
        else:
            weights_str = ' '.join(1024 * ['0'])
            status += ' zeroed'
        cbf.req.dbe_k7_beam_weights(bf, inp, weights_str)
        user_logger.info(status)


class BeamformerSession(object):
    """Context manager that ensures that beamformer is switched off."""
    def __init__(self, cbf, instrument):
        self.cbf = cbf
        self.instrument = instrument

    def __enter__(self):
        """Enter the data capturing session, starting capture."""
        user_logger.info('starting beamformer')
        self.cbf.req.dbe_capture_start(self.instrument)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the data capturing session, stopping the capture."""
        if exc_value is not None:
            exc_msg = str(exc_value)
            msg = "Session interrupted by exception (%s%s)" % \
                  (exc_value.__class__.__name__,
                   (": '%s'" % (exc_msg,)) if exc_msg else '')
            if exc_type is KeyboardInterrupt:
                user_logger.warning(msg)
            else:
                user_logger.error(msg, exc_info=True)
        self.cbf.req.dbe_capture_stop(self.instrument)
        user_logger.info('beamformer stopped')
        # Suppress KeyboardInterrupt so as not to scare the lay user,
        # but allow other exceptions that occurred in the body of with-statement
        return exc_type is KeyboardInterrupt


# Set up standard script options
usage = "%prog [options] <'target'> [<'cal_target'>]"
description = "Perform a beamforming run on a specified target, optionally " \
              "visiting a gain calibrator beforehand to set beamformer weights."
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-a', '--ants', default='all',
                  help="Antennas to include in beamformer (default='%default')")
parser.add_option('-b', '--beamformer', default='bf0',
                  help='Name of beamformer instrument to use '
                       '(bf0 -> H pol, bf1 -> V pol, default=%default)')
parser.add_option('-t', '--target-duration', type='float', default=20,
                  help='Minimum duration to track the beamforming target, '
                       'in seconds (default=%default)')
parser.add_option('-c', '--cal-duration', type='float', default=120,
                  help='Minimum duration to track calibrator, in seconds '
                       '(default=%default)')
parser.add_option('--fix-amp', action='store_true', default=False,
                  help='Fix amplitude as well as phase in beamformer weights')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Beamformer calibration', nd_params='off',
                    dump_rate=1.0, mode='bc16n400M1k')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments and connect to KAT proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target (and optionally calibrator) "
                     "to observe as arguments")
with verify_and_connect(opts) as kat:
    cbf = kat.dbe7
    # Antennas and polarisations forming beamformer
    ants = ant_array(kat, opts.ants)
    pol = 'h' if opts.beamformer == 'bf0' else 'v'
    # Pick first target as main beamformer target
    target = collect_targets(kat, args[:1]).targets[0]

    # Determine beamformer weights if calibrator is provided
    if len(args) > 1:
        cal_target = collect_targets(kat, args[1:]).targets[0]
        cal_target = cal_target.add_tags('gaincal')
        user_logger.info('Obtaining beamformer weights on calibrator source %r' %
                         (cal_target.name))
        if not cal_target.flux_model:
            raise ValueError("Calibrator '%s' has no flux density model" %
                             (cal_target.description,))
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            session.capture_start()
            session.label('track')
            session.track(cal_target, duration=opts.cal_duration)

    user_logger.info('Phasing up beamformer based on %d antennas' % (len(ants),))
    weights = get_weights(cbf)
    if not weights:
        raise ValueError('No beamformer weights are available')
    phase_up(cbf, weights, ants=[(ant.name + pol) for ant in ants],
             bf=opts.beamformer, phase_only=not opts.fix_amp)

    user_logger.info("Initiating %g-second track on target '%s'" %
                     (opts.target_duration, target.name))
    ants.req.target(target)
    cbf.req.target(target)
    # We need delay tracking
    cbf.req.auto_delay()
    user_logger.info('slewing to target')
    # Start moving each antenna to the target
    ants.req.mode('POINT')
    # Wait until they are all in position (with 5 minute timeout)
    ants.req.sensor_sampling('lock', 'event')
    ants.wait('lock', True, 300)
    user_logger.info('target reached')

    with BeamformerSession(cbf, opts.beamformer):
        user_logger.info('tracking target')
        time.sleep(opts.target_duration)
        user_logger.info('target tracked for %g seconds' % (opts.target_duration,))
