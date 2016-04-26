#!/usr/bin/python
# Dual polarisation beamforming: Track target and possibly calibrator for beamforming.

# The *with* keyword is standard in Python 2.6, but has to be explicitly
# imported in Python 2.5
from __future__ import with_statement

import time
import StringIO
import logging

import numpy as np

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)

def bf_inputs(data, stream):
    """Input labels associated with specified beamformer stream."""
    reply = data.req.cbf_input_labels() # do away with once get CAM sensor
    if not reply.succeeded:
        return []
    inputs = reply.messages[0].arguments[1:]
    return inputs[0::2] if stream.endswith('x') else inputs[1::2]


def select_ant(cbf, input, bf='beam_0x'):
    """Only use one antenna in specified beamformer."""
    # Iterate over *all* inputs going into the given beam
    for inp in bf_inputs(cbf, bf):
        status = 'beamformer input ' + inp + ':'
        weight = '1' if inp == input else '0'
        status += ' kept' if inp == input else ' zeroed'
        cbf.req.beam_weights(bf, inp, *(1024 * [weight]))
        user_logger.info(status)


def get_weights(cbf):
    """Retrieve the latest gain corrections and their corresponding update times."""
    weights, times = {}, {}
    for sensor_name in vars(cbf.sensor):
        if sensor_name.endswith('_gain_correction_per_channel'):
            sensor = getattr(cbf.sensor, sensor_name)
            weights[sensor_name.split('_')[1]] = sensor.get_value()
            times[sensor_name.split('_')[1]] = sensor.value_seconds
    return weights, times


def phase_up(cbf, weights, inputs=None, bf='beam_0x', style='flatten'):
    """Phase up a group of antennas using latest gain corrections.

    The *style* parameter determines how the complex gain corrections obtained
    on the latest calibrator source will be turned into beamformer weights:

      - 'norm': Apply the complex gain corrections unchanged as weights,
        thereby normalising both amplitude and phase per channel.
      - 'phase': Only apply the phase correction, leaving the weight amplitudes
        equal to 1. This has the advantage of not boosting weaker inputs that
        will increase the noise level, but it also does not flatten the band.
      - 'flatten': Apply both amplitude and phase corrections, but preserve
        mean gain of each input. This flattens the band while also not boosting
        noise levels on weaker inputs.
      - 'scramble': Apply random phase corrections, just for the heck of it.

    Parameters
    ----------
    cbf : client object
        Object providing access to CBF (typically via proxy)
    weights : string->string mapping
        Gain corrections per input as returned by appropriate sensor
    inputs : None or sequence of strings, optional
        Names of inputs in use in given beamformer (default=all)
    bf : string, optional
        Name of beamformer stream (one per polarisation)
    style : {'flatten', 'norm', 'phase', 'scramble'}, optional
        Processing done to gain corrections to turn them into weights

    """
    # Iterate over *all* inputs going into the given beam
    all_inputs = bf_inputs(cbf, bf)
    num_inputs = len(all_inputs)
    for inp in all_inputs:
        status = 'beamformer input ' + inp + ':'
        if (inputs is None or inp in inputs) and inp in weights and weights[inp]:
            # Extract array of complex weights from string representation
            weights_str = weights[inp]
            f = StringIO.StringIO(weights_str)
            orig_weights = np.loadtxt(f, dtype=np.complex, delimiter=' ')
            amp_weights = np.abs(orig_weights)
            phase_weights = orig_weights / amp_weights
            if style == 'norm':
                new_weights = orig_weights  # set B-engine weights to parsed weights
                status += ' normed'
            elif style == 'phase':
                new_weights = phase_weights
                status += ' phased'
            elif style == 'flatten':
                # Get the average gain in the KAT-7 passband
                avg_amp = np.median(amp_weights[256:768])
                new_weights = orig_weights / avg_amp
                status += ' flattened'
            elif style == 'scramble':
                new_weights = np.exp(2j * np.pi * np.random.rand(1024))
                status += ' scrambled'
            else:
                raise ValueError('Unknown phasing-up style %r' % (style,))
            # Normalise weights by number of inputs to avoid overflow
            new_weights /= num_inputs
            # Reconstruct string representation of weights from array
            weights_str = ' '.join([('%+5.3f%+5.3fj' % (w.real, w.imag))
                                    for w in new_weights])
        else:
            # Zero the inputs that are not in use in the beamformer
            weights_str = ' '.join(1024 * ['0'])
            status += ' zeroed'
        cbf.req.beam_weights(bf, inp, weights_str)
        user_logger.info(status)


def report_compact_traceback(tb):
    """Produce a compact traceback report."""
    print '--------------------------------------------------------'
    print 'Session interrupted while doing (most recent call last):'
    print '--------------------------------------------------------'
    while tb:
        f = tb.tb_frame
        print '%s %s(), line %d' % (f.f_code.co_filename, f.f_code.co_name, f.f_lineno)
        tb = tb.tb_next
    print '--------------------------------------------------------'


class BeamformerSession(object):
    """Context manager that ensures that beamformer is switched off."""
    def __init__(self, cbf):
        self.cbf = cbf   # cbf = CBF + SP data proxy i.e. rename to "data"

    def __enter__(self):
        """Enter the data capturing session."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the data capturing session, closing all streams."""
        if exc_value is not None:
            exc_msg = str(exc_value)
            msg = "Session interrupted by exception (%s%s)" % \
                  (exc_value.__class__.__name__,
                   (": '%s'" % (exc_msg,)) if exc_msg else '')
            if exc_type is KeyboardInterrupt:
                user_logger.warning(msg)
            else:
                user_logger.error(msg, exc_info=True)
        self.capture_stop()
        self.capture_done()

        # Suppress KeyboardInterrupt so as not to scare the lay user,
        # but allow other exceptions that occurred in the body of with-statement
        if exc_type is KeyboardInterrupt:
            report_compact_traceback(traceback)
            return True
        else:
            return False

    def stream_start(self, stream):
        """Start given CBF stream."""
        self.cbf.req.capture_start(stream)
        user_logger.info('waiting 1s for stream %r to start' % (stream,))
        time.sleep(1)

    def stream_stop(self, stream):
        """Stop given  stream."""
        self.cbf.req.capture_stop(stream)
        user_logger.info('waiting 1s for stream %r to stop' % (stream,))
        time.sleep(1)

    def capture_start(self):
        """Enter the data capturing session, starting capture."""
        # Starting streams will issue metadata for capture
        self.stream_start('beam_0x')
        self.stream_start('beam_0y')

    def capture_stop(self):
        """Exit the data capturing session, stopping the capture."""
        # End all receivers
        self.stream_stop('beam_0x')
        self.stream_stop('beam_0y')

# Set up standard script options
usage = "%prog [options] <'target'>"
description = "Perform a beamforming run on a target. It is assumed that " \
              "the beamformer is already phased up on a calibrator."
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('--ants',
                  help='Comma-separated list of antennas to use in beamformer '
                       '(default=all antennas in subarray)')
parser.add_option('-t', '--target-duration', type='float', default=20,
                  help='Minimum duration to track the beamforming target, '
                       'in seconds (default=%default)')
parser.add_option('-B', '--beam-bandwidth', type='float', default=40.0,
                  help="Beamformer bandwidth, in MHz (default=%default)")
parser.add_option('-F', '--beam-centre-freq', type='float', default=920.0,
                  help="Beamformer bandwidth, in MHz (default=%default)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Beamformer observation', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments and connect to KAT proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target")

with verify_and_connect(opts) as kat:
    bf_ants = opts.ants.split(',') if opts.ants else [ant.name for ant in kat.ants]
    bf_streams = ('beam_0x', 'beam_0y')
    for stream in bf_streams:
        kat.data.req.cbf_beam_passband(stream, int(opts.beam_bandwidth * 1e6),
                                               int(opts.beam_centre_freq * 1e6))
        for inp in bf_inputs(kat.data, stream):
            weight = 1.0 if inp[:-1] in bf_ants else 0.0
            kat.data.req.cbf_beam_weights(stream, inp, weight)

    # We are only interested in first target (use default catalogue if no pulsar specified)
    user_logger.info('Looking up main beamformer target...')
    target = collect_targets(kat, args[:1]).targets[0]

    # Ensure that the target is up
    target_elevation = np.degrees(target.azel()[1])
    if target_elevation < opts.horizon:
        raise ValueError("The desired target to be observed is below the horizon")

    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.data.req.auto_delay('on')
        # Assume correlator stream is bc product name without the 'b'
        session.data.req.capture_start(opts.product[1:])
        # Get onto beamformer target
        session.label('track')
        session.track(target, duration=0)
        # Only start capturing with beamformer once we are on target
        for stream in bf_streams:
            session.data.req.capture_start(stream)
        session.track(target, duration=opts.target_duration)
