#!/usr/bin/python
# Dual polarisation beamforming: Track target and possibly calibrator for beamforming.

# The *with* keyword is standard in Python 2.6, but has to be explicitly
# imported in Python 2.5
from __future__ import with_statement

import time
import StringIO
import logging

import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger, ant_array)
import fbf_katcp_wrapper as fbf


class BeamformerReceiver(fbf.FBFClient):
    """KATCP client to beamformer receiver, with added metadata."""
    def __init__(self, name, server, rx_port, pol, meta_port, data_port, data_drive):
        user_logger.info('Connecting to server %r for beam %r...' % (server, name))
        logger = logging.getLogger('katcp')
        super(BeamformerReceiver, self).__init__(host=server, port=rx_port,
                                                 timeout=60, logger=logger)
        while not self.is_connected():
            user_logger.info('Waiting for TCP link to receiver server...')
            time.sleep(1)
        user_logger.info('Connected to server %r for beam %r' % (server, name))
        self.name = name
        self.pol = pol
        self.meta_port = meta_port
        self.data_port = data_port
        self.data_drive = data_drive
        self.obs_meta = {}

    def __repr__(self):
        return "<BeamformerReceiver %r -> %r at 0x%x>" % (self.name, self.pol, id(self))

    @property
    def inputs(self):
        return self.obs_meta.get('ants', [])


# Server where beamformer receivers are run
server = 'kat-dc2.karoo'
# beams = {'bf0': {'pol':'h', 'meta_port':'7152', 'data_port':'7150', 'rx_port':1235, 'data_drive':'/data1'},
#          'bf1': {'pol':'v', 'meta_port':'7153', 'data_port':'7151', 'rx_port':1236, 'data_drive':'/data2'}}
beams = [BeamformerReceiver('bf0', server, rx_port=1235, pol='h', meta_port=7152,
                            data_port=7150, data_drive='/data1'),
         BeamformerReceiver('bf1', server, rx_port=1236, pol='v', meta_port=7153,
                            data_port=7151, data_drive='/data2')]


def bf_inputs(cbf, bf):
    """Input labels associated with specified beamformer (*all* inputs)."""
    reply = cbf.req.dbe_label_input()
    return [] if not reply.succeeded else \
           [m.arguments[0] for m in reply.messages[1:] if m.arguments[3] == bf]


def select_ant(cbf, input, bf='bf0'):
    """Only use one antenna in specified beamformer."""
    # Iterate over *all* inputs going into the given beam
    for inp in bf_inputs(cbf, bf):
        status = 'beamformer input ' + inp + ':'
        weight = '1' if inp == input else '0'
        status += ' kept' if inp == input else ' zeroed'
        cbf.req.dbe_k7_beam_weights(bf, inp, *(1024 * [weight]))
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


def phase_up(cbf, weights, inputs=None, bf='bf0', style='flatten'):
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
        Name of beamformer instrument (one per polarisation)
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
                new_weights = orig_weights
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
        cbf.req.dbe_k7_beam_weights(bf, inp, weights_str)
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
    def __init__(self, cbf, beams):
        self.cbf = cbf
        self.beams = beams

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
        # Suppress KeyboardInterrupt so as not to scare the lay user,
        # but allow other exceptions that occurred in the body of with-statement
        if exc_type is KeyboardInterrupt:
            report_compact_traceback(traceback)
            return True
        else:
            return False

    def instrument_start(self, instrument):
        """Start given CBF instrument."""
        self.cbf.req.dbe_capture_start(instrument)
        user_logger.info('waiting 10s for stream %r to start' % (instrument,))
        time.sleep(10)

    def instrument_stop(self, instrument):
        """Stop given CBF instrument."""
        self.cbf.req.dbe_capture_stop(instrument)
        user_logger.info('waiting 10s for stream %r to stop' % (instrument,))
        time.sleep(10)

    def capture_start(self):
        """Enter the data capturing session, starting capture."""
        user_logger.info('Starting correlator (used for signal displays)')
        # Starting streams will issue metadata for capture
        # Allow long 10sec intervals to allow enough time to initiate data capture and to capture metadata
        # Else there will be collisions between the 2 beams
        for beam in self.beams:
            # Initialise receiver and setup server for data capture
            user_logger.info('Initialising receiver and stream for beam %r' %
                             (beam.name,))
            if not beam.rx_init(beam.data_drive, beam.obs_meta['half_band'],
                                beam.obs_meta['transpose']):
                raise RuntimeError('Could not initialise %r receiver' %
                                   (beam.name,))
            # Start metadata receiver before starting data transmit
            beam.rx_meta_init(beam.meta_port) # port
            self.instrument_start(beam.name)
            user_logger.info('beamformer metadata')
            beam.rx_meta(beam.obs_meta) # additional obs related info
            user_logger.info('waiting 10s to write metadata for beam %r' %
                             (beam.name,))
            time.sleep(10)
            # Start transmitting data
            user_logger.info('beamformer data for beam %r' % (beam.name,))
            beam.rx_beam(pol=beam.pol, port=beam.data_port)
            time.sleep(1)

    def capture_stop(self):
        """Exit the data capturing session, stopping the capture."""
        # End all receivers
        for beam in self.beams:
            user_logger.info('Stopping receiver and stream for beam %r' %
                             (beam.name,))
            beam.rx_stop()
            time.sleep(5)
            self.instrument_stop(beam.name)
            user_logger.info(beam.rx_close())
        user_logger.info('Stopping correlator (used for signal displays)')


# Set up standard script options
usage = "%prog [options] <'target'> [<'cal_target'>]"
description = "Perform a beamforming run on a specified target, optionally " \
              "visiting a gain calibrator beforehand to set beamformer weights."
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-a', '--ants', default='all',
                  help="Antennas to include in beamformer (default='%default')")
parser.add_option('-t', '--target-duration', type='float', default=20,
                  help='Minimum duration to track the beamforming target, '
                       'in seconds (default=%default)')
parser.add_option('--half-band', action='store_true', default=False,
                  help='Use only inner 50% of output band')
parser.add_option('--reset', action="store_true", default=False,
                  help='Reset the gains to 160.')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Beamformer observation', nd_params='off',
                    dump_rate=1.0, mode='bc16n400M1k')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments and connect to KAT proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target")
with verify_and_connect(opts) as kat:
    cbf = kat.dbe7
    ants = kat.ants
    # We are only interested in the first target
    user_logger.info('Looking up main beamformer target...')
    target = collect_targets(kat, args[:1]).targets[0]
    # Ensure that the target is up
    target_elevation = np.degrees(target.azel()[1])
    if target_elevation < opts.horizon:
        raise ValueError("The desired target to be observed is below the horizon")

    # Start correlator capture session
    with start_session(kat, **vars(opts)) as corr_session:
        corr_session.standard_setup(**vars(opts))
        corr_session.dbe.req.auto_delay('on')
        corr_session.capture_start()


        # Dictionary to hold observation metadata to send over to beamformer receiver
        for beam in beams:
            beam.obs_meta.update(vars(opts))
            beam.obs_meta['ants'] = [(ant.name + beam.pol) for ant in ants]
            beam.obs_meta['target'] = target.description
            if cal_target and len(ants) >= 4:
                beam.obs_meta['cal_target'] = cal_target.description

        if len(ants) > 1:
            user_logger.info('Setting beamformer weight to 1 for %d antennas' % (len(ants),))
            inputs = reduce(lambda inp, beam: inp + beam.inputs, beams, [])
            # set the beamformer weights to 1 as the phaseing is done in the f-engine
            weights = {}
            bf_weights_str = ' '.join(1024 * ['1'])
            for inp in inputs:
                weights[inp] = bf_weights_str
            for beam in beams:
                phase_up(cbf, weights, inputs=beam.inputs, bf=beam.name, style='norm')
                time.sleep(1)
        else:
            # The single-dish case does not need beamforming
            user_logger.info('Set beamformer weights to select single dish')
            for beam in beams:
                select_ant(cbf, input=beam.inputs[0], bf=beam.name)
                time.sleep(1)

        # Start beamformer session
        with BeamformerSession(cbf, beams) as bf_session:
            # Get onto beamformer target
            corr_session.label('track')
            corr_session.track(target, duration=0)
            # Only start capturing with beamformer once we are on target
            bf_session.capture_start()
            corr_session.track(target, duration=opts.target_duration)
