#!/usr/bin/python
# Dual polarisation beamforming: Track target and possibly calibrator for beamforming.

# The *with* keyword is standard in Python 2.6, but has to be explicitly
# imported in Python 2.5
from __future__ import with_statement

import time
import StringIO

import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger, ant_array)
import katpoint

import fbf_katcp_wrapper as katcp
import copy
import logging
import sys
import time

log = logging.getLogger("katcp")

DEBUG=False
## CONFIG VALUES ##
server = 'kat-dc2.karoo'
beams = {'bf0': {'pol':'h', 'meta_port':'7152', 'data_port':'7150', 'rx_port':1235, 'data_drive':'/data1'},
         'bf1': {'pol':'v', 'meta_port':'7153', 'data_port':'7151', 'rx_port':1236, 'data_drive':'/data2'}}
## CONFIG VALUES ##


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

def natural_weights(cbf, ants=None, bf='bf0'):
    """Assign a natural weight of 1 to a group of antennas."""
    for inp in bf_inputs(cbf, bf):
        status = 'beamformer input ' + inp + ':'
        if (ants is None or inp in ants):
            weights_str = ' '.join(1024 * ['1'])
            status += ' unity'
        else:
            weights_str = ' '.join(1024 * ['0'])
            status += ' zeroed'
        cbf.req.dbe_k7_beam_weights(bf, inp, weights_str)
        user_logger.info(status)


class BeamformerSession(object):
    """Context manager that ensures that beamformer is switched off."""
    def __init__(self, cbf):
        self.cbf = cbf

    def capture_start(self, instrument):
        """Enter the data capturing session, starting capture."""
        user_logger.info('starting beamformer')
        self.cbf.req.dbe_capture_start(instrument)
        user_logger.info('waiting 10s for stream %s to start'%instrument)
        time.sleep(10)

    def capture_stop(self, instrument):
        """Exit the data capturing session, stopping the capture."""
        self.cbf.req.dbe_capture_stop(instrument)
        user_logger.info('waiting 10s for stream %s to stop'%instrument)
        time.sleep(10)
        user_logger.info('beamformer stopped')

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
parser.add_option('-c', '--cal-duration', type='float', default=120,
                  help='Minimum duration to track calibrator, in seconds '
                       '(default=%default)')
parser.add_option('--fix-amp', action='store_true', default=False,
                  help='Fix amplitude as well as phase in beamformer weights')
parser.add_option('--half-band', action='store_true', default=False,
                  help='Use only inner 50% of otuput band')
parser.add_option('--transpose', action='store_true', default=False,
                  help='Transpose time frequency blocks from correlator')
parser.add_option('--phase-once', action='store_true', default=False,
                  help='Transpose time frequency blocks from correlator')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Beamformer observation', nd_params='off',
                    dump_rate=1.0, mode='bc16n400M1k')
# Parse the command line
opts, args = parser.parse_args()

##System: Set up all connections and objects
# Connecting to server
# if the correlator is running on a different server -- the ip of that server is specified as a string in 'host'
sys.stdout.write('Connecting to server %s...\n'%(server))
try:
    # Create katcp correlator instance
    katcp_bf0 = katcp.FBFClient(host=server, port=beams['bf0']['rx_port'], timeout=60, logger=log)
    katcp_bf1 = katcp.FBFClient(host=server, port=beams['bf1']['rx_port'], timeout=60, logger=log)
    while not katcp_bf0.is_connected() or not katcp_bf1.is_connected():
        sys.stdout.write('Waiting for TCP link to KATCP...\n')
        sys.stdout.flush()
        time.sleep(1)
except Exception as e:
    sys.stderr.write('ERROR connecting to server %s...\n'%(server))
    print e
    sys.exit(1)
sys.stdout.write('Connection established to server %s...\n'%(server))
# Dictionary to hold observation metadata to send over the receiver
obs_meta={}
for beam in beams.keys():
    obs_meta[beam]={}

# Check options and arguments and connect to KAT proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target (and optionally calibrator) "
                     "to observe as arguments")
cal_targets = []
done = False
with verify_and_connect(opts) as kat:
    cbf = kat.dbe7
    for beam in beams.keys(): obs_meta[beam] = copy.copy(vars(opts))
    # Antennas and polarisations forming beamformer
    ants = ant_array(kat, opts.ants)
    for beam in beams.keys(): obs_meta[beam]['ants'] = [(ant.name + beams[beam]['pol']) for ant in ants]
    observation_sources = collect_targets(kat, args[:1])
    if len(args) > 1:
        cal_targets = collect_targets(kat, args[1:]).targets

    for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
        for beam in beams.keys(): obs_meta[beam]['target'] = target.description

        # Phase up for each target
        if not done:
            if opts.phase_once: done=True
            # Determine beamformer weights if calibrator is provided
            if len(cal_targets) > 0 and len(ants) > 4: # must have at least 5 antennas for phases to be computed
                if len(cal_targets) > 1: # find closest calibrator source
                    dist = []
                    for cal_target in cal_targets:
                        ra2 = float(katpoint.rad2deg(cal_target.radec()[0])-katpoint.rad2deg(target.radec()[0]))*(katpoint.rad2deg(cal_target.radec()[0])-katpoint.rad2deg(target.radec()[0]))
                        dec2 = float(katpoint.rad2deg(cal_target.radec()[1])-katpoint.rad2deg(target.radec()[1]))*(katpoint.rad2deg(cal_target.radec()[1])-katpoint.rad2deg(target.radec()[1]))
                        dist.append(np.sqrt(ra2+dec2))
                    cal_target = cal_targets[np.argmin(dist)]
                else: cal_target = cal_targets[0]
                cal_target = cal_target.add_tags('gaincal')
                user_logger.info('Obtaining beamformer weights on calibrator source %r' %
                                 (cal_target.name))
                if not cal_target.flux_model:
                    raise ValueError("Calibrator '%s' has no flux density model" %
                                     (cal_target.description,))
                for beam in beams.keys(): obs_meta[beam]['cal_target'] = cal_target.description
                with start_session(kat, **vars(opts)) as session:
                    session.standard_setup(**vars(opts))
                    session.capture_start()
                    session.label('track')
                    session.track(cal_target, duration=opts.cal_duration)

                user_logger.info('Phasing up beamformer based on %d antennas' % (len(ants),))
                weights = get_weights(cbf)
                time.sleep(5)
                if not weights:
                    raise ValueError('No beamformer weights are available')
                for beam in beams.keys():
                    phase_up(cbf, weights, ants=[(ant.name + beams[beam]['pol']) for ant in ants],
                             bf=beam, phase_only=not opts.fix_amp)
                    time.sleep(1)

            else: # Use natural weighting
                for beam in beams.keys():
                    natural_weights(cbf, ants=[(ant.name + beams[beam]['pol']) for ant in ants], bf=beam)

        # Beamformer data capture
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

        # start remote receiver
        fbf_obj = BeamformerSession(cbf)
        user_logger.info('Initialising correlator receiver k7')
        fbf_obj.capture_start('k7')

        # Starting streams will issue metadata for capture
        # Allow long 10sec intervals to allow enough time to initiate data capture and to capture metadata
        # Else there will be collisions between the 2 beams
        for beam in beams.keys():
            user_logger.info('Initialising beamformer receiver for beam %s' % beam)
            # Initialise receiver and setup kat-dc2.karoo for output
            if beam == 'bf0': katcp_bfX = katcp_bf0
            elif beam == 'bf1': katcp_bfX = katcp_bf1
            else: raise RuntimeError('Unknown katcp client')
            if not katcp_bfX.rx_init(beams[beam]['data_drive'], opts.half_band, opts.transpose):
                raise RuntimeError('\nCould not initialise %s beamformer receiver.\n'%beam)
            # Start metadata receiver before starting data transmit
            katcp_bfX.rx_meta_init(beams[beam]['meta_port']) # port
            fbf_obj.capture_start(beam)
            user_logger.info('beamformer metadata')
            katcp_bfX.rx_meta(obs_meta[beam]) # additional obs related info
            user_logger.info('waiting 10s for metadata for stream %s to write'%beam)
            time.sleep(10)
            # Start transmitting data
            user_logger.info('beamformer data for beam %s'%beam)
            katcp_bfX.rx_beam(pol=beams[beam]['pol'], port=beams[beam]['data_port'])
            time.sleep(1)
        # Capture data
        user_logger.info('track target for %g seconds' % (opts.target_duration,))
        time.sleep(opts.target_duration)
        user_logger.info('target tracked for %g seconds' % (opts.target_duration,))
        # End all receivers
        for beam in beams.keys():
            user_logger.info('safely stopping receivers and tearing down beam %s' % beam)
            if beam == 'bf0': katcp_bfX = katcp_bf0
            elif beam == 'bf1': katcp_bfX = katcp_bf1
            katcp_bfX.rx_stop()
            time.sleep(5)

        # Stop all transmit
        fbf_obj.capture_stop('bf1')
        fbf_obj.capture_stop('bf0')
        fbf_obj.capture_stop('k7')


    # Closing and tidy up
    user_logger.info('Tidy up output for bf1')
    print katcp_bf1.rx_close()
    user_logger.info('Tidy up output for bf0')
    print katcp_bf0.rx_close()

# -fin-


