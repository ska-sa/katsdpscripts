#!/usr/bin/env python
# Dual polarisation beamforming: Track target for beamforming.

import argparse

import numpy as np
import katpoint
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                SessionCBF, SessionSDP)


def verify_digifits_backend_args(backend_args):
    parser = argparse.ArgumentParser(description='Grab arguments')
    parser.add_argument('-t', type=float, help='integration time (s) per output sample (default=64mus)')
    parser.add_argument('-overlap', action='store_true', help='disable input buffering')
    parser.add_argument('-header', help='command line arguments are header values (not filenames)')
    parser.add_argument('-S', type=int, help='start processing at t=seek seconds')
    parser.add_argument('-T', type=int, help='process only t=total seconds')
    parser.add_argument('-set', help='key=value set observation attributes')
    parser.add_argument('-r', action='store_true', help='report time spent performing each operation')
    parser.add_argument('-dump', action='store_true', help='dump time series before performing operation')
    parser.add_argument('-D', type=float, help='set the dispersion measure')
    parser.add_argument('-do_dedisp', action='store_true', help='enable coherent dedispersion (default: false)')
    parser.add_argument('-c', action='store_true', help='keep offset and scale constant')
    parser.add_argument('-I', type=int, help='rescale interval in seconds')
    parser.add_argument('-p', type=int, choices=[1, 2, 4],
                        help='output 1 (Intensity), 2 (AABB), or 4 (Coherence) products')
    parser.add_argument('-b', type=int, choices=[1, 2, 4, 8], help='number of bits per sample output to file [1,2,4,8]')
    parser.add_argument('-F', type=int, help='nchan[:D] * create a filterbank (voltages only)')
    parser.add_argument('-nsblk', type=int, help='output block size in samples (default=2048)')
    parser.add_argument('-k', action='store_true', help='remove inter-channel dispersion delays')
    parser.parse_args(backend_args.split(" "))


def verify_dspsr_backend_args(backend_args):
    parser = argparse.ArgumentParser(description='Grab arguments')
    parser.add_argument('-overlap', action='store_true', help='disable input buffering')
    parser.add_argument('-header', help='command line arguments are header values (not filenames)')
    parser.add_argument('-S', type=int, help='start processing at t=seek seconds')
    parser.add_argument('-T', type=int, help='process only t=total seconds')
    parser.add_argument('-set', help='key=value     set observation attributes')
    parser.add_argument('-W', action='store_true', help='disable weights (allow bad data)')
    parser.add_argument('-r', action='store_true', help='report time spent performing each operation')
    parser.add_argument('-B', type=float, help='set the bandwidth in MHz')
    parser.add_argument('-f', type=float, help='set the centre frequency in MHz')
    parser.add_argument('-k', help='set the telescope name')
    parser.add_argument('-N', help='set the source name')
    parser.add_argument('-C', type=float, help='adjust clock byset the source name')
    parser.add_argument('-m', help='set the start MJD of the observation')
    parser.add_argument('-2', dest='two', action='store_true', help='unpacker options ("2-bit" excision)')
    parser.add_argument('-skz', action='store_true', help='apply spectral kurtosis filterbank RFI zapping')
    parser.add_argument('-noskz_too', action='store_true', help='also produce un-zapped version of output')
    parser.add_argument('-skzm', type=int, help='samples to integrate for spectral kurtosis statistics')
    parser.add_argument('-skzs', type=int, help='number of std deviations to use for spectral kurtosis excisions')
    parser.add_argument('-skz_start', type=int, help='first channel where signal is expected')
    parser.add_argument('-skz_end', type=int, help='last channel where signal is expected')
    parser.add_argument('-skz_no_fscr', action='store_true', help=' do not use SKDetector Fscrunch feature')
    parser.add_argument('-skz_no_tscr', action='store_true', help='do not use SKDetector Tscrunch feature')
    parser.add_argument('-skz_no_ft', action='store_true', help='do not use SKDetector despeckeler')
    parser.add_argument('-sk_fold', action='store_true', help='fold the SKFilterbank output')
    parser.add_argument('-F', help='<N>[:D] * create an N-channel filterbank')
    parser.add_argument('-G', type=int, help='nbin create phase-locked filterbank')
    parser.add_argument('-cyclic', type=int, help='form cyclic spectra with N channels (per input channel)')
    parser.add_argument('-cyclicoversample', type=int,
                        help='use M times as many lags to improve cyclic channel isolation (4 is recommended)')
    parser.add_argument('-D', type=float, help='over-ride dispersion measure')
    parser.add_argument('-K', type=float, help='remove inter-channel dispersion delays')
    parser.add_argument('-d', type=int, choices=[1, 2, 3, 4], help='1=PP+QQ, 2=PP,QQ, 3=(PP+QQ)^2 4=PP,QQ,PQ,QP')
    parser.add_argument('-n', action='store_true', help='[experimental] ndim of output when npol=4')
    parser.add_argument('-4', dest='four', action='store_true', help='compute fourth-order moments')
    parser.add_argument('-b', type=int, help='number of phase bins in folded profile')
    parser.add_argument('-c', type=float, help='folding period (in seconds)')
    parser.add_argument('-cepoch', help='MJD reference epoch for phase=0 (when -c is used)')
    parser.add_argument('-p', type=float, help='reference phase of rising edge of bin zero')
    parser.add_argument('-E', help='pulsar ephemeris used to generate predictor')
    parser.add_argument('-P', help='phase predictor used for folding')
    parser.add_argument('-X', help='additional pulsar to be folded')
    parser.add_argument('-asynch-fold', action='store_true', help='fold on CPU while processing on GPU')
    parser.add_argument('-A', action='store_true', help='output single archive with multiple integrations')
    parser.add_argument('-nsub', type=int, help='output archives with N integrations each')
    parser.add_argument('-s', action='store_true', help='create single pulse sub-integrations')
    parser.add_argument('-turns', type=int, help='create integrations of specified number of spin periods')
    parser.add_argument('-L', type=float, help='create integrations of specified duration')
    parser.add_argument('-Lepoch', help='start time of first sub-integration (when -L is used)')
    parser.add_argument('-Lmin', type=float, help='minimum integration length output')
    parser.add_argument('-y', action='store_true', help='output partially completed integrations')
    parser.parse_args(backend_args.split(" "))


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
parser.add_option('-B', '--beam-bandwidth', type='float', default=853.75,
                  help="Beamformer bandwidth, in MHz (default=%default)")
parser.add_option('-F', '--beam-centre-freq', type='float', default=1284.0,
                  help="Beamformer centre frequency, in MHz (default=%default)")
parser.add_option('--test-snr', action='store_true', default=False,
                  help="Perform SNR test by switching off inputs (default=no)")
parser.add_option('--backend', type='choice', default='digifits',
                  choices=['digifits', 'dspsr', 'dada_dbdisk',
                           'digifits profile', 'dspsr profile'],
                  help="Choose backend (default=%default)")
parser.add_option('--backend-args',
                  help="Arguments for backend processing")
parser.add_option('--drift-scan', action='store_true', default=False,
                  help="Perform drift scan instead of standard track (default=no)")
parser.add_option('--noise-source', type=str, default=None,
                  help="Initiate a noise diode pattern on all antennas, '<cycle_length_sec>,<on_fraction>'")
nd_cycles = ['all', 'cycle']
parser.add_option('--noise-cycle', type=str, default=None,
                  help="How to apply the noise diode pattern: \
'%s' to set the pattern to all dishes simultaneously (default), \
'%s' to set the pattern so loop through the antennas in some fashion, \
'm0xx' to set the pattern to a single selected antenna." % (nd_cycles[0], nd_cycles[1]))
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Beamformer observation', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments and connect to KAT proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target")

with verify_and_connect(opts) as kat:
    bf_ants = opts.ants.split(',') if opts.ants else [ant.name for ant in kat.ants]
    cbf = SessionCBF(kat)
    for stream in cbf.beamformers:
        # ROACH implementation only, need to set passband and centerfrequency
        # Values hard coded: CBF is expecting values
        # (856000000.00 - 2250000.0, 1284000000.000)
        # Hardcoded and magic number usage to be removed in future update
        reply = stream.req.passband(int(opts.beam_bandwidth * 1e6),
                                    int(opts.beam_centre_freq * 1e6))
        if reply.succeeded:
            try:
                actual_bandwidth = float(reply.messages[0].arguments[2])
                actual_centre_freq = float(reply.messages[0].arguments[3])
            except IndexError:
                # In a dry run the reply will succeed but with no return values
                actual_bandwidth = actual_centre_freq = 0.0
            user_logger.info("Beamformer %r has bandwidth %g Hz and centre freq %g Hz",
                             stream.name, actual_bandwidth, actual_centre_freq)
        else:
            # SKARAB no longer has bandwidth and centre freq registers
            if str(reply.messages[0]).find('AttributeError'):
                # will be replaced with a "unimplemented" response
                user_logger.info("SKARAB, skipping bandwidth and centre freq setting")
            else:  # ROACH setting error
                raise ValueError("Could not set beamformer %r passband - (%s)" %
                                 (stream.name, ' '.join(reply.messages[0].arguments)))
        user_logger.info('Setting beamformer weights for stream %r:', stream.name)
        for inp in stream.inputs:
            weight = 1.0 / np.sqrt(len(bf_ants)) if inp[:-1] in bf_ants else 0.0
            reply = stream.req.weights(inp, weight)
            if reply.succeeded:
                user_logger.info('  input %r got weight %f', inp, weight)
            else:
                user_logger.warning('  input %r weight could not be set', inp)

    # We are only interested in first target
    user_logger.info('Looking up main beamformer target...')
    tgt_with_spaces = [tgt for tgt in args[:1] if len(tgt.strip().split()) > 1]
    if len(tgt_with_spaces) > 0:
        user_logger.error("Please replace '%s' with '%s'",
                          '& '.join(tgt_with_spaces),
                          '& '.join([''.join(tgt.strip().split())
                                     for tgt in tgt_with_spaces]))
        raise ValueError('Found spaces in target names, which will cause an error in digifits')

    target = collect_targets(kat, args[:1]).targets[0]

    # Verify backend_args
    if opts.backend.split(' ')[0] == "dspsr" and opts.backend_args:
        verify_dspsr_backend_args(opts.backend_args)
    elif opts.backend.split(' ')[0] == "digifits" and opts.backend_args:
        verify_digifits_backend_args(opts.backend_args)

    # Save script parameters before session capture-init's the SDP subsystem
    sdp = SessionSDP(kat)
    if not sdp.telstate:
        raise ValueError("Could not connect to telstate on " + str(sdp))
    script_args = vars(opts)
    script_args['targets'] = args
    sdp.telstate.add('obs_script_arguments', script_args)

    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Ensure that the target is up
        target_elevation = np.degrees(target.azel()[1])
        if target_elevation < opts.horizon:
            raise ValueError("The target %r is below the horizon" % (target.description,))
        # Force delay tracking to be on
        opts.no_delays = False
        session.standard_setup(**vars(opts))
        session.capture_init()

        user_logger.info('Set noise-source pattern')
        if opts.noise_source is not None:
            import time
            cycle_length, on_fraction = np.array([el.strip() for el in opts.noise_source.split(',')],
                                                 dtype=float)
            user_logger.info('Setting noise source pattern to %.3f [sec], %.3f fraction on',
                             cycle_length, on_fraction)
            if opts.noise_cycle is None or opts.noise_cycle == 'all':
                # Noise Diodes are triggered on all antennas in array simultaneously
                timestamp = time.time() + 1  # add a second to ensure all digitisers set at the same time
                user_logger.info('Set all noise diode with timestamp %d (%s)',
                                 int(timestamp), time.ctime(timestamp))
                kat.ants.req.dig_noise_source(timestamp, on_fraction, cycle_length)
            elif opts.noise_cycle in bf_ants:
                # Noise Diodes are triggered for only one antenna in the array
                ant_name = opts.noise_cycle.strip()
                user_logger.info('Set noise diode for antenna %s', ant_name)
                ped = getattr(kat, ant_name)
                ped.req.dig_noise_source('now', on_fraction, cycle_length)
            elif opts.noise_cycle == 'cycle':
                timestamp = time.time() + 1  # add a second to ensure all digitisers set at the same time
                for ant in bf_ants:
                    user_logger.info('Set noise diode for antenna %s with timestamp %f',
                                     ant, timestamp)
                    ped = getattr(kat, ant)
                    ped.req.dig_noise_source(timestamp, on_fraction, cycle_length)
                    timestamp += cycle_length * on_fraction
            else:
                raise ValueError("Unknown ND cycle option, please select: %s or any one of %s" %
                                 (', '.join(nd_cycles), ', '.join(bf_ants)))

        # Get onto beamformer target
        session.track(target, duration=0)
        # Perform a drift scan if selected
        if opts.drift_scan:
            transit_time = katpoint.Timestamp() + opts.target_duration / 2.0
            # Stationary transit point becomes new target
            az, el = target.azel(timestamp=transit_time)
            target = katpoint.construct_azel_target(katpoint.wrap_angle(az), el)
            # Go to transit point so long
            session.track(target, duration=0)
        # Only start capturing once we are on target
        session.capture_start()
        if not opts.test_snr:
            # Basic observation
            session.label('track')
            session.track(target, duration=opts.target_duration)
        else:
            duration_per_slot = opts.target_duration / (len(bf_ants) + 1)
            session.label('snr_all_ants')
            session.track(target, duration=duration_per_slot)
            # Perform SNR test by cycling through all inputs to the beamformer
            for n, ant in enumerate(bf_ants):
                # Switch on selected antenna only
                for stream in cbf.beamformers:
                    for inp in stream.inputs:
                        weight = 1.0 if inp[:-1] == ant else 0.0
                        stream.req.weights(inp, weight)
                session.label('snr_' + ant)
                session.track(target, duration=duration_per_slot)
        # XXX Clear the script arguments in telstate as these may inadvertently
        # start an unwanted backend on a future SDP capture-init in the same
        # subarray product (since it keeps the same telstate). A more elegant
        # solution needed? See Jira ticket SR-822.
        sdp.telstate.add('obs_script_arguments', {})

    # switch noise-source pattern off
    if opts.noise_source is not None:
        user_logger.info('Ending noise source pattern')
        kat.ants.req.dig_noise_source('now', 0)
