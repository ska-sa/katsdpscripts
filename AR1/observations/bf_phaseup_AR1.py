#!/usr/bin/env python
#
# Track calibrator target for a specified time.
# Obtain calibrated gains and apply them to the F-engine afterwards.

import time

import numpy as np

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)
from katsdptelstate import TelescopeState


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


class NoGainsAvailableError(Exception):
    """No gain solutions are available from the cal pipeline."""


def get_cbf_inputs(data):
    """Input labels associated with correlator."""
    reply = data.req.cbf_input_labels()
    return reply.messages[0].arguments[1:] if reply.succeeded else []


def get_telstate(data, sub):
    """Get TelescopeState object associated with current data product."""
    subarray_product = 'array_%s_%s' % (sub.sensor.sub_nr.get_value(),
                                        sub.sensor.product.get_value())
    reply = data.req.spmc_telstate_endpoint(subarray_product)
    return TelescopeState(reply.messages[0].arguments[1]) if reply.succeeded else {}


def get_delaycal_solutions(telstate):
    """Retrieve delay calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_K' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant + pol for pol in 'hv' for ant in ants]
    solutions = telstate['cal_product_K']
    return dict(zip(inputs, solutions.real.flat))


def get_bpcal_solutions(telstate):
    """Retrieve bandpass calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_B' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant + pol for pol in 'hv' for ant in ants]
    solutions = telstate['cal_product_B']
    return dict(zip(inputs, solutions.reshape((solutions.shape[0], -1)).T))


def get_gaincal_solutions(telstate):
    """Retrieve gain calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_G' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant + pol for pol in 'hv' for ant in ants]
    solutions = telstate['cal_product_G']
    return dict(zip(inputs, solutions.flat))


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track one or more sources for a specified time and calibrate ' \
              'gains based on them. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('--reset', action='store_true', default=False,
                  help='Reset the gains to the default value afterwards')
parser.add_option('--default-gain', type='int', default=200,
                  help='Default correlator F-engine gain (default=%default)')
parser.add_option('--fft-shift', type='int',
	          help='Set correlator F-engine FFT shift (default=leave as is)')
parser.add_option('--reconfigure-sdp', action="store_true", default=False,
                  help='Reconfigure SDP subsystem at the start to clear crashed containers')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Phase-up observation that sets the F-engine weights')
# Parse the command line
opts, args = parser.parse_args()

# Very bad hack to circumvent SB verification issues
# with anything other than session objects (e.g. kat.data).
# The *near future* will be modelled CBF sessions.
# The *distant future* will be fully simulated sessions via kattelmod.
if opts.dry_run:
    import sys
    sys.exit(0)

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        raise NoTargetsUpError("No targets are currently visible - please re-run the script later")
    if opts.reconfigure_sdp:
        sub, data = kat.sub, kat.data
        subarray_product, streams = data.sensor.stream_addresses.get_value().split(' ')
        product = subarray_product.split('_')[-1]
        resources = sub.sensor.pool_resources.get_value().split(',')
        receptors = ','.join([res for res in resources if not res.startswith('data_')])
        dump_rate = sub.sensor.dump_rate.get_value()
        channels = 32768 if product.endswith('32k') else 4096
        beams = 1 if product.startswith('b') else 0
        user_logger.info("Deconfiguring SDP subsystem for subarray product %r" %
                         (subarray_product,))
        data.req.spmc_data_product_configure(subarray_product, 0, timeout=30)
        user_logger.info("Reconfiguring SDP subsystem")
        data.req.spmc_data_product_configure(subarray_product, receptors, channels,
                                             dump_rate, beams, streams, timeout=200)
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        inputs = get_cbf_inputs(session.data)
        # Assume correlator stream is bc product name without the 'b'
        product = kat.sub.sensor.product.get_value()
        corr_stream = product if product.startswith('c') else product[1:]
        if opts.fft_shift is not None:
            session.data.req.cbf_fft_shift(opts.fft_shift)
        session.data.req.capture_start(corr_stream)

        for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
            if target.flux_model is None:
                user_logger.warning("Target has no flux model (katsdpcal will need it in future)")
            user_logger.info("Resetting F-engine gains to %g to allow phasing up"
                             % (opts.default_gain,))
            for inp in inputs:
                session.data.req.cbf_gain(inp, opts.default_gain)
            session.label('un_corrected')
            user_logger.info("Initiating %g-second track on target '%s'" %
                             (opts.track_duration, target.name,))
            session.track(target, duration=opts.track_duration, announce=False)
            # Attempt to jiggle cal pipeline to drop its gains
            session.ants.req.target('')
            user_logger.info("Waiting for gains to materialise in cal pipeline")
            time.sleep(180)
            telstate = get_telstate(session.data, kat.sub)
            delays = get_delaycal_solutions(telstate)
            bp_gains = get_bpcal_solutions(telstate)
            gains = get_gaincal_solutions(telstate)
            if not gains:
                raise NoGainsAvailableError("No gain solutions found in telstate %r" % (telstate,))
            user_logger.info("Setting F-engine gains to phase up antennas")
	    session.label('corrected')
            for inp in set(inputs) and set(gains):
                orig_weights = gains[inp]
                if inp in bp_gains:
                    orig_weights *= bp_gains[inp]
                if inp in delays:
                    # XXX Hacky hack
                    centre_freq = 1284e6
                    num_chans = 32768 if product.endswith('32k') else 4096
                    sideband = 1
                    channel_width = 856e6 / num_chans
                    channel_freqs = centre_freq + sideband * channel_width * (np.arange(num_chans) - num_chans / 2)
                    delay_weights = np.exp(2.0j * np.pi * delays[inp] * channel_freqs)
                    # Guess which direction to apply delays as katcal has a bug here
                    orig_weights *= delay_weights
                amp_weights = np.abs(orig_weights)
                phase_weights = orig_weights / amp_weights
                # Cop out on the gain amplitude but at least correct the phase
                new_weights = opts.default_gain * phase_weights.conj()
                weights_str = [('%+5.3f%+5.3fj' % (w.real, w.imag)) for w in new_weights]
                session.data.req.cbf_gain(inp, *weights_str)
            user_logger.info("Revisiting target %r for %g seconds to see if phasing worked" %
                             (target.name, opts.track_duration))
            session.track(target, duration=opts.track_duration, announce=False)
        if opts.reset:
            user_logger.info("Resetting F-engine gains to %g" % (opts.default_gain,))
            for inp in inputs:
                session.data.req.cbf_gain(inp, opts.default_gain)
