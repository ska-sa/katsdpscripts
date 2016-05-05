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


def get_bpcal_solutions(telstate):
    """Retrieve bandpass calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_B' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant+pol for pol in ('h', 'v') for ant in ants]
    solutions = telstate['cal_product_B']
    return dict(zip(inputs, solutions.reshape((solutions.shape[0], -1)).T))


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
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
parser.add_option('--default-gain', type='int', default=200,
                  help='Default correlator F-engine gain (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Phase-up observation that sets the F-engine weights')
# Parse the command line
opts, args = parser.parse_args()


# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        raise NoTargetsUpError("No targets are currently visible - please re-run the script later")
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        if not opts.no_delays and not kat.dry_run:
            if session.data.req.auto_delay('on'):
                user_logger.info("Turning on delay tracking.")
            else:
                user_logger.error('Unable to turn on delay tracking.')
        elif opts.no_delays and not kat.dry_run:
            if session.data.req.auto_delay('off'):
                user_logger.info("Turning off delay tracking.")
            else:
                user_logger.error('Unable to turn off delay tracking.')

        session.standard_setup(**vars(opts))
        inputs = get_cbf_inputs(session.data)
        # Assume correlator stream is bc product name without the 'b'
        session.data.req.capture_start(opts.product[1:])
#        session.data.req.cbf_capture_meta(opts.product[1:])

        for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
            if target.flux_model is None:
                user_logger.warning("Target has no flux model (katsdpcal will need it in future)")
            user_logger.info("Resetting F-engine gains to %g to allow phasing up"
                             % (opts.default_gain,))
            for inp in inputs:
                session.data.req.cbf_gain(inp, opts.default_gain)
            target.add_tags('bpcal')
            session.label('track')
            user_logger.info("Initiating %g-second track on target '%s'" %
                             (opts.track_duration, target.name,))
            session.track(target, duration=opts.track_duration, announce=False)
            # Attempt to jiggle cal pipeline to drop its gains
            session.ants.req.target('')
            user_logger.info("Waiting for gains to materialise in cal pipeline")
            time.sleep(10)
            telstate = get_telstate(session.data, kat.sub)
            gains = get_bpcal_solutions(telstate)
            if not gains:
                raise NoGainsAvailableError("No bpcal gain solutions found in telstate %r" % (telstate,))
            user_logger.info("Setting F-engine gains to phase up antennas")
            for inp in set(inputs) and set(gains):
                orig_weights = gains[inp]
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
