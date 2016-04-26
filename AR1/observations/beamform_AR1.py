#!/usr/bin/python
# Dual polarisation beamforming: Track target and possibly calibrator for beamforming.

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

    # We are only interested in first target
    user_logger.info('Looking up main beamformer target...')
    target = collect_targets(kat, args[:1]).targets[0]

    # Ensure that the target is up
    target_elevation = np.degrees(target.azel()[1])
    if target_elevation < opts.horizon:
        raise ValueError("The target %r is below the horizon" % (target.description,))

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
