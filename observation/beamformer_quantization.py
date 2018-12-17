#!/usr/bin/env python
# Track target(s) for a specified time.

import time
import numpy as np
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger, SessionCBF)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track one or more sources for a specified time. At least one ' \
              'target must be specified. Note also some **required** options below.'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=45.0,
                  help='Length of time to track each source, in seconds '
                       '(default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which '
                       'script will end as soon as the current track finishes '
                       '(no limit by default)')
parser.add_option('--bgain', default=[0.01, 1, 5], type='float', nargs=3,
                  help='Values of the B-engine gains. Takes 3 arguments '
                       'in the form start_value stop_value number_of_values_in_this_range '
                       '(default=%default)')
parser.add_option('--fft-shift', type='int',
                  help='Set correlator F-engine FFT shift (default=leave as is)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track', nd_params='coupler,30,0,-1')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    args.append('SCP, radec, 0, -90')

b_start, b_end, b_step = opts.bgain[0], opts.bgain[1], int(opts.bgain[2])

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    cbf = SessionCBF(kat)
    targets = collect_targets(kat, args)
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(targets.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        if not kat.dry_run and not session.cbf.fengine.inputs:
            raise RuntimeError("Failed to get correlator input labels, "
                               "cannot set the F-engine gains")
        # Quit early if the B-Engine quants are not setable
        for stream in cbf.beamformers:
            if not stream.req.quant_gains:
                raise RuntimeError("Failed to set B-Engine quantisation. Quitting.")
        session.standard_setup(**vars(opts))
        if opts.fft_shift is not None:
            session.cbf.fengine.req.fft_shift(opts.fft_shift)
        session.capture_start()

        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            keep_going = opts.max_duration is not None
            targets_before_loop = len(targets_observed)
            # Iterate through source list, picking the next one that is up
            bgain_list = np.linspace(b_start, b_end, b_step)
            for bgain in bgain_list:
                for target in targets.iterfilter(el_limit_deg=opts.horizon):
                    duration = opts.track_duration
                    if opts.max_duration is not None:
                        time_left = opts.max_duration - (time.time() - start_time)
                        # Cut the track short if time runs out
                        if time_left <= 0.:
                            user_logger.warning("Maximum duration of %g seconds "
                                                "has elapsed - stopping script",
                                                opts.max_duration)
                            keep_going = False
                            break
                        duration = min(duration, time_left)
                    # Set the b-gain
                    session.label('track_bgain, %g' % bgain)
                    for stream in cbf.beamformers:
                        stream.req.quant_gains(bgain)
                        user_logger.info("B-engine %s quantisation gain set to %g",
                                         stream, bgain)
                    if session.track(target, duration=duration):
                        targets_observed.append(target.description)
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - "
                                        "stopping script instead of hanging around")
                    keep_going = False
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
