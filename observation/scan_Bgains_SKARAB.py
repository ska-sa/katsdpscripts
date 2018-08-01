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
#parser.add_option('--gain', default='10,5000,500',
#                  help='Values of the correlator F-engine gain '
#                       'in the form "start,stop,number of steps" '
#                       '(default=%default)')
parser.add_option('--bgain', default='0.01,1,5',
                  help='Values of the B-engine gains '
                       'in the form "start,stop,number of steps" '
                       '(default=%default)')
parser.add_option('--fft-shift', type='int',
                  help='Set correlator F-engine FFT shift (default=leave as is)')

## This is the set of bgains I want to work through 2 at a time:
# [0.01 0.013 0.016 0.021 0.026 0.034
# 0.043 0.055 0.070 0.089 0.113 0.144
# 0.183 0.234 0.298 0.379 0.483 0.616
# 0.785 1.        ]

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track', nd_params='coupler,30,0,-1')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    args.append('SCP, radec, 0, -90') 

#g_start,g_end,g_step =np.array(opts.gain.split(',') ).astype(float)
b_start,b_end,b_step =np.array(opts.bgain.split(',') ).astype(float)

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
        #add similar RuntimeError for if B-Engine quants are to setable?    
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
            # for gain in np.logspace(np.log10(g_start),np.log10(g_end),g_step):
            # for bgain in np.linspace(b_start, b_end, b_step):
            # non_zero_bgains = np.linspace(b_start, b_end, b_step)
            # bgain_list = np.logspace(np.log10(b_start), np.log10(b_end), b_step)
            bgain_list = np.linspace(b_start, b_end, b_step)
            #zero_bgains = np.zeros(int(b_step))
            #bgain_list = np.ravel(np.column_stack((zero_bgains,non_zero_bgains)))
            # interleave the bgains with zeros to be able to see when the bgain
            # settings are changed. 
            for bgain in bgain_list:
                for target in targets.iterfilter(el_limit_deg=opts.horizon):
                    # Cut the track short if time runs out
                    #if bgain != 0.:
                    duration = opts.track_duration
                    #else:
                        #duration = 0.01
                        # set to small number to minimize nr zeros (sample rate=200000 per sec)
                    if opts.max_duration is not None:
                        time_left = opts.max_duration - (time.time() - start_time)
                        if time_left <= 0.:
                            user_logger.warning("Maximum duration of %g seconds "
                                                "has elapsed - stopping script",
                                                opts.max_duration)
                            keep_going = False
                            break
                        duration = min(duration, time_left)
                    # Set the gain to a single non complex number if needed
                    #session.label('track_gain,%g,%gi'%(gain.real,gain.imag))
                    session.label('track_bgain,%g'%bgain)
                    #for inp in session.cbf.fengine.inputs:
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
