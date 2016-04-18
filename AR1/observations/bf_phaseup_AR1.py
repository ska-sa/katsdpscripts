#!/usr/bin/env python
#
# Track calibrator target for a specified time.
# Obtain calibrated gains and apply them to the F-engine afterwards.

import time
import pickle
import StringIO

import numpy as np
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)
import katpoint


# read the bandpass solutions from a pickle file
# raw_data,bpass_h,bpass_v=pickle.load(open('/home/kat/comm/scripts/bpass.pikl'))

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time and calibrate gains '
                                             'based on them. At least one target must be specified.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('--reset', action='store_true', default=False,
                  help='Reset the gains to the default value afterwards')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
parser.add_option('--default-gain', type='int', default=160,
                  help='Default correlator F-engine gain (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Phase-up observation that sets the F-engine weights')
# Parse the command line
opts, args = parser.parse_args()


# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    ants = kat.ants
    obs_ants = [ant.name for ant in ants]
    observation_sources = collect_targets(kat, args)
    # Find out which inputs are currently active
    reply = kat.data.req.cbf_label_inputs()
    inputs = [m.arguments[0] for m in reply.messages[3:]]

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
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

            user_logger.info("Resetting F-engine gains to %g to allow phasing up" % (opts.default_gain,))
            for inp in inputs:
                kat.data.req.cbf_gain(inp, opts.default_gain)
                #because we are phasing in the f-engine set the b-engine weights to 1
                bf_weights_str = ' '.join(1024 * ['1'])
                if inp[-1] == 'h':
                    kat.data.req.cbf_beam_weights('beam_0x', inp, bf_weights_str)
                else:
                    kat.data.req.cbf_beam_weights('beam_0y', inp, bf_weights_str)

            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                if target.flux_model is None:
                    user_logger.warning("Target has no flux model - stopping script")
                    break
                target.add_tags('bpcal')
                session.label('track')
                user_logger.info("Initiating %g-second track on target '%s'" %
                                 (opts.track_duration, target.name,))
                session.track(target, duration=opts.track_duration + 5, announce=False)
                # get and set the weights
                for inp in inputs:
                    ant, pol = inp[:-1], inp[-1]
                    if ant not in obs_ants:
                        continue
                    # if pol == 'v':
                    #     gains = bpass_v[inp[:-1]]
                    # else:
                    #     gains = bpass_h[inp[:-1]]
                    # # gains = np.hstack((np.zeros(1), gains))
                    # gains = np.r_[0.0, gains]
                    weights = getattr(kat.data.sensor,'k7w_'+inp+'_gain_correction_per_channel').get_reading().value
                    update = getattr(kat.data.sensor,'k7w_'+inp+'_gain_correction_per_channel').get_reading().timestamp
                    user_logger.info("Gain sensors updated at %s"%katpoint.Timestamp(update).local())
                    # f = StringIO.StringIO(weights)
                    # orig_weights = np.loadtxt(f, dtype=np.complex,delimiter=' ')
                    # amp_weights = np.abs(orig_weights)
                    # phase_weights = orig_weights / amp_weights
                    # ind = np.repeat(False,1024)
                    # # here is where you hack things to get "fuller" band
                    # #ind[slice(10,1000)]=True # this will not work! but do not have time to repair that
                    # ind[slice(200,800)]=True
                    # gains[~ind] = 160.0
                    # N = phase_weights[ind].shape[0]
                    # z = np.polyfit(np.arange(N),np.unwrap(np.angle(phase_weights)[ind]),1)
                    # #print z
                    # phase = np.zeros(1024)
                    # #phase[ind] = np.angle(phase_weights[ind])
                    # phase[ind] = z[0]*np.arange(N)+z[1]
                    # new_weights = (160.0 / gains ) * np.exp(1j * phase)
                    weights_str = ' '.join([('%+5.3f%+5.3fj' % (w.real,w.imag)) for w in new_weights])
                    kat.data.req.cbf_gain(inp, weights_str)
            if opts.reset:
                user_logger.info("Resetting F-engine gains to %g" % (opts.default_gain,))
                for inp in inputs:
                    kat.data.req.cbf_gain(inp, opts.default_gain)
