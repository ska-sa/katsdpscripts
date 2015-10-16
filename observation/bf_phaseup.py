#!/usr/bin/python
# Track target(s) for a specified time.
# Also set the dbe7 gains before and after the track

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import ant_array,standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint
import pickle
import numpy as np
import StringIO
import logging 
#import fbf_katcp_wrapper as fbf



# read the bandpass solutions from a pickle file
raw_data,bpass_h,bpass_v=pickle.load(open('/home/kat/comm/scripts/bpass.pikl'))

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
#parser.add_option('--project-id',
#                  help='Project ID code the observation (**required**) This is a required option')
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum duration (which must be set for this)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
parser.add_option('--reset', action="store_true", default=False,
                          help='Reset the gains to 160.')
parser.add_option('--half-band', action='store_true', default=True,
                          help='Use only inner 50% of output band')
parser.add_option('--transpose', action='store_true', default=False,
                          help='Transpose time frequency blocks from correlator')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test',nd_params='off',project_id='COMMTEST',description='Phaseup observation setting f-engine weights',dump_rate=1.0)
# Parse the command line
opts, args = parser.parse_args()


# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    ants = kat.ants
    obs_ants = [ant.name for ant in ants]
    observation_sources = collect_targets(kat,args)
    # Find out what inputs are curremtly active
    reply = kat.dbe7.req.dbe_label_input()
    inputs = [m.arguments[0] for m in reply.messages[3:]]
    user_logger.info("Resetting f-engine gains to 160 to allow phasing up")
    for inp in inputs:
       kat.dbe7.req.dbe_k7_gain(inp,160)

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            if not opts.no_delays and not kat.dry_run :
                if session.dbe.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
                if session.dbe.req.auto_delay('off'):
                    user_logger.info("Turning off delay tracking.")
                else:
                    user_logger.error('Unable to turn off delay tracking.')
                if session.dbe.req.zero_delay():
                    user_logger.info("Zeroed the delay values.")
                else:
                    user_logger.error('Unable to zero delay values.')

            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    if target.flux_model is None:
                        user_logger.warning("Target has no flux model - stopping script")
                        keep_going=False
                        break
                    # observe the target for 60 seconds to determine the
                    # antenna gains
                    target.add_tags('bpcal')
                    session.label('track')
                    user_logger.info("Initiating %g-second track on target '%s'" % (60,target.name,))
                    session.track(target, duration=60, announce=False)
                    time.sleep(5)
                    # get and set the weights
                    for inp in inputs:
                        if inp[:-1] not in obs_ants : continue
                        if inp[-1] == 'v':
                            gains = bpass_v[inp[:-1]]
                        else:
                            gains = bpass_h[inp[:-1]]
                        gains = np.hstack((np.zeros(1),gains))
                        weights = getattr(kat.dbe7.sensor,'k7w_'+inp+'_gain_correction_per_channel').get_stored_history()[1][-1]
			# added print statement - weigths empty?
                        update = getattr(kat.dbe7.sensor,'k7w_'+inp+'_gain_correction_per_channel').get_stored_history()[0][-1]
                        print katpoint.Timestamp(update).local()
                        f = StringIO.StringIO(weights)
                        orig_weights = np.loadtxt(f, dtype=np.complex,delimiter=' ')
                        amp_weights = np.abs(orig_weights)
                        phase_weights = orig_weights / amp_weights
                        ind = np.repeat(False,1024)
                        # here is where you hack things to get "fuller" band
                        #ind[slice(10,1000)]=True # this will not work! but do not have time to repair that
                        ind[slice(200,800)]=True
                        gains[~ind] = 160.0
                        N = phase_weights[ind].shape[0]
                        z = np.polyfit(np.arange(N),np.unwrap(np.angle(phase_weights)[ind]),1)
                        #print z
                        phase = np.zeros(1024)
                        #phase[ind] = np.angle(phase_weights[ind]) 
                        phase[ind] = z[0]*np.arange(N)+z[1]
                        new_weights = (160.0 / gains ) * np.exp(1j * phase)
                        weights_str = ' '.join([('%+5.3f%+5.3fj' % (w.real,w.imag)) for w in new_weights])
                        kat.dbe7.req.dbe_k7_gain(inp,weights_str)
                        #because we are phasing in the f-engine set the b-engine weights to 1
                        bf_weights_str = ' '.join(1024 * ['1'])
                        for beam in ['bf0','bf1']:
                            kat.dbe7.req.dbe_k7_beam_weights(beam,inp,bf_weights_str)
                    user_logger.info("Initiating %g-second track on target '%s'" % (60,target.name,))
                    session.track(target, duration=60, announce=False)
                keep_going = False
            if opts.reset:
                user_logger.info("Resetting f-engine gains to 160")
                for inp in inputs:
                    kat.dbe7.req.dbe_k7_gain(inp,160)

