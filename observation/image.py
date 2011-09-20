#!/usr/bin/python
# Track target and calibrators for imaging.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, start_session, user_logger

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target'> <'bandpass calibrator'> <'gain calibrator'> [<extra gain cals>*]",
                                 description="Perform an imaging run of a specified target, visiting the bandpass " +
                                             "and gain calibrators along the way.")
# Add experiment-specific options
parser.add_option('-t', '--target_duration', type='int', default=5*60,
                  help='Duration to track the imaging target per visit, in integer secs (default="%default")')
parser.add_option('-b', '--bpcal_duration', type='int', default=5*60,
                  help='Duration to track bandpass calibrator per visit, in integer secs (default="%default")')
parser.add_option('-i', '--bpcal_interval', type='int', default=60*60,
                  help='Interval between bandpass calibrator visits, in integer secs (default="%default")')
parser.add_option('-g', '--gaincal_duration', type='int', default=1*60,
                  help='Duration to track gain calibrator per visit, in integer secs (default="%default")')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of script, in secs (default is no limit)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Imaging run.', nd_params='coupler,0,0,-1')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) < 3:
    raise ValueError("Please specify the target, bandpass calibrator and at least one gain calibrator")

with verify_and_connect(opts) as kat:

    targets = lookup_targets(kat, args)
    target, bpcal, gaincals = targets[0], targets[1], targets[2:]
    user_logger.info("Imaging target is '%s'" % (target,))
    user_logger.info("Bandpass calibrator is '%s'" % (bpcal,))
    user_logger.info("Gain calibrators are [%s]" % (', '.join([("'%s'" % (gaincal,)) for gaincal in gaincals]),))
    
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        time_till_bpcal = opts.bpcal_interval
        start_time = time.time()

        while True:
            if time_till_bpcal >= opts.bpcal_interval:
                session.track(bpcal, duration=opts.bpcal_duration, label='bandpass_cal')
                time_till_bpcal = opts.bpcal_duration
            for gaincal in gaincals:
                session.track(gaincal, duration=opts.gaincal_duration, label='gain_cal')
                time_till_bpcal += opts.gaincal_duration

            if opts.max_duration and (time.time() > start_time + opts.max_duration):
                user_logger.warning("Stopping experiment, as maximum script duration was exceeded")
                break

            if not session.target_visible(target, opts.target_duration, horizon=5.) or \
               not session.target_visible(gaincal, opts.target_duration + len(gaincals) * opts.gaincal_duration, horizon=5.):
                user_logger.warning("Stopping experiment, as imaging target or following gain cal(s) will be below horizon")
                break

            session.track(target, duration=opts.target_duration, label='target')
            time_till_bpcal += opts.target_duration
