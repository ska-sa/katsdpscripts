#!/usr/bin/python
# Track target and calibrator(s) for imaging.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, \
                             user_logger, CaptureSession, TimeSession
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target'> <'calsource'> [<'calsource'>*]",
                                 description="Perform an imaging run of a specified target. " +
                                             "A target and one or more cal sources are specified.")
# Add experiment-specific options
parser.add_option('-t', '--target_duration', type='int', default=5*60,
                  help='Duration to track the imaging target per visit, in integer secs (default="%default")')
parser.add_option('-c', '--cal_duration', type='int', default=1*60,
                  help='Duration to track each cal source per visit, in integer secs (default="%default")')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Imaging run.')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) < 2:
    raise ValueError("Please specify at least a target and one cal source")

with verify_and_connect(opts) as kat:

    targets = lookup_targets(kat, args)
    target = targets[0]
    target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
    calibrators = targets[1:]
    user_logger.info("Imaging target is '%s'" % (target.description,))
    user_logger.info("Calibrator(s) are %s" % (calibrators,))
    full_duration = opts.target_duration + len(calibrators) * opts.cal_duration

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))

        while True:
            # Visit each calibrator
            for cal in calibrators:
                session.track(cal, duration=opts.cal_duration, label='cal_source')

            user_logger.info("Initiating %g-second track on target '%s'" % (opts.target_duration, target.name))
            if not session.target_visible(target, full_duration, horizon=10.):
                user_logger.warning("Stopping experiment, as imaging target will be below horizon")
                break

            # Split the total track on one target into segments lasting as long as the noise diode period
            # This ensures the maximum number of noise diode firings
            total_track_time = 0.
            while total_track_time < opts.target_duration:
                next_track = opts.target_duration - total_track_time
                if opts.nd_params['period'] > 0:
                    next_track = min(next_track, opts.nd_params['period'])
                session.track(target, duration=next_track, label='', announce=False)
                total_track_time += next_track
