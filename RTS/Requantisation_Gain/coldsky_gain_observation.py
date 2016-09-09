#!/usr/bin/python
# Capture SCP auto-correlation data

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import numpy, time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint
import time

# code snippet stolen from Sean's dbe_gain_track.py script
def set_gains(kat,value):
    ants = kat.ants
    for ant in ants:
        for pol in ['h','v']:
            user_logger.info("Setting gain %d to antenna %s" % (int(value), '%s%s'%(ant.name,pol)))
            kat.data_rts.req.cbf_gain('%s%s'%(ant.name,pol), int(value))
	    time.sleep(1)


if __name__ == '__main__':

    # Set up standard script options
    usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
    description="Perform an imaging run of SCP setting requantisation gain values"
    parser = standard_script_options(usage=usage,
                                     description=description)
    # Add experiment-specific options
    parser.add_option('-t', '--target-duration', type='float', default=30,
                      help='Minimum duration to track the imaging target per visit, in seconds (default="%default")')
    parser.add_option('--step', type='int', default=1,
                      help='Integer increment size over gain range (default=%default)')
    parser.add_option('--min-gain', type='int', default=1,
                      help='Integer minimum requantisation gain setting (default=%default)')
    parser.add_option('--max-gain', type='int', default=300,
                      help='Integer maximum requantisation gain setting (default=%default)')

    # Set default value for any option (both standard and experiment-specific options)
    parser.set_defaults(description='Requantisation Gain Evaluation', nd_params='coupler,0,0,-1',dump_rate=0.1)
    # Parse the command line
    opts, args = parser.parse_args()

    # Check options and arguments, and build KAT configuration, connecting to proxies and devices
    if len(args) == 0:
        raise ValueError("Please specify the target(s) and calibrator(s) to observe as arguments, either as "
                         "description strings or catalogue filenames")
    with verify_and_connect(opts) as kat:
        if not kat.dry_run and kat.ants.req.mode('STOP') :
            user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
            time.sleep(3)
        else:
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

        sources = collect_targets(kat, args)
        user_logger.info("Imaging targets are [%s]" %
                         (', '.join([("'%s'" % (target.name,)) for target in sources]),))
        kat.ants.req.mode('STOP')
        time.sleep(3)
        with start_session(kat, **vars(opts)) as session:
            # Start capture session, which creates HDF5 file
            session.standard_setup(**vars(opts))
            session.capture_start()

            for gain in range(opts.min_gain, opts.max_gain, opts.step):
                try:
                    if not opts.dry_run: set_gains(kat, int(gain))
                except Exception,  e: print e
                session.label('%s' % gain)
                user_logger.info("Set digital gain on selected DBE to %d." % gain)
                # Loop over sources in sequence
                for target in sources:
                    target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
                    user_logger.info("Initiating %g-second track on target '%s'" % (opts.target_duration, target.name))
                    # Set the default track duration for a target with no recognised tags
                    session.label('track')
                    session.track(target, duration=opts.target_duration)
	            time.sleep(3)	

# - fin -
