#!/usr/bin/python
# Track target and calibrators for imaging.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement
import katpoint
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
parser.add_option('-i', '--bpcal_interval', type='float', default=None,
                  help='Interval between bandpass calibrator visits, in secs no pereodic (default="%default")')
parser.add_option('-g', '--gaincal_duration', type='int', default=1*60,
                  help='Duration to track gain calibrator per visit, in integer secs (default="%default")')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of script, in secs (default is no limit)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Imaging run.', nd_params='coupler,0,0,-1')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    if len(args)>0:
        args_target_list = []
        observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
        for catfile in args:
            try:
                observation_sources.add(file(catfile))
            except IOError: # If the file failed to load assume it is a target string
                args_target_list.append(catfile)
        num_catalogue_targets = len(observation_sources.targets)
        args_target_obj = []
        if len(args_target_list) > 0 :
            args_target_obj = lookup_targets(kat,args_target_list)
            observation_sources.add(args_target_obj)
        user_logger.info("Found %d targets from Command line and %d targets from %d Catalogue(s) " % (len(args_target_obj),num_catalogue_targets,len(args)-len(args_target_list),))

    user_logger.info("Imaging targets are [%s]" % (', '.join([("'%s'" % (target.name,)) for target  in observation_sources.filter(['~bpcal','~gaincal'])]),))
    user_logger.info("Bandpass calibrators are [%s]" % (', '.join([("'%s'" % (bpcal.name,)) for bpcal in observation_sources.filter('bpcal')]),))
    user_logger.info("Gain calibrators are [%s]" % (', '.join([("'%s'" % (gaincal.name,)) for gaincal in observation_sources.filter('gaincal')]),))
    time_lookup = {'gaincal':opts.gaincal_duration,'target':opts.target_duration,'bpcal':opts.bpcal_duration}
    def sources_up(sources):
        for target in sources:
            if session.target_visible(target, horizon=5.) : return True
        return False
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        if opts.bpcal_interval is not None :
            time_till_bpcal = opts.bpcal_interval
        else:
            time_till_bpcal = 0
        start_time = time.time()
        loop = True
        while loop or opts.max_duration and (time.time() < start_time + opts.max_duration) and sources_up(observation_sources):
            for current_source in observation_sources:
                if loop or opts.max_duration and (time.time() < start_time + opts.max_duration) and sources_up([current_source]):
                    if opts.bpcal_interval is not None and time_till_bpcal >= opts.bpcal_interval:
                        time_till_bpcal = 0.0
                        for  bpcal_source in observation_sources.filter('bpcal'):
                            session.track(bpcal_source, duration=time_lookup['bpcal'])
                            time_till_bpcal += opts.bpcal_duration
                    if  'bpcal' not in current_source.tags or opts.bpcal_interval is None:
                        track_duration = opts.target_duration
                        for tmp in current_source.tags:
                            time_lookup.get(tmp,track_duration)
                        session.track(current_source, duration=track_duration)
                        time_till_bpcal += track_duration
            loop = False
