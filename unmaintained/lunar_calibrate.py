#!/usr/bin/env python
# Track target and calibrators for imaging.

import time
from collections import defaultdict
import katpoint
from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)


# Set up standard script options
description = "Perform an imaging run of a specified target, visiting the " \
              "bandpass and gain calibrators along the way. The calibrators " \
              "are identified by tags in their description strings ('bpcal' " \
              "and 'gaincal', respectively), while the imaging targets may " \
              "optionally have a tag of 'target'."
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description=description)
# Add experiment-specific options
parser.add_option('-t', '--target-duration', type='float', default=300,
                  help='Minimum duration to track the imaging target per visit, in seconds '
                       '(default="%default")')
parser.add_option('-b', '--bpcal-duration', type='float', default=300,
                  help='Minimum duration to track bandpass calibrator per visit, in seconds '
                       '(default="%default")')
parser.add_option('-i', '--bpcal-interval', type='float',
                  help='Minimum interval between bandpass calibrator visits, in seconds '
                       '(visits each source in turn by default)')
parser.add_option('--lunar-interval', type='float',default=1800
                  help='Minimum interval between Lunar calibrator visits, in seconds '
                       '(default="%default")')
parser.add_option('-g', '--gaincal-duration', type='float', default=60,
                  help='Minimum duration to track gain calibrator per visit, in seconds '
                       '(default="%default")')
parser.add_option('-m', '--max-duration', type='float',
                  help='Maximum duration of script, in seconds '
                       '(the default is to keep observing until all sources have set)')
parser.set_defaults(description='Imaging run', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target(s) and calibrator(s) to observe as arguments, either as "
                     "description strings or catalogue filenames")

with verify_and_connect(opts) as kat:
    sources = collect_targets(kat, args)

    user_logger.info("Imaging targets are [%s]",
                     ', '.join([repr(target.name) for target in sources.filter(['~bpcal', '~gaincal'])]))
    user_logger.info("Bandpass calibrators are [%s]",
                     ', '.join([repr(bpcal.name) for bpcal in sources.filter('bpcal')]))
    user_logger.info("Gain calibrators are [%s]",
                     ', '.join([repr(gaincal.name) for gaincal in sources.filter('gaincal')]))
    duration = {'target': opts.target_duration, 'bpcal': opts.bpcal_duration,
                'gaincal': opts.gaincal_duration}

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        # If bandpass interval is specified, force the first visit to be to the bandpass calibrator(s)
        time_of_last_bpcal = 0
        time_of_last_lunar_cal = 0
        loop = True
        source_total_duration = defaultdict(float)
        
        while loop:
            # Loop over sources in catalogue in sequence
            source_observed = defaultdict(bool)
            for source in sources:
                # If it is time for a bandpass calibrator to be visited on an interval basis, do so
                if opts.bpcal_interval is not None and time.time() - time_of_last_bpcal >= opts.bpcal_interval:
                    time_of_last_bpcal = time.time()
                    for bpcal in sources.filter('bpcal'):
                        session.label('track')
                        track_status = session.track(bpcal, duration=duration['bpcal'])
                        
                        if track_status:
                            source_total_duration[bpcal] += duration['bpcal']
                if time.time() - time_of_last_lunar_cal >= opts.lunar_interval:
                    time_of_last_lunar_cal = time.time()
                    session.label('Moon')
                        track_status = session.track(katpoint.Target('Moon,special'), duration=duration['bpcal'])
                        if track_status:
                            source_total_duration[katpoint.Target('Moon,special')] += duration['bpcal']
                # Visit source if it is not a bandpass calibrator
                # (or bandpass calibrators are not treated specially)
                # If there are no targets specified, assume the calibrators are the targets, else
                targets = [target for target in sources.filter(['~bpcal', '~gaincal'])]
                if opts.bpcal_interval is None or 'bpcal' not in source.tags or not targets:
                    # Set the default track duration for a target with no recognised tags
                    track_duration = opts.target_duration
                    for tag in source.tags:
                        track_duration = duration.get(tag, track_duration)
                    session.label('track')
                    track_status = source_observed[source] = session.track(source, duration=track_duration)
                    
                    if track_status:
                        source_total_duration[source] += track_duration
                        
                if opts.max_duration and time.time() > start_time + opts.max_duration:
                    user_logger.info('Maximum script duration (%d s) exceeded, stopping script',
                                     opts.max_duration)
                    loop = False
                    break
            if loop and not any(source_observed.values()):
                user_logger.warning('All imaging targets and gain cals are '
                                    'currently below horizon, stopping script')
                loop = False
        for source in sources:
            user_logger.info('Source %s observed for %.2f hrs',
                             source.description, source_total_duration[source] / 3600.0)
