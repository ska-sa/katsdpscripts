#!/usr/bin/python
# Track various point sources as specified in a catalogue for the purpose of baseline calibration.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time

from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, CaptureSession, TimeSession
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] [<catalogue files>]",
                                 description="Track various point sources from the specified catalogue file(s), or use \
                                              the default catalogue if none is specified. This is useful for baseline \
                                              (antenna location) calibration. Remember to specify the observer and \
                                              antenna options, as these are **required**.")
# Add experiment-specific options
parser.add_option('-m', '--min_time', type='float', default=-1.0,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Baseline calibration', nd_params='pin,0,0,-1')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    # Create baseline calibrator catalogue
    baseline_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    # Load catalogue files if given
    if len(args) > 0:
        for catfile in args:
            baseline_sources.add(file(catfile))
    else:
        # Prune the standard catalogue to only contain sources that are good for baseline calibration
        great_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A', 'Pictor A']
        good_sources =  ['3C48', '3C84', 'J0408-6545', 'J0522-3627', '3C161', 'J1819-6345', 'J1939-6342', '3C433', 'J2253+1608']
        baseline_sources.add([kat.sources[src] for src in great_sources + good_sources])

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            # Iterate through baseline sources that are up
            for target in baseline_sources.iterfilter(el_limit_deg=5):
                session.track(target, duration=120.0, drive_strategy='longest-track')
                targets_observed.append(target.name)
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break
        print "Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed)))
