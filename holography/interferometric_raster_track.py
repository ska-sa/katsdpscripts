#!/usr/bin/env python
# Track target(s) for a specified time.

import time
import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)

import katpoint
import ephem
import datetime

class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track one or more sources for a specified time. At least one ' \
              'target must be specified. Note also some **required** options below.'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option('--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds '
                       '(default=%default)')
parser.add_option('--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which '
                       'script will end as soon as the current track finishes '
                       '(no limit by default)')
parser.add_option('--cal-duration', type='float', default=180.0,
                  help='Minimum duration to track bandpass calibrator per visit, in seconds '
                       '(default="%default")')
parser.add_option('--cal-interval', type='float', default=3600.0,
                  help='Minimum interval between calibrator visits, in seconds '
                       '(visits each source in turn by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum '
                       'duration (which must be set for this)')
parser.add_option('--reset-gain', type='int', default=None,
                  help='Value for the reset of the correlator F-engine gain '
                       '(default=%default)')
parser.add_option('--fft-shift', type='int',
                  help='Set correlator F-engine FFT shift (default=leave as is)')
parser.add_option('--moon-snapshot-duration', type='float', default=16.0,
                  help="Lunar snapshot time in seconds. The snapshot ra/dec is "
                  "centred at the mid-point of the snapshot timespan.")
parser.add_option('--moon-nsnapshots-per-track', type='int', default=19,
                  help="Number of lunar snapshots to take during lunar track")
parser.add_option('--moon-repeat', type="float", default=20*60.0,
                  help="Repeat lunar observation every specified seconds")

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)
    user_logger.info("Transfer targets are [%s]",
                     ', '.join([repr(target.name) for target in targets.filter(['~bpcal', '~gaincal'])]))
    user_logger.info("Bandpass calibrators are [%s]",
                     ', '.join([repr(bpcal.name) for bpcal in targets.filter('bpcal')]))
    user_logger.info("Polarization calibrators are [%s]",
                     ', '.join([repr(polcal.name) for polcal in targets.filter('polcal')]))

    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(targets.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Set the gain to a single non complex number if needed
        if opts.reset_gain is not None:
            if not session.cbf.fengine.inputs:
                raise RuntimeError("Failed to get correlator input labels, "
                                   "cannot set the F-engine gains")
            for inp in session.cbf.fengine.inputs:
                session.cbf.fengine.req.gain(inp, opts.reset_gain)
                user_logger.info("F-engine %s gain set to %g",
                                 inp, opts.reset_gain)

        session.standard_setup(**vars(opts))
        if opts.fft_shift is not None:
            session.cbf.fengine.req.fft_shift(opts.fft_shift)
        session.capture_start()

        start_time = time.time()

        targets_observed = []
        # Keep going until the time is up
        target_total_duration = [0.0] * len(targets)
        keep_going = True
        
        last_lunar_scan = -np.inf
        last_cal_visit = -np.inf
        while keep_going:
            keep_going = (opts.max_duration is not None) and opts.repeat
            targets_before_loop = len(targets_observed)
            # Iterate through source list, picking the next one that is up
            target_list = list(targets.filter(['~bpcal', '~gaincal']))
            n = 0
            while n < len(target_list):
                # Cut the track short if time ran out
                def _time_left(opts, start_time, track_duration):
                    """ compute duration based on time remaining """
                    if track_duration is not None:
                        time_left = opts.max_duration - (time.time() - start_time)
                        if time_left <= 0.:
                            user_logger.warning("Maximum duration of %g seconds "
                                                "has elapsed - stopping script",
                                                opts.max_duration)
                            return 0, False
                        duration = min(track_duration, time_left)
                        return duration, True

                def _go_observe(session, target, targets_observed, target_total_duration, duration):
                    session.label('track')
                    if session.track(target, duration=duration):
                        targets_observed.append(target.description)
                        target_total_duration[n] += duration
                # select between calibrator, moon or target
                if time.time() - last_lunar_scan > opts.moon_repeat:
                    # time to observe moon
                    last_lunar_scan = time.time()
                    for lunar_track in range(opts.moon_nsnapshots_per_track):
                        duration, keep_going = _time_left(opts, start_time, opts.moon_snapshot_duration)
                        if not keep_going: 
                            #user_logger.info("DEBUG: Out of time. Stopping")
                            break
                        observer = ephem.Observer()
                        observer.lon='21:24:38.5'
                        observer.lat='-30:43:17.3'
                        observer.elevation = 1038.0
                        # ra, dec to be computed based on the centre of the snapshot
                        # it is up to the observer to pick sensible uv cuts to ensure
                        # moon does not move more than a fractional synthesized beam width
                        dt = datetime.datetime.utcfromtimestamp(time.time() + duration / 2.0)
                        observer.date = str(dt)
                        #user_logger.info("DEBUG: Setting lunar ephemaris to {}".format(observer.date))
                        moon = ephem.Moon(observer)
                        moon.compute(observer)
                        target = katpoint.construct_radec_target(moon.ra, moon.dec)
                        session.label('track')
                        user_logger.info("Initiating %g-second Lunar snapshot (%d/%d) (%.2f, %.2f)" % 
                                        (duration, 
                                        lunar_track + 1,
                                        opts.moon_nsnapshots_per_track,
                                        katpoint.rad2deg(float(moon.ra)), 
                                        katpoint.rad2deg(float(moon.dec)),))
                        _go_observe(session, target, targets_observed, target_total_duration, duration)
                    if not keep_going: 
                        #user_logger.info("DEBUG: Out of time. Stopping")
                        break
                elif time.time() - last_cal_visit > opts.cal_interval:
                    # time to observe calibrators
                    last_cal_visit = time.time()
                    for t in list(targets.filter('bpcal')) + list(targets.filter('polcal')):
                        duration, keep_going = _time_left(opts, start_time, opts.cal_duration)    
                        if not keep_going: 
                            #user_logger.info("DEBUG: Out of time. Stopping")
                            break
                        _go_observe(session, t, targets_observed, target_total_duration, duration)
                else:
                    # select next raster target
                    duration, keep_going = _time_left(opts, start_time, opts.track_duration)
                    if not keep_going: 
                        #user_logger.info("DEBUG: Out of time. Stopping")
                        break
                    target = target_list[n]
                    _go_observe(session, target, targets_observed, target_total_duration, duration)
                    n += 1 # next target

            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - "
                                    "stopping script instead of hanging around")
                keep_going = False
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
        # print out a sorted list of target durations
        ind = np.argsort(target_total_duration)
        for i in reversed(ind):
            user_logger.info('Source %s observed for %.2f hrs',
                             targets.targets[i].description, target_total_duration[i] / 3600.0)
