#!/usr/bin/env python
# Track target and calibrators for imaging.

import time
from collections import defaultdict

import numpy as np

from katcorelib import (
    standard_script_options,
    verify_and_connect,
    collect_targets,
    start_session,
    user_logger,
)


ND_LEAD_TIME = 12.0


def sdp_dump_info(session):
    """Get Unix timestamp of start of first SDP dump and its period."""
    view = session._telstate_capture_stream('sdp_l0')
    dump_period = view['int_time']
    # The first_timestamp is in the middle of first SDP dump, relative to sync time
    first_dump_start = view['sync_time'] + view['first_timestamp'] - 0.5 * dump_period
    return first_dump_start, dump_period


def next_event(past_event, period, lead_time=0.0):
    """Time of next periodic event more than `lead_time` into the future."""
    earliest_time = time.time() + lead_time
    n_periods_since_past_event = np.ceil((earliest_time - past_event) / period)
    return past_event + n_periods_since_past_event * period


def trigger_noise_diode(
    session, first_dump_start, dump_period, requested_nd_period, nd_on_fraction
):
    """Set noise diode firing pattern on all antennas, aligned with SDP dumps."""
    if nd_on_fraction <= 0.0:
        user_logger.info("Noise diode not triggered")
        return
    nd_period = np.ceil(requested_nd_period / dump_period) * dump_period
    next_dump_start = next_event(first_dump_start, dump_period, ND_LEAD_TIME)
    nd_start_time = next_dump_start + 0.05 * dump_period
    session.ants.req.dig_noise_source(nd_start_time, nd_on_fraction, nd_period)


# Set up standard script options
description = "Perform an on-the-fly (OTF) imaging scan."
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option(
    '-t', '--target-duration', type='float', default=150,
    help='Minimum duration to track the imaging target per linear scan, '
         'in seconds (default="%default")')
parser.add_option(
    '-b', '--bpcal-duration', type='float', default=300,
    help='Minimum duration to track bandpass calibrator per visit, '
         'in seconds (default="%default")'
)
parser.add_option(
    '-g', '--gaincal-duration', type='float', default=120,
    help='Minimum duration to track gain calibrator per visit, '
         'in seconds (default="%default")'
)
parser.add_option(
    '--elevation', type='float', default=36,
    help='Fixed elevation for OTF scan, in degrees (default="%default")'
)
parser.add_option(
    '--az-start', type='float', default=-77,
    help='Initial azimuth for OTF scan, in degrees (default="%default")'
)
parser.add_option(
    '--az-stop', type='float', default=-55,
    help='Final azimuth for OTF scan, in degrees (default="%default")'
)
parser.add_option(
    '--nd-period', type='float', default=19.5,
    help='Minimum noise diode period, in seconds (default="%default")'
)
parser.add_option(
    '--nd-on-fraction', type='float', default=0.03,
    help='Minimum noise diode on fraction / duty cycle (default="%default")'
)
parser.set_defaults(description='OTF scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration,
# connecting to proxies and devices
if len(args) == 0:
    raise ValueError("Please specify the target(s) and calibrator(s) "
                     "to observe as arguments, either as description "
                     "strings or catalogue filenames")

with verify_and_connect(opts) as kat:
    sources = collect_targets(kat, args)

    bpcals = sources.filter("bpcal")
    if not bpcals:
        raise ValueError("Please specify a bandpass calibrator")
    bpcal = bpcals.targets[0]
    user_logger.info("Bandpass calibrator is %s", bpcal.name)

    gaincals = sources.filter("gaincal")
    if not gaincals:
        raise ValueError("Please specify a gain calibrator")
    gaincal = gaincals.targets[0]
    user_logger.info("Gain calibrator is %s", gaincal.name)

    scan_start = 'azel, {}, {}'.format(opts.az_start, opts.elevation)

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_of_first_dump, dump_period = sdp_dump_info(session)

        trigger_noise_diode(session, start_of_first_dump, dump_period,
                            opts.nd_period, opts.nd_on_fraction)

        # track bpcal
        session.label('track')
        session.track(bpcal, duration=opts.bpcal_duration)

        # track gaincal
        session.label('track')
        session.track(gaincal, duration=opts.gaincal_duration)

        # goto start of scan (pure slew, no CBF target)
        # set_target? on_target? -> ensure_target? slew_to?

        # loop:
        #   figure out ra, dec and set correlator (maybe untangle set_target?)
        #   load scan 30 (10?) seconds in advance

        #   - AP stores 3000 samples
        #   - AP spline needs 4 samples in future in stack at all times (CAM does 30)
        #   - minimum time interval between samples: 50 ms
        #   - command ?track-az-el has microdegree resolution, natural dish coords
        #   - CAM uses 200 ms samples => buffer can store 600 s = 10 min
        #   - you still need a target...

        #   halfway through, load next one
        #   consider three sections: ramp-up, scan, ramp-down
        #   keep the templates in dump index format and scale appropriately

        # BSpline(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), [0, 0, 0, 0.5, 1.0], 4)
        # final slope: 2, going from 0 to 1

        # inputs:
        # - speed (0.1 deg/s = 6 arcmin/s)
        # - range (22 deg -> 220 s)
        # - ramp up: 4 s? Then 2 s @ 0.1 deg/s = start 0.2 deg away
        # - 200 ms samples = 20 samples in ramp-up

        # track gaincal
        session.label('track')
        session.track(gaincal, duration=opts.gaincal_duration)






















    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        # If bandpass interval is specified, force the first visit to be to the bandpass calibrator(s)
        time_of_last_bpcal = 0
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
