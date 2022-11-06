# TODO: @mamkhari

import time

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger,
                                CalSolutionsUnavailable)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""

# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Perform offset pointings on the first source and obtain ' \
              'pointing offsets based on interferometric gains. At least ' \
              'one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=16.0,
                  help='Duration of each offset pointing, in seconds (default=%default)')
parser.add_option('--max-extent', type='float', default=1.0,
                  help='Maximum distance of offset from target, in degrees')
parser.add_option('--pointings', type='int', default=10,
                  help='Number of offset pointings')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Reference pointing', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Build up sequence of pointing offsets running linearly in x and y directions
scan = np.linspace(-opts.max_extent, opts.max_extent, opts.pointings // 2)
offsets_along_x = np.c_[scan, np.zeros_like(scan)]
offsets_along_y = np.c_[np.zeros_like(scan), scan]
offsets = np.r_[offsets_along_y, offsets_along_x]
offset_end_times = np.zeros(len(offsets))
middle_time = 0.0
weather = {}

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe or not enough antennas
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        session.capture_start()

        # XXX Eventually pick closest source as our target, now take first one
        target = observation_sources.targets[0]
        target.add_tags('bfcal single_accumulation')
        session.label('interferometric_pointing')
        user_logger.info("Initiating interferometric pointing scan on target "
                         "'%s' (%d pointings of %g seconds each)",
                         target.name, len(offsets), opts.track_duration)
        session.track(target, duration=0, announce=False)
        # Point to the requested offsets and collect extra data at middle time
        for n, offset in enumerate(offsets):
            user_logger.info("initiating track on offset of (%g, %g) degrees", *offset)
            session.ants.req.offset_fixed(offset[0], offset[1], opts.projection)
            session.track(target, duration=opts.track_duration, announce=False)
            offset_end_times[n] = time.time()
            if not kat.dry_run and n == len(offsets) // 2 - 1:
                # Get weather data for refraction correction at middle time
                temperature = kat.sensor.anc_air_temperature.get_value()
                pressure = kat.sensor.anc_air_pressure.get_value()
                humidity = kat.sensor.anc_air_relative_humidity.get_value()
                weather = {'temperature': temperature, 'pressure': pressure,
                           'humidity': humidity}
                middle_time = offset_end_times[n]
                user_logger.info("reference time = %.1f, weather = "
                                 "%.1f deg C | %.1f hPa | %.1f %%",
                                 middle_time, temperature, pressure, humidity)
        # Clear offsets in order to jiggle cal pipeline to drop its final gains
        # XXX We assume that the final entry in `offsets` is not the origin
        user_logger.info("returning to target to complete the scan")
        session.ants.req.offset_fixed(0., 0., opts.projection)
        session.track(target, duration=0, announce=False)
        user_logger.info("Waiting for gains to materialise in cal pipeline")

        # Perform basic interferometric pointing reduction
        if not kat.dry_run:
            # Wait for last piece of the cal puzzle (crash if not on time)
            last_offset_start = offset_end_times[-1] - opts.track_duration
            session.get_cal_solutions('G', timeout=300.,
                                      start_time=last_offset_start)
            user_logger.info('Retrieving gains, fitting beams, storing offsets')
            data_points = get_offset_gains(session, offsets, offset_end_times,
                                           opts.track_duration)
            beams = fit_primary_beams(session, data_points)
            pointing_offsets = calc_pointing_offsets(session, beams, target,
                                                     middle_time, **weather)
            save_pointing_offsets(session, pointing_offsets, middle_time)
