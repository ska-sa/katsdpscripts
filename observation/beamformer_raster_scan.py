#!/usr/bin/env python
#
# Perform raster scan using tied-array beam.
#
# Ludwig Schwardt
# 11 August 2021
#

import time

import numpy as np

from katcorelib.observe import (
    SessionCBF,
    collect_targets,
    standard_script_options,
    start_session,
    user_logger,
    verify_and_connect,
)
from katcorelib.targets import Offset


def set_beamformer_gains(cbf, ants, gain):
    """"""
    for stream in cbf.beamformers:
        if not stream.name.endswith('0x'):
            continue
        # Set beamformer quantization gain.
        bq_gain = gain / np.sqrt(len(ants))
        reply = stream.req.quant_gains(bq_gain)
        if reply.succeeded:
            user_logger.info('Beamformer stream %s quantization gain set to '
                             '%g for %d antennas.', stream, bq_gain, len(ants))
        else:
            user_logger.warning('Failed to set beamformer quantization gain %g on '
                                'stream %s!', bq_gain, stream)


def set_beamformer_weights(cbf, ants):
    """"""
    for stream in cbf.beamformers:
        if not stream.name.endswith('0x'):
            continue
        # Set the beamformer weights (TAB shaping).
        user_logger.info('Setting beamformer weights for stream %s:', stream)
        weights = []
        for inp in stream.inputs:
            weight = 1.0 if inp[:-1] in ants else 0.0
            weights.append(weight)
            user_logger.info('  input %r will get weight %f', inp, weight)
        reply = stream.req.weights(*weights)
        if reply.succeeded:
            user_logger.info('Set input weights successfully')
        else:
            user_logger.warning('Failed to set input weights!')


def point(session, target, duration):
    """"""
    user_logger.info('  pointing at %s for %g seconds', target, duration)
    before = time.time()
    for stream in session.cbf.beamformers:
        if not stream.name.endswith('0x'):
            continue
        session.cbf.req.set_beam_target('wide.tied-array-channelised-voltage.0x', target)
        # user_logger.info('  (%s -> %s)', stream.name, target)
    time_spent = time.time() - before
    time.sleep(duration - time_spent)


def scan(session, target, scan_duration, pointing_duration, start, end,
         index, projection, coord_system):
    """"""
    projection = Offset.PROJECTIONS.get(projection, projection)
    num_pointings = int(scan_duration / pointing_duration)
    x_steps = np.linspace(start[0], end[0], num_pointings)
    y_steps = np.linspace(start[1], end[1], num_pointings)
    user_logger.info('performing scan %d (%d pointings)', index, num_pointings)
    session.activity('scan')
    for x, y in zip(x_steps, y_steps):
        mid_pointing = time.time() + pointing_duration / 2.
        lon, lat = target.plane_to_sphere(np.radians(x), np.radians(y),
                                          timestamp=mid_pointing,
                                          projection_type=projection,
                                          coord_system=coord_system)
        pointing = '{}, {}, {}'.format(coord_system, np.degrees(lon), np.degrees(lat))
        point(session, pointing, pointing_duration)


def raster_scan(session, target, num_scans, scan_duration, pointing_duration,
                scan_extent, scan_spacing, scan_offset, scan_in_longitude=True,
                projection='zenithal-equidistant', coord_system='azel', announce=True):
    """Perform raster scan on target."""
    if announce:
        user_logger.info("Initiating raster scan (%d %g-second scans "
                            "extending %g degrees) on target %r", num_scans,
                            scan_duration, scan_extent, target.name)
    # Check whether target is visible for entire duration of raster scan
    if not session.target_visible(target, scan_duration * num_scans):
        user_logger.warning("Skipping raster scan, as target %r will be "
                            "below horizon", target.name)
        return False
    # Create start and end positions of each scan, based on scan parameters
    scan_levels = np.arange(-(num_scans // 2), num_scans // 2 + 1)
    scanning_coord = (scan_extent / 2.0) * (-1.0) ** scan_levels
    stepping_coord = scan_spacing * scan_levels
    # Flip sign of elevation offsets to ensure that the first scan always
    # starts at the top left of target
    if scan_in_longitude:
        scan_starts = list(zip(scanning_coord + scan_offset[0],
                               -stepping_coord + scan_offset[1]))
        scan_ends = list(zip(-scanning_coord + scan_offset[0],
                             -stepping_coord + scan_offset[1]))
    else:
        scan_starts = list(zip(stepping_coord + scan_offset[0],
                               -scanning_coord + scan_offset[1]))
        scan_ends = list(zip(stepping_coord + scan_offset[0],
                             scanning_coord + scan_offset[1]))
    # Perform multiple scans across the target
    for scan_index, (start, end) in enumerate(zip(scan_starts, scan_ends)):
        scan(session, target, scan_duration, pointing_duration, start, end,
             scan_index, projection, coord_system)
    return True


# The minimum time per beamformer pointing, since the target cannot be updated faster
DELAY_UPDATE_PERIOD = 5  # seconds

# Set up standard script options
usage = '%prog [options] <target>'
description = (
    'Perform a raster scan on <target> using the first tied-array beam. '
    'It is assumed that the beamformer is already phased up on a calibrator.'
)
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('--beam-quant-gain', type=float, default=0.6,
                  help='Set beamformer quantisation gain (default=%default/sqrt(N_ants))')
parser.add_option('--num-scans', type=int, default=-1,
                  help='Number of scans across target, usually an odd number '
                       '(the default automatically selects this based on scan '
                       'extent and spacing in order to create a uniform grid '
                       'of dots in raster)')
parser.add_option('--scan-duration', type=float, default=-1.0,
                  help='Minimum duration of each scan across target, in seconds '
                       '(the default automatically selects this based on scan '
                       'extent and spacing in order to create a uniform grid '
                       'of dots in raster)')
parser.add_option('--scan-extent', type=float, default=0.5,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('--scan-spacing', type=float, default=1. / 60.,
                  help='Separation between scans, in degrees (default=%default)')
parser.add_option('--scan-offset-x', type=float, default=0.,
                  help='Offset of scan pattern in x direction, in degrees (default=%default)')
parser.add_option('--scan-offset-y', type=float, default=0.,
                  help='Offset of scan pattern in y direction, in degrees (default=%default)')
parser.add_option('--scan-in-elevation', action='store_true', default=False,
                  help='Scan in elevation rather than in azimuth (default=%default)')
parser.add_option('--pointing-duration', type=float, default=DELAY_UPDATE_PERIOD,
                  help='Time spent on each pointing, in seconds (default=%default)')
parser.add_option('--coord-system', type='choice', choices=('azel', 'radec'), default='radec',
                  help='Coordinate system in which to perform projections (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Beamformer raster scan', nd_params='off', no_delays=False)
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# For the default "classic" look of a square raster with square pixels,
# choose the scan duration so that spacing between dumps is close to
# spacing between scans, and set number of scans equal to number of dumps per scan
classic_dumps_per_scan = int(opts.scan_extent / opts.scan_spacing)
if classic_dumps_per_scan % 2 == 0:
    classic_dumps_per_scan += 1
if opts.num_scans <= 0:
    opts.num_scans = classic_dumps_per_scan
if opts.scan_duration <= 0.0:
    opts.scan_duration = classic_dumps_per_scan * opts.pointing_duration

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    target = collect_targets(kat, args).targets[0]

    cbf = SessionCBF(kat)
    bf_ants = [ant.name for ant in kat.ants]
    set_beamformer_gains(cbf, bf_ants, opts.beam_quant_gain)
    set_beamformer_weights(cbf, bf_ants)

    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        session.label('track')
        session.track(target, duration=opts.pointing_duration)

        session.label('raster')
        raster_scan(session, target, opts.num_scans, opts.scan_duration,
                    opts.pointing_duration, opts.scan_extent, opts.scan_spacing,
                    (opts.scan_offset_x, opts.scan_offset_y),
                    not opts.scan_in_elevation,
                    opts.projection, opts.coord_system)
