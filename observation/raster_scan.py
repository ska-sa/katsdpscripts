#!/usr/bin/python
# Perform raster scan on specified target(s). Mostly used for beam pattern mapping.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Perform raster scan across one or more sources. Mostly used for '
                                             'beam pattern mapping and on-the-fly mapping. Some options are '
                                             '**required**.')
# Add experiment-specific options
parser.add_option('-k', '--num-scans', type='int', default=-1,
                  help='Number of scans across target, usually an odd number (the default automatically selects '
                       'this based on scan extent and spacing in order to create a uniform grid of dots in raster)')
parser.add_option('-t', '--scan-duration', type='float', default=-1.0,
                  help='Minimum duration of each scan across target, in seconds (the default automatically selects '
                       'this based on scan extent and spacing in order to create a uniform grid of dots in raster)')
parser.add_option('-l', '--scan-extent', type='float', default=2.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('-m', '--scan-spacing', type='float', default=0.125,
                  help='Separation between scans, in degrees (default=%default)')
parser.add_option('-e', '--scan-in-elevation', action='store_true', default=False,
                  help='Scan in elevation rather than in azimuth (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Raster scan')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# For the default "classic" look of a square raster with square pixels, choose the scan duration so that spacing
# between dumps is close to spacing between scans, and set number of scans equal to number of dumps per scan
classic_dumps_per_scan = int(opts.scan_extent / opts.scan_spacing)
if classic_dumps_per_scan % 2 == 0:
    classic_dumps_per_scan += 1
if opts.num_scans <= 0:
    opts.num_scans = classic_dumps_per_scan
if opts.scan_duration <= 0.0:
    opts.scan_duration = classic_dumps_per_scan / opts.dump_rate

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)

    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        for target in observation_sources:
            session.label('raster')
            session.raster_scan(target, num_scans=opts.num_scans, scan_duration=opts.scan_duration,
                                scan_extent=opts.scan_extent, scan_spacing=opts.scan_spacing,
                                scan_in_azimuth=not opts.scan_in_elevation, projection=opts.projection)
