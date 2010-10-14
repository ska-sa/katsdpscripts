#!/usr/bin/python
# Perform large raster scan on specified target(s). Mostly used for beam pattern mapping.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, CaptureSession, TimeSession

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target 1'> [<'target 2'> ...]",
                                 description="Perform large raster scan across one or more sources. Mostly used for \
                                              beam pattern mapping and on-the-fly mapping. Some options are \
                                              **required**.")
# Add experiment-specific options
parser.add_option('-p', '--scan_spacing', type='float', default=0.125,
                  help='Separation between scans, in degrees (default="%default")')
parser.add_option('-x', '--scan_extent', type='int', default=2,
                  help='Length of each scan, in degrees (default="%default")')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Raster scan')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) == 0:
    raise ValueError("Please specify at least one target argument \
                      (via name, e.g. 'Cygnus A' or description, e.g. 'azel, 20, 30')")

with verify_and_connect(opts) as kat:

    targets = lookup_targets(kat, args)

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        for target in targets:
            # Do raster scan on target, designed to have equal spacing in azimuth and elevation, for a "classic" look
            scan_extent = opts.scan_extent
            scan_spacing = opts.scan_spacing
            # Put odd number of dumps along each scan so that spacing between dumps is close to spacing between scans.
            # Also let number of scans be equal to number of dumps per scan - this creates a "classic look" of a
            # square raster with square pixels.
            dumps_per_scan = int(scan_extent / scan_spacing)
            if dumps_per_scan % 2 == 0:
                dumps_per_scan += 1
            scan_duration = dumps_per_scan / opts.dump_rate

            session.raster_scan(target, num_scans=dumps_per_scan, scan_duration=scan_duration,
                                scan_extent=scan_extent, scan_spacing=scan_spacing, drive_strategy='longest-track')
