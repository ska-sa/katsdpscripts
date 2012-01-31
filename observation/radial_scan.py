#!/usr/bin/python
# Perform radial raster scan on specified target(s). Mostly used for beam pattern mapping.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import numpy as np
from katuilib.observe import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Perform radial raster scan across one or more sources. Mostly used for '
                                             'beam pattern mapping and on-the-fly mapping. Some options are '
                                             '**required**.')
# Add experiment-specific options
parser.add_option('-k', '--num-scans', type='int', default=3,
                  help='Number of scans across target (default=%default)')
parser.add_option('-t', '--scan-duration', type='float', default=20.0,
                  help='Minimum duration of each scan across target, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=2.0,
                  help='Length of each scan, in degrees (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Radial raster scan')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)

    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        for target in observation_sources:
            session.label('raster')
            user_logger.info("Initiating radial scan (%d %g-second scans extending %g degrees) on target '%s'" %
                             (opts.num_scans, opts.scan_duration, opts.scan_extent, target.name))
            # Calculate average time that noise diode is operated per scan, to add to scan duration in check below
            nd_time = session.nd_params['on'] + session.nd_params['off']
            nd_time *= opts.scan_duration / max(session.nd_params['period'], opts.scan_duration)
            nd_time = nd_time if session.nd_params['period'] >= 0 else 0.
            # Check whether the target will be visible for entire duration of radial scan
            if not session.target_visible(target, (opts.scan_duration + nd_time) * opts.num_scans):
                user_logger.warning("Skipping radial scan, as target '%s' will be below horizon" % (target.name,))
                continue
            # Iterate through angles and scan across target
            for ind, angle in enumerate(np.arange(0., np.pi, np.pi / opts.num_scans)):
                offset = np.array((np.cos(angle), -np.sin(angle))) * opts.scan_extent / 2. * (-1) ** ind
                session.scan(target, duration=opts.scan_duration, start=-offset, end=offset, index=ind,
                             projection=opts.projection, announce=False)
