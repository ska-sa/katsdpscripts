#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, CaptureSession

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target 1'> [<'target 2'> ...]",
                                 description="Track one or more sources for a specified time. At least one \
                                              target must be specified. Note also some **required** options below.")
# Add experiment-specific options
parser.add_option('-t', '--track_duration', type='int', default=60,
                  help='Length of time to track each source, in integer secs (default="%default")')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) == 0:
    raise ValueError("Please specify at least one target argument \
                      (via name, e.g. 'Cygnus A' or description, e.g. 'azel, 20, 30')")

with verify_and_connect(opts) as kat:

    targets = lookup_targets(kat, args)

    # Create a data capturing session with the selected sub-array of antennas
    with CaptureSession(kat, **vars(opts)) as session:
        for target in targets:
            # Track target
            session.track(target, duration=opts.track_duration, drive_strategy='longest-track', label='track')
