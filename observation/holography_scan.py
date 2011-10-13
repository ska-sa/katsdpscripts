#!/usr/bin/python
# Template script

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

# Import script helper functions from observe.py
from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, start_session, user_logger
from katuilib.array import Array
# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'Target 1'> ",
                                 description='This script points the Antennas at a source specified as an argument. '
                                             'A sub-group of antennas will then scan across sky while the reference '
                                             'antennas remain traking the target. '
                                             'Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-k', '--num-scans', type='int', default=1,
                  help='Number of scans across target, (default=%default) ')
parser.add_option('-t', '--scan-duration', type='float', default=30.0,
                  help='Minimum duration of each scan across target, (default=%default) ')
parser.add_option('-l', '--scan-extent', type='float', default=5.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('-m', '--scan-spacing', type='float', default=0.0,
                  help='Separation between scans, in degrees (default=%default)')

parser.add_option('-b', '--scan-ants', type='str', default=None,
                  help='This option specifies the antennas that will scan across the sky while the other remain pointing at the source (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='holography scan.')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if not len(args) == 1:
    raise ValueError("Please specify at least one Target")

# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:

    # If the arguments are a list of target names, look them up in the default kat.sources catalogue
    targets = lookup_targets(kat, args)
    if  len(targets) > 0 :
        # Initialise a capturing session (which typically opens an HDF5 file)
        with start_session(kat, **vars(opts)) as session:
            # Use the command-line options to set up the system
            session.standard_setup(**vars(opts))
            All_ants = session.ants
            try:
                Sub_ants =  Array('ants', [getattr(kat, ant.strip()) for ant in opts.scan_ants.split(',')])
            except AttributeError:
                raise ValueError("Antenna '%s' not found (i.e. no kat.%s exists)" % (ant, ant))

            session.capture_start()
            for target in targets:
                session.ants = All_ants
                session.track(target, duration=0,label='', announce=False)
                for scan_no in range(opts.num_scans):
                    offset = ((opts.num_scans-1.0)*opts.scan_spacing)/2.0 - opts.scan_spacing*scan_no
                    session.ants = Sub_ants  # Move the Sub section of the array
                    nd_params = session.nd_params
                    session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
                    session.scan(target, duration=opts.scan_duration, start=(-opts.scan_extent*((-1.0)**scan_no), offset),
                                 end=(opts.scan_extent*((-1.0)**scan_no), offset),projection=opts.projection, announce=False)
                    session.nd_params = nd_params
                    session.ants = All_ants
                    session.fire_noise_diode(announce=False, **session.nd_params)
