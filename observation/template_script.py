#!/usr/bin/python
# Template script

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

# Import script helper functions from observe.py
from katcorelib.observe import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Explain what the script does and how to run it. '
                                             'Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--special-option', type='float', default=60.0,
                  help='This option only applies to this script (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='My observation does this.')
# Parse the command line
opts, args = parser.parse_args()

# Check options and arguments, and build KAT configuration, connecting to proxies and devices
if len(args) == 0:
    # Don't use sys.exit() to quit the script - raise an exception instead!
    # This allows the script to be run within an IPython session without taking the whole python session down
    raise ValueError("Please specify at least one argument")

# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:

    # The default target collector extracts targets from the argument list containing a mixture of catalogue files,
    # target description strings and / or target names (the latter looked up in the default kat.sources catalogue)
    targets = collect_targets(kat, args)

    # Initialise a capturing session (which typically opens an HDF5 file)
    with start_session(kat, **vars(opts)) as session:
        # Use the command-line options to set up the system
        session.standard_setup(**vars(opts))
        # Start capturing data (i.e. the correlator starts running)
        session.capture_start()

        # Whatever your script does goes here....
