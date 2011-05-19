#!/usr/bin/python
# Perform tipping curve scan for a specified azimuth position.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, start_session, user_logger

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform tipping curve scan for a specified azimuth position. \
                                              Some options are **required**.")
# Add experiment-specific options
parser.add_option('-z', '--az', type="float", default=168.0,
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Tipping curve')
# Parse the command line
opts, args = parser.parse_args()

# Ensure that azimuth is in valid physical range of -185 to 275 degrees
if (opts.az < -185.) or (opts.az > 275.):
    opts.az = (opts.az + 180.) % 360. - 180.

with verify_and_connect(opts) as kat:

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        # Iterate through elevation angles
        for el in [10, 12.5, 15, 17.5,  20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5,  50, 52.5,  55, 57.5,  60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90]:
            session.track('azel, %f, %f' % (opts.az, el), duration=15.0)
            session.fire_noise_diode('coupler', on=10, off=10)
