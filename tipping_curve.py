#!/usr/bin/python
# Perform tipping curve scan for a specified azimuth position.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, CaptureSession, TimeSession, user_logger

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

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))

        # Iterate through elevation angles
        for el in [2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
            session.track('azel, %f, %f' % (opts.az, el), duration=15.0)
            session.fire_noise_diode('coupler', on=10, off=10)
