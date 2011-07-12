#!/usr/bin/python
# Perform tipping curve scan for a specified azimuth position.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

from katuilib.observe import standard_script_options, verify_and_connect, start_session, user_logger
import numpy as np
# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform tipping curve scan for a specified azimuth position. \
                                              Some options are **required**.")
# Add experiment-specific options
parser.add_option('-z', '--az', type="float", default=168.0,
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')

parser.add_option('--spacing', type="float", default=1.0,
                  help='The Spacing along the elevation axis of the tipping curve that measuremnts are taken, in degrees (default="%default")')
parser.add_option( '--tip-both-directions', action="store_true" , default=False,
                  help='Do tipping curve from low to high elevation and then from high to low elevation')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Tipping curve')
# Parse the command line
opts, args = parser.parse_args()

# Ensure that azimuth is in valid physical range of -185 to 275 degrees
if (opts.az < -185.) or (opts.az > 275.):
    opts.az = (opts.az + 180.) % 360. - 180.
user_logger.info("Tipping Curve at Azimuth=%f"%(opts.az,))
with verify_and_connect(opts) as kat:

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        # Iterate through elevation angles
        spacings = list(np.arange(10.0,90.1,opts.spacing))
        if opts.tip_both_directions :
            spacings += list(np.arange(90.0,9.,-opts.spacing))
        for el in spacings:
            session.track('azel, %f, %f' % (opts.az, el), duration=15.0)
            session.fire_noise_diode('coupler', on=10, off=10)
