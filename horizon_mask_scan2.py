# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5

# run a simple scan script to derive the horizon mask for KAT-7
# scan over constant elevation range but loop over azimuth

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import numpy as np

from katuilib.observe import standard_script_options, verify_and_connect, CaptureSession, TimeSession

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Derive the horizon mask for KAT-7. Note also some **required** options below.")
# Add experiment-specific options
parser.add_option("-z", "--az", dest="az", type="string", default='0,360',
                 help='Azimuth angle along which to do horizon mask, in degrees (default="%default")')
parser.add_option('-e', '--el', dest='el', type="float", default=10.0,
                help='Elevation angle along which to do horizon mask, in degrees (default="%default")')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Horizon mask data')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    start_azim = int(opts.az.split(',')[0])
    stop_azim = int(opts.az.split(',')[1])
    azim_size = np.abs(stop_azim - start_azim)
    scan_duration = azim_size / 1.0 # scan at one degree per second
    center_azim = (stop_azim + start_azim) / 2.0
    offset = np.abs(stop_azim)

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        # Iterate through azimuth and elevation angles
        for el in [opts.el]:
            session.scan('azel,%f,%f' % (center_azim, el), duration=scan_duration,
                         scan_in_azimuth=True, start=-offset, end=offset)
            session.fire_noise_diode('coupler')
