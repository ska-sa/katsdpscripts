# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5

# run a simple scan script to derive the horizon mask for KAT-7
# scan over constant elevation range but loop over azimuth

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import numpy as np

from katuilib.observe import standard_script_options, verify_and_connect, start_session

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Derive the horizon mask for KAT-7. Scan over constant elevation range \
                                              but loop over azimuth. Note also some **required** options below.")
# Add experiment-specific options
#parser.add_option('-m', '--max_time', dest='max_time', type="float", default=-1.0,
#                                         help="Time limit on experiment, in seconds (default=no limit)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Horizon mask data')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    azimuth_angle = np.arange(-180.0, 181.0)
    elev_center = 8.5
    elev_offset = 6.5

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        # Iterate through azimuth and elevation angles
        for az in azimuth_angle:
            session.scan('azel, %f, %f' % (az, elev_center), duration=15.0, start=(0, -elev_offset), end=(0, elev_offset))
            session.fire_noise_diode('coupler')
