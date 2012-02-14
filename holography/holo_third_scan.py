#!/usr/bin/python

# IMPORTANT: update TLE's before doing this!
#
# Scans through EUTELSAT W2M, with dwells on-target between each scan for 5s.
# Expects an az and el offset provided by the calling script, determined by first peaking
# up on the target. This can be done as follows:
#   kat.ant1.req.target(kat.sources["EUTELSAT W2M"])
#   az=0.0;el=0.0; kat.ant1.req.offset_fixed(az,el,'stereographic');
# Now iterate\ively increase az, el to maximize the magnitude
#   kat.dh.sd.plot_time_series('mag', products=[(1,2,'HH')], start_channel=377,stop_channel=378)

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import sys
import time
import numpy as np

from katcorelib import standard_script_options, verify_and_connect, user_logger, CaptureSession

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform raster scan across holography source. Some options are **required**.")
# Add experiment-specific options
parser.add_option('-e', '--scan_in_elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth, (default=%default)")
parser.add_option('-X', '--extent', dest='extent_deg', type="float", default=1.5,
                  help="Angular extent of the scan (same for X & Y), in degrees (default=%default)")
parser.add_option('-x', '--step', dest='step_deg', type="float", default=0.075,
                  help="Angular spacing of scans (in az or el, depending on scan direction), in degrees (default=%default)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Holography script version 3')
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:

    # Source to scan across
    tgt = kat.sources['EUTELSAT W2M']
    try:
        kat.ant1.req.antenna_rotation(-27) # XDM-only, "shallow" optimal value
    except:
        pass

    # The real experiment: Create a data capturing session with the selected sub-array of antennas
    with CaptureSession(kat, **vars(opts)) as session:
        kat.ant1.req.sensor_sampling("lock","event")
        kat.ant1.req.sensor_sampling("scan.status","event")
        kat.dbe.req.capture_setup()
        kat.dbe.req.capture_start()
        kat.dbe.req.k7w_new_scan('slew')
        kat.ant1.req.target(tgt.description)
        kat.ant1.req.mode("POINT")
        #kat.ant1.wait("lock",1,300)
         # wait for lock on boresight target
        #t_az = kat.ant1.sensor.pos_actual_scan_azim.get_value()
        #t_el = kat.ant1.sensor.pos_actual_scan_elev.get_value()
        #if az is not None:
        #    print "Adding azimuth offset of:",az
        #    t_az += az
        #if el is not None:
        #    print "Adding elevation offset of:",el
        #    t_el += el
        #kat.ant1.req.target_azel(t_az,t_el)
        # Alternative way to set the offset while still tracking the TLE
        kat.ant1.req.offset_fixed(az,el,'stereographic')
        kat.ant1.wait("lock",1,300)
        kat.dbe.req.k7w_new_scan('cal')
        time.sleep(5)
        kat.dbe.req.k7w_new_scan('slew')
        degrees=opts.extent_deg
        step=opts.step_deg
        nrsteps=degrees/(2.0*step)
        for x in np.arange(-nrsteps,nrsteps,1):
            offset = x * step
            user_logger.info("Scan %i: (%.2f,%.2f,%.2f,%.2f)" % (x, -0.5*degrees, offset, 0.5*degrees,offset))
            if opts.scan_in_elevation:
                kat.ant1.req.scan_asym(offset, -0.5*degrees, offset, 0.5*degrees, 25*degrees, "stereographic")
            else:
                kat.ant1.req.scan_asym(-0.5*degrees, offset, 0.5*degrees, offset, 25*degrees, "stereographic")
            kat.ant1.wait("lock",1,300)
             # wait for lock at start of scan
            kat.dbe.req.k7w_new_scan('scan')
            kat.ant1.req.mode("SCAN")
            time.sleep(25*degrees)
            kat.ant1.wait('scan_status', 'after', 300)
             # wait for scan to finish
            sys.stdout.flush()
            user_logger.info("Finished scan. Doing cal...")
            kat.ant1.req.mode("POINT")
            kat.dbe.req.k7w_new_scan('slew')
            #kat.ant1.req.target_azel(t_az,t_el)
            # Alternative way to set the offset while still tracking the TLE
            kat.ant1.req.target(tgt.description)
            kat.ant1.req.offset_fixed(az,el,'stereographic')
            kat.ant1.wait("lock",1,300)
            kat.dbe.req.k7w_new_scan('cal')
            time.sleep(5)
             # wait for 5seconds on cal
            kat.dbe.req.k7w_new_scan('slew')
            sys.stdout.flush()
            sys.stderr.flush()
    user_logger.info("Done...")
