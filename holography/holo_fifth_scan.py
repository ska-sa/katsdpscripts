#!/usr/bin/python

# IMPORTANT: update TLE's before doing this!
#
# Scans through a source (default EUTELSAT W2M), with dwells on-target between each scan for 5s.
# Expects an az and el offset provided by the calling script, determined by first peaking
# up on the target. This can be done as follows:
#   kat.ant1.req.target(kat.sources["EUTELSAT W2M"]) # or whatever your source
#   az=0.0;el=0.0; kat.ant1.req.offset_fixed(az,el,'stereographic');
# Now iteratively increase az, el to maximize the magnitude
#   kat.dh.sd.plot_time_series('mag', products=[(1,2,'HH')], start_channel=377,stop_channel=378)

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import sys
import time
import numpy as np

from katcorelib.observe import standard_script_options, verify_and_connect, user_logger, CaptureSession

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform multiple raster scans across holography source. Some options are **required**.")
# Remove options not applicable to holography
parser.remove_option('--centre-freq')
parser.remove_option('--dump-rate')
parser.remove_option('--nd-params')
parser.remove_option('--dry-run')
# Add experiment-specific options
parser.add_option('-t', '--target', default="EUTELSAT W2M",
                  help="Name of the source, as accepted by kat.sources (default=%default)")
parser.add_option('-e', '--scan_in_elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth, (default=%default)")
parser.add_option('-X', '--extent', dest='extent_deg', type="float", default=1.5,
                  help="Angular extent of the scan (same for X & Y), in degrees (default=%default)")
parser.add_option('-x', '--step', dest='step_deg', type="float", default=0.075,
                  help="Angular spacing of scans (in az or el, depending on scan direction), in degrees (default=%default)")
parser.add_option('-r', '--repeat', type="int", default=1,
                  help="Number of repeat rasters (default=%default)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Holography script version 5')
# Override antenna option help string
opt_ants = parser.get_option('--ants')
opt_ants.help = "Main holography antenna is hard-coded to %default, which is what it always must be."
# Parse the command line
opts, args = parser.parse_args()
# Hard-code main holography antenna
opts.ants = 'ant1'

with verify_and_connect(opts) as kat:

    # Source to scan across
    tgt = kat.sources[opts.target]
    try:
        kat.ant1.req.antenna_rotation(-27) # XDM-only, "shallow" optimal value
    except:
        pass

    for i in range(opts.repeat):
        print "="*40
        T0 = time.time()
        # The real experiment starts here, by creating a data capturing session
        with CaptureSession(kat, **vars(opts)) as session:
            # Setup strategies for the sensors we want to wait() on
            kat.ant1.req.sensor_sampling("lock","event")
            kat.ant1.req.sensor_sampling("scan.status","event")
            kat.dbe.req.capture_setup()
            kat.dbe.req.capture_start()
            kat.dbe.req.k7w_new_scan('slew')
            kat.ant1.req.target(tgt)
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
            samples_per_pointing = 5
            duration = samples_per_pointing*degrees//step*0.5
            for x in np.arange(-nrsteps,nrsteps,1):
                T1 = time.time()
                offset = x * step
                print "Scan %i: (%.2f,%.2f,%.2f,%.2f)" % (x, -0.5*degrees, offset, 0.5*degrees,offset)
                if opts.scan_in_elevation:
                    kat.ant1.req.scan_asym(offset, -0.5*degrees, offset, 0.5*degrees, duration, "stereographic")
                else:
                    kat.ant1.req.scan_asym(-0.5*degrees, offset, 0.5*degrees, offset, duration, "stereographic")
                kat.ant1.wait("lock",1,300)
                # wait for lock at start of scan
                T2 = time.time()
                print ">> %.1f seconds cal -> start of scan"%(T2-T1)
                kat.dbe.req.k7w_new_scan('scan')
                kat.ant1.req.mode("SCAN")
                kat.ant1.wait('scan_status', 'after', int(duration)+30) # Allow extra for projection stretch
                # wait for scan to finish
                T3 = time.time()
                print ">> %.1f seconds to scan"%(T3-T2)
                sys.stdout.flush()
                print "Finished scan. Doing cal..."
                kat.ant1.req.mode("POINT")
                kat.dbe.req.k7w_new_scan('slew')
                #kat.ant1.req.target_azel(t_az,t_el)
                # Alternative way to set the offset while still tracking the TLE
                kat.ant1.req.target(tgt.description)
                kat.ant1.req.offset_fixed(az,el,'stereographic')
                kat.ant1.wait("lock",1,300)
                T4 = time.time()
                print ">> %.1f seconds scan - > start of cal"%(T4-T3)
                kat.dbe.req.k7w_new_scan('cal')
                time.sleep(5)
                # wait for 5seconds on cal
                kat.dbe.req.k7w_new_scan('slew')
                T5 = time.time()
                print ">> Scan & cal took %.1f seconds"%(T5-T1)
                sys.stdout.flush()
        print "Session started at %s completed at %s, after %.f mins\n"%(time.ctime(T0),time.ctime(),(time.time()-T0)/60.)
        print "="*40
    print "Done..."
