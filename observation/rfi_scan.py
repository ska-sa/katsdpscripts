#!/usr/bin/python
# run a simple scan script to derive the horizon mask for KAT-7
# scan over constant elevation range but loop over azimuth

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement
import numpy as np
import time
from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger

# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform a rfi scan with the KAT-7. Scan over constant elevation "
                                             "with 3 scans at 3.1,9.1,15.1 degrees. This takes the form of 2x180 raster scans "
                                             "in opposite directions, with 180 seconds per scan. "
                                             "There are non-optional options.(Antennas)")
# Add experiment-specific options
parser.add_option('-m','--min-duration', dest='min_duration', type="float", default=None,
                  help="The The minimum time to repeat the rfi scan over (default=%default)")
parser.remove_option('-f')
parser.remove_option('-r')
parser.remove_option('-p')
parser.add_option('-f', '--centre-freq', default='1328.0,1575.0,1822.0',
                      help='Centre frequency, in MHz')
## Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Basic RFI Scan')
# Parse the command line
opts, args = parser.parse_args()

opts.description = ("Basic RFI Scan: %s" % (opts.description,)) if opts.description != "Basic RFI Scan" else opts.description

el_start =  3.1
el_end =15.1
scan_spacing = 6.0
num_scans = 3
scan_duration = 180.
scan_extent = 180.
freq = np.array(opts.centre_freq.split(',')).astype(float).tolist()
opts.centre_freq = freq[0]
opts.dump_rate = 1.
with verify_and_connect(opts) as kat:
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        if not kat.dry_run:
            if session.dbe.req.auto_delay('off'):
                user_logger.info("Turning off delay tracking.")
            else:
                user_logger.error('Unable to turn off delay tracking.')
            if session.dbe.req.zero_delay():
                user_logger.info("Zeroed delay values.")
            else:
                user_logger.error('Unable to zero the delay values.')
        session.capture_start()
        start_time = time.time()
        nd_params = session.nd_params.copy()
        nd_params['period'] =  0
        end_loop_now = False
        while ( opts.min_duration is None or (time.time() - start_time) < opts.min_duration) and not end_loop_now:
            if opts.min_duration is None: end_loop_now = True
            for curr_freq in freq:
                if  (time.time() - start_time)< opts.min_duration or  opts.min_duration is None:
                    opts.centre_freq = curr_freq  # Not needed
                    user_logger.info("Change Frequency to %d MHz" % (float(curr_freq)))
                    if not kat.dry_run: kat.rfe7.req.rfe7_lo1_frequency(4200.0 + float(curr_freq), 'MHz')
                    session.fire_noise_diode(announce=False, **nd_params) #
                    # First Half
                    scan_time = time.time()
                    azimuth_angle = abs(-90.0 - 270.0) / 4. # should be 90 deg.
                    target1 = 'azel, %f, %f' % (-90. + azimuth_angle, (el_end + el_start) / 2.)
                    session.label('raster')
                    session.raster_scan(target1, num_scans=num_scans, scan_duration=scan_duration, scan_extent=scan_extent,
                                        scan_spacing=scan_spacing, scan_in_azimuth=True, projection='plate-carree')
                    user_logger.info("Observed horizon part 1/2 for %d seconds" % (time.time() - scan_time))
                    # Second Half
                    half_time = time.time()
                    target2 = 'azel, %f, %f' % (-90. + azimuth_angle * 3., (el_end + el_start) / 2.)
                    session.label('raster')
                    session.raster_scan(target2, num_scans=num_scans, scan_duration=scan_duration, scan_extent=scan_extent,
                                        scan_spacing=scan_spacing, scan_in_azimuth=True, projection='plate-carree')
                    user_logger.info("Observed horizon part 2/2 for %d Seconds (%d Seconds in Total)" %
                                     ((time.time() - half_time), (time.time() - start_time)))
if kat.dry_run : user_logger.info("!! Dry run time is not Accurate !!   Assume a time of about 1400 seconds per frequency for the scan. or 70 Miniuts for the default frequency set" )
