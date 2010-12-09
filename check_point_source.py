#! /usr/bin/python
# Do mini-raster scan on nearest strong point source and reduce data to check system performance.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os
import time
import numpy as np
try:
    import matplotlib.pyplot as plt
    plot = True
except ImportError:
    plot = False
import h5py

from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, \
                             CaptureSession, TimeSession, user_logger
import arutils
import katpoint
import scape

# Set up standard script options
parser = standard_script_options(usage="%prog [options] ['target']",
                                 description="Perform mini raster scan across (nearest) point source and reduce \
                                              data to check system performance. Some options are **required**.")
# Add experiment-specific options
parser.add_option("-c", "--channels", default='100,400',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source check', observer='check', nd_params='coupler,0,0,-1', dump_rate=2.0)
# Parse the command line
opts, args = parser.parse_args()

# Frequency channels to keep
start_freq_channel = int(opts.channels.split(',')[0])
end_freq_channel = int(opts.channels.split(',')[1])

with verify_and_connect(opts) as kat:

    # Select either a CaptureSession for the real experiment, or a fake TimeSession
    Session = TimeSession if opts.dry_run else CaptureSession
    with Session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))

        # Pick a target, either explicitly or the closest strong one
        if len(args) > 0:
            target = lookup_targets(kat, [args[0]])[0]
        else:
            # Get current position of first antenna in the list (assume the rest are the same or close)
            if opts.dry_run:
                current_az, current_el = session.ants[0][2:]
            else:
                current_az = session.ants.devs[0].sensor.pos_actual_scan_azim.get_value()
                current_el = session.ants.devs[0].sensor.pos_actual_scan_elev.get_value()
            current_pos = katpoint.construct_azel_target(katpoint.deg2rad(current_az), katpoint.deg2rad(current_el))
            # Get closest strong source that is up
            strong_sources = kat.sources.filter(el_limit_deg=15, flux_limit_Jy=100, flux_freq_MHz=opts.centre_freq)
            target = strong_sources.targets[np.argmin([t.separation(current_pos) for t in strong_sources])]

        session.fire_noise_diode('coupler', 4, 4)
        session.raster_scan(target, num_scans=3, scan_duration=15, scan_extent=5.0, scan_spacing=0.5)

    # Obtain the name of the file currently being written to
    reply = kat.dbe.req.k7w_get_current_file()
    if not reply.succeeded:
        raise RuntimeError('Could not obtain name of HDF5 file that was recorded')
        
    cfg = kat.config_file
    h5path = reply[1].replace('writing.', '')
    h5file = os.path.basename(h5path) if cfg.find('local') < 0 else h5path
#    simulated_dbe = hasattr(kat.dbe.req, 'dbe_test_target') and hasattr(kat.dbe.req, 'dbe_pointing_az')

if not opts.dry_run:
    ### HACK TO DEDUCE CORRECT ARCHIVE ###
    archive = None
    if cfg.find('karoo') >= 0:
        archive = arutils.karoo_archive
    elif cfg.find('lab') >= 0:
        archive = arutils.lab_archive
    elif cfg.find('local') < 0:
        raise RuntimeError("Could not deduce archive associated with configuration '%s'" % cfg)
    # Wait until desired HDF5 file appears in the archive (or a timeout occurs)
    user_logger.info("Waiting for HDF5 file '%s' to appear in archive" % (h5file,))
#    timeout = 40
#    while timeout > 0:
    while True:
        if archive:
            ar = arutils.ArchiveBrowser(archive)
            ar.filter_by_filename(h5file)
            if len(ar.kath5s) > 0:
                break
        elif os.path.exists(h5file):
            break
        time.sleep(1)
#        timeout -= 1
#    if timeout == 0:
#        raise RuntimeError("Timed out waiting for HDF5 file '%s' to appear in '%s' archive" %
#                           (h5file, archive.name if archive else 'local'))
    # Copy file to local machine if needed
    if archive:
        os.system(ar.generate_script())
    if not os.path.isfile(h5file):
        raise RuntimeError("Could not copy file '%s' from archive to local machine" % (h5file,))

    # Obtain list of antennas present in data set
    user_logger.info('Loading HDF5 file into scape and reducing the data')
    f = h5py.File(h5file, 'r')
    antennas = [int(ant[7:]) for ant in f['Antennas'].iternames()]
    f.close()
    # Iterate through antennas
    for ant in antennas:
        # Load file and do standard processing
        d = scape.DataSet(h5file, baseline='A%dA%d' % (ant, ant))
        d = d.select(freqkeep=range(start_freq_channel, end_freq_channel + 1))
        channel_freqs = d.freqs
        d.convert_power_to_temperature()
        d = d.select(labelkeep='scan', copy=False)
        d.average()
        d.fit_beams_and_baselines()
        # Only use the first compound scan for fitting beam and baseline
        compscan = d.compscans[0]
        beam = compscan.beam
        baseline = compscan.baseline_height()
        # Calculate average target flux over entire band
        flux_spectrum = [compscan.target.flux_density(freq) for freq in channel_freqs]
        average_flux = np.mean([flux for flux in flux_spectrum if flux])
        # Obtain middle timestamp of compound scan, where all pointing calculations are done
        compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
        middle_time = np.median(compscan_times, axis=None)
        # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
        requested_azel = compscan.target.azel(middle_time)
        requested_azel = katpoint.rad2deg(np.array(requested_azel))
        # The offset is very simplistic and doesn't take into account refraction (see a_p_s_s for more correct way)
        offset_azel = katpoint.rad2deg(np.array(beam.center)) if beam else np.zeros(2)

        user_logger.info("Antenna %d" % (ant,))
        user_logger.info("---------")
        if beam is None:
            user_logger.info("no beam found")
        else:
            user_logger.info("beam height = %g %s" % (beam.height, d.data_unit))
            user_logger.info("beam width = %g arcmin" % (60 * katpoint.rad2deg(beam.width),))
            user_logger.info("beam offset = (%g, %g) arcmin" % (60 * offset_azel[0], 60 * offset_azel[1]))
        if baseline is None:
            user_logger.info("no baseline found")
        else:
            user_logger.info("baseline height = %g %s" % (baseline, d.data_unit))

        if plot:
            plt.figure(ant)
            plt.clf()
            plt.subplot(211)
            scape.plot_compound_scan_in_time(compscan)
            plt.title("%s '%s'\nazel=(%.1f, %.1f) deg, offset=(%.1f, %.1f) arcmin" %
                      (d.antenna.name, compscan.target.name, requested_azel[0], requested_azel[1],
                       60. * offset_azel[0], 60. * offset_azel[1]), size='medium')
            plt.ylabel('Total power (%s)' % d.data_unit)
            plt.subplot(212)
            scape.plot_compound_scan_on_target(compscan)
            plt.show()
            # if beam:
            #     plt.figtext(0.05, 0.05, ("Beamwidth = %.1f' (expected %.1f')\n" +
            #                              "Beam height = %.1f %s\n" +
            #                              "HH/VV gain = %.3f/%.3f Jy/%s\n" +
            #                              "Baseline height = %.1f %s") %
            #                   (60. * katpoint.rad2deg(beam.width),
            #                    60. * katpoint.rad2deg(beam.expected_width),
            #                    beam.height, d.data_unit,
            #                    average_flux / , average_flux / beam_params[8], current_dataset.data_unit,
            #                    baseline_height_I, current_dataset.data_unit)
            #     , va='bottom', ha='left')
                
    os.remove(h5file)
