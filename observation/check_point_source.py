#! /usr/bin/python
# Do mini-raster scan on nearest strong point source and reduce data to check system performance.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os
import time
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from katuilib.observe import standard_script_options, verify_and_connect, lookup_targets, start_session, user_logger
import arutils
import katpoint
import katfile
import scape

# Set up standard script options
parser = standard_script_options(usage="%prog [options] ['target']",
                                 description='Perform mini raster scan across (nearest) point source and reduce '
                                             'data to check system performance. Some options are **required**.')
# Add experiment-specific options
parser.add_option('-c', '--channels', default='100,400',
                  help="Frequency channel range to keep (zero-based, specified as 'start,end', default='%default')")
parser.add_option('--no-plots', action='store_true', default=False, help='Disable plotting')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source check', observer='check', nd_params='coupler,0,0,-1', dump_rate=2.0)
# Parse the command line
opts, args = parser.parse_args()

# Frequency channels to keep
start_freq_channel = int(opts.channels.split(',')[0])
end_freq_channel = int(opts.channels.split(',')[1])
if opts.no_plots:
    plt = None

with verify_and_connect(opts) as kat:

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

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
            strong_sources = kat.sources.filter(el_limit_deg=[15, 75], flux_limit_Jy=100, flux_freq_MHz=opts.centre_freq)
            target = strong_sources.targets[np.argmin([t.separation(current_pos) for t in strong_sources])]

        session.fire_noise_diode('coupler', 4, 4)
        session.raster_scan(target, num_scans=3, scan_duration=15, scan_extent=5.0, scan_spacing=0.5)

if not opts.dry_run:
    cfg = kat.config_file
    h5file = session.output_file
    if not h5file:
        raise RuntimeError('Could not obtain name of HDF5 file that was recorded')

    # Obtain archive where file is stored
    archive_name = session.dbe.sensor.archiver_archive.get_value()
    archive = arutils.DataArchive(archive_name)
    # Wait until desired HDF5 file appears in the archive (this could take quite a while...)
    # For now, the timeout option is disabled, as it is safe to wait until the user quits the script
    user_logger.info("Waiting for HDF5 file '%s' to appear in archive" % (h5file,))
    # timeout = 300
    # while timeout > 0:
    while True:
        if archive:
            ar = arutils.ArchiveBrowser(archive)
            ar.filter_by_filenames(h5file)
            if len(ar.kath5s) > 0:
                break
        elif os.path.exists(h5file):
            break
        time.sleep(1)
    #     timeout -= 1
    # if timeout == 0:
    #     raise RuntimeError("Timed out waiting for HDF5 file '%s' to appear in '%s' archive" %
    #                        (h5file, archive.name if archive else 'local'))
    # Copy file to local machine if needed
    if archive:
        os.system(ar.generate_script())
    if not os.path.isfile(h5file):
        raise RuntimeError("Could not copy file '%s' from archive to local machine" % (h5file,))

    # Obtain list of antennas and polarisations present in data set
    user_logger.info('Loading HDF5 file into scape and reducing the data')
    h5 = katfile.h5_data(h5file)
    # Iterate through antennas
    for ant in h5.ants:
        ant_num = int(ant.name[3:])
        # Load file and do standard processing
        d = scape.DataSet(h5file, baseline='A%dA%d' % (ant_num, ant_num))
        d = d.select(freqkeep=range(start_freq_channel, end_freq_channel + 1))
        channel_freqs = d.freqs
        d.convert_power_to_temperature()
        d = d.select(labelkeep='scan', copy=False)
        d.average()
        # Only use the first compound scan for fitting beam and baseline
        compscan = d.compscans[0]
        # Calculate average target flux over entire band
        flux_spectrum = [compscan.target.flux_density(freq) for freq in channel_freqs]
        average_flux = np.mean([flux for flux in flux_spectrum if flux])
        # Fit individual polarisation beams first, to get gains and system temperatures
        gain_hh, gain_vv = None, None
        baseline_hh, baseline_vv = None, None
        if (ant.name + 'H') in h5.inputs:
            d.fit_beams_and_baselines(pol='HH', circular_beam=False)
            if compscan.beam is not None and d.data_unit == 'K':
                gain_hh = compscan.beam.height / average_flux
                baseline_hh = compscan.baseline_height()
        if (ant.name + 'V') in h5.inputs:
            d.fit_beams_and_baselines(pol='VV', circular_beam=False)
            if compscan.beam is not None and d.data_unit == 'K':
                gain_vv = compscan.beam.height / average_flux
                baseline_vv = compscan.baseline_height()
        d.fit_beams_and_baselines(pol='I', circular_beam=True)
        beam = compscan.beam
        # Obtain middle timestamp of compound scan, where all pointing calculations are done
        compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
        middle_time = np.median(compscan_times, axis=None)
        # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
        requested_azel = compscan.target.azel(middle_time)
        requested_azel = katpoint.rad2deg(np.array(requested_azel))
        # The offset is very simplistic and doesn't take into account refraction (see a_p_s_s for more correct way)
        offset_azel = katpoint.rad2deg(np.array(beam.center)) if beam else np.zeros(2)

        user_logger.info("Antenna %s" % (ant.name,))
        user_logger.info("------------")
        user_logger.info("Target = '%s', azel around (%.1f, %.1f) deg" %
                         (compscan.target.name, requested_azel[0], requested_azel[1]))
        if beam is None:
            user_logger.info("No total power beam found")
        else:
            user_logger.info("Beam height = %g %s" % (beam.height, d.data_unit))
            user_logger.info("Beamwidth = %.1f' (expected %.1f')" %
                             (60 * katpoint.rad2deg(beam.width), 60 * katpoint.rad2deg(beam.expected_width)))
            user_logger.info("Beam offset = (%.1f', %.1f') (expected (0', 0'))" %
                             (60 * offset_azel[0], 60 * offset_azel[1]))
        if gain_hh is None:
            user_logger.info("HH parameters could not be determined (no HH data or noise diode cal / beam fit failed)")
        else:
            user_logger.info("HH gain = %.5f K/Jy" % (gain_hh,))
            user_logger.info("HH Tsys = %.1f K" % (baseline_hh,))
        if gain_vv is None:
            user_logger.info("VV parameters could not be determined (no VV data or noise diode cal / beam fit failed)")
        else:
            user_logger.info("VV gain = %.5f K/Jy" % (gain_vv,))
            user_logger.info("VV Tsys = %.1f K" % (baseline_vv,))

        if plt is not None:
            plt.figure(ant_num, figsize=(10, 10))
            plt.clf()
            plt.subplots_adjust(bottom=0.3)
            plt.subplot(211)
            scape.plot_compound_scan_in_time(compscan)
            plt.title("%s '%s'\nazel=(%.1f, %.1f) deg, offset=(%.1f', %.1f')" %
                      (d.antenna.name, compscan.target.name, requested_azel[0], requested_azel[1],
                       60. * offset_azel[0], 60. * offset_azel[1]), size='medium')
            plt.ylabel('Total power (%s)' % d.data_unit)
            plt.subplot(212)
            scape.plot_compound_scan_on_target(compscan)
            # Print additional info as text labels
            beam_str = "Beamwidth = %.1f' (expected %.1f')\nBeam height = %g %s" % \
                       (60 * katpoint.rad2deg(beam.width), 60 * katpoint.rad2deg(beam.expected_width),
                        beam.height, d.data_unit) if beam is not None else "No I beam"
            plt.figtext(0.05, 0.2, beam_str, va='top', ha='left')
            hh_str = "HH gain = %.5f K/Jy\nHH Tsys = %.1f K" % (gain_hh, baseline_hh) \
                     if gain_hh is not None else "No HH parameters"
            plt.figtext(0.05, 0.1, hh_str, va='top', ha='left')
            vv_str = "VV gain = %.5f K/Jy\nVV Tsys = %.1f K" % (gain_vv, baseline_vv) \
                     if gain_vv is not None else "No VV parameters"
            plt.figtext(0.55, 0.1, vv_str, va='top', ha='left')
            plt.show()

    os.remove(h5file)
