#!/usr/bin/python
# Example script that uses scape to reduce data consisting of scans across multiple point sources.
#
# This can be used to determine gain curves, tipping curves and pointing models.
# The user can interactively observe reduction results and discard bad data. The
# end product is a file containing pointing, fitted beam parameters, baseline
# height and weather measurements, etc.
#
# Ludwig Schwardt
# 13 July 2009
#

from __future__ import with_statement

import os.path
import logging
import optparse
import glob

import numpy as np

import scape
import katpoint

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <directories or files>",
                               description="This processes one or more datasets (FITS or HDF5) and extracts \
                                            fitted beam parameters from them. It runs interactively by default, \
                                            which allows the user to inspect results and discard bad scans. \
                                            By default all datasets in the current directory and all \
                                            subdirectories are processed.")
parser.add_option('-a', '--baseline', dest='baseline', type="string", metavar='BASELINE', default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-b", "--batch", dest="batch", action="store_true",
                  help="True if processing is to be done in batch mode without user interaction")
parser.add_option("-c", "--catalogue", dest="catfilename", type="string", default='',
                  help="Name of optional source catalogue file used to override XDM FITS targets")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='90,424',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
parser.add_option("-k", "--keep", dest="keepfilename", type="string", default='',
                  help="Name of optional CSV file used to select compound scans from datasets (implies batch mode)")
parser.add_option("-n", "--nd_models", dest="nd_dir", type="string", default='',
                  help="Name of optional directory containing noise diode model files")
parser.add_option("-o", "--output", dest="outfilebase", type="string", default='point_source_scans',
                  help="Base name of output files (*.csv for output data and *.log for messages)")
parser.add_option("-p", "--pointing_model", dest="pmfilename", type="string", default='',
                  help="Name of optional file containing pointing model parameters in degrees (needed for XDM)")
parser.add_option("-s", "--plot_spectrum", dest="plot_spectrum", action="store_true",
                  help="True to include spectral plot")

(opts, args) = parser.parse_args()
if len(args) < 1:
    args = ['.']

# Set up logging: logging everything (DEBUG & above), both to console and file
logger = logging.root
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(opts.outfilebase + '.log', 'w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(fh)

# Load catalogue used to convert ACSM targets to katpoint ones (only needed for XDM data files)
cat = None
if opts.catfilename:
    cat = katpoint.Catalogue(file(opts.catfilename))
    logger.debug("Loaded catalogue with %d source(s) from '%s'" % (len(cat.targets), opts.catfilename))
# Load old pointing model parameters (useful if it is not in data file, like on XDM and early KAT-7)
pm = None
if opts.pmfilename:
    pm = file(opts.pmfilename).readline().strip()
    logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(pm.split(',')), opts.pmfilename))
# Load old CSV file used to select compound scans from datasets
keep_scans = keep_datasets = None
if opts.keepfilename:
    ant_name = katpoint.Antenna(file(opts.keepfilename).readline().strip().partition('=')[2]).name
    try:
        data = np.loadtxt(opts.keepfilename, dtype='string', comments='#', delimiter=', ')
    except ValueError:
        raise ValueError("CSV file '%s' contains rows with a different number of columns/commas" % opts.keepfilename)
    try:
        fields = data[0].tolist()
        id_fields = [fields.index('dataset'), fields.index('target'), fields.index('timestamp_ut')]
    except (IndexError, ValueError):
        raise ValueError("CSV file '%s' do not have the expected columns" % opts.keepfilename)
    keep_scans = set([ant_name + ' ' + ' '.join(line) for line in data[1:, id_fields]])
    keep_datasets = set(data[1:, id_fields[0]])
    # Switch to batch mode if CSV file is given
    opts.batch = True
    logger.debug("Loaded CSV file '%s' containing %d dataset(s) and %d compscan(s)" %
                 (opts.keepfilename, len(keep_datasets), len(keep_scans)))
# Frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])

# Only import matplotlib if not in batch mode
if not opts.batch:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets

# Find all data sets (HDF5 or FITS) mentioned, and add them to datasets
datasets = []
def walk_callback(arg, directory, files):
    datasets.extend([os.path.join(directory, f) for f in files if f.endswith('.h5') or f.endswith('_0000.fits')])
for arg in args:
    if os.path.isdir(arg):
        os.path.walk(arg, walk_callback, None)
    else:
        datasets.extend(glob.glob(arg))
if len(datasets) == 0:
    raise ValueError('No data sets (HDF5 or XDM FITS) found')
# Indices to step through data sets and compound scans as the buttons are pressed
dataset_index = compscan_index = 0
# Remember current data set (useful when iterating through multiple compscans inside the set)
current_dataset = None
unaveraged_dataset = None
beam_data = [[] for dataset in datasets]
output_data = []
antenna = None

def dataset_name(filename):
    """Convert filename to more compact data set name."""
    if filename.endswith('.fits'):
        dirs = filename.split(os.path.sep)
        if dirs[0] == '.':
            dirs.pop(0)
        if len(dirs) > 2:
            return '%s_%s' % tuple(dirs[-3:-1])
        else:
            return '%s' % (dirs[0],)
    else:
        return os.path.splitext(os.path.basename(filename))[0]

def load_reduce(index):
    """Load data set and do data reduction on data set level, storing beam fits per compound scan."""
    # Global variables that will be modified inside this function
    global unaveraged_dataset, current_dataset, beam_data, antenna

    filename = datasets[index]
    # Avoid loading the data set if it does not appear in specified CSV file
    if keep_datasets and dataset_name(filename) not in keep_datasets:
        logger.info("Skipping dataset '%s' (based on CSV file)" % (filename,))
        return False
    logger.info("Loading dataset '%s'" % (filename,))
    current_dataset = scape.DataSet(filename, catalogue=cat, baseline=opts.baseline)

    # Skip data set if antenna differs from the first antenna found, or no scans found
    if antenna is None or (antenna.name == current_dataset.antenna.name):
        antenna = current_dataset.antenna
    else:
        logger.warning('Data set has different antenna (expected "%s", found "%s"), skipping data set' %
                       (antenna.name, current_dataset.antenna.name))
        return False
    if len(current_dataset.compscans) == 0 or len(current_dataset.scans) == 0:
        logger.warning('No scans found in file, skipping data set')
        return False
    # Override pointing model if it is specified
    if pm is not None:
        antenna.pointing_model = katpoint.PointingModel(pm, strict=False)

    # Standard reduction for XDM, more hard-coded version for FF / KAT-7
    if antenna.name == 'XDM':
        current_dataset = current_dataset.select(freqkeep=current_dataset.channel_select)
        current_dataset.convert_power_to_temperature()
    else:
        # Hard-code the FF frequency band
        current_dataset = current_dataset.select(freqkeep=range(start_freq_channel, end_freq_channel+1))
        # If noise diode models are supplied, insert them into data set before converting to temperature
        if antenna.name[:3] == 'ant' and os.path.isdir(opts.nd_dir):
            try:
                nd_hpol_file = os.path.join(opts.nd_dir, 'T_nd_A%sH_coupler.txt' % (antenna.name[3],))
                nd_vpol_file = os.path.join(opts.nd_dir, 'T_nd_A%sV_coupler.txt' % (antenna.name[3],))
                logger.info("Loading noise diode model '%s'" % (nd_hpol_file,))
                nd_hpol = np.loadtxt(nd_hpol_file, delimiter=',')
                logger.info("Loading noise diode model '%s'" % (nd_vpol_file,))
                nd_vpol = np.loadtxt(nd_vpol_file, delimiter=',')
                nd_hpol[:, 0] /= 1e6
                nd_vpol[:, 0] /= 1e6
                current_dataset.nd_model = scape.gaincal.NoiseDiodeModel(nd_hpol, nd_vpol, std_temp=0.04)
                current_dataset.convert_power_to_temperature()
            except IOError:
                logger.warning('Could not load noise diode model files, should be named T_nd_A1H_coupler.txt etc.')
    current_dataset = current_dataset.select(labelkeep='scan', copy=False)
    # Make a copy of the dataset before averaging the channels so that we keep the spectral information
    unaveraged_dataset = current_dataset.select(copy=True)
    current_dataset.average()
    if len(current_dataset.compscans) == 0 or len(current_dataset.scans) == 0:
        logger.warning('No scans left after standard reduction, skipping data set (no scans labelled "scan", perhaps?)')
        return False

    # First fit HH and VV data, and extract beam and baseline heights and refined scan count
    current_dataset.fit_beams_and_baselines(pol='HH', circular_beam=False)
    beam_height_HH = [compscan.beam.height if compscan.beam else np.nan for compscan in current_dataset.compscans]
    beam_width_HH = [katpoint.rad2deg(np.mean(compscan.beam.width)) if compscan.beam else np.nan
                     for compscan in current_dataset.compscans]
    baseline_height_HH = [compscan.baseline_height() for compscan in current_dataset.compscans]
    baseline_height_HH = [bh if bh is not None else np.nan for bh in baseline_height_HH]
    refined_HH = [compscan.beam.refined if compscan.beam else 0 for compscan in current_dataset.compscans]

    current_dataset.fit_beams_and_baselines(pol='VV', circular_beam=False)
    beam_height_VV = [compscan.beam.height if compscan.beam else np.nan for compscan in current_dataset.compscans]
    beam_width_VV = [katpoint.rad2deg(np.mean(compscan.beam.width)) if compscan.beam else np.nan
                     for compscan in current_dataset.compscans]
    baseline_height_VV = [compscan.baseline_height() for compscan in current_dataset.compscans]
    baseline_height_VV = [bh if bh is not None else np.nan for bh in baseline_height_VV]
    refined_VV = [compscan.beam.refined if compscan.beam else 0 for compscan in current_dataset.compscans]

    # Now fit Stokes I, as this will be used for pointing and plots as well
    current_dataset.fit_beams_and_baselines(pol='I')
    # Calculate beam and baseline height and refined scan count
    beam_height_I = [compscan.beam.height if compscan.beam else np.nan for compscan in current_dataset.compscans]
    beam_width_I = [katpoint.rad2deg(np.mean(compscan.beam.width)) if compscan.beam else np.nan
                    for compscan in current_dataset.compscans]
    baseline_height_I = [compscan.baseline_height() for compscan in current_dataset.compscans]
    baseline_height_I = [bh if bh is not None else np.nan for bh in baseline_height_I]
    refined_I = [compscan.beam.refined if compscan.beam else 0 for compscan in current_dataset.compscans]

    beam_data[index] = np.array([beam_height_I, beam_width_I, baseline_height_I, refined_I,
                                 beam_height_HH, beam_width_HH, baseline_height_HH, refined_HH,
                                 beam_height_VV, beam_width_VV, baseline_height_VV, refined_VV]).transpose()
    return True

def next_load_reduce_plot(fig=None):
    """Load and reduce next data set, update the plots in given figure and store output data."""
    # Global variables that will be modified inside this function
    global dataset_index, compscan_index, output_data
    # Extract plot axes
    if not opts.batch:
        if opts.plot_spectrum:
            (ax1, ax2, ax3), info = fig.axes[:3], fig.texts[0]
        else:
            (ax1, ax2), info = fig.axes[:2], fig.texts[0]
    # Total number of compound scans in data sets prior to the current one
    compscans_in_previous_datasets = np.sum([len(bd) for bd in beam_data[:dataset_index]], dtype=np.int)
    # Move to next compound scan
    if current_dataset is not None:
        compscan_index += 1
    # Load next data set if last compscan has been reached
    if (dataset_index >= len(datasets)) or \
       (compscan_index - compscans_in_previous_datasets >= len(beam_data[dataset_index])):
        if current_dataset is not None:
            dataset_index += 1
        # If there are no more data sets, save output data to file and exit
        if dataset_index >= len(datasets):
            f = file(opts.outfilebase + '.csv', 'w')
            f.write('# antenna = %s\n' % antenna.description)
            f.write('dataset, target, timestamp_ut, azimuth, elevation, delta_azimuth, delta_elevation, data_unit, ' +
                    'beam_height_I, beam_width_I, baseline_height_I, refined_I, beam_height_HH, beam_width_HH, ' +
                    'baseline_height_HH, refined_HH, beam_height_VV, beam_width_VV, baseline_height_VV, refined_VV, ' +
                    'frequency, flux, temperature, pressure, humidity, wind_speed\n')
            f.writelines([(('%s, %s, %s, %.7f, %.7f, %.7f, %.7f, %s, %.7f, %.7f, %.7f, %d, %.7f, %.7f, %.7f, %d, ' +
                            '%.7f, %.7f, %.7f, %d, %.7f, %.4f, %.2f, %.2f, %.2f, %.2f\n') % tuple(p))
                          for p in output_data if p])
            f.close()
            if not opts.batch:
                plt.close('all')
            return False
        # Load next data set
        loaded = load_reduce(dataset_index)
        name = dataset_name(datasets[dataset_index])
        if not loaded:
            if not opts.batch:
                ax1.clear()
                ax1.set_title("%s - data set skipped" % name, size='medium')
                ax2.clear()
                if opts.plot_spectrum:
                    ax3.clear()
                plt.draw()
            return True
        compscans_in_previous_datasets = np.sum([len(bd) for bd in beam_data[:dataset_index]], dtype=np.int)

    # Select current compound scan and related beam data
    compscan = current_dataset.compscans[compscan_index - compscans_in_previous_datasets]
    unaveraged_compscan = unaveraged_dataset.compscans[compscan_index - compscans_in_previous_datasets]
    name = dataset_name(datasets[dataset_index])
    beam_params = beam_data[dataset_index][compscan_index - compscans_in_previous_datasets].tolist()
    beam_height_I, baseline_height_I = beam_params[0], beam_params[2]

    # Calculate average target flux over entire band
    flux_spectrum = [compscan.target.flux_density(freq) for freq in unaveraged_dataset.freqs]
    average_flux = np.mean([flux for flux in flux_spectrum if flux])

    # Interpolate environmental sensor data
    def interp_sensor(quantity, default):
        try:
            sensor = current_dataset.enviro[quantity]
        except KeyError:
            return (lambda times: default)
        else:
            interp = scape.fitting.PiecewisePolynomial1DFit(max_degree=0)
            interp.fit(sensor['timestamp'], sensor['value'])
            return interp
    # Obtain environmental data averaged across the compound scan
    compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
    temperature = np.mean(interp_sensor('temperature', 35.0)(compscan_times))
    pressure = np.mean(interp_sensor('pressure', 950.0)(compscan_times))
    humidity = np.mean(interp_sensor('humidity', 15.0)(compscan_times))
    wind_speed = np.mean(interp_sensor('wind_speed', 0.0)(compscan_times))

    # Calculate pointing offset
    # Obtain middle timestamp of compound scan, where all pointing calculations are done
    middle_time = np.median(compscan_times, axis=None)
    # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
    requested_azel = compscan.target.azel(middle_time)
    # Correct for refraction, which becomes the requested value at input of pointing model
    rc = katpoint.RefractionCorrection()
    requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
    requested_azel = katpoint.rad2deg(np.array(requested_azel))
    if compscan.beam:
        # Fitted beam center is in (x, y) coordinates, in projection centred on target
        beam_center_xy = compscan.beam.center
        # Convert this offset back to spherical (az, el) coordinates
        beam_center_azel = compscan.target.plane_to_sphere(beam_center_xy[0], beam_center_xy[1], middle_time)
        # Now correct the measured (az, el) for refraction and then apply the old pointing model
        # to get a "raw" measured (az, el) at the output of the pointing model
        beam_center_azel = [beam_center_azel[0], rc.apply(beam_center_azel[1], temperature, pressure, humidity)]
        beam_center_azel = current_dataset.antenna.pointing_model.apply(*beam_center_azel)
        beam_center_azel = katpoint.rad2deg(np.array(beam_center_azel))
        # Make sure the offset is a small angle around 0 degrees
        offset_azel = scape.stats.angle_wrap(beam_center_azel - requested_azel, 360.)
    else:
        offset_azel = np.array([np.nan, np.nan])

    # Display compound scan
    if not opts.batch:
        ax1.clear()
        scape.plot_compound_scan_in_time(compscan, ax=ax1)
        ax1.set_title("%s %s '%s'\nazel=(%.1f, %.1f) deg, offset=(%.1f, %.1f) arcmin" %
                      (name, antenna.name, compscan.target.name, requested_azel[0], requested_azel[1],
                       60. * offset_azel[0], 60. * offset_azel[1]), size='medium')
        ax1.set_ylabel('Total power (%s)' % current_dataset.data_unit)
        ax2.clear()
        scape.plot_compound_scan_on_target(compscan, ax=ax2)
        if opts.plot_spectrum:
            ax3.clear()
            scape.plot_xyz(unaveraged_compscan, 'freq', 'amp', labels=[], power_in_dB=True, ax=ax3)
        if compscan.beam:
            info.set_text(("Beamwidth = %.1f' (expected %.1f')\nBeam height = %.1f %s\n" +
                           "HH/VV gain = %.3f/%.3f Jy/%s\nBaseline height = %.1f %s") %
                          (60. * katpoint.rad2deg(compscan.beam.width),
                           60. * katpoint.rad2deg(compscan.beam.expected_width),
                           beam_height_I, current_dataset.data_unit,
                           average_flux / beam_params[4], average_flux / beam_params[8], current_dataset.data_unit,
                           baseline_height_I, current_dataset.data_unit))
        else:
            info.set_text("No beam\nBaseline height = %.2f %s" % (baseline_height_I, current_dataset.data_unit))
        plt.draw()

    # If list of scans to keep are provided, follow it religiously
    if keep_scans:
        # Look up compscan identity in list of compscans to keep (if provided)
        compscan_key = ' '.join([antenna.name, name, compscan.target.name, str(katpoint.Timestamp(middle_time))])
        keep_compscan = compscan_key in keep_scans
        logger.info("%s compscan '%s' (based on CSV file)" % ('Keeping' if keep_compscan else 'Skipping', compscan_key))
    else:
        # If beam is marked as invalid, discard scan only if in batch mode (otherwise discard button has to do it)
        keep_compscan = compscan.beam and (not opts.batch or compscan.beam.is_valid)
    if keep_compscan:
        output_data.append([name, compscan.target.name, katpoint.Timestamp(middle_time),
                            requested_azel[0], requested_azel[1], offset_azel[0], offset_azel[1],
                            current_dataset.data_unit] + beam_params + [current_dataset.freqs.mean(),
                            average_flux, temperature, pressure, humidity, wind_speed])
    else:
        output_data.append(None)

    # Indicate that more data is to come
    return True

### BATCH MODE ###

# This will cycle through all data sets and stop when done
if opts.batch:
    while next_load_reduce_plot():
        pass

### INTERACTIVE MODE ###
else:
    # Set up figure with buttons
    plt.ion()
    fig = plt.figure(1)
    plt.clf()
    if opts.plot_spectrum:
        plt.subplot(311)
        plt.subplot(312)
        plt.subplot(313)
    else:
        plt.subplot(211)
        plt.subplot(212)
    plt.subplots_adjust(bottom=0.2, hspace=0.25)
    plt.figtext(0.05, 0.05, '', va='bottom', ha='left')

    # Make button context manager that disables buttons during processing and re-enables it afterwards
    class DisableButtons(object):
        def __init__(self):
            """Start with empty button list."""
            self.buttons = []
        def append(self, button):
            """Add button to list."""
            self.buttons.append(button)
        def __enter__(self):
            """Disable buttons on entry."""
            if plt.fignum_exists(1):
                for button in self.buttons:
                    button.eventson = False
                    button.hovercolor = '0.85'
                    button.label.set_color('gray')
                plt.draw()
        def __exit__(self, exc_type, exc_value, traceback):
            """Re-enable buttons on exit."""
            if plt.fignum_exists(1):
                for button in self.buttons:
                    button.eventson = True
                    button.hovercolor = '0.95'
                    button.label.set_color('k')
                plt.draw()
    all_buttons = DisableButtons()

    # Create buttons and their callbacks
    spectrogram_button = widgets.Button(plt.axes([0.37, 0.05, 0.1, 0.075]), 'Spectrogram')
    def spectrogram_callback(event):
        with all_buttons:
            plt.figure(2)
            plt.clf()
            compscans_in_previous_datasets = np.sum([len(bd) for bd in beam_data[:dataset_index]], dtype=np.int)
            unaveraged_compscan = unaveraged_dataset.compscans[compscan_index - compscans_in_previous_datasets]
            ax = scape.plot_xyz(unaveraged_compscan, 'time', 'freq', 'amp', power_in_dB=True)
            ax.set_title(unaveraged_compscan.target.name, size='medium')
    spectrogram_button.on_clicked(spectrogram_callback)
    all_buttons.append(spectrogram_button)

    keep_button = widgets.Button(plt.axes([0.48, 0.05, 0.1, 0.075]), 'Keep')
    def keep_callback(event):
        with all_buttons:
            next_load_reduce_plot(fig)
    keep_button.on_clicked(keep_callback)
    all_buttons.append(keep_button)

    discard_button = widgets.Button(plt.axes([0.59, 0.05, 0.1, 0.075]), 'Discard')
    def discard_callback(event):
        with all_buttons:
            if len(output_data) > 0:
                output_data[-1] = None
            next_load_reduce_plot(fig)
    discard_button.on_clicked(discard_callback)
    all_buttons.append(discard_button)

    back_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Back')
    def back_callback(event):
        with all_buttons:
            global dataset_index, compscan_index
            # Ignore back button unless there are two compscans in the pipeline
            if compscan_index > 0:
                compscan_index -= 2
                # Go back to previous data sets until the previous good compound scan is found
                while compscan_index < np.sum([len(bd) for bd in beam_data[:dataset_index]], dtype=np.int):
                    dataset_index -= 1
                output_data.pop()
                output_data.pop()
                next_load_reduce_plot(fig)
    back_button.on_clicked(back_callback)
    all_buttons.append(back_button)

    done_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
    def done_callback(event):
        with all_buttons:
            global dataset_index, compscan_index
            compscan_index = np.inf
            dataset_index = len(datasets)
            next_load_reduce_plot(fig)
    done_button.on_clicked(done_callback)
    all_buttons.append(done_button)

    # Start off the processing
    next_load_reduce_plot(fig)
    # Display plots - this should be called ONLY ONCE, at the VERY END of the script
    # The script stops here until you close the plots...
    plt.show()
