#!/usr/bin/python
# Script that uses scape to reduce data consisting of scans across multiple point sources.
#
# This can be used to determine gain curves, tipping curves and pointing models.
# The user can interactively observe reduction results and discard bad data. The
# end product is a file containing pointing, fitted beam parameters, baseline
# height and weather measurements, etc.
#
# The latest version incorporates uncertainties and supports HDF5 only (poor XDM!).
#
# Ludwig Schwardt
# 3 June 2011
#

from __future__ import with_statement

import os.path
import logging
import optparse

import numpy as np

import scape
import katpoint

# These packages are only imported once the script options are checked
plt = widgets = None

################################################### Helper routines ###################################################

def interp_sensor(compscan, quantity, default):
    """Interpolate environmental sensor data."""
    try:
        sensor = compscan.dataset.enviro[quantity]
    except KeyError:
        return (lambda times: default)
    else:
        interp = scape.fitting.PiecewisePolynomial1DFit(max_degree=0)
        interp.fit(sensor['timestamp'], sensor['value'])
        return interp

def compscan_key(compscan):
    """List of strings that identifies compound scan."""
    # Name of data set that contains compound scan
    path = compscan.scans[0].path
    filename_end = path.find('.h5')
    dataset_name = os.path.basename(path[:filename_end]) if filename_end > 0 else os.path.basename(path)
    # Time when compound scan is exactly half-way through its operation (i.e. 50% complete)
    middle_time = np.median(np.hstack([scan.timestamps for scan in compscan.scans]), axis=None)
    return compscan.dataset.antenna.name, dataset_name, compscan.target.name, str(katpoint.Timestamp(middle_time))

def reduce_compscan(compscan, cal_dataset, beam_pols=['HH', 'VV', 'I'], **kwargs):
    """Do complete point source reduction on a compound scan (gain cal + beam fit)."""
    # Calculate average target flux over entire band
    flux_spectrum = compscan.target.flux_density(compscan.dataset.freqs)
    average_flux = np.mean([flux for flux in flux_spectrum if not np.isnan(flux)])

    # Estimate gain on noise diode firings, then apply this gain to individual compound scan
    cal_dataset.convert_power_to_temperature()
    compscan.dataset.nd_gain = cal_dataset.nd_gain
    compscan.dataset.convert_power_to_temperature()
    compscan.dataset.average()

    # Fit the requested beams and extract beam/baseline parameters
    beams = []
    for pol in beam_pols:
        compscan.fit_beam_and_baselines(pol)
        bh = compscan.baseline_height()
        if bh is None:
            bh = np.nan
        beam_params = [compscan.beam.height, katpoint.rad2deg(np.mean(compscan.beam.width)), bh,
                       float(compscan.beam.refined)] if compscan.beam else [np.nan, np.nan, bh, 0.]
        beams.append((pol, beam_params))

    # Obtain environmental data averaged across the compound scan
    compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
    temperature = np.mean(interp_sensor(compscan, 'temperature', 35.0)(compscan_times))
    pressure = np.mean(interp_sensor(compscan, 'pressure', 950.0)(compscan_times))
    humidity = np.mean(interp_sensor(compscan, 'humidity', 15.0)(compscan_times))
    wind_speed = np.mean(interp_sensor(compscan, 'wind_speed', 0.0)(compscan_times))

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
        expected_width = katpoint.rad2deg(np.mean(compscan.beam.expected_width))
        # Fitted beam center is in (x, y) coordinates, in projection centred on target
        beam_center_xy = compscan.beam.center
        # Convert this offset back to spherical (az, el) coordinates
        beam_center_azel = compscan.target.plane_to_sphere(beam_center_xy[0], beam_center_xy[1], middle_time)
        # Now correct the measured (az, el) for refraction and then apply the old pointing model
        # to get a "raw" measured (az, el) at the output of the pointing model
        beam_center_azel = [beam_center_azel[0], rc.apply(beam_center_azel[1], temperature, pressure, humidity)]
        beam_center_azel = compscan.dataset.antenna.pointing_model.apply(*beam_center_azel)
        beam_center_azel = katpoint.rad2deg(np.array(beam_center_azel))
        # Make sure the offset is a small angle around 0 degrees
        offset_azel = scape.stats.angle_wrap(beam_center_azel - requested_azel, 360.)
    else:
        expected_width = np.nan
        offset_azel = np.array([np.nan, np.nan])
    # Outputs that are not expected to change if visibility data is perturbed
    fixed_names = 'antenna dataset target timestamp_ut data_unit frequency flux ' \
                  'temperature pressure humidity wind_speed azimuth elevation beam_expected_width_I'
    fixed = list(compscan_key(compscan)) + [compscan.dataset.data_unit, compscan.dataset.freqs[0]] + \
            [average_flux, temperature, pressure, humidity, wind_speed] + requested_azel.tolist() + [expected_width]
    # Outputs that are expected to change if visibility data is perturbed
    var_names = 'delta_azimuth delta_elevation'
    variable = offset_azel.tolist()
    for beam in beams:
        var_names += ' beam_height_%s beam_width_%s baseline_height_%s refined_%s' % tuple([beam[0]] * 4)
        variable += list(beam[1])
    return np.rec.fromrecords([tuple(fixed)], names=fixed_names.split()), \
           np.rec.fromrecords([tuple(variable)], names=var_names.split())

def join_recarrays(*args):
    """Join the records of multiple independent recarrays of the same shape."""
    names, data = [], []
    for arg in args:
        names += [name for name in arg.dtype.names]
        data += [arg[name] for name in arg.dtype.names]
    return np.rec.fromarrays(data, names=names)

def reduce_compscan_with_uncertainty(dataset, compscan_index=0, mc_iterations=1, **kwargs):
    """Do complete point source reduction on a compound scan, with uncertainty."""
    scan_dataset = dataset.select(labelkeep='scan', copy=False)
    compscan = scan_dataset.compscans[compscan_index]
    logger.info("==== Processing compound scan '%s' ====" % (' '.join(compscan_key(compscan)),))
    # Build data set containing a single compound scan at a time (make copy, as reduction modifies it)
    scan_dataset.compscans = [compscan]
    compscan_dataset = scan_dataset.select(copy=True)
    # Extract noise diode firings only (make copy, as this will be modified by gain cal)
    cal_dataset = dataset.select(labelkeep='cal', copy=True)
    # Do first reduction run
    main_compscan = compscan_dataset.compscans[0]
    fixed, variable = reduce_compscan(main_compscan, cal_dataset)
    # Produce data set that has counts converted to Kelvin, but no averaging (for spectral plots)
    unavg_compscan_dataset = scan_dataset.select(copy=True)
    unavg_compscan_dataset.nd_gain = cal_dataset.nd_gain
    unavg_compscan_dataset.convert_power_to_temperature()
    # Add data from Monte Carlo perturbations
    iter_outputs = [variable]
    for m in range(mc_iterations - 1):
        logger.info("---- Monte Carlo iteration %d of %d ----" % (m + 2, mc_iterations))
        compscan_dataset = scan_dataset.select(copy=True).perturb()
        cal_dataset = dataset.select(labelkeep='cal', copy=True).perturb()
        fixed, variable = reduce_compscan(compscan_dataset.compscans[0], cal_dataset)
        iter_outputs.append(variable)
    # Get mean and uncertainty of variable part of output data
    var_output = np.concatenate(iter_outputs).view(np.float).reshape(mc_iterations, -1)
    var_mean = np.rec.fromrecords([tuple(var_output.mean(axis=0))], names=iter_outputs[0].dtype.names)
    var_std = np.rec.fromrecords([tuple(var_output.std(axis=0))],
                                 names=[name + '_std' for name in iter_outputs[0].dtype.names])
    out_record = np.squeeze(join_recarrays(fixed, var_mean, var_std))
    return main_compscan, out_record, unavg_compscan_dataset

def reduce_and_plot(dataset, compscan_index, output_data, opts, fig=None):
    """Reduce compound scan, update the plots in given figure and save output data when done."""
    # Save output data and return after last compound scan is done
    if compscan_index >= len(output_data):
        output_recs = [np.atleast_1d(p) for p in output_data if p]
        output_recs = np.concatenate(output_recs) if len(output_recs) > 0 else []
        output_fields = '%(dataset)s, %(target)s, %(timestamp_ut)s, %(azimuth).7f, %(elevation).7f, ' \
                        '%(delta_azimuth).7f, %(delta_azimuth_std).7f, %(delta_elevation).7f, %(delta_elevation_std).7f, ' \
                        '%(data_unit)s, %(beam_height_I).7f, %(beam_height_I_std).7f, %(beam_width_I).7f, ' \
                        '%(beam_width_I_std).7f, %(baseline_height_I).7f, %(baseline_height_I_std).7f, %(refined_I).7f, ' \
                        '%(beam_height_HH).7f, %(beam_width_HH).7f, %(baseline_height_HH).7f, %(refined_HH).7f, ' \
                        '%(beam_height_VV).7f, %(beam_width_VV).7f, %(baseline_height_VV).7f, %(refined_VV).7f, ' \
                        '%(frequency).7f, %(flux).4f, %(temperature).2f, %(pressure).2f, %(humidity).2f, %(wind_speed).2f\n'
        output_field_names = [name.partition(')')[0] for name in output_fields[2:].split(', %(')]
        f = file(opts.outfilebase + '.csv', 'w')
        f.write('# antenna = %s\n' % dataset.antenna.description)
        f.write(', '.join(output_field_names) + '\n')
        f.writelines([output_fields % rec for rec in output_recs])
        f.close()
        if not opts.batch:
            # This closes the GUI and effectively exits the program in the interactive case
            plt.close('all')
        return

    # Reduce compound scan
    compscan, rec, unavg_dataset = reduce_compscan_with_uncertainty(dataset, compscan_index, opts.monte_carlo)
    # If beam is marked as invalid, discard scan only if in batch mode (otherwise discard button has to do it)
    output_data[compscan_index] = rec if (compscan.beam and (not opts.batch or compscan.beam.is_valid)) else None

    # Display compound scan
    if not opts.batch:
        ax1, ax2, info = fig.axes[0], fig.axes[1], fig.texts[0]
        ax1.clear()
        scape.plot_compound_scan_in_time(compscan, ax=ax1)
        ax1.set_title(("%(dataset)s %(antenna)s '%(target)s'\nazel=(%(azimuth).1f, %(elevation).1f) deg," % rec) +
                      (" offset=(%.1f, %.1f) arcmin" % (60. * rec['delta_azimuth'], 60. * rec['delta_elevation'])),
                      size='medium')
        ax1.set_ylabel('Total power (%(data_unit)s)' % rec)
        ax2.clear()
        scape.plot_compound_scan_on_target(compscan, ax=ax2)
        if opts.plot_spectrum:
            ax3 = fig.axes[2]
            ax3.clear()
            scape.plot_xyz(unavg_dataset, 'freq', 'amp', labels=[], power_in_dB=True, ax=ax3)
        if compscan.beam:
            info.set_text(("Beamwidth = %.1f' (expected %.1f')\nBeam height = %.1f %s\n" +
                           "HH/VV gain = %.3f/%.3f Jy/%s\nBaseline height = %.1f %s") %
                          (60. * rec['beam_width_I'], 60. * rec['beam_expected_width_I'], rec['beam_height_I'],
                           rec['data_unit'], rec['flux'] / rec['beam_height_HH'], rec['flux'] / rec['beam_height_VV'],
                           rec['data_unit'], rec['baseline_height_I'], rec['data_unit']))
        else:
            info.set_text("No beam\nBaseline height = %(baseline_height_I).2f %(data_unit)s" % rec)
        plt.draw()
        # Also store data in figure so that button callbacks can access it
        fig.user_data = (dataset, compscan_index, output_data, opts, compscan, rec, unavg_dataset)

#################################################### Main function ####################################################

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and extracts fitted beam parameters "
                                           "from the compound scans in it. It runs interactively by default, "
                                           "which allows the user to inspect results and discard bad scans.")
parser.add_option("-a", "--baseline", default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-b", "--batch", action="store_true",
                  help="Flag to do processing in batch mode without user interaction")
parser.add_option("-f", "--freq-chans", default='90,424',
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default %default)")
parser.add_option("-k", "--keep", dest="keepfilename",
                  help="Name of optional CSV file used to select compound scans from dataset (implies batch mode)")
parser.add_option("-m", "--monte-carlo", type='int', default=1,
                  help="Number of Monte Carlo iterations to estimate uncertainty (20-30 suggested, default off)")
parser.add_option("-n", "--nd-models", help="Name of optional directory containing noise diode model files")
parser.add_option("-o", "--output", dest="outfilebase",
                  help="Base name of output files (*.csv for output data and *.log for messages, "
                       "default is '<dataset_name>_point_source_scans')")
parser.add_option("-p", "--pointing-model",
                  help="Name of optional file containing pointing model parameters in degrees")
parser.add_option("-s", "--plot-spectrum", action="store_true", help="Flag to include spectral plot")
parser.add_option("-t", "--time-offset", type='float', default=0.0,
                  help="Time offset to add to DBE timestamps, in seconds (default = %default)")
(opts, args) = parser.parse_args()

if len(args) != 1 or not args[0].endswith('.h5'):
    raise RuntimeError('Please specify a single HDF5 file as argument to the script')
filename = args[0]
dataset_name = os.path.splitext(os.path.basename(filename))[0]
    
# Default output file names are based on input file name
if opts.outfilebase is None:
    opts.outfilebase = dataset_name + '_point_source_scans'

# Set up logging: logging everything (DEBUG & above), both to console and file
logger = logging.root
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(opts.outfilebase + '.log', 'w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(fh)

# Load old CSV file used to select compound scans from dataset
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
    logger.debug("Loaded CSV file '%s' containing %d dataset(s) and %d compscan(s) for antenna '%s'" %
                 (opts.keepfilename, len(keep_datasets), len(keep_scans), ant_name))
    # Ensure we are using antenna found in CSV file (assume ant name = "ant" + number)
    opts.baseline = 'A%sA%s' % (ant_name[3:], ant_name[3:])

# Only import matplotlib if not in batch mode
if not opts.batch:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets

# Avoid loading the data set if it does not appear in specified CSV file
if keep_datasets and dataset_name not in keep_datasets:
    raise RuntimeError("Skipping dataset '%s' (based on CSV file)" % (filename,))

# Load data set
logger.info("Loading dataset '%s'" % (filename,))
dataset = scape.DataSet(filename, baseline=opts.baseline, nd_models=opts.nd_models, time_offset=opts.time_offset)
# Select frequency channels
start_freq_channel = int(opts.freq_chans.split(',')[0])
end_freq_channel = int(opts.freq_chans.split(',')[1])
chan_range = range(start_freq_channel, end_freq_channel + 1)
dataset = dataset.select(freqkeep=chan_range)

# Check scan count
if len(dataset.compscans) == 0 or len(dataset.scans) == 0:
    raise RuntimeError('No scans found in file, skipping data set')
scan_dataset = dataset.select(labelkeep='scan', copy=False)
if len(scan_dataset.compscans) == 0 or len(scan_dataset.scans) == 0:
    raise RuntimeError('No scans left after standard reduction, skipping data set (no scans labelled "scan", perhaps?)')
# Override pointing model if it is specified (useful if it is not in data file, like on early KAT-7)
if opts.pointing_model:
    pm = file(opts.pointing_model).readline().strip()
    logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(pm.split(',')), opts.pointing_model))
    dataset.antenna.pointing_model = katpoint.PointingModel(pm, strict=False)

# Presized list of output records (None indicates a discarded compound scan)
output_data = [None] * len(scan_dataset.compscans)

### BATCH MODE ###

# This will cycle through all data sets and stop when done
if opts.batch:
    # Go one past the end of compscan list to write the output data out to CSV file
    for compscan_index in range(len(scan_dataset.compscans) + 1):
        # Look up compscan key in list of compscans to keep (if provided, only applicable to batch mode anyway)
        if keep_scans and (compscan_index < len(scan_dataset.compscans)):
            cs_key = ' '.join(compscan_key(scan_dataset.compscans[compscan_index]))
            if cs_key not in keep_scans:
                logger.info("==== Skipping compound scan '%s' (based on CSV file) ====" % (cs_key,))
                continue
        reduce_and_plot(dataset, compscan_index, output_data, opts)

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
            dataset, compscan_index, output_data, opts, compscan, rec, unavg_dataset = fig.user_data
            ax = scape.plot_xyz(unavg_dataset, 'time', 'freq', 'amp', power_in_dB=True)
            ax.set_title(rec['target'], size='medium')
    spectrogram_button.on_clicked(spectrogram_callback)
    all_buttons.append(spectrogram_button)

    keep_button = widgets.Button(plt.axes([0.48, 0.05, 0.1, 0.075]), 'Keep')
    def keep_callback(event):
        with all_buttons:
            dataset, compscan_index, output_data, opts, compscan, rec, unavg_dataset = fig.user_data
            compscan_index += 1
            reduce_and_plot(dataset, compscan_index, output_data, opts, fig)
    keep_button.on_clicked(keep_callback)
    all_buttons.append(keep_button)

    discard_button = widgets.Button(plt.axes([0.59, 0.05, 0.1, 0.075]), 'Discard')
    def discard_callback(event):
        with all_buttons:
            dataset, compscan_index, output_data, opts, compscan, rec, unavg_dataset = fig.user_data
            output_data[compscan_index] = None
            compscan_index += 1
            reduce_and_plot(dataset, compscan_index, output_data, opts, fig)
    discard_button.on_clicked(discard_callback)
    all_buttons.append(discard_button)

    back_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Back')
    def back_callback(event):
        with all_buttons:
            dataset, compscan_index, output_data, opts, compscan, rec, unavg_dataset = fig.user_data
            if compscan_index > 0:
                compscan_index -= 1
                reduce_and_plot(dataset, compscan_index, output_data, opts, fig)
    back_button.on_clicked(back_callback)
    all_buttons.append(back_button)

    done_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
    def done_callback(event):
        with all_buttons:
            dataset, compscan_index, output_data, opts, compscan, rec, unavg_dataset = fig.user_data
            compscan_index = len(output_data)
            reduce_and_plot(dataset, compscan_index, output_data, opts, fig)
    done_button.on_clicked(done_callback)
    all_buttons.append(done_button)

    # Start off the processing on the first compound scan
    reduce_and_plot(dataset, 0, output_data, opts, fig)
    # Display plots - this should be called ONLY ONCE, at the VERY END of the script
    # The script stops here until you close the plots...
    plt.show()
