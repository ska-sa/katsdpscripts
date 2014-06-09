#!/usr/bin/python
# Script that uses scape to reduce data consisting of scans across multiple point sources.
#

#################################################### Main function ####################################################
import optparse
import os
import logging
import katpoint
import numpy as np
import scape

from katsdpdata.reduction import compscan_key
from katsdpdata.reduction import reduce_and_plot

# These packages are only imported once the script options are checked
plt = widgets = None

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and extracts fitted beam parameters "
                                           "from the compound scans in it. It runs interactively by default, "
                                           "which allows the user to inspect results and discard bad scans.")
parser.add_option("-a", "--baseline", default='sd',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-b", "--batch", action="store_true",
                  help="Flag to do processing in batch mode without user interaction")
parser.add_option("-f", "--freq-chans",
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass)")
parser.add_option("-k", "--keep", dest="keepfilename",
                  help="Name of optional CSV file used to select compound scans from dataset (implies batch mode)")
parser.add_option("-m", "--monte-carlo", dest="mc_iterations", type='int', default=1,
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
parser.add_option("--old-loader", action="store_true", help="Use old SCAPE loader to open HDF5 file instead of katfile")
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
    csv_baseline = 'A%sA%s' % (ant_name[3:], ant_name[3:])
    if opts.baseline != 'sd' and opts.baseline != csv_baseline:
        logger.warn("Requested baseline '%s' does not match baseline '%s' in CSV file '%s'" %
                    (opts.baseline, csv_baseline, opts.keepfilename))
    logger.warn("Using baseline '%s' found in CSV file '%s'" % (csv_baseline, opts.keepfilename))
    opts.baseline = csv_baseline

# Only import matplotlib if not in batch mode
if not opts.batch:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets

# Avoid loading the data set if it does not appear in specified CSV file
if keep_datasets and dataset_name not in keep_datasets:
    raise RuntimeError("Skipping dataset '%s' (based on CSV file)" % (filename,))

# Load data set
logger.info("Loading dataset '%s'" % (filename,))
dataset = scape.DataSet(filename, baseline=opts.baseline, nd_models=opts.nd_models,
                        time_offset=opts.time_offset, katfile=not opts.old_loader)

# Select frequency channels and setup defaults if not specified
num_channels = len(dataset.channel_select)
if opts.freq_chans is None:
    # Default is drop first and last 25% of the bandpass
    start_chan = num_channels // 4
    end_chan   = start_chan * 3
else:
    start_chan = int(opts.freq_chans.split(',')[0])
    end_chan = int(opts.freq_chans.split(',')[1])
chan_range = range(start_chan,end_chan+1)
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

# Initialise the output data cache (None indicates the compscan has not been processed yet)
reduced_data = [None] * len(scan_dataset.compscans)

### BATCH MODE ###

# This will cycle through all data sets and stop when done
if opts.batch:
    # Go one past the end of compscan list to write the output data out to CSV file
    for current_compscan in range(len(scan_dataset.compscans) + 1):
        # Look up compscan key in list of compscans to keep (if provided, only applicable to batch mode anyway)
        if keep_scans and (current_compscan < len(scan_dataset.compscans)):
            cs_key = ' '.join(compscan_key(scan_dataset.compscans[current_compscan]), logger=logger)
            if cs_key not in keep_scans:
                logger.info("==== Skipping compound scan '%s' (based on CSV file) ====" % (cs_key,))
                continue
        reduce_and_plot(dataset, current_compscan, reduced_data, opts, logger=logger)

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
    plt.figtext(0.05, 0.945, '', va='bottom', ha='left')

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
            out = reduced_data[fig.current_compscan]
            ax = scape.plot_xyz(out['unavg_dataset'], 'time', 'freq', 'amp', power_in_dB=True)
            ax.set_title(out['target'], size='medium')
    spectrogram_button.on_clicked(spectrogram_callback)
    all_buttons.append(spectrogram_button)

    keep_button = widgets.Button(plt.axes([0.48, 0.05, 0.1, 0.075]), 'Keep')
    def keep_callback(event):
        with all_buttons:
            reduced_data[fig.current_compscan]['keep'] = True
            fig.current_compscan += 1
            reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig)
    keep_button.on_clicked(keep_callback)
    all_buttons.append(keep_button)

    discard_button = widgets.Button(plt.axes([0.59, 0.05, 0.1, 0.075]), 'Discard')
    def discard_callback(event):
        with all_buttons:
            reduced_data[fig.current_compscan]['keep'] = False
            fig.current_compscan += 1
            reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig)
    discard_button.on_clicked(discard_callback)
    all_buttons.append(discard_button)

    back_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Back')
    def back_callback(event):
        with all_buttons:
            if fig.current_compscan > 0:
                fig.current_compscan -= 1
                reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig)
    back_button.on_clicked(back_callback)
    all_buttons.append(back_button)

    done_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
    def done_callback(event):
        with all_buttons:
            fig.current_compscan = len(reduced_data)
            reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig)
    done_button.on_clicked(done_callback)
    all_buttons.append(done_button)

    # Start off the processing on the first compound scan
    fig.current_compscan = 0
    reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig)
    # Display plots - this should be called ONLY ONCE, at the VERY END of the script
    # The script stops here until you close the plots...
    plt.show()
