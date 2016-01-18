#!/usr/bin/python
# Script that uses scape to reduce data consisting of scans across a point source.
#

from matplotlib.backends.backend_pdf import PdfPages
import logging, optparse, os, pickle
import numpy as np
import katdal
import scape
import katpoint
from analyse_point_source_scans import reduce_and_plot
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# reduced version that deals with current bug in scape
def analyse_point_source_scans(filename, opts):
    print ('Loading HDF5 file %s into scape and reducing the data'%filename)
    h5file = katdal.open(filename)
    if len(h5file.catalogue.targets) > 1:
        print ('Removing first dummy scan caused by premature noise diode fire')
        h5file.sensor.get('Observation/target_index').remove(1)
        s=h5file.sensor.get('Observation/target')
        s.remove(s.unique_values[0])
        h5file.catalogue.remove(h5file.catalogue.targets[0].name)

    # Default output file names are based on input file name
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    if opts.outfilebase is None:
        opts.outfilebase = dataset_name + '_point_source_scans'

    # Set up logging: logging everything (DEBUG & above), both to console and file
    logger = logging.root
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler(opts.outfilebase + '.log', 'w')
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    # logger.addHandler(fh)

    kwargs={}

    #Force centre freqency if ku-band option is set
    if opts.ku_band:
        kwargs['centre_freq'] = 12.5005e9

    # Produce canonical version of baseline string (remove duplicate antennas)
    baseline_ants = opts.baseline.split(',')
    if len(baseline_ants) == 2 and baseline_ants[0] == baseline_ants[1]:
        opts.baseline = baseline_ants[0]

    # Load data set
    # logger.info("Loading dataset '%s'" % (filename,))
    dataset = scape.DataSet(h5file, baseline=opts.baseline, nd_models=opts.nd_models,
                            time_offset=opts.time_offset, **kwargs)

    # Select frequency channels and setup defaults if not specified
    num_channels = len(dataset.channel_select)
    if opts.freq_chans is None:
        # Default is drop first and last 25% of the bandpass
        start_chan = num_channels // 4
        end_chan   = start_chan * 3
    else:
        start_chan = int(opts.freq_chans.split(',')[0])
        end_chan = int(opts.freq_chans.split(',')[1])
    chan_select = range(start_chan,end_chan+1)

    # Check if a channel mask is specified and apply
    if opts.channel_mask:
        mask_file = open(opts.channel_mask)
        chan_select = ~(pickle.load(mask_file))
        mask_file.close()
        if len(chan_select) != num_channels:
            raise ValueError('Number of channels in provided mask does not match number of channels in data')
        chan_select[:start_chan] = False
        chan_select[end_chan:] = False
    dataset = dataset.select(freqkeep=chan_select)

    # Check scan count
    if len(dataset.compscans) == 0 or len(dataset.scans) == 0:
        raise RuntimeError('No scans found in file, skipping data set')
    scan_dataset = dataset.select(labelkeep='scan', copy=False)
    if len(scan_dataset.compscans) == 0 or len(scan_dataset.scans) == 0:
        raise RuntimeError('No scans left after standard reduction, skipping data set (no scans labelled "scan", perhaps?)')
    # Override pointing model if it is specified (useful if it is not in data file, like on early KAT-7)
    if opts.pointing_model:
        pm = file(opts.pointing_model).readline().strip()
        # logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(pm.split(',')), opts.pointing_model))
        dataset.antenna.pointing_model = katpoint.PointingModel(pm, strict=False)

    # Remove any noise diode models if the ku band option is set and flag for spikes
    if opts.ku_band:
        dataset.nd_h_model=None
        dataset.nd_v_model=None
        for i in range(len(dataset.scans)):
            dataset.scans[i].data = scape.stats.remove_spikes(dataset.scans[i].data,axis=1,spike_width=3,outlier_sigma=5.)

    # Initialise the output data cache (None indicates the compscan has not been processed yet)
    reduced_data = [{} for n in range(len(scan_dataset.compscans))]

    # Go one past the end of compscan list to write the output data out to CSV file
    for current_compscan in range(len(scan_dataset.compscans) + 1):
        # make things play nice
        opts.batch = True
        try:
            the_compscan   = scan_dataset.compscans[current_compscan]
        except: the_compscan = None
        fig = plt.figure(1,figsize = (8,8))
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
        # Start off the processing on the first compound scan
        print opts
        fig.current_compscan = 0
        reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)

    # Generate output report
    with PdfPages(opts.outfilebase+'.pdf') as pdf:
        out=reduced_data[0]
        offset_az, offset_el = "%.1f" % (60. * out['delta_azimuth'],), "%.1f" % (60. * out['delta_elevation'],)
        beam_width, beam_height = "%.1f" % (60. * out['beam_width_I'],), "%.2f" % (out['beam_height_I'],)
        baseline_height = "%.1f" % (out['baseline_height_I'],)
        pagetext  = "\nCheck Point Source Scan"
        pagetext += "\n\nDescription: %s\nName: %s\nExperiment ID: %s" %(h5file.description, h5file.name, h5file.experiment_id)
        pagetext  = pagetext + "\n"
        pagetext += "\n\nTest Setup:"
        pagetext += "\nRaster Scan across bright source"
        pagetext += "\n\nAntenna %(antenna)s" % out
        pagetext += "\n------------"
        pagetext += ("\nTarget = '%(target)s', azel=(%(azimuth).1f, %(elevation).1f) deg, " % out) +\
                    (u"offset=(%s, %s) arcmin" % (offset_az, offset_el))
        pagetext += (u"\nBeam height = %s %s") % (beam_height, out['data_unit'])
        pagetext += (u"\nBeamwidth = %s' (expected %.1f')") % (beam_width, 60. * out['beam_expected_width_I'])
        pagetext += (u"\nHH gain = %.3f Jy/%s") % (out['flux'] / out['beam_height_HH'], out['data_unit'])
        pagetext += (u"\nVV gain = %.3f Jy/%s") % (out['flux'] / out['beam_height_VV'], out['data_unit'])
        pagetext += (u"\nBaseline height = %s %s") % (baseline_height, out['data_unit'])
        plt.figure(None,figsize = (16,8))
        plt.axes(frame_on=False)
        plt.xticks([])
        plt.yticks([])
        plt.title("RTS Report %s"%opts.outfilebase ,fontsize=14, fontweight="bold")
        plt.text(0,0,pagetext,fontsize=12)
        pdf.savefig()
        plt.close()
        pdf.savefig(fig)

        d = pdf.infodict()
        import datetime
        d['Title'] = h5file.description
        d['Author'] = 'Ruby van Rooyen'
        d['Subject'] = 'RTS check point source scan'
        d['CreationDate'] = datetime.datetime(2015, 8, 13)
        d['ModDate'] = datetime.datetime.today()

## -- Main --
if __name__ == '__main__':

    # Parse command-line opts and arguments
    parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                                   description="Reduced from of analyse_point_source_scan for quick evaluation of single dish performance.")
    parser.add_option("-a", "--baseline", default='sd',
                      help="Baseline to load (e.g. 'ant1' for antenna 1 or 'ant1,ant2' for 1-2 baseline), "
                           "default is first single-dish baseline in file")
    parser.add_option("-c", "--channel-mask", default=None,
                      help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")
    parser.add_option("-f", "--freq-chans",
                      help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass)")
    parser.add_option("-m", "--monte-carlo", dest="mc_iterations", type='int', default=1,
                      help="Number of Monte Carlo iterations to estimate uncertainty (20-30 suggested, default off)")
    parser.add_option("-n", "--nd-models",
                      help="Name of optional directory containing noise diode model files")
    parser.add_option("-o", "--output", dest="outfilebase",
                      help="Base name of output files (*.csv for output data and *.log for messages, "
                           "default is '<dataset_name>_point_source_scans')")
    parser.add_option("-p", "--pointing-model",
                      help="Name of optional file containing pointing model parameters in degrees")
    parser.add_option("-s", "--plot-spectrum", action="store_true",
                      help="Flag to include spectral plot")
    parser.add_option("-t", "--time-offset", type='float', default=0.0,
                      help="Time offset to add to DBE timestamps, in seconds (default = %default)")
    parser.add_option("-u", "--ku-band", action="store_true",
                      help="Force center frequency to be 12500.5 MHz")
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False,
                      help='Display raster scan beam fit')
    parser.add_option('--keep-all', action='store_true', dest='keep_all', default=False,
                      help='Keep scans with or without a valid beam in batch mode')
    (opts, args) = parser.parse_args()

    if len(args) != 1 or not args[0].endswith('.h5'):
        parser.print_usage()
        raise RuntimeError('Please specify a single HDF5 file as argument to the script')

    analyse_point_source_scans(args[0], opts)

    # Display plots - this should be called ONLY ONCE, at the VERY END of the script
    # The script stops here until you close the plots...
    if opts.verbose: plt.show()

    # cleanup before exit
    try: plt.close('all')
    except: pass # nothing to close

# -fin-
