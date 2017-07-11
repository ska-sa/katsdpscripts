#!/usr/bin/python
##Quick Point Source Scan analysis
# Script that uses scape to reduce data consisting of scans across a point source.
#
# Revision History:
# * (Ruby) Initial version based on old KAT-7 analysis
# * (Ruby) Get things working with differences between AR1 and RTS
# * (Ruby/Sean) Add simple beam fit using pointing model calculations for single compscan

from matplotlib.backends.backend_pdf import PdfPages
import logging, optparse, os, pickle
import numpy as np
import katdal
import scape
import katpoint
from katsdpscripts.reduction.analyse_point_source_scans import reduce_and_plot
from katsdpscripts.reduction.analyse_point_source_scans import extract_cal_dataset,reduce_compscan
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def reduce_compscan_with_uncertainty(dataset, compscan_index=0, mc_iterations=1, batch=True, **kwargs):
    """Do complete point source reduction on a compound scan, with uncertainty."""
    print "Do complete point source reduction on a compound scan, with uncertainty."
    scan_dataset = dataset.select(labelkeep='scan', copy=False)
    compscan = scan_dataset.compscans[compscan_index]

    # Build data set containing a single compound scan at a time (make copy, as reduction modifies it)
    scan_dataset.compscans = [compscan]
    compscan_dataset = scan_dataset.select(flagkeep='~nd_on', copy=True)
    cal_dataset = extract_cal_dataset(dataset)
    # Do first reduction run
    main_compscan = compscan_dataset.compscans[0]
    fixed, variable = reduce_compscan(main_compscan, cal_dataset, **kwargs)
    # Produce data set that has counts converted to Kelvin, but no averaging (for spectral plots)
    unavg_compscan_dataset = scan_dataset.select(flagkeep='~nd_on', copy=True)
    unavg_compscan_dataset.nd_gain = cal_dataset.nd_gain
    unavg_compscan_dataset.convert_power_to_temperature()
    # Add data from Monte Carlo perturbations
    iter_outputs = [np.rec.fromrecords([tuple(variable.values())], names=variable.keys())]
    for m in range(mc_iterations - 1):
        compscan_dataset = scan_dataset.select(flagkeep='~nd_on', copy=True).perturb()
        cal_dataset = extract_cal_dataset(dataset).perturb()
        fixed, variable = reduce_compscan(compscan_dataset.compscans[0], cal_dataset, **kwargs)
        iter_outputs.append(np.rec.fromrecords([tuple(variable.values())], names=variable.keys()))
    # Get mean and uncertainty of variable part of output data (assumed to be floats)
    var_output = np.concatenate(iter_outputs).view(np.float).reshape(mc_iterations, -1)
    var_mean = dict(zip(variable.keys(), var_output.mean(axis=0)))
    var_std = dict(zip([name + '_std' for name in variable], var_output.std(axis=0)))
    # Keep scan only with a valid beam in batch mode (otherwise keep button has to do it explicitly)
    keep = batch and main_compscan.beam and main_compscan.beam.is_valid
    output_dict = {'keep' : True, 'compscan' : main_compscan, 'unavg_dataset' : unavg_compscan_dataset}
    output_dict.update(fixed)
    output_dict.update(var_mean)
    output_dict.update(var_std)
    return output_dict

class SuppressErrors(object):
    """Don't crash on exceptions but at least report them."""
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        """Enter the error suppression context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the error suppression context, reporting any errors."""
        if exc_value is not None:
            exc_msg = str(exc_value)
            msg = "Reduction interrupted by exception (%s%s)" % \
                   (exc_value.__class__.__name__,
                   (": '%s'" % (exc_msg,)) if exc_msg else '')
            self.logger.error(msg, exc_info=True)
        # Suppress those exceptions
        return True

def local_reduce_and_plot(dataset, current_compscan, reduced_data, opts, fig=None, **kwargs):
    """Reduce compound scan, update the plots in given figure and save reduction output when done."""
    # Save reduction output and return after last compound scan is done

    if current_compscan >= len(reduced_data):
        output_fields = '%(dataset)s, %(target)s, %(timestamp_ut)s, %(azimuth).7f, %(elevation).7f, ' \
                        '%(delta_azimuth).7f, %(delta_azimuth_std).7f, %(delta_elevation).7f, %(delta_elevation_std).7f, ' \
                        '%(data_unit)s, %(beam_height_I).7f, %(beam_height_I_std).7f, %(beam_width_I).7f, ' \
                        '%(beam_width_I_std).7f, %(baseline_height_I).7f, %(baseline_height_I_std).7f, %(refined_I).7f, ' \
                        '%(beam_height_HH).7f, %(beam_width_HH).7f, %(baseline_height_HH).7f, %(refined_HH).7f, ' \
                        '%(beam_height_VV).7f, %(beam_width_VV).7f, %(baseline_height_VV).7f, %(refined_VV).7f, ' \
                        '%(frequency).7f, %(flux).4f, %(temperature).2f, %(pressure).2f, %(humidity).2f, %(wind_speed).2f\n'
        output_field_names = [name.partition(')')[0] for name in output_fields[2:].split(', %(')]
        output_data = [output_fields % out for out in reduced_data if out and out['keep']]
        #return the recarray
        to_keep=[]
        for field in output_field_names: 
            to_keep.append([data[field] for data in reduced_data if data and data['keep']])
        output_data = np.rec.fromarrays(to_keep, dtype=zip(output_field_names,[np.array(tk).dtype for tk in to_keep]))
        return (dataset.antenna, output_data,)

    # Reduce current compound scan if results are not cached
    if not reduced_data[current_compscan]:
        with SuppressErrors(kwargs['logger']):
            reduced_data[current_compscan] = reduce_compscan_with_uncertainty(dataset, current_compscan,
                                                                              opts.mc_iterations, opts.batch, **kwargs)
    if True:
        out = reduced_data[current_compscan]

    # Reduce next compound scan so long, as this will improve interactiveness (i.e. next plot will be immediate)
    if (current_compscan < len(reduced_data) - 1) and not reduced_data[current_compscan + 1]:
        with SuppressErrors(kwargs['logger']):
            reduced_data[current_compscan + 1] = reduce_compscan_with_uncertainty(dataset, current_compscan + 1,
                                                                                  opts.mc_iterations, opts.batch, **kwargs)

# reduced version that deals with current bug in scape
def analyse_point_source_scans(filename, h5file, opts):
    # Default output file names are based on input file name
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    if opts.outfilebase is None:
        opts.outfilebase = dataset_name + '_point_source_scans'

    kwargs={}

    #Force centre freqency if ku-band option is set
    if opts.ku_band:
        kwargs['centre_freq'] = 12.5005e9

    # Produce canonical version of baseline string (remove duplicate antennas)
    baseline_ants = opts.baseline.split(',')
    if len(baseline_ants) == 2 and baseline_ants[0] == baseline_ants[1]:
        opts.baseline = baseline_ants[0]

    # Load data set
    if opts.baseline not in [ant.name for ant in h5file.ants]:
        raise RuntimeError('Cannot find antenna %s in dataset'%opts.baseline)
    # dataset = scape.DataSet(h5file, baseline=opts.baseline, nd_models=opts.nd_models,
    #                         time_offset=opts.time_offset, **kwargs)
    dataset = scape.DataSet(filename, baseline=opts.baseline, nd_models=opts.nd_models,
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
        if opts.pointing_model.split('/')[-2] =='mkat':
            if opts.ku_band: band='ku'
            else: band='l'
            pt_file = os.path.join(opts.pointing_model, '%s.%s.pm.csv' % (opts.baseline,band))
        else:
            pt_file = os.path.join(opts.pointing_model, '%s.pm.csv' % (opts.baseline))
        if not os.path.isfile(pt_file):
            raise RuntimeError('Cannot find file %s' %(pt_file))
        pm = file(pt_file).readline().strip()
        dataset.antenna.pointing_model = katpoint.PointingModel(pm)

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
        logger = logging.root
        fig.current_compscan = 0
        reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)

    # Initialise the output data cache (None indicates the compscan has not been processed yet)
    reduced_data = [{} for n in range(len(scan_dataset.compscans))]
    # Go one past the end of compscan list to write the output data out to CSV file
    for current_compscan in range(len(scan_dataset.compscans) + 1):
        # make things play nice
        opts.batch = True
        try:
            the_compscan   = scan_dataset.compscans[current_compscan]
        except: the_compscan = None
        logger = logging.root
        output = local_reduce_and_plot(dataset, current_compscan, reduced_data, opts, logger=logger)
    offsetdata = output[1]
    from katpoint import  deg2rad
    def angle_wrap(angle, period=2.0 * np.pi):
        """wrap angle into the interval -*period* / 2 ... *period* / 2."""
        return (angle + 0.5 * period) % period - 0.5 * period
    az, el = angle_wrap(deg2rad(offsetdata['azimuth'])), deg2rad(offsetdata['elevation'])
    model_delta_az, model_delta_el = ant.pointing_model.offset(az, el)
    measured_delta_az = offsetdata['delta_azimuth'] - model_delta_az # pointing model correction
    measured_delta_el = offsetdata['delta_elevation'] - model_delta_el# pointing model correction
    """determine new residuals from current pointing model"""
    residual_az = measured_delta_az - model_delta_az
    residual_el = measured_delta_el - model_delta_el
    residual_xel  = residual_az * np.cos(el)
    # Initialise new pointing model and set default enabled parameters
    keep = np.ones((len(offsetdata)),dtype=np.bool)
    min_rms=np.sqrt(2) * 60. * 1e-12
    use_stats=True
    new_model = katpoint.PointingModel()
    num_params = len(new_model)
    default_enabled = np.array([1, 3, 4, 5, 6, 7]) - 1
    enabled_params = np.tile(False, num_params)
    enabled_params[default_enabled] = True
    enabled_params = enabled_params.tolist()
    # Fit new pointing model
    az, el = angle_wrap(deg2rad(offsetdata['azimuth'])), deg2rad(offsetdata['elevation'])
    measured_delta_az, measured_delta_el = deg2rad(offsetdata['delta_azimuth']), deg2rad(offsetdata['delta_elevation'])
    # Uncertainties are optional
    min_std = deg2rad(min_rms  / 60. / np.sqrt(2))
    std_delta_az = np.clip(deg2rad(offsetdata['delta_azimuth_std']), min_std, np.inf) \
    if 'delta_azimuth_std' in offsetdata.dtype.fields and use_stats else np.tile(min_std, len(az))
    std_delta_el = np.clip(deg2rad(offsetdata['delta_elevation_std']), min_std, np.inf) \
    if 'delta_elevation_std' in offsetdata.dtype.fields and use_stats else np.tile(min_std, len(el))

    params, sigma_params = new_model.fit(az[keep], el[keep], measured_delta_az[keep], measured_delta_el[keep],
                                         std_delta_az[keep], std_delta_el[keep], enabled_params)
    """Determine new residuals from new fit"""
    newmodel_delta_az, newmodel_delta_el = new_model.offset(az, el)
    residual_az = measured_delta_az - newmodel_delta_az
    residual_el = measured_delta_el - newmodel_delta_el
    residual_xel  = residual_az * np.cos(el)

    # Show actual scans
    h5file.select(scans='scan')
    fig1 = plt.figure(2, figsize=(8,8))
    plt.scatter(h5file.ra, h5file.dec, s=np.mean(np.abs(h5file.vis[:,2200:2400,1]), axis=1))
    plt.title('Raster scan over target')
    plt.ylabel('Dec [deg]')
    plt.xlabel('Ra [deg]')

    # Try to fit beam
    for c in h5file.compscans():
        if not dataset is None:
            dataset = dataset.select(flagkeep='~nd_on')
        dataset.average()
        dataset.fit_beams_and_baselines()


    # Generate output report
    with PdfPages(opts.outfilebase+'_'+opts.baseline+'.pdf') as pdf:
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
        pagetext  = pagetext + "\n"
        pagetext += (u"\nCurrent model AzEl=(%.3f, %.3f) deg" % (model_delta_az[0], model_delta_el[0]))
        pagetext += (u"\nMeasured coordinates using rough fit")
        pagetext += (u"\nMeasured AzEl=(%.3f, %.3f) deg" % (measured_delta_az[0], measured_delta_el[0]))
        pagetext  = pagetext + "\n"
        pagetext += (u"\nDetermine residuals from current pointing model")
        residual_az = measured_delta_az - model_delta_az
        residual_el = measured_delta_el - model_delta_el
        pagetext += (u"\nResidual AzEl=(%.3f, %.3f) deg" % (residual_az[0], residual_el[0]))
        if dataset.compscans[0].beam is not None:
            if not dataset.compscans[0].beam.is_valid:
                pagetext += (u"\nPossible bad fit!")
        if (residual_az[0] < 1.) and (residual_el[0] < 1.):
            pagetext += (u"\nResiduals withing L-band beam")
        else:
            pagetext += (u"\nMaximum Residual, %.2f, larger than L-band beam"%(numpy.max(residual_az[0], residual_el[0])))
        pagetext  = pagetext + "\n"
        pagetext += (u"\nFitted parameters \n%s" % str(params[:5]))

        plt.figure(None,figsize = (16,8))
        plt.axes(frame_on=False)
        plt.xticks([])
        plt.yticks([])
        plt.title("AR1 Report %s"%opts.outfilebase ,fontsize=14, fontweight="bold")
        plt.text(0,0,pagetext,fontsize=12)
        pdf.savefig()
        plt.close()
        pdf.savefig(fig)
        pdf.savefig(fig1)

        d = pdf.infodict()
        import datetime
        d['Title'] = h5file.description
        d['Author'] = 'AR1'
        d['Subject'] = 'AR1 check point source scan'
        d['CreationDate'] = datetime.datetime(2015, 8, 13)
        d['ModDate'] = datetime.datetime.today()

## -- Main --
if __name__ == '__main__':

    # Parse command-line opts and arguments
    parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                                   description="Reduced from of analyse_point_source_scan for quick evaluation of single dish performance.")
    parser.add_option("-a", "--baseline",
                      default='all',
                      help="Baseline to load (e.g. 'ant1' for antenna 1 or 'ant1,ant2' for 1-2 baseline), "
                           "default is all baseline in file")
    parser.add_option("-c", "--channel-mask",
                      default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle',
                      help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")
    parser.add_option("-f", "--freq-chans",
                      help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass)")
    parser.add_option("-m", "--monte-carlo",
                      dest="mc_iterations",
                      type='int',
                      default=1,
                      help="Number of Monte Carlo iterations to estimate uncertainty (20-30 suggested, default off)")
    parser.add_option("-n", "--nd-models",
                      default='/var/kat/katconfig/user/noise-diode-models/mkat/',
                      help="Name of optional directory containing noise diode model files")
    parser.add_option("-o", "--output",
                      dest="outfilebase",
                      help="Base name of output files (*.csv for output data and *.log for messages, "
                           "default is '<dataset_name>_point_source_scans')")
    parser.add_option("-p", "--pointing-model",
                      default='/var/kat/katconfig/user/pointing-models/mkat/',
                      help="Name of optional file containing pointing model parameters in degrees")
    parser.add_option("-s", "--plot-spectrum",
                      action="store_true",
                      default='False',
                      help="Flag to include spectral plot")
    parser.add_option("-t", "--time-offset",
                      type='float',
                      default=0.0,
                      help="Time offset to add to DBE timestamps, in seconds (default = %default)")
    parser.add_option("-u", "--ku-band",
                      action="store_true",
                      default=False,
                      help="Force center frequency to be 12500.5 MHz")
    parser.add_option('-v', '--verbose',
                      action='store_true',
                      dest='verbose',
                      default=False,
                      help='Display raster scan beam fit')
    parser.add_option('--keep-all',
                      action='store_true',
                      dest='keep_all',
                      default=False,
                      help='Keep scans with or without a valid beam in batch mode')
    (opts, args) = parser.parse_args()

    # Set defaults for pipeline workflow processing
    opts.keep_all=True
    opts.plot_spectrum=True

    if len(args) != 1 or not args[0].endswith('.h5'):
        parser.print_usage()
        raise RuntimeError('Please specify a single HDF5 file as argument to the script')

    print ('Loading HDF5 file %s into scape and reducing the data'%args[0])
    h5file = katdal.open(args[0])
    if opts.baseline == 'all': ants = [ant.name for ant in h5file.ants]
    else: ants = [opts.baseline]
    for ant in ants:
        opts.baseline = ant
        print("Loading dataset '%s'" % (opts.baseline,))
        analyse_point_source_scans(args[0], h5file, opts)

    # Display plots - this should be called ONLY ONCE, at the VERY END of the script
    # The script stops here until you close the plots...
    if opts.verbose: plt.show()

    # cleanup before exit
    try: plt.close('all')
    except: pass # nothing to close

# -fin-
