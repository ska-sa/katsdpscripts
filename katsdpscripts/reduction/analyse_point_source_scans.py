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

import os.path
import logging
import pickle

import numpy as np
import scape
import katpoint
import scikits.fitting as fit

try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets
except ImportError:
    plt = widgets = None


def interp_sensor(compscan, quantity, default):
    """Interpolate environmental sensor data."""
    try:
        sensor = compscan.dataset.enviro[quantity]
    except KeyError:
        return (lambda times: default)
    else:
        interp = fit.PiecewisePolynomial1DFit(max_degree=0)
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
    beams_std = []
    for pol in beam_pols:
        compscan.fit_beam_and_baselines(pol='abs' + pol)
        bh = compscan.baseline_height()
        if bh is None:
            bh = np.nan
        beam_params = [compscan.beam.height, katpoint.rad2deg(np.mean(compscan.beam.width)), bh,
                       float(compscan.beam.refined)] if compscan.beam else [np.nan, np.nan, bh, 0.]
        beams.append((pol, beam_params))
        beams_std.append((pol, [compscan.beam.std_height, compscan.beam.std_width] if compscan.beam
                               else [np.nan, np.nan]))

    # Obtain environmental data averaged across the compound scan
    compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
    temperature = np.mean(interp_sensor(compscan, 'temperature', 35.0)(compscan_times))
    pressure = np.mean(interp_sensor(compscan, 'pressure', 950.0)(compscan_times))
    humidity = np.mean(interp_sensor(compscan, 'humidity', 15.0)(compscan_times))
    # Do a 2-D vector average of wind speed + direction
    raw_wind_speed = interp_sensor(compscan, 'wind_speed', 0.0)(compscan_times)
    raw_wind_direction = interp_sensor(compscan, 'wind_direction', 0.0)(compscan_times)
    mean_north_wind = np.mean(raw_wind_speed * np.cos(np.radians(raw_wind_direction)))
    mean_east_wind = np.mean(raw_wind_speed * np.sin(np.radians(raw_wind_direction)))
    wind_speed = np.sqrt(mean_north_wind ** 2 + mean_east_wind ** 2)
    wind_direction = np.degrees(np.arctan2(mean_east_wind, mean_north_wind))
    wind_std = np.std(raw_wind_speed)
    sun = katpoint.Target('Sun, special')
    # Calculate pointing offset
    # Obtain middle timestamp of compound scan, where all pointing calculations are done
    middle_time = np.median(compscan_times, axis=None)
    # work out the sun's angle
    sun_azel = katpoint.rad2deg(np.array(sun.azel(middle_time, antenna=compscan.dataset.antenna)))
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
                  'temperature pressure humidity wind_speed wind_direction wind_std ' \
                  'azimuth elevation beam_expected_width_I sun_az sun_el timestamp'
    fixed = list(compscan_key(compscan)) + \
        [compscan.dataset.data_unit, compscan.dataset.freqs[0]] + \
        [average_flux, temperature, pressure, humidity, wind_speed, wind_direction, wind_std] + \
        requested_azel.tolist() + [expected_width] + sun_azel.tolist() + [middle_time]
    # Outputs that are expected to change if visibility data is perturbed
    var_names = 'delta_azimuth delta_elevation'
    variable = offset_azel.tolist()
    for beam in beams:
        var_names += ' beam_height_%s beam_width_%s baseline_height_%s refined_%s' % tuple([beam[0]] * 4)
        variable += list(beam[1])
    for beam in beams_std:
        var_names += ' beam_height_%s_std beam_width_%s_std' % tuple([beam[0]] * 2)
        variable += list(beam[1])
    return dict(zip(fixed_names.split(), fixed)), dict(zip(var_names.split(), variable))


def extract_cal_dataset(dataset):
    """Build data set from scans in original dataset containing noise diode firings."""
    compscanlist = []
    for compscan in dataset.compscans:
        # Extract scans containing noise diode firings (make copy, as this will be modified by gain cal)
        # Don't rely on 'cal' labels, as the KAT-7 system does not produce it anymore
        scanlist = [scan.select(copy=True) for scan in compscan.scans
                    if 'nd_on' in scan.flags.dtype.fields and scan.flags['nd_on'].any()]
        if scanlist:
            compscanlist.append(scape.CompoundScan(scanlist, compscan.target))
    return scape.DataSet(None, compscanlist, dataset.experiment_id, dataset.observer,
                         dataset.description, dataset.data_unit, dataset.corrconf.select(copy=True),
                         dataset.antenna, dataset.antenna2, dataset.nd_h_model, dataset.nd_v_model, dataset.enviro)


def reduce_compscan_with_uncertainty(dataset, compscan_index=0, mc_iterations=1, batch=True,
                                     keep_all=True, num_compscans=0, **kwargs):
    """Do complete point source reduction on a compound scan, with uncertainty."""
    dataset = scape.DataSet(None, [dataset.compscans[compscan_index]], dataset.experiment_id, dataset.observer,
                            dataset.description, dataset.data_unit, dataset.corrconf,
                            dataset.antenna, dataset.antenna2, dataset.nd_h_model, dataset.nd_v_model, dataset.enviro)
    scan_dataset = dataset.select(labelkeep='scan', copy=False)
    compscan = scan_dataset.compscans[0]
    if 'logger' in kwargs:
        kwargs['logger'].info("==== Processing compound scan %d of %d: '%s' ====",
                              compscan_index + 1, num_compscans, ' '.join(compscan_key(compscan)))
    # Build data set containing a single compound scan at a time (make copy, as reduction modifies it)
    scan_dataset.compscans = [compscan]
    # If there are no noise diode models assume that there are no noise diodes
    if dataset.nd_h_model is not None and dataset.nd_v_model is not None:
        compscan_dataset = scan_dataset.select(flagkeep='~nd_on', copy=True)
    else:
        compscan_dataset = dataset.select(labelkeep='scan', copy=True)
    cal_dataset = extract_cal_dataset(dataset)
    # Do first reduction run
    main_compscan = compscan_dataset.compscans[0]
    fixed, variable = reduce_compscan(main_compscan, cal_dataset, **kwargs)
    # Produce data set that has counts converted to Kelvin, but no averaging (for spectral plots)
    unavg_compscan_dataset = scan_dataset.select(flagkeep='~nd_on', copy=True)
    unavg_compscan_dataset.nd_gain = cal_dataset.nd_gain
    unavg_compscan_dataset.convert_power_to_temperature()
    # Add data from Monte Carlo perturbations
    iter_outputs = [np.rec.fromrecords([tuple(variable.values())], names=list(variable.keys()))]
    for m in range(mc_iterations - 1):
        if 'logger' in kwargs:
            kwargs['logger'].info("---- Monte Carlo iteration %d of %d ----",
                                  m + 2, mc_iterations)
        compscan_dataset = scan_dataset.select(flagkeep='~nd_on', copy=True).perturb()
        cal_dataset = extract_cal_dataset(dataset).perturb()
        fixed, variable = reduce_compscan(compscan_dataset.compscans[0], cal_dataset, **kwargs)
        iter_outputs.append(np.rec.fromrecords([tuple(variable.values())], names=list(variable.keys())))
    # Get mean and uncertainty of variable part of output data (assumed to be floats)
    var_output = np.concatenate(iter_outputs).view(float).reshape(mc_iterations, -1)
    var_mean = dict(zip(variable.keys(), var_output.mean(axis=0)))
    var_std = {k+"_std":v for k,v in zip(variable, var_output.std(axis=0)) if not k.endswith("_std")}
    if (mc_iterations <= 1): # If no Monte Carlo then take forward the first reduction run's residuals
        var_std.update({k:v for k,v in variable.items() if k.endswith("_std")})
    # Keep scan only with a valid beam in batch mode (otherwise keep button has to do it explicitly)
    keep = batch and main_compscan.beam is not None and (keep_all or main_compscan.beam.is_valid)
    if 'logger' in kwargs:
        kwargs['logger'].debug("keep_all=%s, main_compscan.beam.is_valid=%s, keep=%s",
                               keep_all, main_compscan.beam is not None and main_compscan.beam.is_valid, keep)
    output_dict = {'keep': keep, 'compscan': main_compscan, 'unavg_dataset': unavg_compscan_dataset}
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


def reduce_and_plot(dataset, current_compscan, reduced_data, opts, fig=None, **kwargs):
    """Reduce compound scan, update the plots in given figure and save reduction output when done."""
    # Save reduction output and return after last compound scan is done
    if current_compscan >= len(reduced_data):
        output_fields = '%(dataset)s, %(target)s, %(timestamp_ut)s, %(azimuth).7f, %(elevation).7f, ' \
                        '%(delta_azimuth).7f, %(delta_azimuth_std).7f, ' \
                        '%(delta_elevation).7f, %(delta_elevation_std).7f, ' \
                        '%(data_unit)s, %(beam_height_I).7f, %(beam_height_I_std).7f, ' \
                        '%(beam_width_I).7f, %(beam_width_I_std).7f, ' \
                        '%(baseline_height_I).7f, %(baseline_height_I_std).7f, %(refined_I).7f, ' \
                        '%(beam_height_HH).7f, %(beam_width_HH).7f, %(baseline_height_HH).7f, %(refined_HH).7f, ' \
                        '%(beam_height_VV).7f, %(beam_width_VV).7f, %(baseline_height_VV).7f, %(refined_VV).7f, ' \
                        '%(frequency).7f, %(flux).4f, ' \
                        '%(temperature).2f, %(pressure).2f, %(humidity).2f, %(wind_speed).2f, ' \
                        '%(wind_direction).2f, %(wind_std).2f, %(sun_az).7f, %(sun_el).7f, %(timestamp)i \n'
        output_field_names = [name.partition(')')[0] for name in output_fields[2:].split(', %(')]
        output_data = [output_fields % out for out in reduced_data if out and out['keep']]
        f = open(opts.outfilebase + '.csv', 'w')
        f.write('# antenna = %s\n' % dataset.antenna.description)
        f.write(', '.join(output_field_names) + '\n')
        f.writelines(output_data)
        f.close()
        if not opts.batch:
            # This closes the GUI and effectively exits the program in the interactive case
            plt.close('all')
        # return the recarray
        to_keep = []
        for field in output_field_names:
            to_keep.append([data[field] for data in reduced_data if data and data['keep']])
        output_data = np.rec.fromarrays(to_keep, dtype=list(zip(output_field_names, [np.array(tk).dtype for tk in to_keep])))
        return (dataset.antenna, output_data,)

    # Reduce current compound scan if results are not cached
    if not reduced_data[current_compscan]:
        with SuppressErrors(kwargs['logger']):
            reduced_data[current_compscan] = \
                reduce_compscan_with_uncertainty(dataset, current_compscan, opts.mc_iterations,
                                                 opts.batch, opts.keep_all, len(reduced_data), **kwargs)

    # Display compound scan
    if fig:
        ax1, ax2, info, counter = fig.axes[0], fig.axes[1], fig.texts[0], fig.texts[1]
        ax1.clear()
        ax2.clear()
        if opts.plot_spectrum:
            ax3 = fig.axes[2]
            ax3.clear()
        out = reduced_data[current_compscan]
        if 'compscan' in out:
            # Display uncertainties if we are doing Monte Carlo
            if opts.mc_iterations > 1:
                offset_az = "%.1f+/-%.3f" % (60. * out['delta_azimuth'], 60. * out['delta_azimuth_std'])
                offset_el = "%.1f+/-%.3f" % (60. * out['delta_elevation'], 60. * out['delta_elevation_std'])
                beam_width = "%.1f+/-%.2f" % (60. * out['beam_width_I'], 60. * out['beam_width_I_std'])
                beam_height = "%.2f+/-%.5f" % (out['beam_height_I'], out['beam_height_I_std'])
                baseline_height = "%.1f+/-%.4f" % (out['baseline_height_I'], out['baseline_height_I_std'])
            else:
                offset_az, offset_el = "%.1f" % (60. * out['delta_azimuth'],), "%.1f" % (60. * out['delta_elevation'],)
                beam_width, beam_height = "%.1f" % (60. * out['beam_width_I'],), "%.2f" % (out['beam_height_I'],)
                baseline_height = "%.1f" % (out['baseline_height_I'],)
            scape.plot_compound_scan_in_time(out['compscan'], ax=ax1)
            ax1.set_title(("%(dataset)s %(antenna)s '%(target)s'\nazel=(%(azimuth).1f, %(elevation).1f) deg, " % out) +
                          ("offset=(%s, %s) arcmin" % (offset_az, offset_el)), size='medium')
            ax1.set_ylabel('Total power (%(data_unit)s)' % out)
            scape.plot_compound_scan_on_target(out['compscan'], ax=ax2)
            if opts.plot_spectrum:
                scape.plot_xyz(out['unavg_dataset'], 'freq', 'amp', labels=[], power_in_dB=True, ax=ax3)
            if out['compscan'].beam:
                info.set_text(("Beamwidth = %s' (expected %.1f')\nBeam height = %s %s\n"
                               "HH/VV gain = %.3f/%.3f Jy/%s\nBaseline height = %s %s") %
                              (beam_width, 60. * out['beam_expected_width_I'], beam_height, out['data_unit'],
                               out['flux'] / out['beam_height_HH'], out['flux'] / out['beam_height_VV'],
                               out['data_unit'], baseline_height, out['data_unit']))
            else:
                info.set_text("No beam\nBaseline height = %s %s" % (baseline_height, out['data_unit']))
        else:
            info.set_text("Reduction failed")
        counter.set_text("compscan %d of %d" % (current_compscan + 1, len(reduced_data)))
        plt.draw()

    # Reduce next compound scan so long, as this will improve interactiveness (i.e. next plot will be immediate)
    if (current_compscan < len(reduced_data) - 1) and not reduced_data[current_compscan + 1]:
        with SuppressErrors(kwargs['logger']):
            reduced_data[current_compscan + 1] = \
                reduce_compscan_with_uncertainty(dataset, current_compscan + 1, opts.mc_iterations,
                                                 opts.batch, opts.keep_all, len(reduced_data), **kwargs)


def analyse_point_source_scans(dataset, opts):
    """ 'dataset' may be either a string file name or any data object accepted by scape.DataSet(). """
    if isinstance(dataset, str):
        filename = dataset
    else: # Get filename from dataset. This code should work for katdal datasets including v4
        filename = dataset.name.split("|")[0]

    # Produce canonical version of baseline string (remove duplicate antennas)
    baseline_ants = opts.baseline.split(',')
    if len(baseline_ants) == 2 and baseline_ants[0] == baseline_ants[1]:
        opts.baseline = baseline_ants[0]

    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    # Default output file names are based on input file name
    if opts.outfilebase is None:
        opts.outfilebase = dataset_name + '_' + opts.baseline + '_point_source_scans'

    # Set up logging: logging everything (DEBUG & above), both to console and file
    logger = logging.root
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(opts.outfilebase + '.log', 'w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(fh)

    kwargs = {}

    # Force centre freqency if ku-band option is set
    if opts.ku_band:
        kwargs['centre_freq'] = 12.5005e9

    if opts.freq_centre is not None:
        kwargs['centre_freq'] = float(opts.freq_centre) * 1e6

    # Load old CSV file used to select compound scans from dataset
    keep_scans = keep_datasets = None
    if opts.keepfilename:
        ant_name = katpoint.Antenna(open(opts.keepfilename).readline().strip().partition('=')[2]).name
        try:
            data = np.loadtxt(opts.keepfilename, dtype='string', comments='#', delimiter=', ')
        except ValueError:
            raise ValueError("CSV file %r contains rows with a different number of columns/commas" % opts.keepfilename)
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
        # Ensure we are using antenna found in CSV file (this assumes single dish setup for now)
        csv_baseline = ant_name
        if opts.baseline != 'sd' and opts.baseline != csv_baseline:
            logger.warn("Requested baseline '%s' does not match baseline '%s' in CSV file '%s'" %
                        (opts.baseline, csv_baseline, opts.keepfilename))
        logger.warn("Using baseline '%s' found in CSV file '%s'" % (csv_baseline, opts.keepfilename))
        opts.baseline = csv_baseline

    # Avoid loading the data set if it does not appear in specified CSV file
    if keep_datasets and dataset_name not in keep_datasets:
        raise RuntimeError("Skipping dataset '%s' (based on CSV file)" % (filename,))

    # Load data set
    logger.info("Loading dataset '%s'" % (filename,))
    dataset = scape.DataSet(dataset, baseline=opts.baseline, nd_models=opts.nd_models,
                            time_offset=opts.time_offset, katfile=not opts.old_loader, **kwargs)

    # Select frequency channels and setup defaults if not specified
    num_channels = len(dataset.channel_select)
    if opts.freq_chans is None:
        # Default is drop first and last 25% of the bandpass
        start_chan = num_channels // 4
        end_chan = start_chan * 3
    else:
        start_chan = int(opts.freq_chans.split(',')[0])
        end_chan = int(opts.freq_chans.split(',')[1])
    chan_select = list(range(start_chan, end_chan + 1))
    # Check if a channel mask is specified and apply
    if opts.channel_mask:
        chan_select = ~(load_rfi_static_mask(opts.channel_mask, dataset.freqs))
        chan_select[:start_chan] = False
        chan_select[end_chan:] = False
    dataset = dataset.select(freqkeep=chan_select)

    # Check scan count
    if len(dataset.compscans) == 0 or len(dataset.scans) == 0:
        raise RuntimeError('No scans found in file, skipping data set')
    scan_dataset = dataset.select(labelkeep='scan', copy=False)
    if len(scan_dataset.compscans) == 0 or len(scan_dataset.scans) == 0:
        raise RuntimeError('No scans left after standard reduction, skipping '
                           'data set (no scans labelled "scan", perhaps?)')
    # Override pointing model if it is specified (useful if it is not in data file, like on early KAT-7)
    if opts.pointing_model:
        pm = open(opts.pointing_model).readline().strip()
        logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(pm.split(',')), opts.pointing_model))
        dataset.antenna.pointing_model = katpoint.PointingModel(pm)

    # Remove any noise diode models if the ku band option is set and flag for spikes
    if opts.ku_band:
        dataset.nd_h_model = None
        dataset.nd_v_model = None
    if opts.remove_spikes:
        for i in range(len(dataset.scans)):
            dataset.scans[i].data = scape.stats.remove_spikes(dataset.scans[i].data, axis=1,
                                                              spike_width=3, outlier_sigma=5.)

    # Initialise the output data cache (None indicates the compscan has not been processed yet)
    reduced_data = [{} for n in range(len(scan_dataset.compscans))]

    # BATCH MODE #

    # This will cycle through all data sets and stop when done
    if opts.batch:
        # Go one past the end of compscan list to write the output data out to CSV file
        for current_compscan in range(len(scan_dataset.compscans) + 1):
            # Look up compscan key in list of compscans to keep (if provided, only applicable to batch mode anyway)
            if keep_scans and (current_compscan < len(scan_dataset.compscans)):
                cs_key = ' '.join(compscan_key(scan_dataset.compscans[current_compscan]))
                if cs_key not in keep_scans:
                    logger.info("==== Skipping compound scan '%s' (based on CSV file) ====" % (cs_key,))
                    continue
            output = reduce_and_plot(dataset, current_compscan, reduced_data, opts, logger=logger)
        return output

    # INTERACTIVE MODE #
    else:
        if not plt:
            raise ImportError('Interactive use of this script requires matplotlib - '
                              'please install it or run in batch mode')
        # Set up figure with buttons
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

        # Store reference to widget.Buttons on figure to keep them from being
        # garbage collected, resulting in unresponsive buttons. This is especially
        # an issue with IPython where the script does not block on plt.show().
        # See the docstring of matplotlib.widgets.AxesWidget for more details.
        fig.temporarily_disable_all_buttons = DisableButtons()

        # Create buttons and their callbacks
        def spectrogram_callback(event):
            with fig.temporarily_disable_all_buttons:
                plt.figure(2)
                plt.clf()
                out = reduced_data[fig.current_compscan]
                ax = scape.plot_xyz(out['unavg_dataset'], 'time', 'freq', 'amp', power_in_dB=True)
                ax.set_title(out['target'], size='medium')
        spectrogram_button = widgets.Button(plt.axes([0.37, 0.05, 0.1, 0.075]), 'Spectrogram')
        spectrogram_button.on_clicked(spectrogram_callback)
        fig.temporarily_disable_all_buttons.append(spectrogram_button)

        def keep_callback(event):
            with fig.temporarily_disable_all_buttons:
                reduced_data[fig.current_compscan]['keep'] = True
                fig.current_compscan += 1
                reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)
        keep_button = widgets.Button(plt.axes([0.48, 0.05, 0.1, 0.075]), 'Keep')
        keep_button.on_clicked(keep_callback)
        fig.temporarily_disable_all_buttons.append(keep_button)

        def discard_callback(event):
            with fig.temporarily_disable_all_buttons:
                reduced_data[fig.current_compscan]['keep'] = False
                fig.current_compscan += 1
                reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)
        discard_button = widgets.Button(plt.axes([0.59, 0.05, 0.1, 0.075]), 'Discard')
        discard_button.on_clicked(discard_callback)
        fig.temporarily_disable_all_buttons.append(discard_button)

        def back_callback(event):
            with fig.temporarily_disable_all_buttons:
                if fig.current_compscan > 0:
                    fig.current_compscan -= 1
                    reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)
        back_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Back')
        back_button.on_clicked(back_callback)
        fig.temporarily_disable_all_buttons.append(back_button)

        def done_callback(event):
            with fig.temporarily_disable_all_buttons:
                fig.current_compscan = len(reduced_data)
                reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)
        done_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
        done_button.on_clicked(done_callback)
        fig.temporarily_disable_all_buttons.append(done_button)

        # Start off the processing on the first compound scan
        fig.current_compscan = 0
        reduce_and_plot(dataset, fig.current_compscan, reduced_data, opts, fig, logger=logger)
        # Display plots - this should be called ONLY ONCE, at the VERY END of the script
        # The script stops here until you close the plots...
        plt.show()


def batch_mode_analyse_point_source_scans(filename, outfilebase=None, keepfilename=None, baseline='sd',
                                          mc_iterations=1, time_offset=0.0, pointing_model=None,
                                          freq_chans=None, old_loader=None, nd_models=None,
                                          ku_band=False, channel_mask=None, keep_all=False,
                                          remove_spikes=False, freq_centre=None):

    class FakeOptsForBatch(object):
        batch = True  # always batch
        plot_spectrum = False  # never plot

        def __init__(self, outfilebase, keepfilename, baseline,
                     mc_iterations, time_offset, pointing_model, freq_chans,
                     old_loader, nd_models, ku_band, channel_mask, keep_all,
                     remove_spikes, freq_centre):
            self.outfilebase = outfilebase
            self.keepfilename = keepfilename
            self.baseline = baseline
            self.mc_iterations = mc_iterations
            self.time_offset = time_offset
            self.pointing_model = pointing_model
            self.freq_chans = freq_chans
            self.old_loader = old_loader
            self.nd_models = nd_models
            self.ku_band = ku_band
            self.channel_mask = channel_mask
            self.channel_mask = channel_mask
            self.keep_all = keep_all
            self.remove_spikes = remove_spikes
            self.freq_centre = freq_centre

    fake_opts = FakeOptsForBatch(outfilebase=outfilebase, keepfilename=keepfilename,
                                 baseline=baseline, mc_iterations=mc_iterations,
                                 time_offset=time_offset, pointing_model=pointing_model,
                                 freq_chans=freq_chans, old_loader=old_loader,
                                 nd_models=nd_models, ku_band=ku_band,
                                 channel_mask=channel_mask, keep_all=keep_all,
                                 remove_spikes=remove_spikes, freq_centre=freq_centre)
    (dataset_antenna, output_data,) = analyse_point_source_scans(filename, fake_opts)

    return dataset_antenna, output_data


# Copied from AR1/reduction/interferometric_pointing/analyse_interferometric_pointing.py
def load_rfi_static_mask(filename, freqs, debug_chunks=0):
    # Construct a mask either from a pickle file, or a text file with frequency ranges
    nchans = len(freqs)
    channel_width = abs(freqs[1]-freqs[0])
    try:
        with open(filename, "rb") as pickle_file:
            channel_flags = pickle.load(pickle_file)
        nflags = len(channel_flags)
        if (nchans != nflags):
            print("Warning channel mask (%d) is stretched to fit dataset (%d)!"%(nflags,nchans))
            N = nchans/float(nflags)
            channel_flags = np.repeat(channel_flags, int(N+0.5)) if (N > 1) else channel_flags[::int(1/N)]
        channel_flags = channel_flags[:nchans] # Clip, just in case
    except pickle.UnpicklingError: # Not a pickle file, perhaps a plain text file with frequency ranges in MHz?
        mask_ranges = np.loadtxt(filename, comments='#', delimiter=',')
        channel_flags = np.full((nchans,), False)
        low = freqs - 0.5 * channel_width
        high = freqs + 0.5 * channel_width
        for r in mask_ranges:
            in_range = (low <= r[1]*1e6) & (r[0]*1e6 <= high)
            idx = np.where(in_range)[0]
            channel_flags[idx] = True
    if debug_chunks > 0:
        for chunk in range(debug_chunks):
            freq = slice(chunk*(nchans//debug_chunks),(chunk+1)*(nchans//debug_chunks))
            masked_f = freqs[freq][channel_flags[freq]]
            if (len(masked_f) > 0):
                mBW = len(masked_f)*(freqs[1]-freqs[0])
                print("\tFreq. chunk %d: mask omits %.1fMHz between (%.1f - %.1f)MHz"%(chunk,mBW/1e6,np.min(masked_f)/1e6,np.max(masked_f)/1e6))
            else:
                print("\tFreq. chunk %d: mask omits nothing"%chunk)
    return channel_flags
