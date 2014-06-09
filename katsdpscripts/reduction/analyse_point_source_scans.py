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
        compscan.fit_beam_and_baselines('abs' + pol)
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

def reduce_compscan_with_uncertainty(dataset, compscan_index=0, mc_iterations=1, batch=True, **kwargs):
    """Do complete point source reduction on a compound scan, with uncertainty."""
    scan_dataset = dataset.select(labelkeep='scan', copy=False)
    compscan = scan_dataset.compscans[compscan_index]
    logger.info("==== Processing compound scan %d of %d: '%s' ====" % (compscan_index + 1, len(scan_dataset.compscans),
                                                                       ' '.join(compscan_key(compscan)),))
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
        logger.info("---- Monte Carlo iteration %d of %d ----" % (m + 2, mc_iterations))
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
    output_dict = {'keep' : keep, 'compscan' : main_compscan, 'unavg_dataset' : unavg_compscan_dataset}
    output_dict.update(fixed)
    output_dict.update(var_mean)
    output_dict.update(var_std)
    return output_dict

def reduce_and_plot(dataset, current_compscan, reduced_data, opts, fig=None, **kwargs):
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
        f = file(opts.outfilebase + '.csv', 'w')
        f.write('# antenna = %s\n' % dataset.antenna.description)
        f.write(', '.join(output_field_names) + '\n')
        f.writelines([output_fields % out for out in reduced_data if out and out['keep']])
        f.close()
        if not opts.batch:
            # This closes the GUI and effectively exits the program in the interactive case
            plt.close('all')
        return

    # Reduce current compound scan if results are not cached
    if not reduced_data[current_compscan]:
        reduced_data[current_compscan] = reduce_compscan_with_uncertainty(dataset, current_compscan,
                                                                          opts.mc_iterations, opts.batch, **kwargs)

    # Display compound scan
    if fig:
        out = reduced_data[current_compscan]
        # Display uncertainties if we are doing Monte Carlo
        if opts.mc_iterations > 1:
            offset_az = u"%.1f\u00B1%.3f" % (60. * out['delta_azimuth'], 60. * out['delta_azimuth_std'])
            offset_el = u"%.1f\u00B1%.3f" % (60. * out['delta_elevation'], 60. * out['delta_elevation_std'])
            beam_width = u"%.1f\u00B1%.2f" % (60. * out['beam_width_I'], 60. * out['beam_width_I_std'])
            beam_height = u"%.2f\u00B1%.5f" % (out['beam_height_I'], out['beam_height_I_std'])
            baseline_height = u"%.1f\u00B1%.4f" % (out['baseline_height_I'], out['baseline_height_I_std'])
        else:
            offset_az, offset_el = "%.1f" % (60. * out['delta_azimuth'],), "%.1f" % (60. * out['delta_elevation'],)
            beam_width, beam_height = "%.1f" % (60. * out['beam_width_I'],), "%.2f" % (out['beam_height_I'],)
            baseline_height = "%.1f" % (out['baseline_height_I'],)
        ax1, ax2, info, counter = fig.axes[0], fig.axes[1], fig.texts[0], fig.texts[1]
        ax1.clear()
        scape.plot_compound_scan_in_time(out['compscan'], ax=ax1)
        ax1.set_title(("%(dataset)s %(antenna)s '%(target)s'\nazel=(%(azimuth).1f, %(elevation).1f) deg, " % out) +
                      (u"offset=(%s, %s) arcmin" % (offset_az, offset_el)), size='medium')
        ax1.set_ylabel('Total power (%(data_unit)s)' % out)
        ax2.clear()
        scape.plot_compound_scan_on_target(out['compscan'], ax=ax2)
        if opts.plot_spectrum:
            ax3 = fig.axes[2]
            ax3.clear()
            scape.plot_xyz(out['unavg_dataset'], 'freq', 'amp', labels=[], power_in_dB=True, ax=ax3)
        if out['compscan'].beam:
            info.set_text((u"Beamwidth = %s' (expected %.1f')\nBeam height = %s %s\n"
                           u"HH/VV gain = %.3f/%.3f Jy/%s\nBaseline height = %s %s") %
                          (beam_width, 60. * out['beam_expected_width_I'], beam_height, out['data_unit'],
                           out['flux'] / out['beam_height_HH'], out['flux'] / out['beam_height_VV'], out['data_unit'],
                           baseline_height, out['data_unit']))
        else:
            info.set_text(u"No beam\nBaseline height = %s %s" % (baseline_height, out['data_unit']))
        counter.set_text("compscan %d of %d" % (current_compscan + 1, len(reduced_data)))
        plt.draw()

    # Reduce next compound scan so long, as this will improve interactiveness (i.e. next plot will be immediate)
    if (current_compscan < len(reduced_data) - 1) and not reduced_data[current_compscan + 1]:
        reduced_data[current_compscan + 1] = reduce_compscan_with_uncertainty(dataset, current_compscan + 1,
                                                                              opts.mc_iterations, opts.batch, **kwargs)

