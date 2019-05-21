from __future__ import absolute_import
from __future__ import print_function
import os
import shutil
import time
import itertools
import multiprocessing

import katdal
from katdal.h5datav3 import FLAG_NAMES
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.lazy_indexer import LazyTransform
import katpoint

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import dask.array as da
import pickle
import h5py
import concurrent.futures

from katsdpsigproc.rfi.twodflag import SumThresholdFlagger
import six
from six.moves import range


def plot_RFI_mask(pltobj, main=True, extra=None, channelwidth=1e6):

    if main:
        pltobj.axvspan(1674e6, 1677e6, alpha=0.3, color='grey')  # Meteosat
        pltobj.axvspan(1667e6, 1667e6, alpha=0.3, color='grey')  # Fengun
        pltobj.axvspan(1682e6, 1682e6, alpha=0.3, color='grey')  # Meteosat
        pltobj.axvspan(1685e6, 1687e6, alpha=0.3, color='grey')  # Meteosat
        pltobj.axvspan(1687e6, 1687e6, alpha=0.3, color='grey')  # Fengun
        pltobj.axvspan(1690e6, 1690e6, alpha=0.3, color='grey')  # Meteosat
        pltobj.axvspan(1699e6, 1699e6, alpha=0.3, color='grey')  # Meteosat
        pltobj.axvspan(1702e6, 1702e6, alpha=0.3, color='grey')  # Fengyun
        pltobj.axvspan(1705e6, 1706e6, alpha=0.3, color='grey')  # Meteosat
        pltobj.axvspan(1709e6, 1709e6, alpha=0.3, color='grey')  # Fengun
        pltobj.axvspan(1501e6, 1570e6, alpha=0.3, color='blue')  # Inmarsat
        pltobj.axvspan(1496e6, 1585e6, alpha=0.3, color='blue')  # Inmarsat
        pltobj.axvspan(1574e6, 1576e6, alpha=0.3, color='blue')  # Inmarsat
        pltobj.axvspan(1509e6, 1572e6, alpha=0.3, color='blue')  # Inmarsat
        pltobj.axvspan(1574e6, 1575e6, alpha=0.3, color='blue')  # Inmarsat
        pltobj.axvspan(1512e6, 1570e6, alpha=0.3, color='blue')  # Thuraya
        pltobj.axvspan(1450e6, 1498e6, alpha=0.3, color='red')  # Afristar
        pltobj.axvspan(1652e6, 1694e6, alpha=0.2, color='red')  # Afristar
        pltobj.axvspan(1542e6, 1543e6, alpha=0.3, color='cyan')  # Express AM1
        pltobj.axvspan(1554e6, 1554e6, alpha=0.3, color='cyan')  # Express AM 44
        pltobj.axvspan(1190e6, 1215e6, alpha=0.3, color='green')  # Galileo
        pltobj.axvspan(1260e6, 1300e6, alpha=0.3, color='green')  # Galileo
        pltobj.axvspan(1559e6, 1591e6, alpha=0.3, color='green')  # Galileo
        pltobj.axvspan(1544e6, 1545e6, alpha=0.3, color='green')  # Galileo
        pltobj.axvspan(1190e6, 1217e6, alpha=0.3, color='green')  # Beidou
        pltobj.axvspan(1258e6, 1278e6, alpha=0.3, color='green')  # Beidou
        pltobj.axvspan(1559e6, 1563e6, alpha=0.3, color='green')  # Beidou
        pltobj.axvspan(1555e6, 1596e6, alpha=0.3, color='green')  # GPS L1  1555 -> 1596
        pltobj.axvspan(1207e6, 1238e6, alpha=0.3, color='green')  # GPS L2  1207 -> 1248
        pltobj.axvspan(1378e6, 1384e6, alpha=0.3, color='green')  # GPS L3
        pltobj.axvspan(1588e6, 1615e6, alpha=0.3, color='green')  # GLONASS  1588 -> 1615 L1
        pltobj.axvspan(1232e6, 1259e6, alpha=0.3, color='green')  # GLONASS  1232 -> 1259 L2
        pltobj.axvspan(1616e6, 1630e6, alpha=0.3, color='grey')  # IRIDIUM
    if extra is not None:
        for i in range(extra.shape[0]):
            pltobj.axvspan(extra[i]-channelwidth/2, extra[i]+channelwidth/2, alpha=0.1, color='Maroon')


def get_flag_stats(mvf, thisdata=None, flags=None, flags_to_show=None, norm_spec=None):
    """
    Given a katdal object, remove a dc offset for each record
    (ignoring severe spikes) then obtain an average spectrum of
    all of the scans in the data.
    Return the average spectrum with dc offset removed and the number of times
    each channel is flagged (flags come optionally from 'flags' else from the
    flags in the input katdal object). Optinally provide a
    spectrum (norm_spec) to divide into the calculated bandpass.
    """
    targets = mvf.catalogue.targets
    flag_stats = {}
    if flags is None:
        flags = np.empty(mvf.shape, dtype=np.bool)
        mvf.select(flags=flags_to_show)
        for dump in range(mvf.shape[0]):
            flags[dump] = mvf.flags[dump]
    # Squeeze here removes stray axes left over by LazyIndexer
    if thisdata is None:
        thisdata = np.empty(mvf.shape, dtype=np.float32)
        for dump in range(mvf.shape[0]):
            thisdata[dump] = np.abs(mvf.vis[dump])
    if norm_spec is not None:
        thisdata /= norm_spec[np.newaxis, :]
    # Get DC height (median rather than mean is more robust...)
    data = np.ma.MaskedArray(thisdata, mask=flags, copy=False).filled(fill_value=np.nan)
    offset = np.nanmedian(data, axis=1)
    # Remove the DC height
    weights = np.logical_not(flags).astype(np.int8)
    data /= np.expand_dims(offset, axis=1)
    # Get the results for all of the data
    weightsum = weights[mvf.dumps].sum(axis=0, dtype=np.int)
    averagespec = np.nanmean(data[mvf.dumps], axis=0)
    flagfrac = 1. - weightsum/mvf.shape[0].astype(np.float)
    flag_stats['all_data'] = {'spectrum': averagespec, 'numrecords_tot': mvf.shape[0],
                              'flagfrac': flagfrac, 'channel_freqs': mvf.channel_freqs,
                              'dump_period': mvf.dump_period, 'corr_products': mvf.corr_products}
    # And for each target
    for t in targets:
        mvf.select(targets=t, scans='~slew')
        weightsum = (weights[mvf.dumps]).sum(axis=0, dtype=np.int).squeeze()
        averagespec = np.nanmean(data[mvf.dumps], axis=0)
        flagfrac = 1. - weightsum/mvf.shape[0].astype(np.float)
        flag_stats[t.name] = {'spectrum': averagespec, 'numrecords_tot': mvf.shape[0],
                              'flagfrac': flagfrac, 'channel_freqs': mvf.channel_freqs,
                              'dump_period': mvf.dump_period, 'corr_products': mvf.corr_products}
    mvf.select(reset='T')
    return flag_stats


def plot_flag_data(label, spectrum, flagfrac, freqs, pdf, mask=None):
    """
    Produce a plot of the average spectrum in H and V
    after flagging and attach it to the pdf output.
    Also show fraction of times flagged per channel.
    """
    from katsdpscripts import git_info

    repo_info = git_info()

    # Set up the figure
    fig = plt.figure(figsize=(11.7, 8.3))
    # Plot the spectrum
    ax1 = fig.add_subplot(211)
    ax1.text(0.01, 0.90, repo_info, horizontalalignment='left', fontsize=10, transform=ax1.transAxes)
    ax1.set_title(label)
    plt.plot(freqs, spectrum, linewidth=.5)
    ticklabels = ax1.get_xticklabels()
    plt.setp(ticklabels, visible=False)
    ticklabels = ax1.get_yticklabels()
    plt.setp(ticklabels, visible=False)
    plt.xlim((min(freqs), max(freqs)))
    plt.ylabel('Mean amplitude\n(arbitrary units)')
    # Plot the flags occupancies
    ax = fig.add_subplot(212, sharex=ax1)
    plt.plot(freqs, flagfrac, 'r-', linewidth=.5)
    plt.ylim((0., 1.))
    plt.axhline(0.8, color='red', linestyle='dashed', linewidth=.5)
    plt.xlim((min(freqs), max(freqs)))
    minorLocator = ticker.MultipleLocator(10e6)
    plt.ylabel('Fraction flagged')
    ticklabels = ax.get_yticklabels()
    # Convert ticks to MHZ
    ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
    ax.xaxis.set_major_formatter(ticks)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.xlabel('Frequency (MHz)')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close('all')


def plot_waterfall_subsample(visdata, flagdata, freqs=None, times=None, label='', resolution=150, output=None):
    """
    Make a waterfall plot from visdata with flags overplotted.
    """
    from datetime import datetime as dt
    import matplotlib.dates as mdates
    from katsdpscripts import git_info

    repo_info = git_info()

    fig = plt.figure(figsize=(8.3, 11.7))
    ax = plt.subplot(111)
    ax.set_title(label)
    ax.text(0.01, 0.02, repo_info, horizontalalignment='left', fontsize=10, transform=ax.transAxes)
    display_limits = ax.get_window_extent()
    if freqs is None:
        freqs = list(range(0, visdata.shape[1]))
    # 300dpi, and one pixel per desired data-point in pixels at 300dpi
    display_width = display_limits.width * resolution/72.
    display_height = display_limits.height * resolution/72.
    x_step = max(int(visdata.shape[1]/display_width), 1)
    y_step = max(int(visdata.shape[0]/display_height), 1)
    x_slice = slice(0, -1, x_step)
    y_slice = slice(0, -1, y_step)
    data = np.log10(np.abs(visdata[y_slice, x_slice]))
    flags = flagdata[y_slice, x_slice]
    plotflags = np.zeros(flags.shape[0:2]+(4,))
    plotflags[:, :, 0] = 1.0
    plotflags[:, :, 3] = flags
    if times is None:
        starttime = 0
        endtime = visdata.shape[0]
    else:
        starttime = mdates.date2num(dt.fromtimestamp(times[0]))
        endtime = mdates.date2num(dt.fromtimestamp(times[-1]))
    kwargs = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'none',
              'extent': (freqs[0], freqs[-1], starttime, endtime)}
    image = ax.imshow(data, **kwargs)
    image.set_cmap('Greys')
    ax.imshow(plotflags, alpha=0.5, **kwargs)
    ampsort = np.sort(data[~flags], axis=None)
    arrayremove = int(len(ampsort)*(1.0 - 0.80)/2.0)
    lowcut, highcut = ampsort[arrayremove], ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    plt.xlim((min(freqs), max(freqs)))
    if times is not None:
        ax.yaxis_date()
        plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.ylabel('Time (SAST)')
    else:
        plt.ylabel('Time (Dumps)')
    # Convert ticks to MHZ
    ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
    ax.xaxis.set_major_formatter(ticks)
    plt.xlabel('Frequency (MHz)')
    if output:
        output.savefig(fig)
    else:
        plt.show()
    plt.close('all')


def plot_waterfall(visdata, flags=None, channel_range=None, output=None):
    fig = plt.figure(figsize=(8.3, 11.7))
    data = np.log10(np.squeeze(np.abs(visdata[:, :])))
    if channel_range is None:
        channel_range = [0, visdata.shape[1]]
    kwargs = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'none',
              'extent': (channel_range[0], channel_range[1], -0.5, data.shape[0] - 0.5)}
    image = plt.imshow(data, **kwargs)
    image.set_cmap('Greys')
    # Make an array of RGBA data for the flags (initialize to alpha=0)
    if flags is not None:
        plotflags = np.zeros(flags.shape[0:2]+(4,))
        plotflags[:, :, 0] = 1.0
        plotflags[:, :, 3] = flags[:, :]
        plt.imshow(plotflags, alpha=0.5, **kwargs)
    else:
        flags = np.zeros_like(data, dtype=np.bool)
    ampsort = np.sort(data[~flags], axis=None)
    arrayremove = int(len(ampsort)*(1.0 - 0.80)/2.0)
    lowcut, highcut = ampsort[arrayremove], ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time')
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


def or_flags_pols(flags, corr_prods, ants):
    """
    OR the flags across polarisation for a given baseline
    """
    antnames = [ant.name for ant in ants]
    corrprod_baselines = [[prod[0][:4],prod[1][:4]] for prod in corr_prods]
    # Get indices in corr_prods of all pols for a baseline
    all_baselines = [list(pair) for pair in itertools.combinations_with_replacement(antnames, 2)]
    for bl in all_baselines:
        bl_indices =  [i[0] for i in enumerate(corrprod_baselines) if i[1] == bl]
        bl_flags = flags[:, :, bl_indices]
        or_flags = np.any(bl_flags, axis=2)
        flags[:, :, bl_indices] = or_flags[:, :, np.newaxis]
    return flags


def get_baseline_mask(corr_prods, ants, limit):
    """
    Compute a mask of the same length as corr_products that indicates
    whether the baseline length of the given correlation product is
    shorter than limit in meters
    """
    baseline_mask = np.zeros(corr_prods.shape[0], dtype=np.bool)
    antlookup = {}
    for ant in ants:
        antlookup[ant.name] = ant
    for prod, baseline in enumerate(corr_prods):
        bl_vector = antlookup[baseline[0][:4]].baseline_toward(antlookup[baseline[1][:4]])
        bl_length = np.linalg.norm(bl_vector)
        if bl_length < limit:
            baseline_mask[prod] = True
    return baseline_mask


def load_data(data_list, slices):
    """
    Load datasets in data_list using :meth:`get`
    if it is available, otherwise just read from the datasets
    one at a time.
    """
    if isinstance(data_list[0], DaskLazyIndexer):
        return data_list[0].get(data_list, slices)
    else:
        return [data[slices] for data in data_list]


def generate_flag_table(input_file, output_root='.', static_flags=None,
                        freq_chans=None, use_file_flags=True, outlier_nsigma=4.5,
                        width_freq=1.5, width_time=100.0, time_extend=3, freq_extend=3,
                        max_scan=600, write_into_input=False, average_freq=1, mask_non_tracks=False,
                        tracks_only=False, mask_limit=1000., or_pols=False, **kwargs):

    """
    Flag the visibility data in the mvf file ignoring the channels specified in
    static_flags, and the channels already flagged if use_file_flags=True.
    This will write a list of flags per scan to the output h5 file or overwrite
    the flag table in the input file if write_into_input=True

    Inputs
    ======
    input_file - name of mvf file to process
    output_root - disk location to write output
    static_flags - input pickle of static mask flags
    freq_chans - range of channels to process (default is central 90% of the band)
    use_file_flags - True if flags already in the file should be used as an input mask for the flagger
    outlier_nsigma - number of sigma for flag threshold
    width_freq - smoothing width in frequency axis (MHz) for background fitting
    width_time - smoothing width in time axis (seconds) for background fitting
    time_extend - size in dumps by which to extend flags after detection
    freq_extend - size in channels by which to extend flags after detection
    max_scan - largest scan length to process (longer scans will be split into chunks in time)
    write_into_input - make a copy of the input file to 'output_root' and insert flags there (v3 only)
    average_freq - average width in channels before flagging (detected flags are extend to full width)
    mask_non_tracks - mask any antennas that are not tracking (added to 'cam_flags' bit)
    tracks_only - only flag tracks (not slews or stops etc.)
    mask_limit - the maximum baseline length in meters to apply the mask
    or_pols - OR the flags across polarisations (HH,VV,HV,HV)
    """

    import logging
    logging.getLogger().setLevel(logging.CRITICAL)

    start_time = time.time()
    mvf = katdal.open(input_file)
    # Only support version 3.x/4.x
    if mvf.version[0] not in ['3', '4']:
        raise Exception("Only mvf version 3.x and 4.x files are supported")
    if write_into_input:
        if mvf.version[0] != '3':
            raise Exception("--write-input will only work for mvf v3 files")
        output_file = os.path.join(output_root, input_file.split('/')[-1])
        if not os.path.exists(output_file) or not os.path.samefile(input_file, output_file):
            print("Copying input file from %s to %s" % (input_file, os.path.abspath(output_root),))
            shutil.copy(input_file, output_root)
        mvf = katdal.open(os.path.join(output_file), mode='r+')
        outfile = mvf.file
        flags_dataset = mvf._flags
    else:
        if mvf.version[0] == '3':
            in_flags_dataset = da.from_array(mvf._flags, chunks=(1, 1024, mvf.shape[2]))
        elif mvf.version[0] == '4':
            in_flags_dataset = mvf.source.data.flags
        basename = os.path.join(output_root, os.path.splitext(os.path.basename(input_file))[0]+'_flags')
        corr_products = mvf.corr_products
        if corr_products.dtype.kind == 'U':
            # HDF5 can't store fixed-length Unicode, so encode as ASCII
            corr_products = np.core.defchararray.encode(mvf.corr_products, 'ascii', 'strict')
        da.to_hdf5(basename + '.h5', {'/corr_products': da.from_array(corr_products, 1), '/flags': in_flags_dataset})
        # Use the local copy of the flags to avoid reading over the network again
        outfile = h5py.File(basename + '.h5', mode='r+')
        flags_dataset = outfile['flags']
        if mvf.version[0] == '4':
            mvf.source.data.flags = da.from_array(flags_dataset, chunks=mvf.source.data.flags.chunksize)
        elif mvf.version[0] == '3':
            mvf._flags = flags_dataset

    # Read static flags from pickle
    if static_flags:
        sff = open(static_flags)
        static_flags = pickle.load(sff)
        sff.close()
    else:
        # Create dummy static flag array if no static flags are specified.
        static_flags = np.zeros(mvf.shape[1], dtype=np.bool)

    # Work out which baselines to use the mask
    bl_mask = get_baseline_mask(mvf.corr_products, mvf.ants, mask_limit)

    # Set up the mask for broadcasting
    mask_array = static_flags[np.newaxis, :, np.newaxis]

    # Convert spike width from frequency and time to channel and dump for the flagger.
    width_freq_channel = width_freq*1.e6/mvf.channel_width
    width_time_dumps = width_time/mvf.dump_period

    cut_chans = (mvf.shape[1]//20, mvf.shape[1]-mvf.shape[1]//20,) if freq_chans is None \
        else (int(freq_chans.split(',')[0]), int(freq_chans.split(',')[1]),)
    freq_range = slice(cut_chans[0], cut_chans[1])

    flagger = SumThresholdFlagger(outlier_nsigma=outlier_nsigma, freq_chunks=7,
                                  spike_width_freq=width_freq_channel, spike_width_time=width_time_dumps,
                                  time_extend=time_extend, freq_extend=freq_extend, average_freq=average_freq)

    for scan, state, target in mvf.scans():
        # We only want the abs of vis
        if int(mvf.version[0]) >= 4:
            mvf.vis.add_transform(da.absolute)
        if tracks_only and state != 'track':
            continue
        # Take slices through scan if it is too large for memory
        if mvf.shape[0] > max_scan:
            scan_slices = [slice(i, i+max_scan, 1) for i in range(0, mvf.shape[0], max_scan)]
            scan_slices[-1] = slice(scan_slices[-1].start, mvf.shape[0], 1)
        else:
            scan_slices = [slice(0, mvf.shape[0])]
        # loop over slices
        for this_slice in scan_slices:
            if use_file_flags:
                this_data, flags = load_data([mvf.vis, mvf.flags,], np.s_[this_slice, freq_range, :])
            else:
                this_data = load_data([mvf.vis], np.s_[this_slice, freq_range, :])[0]
                flags = np.zeros_like(this_data, dtype=np.bool)
            # OR the mask flags with the flags already in the mvf file
            flags[:, :, bl_mask] |= mask_array[:, freq_range, :]
            with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as pool:
                detected_flags = flagger.get_flags(this_data, flags, pool)
            if or_pols:
                detected_flags = or_flags_pols(detected_flags, mvf.corr_products, mvf.ants)
            print("Scan: %4d, Target: %15s, Dumps: %3d, Flagged %5.1f%%" % \
                  (scan, target.name, mvf.shape[0], (np.sum(detected_flags)*100.)/detected_flags.size,))
            # Add new flags to flag table
            flags = np.zeros((this_slice.stop-this_slice.start, mvf.shape[1], mvf.shape[2],), dtype=np.uint8)
            # Add mask to 'static' flags
            flags[:, :, bl_mask] |= mask_array.astype(np.uint8)*(2**FLAG_NAMES.index('static'))
            # Flag non-tracks and add to 'cam' flags
            if mask_non_tracks:
                # Set up mask for cam flags
                cam_mask = np.zeros((this_slice.stop-this_slice.start, mvf.shape[2],), dtype=np.bool)
                for ant in mvf.ants:
                    ant_corr_prods = [index for index, corr_prod in enumerate(mvf.corr_products)
                                      if ant.name in str(corr_prod)]
                    if mvf.version[0] == '3':
                        ant_activity = mvf.sensor['Antennas/%s/activity' % ant.name][this_slice]
                    elif mvf.version[0] == '4':
                        ant_activity = mvf.sensor['%s_activity' % ant.name][this_slice]
                    non_track_dumps = np.nonzero(ant_activity != 'track')[0]
                    cam_mask[non_track_dumps[:, np.newaxis], ant_corr_prods] = True
                flags |= cam_mask[:, np.newaxis, :].astype(np.uint8)*(2**FLAG_NAMES.index('cam'))
            # Add detected flags to 'cal_rfi'
            flags[:, freq_range, :] |= detected_flags.astype(np.uint8)*(2**FLAG_NAMES.index('cal_rfi'))
            flags_dataset[mvf.dumps[this_slice], :, :] |= flags
    outfile.close()
    print("Flagging processing time: %4.1f minutes." % ((time.time() - start_time) / 60.0))
    return


def generate_rfi_report(input_file, input_flags=None, flags_to_show='all', output_root='.', tracks_only=False,
                        antennas=None, targets=None, freq_chans=None, do_cross=True, **kwargs):
    """
    Create an RFI report- store flagged spectrum and number of flags in an output h5 file
    and produce a pdf report.

    Inputs
    ======
    input_file - input mvf filename
    input_flags - input h5 flags (in '/flags' dataset); will use these flags rather than those in mvf file.
    flags_to_show - select which flag bits to plot. ('all'=all flags)
    output_root - directory where output is to be placed - defailt cwd
    antenna - which antenna to produce report on - default all in file
    targets - which target to produce report on - default None
    freq_chans - which frequency channels to work on format - <start_chan>,<end_chan> default - 90% of bandpass
    do_cross - plot the cross correlations with the autos
    """

    mvf = katdal.open(input_file)
    # Get the selected antenna or default to first file antenna
    ants = antennas.split(',') if antennas else [ant.name for ant in mvf.ants]
    # Frequency range
    num_channels = len(mvf.channels)
    if input_flags is not None:
        input_flags = h5py.File(input_flags)
        if mvf.version[0] == "3":
            mvf._flags = input_flags['flags']
        elif mvf.version[0] == "4":
            mvf.source.data.flags = da.from_array(input_flags['flags'], chunks=(1, 1024, mvf.shape[2],))
    if freq_chans is None:
        # Default is drop first and last 5% of the bandpass
        start_chan = num_channels//20
        end_chan = num_channels - start_chan
    else:
        start_chan = int(freq_chans.split(',')[0])
        end_chan = int(freq_chans.split(',')[1])
    chan_range = list(range(start_chan, end_chan+1))

    if targets is 'all':
        targets = mvf.catalogue.targets
    if targets is None:
        targets = []

    # Report cross correlations if requested
    if do_cross:
        all_blines = [list(pair) for pair in itertools.combinations_with_replacement(ants, 2)]
    else:
        all_blines = [[a, a] for a in ants]
    for bline in all_blines:
        # Set up the output file
        basename = os.path.join(output_root, os.path.splitext(
                                input_file.split('/')[-1])[0] + '_' + ','.join(bline) + '_RFI')
        pdf = PdfPages(basename+'.pdf')
        corrprodselect = [[bline[0] + 'h', bline[1] + 'h'], [bline[0] + 'v', bline[1] + 'v']]
        mvf.select(reset='TFB', corrprods=corrprodselect, flags=flags_to_show)
        mvf.vis.add_transform(np.abs)
        vis, flags = load_data([mvf.vis, mvf.flags], np.s_[:, :, :])
        if tracks_only:
            mvf.select(scans='track')
        # Populate data_dict
        data_dict = get_flag_stats(mvf, thisdata=vis, flags=flags)
        # Output to h5 file
        outfile = h5py.File(basename + '.h5', 'w')
        for targetname, targetdata in six.iteritems(data_dict):
            # Create a group in the h5 file corresponding to the target
            grp = outfile.create_group(targetname)
            # populate the group with the data
            for datasetname, data in six.iteritems(targetdata):
                # h5py doesn't support numpy Unicode arrays, so convert to ASCII.
                # (http://docs.h5py.org/en/stable/strings.html)
                if isinstance(data, np.ndarray) and data.dtype.kind == 'U':
                    data = np.core.defchararray.encode(data, 'ascii', 'strict')
                grp.create_dataset(datasetname, data=data)
        outfile.close()

        # Loop through targets
        for target in targets:
            # Get the target name if it is a target object
            if isinstance(target, katpoint.Target):
                target = target.name
            # Extract target from file
            mvf.select(reset='TFB', targets=target, scans='track', corrprods=corrprodselect, channels=chan_range)
            if mvf.shape[0] == 0:
                print('No data to process for ' + target)
                continue
            # Get HH and VV cross pol indices
            hh_index = np.all(np.char.endswith(mvf.corr_products, 'h'), axis=1)
            vv_index = np.all(np.char.endswith(mvf.corr_products, 'v'), axis=1)
            label = 'Flag info for Target: ' + target + ', Baseline: ' + ','.join(bline) + \
                    ', ' + str(data_dict[target]['numrecords_tot'])+' records'
            plot_flag_data(label + ' H Pol', data_dict[target]['spectrum'][chan_range, hh_index],
                           data_dict[target]['flagfrac'][chan_range, hh_index], mvf.channel_freqs, pdf)
            plot_flag_data(label + ' V Pol', data_dict[target]['spectrum'][chan_range, vv_index],
                           data_dict[target]['flagfrac'][chan_range, vv_index], mvf.channel_freqs, pdf)
            plot_waterfall_subsample(vis[mvf.dumps[:, np.newaxis], mvf.channels, hh_index],
                                     flags[mvf.dumps[:, np.newaxis], mvf.channels, hh_index],
                                     mvf.channel_freqs, None, label + '\nHH polarisation', output=pdf)
            plot_waterfall_subsample(vis[mvf.dumps[:, np.newaxis], mvf.channels, vv_index],
                                     flags[mvf.dumps[:, np.newaxis], mvf.channels, vv_index],
                                     mvf.channel_freqs, None, label + '\nVV polarisation', output=pdf)
        # Reset the selection
        mvf.select(reset='TFB', corrprods=corrprodselect, channels=chan_range, scans='track')

        # Plot the flags for all data in the file
        hh_index = np.all(np.char.endswith(mvf.corr_products, 'h'), axis=1)
        vv_index = np.all(np.char.endswith(mvf.corr_products, 'v'), axis=1)
        label = 'Flag info for all data, Baseline: ' + ','.join(bline) + \
                ', ' + str(data_dict['all_data']['numrecords_tot']) + ' records'
        plot_flag_data(label + ' H Pol', data_dict['all_data']['spectrum'][chan_range, hh_index],
                       data_dict['all_data']['flagfrac'][chan_range, hh_index], mvf.channel_freqs, pdf)
        plot_flag_data(label + ' V Pol', data_dict['all_data']['spectrum'][chan_range, vv_index],
                       data_dict['all_data']['flagfrac'][chan_range, vv_index], mvf.channel_freqs, pdf)
        plot_waterfall_subsample(vis[mvf.dumps[:, np.newaxis], mvf.channels, hh_index],
                                 flags[mvf.dumps[:, np.newaxis], mvf.channels, hh_index],
                                 mvf.channel_freqs, mvf.timestamps, label + '\nHH polarisation', output=pdf)
        plot_waterfall_subsample(vis[mvf.dumps[:, np.newaxis], mvf.channels, vv_index],
                                 flags[mvf.dumps[:, np.newaxis], mvf.channels, vv_index],
                                 mvf.channel_freqs, mvf.timestamps, label + '\nVV polarisation', output=pdf)
        # close the plot
        pdf.close()
