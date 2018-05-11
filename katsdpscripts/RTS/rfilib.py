#Library to contain RFI flagging routines and other RFI related functions
import os
import shutil
import time
import itertools
import multiprocessing

import katdal
import katpoint

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt #; plt.ioff()
from matplotlib import ticker
import numpy as np
import dask.array as da
import pickle
import h5py
import concurrent.futures

from katsdpsigproc.rfi.twodflag import SumThresholdFlagger

def plot_RFI_mask(pltobj,main=True,extra=None,channelwidth=1e6):
    if main:
        pltobj.axvspan(1674e6,1677e6, alpha=0.3, color='grey')#Meteosat
        pltobj.axvspan(1667e6,1667e6, alpha=0.3, color='grey')#Fengun
        pltobj.axvspan(1682e6,1682e6, alpha=0.3, color='grey')#Meteosat
        pltobj.axvspan(1685e6,1687e6, alpha=0.3, color='grey')#Meteosat
        pltobj.axvspan(1687e6,1687e6, alpha=0.3, color='grey')#Fengun
        pltobj.axvspan(1690e6,1690e6, alpha=0.3, color='grey')#Meteosat
        pltobj.axvspan(1699e6,1699e6, alpha=0.3, color='grey')#Meteosat
        pltobj.axvspan(1702e6,1702e6, alpha=0.3, color='grey')#Fengyun
        pltobj.axvspan(1705e6,1706e6, alpha=0.3, color='grey')#Meteosat
        pltobj.axvspan(1709e6,1709e6, alpha=0.3, color='grey')#Fengun
        pltobj.axvspan(1501e6,1570e6, alpha=0.3, color='blue')#Inmarsat
        pltobj.axvspan(1496e6,1585e6, alpha=0.3, color='blue')#Inmarsat
        pltobj.axvspan(1574e6,1576e6, alpha=0.3, color='blue')#Inmarsat
        pltobj.axvspan(1509e6,1572e6, alpha=0.3, color='blue')#Inmarsat
        pltobj.axvspan(1574e6,1575e6, alpha=0.3, color='blue')#Inmarsat
        pltobj.axvspan(1512e6,1570e6, alpha=0.3, color='blue')#Thuraya
        pltobj.axvspan(1450e6,1498e6, alpha=0.3, color='red')#Afristar
        pltobj.axvspan(1652e6,1694e6, alpha=0.2, color='red')#Afristar
        pltobj.axvspan(1542e6,1543e6, alpha=0.3, color='cyan')#Express AM1
        pltobj.axvspan(1554e6,1554e6, alpha=0.3, color='cyan')#Express AM 44
        pltobj.axvspan(1190e6,1215e6, alpha=0.3, color='green')#Galileo
        pltobj.axvspan(1260e6,1300e6, alpha=0.3, color='green')#Galileo
        pltobj.axvspan(1559e6,1591e6, alpha=0.3, color='green')#Galileo
        pltobj.axvspan(1544e6,1545e6, alpha=0.3, color='green')#Galileo
        pltobj.axvspan(1190e6,1217e6, alpha=0.3, color='green')#Beidou
        pltobj.axvspan(1258e6,1278e6, alpha=0.3, color='green')#Beidou
        pltobj.axvspan(1559e6,1563e6, alpha=0.3, color='green')#Beidou  
        pltobj.axvspan(1555e6,1596e6, alpha=0.3, color='green')#GPS L1  1555 -> 1596 
        pltobj.axvspan(1207e6,1238e6, alpha=0.3, color='green')#GPS L2  1207 -> 1248 
        pltobj.axvspan(1378e6,1384e6, alpha=0.3, color='green')#GPS L3  
        pltobj.axvspan(1588e6,1615e6, alpha=0.3, color='green')#GLONASS  1588 -> 1615 L1
        pltobj.axvspan(1232e6,1259e6, alpha=0.3, color='green')#GLONASS  1232 -> 1259 L2
        pltobj.axvspan(1616e6,1630e6, alpha=0.3, color='grey')#IRIDIUM
    if extra is not None: 
        for i in xrange(extra.shape[0]):
            pltobj.axvspan(extra[i]-channelwidth/2,extra[i]+channelwidth/2, alpha=0.1, color='Maroon')

def rolling_window(a, window,axis=-1,pad=False,mode='reflect',**kargs):
    """
     This function produces a rolling window shaped data with the rolled data in the last col
        a      :  n-D array of data  
        window : integer is the window size
        axis   : integer, axis to move the window over
                 default is the last axis.
        pad    : {Boolean} Pad the array to the origanal size
        mode : {str, function} from the function numpy.pad
        One of the following string values or a user supplied function.
        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
                        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
                        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
                      con  vector along each axis.
        'median'        Pads with the median value of all or part of the
                        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
                        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
                        the first and last values of the vector along each
                        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
                        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
                        The first values are used to pad the end and the
                        end values are used to pad the beginning.
        <function>      of the form padding_func(vector, iaxis_pad_width, iaxis, **kwargs)
                        see numpy.pad notes
        **kargs are passed to the function numpy.pad
        
    Returns:
        an array with shape = np.array(a.shape+(window,))
        and the rolled data on the last axis
        
    Example:
        import numpy as np
        data = np.random.normal(loc=1,scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1, size=10000)
        stddata = rolling_window(data, 400).std(axis=-1)
    """
    if axis == -1 : axis = len(a.shape)-1 
    if pad :
        pad_width = []
        for i in xrange(len(a.shape)):
            if i == axis: 
                pad_width += [(window//2,window//2 -1 +np.mod(window,2))]
            else :  
                pad_width += [(0,0)] 
        a = np.pad(a,pad_width=pad_width,mode=mode,**kargs)
    a1 = np.swapaxes(a,axis,-1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1],)
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2,axis) # Move original axis to 

##############################
# End of RFI detection routines
##############################
def get_flag_stats(h5, thisdata=None, flags=None, flags_to_show=None, norm_spec=None):
    """
    Given a katdal object, remove a dc offset for each record
    (ignoring severe spikes) then obtain an average spectrum of
    all of the scans in the data.
    Return the average spectrum with dc offset removed and the number of times
    each channel is flagged (flags come optionally from 'flags' else from the 
    flags in the input katdal object). Optinally provide a 
    spectrum (norm_spec) to divide into the calculated bandpass.
    """
    targets=h5.catalogue.targets
    flag_stats = {}
    if flags is None:
        flags = np.empty(h5.shape,dtype=np.bool)
        h5.select(flags=flags_to_show)
        for dump in range(h5.shape[0]):
            flags[dump] = h5.flags[dump]
    #Squeeze here removes stray axes left over by LazyIndexer
    if thisdata is None:
        thisdata = np.empty(h5.shape,dtype=np.float32)
        for dump in range(h5.shape[0]):
            thisdata[dump] = np.abs(h5.vis[dump])
    if norm_spec is not None: thisdata /= norm_spec[np.newaxis,:]
    #Get DC height (median rather than mean is more robust...)
    data = np.ma.MaskedArray(thisdata,mask=flags,copy=False).filled(fill_value=np.nan)
    offset = np.nanmedian(data,axis=1)
    #Remove the DC height
    weights = np.logical_not(flags).astype(np.int8)
    data /= np.expand_dims(offset,axis=1)
    #Get the results for all of the data
    weightsum = weights.sum(axis=0,dtype=np.int)
    averagespec = np.nanmean(data,axis=0)
    flagfrac = 1. - weightsum/h5.shape[0].astype(np.float)
    flag_stats['all_data'] = {'spectrum': averagespec, 'numrecords_tot': h5.shape[0], 'flagfrac': flagfrac, 'channel_freqs': h5.channel_freqs, \
                                'dump_period': h5.dump_period, 'corr_products': h5.corr_products}
    #And for each target
    for t in targets:
        h5.select(targets=t,scans='~slew')
        weightsum = (weights[h5.dumps]).sum(axis=0,dtype=np.int).squeeze()
        averagespec =  np.nanmean(data[h5.dumps],axis=0) #/weightsum
        flagfrac = 1. - weightsum/h5.shape[0].astype(np.float)
        flag_stats[t.name] = {'spectrum': averagespec, 'numrecords_tot': h5.shape[0], 'flagfrac': flagfrac, 'channel_freqs': h5.channel_freqs, \
                                'dump_period': h5.dump_period, 'corr_products': h5.corr_products}
    h5.select(reset='T')
    return flag_stats

def plot_flag_data(label,spectrum,flagfrac,freqs,pdf,mask=None):
    """
    Produce a plot of the average spectrum in H and V 
    after flagging and attach it to the pdf output.
    Also show fraction of times flagged per channel.
    """
    from katsdpscripts import git_info

    repo_info = git_info() 

    #Set up the figure
    fig = plt.figure(figsize=(11.7,8.3))

    #Plot the spectrum
    ax1 = fig.add_subplot(211)
    ax1.text(0.01, 0.90,repo_info, horizontalalignment='left',fontsize=10,transform=ax1.transAxes)
    ax1.set_title(label)
    plt.plot(freqs,spectrum,linewidth=.5)

    #plot_RFI_mask(ax1)
    ticklabels=ax1.get_xticklabels()
    plt.setp(ticklabels,visible=False)
    ticklabels=ax1.get_yticklabels()
    plt.setp(ticklabels,visible=False)
    plt.xlim((min(freqs),max(freqs)))
    plt.ylabel('Mean amplitude\n(arbitrary units)')
    #Plot the mask
    #plot_RFI_mask(ax1,mask,freqs[1]-freqs[0])
    #Plot the flags occupancies
    ax = fig.add_subplot(212,sharex=ax1)
    plt.plot(freqs,flagfrac,'r-',linewidth=.5)
    plt.ylim((0.,1.))
    plt.axhline(0.8,color='red',linestyle='dashed',linewidth=.5)
    #plot_RFI_mask(ax,mask,freqs[1]-freqs[0])
    plt.xlim((min(freqs),max(freqs)))
    minorLocator = ticker.MultipleLocator(10e6)
    plt.ylabel('Fraction flagged')
    ticklabels=ax.get_yticklabels()
    #Convert ticks to MHZ
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

    fig = plt.figure(figsize=(8.3,11.7))
    ax = plt.subplot(111)
    ax.set_title(label)
    ax.text(0.01, 0.02,repo_info, horizontalalignment='left',fontsize=10,transform=ax.transAxes)
    display_limits = ax.get_window_extent()
    if freqs is None: freqs=range(0,visdata.shape[1])
    #300dpi, and one pixel per desired data-point
    #in pixels at 300dpi
    display_width = display_limits.width * resolution/72.
    display_height = display_limits.height * resolution/72.
    x_step = max(int(visdata.shape[1]/display_width), 1)
    y_step = max(int(visdata.shape[0]/display_height), 1)
    x_slice = slice(0, -1, x_step)
    y_slice = slice(0, -1, y_step)
    data = np.log10(np.abs(visdata[y_slice,x_slice]))
    flags = flagdata[y_slice,x_slice]
    plotflags = np.zeros(flags.shape[0:2]+(4,))
    plotflags[:,:,0] = 1.0
    plotflags[:,:,3] = flags
    if times is None:
        starttime = 0
        endtime = visdata.shape[0]
    else:
        starttime = mdates.date2num(dt.fromtimestamp(times[0]))
        endtime = mdates.date2num(dt.fromtimestamp(times[-1]))
    kwargs = {'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none', 'extent' : (freqs[0],freqs[-1], starttime, endtime)}
    image = ax.imshow(data,**kwargs)
    image.set_cmap('Greys')
    ax.imshow(plotflags,alpha=0.5,**kwargs)
    ampsort = np.sort(data[(data>0.0) | (~flags)], axis=None)
    arrayremove = int(len(ampsort)*(1.0 - 0.80)/2.0)
    lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    plt.xlim((min(freqs),max(freqs)))
    if times is not None:
        ax.yaxis_date()
        plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.ylabel('Time (SAST)')
    else:
        plt.ylabel('Time (Dumps)')
    #Convert ticks to MHZ
    ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
    ax.xaxis.set_major_formatter(ticks)
    plt.xlabel('Frequency (MHz)')
    if output:
        output.savefig(fig)
    else:
        plt.show()
    plt.close('all')


def plot_waterfall(visdata,flags=None,channel_range=None,output=None):
    fig = plt.figure(figsize=(8.3,11.7))
    data=np.log10(np.squeeze(np.abs(visdata[:,:])))
    if channel_range is None:
        channel_range=[0,visdata.shape[1]]
    kwargs={'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none', 'extent' : (channel_range[0],channel_range[1], -0.5, data.shape[0] - 0.5)}
    image=plt.imshow(data,**kwargs)
    image.set_cmap('Greys')
    #flagimage=plt.imshow(flags[:,:,0],**kwargs)
    #Make an array of RGBA data for the flags (initialize to alpha=0)
    if flags is not None:
        plotflags = np.zeros(flags.shape[0:2]+(4,))
        plotflags[:,:,0] = 1.0
        plotflags[:,:,3] = flags[:,:]
        plt.imshow(plotflags,alpha=0.5,**kwargs)
    else: 
        flags=np.zeros_like(data,dtype=np.bool)
    ampsort=np.sort(data[(data>0.0) | (~flags)],axis=None)
    arrayremove = int(len(ampsort)*(1.0 - 0.80)/2.0)
    lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time')
    if output==None:
        plt.show()
    else:
        plt.savefig(output)

def generate_flag_table(input_file, output_root='.', static_flags=None, 
                        freq_chans=None, use_file_flags=True, outlier_nsigma=4.5, 
                        width_freq=1.5, width_time=100.0, time_extend=3, freq_extend=3,
                        max_scan=260, write_into_input=False, speedup=1, mask_non_tracks=False, 
                        drop_beg=4, tracks_only=False):
    """
    Flag the visibility data in the h5 file ignoring the channels specified in 
    static_flags, and the channels already flagged if use_file_flags=True.

    This will write a list of flags per scan to the output h5 file or overwrite 
    the flag table in the input file if write_into_input=True
    """
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)

    start_time = time.time()
    h5 = katdal.open(input_file)
    if write_into_input:
        if h5.version[0] != '3':
            raise Exception("--write-input will only work for mvf v3 files")
        output_file = os.path.join(output_root, input_file.split('/')[-1])
        if not os.path.exists(output_file) or not os.path.samefile(input_file,output_file):
            print "Copying input file from %s to %s"%(input_file, os.path.abspath(output_root),)
            shutil.copy(input_file, output_root)
        h5 = katdal.open(os.path.join(output_file), mode = 'r+')
        outfile = h5.file
        flags_dataset = h5._flags
    else:
        if h5.version[0] == '3':
            in_flags_dataset = da.from_array(h5._flags, chunks=(1, h5.shape[1]//4, h5.shape[2]))
        elif h5.version[0] == '4':
            in_flags_dataset = h5.source.data.flags
        else:
            raise Exception("Only mvf version 3.x and 4.x files are supported")
        basename = os.path.join(output_root,os.path.splitext(os.path.basename(input_file))[0]+'_flags')
        #"Quack" first rows
        beg_elements = da.zeros((drop_beg, h5.shape[1], h5.shape[2],), chunks=(1, h5.shape[1]//4, h5.shape[2]), dtype=np.uint8)
        flags_dataset = da.concatenate([beg_elements, in_flags_dataset[drop_beg:]])
        da.to_hdf5(basename + '.h5', {'/corr_products': da.from_array(h5.corr_products, 1), '/flags': flags_dataset})
        #Use the local copy of the flags to avoid reading over the network again
        outfile = h5py.File(basename + '.h5', mode='r+')
        flags_dataset = outfile['flags']
        if h5.version[0] == '4': 
            h5.source.data.flags = da.from_array(flags_dataset, chunks=(1, h5.shape[1]//4, h5.shape[2]))
        elif h5.version[0] == '3':
            h5._flags = flags_dataset

    freq_length = h5.shape[1]
    
    #Read static flags from pickle
    if static_flags:
        sff = open(static_flags)
        static_flags = pickle.load(sff)
        sff.close()
    else:
        #Create dummy static flag array if no static flags are specified. 
        static_flags=np.zeros(h5.shape[1],dtype=np.bool)
    
    #Set up the mask for broadcasting
    mask_array = static_flags[np.newaxis,:,np.newaxis]

    #Speed up flagging by averaging further if requested.
    average_freq = speedup

    #Convert spike width from frequency and time to channel and dump for the flagger.
    width_freq_channel = width_freq*1.e6/h5.channel_width
    width_time_dumps = width_time/h5.dump_period

    cut_chans = (h5.shape[1]//20,h5.shape[1]-h5.shape[1]//20,) if freq_chans is None \
                        else (int(freq_chans.split(',')[0]),int(freq_chans.split(',')[1]),)
    freq_range = slice(cut_chans[0], cut_chans[1])

    flagger = SumThresholdFlagger(outlier_nsigma=outlier_nsigma, freq_chunks=7,
                                  spike_width_freq=width_freq_channel,spike_width_time=width_time_dumps,
                                  time_extend=time_extend, freq_extend=freq_extend, average_freq=average_freq)

    for scan, state, target in h5.scans():
        if tracks_only and state!='track': continue
        #Take slices through scan if it is too large for memory
        if h5.shape[0]>max_scan:
            scan_slices = [slice(i,i+max_scan,1) for i in range(0,h5.shape[0],max_scan)]
            scan_slices[-1] = slice(scan_slices[-1].start,h5.shape[0],1)
        else:
            scan_slices = [slice(0,h5.shape[0])]
        #loop over slices
        for this_slice in scan_slices:
            #Don't read all of the data in one hit- loop over timestamps instead
            this_data = np.empty((this_slice.stop-this_slice.start,freq_range.stop-freq_range.start,h5.shape[2],),dtype=np.float32)
            flags = np.zeros((this_slice.stop-this_slice.start,freq_range.stop-freq_range.start,h5.shape[2],),dtype=np.bool)
            for index,dump in enumerate(range(*this_slice.indices(h5.shape[0]))):
                this_data[index]=np.abs(h5.vis[dump,freq_range])
                if use_file_flags:
                    flags[index] = h5.flags[dump,freq_range]
            #OR the mask flags with the flags already in the h5 file
            flags = np.logical_or(flags,mask_array[:,freq_range,:])
            with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as pool:
                detected_flags = flagger.get_flags(this_data,flags,pool)
            print "Scan: %4d, Target: %15s, Dumps: %3d, Flagged %5.1f%%"% \
                        (scan,target.name,h5.shape[0],(np.sum(detected_flags)*100.)/detected_flags.size,)
            #Flags are 8 bit:
            #1: 'reserved0' = 0
            #2: 'static' = 1
            #3: 'cam' = 2
            #4: 'reserved3' = 3
            #5: 'ingest_rfi' = 4
            #6: 'predicted_rfi' = 5
            #7: 'cal_rfi' = 6
            #8: 'reserved7' = 7
            #Add new flags to flag table
            flags = np.zeros((this_slice.stop-this_slice.start,h5.shape[1],h5.shape[2],),dtype=np.uint8)
            #Add mask to 'static' flags
            flags |= mask_array.astype(np.uint8)*2
            #Flag non-tracks and add to 'cam' flags
            if mask_non_tracks:
                #Set up mask for cam flags (assumtion here is that these are unused up to now)
                cam_mask = np.zeros((this_slice.stop-this_slice.start,h5.shape[2],),dtype=np.bool)
                for ant in h5.ants:
                    ant_corr_prods = [index for index,corr_prod in enumerate(h5.corr_products) if ant.name in str(corr_prod)]
                    non_track_dumps = np.nonzero(h5.sensor['Antennas/%s/activity'%ant.name][this_slice] != 'track' )[0]
                    cam_mask[non_track_dumps[:,np.newaxis],ant_corr_prods] = True
                flags |= cam_mask[:,np.newaxis,:].astype(np.uint8)*(2**2)
            #Add detected flags to 'cal_rfi'
            flags[:,freq_range,:] |= detected_flags.astype(np.uint8)*(2**6)
            flags_dataset[h5.dumps[this_slice],:,:] += flags
    outfile.close()
    print "Flagging processing time: %4.1f minutes."%((time.time() - start_time)/60.0)
    return

def generate_rfi_report(input_file,input_flags=None,flags_to_show='all',output_root='.',antenna=None,targets=None,freq_chans=None,do_cross=True):
    """
    Create an RFI report- store flagged spectrum and number of flags in an output h5 file
    and produce a pdf report.

    Inputs
    ======
    input_file - input h5 filename
    input_flags - input h5 flags; will overwrite flags in h5 file- h5 file in format returnd from generate_flag_table
    flags_to_show - select which flag bits to plot. ('all'=all flags)
    output_root - directory where output is to be placed - defailt cwd
    antenna - which antenna to produce report on - default all in file
    targets - which target to produce report on - default None
    time_range - Time range of input file to report. sequence of 2 (Start_time,End_time)
                Format: 'YYYY-MM-DD HH:MM:SS.SSS', katpoint.Target or ephem.Date object or float in UTC seconds since Unix epoch
    freq_chans - which frequency channels to work on format - <start_chan>,<end_chan> default - 90% of bandpass
    """

    h5 = katdal.open(input_file)
    #Get the selected antenna or default to first file antenna
    ants=antenna.split(',') if antenna else [ant.name for ant in h5.ants]
    #Frequency range
    num_channels = len(h5.channels)
    if input_flags is not None:
        input_flags = h5py.File(input_flags)
        if h5.version == "3":
            h5._flags = input_flags['flags']
        elif h5.version == "4":
            h5.source.data.flags = da.from_array(input_flags['flags'], chunks = (1, h5.shape[1], h5.shape[2],))
    if freq_chans is None:
        # Default is drop first and last 5% of the bandpass
        start_chan = num_channels//20
        end_chan   = num_channels - start_chan
    else:
        start_chan = int(freq_chans.split(',')[0])
        end_chan = int(freq_chans.split(',')[1])
    chan_range = range(start_chan,end_chan+1)

    if targets is 'all': targets = h5.catalogue.targets
    if targets is None: targets = []

    h5.select(scans = 'track')

    #Report cross correlations if requested
    if do_cross:
        all_blines = [list(pair) for pair in itertools.combinations_with_replacement(ants,2)]
    else:
        all_blines = [[a,a] for a in ants]
    for bline in all_blines:
        # Set up the output file
        basename = os.path.join(output_root,os.path.splitext(input_file.split('/')[-1])[0]+'_' + ','.join(bline) + '_RFI')
        pdf = PdfPages(basename+'.pdf')
        corrprodselect=[[bline[0]+'h',bline[1]+'h']]
        h5.select(reset='TFB',corrprods=corrprodselect)
        vis=np.empty(h5.shape,dtype=np.float32)
        flags=np.empty(h5.shape,dtype=np.bool)
        #Get required vis and flags up front to avoid multiple reads of the data
        h5.select(flags=flags_to_show)
        for dump in range(h5.shape[0]):
            vis[dump]=np.abs(h5.vis[dump])
            flags[dump]=h5.flags[dump]
        #Populate data_dict
        data_dict=get_flag_stats(h5,thisdata=vis,flags=flags)
        #Output to h5 file
        outfile=h5py.File(basename+'.h5','w')
        for targetname, targetdata in data_dict.iteritems():
            #Create a group in the h5 file corresponding to the target
            grp=outfile.create_group(targetname)
            #populate the group with the data
            for datasetname, data in targetdata.iteritems(): grp.create_dataset(datasetname,data=data)
        outfile.close()

        # Loop through targets
        for target in targets:
            #Get the target name if it is a target object
            if isinstance(target, katpoint.Target):
                target = target.name
            #Extract target from file
            h5.select(reset='TFB',targets=target,scans='track',corrprods=corrprodselect,channels=chan_range)
            if h5.shape[0]==0:
                print 'No data to process for ' + target
                continue
            #Get HH and VV cross pol indices
            hh_index=np.all(np.char.endswith(h5.corr_products,'h'),axis=1)
            vv_index=np.all(np.char.endswith(h5.corr_products,'h'),axis=1)
            label = 'Flag info for Target: ' + target + ', Baseline: ' + ','.join(bline) +', '+str(data_dict[target]['numrecords_tot'])+' records'
            plot_flag_data(label + ' H Pol', data_dict[target]['spectrum'][chan_range,hh_index], \
                            data_dict[target]['flagfrac'][chan_range,hh_index], h5.channel_freqs, pdf)
            plot_flag_data(label + ' V Pol', data_dict[target]['spectrum'][chan_range,vv_index], \
                            data_dict[target]['flagfrac'][chan_range,vv_index], h5.channel_freqs, pdf)
            plot_waterfall_subsample(vis[h5.dumps[:,np.newaxis],h5.channels,hh_index],flags[h5.dumps[:,np.newaxis],h5.channels,hh_index], \
                                                    h5.channel_freqs,None,label+'\nHH polarisation',output=pdf)
            plot_waterfall_subsample(vis[h5.dumps[:,np.newaxis],h5.channels,vv_index],flags[h5.dumps[:,np.newaxis],h5.channels,vv_index], \
                                                    h5.channel_freqs,None,label+'\nVV polarisation',output=pdf)
        #Reset the selection
        h5.select(reset='TFB', corrprods=corrprodselect, channels=chan_range, scans='track')

        #Plot the flags for all data in the file
        hh_index=np.all(np.char.endswith(h5.corr_products,'h'),axis=1)
        vv_index=np.all(np.char.endswith(h5.corr_products,'h'),axis=1)
        label = 'Flag info for all data, Baseline: ' + ','.join(bline) +', '+str(data_dict['all_data']['numrecords_tot'])+' records'
        plot_flag_data(label + ' H Pol', data_dict['all_data']['spectrum'][chan_range,hh_index], \
                        data_dict['all_data']['flagfrac'][chan_range,hh_index], h5.channel_freqs, pdf)
        plot_flag_data(label + ' V Pol', data_dict['all_data']['spectrum'][chan_range,vv_index], \
                        data_dict['all_data']['flagfrac'][chan_range,vv_index], h5.channel_freqs, pdf)
        plot_waterfall_subsample(vis[h5.dumps[:,np.newaxis],h5.channels,hh_index],flags[h5.dumps[:,np.newaxis],h5.channels,hh_index], \
                                                h5.channel_freqs,h5.timestamps,label+'\nHH polarisation',output=pdf)
        plot_waterfall_subsample(vis[h5.dumps[:,np.newaxis],h5.channels,vv_index],flags[h5.dumps[:,np.newaxis],h5.channels,vv_index], \
                                                h5.channel_freqs,h5.timestamps,label+'\nVV polarisation',output=pdf)  
        #close the plot
        pdf.close()
