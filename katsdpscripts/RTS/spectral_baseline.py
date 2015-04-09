import os

import numpy as np
from numpy.ma import MaskedArray
import scipy.interpolate as interpolate

import katdal
from katdal import averager

import matplotlib.pyplot as plt
from matplotlib import ticker

from katsdpscripts.RTS import rfilib


def read_and_select_file(file, bline=None, target=None, channels=None, polarisation=None, **kwargs):
    """
    Read in the input h5 file and make a selection based on kwargs and average the data.

    file:   {string} filename of h5 file to open in katdal

    Returns:
        the visibility data to plot, the frequency array to plot, the flags to plot
    """

    data = katdal.open(file)
    #Make selection from dictionary
    select_data={}
    #Antenna
    if bline == None:
        ant1,ant2 = data.ants[0].name,data.ants[0].name
    else:
        ants=bline.split(',')
        (ant1,ant2) = (ants[0],ants[1]) if len(ants)>1 else (ants[0],ants[0]) 
    select_data['ants'] = (ant1,ant2)
    if ant1 != ant2: select_data['corrprods']='cross'

    #Target
    if target == None:
        target = data.catalogue.targets[0]
    select_data['targets']=target

    #Polarisation
    if polarisation is 'I':
        #Need HH and VV for I, can get this from corrprods
        select_data['corrprods']=((ant1 + 'v', ant2 + 'v'),(ant1 + 'h', ant2 + 'h'))  
    else:
        select_data['pol']=polarisation

    #Only tracks- no slews
    select_data['scans']='track'

    # Secect desired channel range
    # Select frequency channels and setup defaults if not specified
    num_channels = len(data.channels)
    if channels is None:
        # Default is drop first and last 20% of the bandpass
        start_chan = num_channels // 4
        end_chan   = start_chan * 3
    else:
        start_chan = int(channels.split(',')[0])
        end_chan = int(channels.split(',')[1])
    chan_range = range(start_chan,end_chan+1)
    select_data['channels']=chan_range

    data.select(strict=False, reset='', **select_data)
    #return the selected data

    return data, ant1 + ant2, polarisation


def getbackground_spline(data,spike_width):

    """ From a 1-d data array determine a background iteratively by fitting a spline
    and removing data more than a few sigma from the spline """

    # Remove the first and last element in data from the fit.
    y=data[:]
    arraysize=y.shape[0]
    x=np.arange(arraysize)

    # Iterate 2 times
    for iteration in range(2):

        # First iteration fits a linear spline with 3 knots.
        if iteration==0:
            npieces=3
            deg=1
        # Second iteration fits a cubic spline to every second data point with 10 knots.
        elif iteration==1:
            npieces=int(arraysize/3)
            deg=3

        # Size of each piece of the spline.
        psize = arraysize/npieces
        firstindex = arraysize%psize + int(psize/2)
        indices = np.trim_zeros(np.arange(firstindex,arraysize,psize))
        
        # Fit the spline
        thisfit = interpolate.LSQUnivariateSpline(x,y,indices,k=deg)
        
        thisfitted_data=np.asarray(thisfit(x),y.dtype)

        # Subtract the fitted spline from the data
        residual = y-thisfitted_data
        this_std = np.std(residual)

        # Reject data more than 5sigma from the residual. 
        flags = residual > 5*this_std

        # Set rejected data value to the fitted value + 1sigma.
        y[flags] = thisfitted_data[flags] + this_std

    # Final iteration has knots separated by "spike_width".
    npieces = int(y.shape[0]/spike_width)
    psize = (x[-1]+1)/npieces
    firstindex = int((y.shape[0]%psize))
    indices = np.trim_zeros(np.arange(firstindex,arraysize,psize))

    # Get the final background.
    finalfit = interpolate.LSQUnivariateSpline(x,y,indices,k=3)
    thisfitted_data = np.asarray(finalfit(x),y.dtype)

    return(thisfitted_data)


def extract_and_average(data, timeav=None, freqav=None, stokesI=False):
    """
    Extract the visibility data from data for plotting. Data are averaged in timeav and chanav chunks
    if no timeav or chanav is given then the shortest track is used as the timeav and chanav is set to produce
    100 channels in the bandpass. The katdal data object is assumed to have had a selection down to 1 spectral axis
    applied elsewhere. If stokesI is True then the data is assumed to to have two spectral axes which 
    are used to form stokes I.

    Parameters
    ==========
    data:     A :class:katdal object selected to have 1 baseline axis
    timeav:   The desired time averaging interval in minutes.
    chanav:   The desired frequency averaging interval in MHz
    stokesI:  If True then form stokes I from the (assumed) HH and VV polarisation axes in the visibility data

    Returns
    =======
    vis_data: An [avtime,avfreq,1] array of averaged visibility data for input to plotting routines
    freq_data: An array of averaged frequencies (the x axis for specturm plots)
    flag_data: A boolean array of remaining flags after averaging- these can also be plotted.
    """

    #Get the shortest and longest scans in dumps
    short_scan = -1
    long_scan = -1
    for scan,state,target in data.scans():
        scan_length = data.timestamps.shape[0]
        if short_scan > -1: short_scan = min((scan_length,short_scan))
        else: short_scan = scan_length
        long_scan = max((long_scan,scan_length))
    #Get the number of dumps to average
    if timeav:
        dumpav = max(1,int(np.round(timeav*60.0 / data.dump_period)))
        if dumpav > long_scan:
            dumpav = short_scan
            print "Time averaging interval of %4.1fmin is longer than the longest scan. Scaling back to %4.1fmin to include all scans."%(timeav,dumpav*(data.dump_period/60.0))
            timeav = dumpav*(data.dump_period/60.0)
    else:
        dumpav = int(short_scan/2)
        timeav = dumpav*(data.dump_period/60.0)
    print "Averaging %d dumps to %3d x %4.1fmin intervals."%(dumpav,int(scan_length*(data.dump_period/60.0)/timeav),timeav)

    #Get the number of channels to average
    freq_width_spec = data.channel_width * len(data.channels)
    if freqav:
        chanav = max(1,int(np.round(freqav*1e6 / data.channel_width)))
        if chanav > len(data.channels):
            chanav = len(data.channels)
    else:
        #~100 final channels as default
        chanav=max(1,len(data.channels) // 100)
        freqav=chanav*data.channel_width/1e6
    print "Averaging frequency to %d x %4.1fMHz intervals."%(len(data.channels)//chanav,freqav)
    
    #Prepare arrays for extracted and averaged data
    vis_data = np.empty((0,data.shape[1]//chanav,data.shape[2]))
    flag_data = np.empty((0,data.shape[1]//chanav,data.shape[2]),dtype=np.bool)
    weight_data = np.empty((0,data.shape[1]//chanav,data.shape[2]))

    #Extract the required arrays from the data object for the averager on a scan by scan basis
    for scan, state, target in data.scans():
        scan_vis_data = data.vis[:]
        scan_weight_data = data.weights()[:]
        scan_flag_data = data.flags()[:]
        scan_timestamps = data.timestamps[:]
        scan_channel_freqs = data.channel_freqs[:]
        
        # Average
        scan_vis_data, scan_weight_data, scan_flag_data, scan_timestamps, scan_channel_freqs = averager.average_visibilities(scan_vis_data, scan_weight_data, scan_flag_data, scan_timestamps, 
                                                                                                    scan_channel_freqs, timeav=dumpav, chanav=chanav, flagav=False)        
        vis_data = np.append(vis_data,scan_vis_data,axis=0)
        flag_data = np.append(flag_data,scan_flag_data,axis=0)
        weight_data = np.append(weight_data, scan_weight_data, axis=0)
        channel_freqs = scan_channel_freqs

    return np.array(vis_data), np.array(channel_freqs), np.array(flag_data), np.array(weight_data), freqav, timeav


def condition_data(vis,flags,weight,polarisation):
    """
    Make the data ameniable for plotting.
    - Convert to stokes I if required, in opts.
    - Make vis into a masked array
    - Normalise vis by the mean of each spectrum
    
    Returns
    =======
    visdata =  the normalised masked array constructed from vis, flags (same shape as vis)
    """

    #Convert to Stokes I
    if polarisation == 'I':
        vis = vis[:,:,0] + vis[:,:,1]
        flags = np.logical_or(flags[:,:,0],flags[:,:,1])
        weight = weight[:,:,0] + weight[:,:,1]
    else:
        vis=np.squeeze(vis)
        flags=np.squeeze(flags)
        weight=np.squeeze(weight)

    # Get the abs (we only care about amplitudes)
    visdata = np.abs(vis)

    #Make a masked array
    visdata = MaskedArray(visdata, mask=flags)

    return visdata, flags, weight


def correct_by_mean(vis, axis="Time"):
    """
    Subtract the median of the visibiliy along a given
    axis (usually either Time or Baseline).
    """

    # Get the mean of each spectrum and normalise by it
    if axis=="Time":
        medianvis = np.mean(vis, axis=0)
        corrected_vis = vis - medianvis[np.newaxis,:]
    elif axis=="Channel":
        medianvis = np.mean(vis, axis=1)
        corrected_vis = vis - medianvis[:,np.newaxis]
    return corrected_vis


def weighted_avg_and_std(values, weights, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=axis, weights=weights)
    variance = np.average((values-average)**2, axis=axis, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def plot_std_results(corr_visdata_std,mean_visdata,freqdata,flagdata, baseline, pol, freqav, timeav,fileprefix='filename'):

    #Frequency Range in MHz
    start_freq = freqdata[0]
    end_freq = freqdata[-1]

    #Get flag frequencies
    #print visdata.shape,np.sum(visdata.mask, axis=0)
    channel_width = freqdata[1]-freqdata[0]
    flagged_chans = np.sum(flagdata, axis=0, dtype=np.float)
    # Show where 50% of data is flagged
    flagged_chans = flagged_chans/flagdata.shape[0] > 0.5
    flag_freqs=freqdata[flagged_chans]

    #Set up the figure
    fig = plt.figure(figsize=(8.3,8.3))

    fig.subplots_adjust(hspace=0.0)

    #Plot the gain vs elevation for each target
    ax1 = plt.subplot(211)
    
    ax1.axhline(0.005,ls='--', color='red')
    ax1.plot(freqdata,corr_visdata_std/mean_visdata*100.0)
    plt.ylabel('Standard Deviation (% of mean)')
    tstring = 'Spectral Baseline, %s'%baseline
    if pol=='I':
        tstring += ', Stokes I'
    else:
        tstring += ', %s pol'%pol

    # Add some pertinent information.
    pstring = 'Time average: %4.1f min.\n'%(timeav)
    pstring += 'Frequency average: %4.1f MHz.\n'%(freqav)
    pstring += 'Median standard deviation: %5.3f%%'%np.median(corr_visdata_std/mean_visdata*100.0)
    plt.figtext(0.5,0.83,pstring)

    #plot title
    plt.title(tstring)
    
    #Plot the spectrum with standard deviations around it
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(freqdata,mean_visdata)
    plt.figtext(0.6,0.47,'Average spectrum')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (MHz)')

    #Overlay rfi
    rfilib.plot_RFI_mask(ax1,flag_freqs,channel_width)
    rfilib.plot_RFI_mask(ax2,flag_freqs,channel_width)
    plt.xlim((end_freq,start_freq))
    
    #Convert ticks to MHZ
    ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
    ax2.xaxis.set_major_formatter(ticks)

    fig.savefig(fileprefix+'_SpecBase_'+baseline+'_'+pol+'.pdf')
    plt.close(fig)


def analyse_spectrum(input_file,output_dir='.',polarisation='I',baseline=None,target=None,freqav=None,timeav=None,freq_chans=None,correct='spline'):
    """
    Plot the mean and standard deviation of the bandpass amplitude for a given target in a file

    Inputs
    ======
    polarisation: Polarisation to produce spectrum, options are I, HH, VV, HV, VH. Default is I.
    baseline: Baseline to load (e.g. 'ant1,ant1' for antenna 1 auto-corr), default is first single-dish baseline in file.
    target: Target to plot spectrum of, default is the first target in the file.
    freqav: Frequency averaging interval in MHz. Default is a bin size that will produce 100 frequency channels.
    timeav: Time averageing interval in minutes. Default is the shortest scan length on the selected target.
    freq_chans: Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass.
    correct: Method to use to correct the spectrum in each average timestamp. Options are 'spline' - fit a cubic spline,'channels' - use the average at each channel Default: 'spline'
    output_dir: Output directory for pdfs. Default is cwd.
    """

    # Get data from h5 file and use 'select' to obtain a useable subset of it.
    data, bline, polarisation = read_and_select_file(input_file, bline=baseline, target=target, channels=freq_chans, polarisation=polarisation)

    # Average the data to the required time a frequency bins
    visdata, freqdata, flagdata, weightdata, freqav, timeav = extract_and_average(data, timeav=timeav, freqav=freqav)

    # Make a masked array out of visdata, get amplitudes and average to stokes I if required
    visdata, flagdata, weightdata = condition_data(visdata, flagdata, weightdata, polarisation)

    # Get the mean visibility spectrum
    vis_mean, vis_std = weighted_avg_and_std(visdata, weightdata, axis=0)

    #Correct the visibilities by subtracting the average of the channels at each timestamp
    #and the average of the timestamps at each channel.
    if correct=='channels':
        corr_vis = correct_by_mean(visdata,axis="Channel")
        corr_vis = correct_by_mean(corr_vis,axis="Time")
    # Correct the background in each time bin by fitting a cubic spline.
    elif correct=='spline':
        #Knots will have to satisfy Schoenberg-Whitney conditions for splie else revert to straight mean of channels
        try:
            corr_vis = np.array([data - getbackground_spline(data, 2) for data in visdata])
        except ValueError:
            corr_vis = correct_by_mean(visdata,axis="Channel")
            corr_vis = correct_by_mean(corr_vis,axis="Time")
    #get weighted standard deviation of corrected visdata
    corr_vis_mean, corr_vis_std = weighted_avg_and_std(corr_vis, weightdata, axis=0)

    fileprefix = os.path.join(output_dir,os.path.splitext(input_file.split('/')[-1])[0])
    plot_std_results(corr_vis_std,vis_mean,freqdata,flagdata,bline, polarisation, freqav, timeav,fileprefix)
