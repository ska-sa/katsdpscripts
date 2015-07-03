import os

import numpy as np
from numpy.ma import MaskedArray
import scipy.interpolate as interpolate

import katdal
from katdal import averager

import matplotlib.pyplot as plt
from matplotlib import ticker

from katsdpscripts.RTS import rfilib
from katsdpscripts.RTS import git_info

import h5py


def read_and_select_file(file, bline=None, target=None, channels=None, polarisation=None, flags_file=None, **kwargs):
    """
    Read in the input h5 file and make a selection based on kwargs.

    file:   {string} filename of h5 file to open in katdal

    Returns:
        A masked array with the visibility data to plot and the frequency array to plot.
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
        # Default is drop first and last 10% of the bandpass
        start_chan = num_channels // 20
        end_chan   = start_chan * 19
    else:
        start_chan = int(channels.split(',')[0])
        end_chan = int(channels.split(',')[1])
    chan_range = range(start_chan,end_chan+1)
    select_data['channels']=chan_range

    data.select(strict=False, reset='', **select_data)

    #Check there is some data left over
    if data.shape[0] == 0:
        raise ValueError('Selection has resulted in no data to process.')

    #Get the selected visibilities and flags (average to stokes I if required and extend flags across corr products)
    vis = np.abs(np.sum(data.vis[:],axis=-1))
    if flags_file is None:
        flags = np.sum(data.flags()[:],axis=-1)
    else:
        #Open the flags file
        ff = h5py.File(flags_file)
        #Select file flages based on h5 file selection
        file_flags=ff['flags'].value
        file_flags = file_flags[data._time_keep]
        file_flags = file_flags[:,data._freq_keep]
        file_flags = file_flags[:,:,data._corrprod_keep]
        #Extend flags
        flags = np.sum(file_flags,axis=-1)
    weights = np.sum(data.weights()[:], axis=-1)
    outputvis = np.ma.masked_array(vis, mask=flags)

    #return the selected data
    return outputvis, weights, data , ant1 + ant2, polarisation


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
        #remove masked indices
        indices = [index for index in indices if ~y.mask[index]]

        # Fit the spline with 0 weights at the mask.
        thisfit = interpolate.LSQUnivariateSpline(x,y,indices,k=deg,w=(~y.mask).astype(np.float))
        
        thisfitted_data=np.asarray(thisfit(x),y.dtype)

        # Subtract the fitted spline from the data
        residual = y-thisfitted_data
        this_std = np.ma.std(residual)

        # Reject data more than 5sigma from the residual. 
        flags = residual > 5*this_std

        # Mask rejected data
        y[flags] = np.ma.masked

    # Final iteration has knots separated by "spike_width".
    npieces = int(y.shape[0]/spike_width)
    psize = (x[-1]+1)/npieces
    firstindex = int((y.shape[0]%psize))
    indices = np.trim_zeros(np.arange(firstindex,arraysize,psize))
    #remove the masked indices
    indices = [index for index in indices if ~y.mask[index]]
    # Get the final background.
    finalfit = interpolate.LSQUnivariateSpline(x,y,indices,k=3,w=(~y.mask).astype(np.float))
    thisfitted_data = np.asarray(finalfit(x),y.dtype)

    return(thisfitted_data)

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
        medianvis = np.ma.mean(vis, axis=0)
        corrected_vis = vis - medianvis[np.newaxis,:]
    elif axis=="Channel":
        medianvis = np.ma.mean(vis, axis=1)
        corrected_vis = vis - medianvis[:,np.newaxis]
    return corrected_vis

def weighted_avg_and_std(values, weights, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.ma.average(values, axis=axis, weights=weights)
    variance = np.ma.average((values-average)**2, axis=axis, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def plot_std_results(corr_visdata_std,mean_visdata,freqdata,flagdata, baseline, pol, freqav, timeav, obs_details, fileprefix='filename'):

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
    ax1.set_yscale('log')
    plt.ylabel('log10 [Standard Deviation (% of mean)]')
    tstring = 'Spectral Baseline, %s'%baseline
    if pol=='I':
        tstring += ', Stokes I'
    else:
        tstring += ', %s pol'%pol

    # Add some pertinent information.
    pstring = 'Time average: %4.1f min.\n'%(timeav)
    pstring += 'Frequency average: %4.1f MHz.\n'%(freqav)
    pstring += 'Median standard deviation: %6.4f%%'%np.ma.median(corr_visdata_std/mean_visdata*100.0)
    plt.figtext(0.5,0.83,pstring)
    plt.grid()

    #plot title
    plt.title(tstring)
    plt.suptitle(obs_details)
    
    #Plot the spectrum with standard deviations around it
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(freqdata,mean_visdata)
    plt.figtext(0.6,0.47,'Average spectrum')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (MHz)')

    #Overlay rfi
    rfilib.plot_RFI_mask(ax1,main=False,extra=flag_freqs,channelwidth=channel_width)
    rfilib.plot_RFI_mask(ax2,main=False,extra=flag_freqs,channelwidth=channel_width)
    if end_freq<start_freq:
        plt.xlim((end_freq,start_freq))
    else:
        plt.xlim((start_freq,end_freq))
    #Convert ticks to MHZ
    ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
    ax2.xaxis.set_major_formatter(ticks)
    plt.grid()
    plt.figtext(0.89, 0.13, git_info(), horizontalalignment='right',fontsize=10)

    fig.savefig(fileprefix+'_SpecBase_'+baseline+'_'+pol+'.pdf')
    plt.close(fig)


def analyse_spectrum(input_file,output_dir='.',polarisation='I',baseline=None,target=None,freqav=None,timeav=None,freq_chans=None,correct='spline',flags_file=None,debug=False):
    """
    Plot the mean and standard deviation of the bandpass amplitude for a given target in a file

    Inputs
    ======
    polarisation: Polarisation to produce spectrum, options are I, HH, VV, HV, VH. Default is I.
    baseline: Baseline to load (e.g. 'ant1,ant1' for antenna 1 auto-corr), default is first single-dish baseline in file.
    target: Target to plot spectrum of, default is the first target in the file.
    freqav: Frequency averaging interval in MHz. Default is a bin size that will produce 100 frequency channels.
    timeav: Time averageing interval in minutes. Default is the shortest scan length on the selected target.
    freq_chans: Range of frequency channels to keep (zero-based, specified as 'start,end', default is 90% of the bandpass.
    correct: Method to use to correct the spectrum in each average timestamp. Options are 'spline' - fit a cubic spline,'channels' - use the average at each channel Default: 'spline'
    output_dir: Output directory for pdfs. Default is cwd.
    flags_file: Name of .h5 file containg flags calculated from 'rfi_report.py'.
    debug: make a debug file containing all of the background fits to the dumps
    """

    # Get data from h5 file and use 'select' to obtain a useable subset of it.
    visdata, weightdata, h5data, bline, polarisation = \
        read_and_select_file(input_file, bline=baseline, target=target, channels=freq_chans, polarisation=polarisation, flags_file=flags_file)

    # Extract visibility data as a masked array containing flags
    # visdata = extract_visibiities(data)
    # Correct the visibilities by subtracting the average of the channels at each timestamp
    #and the average of the timestamps at each channel.
    if correct=='channels':
        corr_vis = correct_by_mean(visdata,axis="Channel")
        corr_vis = correct_by_mean(corr_vis,axis="Time")
    # Correct the background in each time bin by fitting a cubic spline.
    elif correct=='spline':
        #Knots will have to satisfy Schoenberg-Whitney conditions for spline else revert to straight mean of channels
        try:
            corr_vis = np.ma.masked_array([data - getbackground_spline(data, 3) for data in visdata],mask=visdata.mask)
        except ValueError:
            corr_vis = correct_by_mean(visdata,axis="Channel")
            corr_vis = correct_by_mean(corr_vis,axis="Time")

    #Get the number of dumps to average
    dumpav = max(1,int(np.round(timeav*60.0 / h5data.dump_period)))
    if dumpav > len(h5data.timestamps):
        dumpav = 1
        print "Time averaging interval of %4.1fmin is longer than the observation length. No time averaging will be applied."%(timeav)
        timeav = dumpav*(h5data.dump_period/60.0)
    print "Averaging time to %3d x %4.1fmin (%d dump) intervals."%(len(h5data.timestamps)//dumpav,timeav,dumpav)

    #Get the number of channels to average
    chanav = max(1,int(np.round(freqav*1e6 / h5data.channel_width)))
    if chanav > len(h5data.channel_freqs):
        chanav = 1
        print "Frequency averaging interval of %4.1fMHz is wider than available bandwidth. No frequency averaging will be applied."%(freqav)
        freqav = h5data.channel_width/1e6
    print "Averaging frequency to %d x %4.1fMHz intervals."%(len(h5data.channel_freqs)//chanav,freqav)

    #Average the data over all time in chanav channels
    av_visdata = averager.average_visibilities(visdata.data, weightdata, visdata.mask, h5data.timestamps, h5data.channel_freqs, timeav=len(h5data.timestamps), chanav=chanav)

    #Average the background subtracted data in dumpav times and chanav channels
    av_corr_vis = averager.average_visibilities(corr_vis.data, weightdata, corr_vis.mask, h5data.timestamps, h5data.channel_freqs, timeav=dumpav, chanav=chanav)

    #Get the averaged weights and channel frequencies
    av_weightdata = av_corr_vis[1]
    av_channel_freqs = av_corr_vis[4]
    
    #Make a masked array out of the averaged visdata
    av_visdata = np.ma.masked_array(np.squeeze(av_visdata[0]),mask=np.squeeze(av_visdata[2]))

    #Make a masked array out of the averaged background subtracted data
    av_corr_vis = np.ma.masked_array(av_corr_vis[0],mask=av_corr_vis[2])

    #get weighted standard deviation of background subtracted data
    corr_vis_mean, corr_vis_std = weighted_avg_and_std(av_corr_vis, av_weightdata, axis=0)

    fileprefix = os.path.join(output_dir,os.path.splitext(input_file.split('/')[-1])[0])
    obs_duration = np.str(np.round((h5data.end_time.to_mjd() - h5data.start_time.to_mjd())*24*60,2)) + ' min'
    h5name = h5data.name.split('/')[-1]
    obs_details = h5name + ', start ' + h5data.start_time.to_string() + ', duration ' + obs_duration
    plot_std_results(corr_vis_std,np.squeeze(av_visdata),av_channel_freqs,av_corr_vis.mask,bline, polarisation, freqav, timeav,fileprefix)
