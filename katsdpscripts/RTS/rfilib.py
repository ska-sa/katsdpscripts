#Library to contain RFI flagging routines and other RFI related functions
import katdal
import katpoint 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid import Grid

import matplotlib.pyplot as plt; plt.ioff()
import matplotlib.gridspec as gridspec
from matplotlib import ticker

import numpy as np
import optparse
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import math

import pickle

import h5py
import os

#########################
# RFI Detection routines
#########################

#----------------------------------------------------------------------------------
#--- FUNCTION :  getbackground_spline
# Fit the background to the array in "data" using an iterative fit of a spline to the 
# data. On each iteration the number of density of knots in the spline is increased 
# up to "spike_width", and residual peaks more than 5sigma are removed.
#----------------------------------------------------------------------------------
def getbackground_spline(data,spike_width):
    """ From a 1-d data array determine a background iteratively by fitting a spline
    and removing data more than a few sigma from the spline """

    # Remove the first and last element in data from the fit.
    y=np.ma.copy(data[1:-1])
    arraysize=y.shape[0]
    x=np.arange(arraysize)

    # Iterate 4 times
    for iteration in range(4):

        # First iteration fits a linear spline with 3 knots.
        if iteration==0:
            npieces=3
            nsigma=3.0
        # Second iteration fits a quadratic spline with 10 knots.
        elif iteration==1:
            npieces=10
            nsigma=3.0
        # Third and fourth iterations fit a cubic spline with 50 and 75 knots respectively.
        elif iteration>1:
            npieces=iteration*25
            nsigma=4.0
        deg=min(iteration+1,3)
        
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
        this_std = np.std(residual)

        # Reject data more than nsigma from the residual. 
        flags = residual > nsigma*this_std

        # Mask the rejected data
        y[flags] = np.ma.masked
        #y[flags] = thisfitted_data[flags] + this_std

    # Final iteration has knots separated by "spike_width".
    npieces = int(y.shape[0]/spike_width)
    psize = (x[-1]+1)/npieces
    firstindex = int((y.shape[0]%psize))
    indices = np.trim_zeros(np.arange(firstindex,arraysize,psize))
    #remove the masked indices
    indices = [index for index in indices if ~y.mask[index]]
    #fitting_data=interp_edges(y)

    # Get the final background.
    finalfit = interpolate.LSQUnivariateSpline(x,y,indices,k=3,w=(~y.mask).astype(np.float))
    thisfitted_data = np.asarray(finalfit(x),y.dtype)
    
    # Insert the original data at the beginning and ends of the data array.
    thisfitted_data = np.append(thisfitted_data,data[-1])
    thisfitted_data = np.insert(thisfitted_data,0,data[0])

    return(thisfitted_data)

def plot_RFI_mask(pltobj,extra=None,channelwidth=1e6):
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
    if not extra is None:
        for i in xrange(extra.shape[0]):
            pltobj.axvspan(extra[i]-channelwidth/2,extra[i]+channelwidth/2, alpha=0.7, color='Maroon')
                

def detect_spikes_sumthreshold(data, blarray=None, spike_width=5, outlier_sigma=11.0, window_size_auto=[1,3,5], window_size_cross=[2,4,8]):
    """FUNCTION :  detect_spikes_sumthreshold
    Given an array "data" from a baseline:
    determine if the data is an auto-correlation or a cross-correlation.
    Get the background in the data using a median filter for auto_correlations or a 
    cubic spline for cross correlations.
    Make an array of flags from the data using the "sumthreshold" method and return
    this array of flags.
    Parameters
    ----------
    data : array-like
        An N-dimensional numpy array containing the data to flag
    blarray : array-like
        An array of baseline labels used to determine if the 
        baseline index is an auto- or a cross- correlation. Should have the same
        shape as the baseline part of the data array.
    spike_width : integer
        The width of the median filter and the gaps between knots in the spline fit.
    outlier_sigma : float
        The number of sigma in the first iteration of the sumthreshold method.
    buffer_size : int
        The number of timestamps in the data array to average.
    window_size_auto : array of ints
        The sizes of the averaging windows in each sumthreshold iteration for auto-correlations.
    window_size_cross : array of ints 
        The sizes of the averaging windows in each sumthreshold iteration for cross-correlations.
    """

    # Kernel size for the median filter.
    kernel_size = 2 * max(int(spike_width), 0) + 1
    #Init Flags
    flags = np.zeros(data.shape, dtype=np.bool)

    for bl_index in range(data.shape[-1]):
        # Extract this baseline from the data
        this_data_buffer = data[:,bl_index]
        #Separate the auto-correlations and the cross-correlations
        #auto-correlations use a median filter and cross correlations
        #use a fitted spline.
        if blarray is not None:
            bl_name = bline.bls_ordering[bl_index]
        
        # Check if this is an auto or a cross... (treat as auto if we have no bl-ordering)
        if blarray is None or bl_name[0][:-1] == bl_name[1][:-1]:
            #Auto-Correlation.
            #filtered_data = np.asarray(signal.medfilt(this_data_buffer, kernel_size), this_data_buffer.dtype)
            filtered_data = getbackground_spline(this_data_buffer,kernel_size)
            #Use the auto correlation window function
            #filtered_data = ndimage.grey_opening(this_data_buffer, (kernel_size,))
            window_bl = window_size_auto
            this_sigma = outlier_sigma
        else:
            #Cross-Correlation.
            filtered_data = getbackground_spline(this_data_buffer,kernel_size)
            #Use the cross correlation window function
            window_bl = window_size_cross
            # Can lower the threshold a little (10%) for cross correlations
            this_sigma = outlier_sigma * 0.9
    
        av_dev = (this_data_buffer-filtered_data)

        av_abs_dev = np.abs(av_dev)
            
        # Calculate median absolute deviation (MAD)
        med_abs_dev = np.median(av_abs_dev[av_abs_dev>0])
            
        # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
        estm_stdev = 1.4826 * med_abs_dev
            
        # Identify initial outliers (again based on normal assumption), and replace them with local median
        threshold = this_sigma * estm_stdev
        outliers = np.zeros(data.shape[0],dtype=np.bool)
        # Always flag the first element of the array.
        outliers[0] = True 

        for window in window_bl:
            #Set up 'this_data' from the averaged background subtracted buffer 
            bl_data = av_dev.copy()
                
            #The threshold for this iteration is calculated from the initial threshold
            #using the equation from Offringa (2010).
            # rho=1.3 in the equation seems to work better for KAT-7 than rho=1.5 from AO.
            thisthreshold = threshold / pow(1.2,(math.log(window)/math.log(2.0)))
            #Set already flagged values to be the value of this threshold
            bl_data[outliers] = thisthreshold
                
            #Calculate a rolling average array from the data with a windowsize for this iteration
            weight = np.repeat(1.0, window)/window
            avgarray = np.convolve(bl_data, weight,mode='valid')
                
            #Work out the flags from the convolved data using the current threshold.
            #Flags are padded with zeros to ensure the flag array (derived from the convolved data)
            #has the same dimension as the input data.
            this_flags = (avgarray > thisthreshold)
            #Convolve the flags to be of the same width as the current window.
            convwindow = np.ones(window,dtype=np.bool)
            this_outliers = np.convolve(this_flags,convwindow)
            #"OR" the flags with the flags from the previous iteration.
            outliers = outliers | this_outliers
        flags[:,bl_index] = outliers
    return flags



def detect_spikes_mad(data,blarray=None,spike_width=20,outlier_sigma=3):
    """
    FUNCTION :  detect_spikes_mad
    Given an array "data" from a baseline determine flags using the "median absolute
    deviation" method. The data is median filtered (with a kernel defined by "spike_width")
    to find a background and then rfi spikes are found by finding peaks that are 
    "outlier_sigma" from the median of the absolute values of the background subtracted data.
    Parameters
    ----------
    data : array-like
        An N-dimensional numpy array containing the data to flag
    blarray : CorrProdRef object
        Baseline labels used to determine if the 
        baseline index is an auto- or a cross- correlation.
    spike_width : integer
        The width of the median filter and the gaps between knots in the spline fit.
    outlier_sigma : float
        The number of sigma in the first iteration of the sumthreshold method.
    """    
    flags = np.zeros_like(data, dtype=np.uint8)

    for bl_index in range(data.shape[-1]):
        spectral_data = np.abs(data[:,bl_index])
        kernel_size = 2 * max(int(spike_width), 0) + 1
        # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
        #filtered_data = np.asarray(signal.medfilt(spectral_data, kernel_size), spectral_data.dtype)     
        filtered_data =  np.median(rolling_window(spectral_data, kernel_size,pad=True),axis=-1)
        # The deviation is measured relative to the local median in the signal
        abs_dev = np.abs(spectral_data - filtered_data)
        # Calculate median absolute deviation (MAD)
        med_abs_dev = np.median(abs_dev[abs_dev>0])
        #med_abs_dev = signal.medfilt(abs_dev, kernel)
        # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
        estm_stdev = 1.4826 * med_abs_dev
        # Identify outliers (again based on normal assumption), and replace them with local median
        #outliers = ( abs_dev > self.n_sigma * estm_stdev)
        #print outliers
        # Identify only positve outliers
        outliers = (spectral_data - filtered_data > outlier_sigma*estm_stdev)
        flags[:,bl_index] = outliers
        # set appropriate flag bit for detected RFI
    
    return flags


def detect_spikes_median(data,blarray=None,spike_width=10,outlier_sigma=11.0):
    """
    FUNCTION :  detect_spikes_median
    Simple RFI flagging through thresholding.
    Given an array "data" from a baseline determine flags using a simple median filter.
    Trivial thresholder that looks for n sigma deviations from the average
    of the supplied frame.
    Parameters
    ----------
    data : array-like
        An N-dimensional numpy array containing the data to flag
    blarray : CorrProdRef object
        Baseline labels used to determine if the 
        baseline index is an auto- or a cross- correlation.
        This is a dummy varaible
    spike_width : integer
        The width of the median filter and the gaps between knots in the spline fit.
    outlier_sigma : float
        The number of sigma in the first iteration of the sumthreshold method.
    """
    flags = np.zeros(list(data.shape), dtype=np.uint8)
    for bl_index in xrange(data.shape[-1]):
        spectral_data = data[...,bl_index]
        #spectral_data = np.atleast_1d(spectral_data)
        kernel_size=spike_width
        filtered_data =  np.median(rolling_window(spectral_data, kernel_size,pad=True),axis=-1)
        # The deviation is measured relative to the local median in the signal
        abs_dev = spectral_data - filtered_data
        outliers = (abs_dev > np.std(abs_dev)*2.3)# TODO outlier_sigma pram
        flags[:,bl_index] = outliers
    return flags

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


def detect_spikes_orig(data, axis=0, spike_width=2, outlier_sigma=11.0):
    """
    Detect and Remove outliers from data, replacing them with a local median value.

    The data is median-filtered along the specified axis, and any data values
    that deviate significantly from the local median is is marked as outlier.

    Parameters
    ----------
    data : array-like
        N-dimensional numpy array containing data to clean
    axis : int, optional
        Axis along which to perform median, between 0 and N-1
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed. A size
        of <= 0 implies no spike removal. The kernel size for the median filter
        will be 2 * spike_width + 1.
    outlier_sigma : float, optional
        Multiple of standard deviation that indicates an outlier

    Returns
    -------
    cleaned_data : array
        N-dimensional numpy array of same shape as original data, with outliers
        removed

    Notes
    -----
    This is very similar to a *Hampel filter*, also known as a *decision-based
    filter* or three-sigma edit rule combined with a Hampel outlier identifier.

    .. todo::

       TODO: Make this more like a Hampel filter by making MAD time-variable too.

    """
    flags = np.zeros(data.shape, dtype='int32')#making sure that the data is already 1-D
    spectral_data = np.atleast_1d(data)
    kernel_size = 2 * max(int(spike_width), 0) + 1
    # Median filter data along the desired axis, with given kernel size
    kernel = np.ones(spectral_data.ndim, dtype='int32')
    kernel[axis] = kernel_size
    # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
    #filtered_data = np.asarray(signal.medfilt(spectral_data, kernel), spectral_data.dtype)
    filtered_data =  np.median(rolling_window(spectral_data, kernel_size,pad=True),axis=-1)
    
    # The deviation is measured relative to the local median in the signal
    abs_dev = np.abs(spectral_data - filtered_data)
    # Calculate median absolute deviation (MAD)
    med_abs_dev = np.expand_dims(np.median(abs_dev[abs_dev>0],axis), axis)
    #med_abs_dev = signal.medfilt(abs_dev, kernel)
    # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
    estm_stdev = 1.4826 * med_abs_dev
    # Identify outliers (again based on normal assumption), and replace them with local median
    #outliers = ( abs_dev > self.n_sigma * estm_stdev)
    # Identify only positve outliers
    outliers = (spectral_data - filtered_data > outlier_sigma*estm_stdev)
    # assign as my cols to flags as outliers has.
    flags[...] = outliers[...]
    #return flags
    return flags

##############################
# End of RFI detection routines
##############################

def get_flag_stats(h5, flags=None, norm_spec=None):
    """
    Given a katdal object, remove a dc offset for each record
    (ignoring severe spikes) then obtain an average spectrum of
    all of the scans in the data.
    Return the average spectrum with dc offset removed and the number of times
    each channel is flagged (flags come optionally from 'flags' else from the 
    flages in the input katdal object). Optinally provide a 
    spectrum (norm_spec) to divide into the calculated bandpass.
    """

    sumarray=np.zeros((h5.shape[1],4))
    offsetarray=np.zeros((h5.shape[0],4))
    weightsum=np.zeros((h5.shape[1],4),dtype=np.int)
    if flags is None:
        flags = h5.flags()
    for num,thisdata in enumerate(h5.vis):
        #Extract pols
        thisdata = np.abs(thisdata[0,:])
        # normalise if defined
        if norm_spec is not None: thisdata /= norm_spec
        #Get DC height (median rather than mean is more robust...)
        offset = np.median(thisdata[np.where(~flags[num])],axis=0)
        #Make an elevation corrected offset to remove outliers
        offsetarray[num,:] = offset
        #Remove the DC height
        weights = (~flags[num]).astype(np.float)
        thisdata = thisdata/offset
        weightsum += weights
        #Sum the data for this target
        sumarray = sumarray + thisdata*weights
    averagespec = sumarray/(weightsum.astype(np.float)+1.e-10)
    flagfrac = 1. - (weightsum.astype(np.float)/h5.shape[0].astype(np.float))
    return {'spectrum': averagespec, 'numrecords_tot': h5.shape[0], 'flagfrac': flagfrac, 'channel_freqs': h5.channel_freqs, 'dump_period': h5.dump_period}

def plot_flag_data(label,spectrum,flagfrac,vis,flags,freqs,pdf):
    """
    Produce a plot of the average spectrum in H and V 
    after flagging and attach it to the pdf output.
    Also show fraction of times flagged per channel.
    """

    #Set up the figure
    fig = plt.figure(figsize=(8.3,11.7))
    plt.suptitle(label,fontsize=14)

    outer_grid= gridspec.GridSpec(2,1)
    
    for num,pol in enumerate(['HH','VV']):
        data=np.abs(vis[:,:,num-1])
        inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[num-1], hspace=0.0)
        #Plot the spectrum for each target
        ax1 = plt.subplot(inner_grid[0])
        ax1.set_title(pol +' polarisation')
        plt.plot(freqs,spectrum[:,num-1])
        plot_RFI_mask(ax1)
        ticklabels=ax1.get_xticklabels()
        plt.setp(ticklabels,visible=False)
        ticklabels=ax1.get_yticklabels()
        plt.setp(ticklabels,visible=False)
        plt.xlim((min(freqs),max(freqs)))
        plt.ylabel('Mean amplitude\n(arbitrary units)')
        #Plot the average flags
        ax = plt.subplot(inner_grid[1],sharex=ax1)
        plt.plot(freqs,flagfrac[:,num-1],'r-')
        plt.ylim((0.,1.))
        plot_RFI_mask(ax)
        ticklabels=ax.get_xticklabels()
        plt.setp(ticklabels,visible=False)
        plt.xlim((min(freqs),max(freqs)))
        plt.ylabel('Fraction flagged')
        #Waterfall plot
        ax = plt.subplot(inner_grid[2],sharex=ax1)
        kwargs={'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none', 'extent' : (freqs[0],freqs[-1], -0.5, data.shape[0] - 0.5)}
        image=plt.imshow(data,**kwargs)
        ampsort=np.sort(data,axis=None)
        arrayremove = int(len(ampsort)*(1.0 - 0.97)/2.0)
        lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
        image.norm.vmin = lowcut
        image.norm.vmax = highcut
        image.set_cmap('Greys')
        plotflags = np.zeros(flags.shape[0:2]+(4,))
        plotflags[:,:,0] = 1.0
        plotflags[:,:,3] = flags[:,:,num-1]
        flagimage=plt.imshow(plotflags,**kwargs)
        plt.xlim((min(freqs),max(freqs)))
        #Convert ticks to MHZ
        ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
        ax.xaxis.set_major_formatter(ticks)
        ticklabels=ax.get_yticklabels()
        plt.setp(ticklabels,visible=False)
        plt.ylabel('Time')
        plt.xlabel('Frequency (Hz)')

    pdf.savefig(fig)
    plt.close('all')
 
def plot_waterfall(visdata,flags,channel_freqs):
    fig = plt.figure(figsize=(8.3,11.7))
    data=np.squeeze(np.abs(visdata[:,:,0]))
    kwargs={'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none', 'extent' : (channel_freqs[0],channel_freqs[1], -0.5, data.shape[0] - 0.5)}
    image=plt.imshow(data,**kwargs)
    image.set_cmap('Greys')
    #flagimage=plt.imshow(flags[:,:,0],**kwargs)
    #Make an array of RGBA data for the flags (initialize to alpha=0)
    plotflags = np.zeros(flags.shape[0:2]+(4,))
    plotflags[:,:,0] = 1.0
    plotflags[:,:,3] = flags[:,:,0]
    plt.imshow(plotflags,**kwargs)
    ampsort=np.sort(data,axis=None)
    arrayremove = int(len(ampsort)*(1.0 - 0.97)/2.0)
    lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    pdf.savefig(fig)
    plt.close(fig)
    fig = plt.figure(figsize=(8.3,11.7))
    data=np.squeeze(np.abs(visdata[:,:,1]))
    kwargs={'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none', 'extent' : (channel_freqs[0],channel_freqs[1], -0.5, data.shape[0] - 0.5)}
    image=plt.imshow(data,**kwargs)
    image.set_cmap('Greys')
    #flagimage=plt.imshow(flags[:,:,0],**kwargs)
    #Make an array of RGBA data for the flags (initialize to alpha=0)
    plotflags = np.zeros(flags.shape[0:2]+(4,))
    plotflags[:,:,0] = 1.0
    plotflags[:,:,3] = flags[:,:,1]
    flagimage=plt.imshow(plotflags,**kwargs)
    ampsort=np.sort(data,axis=None)
    arrayremove = int(len(ampsort)*(1.0 - 0.97)/2.0)
    lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    pdf.savefig(fig)
    plt.close(fig)

def generate_flag_table(input_file,output_root='.',static_flags=None):
    """
    Flag the visibility data in the h5 file ignoring the channels specified in static_flags.

    This will write a list of flags per scan to the output h5 file.
    """

    basename = os.path.join(output_root,os.path.splitext(input_file.split('/')[-1])[0]+'_flags')
    outfile=h5py.File(basename+'.h5','w')

    h5 = katdal.open(input_file)

    #Read static flags from pickle
    if static_flags:
        sff = open(static_flags)
        static_flags = pickle.load(sff)
        sff.close()
    else:
        #Create dummy static flag array if no static flags are specified. 
        static_flags=np.zeros(num_channels,dtype=np.bool)

    #Set up the mask for broadcasting
    if static_flags is not None:
        mask_array = static_flags[np.newaxis,:,np.newaxis]
    else:
        mask_array = np.zeros((1,h5.vis.shape[1],1),dtype=np.bool)

    #loop through scans
    for scan, state, target in h5.scans():
        this_data = h5.vis[:]
        this_flags = h5.flags()[:]

        #Construct a masked array with flags removed
        mask_flags = np.zeros(this_flags.shape,dtype=np.bool)
        #Broadcast the channel mask to the same shape as this_flags
        mask_flags[:] = mask_array

        #OR the mask flags with the flags already in the h5 file
        this_flags = this_flags | mask_flags

        this_data = np.ma.MaskedArray(np.abs(this_data),mask=this_flags,fill_value=np.nan)

        detected_flags = np.array([detect_spikes_sumthreshold(this_dump,outlier_sigma=8.0,spike_width=13.0) for this_dump in this_data])
        #Flags are 8 bit:
        #1: 'reserved0' 
        #2: 'static' 
        #3: 'cam' 
        #4: 'reserved3' 
        #5: 'detected_rfi' 
        #6: 'predicted_rfi' 
        #7: 'reserved6' 
        #8: 'reserved7'
        all_flags = np.zeros(h5.vis.shape+(8,),dtype=np.uint8)
        all_flags[...,2] = mask_flags
        all_flags[...,5] = detected_flags

        all_flags=np.packbits(all_flags,axis=3).squeeze()

        grp=outfile.create_group(str(scan))
        grp.create_dataset('flags',data=all_flags)

    outfile.close()

    return

def generate_rfi_report(input_file,input_flags=None,output_root='.',antenna=None,targets=None,freq_chans=None):
    """
    Create an RFI report- store flagged spectrum and number of flags in an output h5 file
    and produce a pdf report.

    Inputs
    ======
    input_file - input h5 filename
    input_flags - input h5 flags; will overwrite flags in h5 file- h5 file in format returnd from generate_flag_table
    output_root - directory where output is to be placed - defailt cwd
    antenna - which antenna to produce report on - default first in file
    targets - which target to produce report on - default all
    freq_chans - which frequency channels to work on format - <start_chan>,<end_chan> default - 90% of bandpass
    """

    h5 = katdal.open(input_file)

    #Get the selected antenna or default to first file antenna
    ant=antenna or h5.ants[0].name

    #Frequency range
    num_channels = len(h5.channels)

    if input_flags is not None:
        input_flags = h5py.File(input_flags)

    if freq_chans is None:
        # Default is drop first and last 5% of the bandpass
        start_chan = num_channels//20
        end_chan   = num_channels - start_chan
    else:
        start_chan = int(freq_chans.split(',')[0])
        end_chan = int(freq_chans.split(',')[1])
    chan_range = range(start_chan,end_chan+1)

    # Set up the output file
    basename = os.path.join(output_root,os.path.splitext(input_file.split('/')[-1])[0]+'_' + ant + '_RFI')
    pdf = PdfPages(basename+'.pdf')

    # Select the desired antenna and remove slews from the file
    h5.select(ants=ant)

    if h5.shape[0]==0:
        raise ValueError('Selection has resulted in no data to process.')

    if targets is None: targets = h5.catalogue.targets 

    #Set up the output data dictionary
    data_dict = {}

    # Loop through targets
    for target in targets:
        #Get the target name if it is a target object
        if isinstance(target, katpoint.Target):
            target = target.name
        #Extract target from file
        h5.select(targets=target,scans='~slew')
        #Construct flag, visibility arrays for this target
        if input_flags is not None:
            flags=np.zeros(h5.shape,dtype=np.bool)
            offset=0
            for scan_no in h5.scan_indices:
                scan_flags = input_flags[str(scan_no)+'/flags'].value
                flags[offset:offset+scan_flags.shape[0]]=scan_flags.astype(np.bool)
                offset+=scan_flags.shape[0]
        else:
            #Just use flags from the file
            flags = h5.flags()
        data_dict[target]=get_flag_stats(h5,flags)
        label = 'Flag info for Target: ' + target + ', Antenna: ' + ant +', '+str(data_dict[target]['numrecords_tot'])+' records'
        plot_flag_data(label,data_dict[target]['spectrum'][chan_range],data_dict[target]['flagfrac'][chan_range],h5.vis[:,chan_range,0:2],flags[:,chan_range,:],h5.channel_freqs[chan_range],pdf)

    #Reset the selection
    h5.select(scans='~slew',ants=ant)

    #Construct flag, visibility arrays for this target
    if input_flags is not None:
        flags=np.zeros(h5.shape,dtype=np.bool)
        offset=0
        for scan_no in h5.scan_indices:
            scan_flags = input_flags[str(scan_no)+'/flags'].value
            flags[offset:offset+scan_flags.shape[0]]=scan_flags.astype(np.bool)
            offset+=scan_flags.shape[0]
    else:
        #Just use flags from the file
        flags = h5.flags()

    # Do calculation for all the data and store in the dictionary
    data_dict['all_data']=get_flag_stats(h5,flags)

    #Plot the flags for all data in the file
    label = 'Flag info for all data, Antenna: ' + ant +', '+str(data_dict['all_data']['numrecords_tot'])+' records'
    plot_flag_data(label,data_dict['all_data']['spectrum'][chan_range],data_dict['all_data']['flagfrac'][chan_range],h5.vis[:,chan_range,0:2],flags[:,chan_range,:],h5.channel_freqs[chan_range],pdf)

    #Output to h5 file
    outfile=h5py.File(basename+'.h5','w')
    for targetname, targetdata in data_dict.iteritems():
        #Create a group in the h5 file corresponding to the target
        grp=outfile.create_group(targetname)
        #populate the group with the data
        for datasetname, data in targetdata.iteritems(): grp.create_dataset(datasetname,data=data)
    outfile.close()

    #Finish with a waterfall plot
    #plot_waterfall(h5.vis[:,chan_range,:],all_flags[:,chan_range,:],h5.channel_freqs[chan_range])

    #close the plot
    pdf.close()
