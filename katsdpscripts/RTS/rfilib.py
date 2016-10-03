#Library to contain RFI flagging routines and other RFI related functions
import katdal
import katpoint 
import warnings
warnings.simplefilter('ignore')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid import Grid

import matplotlib.pyplot as plt #; plt.ioff()
import matplotlib.gridspec as gridspec
from matplotlib import ticker

import numpy as np
import optparse
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import skimage
import skimage.morphology
import math

import pickle

import h5py
import os
import shutil
import multiprocessing as mp
import time
import itertools

#Supress warnings
#import warnings
#warnings.simplefilter('ignore')

def running_mean(x, N, axis=None):
    #Fast implementation of a running mean (array x with width N)
    #Stolen from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    #And modified to allow axis selection
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis)
    return np.apply_along_axis(lambda x: (x[N:] - x[:-N])/N, axis, cumsum) if axis else (cumsum[N:] - cumsum[:-N])/N


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

def getbackground_median_filter(in_data,in_flags,broad_iterations=2,fine_iterations=3,spike_width_time=10,spike_width_freq=10,reject_threshold=2.0,interp_nonfinite=True):
    """Determine a smooth background through a 2d data array by iteratively smoothing
    the data with a median filter
    """
    data=in_data[:]
    #Mask array
    mask=np.ones(data.shape,dtype=np.bool)
    mask[:,0]=0
    if in_flags is not None:
        mask[in_flags]=0
    background_func=skimage.filters.median
    morphology_func=skimage.morphology.rectangle
    #First do the broad iterations
    sigma=np.array([max(data.shape[0]//5,1),max(data.shape[1]//5,1)])
    for iteration in range(broad_iterations):
        background=background_func(skimage.img_as_uint(data/data.max()),morphology_func(*sigma),mask=mask)*data.max()/65535.0
        residual=data-background
        # Three more mask runs on the last iteration
        for i in range(2):
            mask[np.abs(residual)>reject_threshold*np.std(residual[np.where(mask)])]=0

    #Next use windows with decreasing width from iterations*spike_width to 1*spike_width
    for extend_factor in range(fine_iterations,0,-1):
        #Convolution sigma
        sigma=np.array([min(spike_width_time*extend_factor,max(data.shape[0]//5,1)),min(spike_width_freq*extend_factor,max(data.shape[1]//5,1))])
        background=background_func(skimage.img_as_uint(data/data.max()),morphology_func(*sigma),mask=mask)*data.max()/65535.0
        residual=data-background
        #Reject outliers
        residual=residual-np.median(residual[np.where(mask)])
        mask[np.abs(residual)>reject_threshold*np.std(residual[np.where(mask)])]=0

    #Final background
    background=background_func(skimage.img_as_uint(data/data.max()),morphology_func(*sigma),mask=mask)*data.max()/65535.0

    #Put nans in background where the median filter has failed
    background[np.where((background==0.0)&(data!=0.0)&(~mask))]=np.nan

    if interp_nonfinite:
        #If requested fill in nonfinite values with background smoothed with gaussian width increased by a factor of 2. 
        #Fill values with 0.0 if gaussian smoothing factor is larger than the scan size.
        nonfinite_values=~np.isfinite(background)
        denominator=5
        while np.any(nonfinite_values):
            denominator-=1
            if denominator==0:
            #If we really can't interpolate any values- just set nonfinite values to zero
                background[nonfinite_values]=0.0
                break
            sigma=np.array([data.shape[0]//denominator,data.shape[1]//denominator])
            weight=ndimage.gaussian_filter(mask,sigma,mode='constant',cval=0.0)
            interp_background=ndimage.gaussian_filter(data*mask,sigma,mode='constant',cval=0.0)/weight
            background[nonfinite_values]=interp_background[nonfinite_values]
            nonfinite_values=~np.isfinite(background)
    return background

def getbackground(data,in_flags=None,iterations=3,spike_width_time=10,spike_width_freq=10,reject_threshold=2.0,interp_nonfinite=True):
    """Determine a smooth background through a 2d data array by iteratively smoothing
    the data with a gaussian
    """
    #Make mask array
    mask=np.ones(data.shape,dtype=np.float)
    #Mask input flags if provided
    if in_flags is not None:
        mask[in_flags]=0.0
    #Filter Brightest spikes
    for i in range(2):
        median = np.nanmedian(data[~in_flags])
        mask[data-median > reject_threshold*3*np.nanstd(data[~in_flags])]=0.0
    #Next convolve with Gaussians with increasing width from iterations*spike_width to 1*spike_width
    for extend_factor in range(iterations,0,-1):
        #Convolution sigma
        sigma=np.array([min(spike_width_time*extend_factor,max(data.shape[0]//10,1)),min(spike_width_freq*extend_factor,data.shape[1]//10)])
        #sigma=np.array([1,min(spike_width_freq*extend_factor,data.shape[1]//10)])
        #Get weight and background convolved in time axis
        weight=ndimage.gaussian_filter(mask,sigma,mode='constant',cval=0.0)
        #Smooth background and apply weight
        background=ndimage.gaussian_filter(data*mask,sigma,mode='constant',cval=0.0)/weight
        residual=data-background
        #Reject outliers
        residual=residual-np.median(residual[np.where(mask)])
        mask[np.abs(residual)>reject_threshold*np.std(residual[np.where(mask)])]=0.0
    weight=ndimage.gaussian_filter(mask,sigma,mode='constant',cval=0.0)
    background=ndimage.gaussian_filter(data*mask,sigma,mode='constant',cval=0.0)/weight

    if interp_nonfinite:
        #If requested fill in nonfinite values with background smoothed with gaussian width increased by a factor of 2. 
        #Fill values with 0.0 if gaussian smoothing factor is larger than the scan size.
        nonfinite_values=~np.isfinite(background)
        denominator=5
        while np.any(nonfinite_values):
            denominator-=1
            if denominator==0:
            #If we really can't interpolate any values- just set nonfinite values to zero
                background[nonfinite_values]=0.0
                break
            sigma=np.array([data.shape[0]//denominator,data.shape[1]//denominator])
            weight=ndimage.gaussian_filter(mask,sigma,mode='constant',cval=0.0)
            interp_background=ndimage.gaussian_filter(data*mask,sigma,mode='constant',cval=0.0)/weight
            background[nonfinite_values]=interp_background[nonfinite_values]
            nonfinite_values=~np.isfinite(background)
    return background


def getbackground_opening_filter():
    """ Determine the background in a 1d array of data using an opening filter.
        This is the process of "erosion" - filtering the array by the minimum
        of the elements in a small area around a given element
        followed by the process of "dilation" -filtering the array by the minimum
        of the elements in a small area around a given element
        the size of the area chosen should correspond to the expected spike width
        in the data.
    """

    return(thisfitted_data)

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

def get_scan_flags(flagger,data,flags):
    """
    Function to run the flagging for a single scan. This is used by multiprocessing
    to avoid pickling problems therein. It can also be used to run the flagger
    independently of multiprocessing.
    Inputs:
        flagger: A flagger object with a method for running the flagging
        data: a 2d array of data to flag
        flags: a 2d array of prior flags
    Outputs:
        a 2d array of derived flags for the scan
    """
    return flagger._detect_spikes_sumthreshold(data,flags)

class sumthreshold_flagger():
    def __init__(self,background_iterations=1, spike_width_time=10, spike_width_freq=10, outlier_sigma=4.5, background_reject=2.0, 
                window_size=[1,3,5,7,9,11], average_time=1, average_freq=1, debug=False):
        self.background_iterations=background_iterations
        self.spike_width_time=spike_width_time
        self.spike_width_freq=spike_width_freq
        self.outlier_sigma=outlier_sigma
        self.background_reject=background_reject
        self.window_size=window_size
        self.average_time=average_time
        self.average_freq=average_freq
        self.debug=debug
        #Internal parameters
        #Fraction of data flagged to extend flag to all data
        self.flag_all_time_frac = 0.6
        self.flag_all_freq_frac = 0.8
        #Extend size of flags in time and frequency
        self.time_extend = 3
        self.freq_extend = 3
        #How many subbands to flag in frequency??
        self.num_freq_chunks = 4
        #Falloff exponent for sumthreshold
        self.rho = 1.3


    def get_flags(self,data,flags=None,num_cores=6):
        if self.debug: start_time=time.time()
        if flags is None:
            in_flags = np.repeat(None,in_data.shape[0]).reshape((in_data.shape[0]))
        out_flags=np.empty(data.shape,dtype=np.bool)
        async_results=[]
        p=mp.Pool(num_cores)
        for i in range(data.shape[-1]):
            async_results.append(p.apply_async(get_scan_flags,(self,data[...,i],flags[...,i],)))
        p.close()
        p.join()
        for i,result in enumerate(async_results):
            out_flags[...,i]=result.get()     
        if self.debug: 
            end_time=time.time()
            print "TOTAL SCAN TIME: %f"%((end_time-start_time)/60.0)
        return out_flags

    def _average(self,data,flags):
        #Only works if self.average_time and self.average_freq divide into data.shape
        new_time_axis = data.shape[0]//self.average_time
        new_freq_axis = data.shape[1]//self.average_freq
        bin_area = self.average_time*self.average_freq
        avg_data = data.reshape(new_time_axis,self.average_time,new_freq_axis,self.average_freq)
        avg_flags = flags.reshape(new_time_axis,self.average_time,new_freq_axis,self.average_freq)
        avg_data = np.nansum(np.nansum(avg_data*(~avg_flags),axis=3),axis=1)
        avg_flags = np.nansum(np.nansum(avg_flags,axis=3),axis=1)
        bin_weight = (bin_area - avg_flags)
        avg_flags = (avg_flags == bin_area)
        #Avoid NaNs where all data is flagged (data will become zero)
        bin_weight[avg_flags==True] = 1
        avg_data /= bin_weight
        return avg_data, avg_flags

    def _detect_spikes_sumthreshold(self,in_data,in_flags):
        if self.debug: start_time=time.time()
        #Create flags array
        if in_flags is None:
            flags = np.zeros(data.shape, dtype=np.bool)
        if self.average_time > 1 or self.average_freq > 1:
            data, flags = self._average(in_data,in_flags)

        freq_chunk_overlap = max(self.window_size*2)
        chunk_size = int(np.ceil(data.shape[1]/float(self.num_freq_chunks)))

        for chunk_num in range(self.num_freq_chunks):
            chunk_start = chunk_size*chunk_num
            chunk=slice(chunk_start,chunk_start+chunk_size+freq_chunk_overlap)
            data_chunk = data[:,chunk]
            flags_chunk = flags[:,chunk]
            background_chunk = getbackground(data_chunk,in_flags=flags_chunk,iterations=self.background_iterations,spike_width_time=self.spike_width_time, \
                                                spike_width_freq=self.spike_width_freq,reject_threshold=self.background_reject,interp_nonfinite=False)
            if self.debug: back_time=time.time()
            flags_chunk = flags_chunk | ~np.isfinite(background_chunk)
            #Subtract background
            av_dev = data_chunk-background_chunk
            #Sumthershold along time axis
            flags_chunk = self._sumthreshold(av_dev,flags_chunk,0,self.window_size,self.outlier_sigma)
            #Sumthreshold along frequency axis
            flags_chunk = self._sumthreshold(av_dev,flags_chunk,1,self.window_size,self.outlier_sigma)
            flags[:,chunk] = flags_chunk
        #Extend flags in freq and time
        if self.freq_extend > 1:
            flags = ndimage.convolve1d(flags, [True]*self.freq_extend, axis=1, mode='reflect')
        if self.time_extend > 1:
            flags = ndimage.convolve1d(flags, [True]*self.time_extend, axis=0, mode='reflect')
        #Flag all freqencies and times if too much is flagged.
        flags[:,np.where(np.sum(flags,dtype=np.float,axis=0)/flags.shape[0] > self.flag_all_time_frac)[0]]=True
        flags[np.where(np.sum(flags,dtype=np.float,axis=1)/flags.shape[1] > self.flag_all_freq_frac)]=True
        if self.average_freq > 1:
            flags=np.repeat(flags,self.average_freq,axis=1)
        if self.average_time > 1:
            flags=np.repeat(flags,self.average_time,axis=0)
        if self.debug:
            end_time=time.time()
            print "Shape %d x %d, BG Time %f, ST Time %f, Tot Time %f"%(data.shape[0],data.shape[1],back_time-start_time, end_time-back_time, end_time-start_time)
        return flags


    def _sumthreshold(self,input_data,flags,axis,window_bl,sigma):
        sd_mask = (input_data==0.)|(flags)
        #Get standard deviations along the axis using MAD
        estm_stdev = 1.4826 * np.ma.median(np.ma.masked_array(np.abs(input_data),mask=sd_mask),axis=axis)
        # Identify initial outliers (again based on normal assumption), and replace them with local median
        threshold = sigma * estm_stdev
        for window in window_bl:
            if window>input_data.shape[axis]: break
            #Set up 'this_data' from the averaged background subtracted buffer
            bl_data = np.abs(input_data)
            #The threshold for this iteration is calculated from the initial threshold
            #using the equation from Offringa (2010).
            thisthreshold = np.expand_dims(threshold / pow(self.rho,(math.log(window)/math.log(2.0))), axis).repeat(bl_data.shape[axis],axis=axis)
            #Set already flagged values to be the value of this threshold
            bl_data = np.where(flags,thisthreshold,bl_data)
            #Calculate a rolling average array from the data with a windowsize for this iteration
            avgarray = running_mean(bl_data, window, axis=axis)
            #Work out the flags from the convolved data using the current threshold.
            #Flags are padded with zeros to ensure the flag array (derived from the convolved data)
            #has the same dimension as the input data.
            this_flags = np.abs(avgarray) > np.expand_dims(np.take(thisthreshold,0,axis),axis)
            #Convolve the flags to be of the same width as the current window.
            convwindow = np.ones(window, dtype=np.bool)
            this_flags = np.apply_along_axis(np.convolve, axis, this_flags, convwindow)
            #"OR" the flags with the flags from the previous iteration.
            flags = flags | this_flags
        return flags


def detect_spikes_mad(data,blarray=None,spike_width=20,outlier_sigma=3):
    """
    FUNCTION :  detect_spikes_mad
    Given an array "data" from a baseline determine flags using the "median absolute deviation"
    method. The data is median filtered (with a kernel defined by "spike_width")
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
        flags[...,bl_index] = outliers
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
        for dump in range(h5.shape[0]):
            flags[dump] = h5.flags(flags_to_show)[dump][0]
    #Squeeze here removes stray axes left over by LazyIndexer
    if thisdata is None:
        thisdata = np.empty(h5.shape,dtype=np.float32)
        for dump in range(h5.shape[0]):
            thisdata[dump] = np.abs(h5.vis[dump][0])
    if norm_spec is not None: thisdata /= norm_spec[np.newaxis,:]
    #Get DC height (median rather than mean is more robust...)
    data = np.ma.MaskedArray(thisdata,mask=flags,copy=False).filled(fill_value=np.nan)
    offset = np.nanmedian(data,axis=1)
    #Remove the DC height
    weights = np.logical_not(flags).view(np.int8)
    data /= np.expand_dims(offset,axis=1)
    #Get the results for all of the data
    weightsum = weights.sum(axis=0,dtype=np.int).squeeze()
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

def plot_flag_data(label,hspectrum,hflagfrac,vspectrum,vflagfrac,freqs,pdf):
    """
    Produce a plot of the average spectrum in H and V 
    after flagging and attach it to the pdf output.
    Also show fraction of times flagged per channel.
    """
    from katsdpscripts import git_info

    repo_info = git_info() 

    #Set up the figure
    fig = plt.figure(figsize=(8.3,11.7))
    plt.suptitle(label,fontsize=14)
    outer_grid = gridspec.GridSpec(2,1)
    spectrum = {'HH': hspectrum, 'VV': vspectrum}
    flagfrac = {'HH': hflagfrac, 'VV': vflagfrac}
    for num,pol in enumerate(['HH','VV']):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[num-1], hspace=0.0)
        #Plot the spectrum for each target
        ax1 = plt.subplot(inner_grid[0])
        ax1.text(0.01, 0.90,repo_info, horizontalalignment='left',fontsize=10,transform=ax1.transAxes)
        ax1.set_title(pol +' polarisation')
        plt.plot(freqs,spectrum[pol])
        plot_RFI_mask(ax1)
        ticklabels=ax1.get_xticklabels()
        plt.setp(ticklabels,visible=False)
        ticklabels=ax1.get_yticklabels()
        plt.setp(ticklabels,visible=False)
        plt.xlim((min(freqs),max(freqs)))
        plt.ylabel('Mean amplitude\n(arbitrary units)')
        #Plot the average flags
        ax = plt.subplot(inner_grid[1],sharex=ax1)
        plt.plot(freqs,flagfrac[pol],'r-')
        plt.ylim((0.,1.))
        plot_RFI_mask(ax)
        plt.xlim((min(freqs),max(freqs)))
        plt.ylabel('Fraction flagged')
        ticklabels=ax.get_yticklabels()
        #Convert ticks to MHZ
        ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
        ax.xaxis.set_major_formatter(ticks)
        plt.xlabel('Frequency (Hz)')
    pdf.savefig(fig)
    plt.close('all')

def plot_waterfall_subsample(visdata, flagdata, freqs=None, times=None, label='', resolution=300):
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
    data = np.log10(np.abs(visdata[y_slice,x_slice][...,0]))
    flags = flagdata[y_slice,x_slice][...,0]
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
    plt.xlabel('Frequency (Hz)')
    return(fig)


def plot_waterfall(visdata,flags=None,channel_range=None,output=None):
    fig = plt.figure(figsize=(8.3,11.7))
    data=np.squeeze(np.abs(visdata[:,:]))
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
    arrayremove = int(len(ampsort)*(1.0 - 0.90)/2.0)
    lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time')
    if output==None:
        plt.show()
    else:
        plt.savefig(output)

def generate_flag_table(input_file,output_root='.',static_flags=None,use_file_flags=True,outlier_sigma=5.0,width_freq=3.0,width_time=15.0,max_scan=200,write_into_input=False,speedup=1,debug=False):
    """
    Flag the visibility data in the h5 file ignoring the channels specified in 
    static_flags, and the channels already flagged if use_file_flags=True.

    This will write a list of flags per scan to the output h5 file or overwrite 
    the flag table in the input file if write_into_input=True
    """

    cpu_count=mp.cpu_count()

    start_time=time.time()
    if write_into_input:
        output_file = os.path.join(output_root,input_file.split('/')[-1])
        if not os.path.exists(output_file) or not os.path.samefile(input_file,output_file):
            print "Copying input file from %s to %s"%(input_file,os.path.abspath(output_root),)
            shutil.copy(input_file,output_root)
        h5 = katdal.open(os.path.join(output_file),mode='r+')
        outfile = h5.file
        flags_dataset = h5._flags
    else:
        h5 = katdal.open(input_file)
        basename = os.path.join(output_root,os.path.splitext(input_file.split('/')[-1])[0]+'_flags')
        outfile=h5py.File(basename+'.h5','w')
        outfile.create_dataset('corr_products',data=h5.corr_products)
        flags_dataset = outfile.create_dataset('flags',h5._flags.shape,dtype=h5._flags.dtype)
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
    width_freq_channel = int(width_freq*1.e6/h5.channel_width/average_freq)
    width_time_dumps = int(width_time/h5.dump_period)

    #loop through scans
    num_bl = h5.shape[-1]
    cores_to_use = min(num_bl, (cpu_count-2))

    #Are we KAT7 or MeerKAT
    if h5.inputs[0][0]=='m':
        #MeerKAT
        flagger = sumthreshold_flagger(outlier_sigma=outlier_sigma,spike_width_freq=width_freq_channel,spike_width_time=width_time_dumps,average_freq=average_freq,debug=debug)
        cut_chans = h5.shape[1]//20
    else:
        #kat-7
        flagger = sumthreshold_flagger(outlier_sigma=outlier_sigma,background_reject=4.5,spike_width_freq=width_freq_channel,spike_width_time=width_time_dumps,average_freq=average_freq,debug=debug)
        cut_chans = h5.shape[1]//7
    #Make sure final size of array divides into averaging width
    remainder = (h5.shape[1]-2*cut_chans)%average_freq
    freq_range = slice(cut_chans-(remainder//2),h5.shape[1]-(cut_chans-(remainder-(remainder//2))))
    for scan, state, target in h5.scans():
        #Take slices through scan if it is too large for memory
        if h5.shape[0]>max_scan:
            scan_slices = [slice(i,i+max_scan,1) for i in range(0,h5.shape[0],max_scan)]
            scan_slices[-1] = slice(scan_slices[-1].start,h5.shape[0],1)
        else:
            scan_slices = [slice(0,h5.shape[0])]
        #loop over slices
        for this_slice in scan_slices:
            #this_data = np.abs(h5.vis)[this_slice,:,:]
            #Don't read all of the data in one hit- loop over timestamps instead
            this_data = np.empty((this_slice.stop-this_slice.start,freq_range.stop-freq_range.start,h5.shape[2],),dtype=np.float32)
            flags = np.zeros((this_slice.stop-this_slice.start,freq_range.stop-freq_range.start,h5.shape[2],),dtype=np.bool)
            for index,dump in enumerate(range(*this_slice.indices(h5.shape[0]))):
                this_data[index]=np.abs(h5.vis[dump,freq_range])
                if use_file_flags:
                    flags[index] = h5.flags('ingest_rfi')[dump,freq_range,:][0]
            #OR the mask flags with the flags already in the h5 file
            flags = np.logical_or(flags,mask_array[:,freq_range,:])
            detected_flags = flagger.get_flags(this_data,flags,num_cores=cores_to_use)
            del this_data
            #Flags are 8 bit:
            #1: 'reserved0' = 0
            #2: 'static' = 1
            #3: 'cam' = 2
            #4: 'reserved3' = 3
            #5: 'ingest_rfi' = 4
            #6: 'predicted_rfi' = 5
            #7: 'cal_rfi' = 6
            #8: 'reserved7' = 7
            #Add new flags to flag table from the mask and the detection
            flags = np.zeros((this_slice.stop-this_slice.start,h5.shape[1],h5.shape[2],),dtype=np.uint8)
            #Add mask to 'static' flags
            flags += mask_array.view(np.uint8)*2
            #Add detected flags to 'cal_rfi'
            flags[:,freq_range,:] += detected_flags.view(np.uint8)*(2**6)
            flags_dataset[h5.dumps[this_slice],:,:] += flags.squeeze()
    outfile.close()
    print "Flagging processing time: %4.1f minutes."%((time.time() - start_time)/60.0)
    return

def generate_rfi_report(input_file,input_flags=None,flags_to_show=None,output_root='.',antenna=None,targets=None,freq_chans=None,do_cross=True):
    """
    Create an RFI report- store flagged spectrum and number of flags in an output h5 file
    and produce a pdf report.

    Inputs
    ======
    input_file - input h5 filename
    input_flags - input h5 flags; will overwrite flags in h5 file- h5 file in format returnd from generate_flag_table
    flags_to_show - select which flag bits to plot. (None=all flags)
    output_root - directory where output is to be placed - defailt cwd
    antenna - which antenna to produce report on - default all in file
    targets - which target to produce report on - default all
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
        h5._flags = input_flags['flags']

    if freq_chans is None:
        # Default is drop first and last 5% of the bandpass
        start_chan = num_channels//20
        end_chan   = num_channels - start_chan
    else:
        start_chan = int(freq_chans.split(',')[0])
        end_chan = int(freq_chans.split(',')[1])
    chan_range = range(start_chan,end_chan+1)

    if targets is None: targets = h5.catalogue.targets

    for ant in ants:
        # Set up the output file
        basename = os.path.join(output_root,os.path.splitext(input_file.split('/')[-1])[0]+'_' + ant + '_RFI')
        pdf = PdfPages(basename+'.pdf')
        #Only select single pol
        corrprodselect=[[ant+'h']*2,[ant+'v']*2]
        h5.select(reset='TFB',corrprods=corrprodselect)
        vis=np.empty(h5.shape,dtype=np.float32)
        flags=np.empty(h5.shape,dtype=np.bool)
        #Get required vis and flags up front to avoid multiple reads of the data
        #vis=np.abs(h5.vis).squeeze()
        for dump in range(h5.shape[0]):
            vis[dump]=np.abs(h5.vis[dump][0])
            flags[dump]=h5.flags(flags_to_show)[dump][0]
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
            #Extract target and time range from file
            h5.select(targets=target,scans='~slew',ants=ant,channels=chan_range)
            if h5.shape[0]==0:
                print 'No data to process for .' + target
                continue
            label = 'Flag info for Target: ' + target + ', Antenna: ' + ant +', '+str(data_dict[target]['numrecords_tot'])+' records'
            #Get index of HH and VV data
            hh_index=np.all(h5.corr_products==ant+'h',axis=1)
            vv_index=np.all(h5.corr_products==ant+'v',axis=1)
            plot_flag_data(label,data_dict[target]['spectrum'][chan_range][:,hh_index],data_dict[target]['flagfrac'][chan_range][:,hh_index], \
                            data_dict[target]['spectrum'][chan_range][:,vv_index],data_dict[target]['flagfrac'][chan_range][:,vv_index],h5.channel_freqs,pdf)
            for pol in ['HH','VV']:
                h5.select(ants=ant,pol=pol)
                fig=plot_waterfall_subsample(vis[h5.dumps[:,np.newaxis],h5.channels],flags[h5.dumps[:,np.newaxis],h5.channels],h5.channel_freqs,None,label+'\n'+pol+' polarisation')
                pdf.savefig(fig)
            plt.close('all')

        #Reset the selection
        h5.select(reset='TFB',corrprods=corrprodselect,channels=chan_range)

        #Plot the flags for all data in the file
        label = 'Flag info for all data, Antenna: ' + ant +', '+str(data_dict['all_data']['numrecords_tot'])+' records'
        #Get index of HH and VV data
        hh_index=np.all(h5.corr_products==ant+'h',axis=1)
        vv_index=np.all(h5.corr_products==ant+'v',axis=1)
        plot_flag_data(label,data_dict['all_data']['spectrum'][chan_range,hh_index],data_dict['all_data']['flagfrac'][chan_range,hh_index], \
                        data_dict['all_data']['spectrum'][chan_range,vv_index],data_dict['all_data']['flagfrac'][chan_range,vv_index],h5.channel_freqs,pdf)
        for pol in ['HH','VV']:
            h5.select(ants=ant,pol=pol)
            fig=plot_waterfall_subsample(vis[:,h5.channels],flags[:,h5.channels],h5.channel_freqs,h5.timestamps,label+'\n'+pol+' polarisation')
            pdf.savefig(fig)
        plt.close('all')

        #close the plot
        pdf.close()

    #Report cross correlations if requested
    all_blines = [list(pair) for pair in itertools.combinations(ants,2) if do_cross]

    for bline in all_blines:
        # Set up the output file
        basename = os.path.join(output_root,os.path.splitext(input_file.split('/')[-1])[0]+'_' + ','.join(bline) + '_RFI')
        pdf = PdfPages(basename+'.pdf')
        corrprodselect=[[bline[0]+'h',bline[1]+'h'],[bline[0]+'v',bline[1]+'v']]
        h5.select(reset='TFB',corrprods=corrprodselect)
        #Get required vis and flags up front to avoid multiple reads of the data
        vis=np.abs(h5.vis).squeeze()
        flags=h5.flags(flags_to_show)[:]
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
            h5.select(reset='TFB',targets=target,scans='~slew',corrprods=corrprodselect,channels=chan_range)
            if h5.shape[0]==0:
                print 'No data to process for ' + target
                continue
            #Get HH and VV cross pol indices
            hh_index=np.all(np.char.endswith(h5.corr_products,'h'),axis=1)
            vv_index=np.all(np.char.endswith(h5.corr_products,'v'),axis=1)
            label = 'Flag info for Target: ' + target + ', Baseline: ' + ','.join(bline) +', '+str(data_dict[target]['numrecords_tot'])+' records'
            plot_flag_data(label,data_dict[target]['spectrum'][chan_range,hh_index],data_dict[target]['flagfrac'][chan_range,hh_index], \
                        data_dict[target]['spectrum'][chan_range,vv_index],data_dict[target]['flagfrac'][chan_range,vv_index],h5.channel_freqs,pdf)
            for pol in ['HH','VV']:
                h5.select(corrprods=corrprodselect,pol=pol)
                fig=plot_waterfall_subsample(vis[h5.dumps[:,np.newaxis],h5.channels],flags[h5.dumps[:,np.newaxis],h5.channels],h5.channel_freqs,None,label+'\n'+pol+' polarisation')
                pdf.savefig(fig)
        #Reset the selection
        h5.select(reset='TFB',corrprods=corrprodselect,channels=chan_range)

        #Plot the flags for all data in the file
        hh_index=np.all(np.char.endswith(h5.corr_products,'h'),axis=1)
        vv_index=np.all(np.char.endswith(h5.corr_products,'v'),axis=1)
        label = 'Flag info for all data, Baseline: ' + ','.join(bline) +', '+str(data_dict['all_data']['numrecords_tot'])+' records'
        plot_flag_data(label,data_dict['all_data']['spectrum'][chan_range,hh_index],data_dict['all_data']['flagfrac'][chan_range,hh_index], \
                        data_dict['all_data']['spectrum'][chan_range,vv_index],data_dict['all_data']['flagfrac'][chan_range,vv_index],h5.channel_freqs,pdf)
        for pol in ['HH','VV']:
            h5.select(corrprods=corrprodselect,pol=pol)
            fig=plot_waterfall_subsample(vis[:,h5.channels],flags[:,h5.channels],h5.channel_freqs,h5.timestamps,label+'\n'+pol+' polarisation')
            pdf.savefig(fig)
    
        #close the plot
        pdf.close()
