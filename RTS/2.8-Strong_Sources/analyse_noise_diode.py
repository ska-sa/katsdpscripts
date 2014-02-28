#!/usr/bin/python
# Script that will search through a .h5 file for noise diode firings and
# calculate change in mean counts in the data due to each noise diode firing.
# Output is written to a file which lists the target that is being observed during
# the noise diode firing and the timestamp of the scan and the noise diode
# jump in counts in the HH and VV polarisations.
#
# This is intended to be used for survivability and strong source tests
# changes in the mean value of noise diode jumps can indicate that the data
# is saturated.
#
# TM: 27/11/2013


import optparse

from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt

import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import math
import os

import scape
from scape.stats import robust_mu_sigma



def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>")
    parser.add_option("-a", "--antenna", type="string", default='sd', help="Antenna to load (e.g. 'A1' for antenna 1), default is first single-dish baseline in file.")
    parser.add_option("-f", "--freq-chans", help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass.")
    parser.add_option("-t", "--targets", default='all', help="List of targets to select (default is all)")
    (opts, args) = parser.parse_args()

    return vars(opts), args


def read_and_select_file(file, bline=None, channels=None, **kwargs):
    """
    Read in the input h5 file using scape and make a selection.

    file:   {string} filename of h5 file to open in katdal

    Returns:
        the visibility data to plot, the frequency array to plot, the flags to plot
    """

    data = scape.DataSet(file, baseline=bline, katfile=True)
    # Secect desired channel range
    # Select frequency channels and setup defaults if not specified
    num_channels = len(data.channel_select)
    if channels is None:
        # Default is drop first and last 20% of the bandpass
        start_chan = num_channels // 4
        end_chan   = start_chan * 3
    else:
        start_chan = int(channels.split(',')[0])
        end_chan = int(channels.split(',')[1])
    chan_range = range(start_chan,end_chan+1)
    data = data.select(freqkeep=chan_range)

    #return the selected data
    return data

#----------------------------------------------------------------------------------
#--- FUNCTION :  getbackground_spline
# Fit the background to the array in "data" using an iterative fit of a spline to the 
# data. On each iteration the number of density of knots in the spline is increased 
# up to "spike_width", and residual peaks more than 5sigma are removed.
#----------------------------------------------------------------------------------
def getbackground_spline(data,spike_width=5):

    """ From a 1-d data array determine a background iteratively by fitting a spline
    and removing data more than a few sigma from the spline """

    # Remove the first and last element in data from the fit.
    y=data[:]
    arraysize=y.shape[0]
    x=np.arange(arraysize)

    # Iterate 4 times
    for iteration in range(4):

        # First iteration fits a linear spline with 3 knots.
        if iteration==0:
            npieces=3
        # Second iteration fits a quadratic spline with 10 knots.
        elif iteration==1:
            npieces=10
        # Third and fourth iterations fit a cubic spline with 50 and 75 knots respectively.
        elif iteration>1:
            npieces=iteration*25
        deg=min(iteration+1,3)

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

def detect_spikes_sumthreshold(data, blarray=None, spike_width=3, outlier_sigma=4.0, window_size_auto=[1,3], window_size_cross=[2,4,8]):
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
    flags = np.zeros(list(data.shape), dtype=np.uint8)

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
            filtered_data = np.asarray(signal.medfilt(this_data_buffer, kernel_size), this_data_buffer.dtype)
            #Use the auto correlation window function
            #filtered_data = ndimage.grey_opening(this_data_buffer, (kernel_size,))
            #filtered_data = getbackground_spline(this_data_buffer, kernel_size)
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
        #outliers[0] = True 

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


def plot_RFI_mask(pltobj, extra=None, channelwidth=1e6):
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
                

def present_results(pdf, temperature, freq, targname, antenna, channelwidth):

    #Set up the figure
    fig = plt.figure(figsize=(8.3,8.3))

    #Flag the data
    flags = detect_spikes_sumthreshold(temperature)

    #Loop over polarisations
    temps=[]
    for polnum,thispol in enumerate(['HH','VV']):
        thisdata = temperature[:,polnum]
        thisflags = flags[:,polnum]
        systemp, err = robust_mu_sigma(thisdata[np.where(thisflags==False)])
        temps.append(systemp)
        ax = plt.subplot(211 + polnum)
        plt.title(targname + ', Antenna: ' + antenna + ', ' + thispol + ' pol')
        ax.text(0.05,0.8,'Tsys: %5.2f'%(systemp),transform=ax.transAxes)
        ax.plot(freq,thisdata)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('System Temperature (K)')
        plot_RFI_mask(ax, extra=freq[np.where(thisflags)], channelwidth=channelwidth)
        plt.xlim(freq[-1], freq[0])
        
    pdf.savefig()
    return temps

def plot_temps_time(alltimes,alltempshh,alltempsvv,antenna):

    fig=plt.figure(figsize=(8.3,8.3))

    ax = plt.subplot(211)
    plt.title("Tsys vs Time, Antenna" + antenna + ", HH polarisation")
    ax.plot(alltimes,alltempshh,'ro')
    plt.xlabel("Time since observation start (hours)")
    plt.ylabel("Tsys")
    plt.xlim(alltimes[0], alltimes[-1])
    ax = plt.subplot(212)
    plt.title("VV polarisation")
    ax.plot(alltimes,alltempsvv,'ro')
    plt.xlabel("Time since observation start (hours)")
    plt.ylabel("Tsys")
    plt.xlim(alltimes[0], alltimes[-1])

    pdf.savefig()


# Print out the 'on' and 'off' values of noise diode firings from an on->off transition to a text file.
opts, args = parse_arguments()

#get the targets
targets=opts.get('targets','all')

# Get data from h5 file and use 'select' to obtain a useable subset of it.
data = read_and_select_file(args[0], bline=opts.get('antenna',None), channels=opts.get('freq_chans',None))
pdf = PdfPages(os.path.splitext(os.path.basename(args[0]))[0] +'_SystemTemp_'+data.antenna.name+'.pdf')
# loop through compscans in file and get noise diode firings
#nd_data = extract_cal_dataset(data)

#Convert the data to kelvin using the noise diode firings
data.convert_power_to_temperature()

#average each required scan in time sensibly and plot the data
alltempshh,alltempsvv,alltimes=[],[],[]
zerotime = data.scans[0].timestamps[0]
for scan in data.scans:
    if (targets == 'all') or (scan.compscan.target.name in targets):
        average_spec, sigma_spec = robust_mu_sigma(scan.data[np.where(scan.flags['nd_on']==False)], axis=0)
        systemp=present_results(pdf, average_spec[:,:2], data.freqs*1.e6, scan.compscan.target.name, data.antenna.name, data.bandwidths[0]*1e6)
        alltempshh.append(systemp[0])
        alltempsvv.append(systemp[1])
        alltimes.append((np.mean(scan.timestamps[np.where(scan.flags['nd_on']==False)])-zerotime)/(60*60))      

plot_temps_time(alltimes,alltempshh,alltempsvv,data.antenna.name)

pdf.close()
