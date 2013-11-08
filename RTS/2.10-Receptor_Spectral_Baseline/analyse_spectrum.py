#!/usr/bin/python

import optparse

import numpy as np

import katdal
from katdal import averager

import matplotlib.pyplot as plt


def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>")
    parser.add_option("-p", "--polarisation", type="string", default="I", help="Polarisation to produce spectrum, options are I, HH, VV, HV, VH. Default is I.")
    parser.add_option("-b", "--baseline", type="string", default=None, help="Baseline to load (e.g. 'ant1,ant1' for antenna 1 auto-corr), default is first single-dish baseline in file.")
    parser.add_option("-t", "--target", type="string", default=None, help="Target to plot spectrum of, default is the first target in the file.")
    parser.add_option("-f", "--freqaverage", type="float", default=None, help="Frequency averaging interval in MHz. Default is a bin size that will produce 100 frequency channels.")
    parser.add_option("-a", "--timeaverage", type="float", default=None, help="Time averageing interval in minutes. Default is the shortest scan length on the selected target.")
    (opts, args) = parser.parse_args()

    return vars(opts), args


def rolling_window(a, window,axis=-1,pad=False,mode='reflect',**kwargs):
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
        **kwargs are passed to the function numpy.pad
        
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
        a = np.pad(a,pad_width=pad_width,mode=mode,**kwargs)
    a1 = np.swapaxes(a,axis,-1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1],)
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2,axis) # Move original axis to 


def read_and_select_file(file, bline=None, target=None, polarisation=None, **kwargs):
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
        ant1,ant2 = bline[:4],bline[5:]
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

    data.select(strict=False, reset='', **select_data)
    #return the selected data
    return data

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
        dumpav = short_scan
        timeav = dumpav*(data.dump_period/60.0)
    print "Averaging %d dumps to %4.1fmin intervals."%(dumpav,timeav)

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
    vis_data = []
    flag_data = []

    #Extract the required arrays from the data object for the averager on a scan by scan basis
    for scan, state, target in data.scans():
        scan_vis_data = data.vis[:]
        scan_weight_data = data.weights()[:]
        scan_flag_data = data.flags()[:]
        scan_timestamps = data.timestamps[:]
        scan_channel_freqs = data.channel_freqs[:]
        #Average
        scan_vis_data, scan_weight_data, scan_flag_data, scan_timestamps, scan_channel_freqs = averager.average_visibilities(scan_vis_data, scan_weight_data, scan_flag_data, scan_timestamps, 
                                                                                                    scan_channel_freqs, timeav=dumpav, chanav=chanav, flagav=True)        
        if scan_vis_data.shape[0] == 1:
            vis_data.append(scan_vis_data[0])
            flag_data.append(scan_flag_data[0])
        else:
            vis_data.append(scan_vis_data)
            flag_data.append(scan_flag_data)
        channel_freqs = scan_channel_freqs

    return np.array(vis_data), np.array(channel_freqs), np.array(flag_data)

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

def plot_results(visdata,freqdata,flagdata):

    #Get flag frequencies
    #print flagdata.shape,np.sum(flagdata, axis=0)


    #get standard deviation of visdata
    vis_std = np.std(np.abs(visdata), axis=0)
    vis_median = np.average(np.abs(visdata), axis=0)
    #plt.plot(freqdata,vis_std)
    plt.plot(freqdata,vis_std/vis_median)
    #plot_RFI_mask(plt)

    plt.show()

opts, args = parse_arguments()

data = read_and_select_file('1374108772.h5', bline=opts.get('baseline',None), target=opts.get('target',None), polarisation=opts.get('polarisation',None))

visdata, freqdata, flagdata = extract_and_average(data, timeav=opts.get('timeaverage',None), freqav=opts.get('freqaverage',None))

if opts.get('polarisation',None) == 'I':
    visdata = visdata[:,:,0] + visdata[:,:,1]
    flagdata = np.logical_and(flagdata[:,:,0],flagdata[:,:,1])

plot_results(visdata,freqdata,flagdata)