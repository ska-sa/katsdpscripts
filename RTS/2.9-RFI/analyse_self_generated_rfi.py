import katfile 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid import Grid

import matplotlib.pyplot as plt; plt.ioff()
import numpy as np
import optparse
import scipy.signal as signal
import scipy.interpolate as interpolate

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
    y=np.copy(data[1:-1])
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


def detect_spikes_sumthreshold(data, bline, spike_width=3, outlier_sigma=11.0, window_size_auto=[1,3], window_size_cross=[2,4,8]):
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
        bl_name = bline.bls_ordering[bl_index]
        
        # Check if this is an auto or a cross...
        if bl_name[0][:-1] == bl_name[1][:-1]:
            #Auto-Correlation.
            filtered_data = np.asarray(signal.medfilt(this_data_buffer, kernel_size), this_data_buffer.dtype)
            #Use the auto correlation window function
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
            thisthreshold = threshold / pow(1.2,(log(window)/log(2.0)))
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

def plot_selection_per_antenna(h5, pol, chan_range,detectfunc=None):
    """FUNCTION :  plot_selection_per_antenna
    """
    target_list = np.copy(h5.target_indices)
    #h5.select(corrprods='cross', pol=pol, channels=chan_range,scans='~slew')
    #crossdata = h5.vis[:,:,:].mean(axis=0)
    if detectfunc is None : detectfunc=detect_spikes_mad
    #external_rfi =  detectfunc(crossdata).mean(axis=-1)# = 1 will be rfi on all baselines (occupancy)
    h5.select(corrprods='auto', pol=pol, channels=chan_range,scans='~slew') 
    autodata = np.zeros(((len(target_list),)+h5.shape[1:3] ))
    spikes = np.zeros(((len(target_list),)+h5.shape[1:3] ))
    text = []
    for i,target  in enumerate(target_list) :
        h5.select(corrprods='auto', pol=pol,targets=target, channels=chan_range,scans='~slew') 
        if h5.shape[0] >  1 :#more than one timestamp 
            autodata[i,:,:] =np.abs(h5.vis[:,:,:].mean(axis=0))
            spikes[i,:,:] =  detectfunc(autodata[i,...]) #internal + external
    rfiflag =  spikes[...].mean(axis=0) > 0.8 
    figlist = []
    for i,ant in  enumerate(h5.ants):
        text.append("\n Flagged Frequencys %s:"%(ant.name))
        d1 = h5.channel_freqs[:]
        for j in xrange(h5.channel_freqs[:].shape[0]):
            d2 = spikes[:,:,i].mean(axis=0)
            if d2[j] > 0.8 :
                text.append('     %f MHz , Percentage of pointings contaminated is %.3f  ' %(d1[j]/1e6,d2[j]*100) )

        fig, ax = plt.subplots()
        ax.plot(h5.channel_freqs,np.abs(autodata[:,:,i].mean(axis=0)))
        plot_RFI_mask(ax,extra=h5.channel_freqs[ rfiflag[:,i]],channelwidth=h5.channel_width)
        if pol == 'H':
            fig.suptitle('Mean Horizontal auto-correlation spectra per Antenna',size = 'small', fontweight='bold')
        elif pol == 'V':
            fig.suptitle('Mean Vertical auto-correlation spectra per Antenna',size = 'small', fontweight='bold')
        #ylim=(0,1.2*d[:,index].max())
        #xlim=(freqs[0],freqs[-1])
        figlist.append(fig)
    return (text, figlist)

#-------------------------------
#--- FUNCTION :  plot_all_antenas_selection_per_pointing
#-------------------------------
def plot_all_antenas_selection_per_pointing(fileopened, pol, antennas, chan_range, targets):
    fig = plt.figure()
    if pol == 'H':
        fig.suptitle('All antennas mean horizontal auto-correlation spectra per pointing',size = 'small', fontweight='bold')
    elif pol == 'V':
        fig.suptitle('All antennas mean vertical auto-correlation spectra per pointing',size = 'small', fontweight='bold')
    grid = Grid(fig, 111,  nrows_ncols =(4,5), axes_pad=0.0, share_all=True)
    all_text = []
    for index, targ in enumerate(targets):
        fileopened.select(corrprods='auto', pol=pol,targets=targ,channels=chan_range, scans='~slew')
        freqs = fileopened.channel_freqs*1.0e-6
        data = np.abs(fileopened.vis[:].mean(axis=0))
        ylim=(0,1.2*data.max())
        xlim=(freqs[0],freqs[-1])
        spikes = detect_spikes_orig(data)
        #detect spikes seen in all antennas per each pointing
        rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
        label = "Flags [MHz]:\n"
        text = targ+'\n'+label+'\n'.join(['%.3f' % num for num in rfi_inall_ants])
        all_text.append(text)
        at = AnchoredText(text,prop=dict(size=4), frameon=True,loc=2)
        grid[index].add_artist(at)
        #print targ, rfi_inall_ants
        for k,ant in enumerate(antennas):
            #detect spikes per antenna for each pointing
            rfi_freqs = [freqs[i] for i,elem in enumerate(spikes[:,k]) if elem]
            rfi_power = [data[:,k][i] for i,elem in enumerate(spikes[:,k]) if elem]
            grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
        grid[index].plot(freqs,data)
        grid[index].add_artist(at)
        plt.setp(grid[index],xticks=[],yticks=[], ylim=ylim, xlim=xlim)
    return ('\n'.join(all_text), fig)

#-------------------------------
#--- FUNCTION :  plot_selection_per_pointing
#-------------------------------
def plot_selection_per_pointing(fileopened, pol, antennas, chan_range, targets):
    # Horizontal selection per pointing
    for ant in antennas:
        fig = plt.figure()
        if pol == 'H':
            fig.suptitle(' '.join([ant.capitalize(),'Mean Horizontal auto-correlation spectra per pointing']),size = 'small', fontweight='bold')
        elif pol == 'V':
            fig.suptitle(' '.join([ant.capitalize(),'Mean Vertical auto-correlation spectra per pointing']),size = 'small', fontweight='bold')
        fig.text(0.5, 0.04, 'Frequency [MHz]', ha='center', va='center',size='x-small', fontweight='bold',style='italic')
        fig.text(0.06, 0.5, 'Power [Units]', ha='center', va='center', size='x-small',rotation='vertical', fontweight='bold',style='italic')
        grid = Grid(fig, 111, (4, 5), axes_pad=0.0, share_all=True)
        all_text = []
        for index,targ in enumerate(targets):
            fileopened.select(ants=ant,corrprods='auto',pol='H',targets=targ,channels=chan_range,scans ='~slew')
            data=np.abs(fileopened.vis[:].mean(axis=0))
            freqs = fileopened.channel_freqs*1.0e-6
            ylim=(0,1.2*data.max())
            xlim=(freqs[0],freqs[-1])
            #at = AnchoredText(targ,prop=dict(size=5), frameon=False,loc=1)
            spikes = detect_spikes_orig(data)
            rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
            rfi_freqs = [freqs[i] for i,elem in enumerate(spikes,0) if elem]
            rfi_power = [data[i] for i,elem in enumerate(spikes,0) if elem]
            grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
            grid[index].plot(freqs,data)
            #A potential bug here - at has not yet been defined
            #grid[index].add_artist(at)
            plt.setp(grid[index],xticks=[],yticks=[],ylim=ylim,xlim=xlim)
            label = "Flags [MHz]:\n"
            text = targ + '\n'+ label +'\n'.join(['%.3f' % num for num in rfi_inall_ants])
            all_text.append(text)
            at = AnchoredText(text,prop=dict(size=3), frameon=True,loc=2)
            grid[index].add_artist(at)
        # pdf.savefig(fig)
        yield ('\n'.join(all_text), fig)

#command-line parameters
parser = optparse.OptionParser(usage="Please specify the input file (Yes, this is a non-optional option)\n\
    USAGE: python analyse_self_generated_rfi.py <inputfile.h5> ",
    description="Evaluate the auto & cross correlation spectra to Find the RFI spikes\
    that appear consistently in all observations/pointings.")

opts, args = parser.parse_args()

# if no enough arguments, raise the runtimeError
if len(args) < 1:
    raise RuntimeError("No File passed as argument to script")

filename = args[0]
h5 = katfile.open(filename)

# user defined variables
print("Please wait while analysis is in progress...")
pdf = PdfPages(filename.split('/')[-1]+'_RFI.pdf')


#antennas = [ant.name for ant in fileopened.ants]
#targets = [('%s' % (i.name)) for i in fileopened.catalogue.targets]
chan_range = slice(1,-1) # remove dc spike
#freqs = fileopened.channel_freqs*1.0e-6
detection_function = detect_spikes_orig
#plot_horizontal_selection_per_antenna
for pol in ['H','V']: 
    (all_text, figlist) = plot_selection_per_antenna(h5, pol, chan_range,)
    line = 0
    for page in xrange(int(np.ceil(len(all_text)/70.))):
        fig = plt.figure(None,figsize = (10,16)) 
        #fig, ax = plt.subplots(figsize = (10,16))
        lineend = line+np.min((70,len(all_text[line:])))
        factadj = 0.8*(1-(lineend-line)/70.)
        plt.figtext(0.1 ,0.15+factadj,'\n'.join(all_text[line:lineend]),fontsize=10)
        line = line + lineend
        fig.savefig(pdf,format='pdf')
        plt.close(fig)

    for fig in figlist:
        pdf.savefig(fig)
        plt.close(fig)

#for pol in ['H','V']: 
#    (all_text, fig) = plot_all_antenas_selection_per_pointing(fileopened, pol, antennas, chan_range, targets,detection_function)
#    pdf.savefig(fig)
#    plt.close(fig)
#for pol in ['H','V']:
#    for (all_text, fig) in plot_selection_per_pointing(fileopened, pol, antennas, chan_range, targets,detection_function):
#        #text_output.write(all_text)
#        pdf.savefig(fig)
#        plt.close(fig)
pdf.close()

print ("Done!")
print("Open the file %s" % (filename.split('/')[-1]+'_RFI.pdf'))
