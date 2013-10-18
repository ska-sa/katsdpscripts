from katfile import open as kfopen
from math import log
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid import Grid

import matplotlib.pyplot as plt; plt.ioff()
import numpy as np
import optparse
import os
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

#----------------------------------------------------------------------------------
#--- FUNCTION :  detect_spikes_sumthreshold
# Given an array "data" from a baseline:
# - determine if the data is an auto-correlation or a cross-correlation.
# - Get the background in the data using a median filter for auto_correlations or a 
#   cubic spline for cross correlations.
# - Make an array of flags from the data using the "sumthreshold" method and return
#   this array of flags.
# Parameters
# ----------
# data : array-like
#     An N-dimensional numpy array containing the data to flag
# blarray : array-like
#     An array of baseline labels used to determine if the 
#     baseline index is an auto- or a cross- correlation. Should have the same
#     shape as the baseline part of the data array.
# spike_width : integer
#     The width of the median filter and the gaps between knots in the spline fit.
# outlier_sigma : float
#     The number of sigma in the first iteration of the sumthreshold method.
# buffer_size : int
#     The number of timestamps in the data array to average.
# window_size_auto : array of ints
#     The sizes of the averaging windows in each sumthreshold iteration for auto-correlations.
# window_size_cross : array of ints 
#     The sizes of the averaging windows in each sumthreshold iteration for cross-correlations.
#
#----------------------------------------------------------------------------------
def detect_spikes_sumthreshold(data, bline, spike_width=3, outlier_sigma=11.0, window_size_auto=[1,3], window_size_cross=[2,4,8]):


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


#----------------------------------------------------------------------------------
#--- FUNCTION :  detect_spikes_mad
# Given an array "data" from a baseline determine flags using the "median absolute
# deviation" method. The data is median filtered (with a kernel defined by "spike_width")
# to find a background and then rfi spikes are found by finding peaks that are 
#"outlier_sigma" from the median of the absolute values of the background subtracted data.
# Parameters
# ----------
# data : array-like
#     An N-dimensional numpy array containing the data to flag
# blarray : CorrProdRef object
#     Baseline labels used to determine if the 
#     baseline index is an auto- or a cross- correlation.
# spike_width : integer
#     The width of the median filter and the gaps between knots in the spline fit.
# outlier_sigma : float
#     The number of sigma in the first iteration of the sumthreshold method.
#
#----------------------------------------------------------------------------------

def detect_spikes_mad(data,blarray,spike_width=6,outlier_sigma=11):
    
    flags = np.zeros(list(data.shape, dtype=np.uint8))

    for bl_index in range(data.shape[-1]):
        spectral_data = np.abs(data[:,bl_index])
        spectral_data = np.atleast_1d(spectral_data)
        kernel_size = 2 * max(int(spike_width), 0) + 1
        
        # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
        filtered_data = np.asarray(signal.medfilt(spectral_data, kernel_size), spectral_data.dtype)
        
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
        
#-----------------------------------------------------------------------------------
#--- FUNCTION :  detect_spikes_median
# Given an array "data" from a baseline determine flags using a simple median filter.
# Parameters
# ----------
# data : array-like
#     An N-dimensional numpy array containing the data to flag
# blarray : CorrProdRef object
#     Baseline labels used to determine if the 
#     baseline index is an auto- or a cross- correlation.
# spike_width : integer
#     The width of the median filter and the gaps between knots in the spline fit.
# outlier_sigma : float
#     The number of sigma in the first iteration of the sumthreshold method.
#
#----------------------------------------------------------------------------------

# BUG: line 277 spectral_data called before it is defined.
# def detect_spikes_median(data,blarray,spike_width=3,outlier_sigma=11.0):
    
#     """Simple RFI flagging through thresholding.

#     Trivial thresholder that looks for n sigma deviations from the average
#     of the supplied frame.

#     Parameters
#     ----------
#     n_sigma : float
#        The number of std deviations allowed

#     """
        
#     flags = np.zeros(list(data.shape), dtype=np.uint8)
#     for bl_index in range(data.shape[-1]):
#         spectral_data = np.atleast_1d(spectral_data)
        
#         kernel_size=spike_width
        
#         # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
#         filtered_data = signal.medfilt(spectral_data, kernel_size)
#         # The deviation is measured relative to the local median in the signal
#         abs_dev = spectral_data - filtered_data
#         # Identify outliers (again based on normal assumption), and replace them with local median
#         outliers = (abs_dev > np.std(abs_dev)*2.3)
#         flags[:,bl_index] = outliers

#     return flags

#-------------------------------
#--- FUNCTION :  detect_spikes
#-------------------------------
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
    flags = np.zeros(data.shape, dtype='int32')
    #making sure that the data is already 1-D
    spectral_data = np.atleast_1d(data)
    kernel_size = 2 * max(int(spike_width), 0) + 1
    # Median filter data along the desired axis, with given kernel size
    kernel = np.ones(spectral_data.ndim, dtype='int32')
    kernel[axis] = kernel_size
    # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
    filtered_data = np.asarray(signal.medfilt(spectral_data, kernel), spectral_data.dtype)
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

#-------------------------------
#--- FUNCTION :  plot_selection_per_antenna
#-------------------------------
def plot_selection_per_antenna(fileopened, pol, antennas, chan_range, targets):
    fileopened.select(corrprods='auto', pol=pol, channels=chan_range,scans='~slew')
    d = np.abs(fileopened.vis[:].mean(axis=0))
    spikes = detect_spikes_orig(d)
    freqs = fileopened.channel_freqs*1.0e-6
    #detects all the spikes seen by all antennas irrespective of pointing
    rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
    fig = plt.figure()
    if pol == 'H':
        fig.suptitle('Mean Horizontal auto-correlation spectra per Antenna',size = 'small', fontweight='bold')
    elif pol == 'V':
        fig.suptitle('Mean Vertical auto-correlation spectra per Antenna',size = 'small', fontweight='bold')
    grid = Grid(fig, 111, nrows_ncols=(3,2), axes_pad=0.0, share_all=True)
    all_text = []
    for index,ant in enumerate(antennas):
        antenna = ant +'\n'
        ylim=(0,1.2*d[:,index].max())
        xlim=(freqs[0],freqs[-1])
        spikes = detect_spikes_orig(d[:,index])
        rfi_freqs = [freqs[i] for i,elem in enumerate(spikes,0) if elem]
        rfi_power = [d[:,index][i] for i,elem in enumerate(spikes,0) if elem]
        label = "Flags [MHz]:\n"
        text = antenna +'\n'+label+'\n'.join(['%.3f' % num for num in rfi_freqs])
        all_text.append(text)
        at = AnchoredText(text,prop=dict(size=3), frameon=True,loc=2)
        grid[index].add_artist(at)
        grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
        grid[index].plot(freqs,d[:,index])
        grid[index].add_artist(at)
        plt.setp(grid[index],xlim=xlim,ylim=ylim,yticks=[],xticks=[])
    label = "Flags in all Ants [MHz]:\n"
    text = label +'\n'.join(['%.3f' % num for num in rfi_inall_ants])
    all_text.append(text)
    at = AnchoredText(text,prop=dict(size=4), frameon=True,loc=2)
    grid[-1].add_artist(at)
    return ('\n'.join(all_text), fig)

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
    raise RuntimeError(parser.usage)

fileopened = kfopen(args[0])

# user defined variables
print("Please wait while analysis is in progress...")
pdf = PdfPages(os.path.basename(args[0]).replace('h5','h5_RFI.pdf'))

#called extract_spectra_data()
# Here's the description in the doc string
# def extract_spectra_data():
#       Extract the horizontal and the vertical spectral data for all th antennas in the loaded hdf5 file and plot their
#       mean visibilies against channel_freqs. The plots are written and saved into the PDF file whose name has the form
#       xxxxxxx.h5_RFI.pdf (where xxxxx is the timestamps representing the KAT7 file names.)

antennas = [ant.name for ant in fileopened.ants]
targets = [('%s' % (i.name)) for i in fileopened.catalogue.targets]
chan_range = slice(10,-10)
#freqs = fileopened.channel_freqs*1.0e-6

text_output = open('rfi.new.txt', 'w')

#plot_horizontal_selection_per_antenna
(all_text, fig) = plot_selection_per_antenna(fileopened, 'H', antennas, chan_range, targets)
text_output.write(all_text)
pdf.savefig(fig)

#re-initialise the oppened file for new selection
fileopened.select()

(all_text, fig) = plot_selection_per_antenna(fileopened, 'V', antennas, chan_range, targets)
text_output.write(all_text)
pdf.savefig(fig)

#re-initialise the oppened file for new selection
fileopened.select()

(all_text, fig) = plot_all_antenas_selection_per_pointing(fileopened, 'H', antennas, chan_range, targets)
text_output.write(all_text)
pdf.savefig(fig)

fileopened.select()

(all_text, fig) = plot_all_antenas_selection_per_pointing(fileopened, 'V', antennas, chan_range, targets)
text_output.write(all_text)
pdf.savefig(fig)

fileopened.select()

# Horizontal selection per pointing
fileopened.select()
for (all_text, fig) in plot_selection_per_pointing(fileopened, 'H', antennas, chan_range, targets):
    text_output.write(all_text)
    pdf.savefig(fig)

# Vertital selection per pointing
fileopened.select()
for (all_text, fig) in plot_selection_per_pointing(fileopened, 'V', antennas, chan_range, targets):
    text_output.write(all_text)
    pdf.savefig(fig)

# put all the contaminated freqs all pointing (like summary)
pdf.close()
plt.close('all')

print "Done!"
print("Open the file %s" % (os.path.basename(args[0]).replace('h5','h5_RFI.pdf')))
