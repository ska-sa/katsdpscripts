###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
# imports
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid import Grid
import matplotlib.pyplot as plt
import scipy.signal as signal
from katfile import open
import numpy as np
import optparse
import os

#command-line parameters
parser = optparse.OptionParser(usage="%prog [options]",
    description="Evaluate the auto & cross correlation spectra to Find the RFI spikes\
    that appear consistently in all observations/pointings.")
parser.add_option('-o', '--output_file', type='string',
    help="A PDF file containing all the evaluated spectra (default=%default)")

opts, args = parser.parse_args()
usage = "Please specify the input file (Yes, this is a non-optional option)\n\
    USAGE: python analyse_self_generated_rfi.py <inputfile.h5> "
    
file_corrupted = "This normaly happened when the input file is corrupted.\n\
    The input file should be a xxxxxx.h5 (An example file would be: 1378901689.h5)."
# if no enough files, system exit
if len(args) < 1:
    raise SystemExit(usage)

# user defined variables
print("Plese wait while analysis in Progress...")
pdf = PdfPages(os.path.basename(args[0]).replace('h5','pdf'))
def load_data(fname):
    """
        load dataset using katfile
    """
    # Exception, catching all possible command-line errors (IOError, TypeError, NameError)
    try:
        f = open(fname)
    except (IOError, TypeError, NameError) as e:
        raise SystemExit(e)

    observer = ('Obsever: %s' % (f.observer))
    name = ('Filename: %s' % os.path.basename(f.name))
    fsize = ('Filesize: %.2f %s' % (f.size*1.0e-9, 'GB'))
    description = ('Description: %s' % (f.description))
    centre_freq = ('Centre Freq [MHz]: %s' % (f.spectral_windows[0].centre_freq*1e-6))
    dump = ('Dump Period: %0.4f' % f.dump_period)
    start_time = ('Start Time: %s' % (f.start_time))
    end_time = ('End Time: %s' % (f.end_time))
    targets = [('%s' % (i.name)) for i in f.catalogue.targets]
    ants = [ ant.name for ant in f.ants]
    freqs = f.channel_freqs
    chans = f.channels
    tstamps = f.timestamps[:]
    meta_data = [observer, name, fsize, description, centre_freq, dump,start_time, end_time]

    return {'metadata':meta_data,'targets':targets,'fileopened':f, 'ants':ants,'freqs':freqs, 'chans':chans, 'tstamps':tstamps}

def extract_spectra_data():
    try:
        load = load_data(args[0])
    except (IndexError) as e:
        raise SystemExit(e)

    fileopened = load['fileopened']
    antennas = load['ants']
    targets = load['targets']
    
    chan_range = slice(10,-10)
    
    fileopened.select(corrprods='auto', pol='H', channels=chan_range,scans='~slew')
    d = np.abs(fileopened.vis[:].mean(axis=0))
    freqs = fileopened.channel_freqs*1.0e-6
    #detect spikes from the data
    spikes = detect_spikes(d)
    #detects all the spikes seen by all antennas irrespective of pointing
    rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
    fig = plt.figure()
    fig.suptitle('Mean Horizontal auto-correlation spectra per Antenna',size = 'small', fontweight='bold')
    grid = Grid(fig, 111, nrows_ncols=(3,2), axes_pad=0.0, share_all=True)
    for index,ant in enumerate(antennas):
        antenna = ant +'\n'
        ylim=(0,1.2*d[:,index].max())
        xlim=(freqs[0],freqs[-1])
        spikes = detect_spikes(d[:,index])
        rfi_freqs = [freqs[i] for i,elem in enumerate(spikes,0) if elem]
        rfi_power = [d[:,index][i] for i,elem in enumerate(spikes,0) if elem]
        label = "Flags [MHz]:\n"
        text = antenna +'\n'+label+'\n'.join(['%.3f' % num for num in rfi_freqs])
        at = AnchoredText(text,prop=dict(size=3), frameon=True,loc=2)
        grid[index].add_artist(at)
        grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
        grid[index].plot(freqs,d[:,index])
        grid[index].add_artist(at)
        plt.setp(grid[index],xlim=xlim,ylim=ylim,yticks=[],xticks=[])
    label = "Flags in all Ants [MHz]:\n"
    text = label +'\n'.join(['%.3f' % num for num in rfi_inall_ants])
    at = AnchoredText(text,prop=dict(size=4), frameon=True,loc=2)
    grid[-1].add_artist(at)
    
    pdf.savefig(fig)

    #re-initialise the oppened file for new selection
    fileopened.select()
    
    fileopened.select(corrprods='auto', pol='V',channels=chan_range, scans='~slew')
    d = np.abs(fileopened.vis[:].mean(axis=0))
    #detect spikes from the data
    spikes = detect_spikes(d)
    #detects all the spikes seen by all antennas irrespective of pointing
    rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
    
    fig = plt.figure()
    grid = Grid(fig, 111, (3,2),axes_pad=0.0, share_all=True)
    fig.suptitle('Mean Vertical auto-correlation spectra per Antenna',size = 'small', fontweight='bold')
    for index,ant in enumerate(antennas):
        antenna = ant + '\n'
        ylim=(0,1.2*d[:,index].max())
        xlim=(freqs[0],freqs[-1])
        spikes = detect_spikes(d[:,index])
        rfi_freqs = [freqs[i] for i,elem in enumerate(spikes,0) if elem]
        rfi_power = [d[:,index][i] for i,elem in enumerate(spikes,0) if elem]
        label = "Flags [MHz]:\n"
        text = antenna +'\n'+label+'\n'.join(['%.3f' % num for num in rfi_freqs])
        at = AnchoredText(text,prop=dict(size=3), frameon=True,loc=2)
        grid[index].add_artist(at)
        grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
        grid[index].plot(freqs,d[:,index].T)
        grid[index].add_artist(at)
        plt.setp(grid[index],xlim=xlim,ylim=ylim,xticks=[],yticks=[])
    # writting out common flags
    label = "Flags in all Ants [MHz]:\n\n"
    text = label +'\n'.join(['%.3f' % num for num in rfi_inall_ants])
    at = AnchoredText(text,prop=dict(size=4), frameon=True,loc=2)
    grid[-1].add_artist(at)
    pdf.savefig(fig)
    
    #re-initialise the oppened file for new selection
    fileopened.select()
    
    fig = plt.figure()
    fig.suptitle('All antennas mean horizontal auto-correlation spectra per pointing',size = 'small', fontweight='bold')
    grid = Grid(fig, 111,  nrows_ncols =(4,5), axes_pad=0.0, share_all=True)
    for index, targ in enumerate(targets):
        fileopened.select(corrprods='auto', pol='H',targets=targ,channels=chan_range, scans='~slew')
        freqs = fileopened.channel_freqs*1.0e-6
        data = np.abs(fileopened.vis[:].mean(axis=0))
        ylim=(0,1.2*data.max())
        xlim=(freqs[0],freqs[-1])
        spikes = detect_spikes(data)
        #detect spikes seen in all antennas per each pointing
        rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
        label = "Flags [MHz]:\n"
        text = targ+'\n'+label+'\n'.join(['%.3f' % num for num in rfi_inall_ants])
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
    pdf.savefig(fig)
    
    fileopened.select()
    
    fig = plt.figure()
    fig.suptitle('All antennas mean vertical auto-correlation spectra per pointing',size = 'small', fontweight='bold')
    grid = Grid(fig, 111, (4,5),axes_pad=0.0, share_all=True)
    for index, targ in enumerate(targets):
        fileopened.select(corrprods='auto', pol='V',targets=targ,channels=chan_range, scans='~slew')
        freqs = fileopened.channel_freqs*1.0e-6
        data = np.abs(fileopened.vis[:].mean(axis=0))
        ylim=(0,1.2*data.max())
        xlim=(freqs[0],freqs[-1])
        spikes = detect_spikes(data)
        rfi_inall_ants = [freqs[i] for i,elem in enumerate(spikes.all(axis=1)) if elem]
        label = "Flags [MHz]:\n"
        text = targ+'\n'+label+'\n'.join(['%.3f' % num for num in rfi_inall_ants])
        at = AnchoredText(text,prop=dict(size=4), frameon=True,loc=2)
        grid[index].add_artist(at)
        for k,ant in enumerate(antennas):
            rfi_freqs = [freqs[i] for i,elem in enumerate(spikes[:,k]) if elem]
            rfi_power = [data[:,k][i] for i,elem in enumerate(spikes[:,k]) if elem]
            grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
        grid[index].plot(freqs,data)
        grid[index].add_artist(at)
        plt.setp(grid[index],xticks=[],yticks=[],xlim=xlim, ylim=ylim)
    pdf.savefig(fig)

    fileopened.select()

    # Horizontal selection per pointing
    fileopened.select()
    for ant in antennas:
        fig = plt.figure()
        fig.suptitle(' '.join([ant.capitalize(),'Mean Horizontal auto-correlation spectra per pointing']),size = 'small', fontweight='bold')
        fig.text(0.5, 0.04, 'Frequency [MHz]', ha='center', va='center',size='x-small', fontweight='bold',style='italic')
        fig.text(0.06, 0.5, 'Power [Units]', ha='center', va='center', size='x-small',rotation='vertical', fontweight='bold',style='italic')
        grid = Grid(fig, 111, (4, 5), axes_pad=0.0, share_all=True)
        for index,targ in enumerate(targets):
            fileopened.select(ants=ant,corrprods='auto',pol='H',targets=targ,channels=chan_range,scans ='~slew')
            data=np.abs(fileopened.vis[:].mean(axis=0))
            ylim=(0,1.2*data.max())
            xlim=(freqs[0],freqs[-1])
            #at = AnchoredText(targ,prop=dict(size=5), frameon=False,loc=1)
            spikes = detect_spikes(data)
            rfi_freqs = [freqs[i] for i,elem in enumerate(spikes,0) if elem]
            rfi_power = [data[i] for i,elem in enumerate(spikes,0) if elem]
            grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
            grid[index].plot(freqs,data)
            grid[index].add_artist(at)
            plt.setp(grid[index],xticks=[],yticks=[],ylim=ylim,xlim=xlim)
            label = "Flags [MHz]:\n"
            text = targ + '\n'+ label +'\n'.join(['%.3f' % num for num in rfi_inall_ants])
            at = AnchoredText(text,prop=dict(size=3), frameon=True,loc=2)
            grid[index].add_artist(at)
        pdf.savefig(fig)

    # Vertital selection per pointing
    fileopened.select()
    for ant in antennas:
        fig = plt.figure()
        fig.suptitle(' '.join([ant.capitalize(),'Mean Vertical auto-correlation spectra per pointing']), size = 'small', fontweight='bold')
        fig.text(0.5, 0.04, 'Frequency [MHz]', ha='center', va='center',size='x-small', fontweight='bold',style='italic')
        fig.text(0.06, 0.5, 'Power [Units]', ha='center', va='center', size='x-small',rotation='vertical', fontweight='bold',style='italic')
        grid = Grid(fig, 111, (4, 5),axes_pad=0.0, share_all=True)
        for index,targ in enumerate(targets):
            fileopened.select(ants=ant,corrprods='auto',pol='V',targets=targ,channels=chan_range,scans ='~slew')
            data=np.abs(fileopened.vis[:].mean(axis=0))
            ylim=(0,1.2*data.max())
            xlim=(freqs[0],freqs[-1])
            #at = AnchoredText(targ,prop=dict(size=5), frameon=False,loc=2)
            spikes = detect_spikes(data)
            rfi_freqs = [freqs[i] for i,elem in enumerate(spikes,0) if elem]
            rfi_power = [data[i] for i,elem in enumerate(spikes,0) if elem]
            grid[index].scatter(rfi_freqs,rfi_power,marker='+',color='Maroon')
            grid[index].plot(freqs,data)
            grid[index].add_artist(at)
            plt.setp(grid[index],xticks=[],yticks=[],ylim=ylim, xlim=xlim)
            label = "Flags [MHz]:\n"
            text = targ + '\n'+ label +'\n'.join(['%.3f' % num for num in rfi_inall_ants])
            at = AnchoredText(text,prop=dict(size=3), frameon=True,loc=2)
            grid[index].add_artist(at)
        pdf.savefig(fig)

    pdf.close()
    plt.show()

#-------------------------------
#--- FUNCTION :  detect_spikes
#-------------------------------

def detect_spikes(data, axis=0, spike_width=2, outlier_sigma=11.0):
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

extract_spectra_data()
print "Done!"
print("Open the file %s" % (os.path.basename(args[0]).replace('h5','pdf')))