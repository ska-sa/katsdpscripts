# Module to search through a .h5 file for noise diode firings and
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

from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt

import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import math
import os

import scape
from scape.stats import robust_mu_sigma

import rfilib

def read_and_select_file(file, bline=None, channels=None, **kwargs):
    """
    Read in the input h5 file using scape and make a selection.

    file:   {string} filename of h5 file to open in katdal

    Returns:
        the visibility data to plot, the frequency array to plot, the flags to plot
    """

    data = scape.DataSet(file, baseline=bline, katfile=True)
    compscan_labels=[]
    # Get compscan names (workaround for broken labelling after selection in scape)
    for compscan in data.compscans:
        compscan_labels.append(compscan.label)
    # Secect desired channel range and tracks
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
    data = data.select(freqkeep=chan_range,labelkeep='track')

    #return the selected data
    return data,compscan_labels


def get_system_temp(temperature):
    
    flags = rfilib.detect_spikes_sumthreshold(temperature)
    temps=[]
    for polnum,thispol in enumerate(['HH','VV']):
        thisdata= temperature[:,polnum]
        thisflags= flags[:,polnum]
        systemp,err = robust_mu_sigma(thisdata[np.where(thisflags==False)])
        temps.append(systemp)
    return temps


def present_results(pdf, temperature, freq, targname, antenna, channelwidth):

    #Set up the figure
    fig = plt.figure(figsize=(8.3,8.3))

    #Flag the data
    flags = rfilib.detect_spikes_sumthreshold(temperature)

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
        rfilib.plot_RFI_mask(ax, extra=freq[np.where(thisflags)], channelwidth=channelwidth)
        plt.xlim(freq[-1], freq[0])
    pdf.savefig()
    plt.close(fig)
    return temps

def plot_temps_time(pdf,alltimes,alltempshh,alltempsvv,antenna):

    fig=plt.figure(figsize=(8.3,8.3))

    ax = plt.subplot(211)
    plt.title("Tsys vs Time, Antenna: " + antenna + ", HH polarisation")
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
    plt.close(fig)

def analyse_noise_diode(input_file,output_dir='.',antenna='sd',targets='all',freq_chans=None):

    # Get data from h5 file and use 'select' to obtain a useable subset of it.
    data,compscan_labels = read_and_select_file(input_file, bline=antenna, channels=freq_chans)
    pdf = PdfPages(os.path.join(output_dir,os.path.splitext(os.path.basename(input_file))[0] +'_SystemTemp_'+data.antenna.name+'.pdf'))
    # loop through compscans in file and get noise diode firings
    #nd_data = extract_cal_dataset(data)

    #Convert the data to kelvin using the noise diode firings
    data.convert_power_to_temperature()

    #average each required scan in time sensibly and plot the data
    for num,compscan in enumerate(data.compscans):
        compscan_data = np.empty((0,len(data.channel_select),4))
        if (targets == 'all') or (compscan.target.name in targets):
            for scan in compscan.scans:
                scan_data = scan.data[np.where(scan.flags['nd_on']==False)]
                compscan_data=np.append(compscan_data,scan_data,axis=0)
            average_spec, sigma_spec = robust_mu_sigma(compscan_data, axis=0)
            plottitle = compscan.target.name + ' ' + compscan_labels[num]
            systemp=present_results(pdf, average_spec[:,:2], data.freqs*1.e6, plottitle, data.antenna.name, data.bandwidths[0]*1e6)

    #Get the system temperature in each scan and plot it
    alltempshh,alltempsvv,alltimes=[],[],[]
    zerotime = data.scans[0].timestamps[0]
    for scan in data.scans:
        if (targets == 'all') or (scan.compscan.target.name in targets):
            scan_data = scan.data[np.where(scan.flags['nd_on']==False)]
            average_spec, sigma_spec = robust_mu_sigma(scan_data, axis=0)
            systemp=get_system_temp(average_spec[:,:2])
            alltempshh.append(systemp[0])
            alltempsvv.append(systemp[1])
            alltimes.append((np.mean(scan.timestamps[np.where(scan.flags['nd_on']==False)])-zerotime)/(60*60))      

    plot_temps_time(pdf,alltimes,alltempshh,alltempsvv,data.antenna.name)

    pdf.close()
