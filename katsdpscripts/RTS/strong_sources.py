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

from __future__ import absolute_import
from __future__ import print_function
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt

import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import math
import os
import pickle

import scape
from scape.stats import robust_mu_sigma

from . import rfilib
from six.moves import range

def read_and_select_file(file, bline=None, channels=None, rfi_mask=None, nd_models=None, **kwargs):
    """
    Read in the input h5 file using scape and make a selection.

    file:   {string} filename of h5 file to open in katdal

    Returns:
        the visibility data to plot, the frequency array to plot, the flags to plot
    """

    data = scape.DataSet(file, baseline=bline, nd_models=nd_models, katfile=True)
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
    chan_select = list(range(start_chan,end_chan+1))
    if rfi_mask:
        mask_file = open(rfi_mask)
        chan_select = ~(pickle.load(mask_file))
        mask_file.close()
        if len(chan_select) != num_channels:
            raise ValueError('Number of channels in provided mask does not match number of channels in data')
        chan_select[:start_chan] = False
        chan_select[end_chan:] = False

    #return the selected data
    return data.select(freqkeep=chan_select,labelkeep='track')


def get_system_temp(temperature):
    flagger=rfilib.sumthreshold_flagger(spike_width_time=1,spike_width_freq=5)
    flags = flagger.get_flags(np.expand_dims(temperature,0)).squeeze()
    temps=[]
    for polnum,thispol in enumerate(['HH','VV']):
        thisdata= temperature[:,polnum]
        thisflags= flags[:,polnum]
        systemp,err = robust_mu_sigma(thisdata[np.where(thisflags==False)])
        temps.append(systemp)
    return temps


def present_results(pdf, temperatures, freq, targnames, antenna, channelwidth):

    #Set up the figure
    fig = plt.figure(figsize=(8.3,8.3))

    flagger=rfilib.sumthreshold_flagger(spike_width_time=1,spike_width_freq=5)

    ax = [plt.subplot(211),plt.subplot(212)]
    for num,targname in enumerate(targnames):
        temperature = temperatures[num]
        #Flag the data
        flags = flagger.get_flags(np.abs(np.expand_dims(temperature,0))).squeeze()
        #Loop over polarisations
        temps=[]
        for polnum,thispol in enumerate(['HH','VV']):
            thisdata = temperature[:,polnum]
            thisflags = flags[:,polnum]
            systemp, err = robust_mu_sigma(thisdata[np.where(thisflags==False)])
            temps.append(systemp)
            ax[polnum].set_title('Antenna: ' + antenna + ', ' + thispol + ' pol')
            ax[polnum].plot(freq,thisdata,label=targname + ' Tsys: %5.2f'%(systemp))
            ax[polnum].set_xlabel('Frequency (MHz)')
            ax[polnum].set_ylabel('System Temperature (K)')
            ax[polnum].set_xlim(min(freq), max(freq))
    ax[0].legend(loc=3)
    ax[1].legend(loc=3)
    pdf.savefig()
    plt.close(fig)
    return temps

def present_difference_results(pdf,temperatures, freq, antenna, channelwidth):

    #Set up the figure
    fig = plt.figure(figsize=(8.3,8.3))

    flagger=rfilib.sumthreshold_flagger(spike_width_time=1,spike_width_freq=5)

    ax = [plt.subplot(211),plt.subplot(212)]
    #ASSUMPTION: before scan is element 0 and after scan is element 1
    temperature_before = temperatures[0]
    temperature_after = temperatures[1]
    #Flag the data
    flags_before = flagger.get_flags(np.abs(np.expand_dims(temperature_before,0))).squeeze()
    flags_after = flagger.get_flags(np.abs(np.expand_dims(temperature_after,0))).squeeze()

    difference = temperature_before-temperature_after
    flags = flags_before | flags_after
    #Loop over polarisations
    temps=[]
    for polnum,thispol in enumerate(['HH','VV']):
        thisdata = difference[:,polnum]
        thisflags = flags[:,polnum]
        systemp, err = robust_mu_sigma(thisdata[np.where(thisflags==False)])
        temps.append(systemp)
        ax[polnum].set_title('Antenna: ' + antenna + ', ' + thispol + ' pol')
        ax[polnum].plot(freq[np.where(thisflags==False)],thisdata[np.where(thisflags==False)],label='track_before - track_after, Mean: %5.2f'%(systemp))
        ax[polnum].set_xlabel('Frequency (MHz)')
        ax[polnum].set_ylabel('System Temperature Difference (K)')
        ax[polnum].set_xlim(min(freq), max(freq))
        #ax[polnum].set_ylim(min(thisdata[np.where(thisflags==False)]), max(thisdata[np.where(thisflags==False)]))
    ax[0].legend(loc=3)
    ax[1].legend(loc=3)
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
    plt.xlim(alltimes[0]-0.1, alltimes[-1]+0.1)
    plt.axhline(alltempshh[0],linestyle='--')
    ax = plt.subplot(212)
    plt.title("VV polarisation")
    ax.plot(alltimes,alltempsvv,'ro')
    plt.xlabel("Time since observation start (hours)")
    plt.ylabel("Tsys")
    plt.xlim(alltimes[0]-0.1, alltimes[-1]+0.1)
    plt.axhline(alltempsvv[0],linestyle='--')
    pdf.savefig()
    plt.close(fig)

def analyse_noise_diode(input_file,output_dir='.',antenna='sd',targets='all',freq_chans=None,rfi_mask=None, nd_models=None):

    # Get data from h5 file and use 'select' to obtain a useable subset of it.
    data = read_and_select_file(input_file, bline=antenna, channels=freq_chans, rfi_mask=rfi_mask, nd_models=nd_models)
    pdf = PdfPages(os.path.join(output_dir,os.path.splitext(os.path.basename(input_file))[0] +'_SystemTemp_'+data.antenna.name+'.pdf'))
    #Convert the data to kelvin using the noise diode firings
    data.convert_power_to_temperature()
    #average each required scan in time sensibly and plot the data for the before and after scans
    average_specs=[]
    plottitles=[]
    compscan_labels=np.array([compscan.label for compscan in data.compscans])
    ba_compscans = np.concatenate((np.where(compscan_labels == 'track_before')[0], np.where(compscan_labels == 'track_after')[0],))
    for num in ba_compscans:
        compscan = data.compscans[num]
        compscan_data = np.empty((0,len(data.channel_select),4))
        if (targets == 'all') or (compscan.target.name in targets):
            for scan in compscan.scans:
                scan_data = scan.data[np.where(scan.flags['nd_on']==False)]
                compscan_data = np.append(compscan_data,scan_data,axis=0)
            average_spec, sigma_spec = robust_mu_sigma(compscan_data, axis=0)
            average_specs.append(average_spec[:,:2])
            plottitles.append(compscan.target.name + ' ' + compscan_labels[num])
    systemp = present_results(pdf, average_specs, data.freqs, plottitles, data.antenna.name, data.bandwidths[0])
    #Plot the (before - after) spectrum
    #Assume before is first and after second
    if len(average_specs)==2:
        difftemp = present_difference_results(pdf, average_specs, data.freqs, data.antenna.name, data.bandwidths[0])
    else:
        print("No before and after tracks. Not plotting difference spectrum.")
    #Plot the scan track strong scans
    average_specs=[]
    plottitles=[]
    strong_compscans = np.where(compscan_labels == 'track_strong')[0]
    for num in strong_compscans:
        compscan = data.compscans[num]
        compscan_data = np.empty((0,len(data.channel_select),4))
        if (targets == 'all') or (compscan.target.name in targets):
            for scan in compscan.scans:
                scan_data = scan.data[np.where(scan.flags['nd_on']==False)]
                compscan_data = np.append(compscan_data,scan_data,axis=0)
            average_spec, sigma_spec = robust_mu_sigma(compscan_data, axis=0)
            average_specs.append(average_spec[:,:2])
            plottitles.append(compscan.target.name + ' ' + compscan_labels[num])
        systemp = present_results(pdf, average_specs, data.freqs, plottitles, data.antenna.name, data.bandwidths[0])
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
