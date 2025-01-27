#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import cm
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import os
import psrchive as psr
from coast_guard import cleaners

def print_metadata(archive):
    """Function to print archive file metadata.
    
    Input:
        archive: PSRCHIVE Archive object.
        
    Output:
        print metadata in a nice table.
    """
    # get metadata from file
    nbin = archive.get_nbin()
    nchan = archive.get_nchan()
    npol = archive.get_npol()
    nsubint = archive.get_nsubint()
    obs_type = archive.get_type()
    telescope_name = archive.get_telescope()
    source_name = archive.get_source()
    ra = archive.get_coordinates().ra()
    dec = archive.get_coordinates().dec()
    centre_frequency = archive.get_centre_frequency()
    bandwidth = archive.get_bandwidth()
    dm = archive.get_dispersion_measure()
    rm = archive.get_rotation_measure()
    is_dedispersed = archive.get_dedispersed()
    is_faraday_rotated = archive.get_faraday_corrected()
    is_pol_calib = archive.get_poln_calibrated()
    data_units = archive.get_scale()
    data_state = archive.get_state()
    obs_duration = archive.integration_length()
    receiver_name = archive.get_receiver_name()
    receptor_basis = archive.get_basis()
    backend_name = archive.get_backend_name()
    low_freq = archive.get_centre_frequency() - archive.get_bandwidth() / 2.0
    high_freq = archive.get_centre_frequency() + archive.get_bandwidth() / 2.0
    subint_duration = np.rint(archive.get_Integration(1).get_duration())
    subint_duration = np.rint(archive.get_Integration(1).get_duration())
    # Print out metadata
    print '====================================================================================='
    print 'Attribute Names  Description                           Value                         '
    print '====================================================================================='
    print 'nbin             Number of pulse phase bins            %s' % nbin
    print 'nchan            Number of frequency channels          %s' % nchan
    print 'npol             Number of polarizations               %s' % npol
    print 'nsubint          Number of sub-integrations            %s' % nsubint
    print 'type             Observation type                      %s' % obs_type
    print 'site             Telescope name                        %s' % telescope_name
    print 'name             Source name                           %s' % source_name
    print 'coord            Source coordinates                    %s%s' % (ra.getHMS(), dec.getDMS())
    print 'freq             Centre frequency (MHz)                %s' % centre_frequency
    print 'bw               Bandwidth (MHz)                       %s' % bandwidth
    print 'dm               Dispersion measure (pc/cm^3)          %s' % dm
    print 'rm               Rotation measure (rad/m^2)            %s' % rm
    print 'dmc              Dispersion corrected                  %s' % is_dedispersed
    print 'rmc              Faraday Rotation corrected            %s' % is_faraday_rotated
    print 'polc             Polarization calibrated               %s' % is_pol_calib
    print 'scale            Data units                            %s' % data_units
    print 'state            Data state                            %s' % data_state
    print 'length           Observation duration (s)              %s' % obs_duration
    print 'rcvr:name        Receiver name                         %s' % receiver_name
    print 'rcvr:basis       Basis of receptors                    %s' % receptor_basis
    print 'be:name          Name of the backend instrument        %s' % backend_name
    
def load_archive_data(path, verbose=False):
    """Function to load .ar files and convert to PSRCHIVE archive objects. 
    
    Input:
        path    : full path to location of the .ar files.
        verbose : option to run in verbose mode (default=False)
        
    Output:
        archives : list of PSRCHIVE archive objects
    """
    files = []
    for file in os.listdir(path):
        if file.endswith('.ar'):
            files.append(file) 
    files.sort()
    archives = [] 
    archives = [psr.Archive_load(path + file) for file in files]
    if verbose:
        print '======================================================================================================'
        print '                                     Files to be processed:                                           '
        print '======================================================================================================'
    for i in range(1, len(archives)): 
        archives[0].append(archives[i]) # add the .ar files (added file is archive[0])
        if verbose:
            print archives[i]
    return archives

def clean_archive_bandwagon(archive_clone, bad_chan_tol=0.9, bad_sub_tol=1.0):
    """Function to clean the archive files using coast_guard cleaner bandwagon.
    
       Input:
            archive_clone:    a clone of the PSRCHIVE archive object to clean.
            bad_chan_tol:     fraction of bad channels to be tolarated before mask (float between 0 - 1).
            bad_sub_tol:      fraction of bad sub-intergrations to be tolerated before mask (float between 0 - 1)
      
    """
    cleaner3 = cleaners.load_cleaner('bandwagon')
    cleaner3.parse_config_string('badchantol=0.99,badsubtol=1.0')
    cleaner3.run(archive_clone)
    
    
def clean_archive_rcvrstd(archive_clone, bad_channels='0:210;3896:4095', bad_frequencies=None, bad_subints=None, \
                          trim_bw=0, trim_frac=0, trim_num=0):
    """Function to clean the archive files using coast_guard cleaner rcvrstd.
    
       Input:
           archive_clone:    a clone of the PSRCHIVE archive object to clean.
           bad_channels:     bad channels to de-weight (default: band edges (0:210, 3896:4095).
           bad_frequencies:  bad frequencies to de-weight (default: None).
           bad_subints:      bad sub-ints to de-weight (default: None).
           trim_bw:          bandwidth of each band-edge (in MHz) to de-weight (default: None).
           trim_frac:        fraction of each band-edge to de-weight, float between 0 - 0.5 (default: None).
           trim_num:         number of channels to de-weight at each edge of the band (default: None).
    """
    cleaner2 = cleaners.load_cleaner('rcvrstd')
    
    cleaner2.parse_config_string('badchans=%s,badfreqs=%s,badsubints=%s,trimbw=%s,trimfrac=%s,trimnum=%s' \
                                 %(str(bad_channels), str(bad_frequencies), str(bad_subints), \
                                   str(trim_bw), str(trim_frac), str(trim_num)))
    cleaner2.run(archive_clone)
    
    
def clean_archive_surgical(archive_clone, chan_threshold=3, subint_threshold=3, chan_numpieces=1, subint_numpieces=1):
    """Function to clean the archive files using coast_guard cleaner surgical.
    
       Input:
           archive_clone:    a clone of the PSRCHIVE archive object.
           chan_threshold:   threshold sigma of a profile in a channel.
           subint_threshold: threshold sigma of a profile in a sub-intergration.
           chan_numpieces:   the number of equally sized peices in each channel (used for detranding in surgical)
           subint_numpieces: the number of equally sized peices in each sub-int (used for detranding in surgical)
    """
    cleaner = cleaners.load_cleaner('surgical')
    cleaner.parse_config_string('chan_numpieces=%s,subint_numpieces=%s,chanthresh=%s,subintthresh=%s'\
                               % (str(chan_numpieces), str(subint_numpieces), str(chan_threshold),\
                                  str(subint_threshold)))
    cleaner.run(archive_clone)
    
    
def print_zero_weight_statistics(weights, nchan, nsubint):
    """Function to determine a fraction of channels set to zero.
       Input:
           weights:    rfi weights from coast_guard cleaned files.
           nchans:     number of channels in the band.
           nsubint:    number of sub-intergrations.
    """
    counter = 0
    zero_weight_channel = 0
    for i in range(nsubint):
        zero_weight_channel = 0
        for j in range(nchan):
            if weights[i][j] == 0.0:
                counter += 1 
                zero_weight_channel += 1 # channels set with zero weights
        subint_proc = (float(zero_weight_channel) / float(nchan)) * 100
        print 'Subint %s has %s channels (%.2f%%) set to zero.' % (i, zero_weight_channel, subint_proc)
    percentage = (float(counter) / float(weights.size)) * 100
    print '%s data points out of %s with weights set to zero.' % (counter, weights.size)
    print '%.2f%% of the data was set to zero.' % (percentage)

def prepare_subint_data(archives):
    """Funtion to prepare PSRCHIVE archive file for plotting.
    
       Input:
           archives:  PSRCHIVE archive file object
       Returns:
           data:      dedispersed and freq-pol scrunched data.
    """
    stack = archives.clone()
    stack.dedisperse()
    stack.remove_baseline()
    stack.centre_max_bin()
    stack.fscrunch()
    stack.pscrunch()
    data = stack.get_data().squeeze()
    return data

def prepare_profile_data(archive):
    """Function to prepare average profile for stokes data.
       
       Input:
           archive:        PSRCHIVE archive file object.
           
       Returns:
           profile_data:   time and frequency scrunched profiles for stokes (I, Q, U, V).
    """
    profile = archive.clone()
    profile.dedisperse()
    profile.remove_baseline()
    profile.centre_max_bin()
    profile.tscrunch()
    profile.fscrunch()
    profile.convert_state('Stokes')
    profile_data = profile.get_data()
    profile_data = profile_data.squeeze()
    return profile_data

def prepare_sokes_phase_freq(archive):
    """Function to prepare average profile for stokes data.
       
       Input:
           archive:        PSRCHIVE archive file object.
           
       Returns:
           stokes_data:   time-freq flux for stokes (I, Q, U, V).
    """
    stokes_flux = archive.clone()
    stokes_flux.dedisperse()
    stokes_flux.remove_baseline()
    stokes_flux.centre_max_bin()    
    stokes_flux.tscrunch()
    stokes_flux.convert_state('Stokes')
    stokes_data = stokes_flux.get_data().squeeze()
    return stokes_data

def prepare_coherency_flux_data(archive):
    """Function to prepare coherancy data.
    
       Input:
           archive:          PSRCHIVE archive file object. 
           
       Returns:
           coherency_flux_phase_data:   time scrunched coherency flux data.
    """
    coherency_flux_phase = archive.clone()
    coherency_flux_phase.dedisperse()
    coherency_flux_phase.remove_baseline()
    coherency_flux_phase.centre_max_bin()
    coherency_flux_phase.tscrunch()
    coherency_flux_phase_data = coherency_flux_phase.get_data().squeeze()
    return coherency_flux_phase_data

def get_mean_bandpass(archive, stokes=False):
    """Function to detetermine the mean bandpass.
    
       Input:
           archive:          PSRCHIVE archive file object. 
           stokes:           option to get bandpass for stokes I, Q, U, V
       Returns:
           bandpass_mean:      mean bandpass
           bandpass_varience:  variance bandpass
    """
    bandpass = archive.clone()
    if stokes:
        bandpass.convert_state('Stokes')
    bandpass.tscrunch()
    (bandpass_mean, bandpass_variance) = bandpass.get_Integration(0).baseline_stats()    
    return bandpass_mean, bandpass_variance



def get_baseline_spectrum(archive, npol, nchan, nsubint, stokes=False):
    """Function to determine teh baseline spectrum
    """
    dynamic_spectrum = archive.clone()
    if stokes:
        dynamic_spectrum.convert_state('Stokes')
    mean = np.zeros((nsubint, npol, nchan))
    variance = np.zeros((nsubint, npol, nchan))
    for subint in range(nsubint):
        m, v = dynamic_spectrum.get_Integration(subint).baseline_stats()
        mean[subint] = m[:, :]
        variance[subint] = v[:, :]
    return mean, variance

        
        
def get_subint_snr(archive, nsubint):
    """Function to determine the subintergration S/N
       Input:
           archive:     PSRCHIVE archive object
           nsubint:     number of sub-intergrations
           
       Return:
           snr_data:    s/n per sub-intergration
    """
    archive_snr = archive.clone()
    new_integration = archive_snr.get_Integration(0).total()
    snr_data = np.zeros(nsubint)
    for i_subintegration in range (1, nsubint):
        next_integration = archive_snr.get_Integration(i_subintegration).total()
        new_integration.combine(next_integration)
        profile = new_integration.get_Profile(0, 0)
        snr_data[i_subintegration] = profile.snr()
    return snr_data


def create_subint_profile_figure(fig_title):
    """Function to create a figure and axis. To be used to compare 
       clean and unclean subintergartion stack, profile, and bandpass.
       
       Input:
           fig_tittle:          title for the whole figure.
           
       Returns:
           fig:                 the figure handle.
           ax:                  figure axes handle.
           clean_stack:         subplot axes handle to plot clean subintergration stack.
           clean_profile:       subplot axes handle to plot clean stokes average profile.
           unclean_stack:       subplot axes handle to plot dirty subintergration stack.
           unclean_profile:     subplot axes handle to plot dirty stokes average profile.
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 15), sharex='col', tight_layout='True')
    # create 2 sub-figures
    grid_spec_1 = GridSpec(3, 3)
    grid_spec_1.update(left=0.05, right=0.48, wspace=0.05, hspace=0.1)
    unclean_profile = plt.subplot(grid_spec_1[0, :])
    unclean_stack = plt.subplot(grid_spec_1[1:, :])
    unclean_profile.tick_params(labelbottom=False)

    
    grid_spec_2 = GridSpec(3, 3)
    grid_spec_2.update(left=0.6, right=0.98, wspace=0, hspace=0.1)
    clean_profile = plt.subplot(grid_spec_2[0, :])
    clean_profile.tick_params(labelbottom=False, labelleft=False)
    clean_stack = plt.subplot(grid_spec_2[1:, :])
    clean_stack.tick_params(labelleft=False)
    
    fig.suptitle(fig_title, fontsize=18)   
    return fig, ax, clean_stack, clean_profile, unclean_stack, unclean_profile

    
def plot_mask_channels(weights, low_freq, high_freq, nchan, nsubint, target):
    """Function to plot image of the rfi masked channels in the band.
       Input:
           weights:    rfi weights from coast_guard cleaned files.
           low_freq:   minimum frequency in the band (MHz).
           high_freq:  maximum frequency in the band (MHz).
           nchans:     number of channels.
           nsubint:    number of sub-intergrations.
           target:     name of the target source observed.
    """
    fig, ax1 = plt.subplots(1, 1, figsize = [15, 10])
    ax1.set_title(target, fontsize=20)
    ax1.set_title('RFI mask ', loc='left', fontsize=20)
    ax1.set_ylabel('Channel number', fontsize=18)
    ax1.yaxis.set_ticks(np.arange(0, nchan - 1, 200))
    ax1.set_xlabel('Subint number', fontsize=18)
    ax1_secondary = ax1.twinx()
    ax1_secondary.set_ylabel('Frequency (MHz)', fontsize=18)
    ax1_secondary.set_ylim(low_freq, high_freq)
    ax1_secondary.yaxis.set_ticks(np.arange(np.rint(low_freq), np.rint(high_freq), 25))
    ax1.imshow(weights.T, origin='lower', aspect='auto', cmap=colors.ListedColormap(['red', 'white']), \
               interpolation='none', extent=(0, nsubint - 1, 0, nchan - 1))

def plot_sub_intergration_stack(data, ax,  nbin, nsubint, obs_duration, title=False):
    """Function to plot the prepared sub intergration data on provided axis.
    
       Input:
           data:           data to be plotted
           ax:             axes to plot
           title:          title of the plot
           nbin:           number of bins for the plot
           nsubint:        number of sub-intergrations
           obs_duration:   observation intergration time
    """
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('Pulse phase (bin)', fontsize=14)    
    ax.xaxis.set_ticks(np.arange(0, nbin - 1, 100))
    ax.imshow(data, cmap=cm.afmhot, aspect='auto', vmax=data.max() * 0.75, 
              interpolation='none', extent=(0, nbin - 1, 0, nsubint - 1))
    
def plot_stokes_profiles(profile_data, target, ax, nbin, label='Clean data'):
    """ Function to plot the stokes profiles.
    
        Input:
            profile_data:    profiles to plot shape (I, Q, U, V) 
            target:          target source name
            ax:              figure axis to plot on.
            nbin:            number of bins.
            label:           plot title label.
    """
    min_max_vals = np.array([profile_data[0, :].max(), profile_data[1, :].max(),
                             profile_data[2, :].max(), profile_data[3, :].max(), 
                             profile_data[0, :].min(), profile_data[1, :].min(), 
                             profile_data[2, :].min(), profile_data[3, :].min()])
    ax.text(20, profile_data.max(), label, fontsize='large')
    ax.set_title(target)
    ax.set_ylabel('Flux (a.u.)')
    ax.set_ylim((min_max_vals.min() * 1.2), min_max_vals.max() + 0.1 * min_max_vals.max())
    ax.set_xlim(0, nbin - 1)
    ax.plot(profile_data[0, :], 'k', label='I')
    ax.plot(profile_data[1, :], 'r', label='Q')
    ax.plot(profile_data[2, :], 'g', label='U')
    ax.plot(profile_data[3, :], 'b', label='V')
    ax.xaxis.set_ticks(np.arange(0, nbin - 1, 50))
    ax.legend()
    
def plot_phase_freq_flux(data, ax, title, low_freq, high_freq, nbin, nchan):
    """Function to plot the phase-frequency image of the flux.
       Input:
           data:    time-frequency data to plot
           ax:      axis to plot.
           
    """
    ax.set_title(title)
    ax.xaxis.set_ticks(np.arange(0, nbin - 1, 100))
    ax.yaxis.set_ticks(np.arange(0, nchan - 1, 50))
    ax.imshow(data, cmap=cm.afmhot, aspect='auto', interpolation='none', extent=(0, nbin - 1.0, low_freq, high_freq))

    
    
def plot_dynamic_spectrum(data, ax, title, low_freq, high_freq, nbin, nchan, minimum, maximum, clean_data=True):
    """Function to plot the phase-frequency image of the flux.
       Input:
           data:    time-frequency data to plot
           ax:      axis to plot.  
    """
    ax.set_title(title)
    ax.yaxis.set_ticks(np.arange(0, nchan - 1, 400))
    if clean_data:
        ax.imshow(data.T, origin='lower', aspect='auto', cmap=cm.afmhot, 
                  norm=colors.Normalize(vmin=minimum, vmax=maximum), 
                  interpolation='none')
    else:
        ax.imshow(data.T, origin='lower', aspect='auto', cmap=cm.afmhot, 
                  norm=colors.SymLogNorm(10, linscale=1.0, vmin=minimum, vmax=maximum), 
                  interpolation='none')
    
    
    
def set_subint_time_label(ax, intergration):
    """Sets twin axes sharing the axis
       
       Input:
           ax:             axis to be shared 
           intergration:   observation duration
    """
    ax_tertiary = ax.twinx()
    ax_tertiary.set_ylabel('Time (seconds)')
    ax_tertiary.yaxis.set_label_position('right')
    ax_tertiary.set_ylim(0, intergration)

def set_phase_label(ax):
    """Sets twin axes sharing the axis
    
       Input:
           ax    : axis to be shared
    """
    ax_secondary = ax.twiny()
    ax_secondary.set_frame_on(True)
    ax_secondary.patch.set_visible(False)
    ax_secondary.xaxis.set_ticks_position('bottom')
    ax_secondary.set_xlabel('Pulse phase (degrees)')
    ax_secondary.xaxis.set_label_position('bottom')
    ax_secondary.spines['bottom'].set_position(('outward', 50))
    ax_secondary.set_xlim(0, 360)
    _ = ax_secondary.xaxis.set_ticks(np.arange(0, 360, 20))
    
def set_channel_label(ax, nchan):
    """Sets twin axes sharing the axis
    
       Input:
           ax:      axis to be shared
           nchan:   number of channels  
    """
    ax_secondary = ax.twinx()
    ax_secondary.set_ylabel('Channel number')
    ax_secondary.set_ylim(0, nchan - 1)
    ax_secondary.yaxis.set_ticks(np.arange(0, nchan - 1 , 200))
    
def set_freq_label(ax, low_freq, high_freq, pos='xaxis'):
    """
    """
    if pos == 'xaxis':
        ax_secondary = ax.twiny()
    elif pos == 'yaxis':
        ax_secondary = ax.twinx()
    ax_secondary.set_frame_on(True)
    ax_secondary.patch.set_visible(False)
    ax_secondary.xaxis.set_ticks_position('bottom')
    ax_secondary.set_xlabel('Frequency (MHz)')
    ax_secondary.xaxis.set_label_position('bottom')
    ax_secondary.spines['bottom'].set_position(('outward', 50))
    ax_secondary.set_xlim(low_freq, high_freq)
    _ = ax_secondary.xaxis.set_ticks(np.arange(np.rint(low_freq), np.rint(high_freq), 50))
    

def set_bandpass_plot_label(bandpass, target, title, ax, nchan, logscale=True):
    """Function to prepare bandpass plot.
    """
    bandpass_max_min_vals = np.array([bandpass[0, :].min(), bandpass[0, :].max(),
                                            bandpass[1, :].min(), bandpass[1, :].max()])
    ax.set_title(target)
    ax.set_title(title, loc='left')
    ax.set_xlim(0, nchan - 1)
    ax.set_ylabel('Flux (a.u.)')
    ax.xaxis.set_ticks(np.arange(0, nchan - 1, 200))
    if logscale:
        ax.set_yscale('log')
