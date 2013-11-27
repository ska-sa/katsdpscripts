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

import numpy as np
import scape
from scape.stats import robust_mu_sigma

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>")
    parser.add_option("-a", "--antenna", type="string", default='sd', help="Antenna to load (e.g. 'A1' for antenna 1), default is first single-dish baseline in file.")
    parser.add_option("-f", "--freq-chans", help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass.")
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

def extract_cal_dataset(dataset):
    """Build data set from scans in original dataset containing noise diode firings."""
    compscanlist = []
    for compscan in dataset.compscans:
        # Extract scans containing noise diode firings (make copy, as this will be modified by gain cal)
        # Don't rely on 'cal' labels, as the KAT-7 system does not produce it anymore
        scanlist = [scan.select(copy=True) for scan in compscan.scans
                    if 'nd_on' in scan.flags.dtype.fields and scan.flags['nd_on'].any()]
        if scanlist:
            compscanlist.append(scape.CompoundScan(scanlist, compscan.target))
    return scape.DataSet(None, compscanlist, dataset.experiment_id, dataset.observer,
                         dataset.description, dataset.data_unit, dataset.corrconf.select(copy=True),
                         dataset.antenna, dataset.antenna2, dataset.nd_h_model, dataset.nd_v_model, dataset.enviro)

def get_noise_diode_data(scan, min_samples=3, max_samples=9):
    """Get arrays of the power in the 'on' state and the 'off' state for
    noise diode firings - the 'off' state is the average of the valid 'offs' 
    before and after the 'on' state. 
    *** Largely pilfered from estimate_nd_jumps in gaincal.py ***

    Parameters
    ----------
    scan : :class:`dataset.scan` object
    min_samples : int, optional
        Minimum number of samples in each time segment, to ensure good estimates
    max_samples : int, optional
        Maximum number of samples in each time segment, to avoid incorporating scans
    """
    # Find indices where noise diode flag changes value, or continue on to the next scan
    jumps = (np.diff(scan.flags['nd_on']).nonzero()[0] + 1).tolist()
    num_times = len(scan.timestamps)
    # Get valid flags array (all valid if no valid flag)
    valid_flag = scan.flags['valid'] if 'valid' in scan.flags.dtype.names else np.tile(True, num_times)
    # The samples immediately before and after the noise diode changes state are invalid for gain calibration
    valid_flag[np.array(jumps) - 1] = False
    valid_flag[jumps] = False
    before_jump = [0] + jumps[:-1]
    at_jump = jumps
    after_jump = jumps[1:] + [num_times]
    # Initialise array for all noise diode jumps in this scan
    all_delta_mu=[]
    # For every jump, obtain segments before and after jump with constant noise diode state
    for start, mid, end in zip(before_jump, at_jump, after_jump):
        if scan.flags['nd_on'][mid]:
            off_segment = valid_flag[start:mid].nonzero()[0] + start
            on_segment = valid_flag[mid:end].nonzero()[0] + mid
        else:
            on_segment = valid_flag[start:mid].nonzero()[0] + start
            off_segment = valid_flag[mid:end].nonzero()[0] + mid

        # Skip the jump if one or both segments are too short
        if min(len(on_segment), len(off_segment)) < min_samples:
            continue
        # Limit segments to small local region around jump to avoid incorporating scans, etc.
        on_segment, off_segment = on_segment[-max_samples:], off_segment[:max_samples]
        nd_off_mu, nd_off_sigma = robust_mu_sigma(scan.data[off_segment, :, :])
        nd_on_mu, nd_on_sigma = robust_mu_sigma(scan.data[on_segment, :, :])
        nd_delta_mu = nd_on_mu - nd_off_mu
        all_delta_mu.append(np.mean(nd_delta_mu, axis=0)[:2])

    #Get the average delta_mu for each scan
    if len(all_delta_mu)>0:
        return np.mean(np.array(all_delta_mu), axis=0)
    else:
        return all_delta_mu



# Print out the 'on' and 'off' values of noise diode firings from an on->off transition to a text file.
opts, args = parse_arguments()

# Get data from h5 file and use 'select' to obtain a useable subset of it.
data = read_and_select_file(args[0], bline=opts.get('antenna',None), channels=opts.get('freq_chans',None))

# loop through compscans in file and get noise diode firings
nd_data = extract_cal_dataset(data)

#Output Filename
output_filename=args[0].strip('.h5') + '_nd_' + data.antenna.name + '.csv'

# Extract noise diode jump strengths from the data and write to csv file
f=file(output_filename,'w')
f.write("#Antenna %s noise diode data.\n"%(data.antenna.name))
f.write("# Target       , Timestamp ,HH Jump,VV Jump\n")
f.write("# name         ,  (sec)    ,     (counts)  \n")
for scan_ind, scan in enumerate(nd_data.scans):
    diode_jump = get_noise_diode_data(scan)
    target = scan.compscan.target.name
    time = np.mean(scan.timestamps)
    if len(diode_jump)>0:
        f.write("%-15s, %10.0f, %6.3f, %6.3f\n"%(target, time, diode_jump[0], diode_jump[1]))
    
f.close()

    #jumps = scan.flags['nd_on'].nonzero()
    #print scan_ind, scan, jumps