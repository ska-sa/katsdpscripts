#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################


import scape
import katpoint
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import optparse
import string
import sys
import logging

#------------------------------------------------------------------------------------------------

logger = logging.root

def LoadHDF5(HDF5Filename, header=False):
    try:
        d = scape.DataSet(HDF5Filename,baseline=opts.baseline)
    except ValueError:
        print "WARNING:THIS FILE",HDF5Filename.split('/')[-1], "IS CORRUPTED AND SCAPE WILL NOT PROCESS IT, YOU MAY NEED TO REAUGMENT IT,BUT ITS AN EXPENSIVE TASK..!!"
    else:
        print "SUCCESSFULLY LOADED: Wellcome to scape Library and scape is busy processing your request"

        lo_freq = 4200.0 + d.freqs[len(d.freqs)/2.0]

        # try to check all the rfi channels across all the channels
        rfi_chan_across_all = d.identify_rfi_channels()

        d = d.select(freqkeep=range(100,420))
        # rfi channels across fringe finder channels ( i.e frequancy range around 100 to 420)
        rfi_channels = d.identify_rfi_channels()
        freqs = d.freqs
        sky_frequency = d.freqs[rfi_channels]
        ant = d.antenna.name
        data_filename = os.path.splitext(os.path.basename(HDF5Filename))[0]+'.h5'
        # obs_date = os.path.splitext(os.path.basename(HDF5Filename))[0]
        #date = time.ctime(float(obs_date))

        for compscan in d.compscans:
            azimuth = np.hstack([scan.pointing['az'] for scan in compscan.scans])
            elevation = np.hstack([scan.pointing['el'] for scan in compscan.scans])
            compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
            compscan_start_time = np.hstack([scan.timestamps[0] for scan in compscan.scans])
            compscan_end_time = np.hstack([scan.timestamps[-1] for scan in compscan.scans])
            middle_time = np.median(compscan_times, axis=None)
            obs_date = katpoint.Timestamp(middle_time)
            middle_start_time = np.median(compscan_start_time)
            middle_end_time = np.median(compscan_end_time)
            end_time = katpoint.Timestamp(middle_end_time)
            min_compscan_az = katpoint.rad2deg(azimuth.min())
            max_compscan_az = katpoint.rad2deg(azimuth.max())
            min_compscan_el = katpoint.rad2deg(elevation.min())
            max_compscan_el = katpoint.rad2deg(elevation.max())
            start_time = katpoint.Timestamp(middle_start_time)
            requested_azel = compscan.target.azel(middle_time)
            #ant_az = katpoint.rad2deg(np.array(requested_azel[0]))
            #ant_el = katpoint.rad2deg(np.array(requested_azel[1]))
            target = compscan.target.name

            f = file(opts.outfilebase + '.csv', 'a')
            for index in range(0,len(rfi_channels)):
                rfi_chan = rfi_channels[index] + 100
                rfi_freq = freqs[rfi_channels[index]]
                f.write('%s, %s, %s, %s, %s,%f, %f,%f,%f, %f, %d, %f\n' % (data_filename,start_time, end_time, ant,target,min_compscan_az,max_compscan_az,\
                min_compscan_el, max_compscan_el,lo_freq, rfi_chan, rfi_freq))
            f.close()

def loop_througth(observationDataDir):
    # data is stored in time stamped directories -- use this to read data files in sequence sorted by date and time
    for file in sorted(os.listdir(observationDataDir)):
            # from the observation directory -- read the fits file
            if os.path.splitext(file)[1] == '.h5':
                print "\nReading HDF5 file:", file,'\n'
                LoadHDF5(os.path.join(observationDataDir, file))

if __name__ == '__main__':

    parser = optparse.OptionParser(usage='prog[options]<data file>',description='This extract the useful data from data file')
    parser.add_option('-a', '--baseline', default='sd',
    		help='Baseline to be loaded (e.g A1A1 for antenna 1) default is the first single-dish baseline in the data file')
    parser.add_option('-p', '--path', dest='data_dir',
            help='Directory containing observation data')
    parser.add_option("-o", "--output", dest="outfilebase", default='rfi_data_points',
                      help="Base name of output files (*.csv for output data)")

    (opts, args) = parser.parse_args()
    if len(args) > 0 or not opts.data_dir:
        parser.print_help()
        print "\nUsage example: \
        \n./extract_horizon_fits_data.py --path=\"data/\" "
        sys.exit(1)

    observationDataDir = os.path.dirname(opts.data_dir)
    # writing out the outup csv file
    data_line = ('FILENAME,START TIME,END TIME,ANT NAME,TARGET NAME, ANT MIN AZIM,ANT MAX AZIM,ANT MIN ELEV, ANT MAX ELEV,LO FREQUENCY, RFI CHANNELS, RFI FREQUENCY')
    f = file(opts.outfilebase + '.csv', 'a')
    f.write('%s\n' % (data_line))
    f.close()
    loop_througth(observationDataDir)
