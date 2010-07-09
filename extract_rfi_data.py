#!/usr/bin/python

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

    d = scape.DataSet(HDF5Filename,baseline = opts.baseline)

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
    obs_date = os.path.splitext(os.path.basename(HDF5Filename))[0]
    date = time.ctime(float(obs_date))

    for compscan in d.compscans:
        compscan_times = np.hstack([scan.timestamps for scan in compscan.scans])
        middle_time = np.median(compscan_times, axis=None)
        requested_azel = compscan.target.azel(middle_time)
        ant_az = katpoint.rad2deg(np.array(requested_azel[0]))
        ant_el = katpoint.rad2deg(np.array(requested_azel[1]))

    f = file('rfi_data_points' + '.csv', 'a')
    for index in range(0,len(rfi_channels)):
        f.write('%s, %s,%s,%s,%f,%f,%f,%d,%f\n' % (data_filename, date, ant, ant_az,ant_el, freqs[index],lo_freq, rfi_channels[index], freqs[rfi_channels[index]]))
    f.close()

def loop_througth(observationDataDir):
    # data is stored in time stamped directories -- use this to read data files in sequence sorted by date and time
    for file in sorted(os.listdir(observationDataDir)):
            # from the observation directory -- read the fits file
            if os.path.splitext(file)[1] == '.h5':
                print "\nReading HDF5 file:", file,
                LoadHDF5(os.path.join(observationDataDir, file))

if __name__ == '__main__':

    parser = optparse.OptionParser(usage='prog[options]<data file>',description='This extract the useful data from data file')
    parser.add_option('-a', '--baseline', dest='baseline', type='string', metavar='BASELINE', default='AxAx',help='Baseline to be loaded (e.g A1A1 for antenna 1) default is the first single-dish baseline in the data file')
    parser.add_option('-p', '--path', dest='data_dir',help='Directory containing observation data')
    parser.add_option("-o", "--output", dest="outfilebase", type="string", default='rfi_data_points',help="Base name of output files (*.csv for output data)")

    (opts, args) = parser.parse_args()

    if len(args) > 0 or not opts.data_dir:
        parser.print_help()
        print "\nUsage example: \
        \n./extract_horizon_fits_data.py --path=\"data/\" "
        sys.exit(1)
    observationDataDir = os.path.dirname(opts.data_dir)
    # writing out the outup csv file
    data_line = ('FILENAME, DATE,ANT NAME,ANT AZIM,ANT ELEV,CENTER FREQUENCY,LO FREQUENCY, RFI CHANNELS, RFI FREQUENCY')
    f = file('rfi_data_points' + '.csv', 'a')
    f.write('%s\n' % (data_line))
    f.close()
    loop_througth(observationDataDir)