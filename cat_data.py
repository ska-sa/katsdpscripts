#! /usr/bin/python
# Summarize the internals of an HDF5 data file (or a directory of files). This works on both augmented and unaugmented files.
#
# Ludwig Schwardt
# 22 January 2010
#

import h5py
import optparse
import os.path
import sys
import time
import glob

# Ripped from katpoint.construct_target_params, to avoid extra dependencies
def preferred_name(description):
    fields = [s.strip() for s in description.split(',')]
    # Extract preferred name from name list (starred or first entry)
    names = [s.strip() for s in fields[0].split('|')]
    if len(names) == 0:
        return ''
    else:
        try:
            ind = [name.startswith('*') for name in names].index(True)
            return names[ind][1:]
        except ValueError:
            return names[0]

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [<directories or files>]")
(opts, args) = parser.parse_args()
if len(args) < 1:
    args = ['.']

# Find all data sets mentioned, and add them to datasets
datasets = []
def walk_callback(arg, directory, files):
    datasets.extend([os.path.join(directory, f) for f in files if f.endswith('.h5')])
for arg in args:
    if os.path.isdir(arg):
        os.path.walk(arg, walk_callback, None)
    else:
        datasets.extend(glob.glob(arg))
if len(datasets) == 0:
    print 'ERROR: No HDF5 data sets found'
    sys.exit(1)

for dataset in datasets:
    try:
        f = h5py.File(dataset, 'r')
        filesize = os.path.getsize(dataset) / 1e6

        # All h5 files have at least compscans, scans, targets and timestamps
        compscans = f['Scans']
        num_compscans, num_scans, num_samples, num_chans = len(compscans), 0, 0, 0
        target, start, single_target = '', '', True
        for cs in compscans:
            num_scans += len(compscans[cs])
            new_target = preferred_name(compscans[cs].attrs['target'])
            # Extract first target encountered, and note if there are more than one
            if not target:
                target = new_target
            elif new_target != target:
                single_target = False
            for s in compscans[cs]:
                scan = compscans[cs][s]
                # Extract first timestamp of first scan
                if not start:
                    start = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(scan['timestamps'][0] // 1000))
                num_samples += scan['data'].shape[0]
                num_chans = scan['data'].shape[1]
        if not single_target:
            target += '...'

        # Augmented file has a bit more info
        if 'augment' in f.attrs:
            ants = [f['Antennas'][ant].attrs['description'].partition(',')[0] for ant in f['Antennas']]
            centre_freq = f['Correlator'].attrs['center_frequency_hz'] / 1e6
            dump_rate = f['Correlator'].attrs['dump_rate_hz']
            print "%s: %s, %s, %s MHz, %2d compscans, %3d scans, %4d samples, %d chans, %s, %.1f MB" % \
                  (dataset, start, ' '.join(ants), centre_freq, num_compscans,
                   num_scans, num_samples, num_chans, target, filesize)
        else:
            print "%s: %s, UNAUGMENTED, %2d compscans, %3d scans, %4d samples, %d chans, %s, %.1f MB" % \
                  (dataset, start, num_compscans, num_scans, num_samples, num_chans, target, filesize)
    except h5py.H5Error, e:
        print "%s: Error reading file (bad format?): %s" % (dataset, e)
