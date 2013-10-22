#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
#
# Apply Van Vleck (quantisation) correction to an HDF5 file. This uses the
# online system itself in an offline mode to perform the correction.
#
# Ludwig Schwardt
# 7 May 2012
#

import optparse
import h5py
import numpy as np
from katcapture import sigproc

parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description="This applies Van Vleck correction to the given HDF5 file.")
(opts, args) = parser.parse_args()
if len(args) < 1:
    raise RuntimeError('Please specify an HDF5 file to correct')
filename = args[0]

# Open HDF5 file with write access
f = h5py.File(filename, 'r+')

version = f.attrs.get('version', '1.x')
if not version.startswith('2.'):
    raise ValueError('Only version 2 (KAT-7) HDF5 files are currently supported (got version %s instead)' % (version,))

data_group, config_group = f['Data'], f['MetaData/Configuration']
vis = data_group['correlator_data']
def get_single_value(group, name):
    """Return single value from attribute or dataset with given name in group."""
    return group.attrs[name] if name in group.attrs else group[name].value[-1]
corrprods = get_single_value(config_group['Correlator'], 'bls_ordering')
accum_per_int = get_single_value(config_group['Correlator'], 'n_accs')

# Handle correlation product mislabelling
if len(corrprods) != vis.shape[2]:
    script_ants = config_group['Observation'].attrs['script_ants'].split(',')
    # Apply k7_capture baseline mask after the fact, in the hope that it fixes correlation product mislabelling
    corrprods = np.array([cp for cp in corrprods if cp[0][:-1] in script_ants and cp[1][:-1] in script_ants])
    # If there is still a mismatch between labels and data shape, file is considered broken (maybe bad labels?)
    if len(corrprods) != vis.shape[2]:
        raise ValueError('Number of baseline labels (containing expected antenna names) '
                         'received from correlator (%d) differs from number of baselines in data (%d)' %
                         (len(corrprods), vis.shape[2]))
    else:
        print 'Reapplied k7_capture baseline mask to fix unexpected number of baseline labels'
# Identify autocorrelations
auto = [n for n, (inpA, inpB) in enumerate(corrprods) if inpA == inpB]
labels = [inpA for inpA, inpB in corrprods[auto]]

# Create online processing block for Van Vleck correction
vanvleck = sigproc.VanVleck(accum_per_int, bls_ordering=corrprods)

# Iterate through visibility data, one dump at a time, keeping track of power statistics
power_before = np.zeros((len(auto), vis.shape[0]))
power_after = np.zeros((len(auto), vis.shape[0]))
aborted = False
for t in range(vis.shape[0]):
    sigproc.ProcBlock.current = vis[t]
    power_before[:, t] = np.median(sigproc.ProcBlock.current[:, auto, 0], axis=0)
    try:
        vanvleck.proc()
    except sigproc.VanVleckOutOfRangeError:
        print 'Van Vleck correction seems to be applied already at dump %d - aborting conversion' % (t,)
        aborted = True
        break
    power_after[:, t] = np.median(sigproc.ProcBlock.current[:, auto, 0], axis=0)
    vis[t] = sigproc.ProcBlock.current

if not aborted:
    # Flush changes to disk
    f.close()

    print "Median power before and after correction:"
    print '\n'.join([("%s: %6.3g %6.3g" % (label, before, after))
                     for label, before, after in zip(labels, np.median(power_before, axis=1),
                                                             np.median(power_after, axis=1))])
