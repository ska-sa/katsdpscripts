#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
# Plot horizon mask

import optparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scape
from katpoint import rad2deg


def remove_rfi(d,width=3,sigma=5,axis=1):
    for i in range(len(d.scans)):
        d.scans[i].data = scape.stats.remove_spikes(d.scans[i].data,axis=axis,spike_width=width,outlier_sigma=sigma)
    return d
def main():
    # Parse command-line options and arguments
    parser = optparse.OptionParser(usage='%prog [options] <data file> [<data file> ...]',
                                   description='Display a horizon mask from a set of data files.')
    parser.add_option('-a', '--baseline', dest='baseline', type="string", metavar='BASELINE', default='A1A1',
                      help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
    parser.add_option('-o', '--output', dest='output', type="string", metavar='OUTPUTFILE', default=None,
                      help="Write out intermediate h5 file")
    parser.add_option('-s', '--split', dest='split', action="store_true", metavar='SPLIT', default=False,
                      help="Whether to split each horizon plot in half")
    parser.add_option('-z', '--azshift', dest='azshift', type='float', metavar='AZIMUTH_SHIFT', default=45.0,
                      help="Degrees to rotate azimuth window by.")
    parser.add_option('--temp-limit', dest='temp_limit', type='float', default=40.0,
                      help="The Tempreture Limit to make the cut-off for the mask. This is calculated "
                           "as the T_sys at zenith plus the atmospheric noise contrabution at 10 degrees"
                           "elevation as per R.T. 199  .")
    parser.add_option("-n", "--nd-models",
                      help="Name of optional directory containing noise diode model files")


    (opts, args) = parser.parse_args()

# Check arguments
    if len(args) < 1:
        raise RuntimeError('Please specify the data file to reduce')

    # Load data set
    gridtemp = []
    for filename in args:
        print 'Loading baseline', opts.baseline, 'from data file', filename
        d = scape.DataSet(filename, baseline=opts.baseline,nd_models=opts.nd_models)
        if len(d.freqs) > 1:
            # Only keep main scans (discard slew and cal scans) a
            d = d.select(freqkeep=range(200, 800))
            d = remove_rfi(d,width=7,sigma=5)
            d = d.convert_power_to_temperature(min_duration=3, jump_significance=4.0)
            d = d.select(flagkeep='~nd_on')
            d = d.select(labelkeep='scan', copy=False)
            # Average all frequency channels into one band
            d.average()

        # Extract azimuth and elevation angle from (azel) target associated with scan, in degrees
        azimuth, elevation, temp = [], [], []
        for s in d.scans:
            azimuth.extend(rad2deg(s.pointing['az']))
            elevation.extend(rad2deg(s.pointing['el']))
            temp.extend(tuple(np.sqrt(s.pol('HH')[:,0]*s.pol('VV')[:,0])))
        assert len(azimuth) == len(elevation) == len(temp), "sizes don't match"

        data = (azimuth, elevation, temp)
        np.array(azimuth)<-89
        print "Gridding the data"
        print "data shape = ",np.shape(data[0]+(np.array(azimuth)[np.array(azimuth)<-89]+360.0).tolist())
        print np.shape(data[1]+np.array(elevation)[np.array(azimuth)<-89].tolist())
        print np.shape(data[2]+np.array(temp)[np.array(azimuth)<-89].tolist())
        gridtemp.append(mlab.griddata(data[0]+(np.array(azimuth)[np.array(azimuth)<-89]+360.0).tolist(), data[1]+np.array(elevation)[np.array(azimuth)<-89].tolist(), data[2]+np.array(temp)[np.array(azimuth)<-89].tolist(), np.arange(-90,271,1), np.arange(4,16,0.1)))
        # The +361 is to ensure that the point are well spaced,
        #this offset is not a problem as it is just for sorting out a boundery condition
        print "Completed Gridding the data"

    print "Making the mask"
    mask = gridtemp[0] >= opts.temp_limit
    for grid in gridtemp:
        mask = mask * (grid >= opts.temp_limit)
    maskr = np.zeros((len(np.arange(-90,271,1)),2))
    for i,az in enumerate(np.arange(-90,271,1)):
        print 'at az %f'%(az,)
        maskr[i] = az,np.max(elevation)
        for j,el in enumerate(np.arange(4,16,0.1)):
            if ~mask.data[j,i] and ~mask.mask[j,i] :
                maskr[i] = az,el
                break
    np.savetxt('horizon_mask_%s.dat'%(opts.baseline),maskr[1:,:])
    #plt.figure()
    #plt.subplot(1, 1, 1)
    #plt.plot(maskr[1:,0],maskr[1:,1])
    #az_title,el_title,big_title = ('Azimuth (deg)', 'Elevation (deg)', 'Mask for %s' % (opts.baseline,))
    #plt.xlabel(az_title)
    #plt.ylabel(el_title)
    #plt.ylim(0,15)
    #plt.title(big_title)
    #plt.show()

if __name__ == "__main__":
    main()
