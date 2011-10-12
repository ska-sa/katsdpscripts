#!/usr/bin/python
# Plot horizon mask

import sys
import optparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import scape
from katpoint import rad2deg

def plot_horizon(xyz_data, xyz_titles, az_lims, el_lims, pow_lims):
    """Calculate and plot horizon.

    Parameters
    ----------
    xyz_data : tuple of (azimuth, elevation, power_db) data lists
        Data to plot.
    xyz_titles : tuple of titles for (azimuth, elevation and power_db) axes
        Titles for axes.
    az_lims : tuple of (min azimuth, max azimuth)
        Azimuth limits for the plot.
    el_lims : tuple of (min elevation, max elevation)
        Elevation limits for the plot.
    pow_lims : tuple of (min power, max power)
        Power limits for the plot.
    """
    azimuth, elevation, power_db = xyz_data
    az_title, el_title, pow_title = xyz_titles
    az_min, az_max = az_lims
    el_min, el_max = el_lims
    pow_min, pow_max = pow_lims
    az_pos = np.linspace(az_min, az_max, (az_max - az_min) / 0.1)
    el_pos = np.linspace(el_min, el_max, (el_max - el_min) / 0.1)
    pow_tick_min, pow_tick_max = np.floor(pow_min), np.ceil(pow_max)
    pow_levels = np.round(np.linspace(pow_tick_min, pow_tick_max, np.round((pow_tick_max - pow_tick_min) / 0.1) + 1), 1)
    pow_ticks = np.round(np.linspace(pow_tick_min, pow_tick_max, np.round((pow_tick_max - pow_tick_min) / 1.0) + 1), 0)
    power_db_pos = mlab.griddata(azimuth, elevation, power_db, az_pos, el_pos)
    cs = plt.contour(az_pos, el_pos, power_db_pos, pow_levels)
    plt.contourf(az_pos, el_pos, power_db_pos, pow_levels, antialiased=True)
    #plt.colorbar(cs, ticks=pow_ticks)
    plt.colorbar()
    plt.xlim(az_min, az_max)
    plt.ylim(el_min, el_max)

    if az_title:
        plt.xlabel(az_title)
    if el_title:
        plt.ylabel(el_title)
    if pow_title:
        plt.title(pow_title)
def main():
    # Parse command-line options and arguments
    parser = optparse.OptionParser(usage='%prog [options] <data file> [<data file> ...]',description='Display a horizon mask from a set of data files.')
    parser.add_option('-a', '--baseline', dest='baseline',type="string", metavar='BASELINE', default='A1A1',help="Baseline to load (e.g. 'A1A1' for antenna 1),default is first single-dish baseline in file")
    parser.add_option('-o', '--output', dest='output', type="string", metavar='OUTPUTFILE', default=None,help="Write out intermediate h5 file")
    parser.add_option('-s', '--split', dest='split', action="store_true", metavar='SPLIT', default=False,help="Whether to split each horizon plot in half")
    parser.add_option('-z', '--azshift', dest='azshift', type='float', metavar='AZIMUTH_SHIFT', default=45.0,help="Degrees to rotate azimuth window by.")
    parser.add_option('-p', '--pol', dest = 'pol',type ='string',metavar ='POLARIZATION', default = None,help = 'Polarization to load (e.g. HH for horizontal polarization ), there is no default polarization')
    (opts, args) = parser.parse_args()

    # Check arguments
    if len(args) < 1:
        print 'Please specify the data file to reduce'
        sys.exit(1)

    if opts.pol is None:
        print "please specify which polarization to load"
        sys.exit(1)
    # Load data set
    combined = None
    for filename in args:
        print 'Loading baseline', opts.baseline, 'from data file', filename
        d = scape.DataSet(filename, baseline=opts.baseline)
        if len(d.freqs) > 1:
            # Only keep main scans (discard slew and cal scans) and restrict frequency band to Fringe Finder band
            d = d.select(labelkeep='scan', freqkeep=range(90, 425))
            # Average all frequency channels into one band
            d.average()
        if combined is None:
            combined = d
        else:
            combined.scans.extend(d.scans)
            combined.compscans.extend(d.compscans)
    if not combined.scans:
        print 'No scans found. Did you specify a data file?'
        sys.exit(1)
    if opts.output is not None:
        combined.save(opts.output)
    # Extract azimuth and elevation angle from (azel) target associated with scan, in degrees
    azimuth, elevation,power_db = [], [], []
    for s in d.scans:
        #azimuth.extend(scape.stats.angle_wrap(rad2deg(s.pointing['az']) + opts.azshift, period=360.0) - opts.azshift)
        azimuth.extend(rad2deg(s.pointing['az']))
        elevation.extend(rad2deg(s.pointing['el']))
        power_db.extend(10.0 * np.log10(s.pol(opts.pol)[:,0]))
    assert len(azimuth) == len(elevation) == len(power_db)
    print "Contour plotting horizon from %d points ..." % len(azimuth)
    # Calculate and plot tipping curves
    #plt.figure(1)
    #plt.clf()
    #plt.subplots_adjust(hspace=0.5)
    data = (azimuth, elevation, power_db)
    titles = ('Azimuth (deg)', 'Elevation (deg)', 'Power (dB) for %s %s' % (opts.pol,opts.baseline,))
    az_max, az_min = np.ceil(max(azimuth)), np.floor(min(azimuth))
    el_max, el_min = np.ceil(max(elevation)), np.floor(min(elevation))
    pow_max, pow_min = max([max(power_db)]), min([min(power_db)])
    az_mid = (az_max + az_min) / 2.0
    el_mid = (el_max + el_min)/2.0
    if opts.split:
        plt.subplot(2, 1, 1)
        plot_horizon(data, titles, (az_min, az_mid), (el_min, el_max), (pow_min, pow_max))
        plt.subplot(2, 1, 2)
        plot_horizon(data,titles, (az_mid, az_max), (el_min, el_max), (pow_min, pow_max))
    else:
        plt.subplot(1, 1, 1)
        plot_horizon(data,titles, (az_min, az_max), (el_min, el_max), (pow_min, pow_max))
    # Display plots - this should be called ONLY ONCE, at the VERY END of the script
    # The script stops here until you close the plots...
    plt.show()

if __name__ == "__main__":
    main()
