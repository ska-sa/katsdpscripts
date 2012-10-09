#!/usr/bin/python
# Plot horizon mask

import sys
import optparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import katfile

def plot_horizon(xyz_data, xyz_titles, az_lims, el_lims):
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
    #pow_min, pow_max = pow_lims
    az_pos = np.linspace(az_min, az_max, (az_max - az_min) / 0.1)
    el_pos = np.linspace(el_min, el_max, (el_max - el_min) / 0.1)
    power_db_pos = mlab.griddata(azimuth[:,0], elevation[:,0], power_db[:,0], az_pos, el_pos)
    plt.imshow(power_db_pos, aspect='auto', origin='lower')
    #cs = plt.contour(az_pos, el_pos, power_db_pos, pow_levels)
    #plt.contourf(az_pos,el_pos, power_db_pos, pow_levels, antialiased=True)
    plt.colorbar()

    if az_title:
        plt.xlabel(az_title)
    if el_title:
        plt.ylabel(el_title)
    if pow_title:
        plt.title(pow_title)

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file> [<data file> ...]',description='Display a horizon mask from a set of data files.')
parser.add_option('-a', '--baseline', dest='baseline',type="string", metavar='BASELINE', default='ant1',help="Baseline to load (e.g. 'ant1' for antenna 1),\
                default is first single-dish baseline in file")
parser.add_option('-o', '--output', dest='output', type="string", metavar='OUTPUTFILE', default=None,help="Write out intermediate h5 file")
parser.add_option('-s', '--split', dest='split', action="store_true", metavar='SPLIT', default=False,help="Whether to split each horizon plot in half")
parser.add_option('-p', '--pol', dest = 'pol',type ='string',metavar ='POLARIZATION', default = 'HH',help = 'Polarization to load (e.g. HH for horizontal polarization ),\
                the default is the horizontal polarization')
    
(opts, args) = parser.parse_args()
# Check arguments
if len(args) < 1:
    print 'Please specify the data file to reduce'
    sys.exit(1)

if opts.pol is None:
    print "please specify which polarization to load"
    sys.exit(1)

print 'Loading baseline', opts.baseline+'-'+opts.pol, 'from data file', args[0]
f = katfile.open(args[0])
f.select(ants=opts.baseline, pol=opts.pol, scans='scan', channels=range(90,425))
# Extract azimuth and elevation angle from (azel), in degrees
azimuth = f.az
elevation = f.el
power_linear = np.abs(f.vis[:])
power_db = 10.0 * np.log10(power_linear).sum(axis=1)
assert len(azimuth) == len(elevation) == len(power_db)
print "Contour plotting horizon from %d points ..." % len(azimuth)
    # Calculate and plot tipping curves
    #plt.figure(1)
    #plt.clf()
    #plt.subplots_adjust(hspace=0.5)
data = (azimuth, elevation, power_db)
titles = ('Azimuth (deg)', 'Elevation (deg)', 'Power (dB) for %s %s' % (opts.baseline,opts.pol))
az_min, az_max = 0.95*min(azimuth), 0.95*max(azimuth)
el_min, el_max = 1.05*min(elevation), 0.95*max(elevation)
pow_max, pow_min = max(power_db[:,0]), min(power_db[:,0])
az_mid = (az_max + az_min) / 2.0
el_mid = (el_max + el_min)/2.0
if opts.split:
    plt.subplot(2, 1, 1)
    plot_horizon(data, titles, (az_min, az_mid), (el_min, el_max))
    plt.subplot(2, 1, 2)
    plot_horizon(data,titles, (az_mid, az_max), (el_min, el_max))
else:
    plt.subplot(1, 1, 1)
    plot_horizon(data,titles, (az_min, az_max), (el_min, el_max))
# Display plots - this should be called ONLY ONCE, at the VERY END of the script
# The script stops here until you close the plots...
plt.show()
"""
## small tool
f =katfile.open('1349722968.h5')
f.select(ants='ant1', pol="HH", scans='scan', channels=range(90,425))
vis = f.vis[:]
abs_vis = np.abs(vis)
x = abs_vis.sum(axis=1)
az = f.az
el = f.el
az_min, az_max = 0.95*min(az), 0.95*max(az)
el_min, el_max = 1.05*min(el), 0.95*max(el)
el_pos = np.linspace(el_min, el_max, (el_max - el_min) / 0.1)
az_pos = np.linspace(az_min, az_max, (az_max - az_min) / 0.1)
p = mlab.griddata(az[:,0], el[:,0], x[:,0], az_pos, el_pos)
"""
