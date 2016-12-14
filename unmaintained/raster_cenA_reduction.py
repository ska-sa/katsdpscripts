#
# Produce image of Centaurus A from data taken by Fringe Finder on 17 March 2010.
#
# Ludwig Schwardt
# 26 March 2010
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scape
import katpoint
import scikits.fitting as fit

# Load temporary noise diode models
a1h = np.loadtxt('noise_diode_models/T_nd_A1H_coupler.txt', delimiter=',')
a1v = np.loadtxt('noise_diode_models/T_nd_A1V_coupler.txt', delimiter=',')
a2h = np.loadtxt('noise_diode_models/T_nd_A2H_coupler.txt', delimiter=',')
a2v = np.loadtxt('noise_diode_models/T_nd_A2V_coupler.txt', delimiter=',')

# Load data set and do standard continuum reduction
d = scape.DataSet('1268855687.h5', baseline='A1A1')
d.nd_model = scape.gaincal.NoiseDiodeModel(a1h, a1v, std_temp=0.04)
d = d.select(freqkeep=range(95, 380))
d.convert_power_to_temperature()
d = d.select(labelkeep='scan', copy=False)
d.average()

# Edit out some RFI
d.scans = d.compscans[0].scans
d.scans[37] = d.scans[37].select(timekeep=range(76), copy=True)
d.scans[38] = d.scans[38].select(timekeep=range(12, len(d.scans[38].timestamps)), copy=True)
d.scans[72] = d.scans[72].select(timekeep=range(2, len(d.scans[72].timestamps)), copy=True)

# Replace target coordinates with (ra,dec) offsets instead of (az,el) offsets
target = d.compscans[0].target
for scan in d.scans:
    ra_dec = np.array([katpoint.construct_azel_target(az, el).radec(t, d.antenna)
                       for az, el, t in zip(scan.pointing['az'], scan.pointing['el'], scan.timestamps)])
    scan.target_coords = np.array(target.sphere_to_plane(ra_dec[:,0], ra_dec[:,1], scan.timestamps, coord_system='radec'))

# Fit standard Gaussian beam and baselines
d.fit_beams_and_baselines(circular_beam=False)
# Fit linear baselines for all scans that did not get refined baselines in the standard fit
for n, scan in enumerate(d.scans):
    if scan.baseline is None:
        scan_power = scan.pol('I').squeeze()
        # Get confidence interval based on radiometer equation
        dof = 2.0 * 2.0 * (d.bandwidths[0] * 1e6) / d.dump_rate
        mean = scan_power.min()
        upper = scape.stats.chi2_conf_interval(dof, mean)[1]
        # Move baseline down as low as possible, taking confidence interval into account
        baseline = fit.Polynomial1DFit(max_degree=1)
        fit_region = np.arange(len(scan_power))
        for iteration in range(7):
            baseline.fit(scan.timestamps[fit_region], scan_power[fit_region])
            bl_resid = scan_power - baseline(scan.timestamps)
            next_fit_region = bl_resid < 1.0 * (upper - mean)
            if not next_fit_region.any():
                break
            else:
                fit_region = next_fit_region
        d.scans[n].baseline = baseline

# Obtain projected ra, dec coordinates and total power
target = d.compscans[0].target
ra, dec = [], []
for scan in d.scans:
    if scan.baseline:
        ra_dec = np.array([katpoint.construct_azel_target(az, el).radec(t, d.antenna)
                           for az, el, t in zip(scan.pointing['az'], scan.pointing['el'], scan.timestamps)])
        x, y = target.sphere_to_plane(ra_dec[:,0], ra_dec[:,1], scan.timestamps, coord_system='radec')
        ra.append(x)
        dec.append(y)
# Remove pointing offset (order of a few arcminutes)
ra = katpoint.rad2deg(np.hstack(ra) - d.compscans[0].beam.center[0])
dec = katpoint.rad2deg(np.hstack(dec) - d.compscans[0].beam.center[1])
power = np.hstack([scan.pol('I').squeeze() - scan.baseline(scan.timestamps) for scan in d.scans if scan.baseline])
power = np.abs(power)

# Grid the raster scan to projected plane
min_num_pixels = 201
interp = fit.Delaunay2DScatterFit(default_val=0.0, jitter=True)
interp.fit([ra, dec], power)
ra_range, dec_range = ra.max() - ra.min(), dec.max() - dec.min()
# Use a square pixel size in projected plane
pixel_size = min(ra_range, dec_range) / min_num_pixels
grid_ra = np.arange(ra.min(), ra.max(), pixel_size)
grid_dec = np.arange(dec.min(), dec.max(), pixel_size)
mesh_ra, mesh_dec = np.meshgrid(grid_ra, grid_dec)
mesh = np.vstack((mesh_ra.ravel(), mesh_dec.ravel()))
# This is already in transposed form, as contour plot expects (x => columns, y => rows)
smooth_power = interp(mesh).reshape(grid_dec.size, grid_ra.size)
smooth_rel_power = smooth_power * 100.0 / power.max()

start_time = d.scans[0].timestamps[0]
target_ra, target_dec = target.radec(start_time)
x, y = grid_ra + katpoint.rad2deg(target_ra), grid_dec + katpoint.rad2deg(target_dec)
scape.save_fits_image('CenA_1836MHz_ant1.fits', np.flipud(x), y, np.fliplr(smooth_power), target_name='Centaurus A',
                      coord_system='radec', projection_type='ARC', data_unit=d.data_unit,
                      freq_Hz=d.freqs[0] * 1e6, bandwidth_Hz=d.bandwidths[0] * 1e6, pol='I',
                      observe_date=katpoint.Timestamp(start_time).to_string().replace(' ', 'T'),
                      create_date=katpoint.Timestamp().to_string().replace(' ', 'T'),
                      telescope='KAT-4', instrument='ant1', observer='schwardt', clobber=True)

# Do contour plot
plt.figure(1, figsize=(9.4, 9.1))
plt.clf()
levels = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100])
cfs = plt.contourf(grid_ra, grid_dec, smooth_rel_power, levels, norm=mpl.colors.LogNorm(0.1, 101.0))
cs = plt.contour(grid_ra, grid_dec, smooth_rel_power, levels, colors='k')
#plt.imshow(mpl.image.imread('cena_clean.png'), extent=[-0.25, 0.25, 0.25, -0.25], zorder=100)
# Add rectangle indicating border of interferometric image
plt.plot(0.25 * np.array([-1, -1, 1, 1, -1]), 0.25 * np.array([-1, 1, 1, -1, -1]), 'w')
ax = plt.gca()
plt.axis('image')
plt.axis([4.25, -2.5, -4.75, 5])
plt.xlabel('RA offset (J2000 degrees)')
plt.ylabel('DEC offset (J2000 degrees)')
#plt.title('Radio continuum emission from Cen A at 1836 MHz')
plt.title('Raster scan image of Centaurus A')
cbar = plt.colorbar(cfs)
plt.colorbar(cs, cax=cbar.ax, ticks=levels)
cbar.set_label('Percentage of peak total power')

# Get beamwidth
lamb = katpoint.lightspeed / (d.freqs[0] * 1e6)
beamwidth_deg = katpoint.rad2deg(1.2 * (lamb / d.antenna.diameter))
# Plot beam ellipse
bmaj, bmin, bpa = beamwidth_deg, beamwidth_deg, 0.0
gap = 0.15
maxwidth = (1.0 + 2.0 * gap) * max(bmaj, bmin)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
corner = (xlim[0], ylim[1])
axisdir = ((np.diff(xlim) / np.abs(np.diff(xlim)))[0],
           (np.diff(ylim) / np.abs(np.diff(ylim)))[0])
beamCenter = (corner[0] + axisdir[0] * maxwidth / 2.0,
              corner[1] - axisdir[1] * maxwidth / 2.0)
cornerClosestToOrigin = (corner[0] + axisdir[0] * maxwidth,
                         corner[1] - axisdir[1] * maxwidth)
beamBorder = mpl.patches.Rectangle(cornerClosestToOrigin, maxwidth,
                                   maxwidth, ec='k', fc='w', zorder=4)
beamEll = mpl.patches.Ellipse(beamCenter, bmaj, bmin, 90.0 - bpa,
                              ec='k', fc='0.3', zorder=5)
ax.add_patch(beamBorder)
ax.add_patch(beamEll)

plt.savefig('CenA_1836MHz_ant1.pdf')

# Do raster plot with superimposed Moon
try:
    # First try to use an image of the Moon
    moon = mpl.image.imread('moon.jpg')
except IOError:
    # If no image is found, use a white disk
    xx, yy = np.arange(-50,51), np.arange(-50,51)
    moon = np.array(xx[np.newaxis, :] ** 2 + yy[:, np.newaxis] ** 2 <= 50 ** 2, dtype=np.float64)
    moon = np.dstack([moon, moon, moon])

plt.figure(2)
plt.clf()
plt.imshow(np.fliplr(10*np.sqrt(smooth_rel_power)), cmap='gray', origin='lower',
           extent=[grid_ra[-1], grid_ra[0], grid_dec[0], grid_dec[-1]])
plt.imshow(moon, extent=[3.75, 3.25, 3.75, 4.25], origin='lower')
plt.axis([4.25, -2.5, -4.75, 5])
plt.xlabel('RA offset (J2000 degrees)')
plt.ylabel('DEC offset (J2000 degrees)')
plt.title('Radio continuum emission from Cen A at 1836 MHz')

plt.savefig('CenA_1836MHz_moon.png')
