#!/usr/bin/python
# Compare simplistic delay calculation to that of CalcServer 1.1.
#
# This modifies checkCalcServer (checkcalc.c) as part of the check.
#
# Also check the fringe rates of first fringes on KAT.
#
# Ludwig Schwardt
# 1 December 2009
#

import katpoint
import ephem
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

# First use original settings in checkcalc.c as far as possible
###############################################################

# Timestamp at which to evaluate delay, as UTC Modified Julian Day (MJD)
t_mjd = 50774.0 + 22.0/24.0 + 2.0/(24.0*60.0)
# Convert to Julian Date (JD) and Dublin Julian Day (DJD), as used by ephem
t_jd = t_mjd + 2400000.5
t_djd = t_jd - 2415020
# Convert to UTC seconds since Unix epoch
t = katpoint.Timestamp(ephem.Date(t_djd))
# Estimated UT1-UTC offset at time t, from EOP values in checkCalcServer
ut1_offset = 0.2818
t = t + ut1_offset

# Station A used to be the centre of the Earth (also reference position?)
ant_a = katpoint.Antenna('EC, 0, 0, -6378137.0, 0.0')
# Station B is Kitt Peak VLBA antenna
kp_lla = katpoint.ecef_to_lla(-1995678.4969, -5037317.8209, 3357328.0825)
kp_lat, kp_long, kp_alt = katpoint.rad2deg(kp_lla[0]), katpoint.rad2deg(kp_lla[1]), kp_lla[2]
ant_b = katpoint.Antenna('KP, %.16f, %.16f, %.16f, 25.0, 0 0 0' % (kp_lat, kp_long, kp_alt))
# Station A is an imaginary antenna close to Kitt Peak (south-east of it)
ant_a = katpoint.Antenna('RF, %.16f, %.16f, %.16f, 25.0, 63.336 -63.336 0' % (kp_lat, kp_long, kp_alt))
a_lla = (ant_a.observer.lat, ant_a.observer.long, ant_a.observer.elevation)
##### Calculate ECEF coords - add these to checkcalc.c as Station A x, y, z coords
a_ecef = katpoint.lla_to_ecef(*a_lla)
##### Set axis offsets of both stations = 0.0 in checkcalc.c

# Source is the first millisecond pulsar, PSR B1937+21
target = katpoint.Target('PSR B1937+21, radec J2000, 19:39:38.560210, 21:34:59.141000')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)

# CALC has:
el_a_calc, az_a_calc = 1.286942e+00, 2.202087e+00
el_b_calc, az_b_calc = 1.286928e+00, 2.202088e+00

print 'Station B azimuth error   = %g arcsec' % (katpoint.rad2deg(az_b - az_b_calc) * 3600,)
print 'Station B elevation error = %g arcsec' % (katpoint.rad2deg(el_b - el_b_calc) * 3600,)

# Convert (az, el) to unit vector in ENU coordinates
def azel_to_enu(az, el):
    sin_az, cos_az = np.sin(az), np.cos(az)
    sin_el, cos_el = np.sin(el), np.cos(el)
    return np.array([sin_az * cos_el, cos_az * cos_el, sin_el])

# Assume Antenna A is the array reference position - baseline is minus offset of ant A
baseline_m = - np.array(ant_a.position_enu)
# Get geometric delay from direct dot product
source_vec = azel_to_enu(az_b, el_b)
geom_delay = - np.dot(baseline_m, source_vec) / katpoint.lightspeed
# Get source position a second later, and use it to derive delay rate
az_b_1s, el_b_1s = target.azel(t + 1.0, ant_b)
source_vec_1s = azel_to_enu(az_b_1s, el_b_1s)
geom_delay_1s = - np.dot(baseline_m, source_vec_1s) / katpoint.lightspeed
delay_rate = geom_delay_1s - geom_delay

# CALC has:
delay_calc, delay_rate_calc = 8.2741374095e-08, -1.2053192132e-11

print 'Delay error = %g ns (%g %%)' % ((geom_delay - delay_calc) * 1e9,
                                       100 * (geom_delay / delay_calc - 1.0))
print 'Delay rate error = %g sec / sec (%g %%)' % (delay_rate - delay_rate_calc,
                                                   100 * (delay_rate / delay_rate_calc - 1.0))

# Next, try some KAT specific settings
################################################

# Pick KAT antennas 1 and 2 as Stations A and B, respectively
ref_lla = '-30:43:17.34, 21:24:38.46, 1038.0'
ant_a = katpoint.Antenna('FF1, %s, 12.0, 18.4 -8.7 0.0' % ref_lla)
ant_b = katpoint.Antenna('FF2, %s, 12.0, 86.2 25.5 0.0' % ref_lla)
##### Calculate ECEF coords - add these to checkcalc.c as Station A and B x, y, z coords
a_lla = (ant_a.observer.lat, ant_a.observer.long, ant_a.observer.elevation)
a_ecef = katpoint.lla_to_ecef(*a_lla)
b_lla = (ant_b.observer.lat, ant_b.observer.long, ant_b.observer.elevation)
b_ecef = katpoint.lla_to_ecef(*b_lla)

# Choose the date and time of first fringes
t = katpoint.Timestamp('2009-12-01 18:03:55')
##### Convert to MJD and insert into checkcalc.c
t_djd = t.to_ephem_date()
t_jd = t_djd + 2415020
t_mjd = t_jd - 2400000.5

# Estimated UT1-UTC offset at time t, from EOP values in checkCalcServer
##### Just change the EOP dates and TAI-UTC value (= 34)
ut1_offset = 0.2820
t = t + ut1_offset

##### Source is Orion A - insert in checkcalc.c
target = katpoint.Target('J0535-0523 | *Orion A | OriA | M42, radec J2000, 5:35:17.3, -5:23:28.0')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)

# CALC has:
el_a_calc, az_a_calc = 1.853262e-01, 1.570418e+00
el_b_calc, az_b_calc = 1.853368e-01, 1.570413e+00

print 'Station B azimuth error   = %g arcsec' % (katpoint.rad2deg(az_b - az_b_calc) * 3600,)
print 'Station B elevation error = %g arcsec' % (katpoint.rad2deg(el_b - el_b_calc) * 3600,)

# Assume Antenna A is the array reference position - baseline is B - A
geom_delay, delay_rate = target.geometric_delay(ant_b, t, ant_a)
# Expected fringe period in seconds
fringe_period = 1.0 / ((1.5 - (350 - 256)/512.*0.4) * 1e9 * delay_rate)
uvw = target.uvw(ant_b, t, ant_a)

# CALC has:
delay_calc, delay_rate_calc = -2.2232772665e-07, -1.5603396713e-12
uvw_calc = (6.476748e+00, -3.580610e+01, -6.665174e+01)

print 'Delay error = %g ns (%g %%)' % ((geom_delay - delay_calc) * 1e9,
                                       100 * (geom_delay / delay_calc - 1.0))
print 'Delay rate error = %g s / s (%g %%)' % (delay_rate - delay_rate_calc,
                                               100 * (delay_rate / delay_rate_calc - 1.0))
print 'Fringe period = %g s' % (fringe_period,)

# We choose the Moon...
t = katpoint.Timestamp('2009-12-01 17:27:00')
target = katpoint.Target('Moon, special')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)
# Get geometric delay from direct dot product
geom_delay, delay_rate = target.geometric_delay(ant_b, t, ant_a)
# Expected fringe period in seconds
fringe_period = 1.0 / ((1.5 - (350 - 256)/512.*0.4) * 1e9 * delay_rate)

# And the Sun...
t = katpoint.Timestamp('2009-12-01 17:00:00')
target = katpoint.Target('Sun, special')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)
# Get geometric delay from direct dot product
geom_delay, delay_rate = target.geometric_delay(ant_b, t, ant_a)
# Expected fringe period in seconds
fringe_period = 1.0 / ((1.5 - (350 - 256)/512.*0.4) * 1e9 * delay_rate)

# How about an all-sky fringe pattern for the first baseline?
#############################################################

baseline_m = ant_a.baseline_toward(ant_b)
lat = ant_a.observer.lat
rf_freq = 1.8e9

# In terms of (az, el)
x_range, y_range = np.linspace(-1., 1., 201), np.linspace(-1., 1., 201)
x_grid, y_grid = np.meshgrid(x_range, y_range)
xx, yy = x_grid.flatten(), y_grid.flatten()
outside_circle = xx * xx + yy * yy > 1.0
xx[outside_circle] = yy[outside_circle] = np.nan
az, el = katpoint.plane_to_sphere['SIN'](0.0, np.pi / 2.0, xx, yy)

source_vec = katpoint.azel_to_enu(az, el)
geom_delay = -np.dot(baseline_m, source_vec) / katpoint.lightspeed
turns = geom_delay * rf_freq
phase = turns - np.floor(turns)

plt.figure(1)
plt.clf()
plt.imshow(phase.reshape(x_grid.shape), origin='lower',
           extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]])

# In terms of (ha, dec)
# One second resolution on hour angle - picks up fast fringes that way
ha_range = np.linspace(-12., 12., 86401.)
dec_range = np.linspace(-90., katpoint.rad2deg(lat) + 90., 101)
ha_grid, dec_grid = np.meshgrid(ha_range, dec_range)
hh, dd = ha_grid.flatten(), dec_grid.flatten()

source_vec = katpoint.hadec_to_enu(hh  / 12. * np.pi, katpoint.deg2rad(dd), lat)
geom_delay = -np.dot(baseline_m, source_vec) / katpoint.lightspeed
geom_delay = geom_delay.reshape(ha_grid.shape)
turns = geom_delay * rf_freq
phase = turns - np.floor(turns)
fringe_rate = np.diff(geom_delay, axis=1) / (np.diff(ha_range) * 3600.) * rf_freq

plt.figure(2)
plt.clf()
plt.imshow(phase, origin='lower', aspect='auto',
           extent=[ha_range[0], ha_range[-1], dec_range[0], dec_range[-1]])
plt.xlabel('Hour angle (hours)')
plt.ylabel('Declination (degrees)')
plt.title('Fringe phase across sky for given baseline')
plt.colorbar()

plt.figure(3)
plt.clf()
plt.imshow(turns, origin='lower', aspect='auto',
           extent=[ha_range[0], ha_range[-1], dec_range[0], dec_range[-1]])
plt.xlabel('Hour angle (hours)')
plt.ylabel('Declination (degrees)')
plt.title('Geometric delay (number of turns) across sky for given baseline')
plt.colorbar()

plt.figure(4)
plt.clf()
plt.imshow(fringe_rate, origin='lower', aspect='auto',
           extent=[ha_range[0], ha_range[-2], dec_range[0], dec_range[-1]])
plt.xlabel('Hour angle (hours)')
plt.ylabel('Declination (degrees)')
plt.title('Geometric fringe rate (turns / s) across sky for given baseline')
plt.colorbar()

# Now predict the visibility magnitude for the Sun across the band
##################################################################

# Jinc function
def jinc(x):
    j = np.ones(x.shape)
    # Handle 0/0 at origin
    nonzero_x = abs(x) > 1e-20
    j[nonzero_x] = 2 * sp.j1(np.pi * x[nonzero_x]) / (np.pi * x[nonzero_x])
    return j

# Use KAT antennas 1 and 2
ref_lla = '-30:43:17.34, 21:24:38.46, 1038.0'
ant1 = katpoint.Antenna('FF1, %s, 12.0, 18.4 -8.7 0.0' % ref_lla)
ant2 = katpoint.Antenna('FF2, %s, 12.0, 86.2 25.5 0.0' % ref_lla)

# Channel frequencies
band_center = 1822.
channel_bw = 400. / 512
num_chans = 512
freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
channels = range(100, 400)
# Equivalent wavelength, in m
lambdas = katpoint.lightspeed / (freqs[channels] * 1e6)
# Timestamps for observation
t = np.array([katpoint.Timestamp('2009-12-10 06:19:40.579')]) + np.linspace(0, 2700., 2700.)

# Set up the Sun as target
target = katpoint.Target('Sun, special')
# Angular diameter of the Sun (about 32 arcminutes), in radians
diam = katpoint.deg2rad(32.0 / 60.0)

# Get (u,v,w) coordinates (in meters) as a function of time
u, v, w = target.uvw(ant2, t, ant1)
# Normalised uv distance, in wavelengths
uvdist = np.outer(np.sqrt(u ** 2 + v ** 2), 1.0 / lambdas)
# Normalised w distance, in wavelengths (= turns of geometric delay) (also add cable delay)
wdist = np.outer(w - 20, 1.0 / lambdas)
# Contribution from sunspot 1034
spot_angle = katpoint.deg2rad(160.)
sunspot_ripple = np.outer(np.cos(spot_angle) * u + np.sin(spot_angle) * v, 1.0 / lambdas)
sunspots = 0.02 * np.exp(1j * 2 * np.pi * 0.96 * 0.5 * diam * sunspot_ripple) + \
           0.02 * np.exp(1j * 2 * np.pi * 0.92 * 0.5 * diam * sunspot_ripple)
# Contribution from limb-brightening
limbs = 0.05 * np.cos(2 * np.pi * 0.9 * 0.5 * diam * np.outer(u, 1.0 / lambdas))
# Calculate normalised coherence function (see Born & Wolf, Section 10.4.2, p. 574-576)
coh = (jinc(diam * uvdist) + sunspots) * np.exp(1j * 2 * np.pi * wdist)

plt.figure(5)
plt.clf()
plt.imshow(np.abs(coh), origin='lower', aspect='auto',
           extent=[0, channels[-1] - channels[0], t[-1] - t[0], 0.0])
plt.colorbar()

plt.figure(6)
plt.clf()
plt.imshow(np.angle(coh), origin='lower', aspect='auto',
           extent=[0, channels[-1] - channels[0], t[-1] - t[0], 0.0])
plt.colorbar()

plt.figure(7)
plt.clf()
plt.subplot(311)
plt.plot(t - t[0], coh[:, 0].real)
plt.subplot(312)
plt.plot(t - t[0], coh[:, 100].real)
plt.subplot(313)
plt.plot(t - t[0], coh[:, 200].real)

plt.show()
