#
# Compare simplistic delay calculation to that of CalcServer 1.1.
#
# This modifies checkCalcServer (checkcalc.c) as part of the check.
#
# Also check the fringe rates of first fringes on Fringe Finder.
#
# Ludwig Schwardt
# 1 December 2009
#

import katpoint
import ephem

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
ant_a = katpoint.construct_antenna('EC, 0, 0, -6378137.0, 0.0')
# Station B is Kitt Peak VLBA antenna
kp_lla = katpoint.ecef_to_lla(-1995678.4969, -5037317.8209, 3357328.0825)
kp_lat, kp_long, kp_alt = katpoint.rad2deg(kp_lla[0]), katpoint.rad2deg(kp_lla[1]), kp_lla[2]
ant_a = katpoint.construct_antenna('KP, %.16f, %.16f, %.16f, 25.0, 0, 0, 0' % (kp_lat, kp_long, kp_alt))
# Station A is an imaginary antenna close to Kitt Peak (south-east of it)
ant_a = katpoint.construct_antenna('RF, %.16f, %.16f, %.16f, 25.0, 63.336, -63.336, 0' % (kp_lat, kp_long, kp_alt))
a_lla = (ant_a.observer.lat, ant_a.observer.long, ant_a.observer.elevation)
##### Calculate ECEF coords - add these to checkcalc.c as Station A x, y, z coords
a_ecef = katpoint.lla_to_ecef(*a_lla)
##### Set axis offsets of both stations = 0.0 in checkcalc.c

# Source is the first millisecond pulsar, PSR B1937+21
target = katpoint.construct_target('PSR B1937+21, radec J2000, 19:39:38.560210, 21:34:59.141000')
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
baseline_m = - np.array(ant_a.offset)
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

# Next, try some Fringe Finder specific settings
################################################

# Pick Fringe Finder antennas 1 and 2 as Stations A and B, respectively
ref_lla = '-30:43:17.34, 21:24:38.46, 1038.0'
ant_a = katpoint.construct_antenna('FF1, %s, 12.0, 18.4, -8.7, 0.0' % ref_lla)
ant_b = katpoint.construct_antenna('FF2, %s, 12.0, 86.2, 25.5, 0.0' % ref_lla)
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
target = katpoint.construct_target('J0535-0523 | *Orion A | OriA | M42, radec J2000, 5:35:17.3, -5:23:28.0')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)

# CALC has:
el_a_calc, az_a_calc = 1.853262e-01, 1.570418e+00
el_b_calc, az_b_calc = 1.853368e-01, 1.570413e+00

print 'Station B azimuth error   = %g arcsec' % (katpoint.rad2deg(az_b - az_b_calc) * 3600,)
print 'Station B elevation error = %g arcsec' % (katpoint.rad2deg(el_b - el_b_calc) * 3600,)

# Assume Antenna A is the array reference position - baseline is B - A
baseline_m = np.array(ant_b.offset) - np.array(ant_a.offset)
# Get geometric delay from direct dot product
source_vec = azel_to_enu(az_b, el_b)
geom_delay = - np.dot(baseline_m, source_vec) / katpoint.lightspeed
# Get source position a second later, and use it to derive delay rate
az_b_1s, el_b_1s = target.azel(t + 1.0, ant_b)
source_vec_1s = azel_to_enu(az_b_1s, el_b_1s)
geom_delay_1s = - np.dot(baseline_m, source_vec_1s) / katpoint.lightspeed
delay_rate = geom_delay_1s - geom_delay
# Expected fringe period in seconds
fringe_period = 1.0 / ((1.5 + (350 - 256)/512.*0.4) * 1e9 * delay_rate)

# CALC has:
delay_calc, delay_rate_calc = -2.2232772665e-07, -1.5603396713e-12

print 'Delay error = %g ns (%g %%)' % ((geom_delay - delay_calc) * 1e9,
                                       100 * (geom_delay / delay_calc - 1.0))
print 'Delay rate error = %g sec / sec (%g %%)' % (delay_rate - delay_rate_calc,
                                                   100 * (delay_rate / delay_rate_calc - 1.0))

# We choose the Moon...
t = katpoint.Timestamp('2009-12-01 17:27:00')
target = katpoint.construct_target('Moon, special')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)
# Get geometric delay from direct dot product
source_vec = azel_to_enu(az_b, el_b)
geom_delay = - np.dot(baseline_m, source_vec) / katpoint.lightspeed
# Get source position a second later, and use it to derive delay rate
az_b_1s, el_b_1s = target.azel(t + 1.0, ant_b)
source_vec_1s = azel_to_enu(az_b_1s, el_b_1s)
geom_delay_1s = - np.dot(baseline_m, source_vec_1s) / katpoint.lightspeed
delay_rate = geom_delay_1s - geom_delay
# Expected fringe period in seconds
fringe_period = 1.0 / ((1.5 + (350 - 256)/512.*0.4) * 1e9 * delay_rate)

# And the Sun...
t = katpoint.Timestamp('2009-12-01 17:00:00')
target = katpoint.construct_target('Sun, special')
az_a, el_a = target.azel(t, ant_a)
az_b, el_b = target.azel(t, ant_b)
# Get geometric delay from direct dot product
source_vec = azel_to_enu(az_b, el_b)
geom_delay = - np.dot(baseline_m, source_vec) / katpoint.lightspeed
# Get source position a second later, and use it to derive delay rate
az_b_1s, el_b_1s = target.azel(t + 1.0, ant_b)
source_vec_1s = azel_to_enu(az_b_1s, el_b_1s)
geom_delay_1s = - np.dot(baseline_m, source_vec_1s) / katpoint.lightspeed
delay_rate = geom_delay_1s - geom_delay
# Expected fringe period in seconds
fringe_period = 1.0 / ((1.5 + (350 - 256)/512.*0.4) * 1e9 * delay_rate)
