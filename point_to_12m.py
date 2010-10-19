#! /usr/bin/python
# Silly script to point 4 antennas at the 12-m container

import katuilib

kat = katuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

print "Stowing all dishes"
kat.ants.req.target_azel(0.0, 90.0)
kat.ants.req.mode('POINT')
kat.ants.wait('lock', True, 300)

print "Pointing dishes at 12-m container"
kat.ant1.req.target_azel(145., 2.)
kat.ant2.req.target_azel(-145., 2.)
kat.ant3.req.target_azel(146., 2.)
kat.ant4.req.target_azel(136., 2.)

kat.ants.req.mode('POINT')
