#!/usr/bin/python
# Produce various scans across a simulated DBE target producing scan data for signal displays and loading into scape (using CaptureSession).

from __future__ import with_statement

import katuilib

target = 'Takreem,azel,45,10'
Session = katuilib.observe.CaptureSession

with katuilib.tbuild() as kat:

    # Tell the DBE sim to make a test target at specfied az and el and with specified flux
    kat.dbe.req.dbe_test_target(45, 10, 100)
    nd_params = {'diode' : 'coupler', 'on' : 3.0, 'off' : 3.0, 'period' : 40.}

    with Session(kat, 'id', 'nobody', 'The scan zoo', kat.ants, True) as session:
        session.standard_setup(1822., 1., nd_params)
        session.track(target, duration=5.0)
        session.fire_noise_diode('coupler', 5.0, 5.0)
        session.scan(target, duration=20.0)
        session.raster_scan(target, scan_duration=20.0)
