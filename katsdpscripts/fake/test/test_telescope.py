from katsdpscripts.fake.telescope import FakeTelescope

kat = FakeTelescope('katsdpscripts/fake/rts_model.cfg')
kat.dry_run = True
kat.rcps.req.sensor_sampling('lock', 'event')
kat.m062.req.sensor_sampling('pos_actual_scan_azim', 'period', 2.0)
kat.m062.req.sensor_sampling('pos_actual_scan_elev', 'period', 2.0)
kat.rcps.req.target('azel, 20, 30')
kat.rcps.req.mode('POINT')
kat.rcps.wait('lock', True, 300)
kat.m062.sensor.activity.get_value()
kat.rcps.req.mode('STOW')
kat.rcps.wait('lock', True, 300)
