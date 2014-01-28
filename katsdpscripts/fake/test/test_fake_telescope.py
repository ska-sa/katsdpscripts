from katscripts.fake_telescope import FakeTelescope

kat = FakeTelescope('katscripts/rts_model.cfg')
kat.dry_run = True
kat.ants.req.sensor_sampling('lock', 'event')
kat.m062.req.sensor_sampling('pos_actual_scan_azim', 'period', 2.0)
kat.m062.req.sensor_sampling('pos_actual_scan_elev', 'period', 2.0)
kat.ants.req.target('azel, 20, 30')
kat.ants.req.mode('POINT')
kat.ants.wait('lock', True, 300)
kat.m062.sensor.activity.get_value()
kat.ants.req.mode('STOW')
kat.ants.wait('lock', True, 300)
