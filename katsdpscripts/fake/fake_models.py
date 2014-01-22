import numpy as np

from katpoint import (Antenna, Target, rad2deg, deg2rad, wrap_angle,
                      construct_azel_target)

class FakeModel(object):
    def update(self, timestamp):
        pass


class AntennaPositionerModel(FakeModel):
    def __init__(self, description, max_slew_azim_dps, max_slew_elev_dps,
                 inner_threshold_deg, **kwargs):
        self.ant = Antenna(description)
        self.req_target('Zenith, azel, 0, 90')
        self.mode = 'POINT'
        self.activity = 'stow'
        self.lock = True
        self.lock_threshold = inner_threshold_deg
        self.pos_actual_scan_azim = self.pos_request_scan_azim = 0.0
        self.pos_actual_scan_elev = self.pos_request_scan_elev = 90.0
        self.max_slew_azim_dps = max_slew_azim_dps
        self.max_slew_elev_dps = max_slew_elev_dps
        self._last_update = 0.0

    def req_target(self, target):
        self.target = target
        self._target = Target(target)
        self._target.antenna = self.ant

    def req_mode(self, mode):
        self.mode = mode

    def req_scan_asym(self):
        pass

    def update(self, timestamp):
        elapsed_time = timestamp - self._last_update if self._last_update else 0.0
        max_delta_az = self.max_slew_azim_dps * elapsed_time
        max_delta_el = self.max_slew_elev_dps * elapsed_time
        az, el = self.pos_actual_scan_azim, self.pos_actual_scan_elev
        requested_az, requested_el = self._target.azel(timestamp)
        requested_az = rad2deg(wrap_angle(requested_az))
        requested_el = rad2deg(requested_el)
        delta_az = wrap_angle(requested_az - az, period=360.)
        delta_el = requested_el - el
        az += np.clip(delta_az, -max_delta_az, max_delta_az)
        el += np.clip(delta_el, -max_delta_el, max_delta_el)
        self.pos_request_scan_azim = requested_az
        self.pos_request_scan_elev = requested_el
        self.pos_actual_scan_azim = az
        self.pos_actual_scan_elev = el
        dish = construct_azel_target(deg2rad(az), deg2rad(el))
        error = rad2deg(self._target.separation(dish, timestamp))
        self.lock = error < self.lock_threshold
        self._last_update = timestamp
#        print 'elapsed: %g, max_daz: %g, max_del: %g, daz: %g, del: %g, error: %g' % (elapsed_time, max_delta_az, max_delta_el, delta_az, delta_el, error)


class CorrelatorBeamformerModel(FakeModel):
    def __init__(self, n_chans, n_accs, n_bls, bls_ordering, bandwidth,
                 sync_time, int_time, scale_factor_timestamp, ref_ant, **kwargs):
        self.dbe_mode = 'c8n856M32k'
        self.ref_ant = Antenna(ref_ant)
        self.req_target('Zenith, azel, 0, 90')
        self.auto_delay = True

    def req_target(self, target):
        self.target = target
        self._target = Target(target)
        self._target.antenna = self.ref_ant


class EnviroModel(FakeModel):
    def __init__(self, **kwargs):
        self.air_pressure = 1020
        self.air_relative_humidity = 60.0
        self.air_temperature = 25.0
        self.wind_speed = 4.2
        self.wind_direction = 90.0


class DigitiserModel(FakeModel):
    def __init__(self, **kwargs):
        self.overflow = False


class ObservationModel(FakeModel):
    def __init__(self, **kwargs):
        self.label = ''
        self.params = ''
