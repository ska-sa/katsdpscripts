import time
import threading
import types

import numpy as np

from katpoint import (Antenna, Target, Catalogue, rad2deg, deg2rad,
                      construct_azel_target)
from katcp import DeviceServer, Sensor
from katcp.kattypes import return_reply, Str
from katcorelib import build_client


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.

    """
    return (angle + 0.5 * period) % period - 0.5 * period


class FakeSensor(object):
    """Fake sensor."""
    def __init__(self, name, sensor_type, description, units=''):
        sensor_type = Sensor.parse_type(sensor_type)
        params = ['unknown'] if sensor_type == Sensor.DISCRETE else None
        self._sensor = Sensor(sensor_type, name, description, units, params)
        self.name = name
        self.description = description

    def get_value(self):
        return self._sensor.value()

    def set_value(self, value, timestamp):
        if self._sensor.stype == 'discrete':
            self._sensor._kattype._values.append(value)
            self._sensor._kattype._valid_values.add(value)
        self._sensor.set(timestamp, Sensor.NOMINAL, value)

    def set_strategy(self, strategy, params=None):
        pass
    def register_listener(self, listener, limit=-1):
        pass
    def unregister_listener(self, listener):
        pass


class IgnoreUnknownMethods(object):
    def __getattr__(self, name):
        return IgnoreUnknownMethods()
    def __call__(self, *args, **kwargs):
        pass


class FakeCamEventServer(DeviceServer):
    """Device server that serves fake CAM events for a simulated observation.

    Parameters
    ----------
    attributes : dict mapping string to string
        Attributes as key-value string pairs which are streamed once upfront
    sensors : file object, string, list of strings or generator
        File-like object or filename of CSV file listing sensors to serve, with
        header row followed by rows with format 'name, description, unit, type'

    """

    VERSION_INFO = ("fake_cam_event", 0, 1)
    BUILD_INFO = ("fake_cam_event", 0, 1, __version__)

    def __init__(self, attributes, sensors, *args, **kwargs):
        self.attributes = attributes
        self.sensors = np.loadtxt(sensors, delimiter=',', skiprows=1, dtype=np.str)
        super(FakeCamEventServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        """Populate sensor objects on server."""
        for fields in self.sensors:
            self.add_sensor(FakeSensor(*[f.strip() for f in fields]))

    @return_reply(Str())
    def request_get_attributes(self, req, msg):
        """Return dictionary of attributes."""
        logger.info('Returning %d attributes' % (len(attributes,)))
        return ("ok", repr(self.attributes))


class FakeClient(object):
    """Fake KATCP client."""
    def __init__(self, name, model, sensors):
        self.name = name
        self.model = model
        self._time = time
        self.req = IgnoreUnknownMethods()
        self.sensor = IgnoreUnknownMethods()
        for sensor in sensors:
            setattr(self.sensor, sensor.name, sensor)
        self._register_sensors()
        self._register_requests()

    def _register_sensors(self):
        def set_sensor_attr(model, attr_name, value):
            sensor = getattr(self.sensor, attr_name, None)
            print self, attr_name, '->', sensor
            if sensor:
                sensor.set_value(value, model._time.time())
            object.__setattr__(model, attr_name, value)
        # Modify __setattr__ on the *class* and not the instance
        # (see e.g. http://stackoverflow.com/questions/13408372)
        setattr(self.model.__class__, '__setattr__', set_sensor_attr)

    def _register_requests(self):
        for attr_name in dir(self.model):
            attr = getattr(self.model, attr_name)
            if callable(attr) and attr_name.startswith('req_'):
                setattr(self.req, attr_name[4:], types.MethodType(attr.im_func, self.model))

    def is_connected(self):
        return True

    def sensor_sampling(self, sensor_name, strategy, params=None):
        sensor = getattr(self.sensor, sensor_name)
        sensor.set_strategy(strategy, params=None)


inner_threshold_deg: 0.01
inner_threshold_min_time_sec: 2

class AntennaModel(object):
    def __init__(self, antenna, slew_rates):
        self.ant = Antenna(antenna)
        self.observer = self.ant.description
        self.req_target('Zenith, azel, 0, 90')
        self.mode = 'POINT'
        self.activity = 'stow'
        self.lock = True
        self.pos_actual_scan_azim = self.pos_request_scan_azim = 0.0
        self.pos_actual_scan_elev = self.pos_request_scan_elev = 90.0
        self.max_az_vel = slew_rates[0]
        self.max_el_vel = slew_rates[1]
        self.last_update = None
        self._time = time

    def req_target(self, target):
        self.target = target
        self._target = Target(target)
        self._target.antenna = self.ant

    def req_mode(self, mode):
        self.mode = mode

    def req_scan_asym(self):
        pass

    def update(self, timestamp):
        elapsed_time = timestamp - self.last_update if self.last_update else 0.0
        max_delta_az = self.max_az_vel * elapsed_time
        max_delta_el = self.max_el_vel * elapsed_time
        az, el = self.pos_actual_scan_azim, self.pos_actual_scan_elev
        requested_az, requested_el = self._target.azel(timestamp)
        requested_az = rad2deg(angle_wrap(requested_az))
        requested_el = rad2deg(requested_el)
        delta_az = angle_wrap(requested_az - az, period=360.)
        delta_el = requested_el - el
        az += np.clip(delta_az, -max_delta_az, max_delta_az)
        el += np.clip(delta_el, -max_delta_el, max_delta_el)
        self.pos_request_scan_azim = requested_az
        self.pos_request_scan_elev = requested_el
        self.pos_actual_scan_azim = az
        self.pos_actual_scan_elev = el
        dish = construct_azel_target(deg2rad(az), deg2rad(el))
        error = rad2deg(self._target.separation(dish, timestamp))
        self.lock = error < 0.01
        print 'elapsed: %g, max_daz: %g, max_del: %g, daz: %g, del: %g, error: %g' % (elapsed_time, max_delta_az, max_delta_el, delta_az, delta_el, error)
        self.last_update = timestamp

    
class FakeAntennaPositioner(FakeClient):
    def __init__(self, name, antenna, slew_rates, sensors, time):
        FakeClient.__init__(self, name, sensors, time)

    def _register_sensors(self):
        self._add_sensor('ant', 'observer', lambda ant: ant.description)
        self._add_sensor('target', extract=lambda tgt: tgt.description)
        self._add_sensor('activity')
        self._add_sensor('az', 'pos_actual_scan_azim')
        self._add_sensor('el', 'pos_actual_scan_elev')
        self._add_sensor('requested_az', 'pos_request_scan_azim')
        self._add_sensor('requested_el', 'pos_request_scan_elev')
        self._add_sensor('lock')

    def _register_requests(self):

        
        
        
        

class FakeCorrelatorBeamformerModel(object):
    def __init__(self, n_chans, n_accs, n_bls, bls_ordering, bandwidth, sync_time, int_time, scale_factor_timestamp):
        self.dbe_mode = 'c8n856M32k'
        self.target = katpoint.Target('Zenith, special')
        self.center_frequency_hz = 1300e6
        self.auto_delay = True

class FakeEnviroModel(object):
    def __init__(self):
        self.air_pressure = 1020
        self.air_relative_humidity = 60.0
        self.air_temperature = 25.0
        self.wind_speed = 4.2
        self.wind_direction = 90.0

class FakeDigitiser(IgnoreUnknownMethods):
    def __init__(self):
        self.overflow = False

class FakeConn(object):
    """Connection object for a simulated KAT system."""
    def __init__(self, system=None, sb_id_code=None, dry_run=False, attributes=None, opts=None, sensors=None):
        self.create_sensors(sensors)
        self.sensors = IgnoreUnknownMethods()
        self.system = system
        self.sb_id_code = sb_id_code
        self.connected_objects = {}
        self.controlled_objects = []
        self.dry_run = dry_run
        self.sources = katpoint.Catalogue()
        self.attributes = attributes if attributes is not None else {}
        self.server = FakeCamEventServer(self.attributes, opts.sensor_list,
                                         host=opts.fake_cam_host,
                                         port=opts.fake_cam_port) if opts else None

def create_sensors(sensors):
    comp_sensors = {}
    sensor_table = np.loadtxt(sensors, delimiter=',', skiprows=1, dtype=np.str)
    for fields in sensor_table:
        fields = [f.strip() for f in fields]
        comp, sensor_args = fields[0], fields[1:]
        cs = comp_sensors.get(comp, [])
        cs.append(FakeSensor(*sensor_args))
        comp_sensors[comp] = cs
    return comp_sensors
