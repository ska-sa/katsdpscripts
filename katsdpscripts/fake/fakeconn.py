import types
from ConfigParser import SafeConfigParser, NoSectionError
import weakref
import csv

import numpy as np

from katpoint import (Antenna, Target, Catalogue, rad2deg, deg2rad, wrap_angle,
                      construct_azel_target)
from katcp import DeviceServer, Sensor
from katcp.kattypes import return_reply, Str
from katcorelib import build_client

from .updater import PeriodicUpdaterThread

__version__ = 'dev'


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
    def __nonzero__(self):
        return False


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
    def __init__(self, name, model, telescope, clock=None):
        self.name = name
        self.model = object.__new__(model)
        self.req = IgnoreUnknownMethods()
        self.sensor = IgnoreUnknownMethods()
        attrs = telescope[name]['attrs']
        sensors = telescope[name]['sensors']
        for sensor_args in sensors:
            sensor = FakeSensor(*sensor_args)
            setattr(self.sensor, sensor.name, sensor)
        self._register_sensors(clock if clock is not None else time)
        self._register_requests()
        self.model.__init__(**attrs)

    def _register_sensors(self, clock):
        self.model._client = weakref.proxy(self)
        self.model._clock = weakref.proxy(clock)
        def set_sensor_attr(model, attr_name, value):
            if hasattr(model, '_client'):
                sensor = getattr(model._client.sensor, attr_name, None)
                if sensor:
                    sensor.set_value(value, model._clock.time())
            object.__setattr__(model, attr_name, value)
        # Modify __setattr__ on the *class* and not the instance
        # (see e.g. http://stackoverflow.com/questions/13408372)
        setattr(self.model.__class__, '__setattr__', set_sensor_attr)

    def _register_requests(self):
        for attr_name in dir(self.model):
            attr = getattr(self.model, attr_name)
            if callable(attr) and attr_name.startswith('req_'):
                # Unbind attr function from model and bind it to req, removing 'req_' prefix
                setattr(self.req, attr_name[4:], types.MethodType(attr.im_func, self.model))

    def is_connected(self):
        return True

    def sensor_sampling(self, sensor_name, strategy, params=None):
        sensor = getattr(self.sensor, sensor_name)
        sensor.set_strategy(strategy, params=None)


class AntennaPositionerModel(object):
    def __init__(self, description, max_azim_slew_degpersec, max_elev_slew_degpersec,
                 inner_threshold_deg, **kwargs):
        self.ant = Antenna(description)
        self.req_target('Zenith, azel, 0, 90')
        self.mode = 'POINT'
        self.activity = 'stow'
        self.lock = True
        self.lock_threshold = inner_threshold_deg
        self.pos_actual_scan_azim = self.pos_request_scan_azim = 0.0
        self.pos_actual_scan_elev = self.pos_request_scan_elev = 90.0
        self.max_azim_slew_degpersec = max_azim_slew_degpersec
        self.max_elev_slew_degpersec = max_elev_slew_degpersec

    def req_target(self, target):
        self.target = target
        self._target = Target(target)
        self._target.antenna = self.ant

    def req_mode(self, mode):
        self.mode = mode

    def req_scan_asym(self):
        pass

    def update(self, timestamp, last_update=None):
        elapsed_time = timestamp - last_update if last_update else 0.0
        max_delta_az = self.max_azim_slew_degpersec * elapsed_time
        max_delta_el = self.max_elev_slew_degpersec * elapsed_time
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
        print 'elapsed: %g, max_daz: %g, max_del: %g, daz: %g, del: %g, error: %g' % (elapsed_time, max_delta_az, max_delta_el, delta_az, delta_el, error)


class CorrelatorBeamformerModel(object):
    def __init__(self, n_chans, n_accs, n_bls, bls_ordering, bandwidth, sync_time, int_time, scale_factor_timestamp, **kwargs):
        self.dbe_mode = 'c8n856M32k'
        self.req_target('Zenith, azel, 0, 90')
        self.auto_delay = True

    def req_target(self, target):
        self.target = target
        self._target = Target(target)
#        self._target.antenna = self.ant


class EnviroModel(object):
    def __init__(self, **kwargs):
        self.air_pressure = 1020
        self.air_relative_humidity = 60.0
        self.air_temperature = 25.0
        self.wind_speed = 4.2
        self.wind_direction = 90.0


class DigitiserModel(object):
    def __init__(self, **kwargs):
        self.overflow = False


class ObservationModel(object):
    def __init__(self, **kwargs):
        self.label = ''
        self.params = ''


def load_config(config_file):
    split = lambda args: csv.reader([args], skipinitialspace=True).next()
    cfg = SafeConfigParser()
    cfg.read(config_file)
    components = dict(cfg.items('Telescope'))
    telescope = {}
    for comp_name, comp_type in components.items():
        telescope[comp_name] = {'class' : comp_type, 'attrs' : {}, 'sensors' : []}
        sections = [':'.join((comp_type, name, item)) for name in ['*', comp_name]
                                                      for item in ['attrs', 'sensors']]
        for section in sections:
            try:
                items = cfg.items(section)
            except NoSectionError:
                continue
            if section.endswith('attrs'):
                attr_items = [(name, eval(value, {})) for name, value in items]
                telescope[comp_name]['attrs'].update(attr_items)
            else:
                sensor_items = [[name] + split(args) for name, args in items]
                telescope[comp_name]['sensors'].extend(sensor_items)
    return telescope


class FakeConn(object):
    """Connection object for a simulated KAT system."""
    def __init__(self, config_file, dry_run=False, start_time=None):
        self._telescope = load_config(config_file)
        self.sensors = IgnoreUnknownMethods()
        self.dry_run = dry_run
        self._timestamp = katpoint.Timestamp(start_time).secs
        for comp_name, component in self._telescope.items():
            model = globals().get(component['class'] + 'Model')
            client = FakeClient(comp_name, model, self._telescope, self)
            setattr(self, comp_name, client)
            for sensor_args in component['sensors']:
                sensor_name = sensor_args[0]
                sensor = getattr(client.sensor, sensor_name)
                setattr(self.sensors, comp_name + '_' + sensor_name, sensor)
        self.updater = PeriodicUpdaterThread(self._telescope.values(),
                                             dry_run, start_time, period=0.1)
        self.updater.start()

    def time(self):
        """Current time in UTC seconds since Unix epoch."""
        return self.updater.time()

    def sleep(self, seconds):
        """Sleep for the requested duration in seconds."""
        self.updater.sleep(seconds)

        
        # self.system = system
        # self.sb_id_code = sb_id_code
        # self.connected_objects = {}
        # self.controlled_objects = []
        # self.sources = katpoint.Catalogue()
        # self.attributes = attributes if attributes is not None else {}
        # self.server = FakeCamEventServer(self.attributes, opts.sensor_list,
        #                                  host=opts.fake_cam_host,
        #                                  port=opts.fake_cam_port) if opts else None
