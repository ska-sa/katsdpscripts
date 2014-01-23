import types
from ConfigParser import SafeConfigParser, NoSectionError
import weakref
import csv
import threading
import time
import logging

import numpy as np

from katpoint import Catalogue, is_iterable
from katcp import DeviceServer, Sensor
from katcp.kattypes import return_reply, Str
from katcp.sampling import SampleStrategy
from katcorelib import build_client

from katscripts.updater import WarpClock, PeriodicUpdaterThread
from katscripts import fake_models


__version__ = 'dev'
user_logger = logging.getLogger("user")


# XXX How about moving this to katcp?
def normalize_strategy_parameters(params):
    # Normalize strategy parameters to be a list of strings, e.g.:
    # ['1.234', # number
    #  'stringparameter']
    if not params:
        return []
    def fixup_numbers(val):
        try:                          # See if it is a number
            return str(float(val))
        except ValueError:
            # ok, it is not a number we know of, perhaps a string
            return str(val)

    if isinstance(params, basestring):
        param_args = [fixup_numbers(p) for p in params.split(' ')]
    else:
        if not is_iterable(params):
            params = (params,)
        param_args = [fixup_numbers(p) for p in params]
    return param_args


# XXX This should ideally live in katcp
def escape_name(name):
    """Helper function for escaping sensor and request names, replacing '.' and '-' with '_' """
    return name.replace(".","_").replace("-","_")


class SensorUpdate(object):
    """"""
    def __init__(self, update_seconds, value_seconds, status, value):
        self.update_seconds = update_seconds
        self.value_seconds = value_seconds
        self.status = status
        self.value = value


class FakeSensor(object):
    """Fake sensor."""
    def __init__(self, name, sensor_type, description, units='', clock=time):
        self.name = name
        sensor_type = Sensor.parse_type(sensor_type)
        params = ['unknown'] if sensor_type == Sensor.DISCRETE else None
        self._sensor = Sensor(sensor_type, name, description, units, params)
        self.__doc__ = self.description = description
        self._clock = clock
        self._listeners = set()
        self._last_update = SensorUpdate(0.0, 0.0, 'unknown', None)
        self._strategy = None
        self._next_period = None
        self.set_strategy('none')

    @property
    def value(self):
        return self._last_update.value

    @property
    def status(self):
        return self._last_update.status

    def get_value(self):
        return self._sensor.value()

    def _set_value(self, value):
        if self._sensor.stype == 'discrete':
            self._sensor._kattype._values.append(value)
            self._sensor._kattype._valid_values.add(value)
        self._sensor.set(self._clock.time(), Sensor.NOMINAL, value)

    def set_strategy(self, strategy, params=None):
        """Set sensor strategy."""
        def inform_callback(sensor_name, timestamp_str, status_str, value_str):
            """Inform callback for sensor strategy."""
            update_seconds = self._clock.time()
            value_seconds = float(timestamp_str)
            value = self._sensor.parse_value(value_str)
            self._last_update = SensorUpdate(update_seconds, value_seconds,
                                             status_str, value)
            for listener in set(self._listeners):
                listener(update_seconds, value_seconds, status_str, value_str)
            print sensor_name, timestamp_str, status_str, value_str

        if self._strategy:
            self._strategy.detach()
        params = normalize_strategy_parameters(params)
        self._strategy = SampleStrategy.get_strategy(strategy, inform_callback,
                                                     self._sensor, *params)
        self._strategy.attach()
        self._next_period = self._strategy.periodic(self._clock.time())

    def update(self, timestamp):
        while self._next_period and timestamp >= self._next_period:
            self._next_period = self._strategy.periodic(self._next_period)

    def register_listener(self, listener, min_wait=-1.0):
        """Add a callback function that is called when sensor value is updated.

        Parameters
        ----------
        listener : function
            Callback signature: listener(update_seconds, value_seconds, status, value)
        min_wait : float, optional
            Minimum waiting period before listener can be called again, used
            to limit the callback rate (zero or negative for no rate limit)
            *This is ignored* as the same effect can be achieved with an
            event-rate strategy on the sensor.

        """
        self._listeners.add(listener)

    def unregister_listener(self, listener):
        """Remove a listener callback added with register_listener().

        Parameters
        ----------
        listener : function
            Reference to the callback function that should be removed

        """
        self._listeners.discard(listener)


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
    def __init__(self, name, model, telescope, clock=time):
        self.name = name
        self.model = object.__new__(model)
        self.req = IgnoreUnknownMethods()
        self.sensor = IgnoreUnknownMethods()
        attrs = telescope[name]['attrs']
        sensors = telescope[name]['sensors']
        for sensor_args in sensors:
            sensor = FakeSensor(*sensor_args, clock=clock)
            setattr(self.sensor, sensor.name, sensor)
        self._clock = clock
        self._register_sensors()
        self._register_requests()
        self.model.__init__(**attrs)

    def _register_sensors(self):
        self.model._client = weakref.proxy(self)
        def set_sensor_attr(model, attr_name, value):
            if hasattr(model, '_client'):
                sensor = getattr(model._client.sensor, attr_name, None)
                if sensor:
                    sensor._set_value(value)
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

    def update(self, timestamp):
        self.model.update(timestamp)
        for sensor in vars(self.sensor).values():
            sensor.update(timestamp)

    def is_connected(self):
        return True

    def sensor_sampling(self, sensor_name, strategy, params=None):
        sensor = getattr(self.sensor, sensor_name)
        sensor.set_strategy(strategy, params)

    def wait(self, sensor_name, condition, timeout=5, status='nominal'):
        sensor_name = escape_name(sensor_name)
        try:
            sensor = getattr(self.sensor, sensor_name)
        except AttributeError:
            raise ValueError("Cannot wait on sensor %r which does not exist "
                             "on client %r" % (sensor_name, self.name))
        full_condition = lambda: sensor.status == status and (
                                 callable(condition) and condition(sensor) or
                                 sensor.value == condition)
        try:
            return self._clock.slave_sleep(timeout, full_condition)
        except KeyboardInterrupt:
            user_logger.info("User requested interrupt of wait on sensor %r "
                             "which has value %r" % (sensor_name, sensor.value))
            raise


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
        self._clock = WarpClock(start_time, dry_run)
        self._clients = []
        for comp_name, component in self._telescope.items():
            model = vars(fake_models).get(component['class'] + 'Model')
            client = FakeClient(comp_name, model, self._telescope, self._clock)
            setattr(self, comp_name, client)
            self._clients.append(client)
            # Add component sensors to the top-level sensors group
            for sensor_args in component['sensors']:
                sensor_name = sensor_args[0]
                sensor = getattr(client.sensor, sensor_name)
                setattr(self.sensors, comp_name + '_' + sensor_name, sensor)
        self.updater = PeriodicUpdaterThread(self._clients, self._clock, period=0.1)
        self.updater.start()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and stop the system."""
        self.stop()
        # Don't suppress exceptions
        return False

    def stop(self):
        """Stop the system."""
        self.updater.stop()
        self.updater.join()

    def time(self):
        """Current time in UTC seconds since Unix epoch."""
        return self._clock.time()

    def sleep(self, seconds):
        """Sleep for the requested duration in seconds."""
        self._clock.slave_sleep(seconds)

    @property
    def dry_run(self):
        return self._clock.warp
    @dry_run.setter
    def dry_run(self, flag):
        self._clock.warp = flag

        # self.system = system
        # self.sb_id_code = sb_id_code
        # self.connected_objects = {}
        # self.controlled_objects = []
        # self.sources = katpoint.Catalogue()
        # self.attributes = attributes if attributes is not None else {}
        # self.server = FakeCamEventServer(self.attributes, opts.sensor_list,
        #                                  host=opts.fake_cam_host,
        #                                  port=opts.fake_cam_port) if opts else None
