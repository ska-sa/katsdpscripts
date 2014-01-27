import time

from katpoint import is_iterable
from katcp import Sensor
from katcp.sampling import SampleStrategy


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

    @property
    def strategy(self):
        return SampleStrategy.SAMPLING_LOOKUP[self._strategy.get_sampling()]

    def get_value(self):
        # XXX Check whether this also triggers a sensor update a la strategy
        return self._sensor.value()

    def _set_value(self, value):
        if self._sensor.stype == 'discrete':
            self._sensor._kattype._values.append(value)
            self._sensor._kattype._valid_values.add(value)
        self._sensor.set(self._clock.time(), Sensor.NOMINAL, value)

    def _update_value(self, timestamp, status_str, value_str):
        update_seconds = self._clock.time()
        value = self._sensor.parse_value(value_str)
        self._last_update = SensorUpdate(update_seconds, timestamp,
                                         status_str, value)
        for listener in set(self._listeners):
            listener(update_seconds, timestamp, status_str, value_str)

    def set_strategy(self, strategy, params=None):
        """Set sensor strategy."""
        def inform_callback(sensor_name, timestamp_str, status_str, value_str):
            """Inform callback for sensor strategy."""
            self._update_value(float(timestamp_str), status_str, value_str)
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
