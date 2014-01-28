import weakref
import types
import logging

from katscripts.fake_sensor import FakeSensor, escape_name


user_logger = logging.getLogger("user")


class IgnoreUnknownMethods(object):
    def __getattr__(self, name):
        return IgnoreUnknownMethods()
    def __call__(self, *args, **kwargs):
        pass
    def __nonzero__(self):
        return False


class FakeClient(object):
    """Fake KATCP client."""
    def __init__(self, name, model, telescope, clock):
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
        self._aggregates = {}
        self._register_sensors()
        self._register_requests()
        self.model.__init__(**attrs)
        self._register_aggregate_sensors()

    def _register_sensors(self):
        self.model._client = weakref.proxy(self)
        def set_sensor_attr(model, attr_name, value):
            client = getattr(model, '_client', None)
            if client:
                sensor = getattr(client.sensor, attr_name, None)
                if sensor:
                    sensor._set_value(value)
            object.__setattr__(model, attr_name, value)
            if client and attr_name in client._aggregates:
                for parent, rule, children in client._aggregates[attr_name]:
                    child_values = [getattr(model, c) for c in children]
                    setattr(model, parent, rule(*child_values))
        # Modify __setattr__ on the *class* and not the instance
        # (see e.g. http://stackoverflow.com/questions/13408372)
        setattr(self.model.__class__, '__setattr__', set_sensor_attr)

    def _req_sensor_sampling(self, sensor_name, strategy, params=None):
        sensor = getattr(self.sensor, sensor_name)
        sensor.set_strategy(strategy, params)

    def _register_requests(self):
        for attr_name in dir(self.model):
            attr = getattr(self.model, attr_name)
            if callable(attr) and attr_name.startswith('req_'):
                # Unbind attr function from model and bind it to req, removing 'req_' prefix
                setattr(self.req, attr_name[4:], types.MethodType(attr.im_func, self.model))
        setattr(self.req, 'sensor_sampling', self._req_sensor_sampling)

    def _register_aggregate_sensors(self):
        for attr_name in dir(self.model):
            attr = getattr(self.model, attr_name)
            if callable(attr) and attr_name.startswith('_aggregate_'):
                agg_parent = attr_name[11:]
                agg_func = attr.im_func.func_code
                agg_children = agg_func.co_varnames[1:agg_func.co_argcount]
                for child in agg_children:
                    self._aggregates[child] = self._aggregates.get(child, []) + \
                                              [(agg_parent, attr, agg_children)]

    def update(self, timestamp):
        self.model.update(timestamp)
        for sensor in vars(self.sensor).values():
            sensor.update(timestamp)

    def is_connected(self):
        return True

    def wait(self, sensor_name, condition, timeout=5, status='nominal'):
        sensor_name = escape_name(sensor_name)
        try:
            sensor = getattr(self.sensor, sensor_name)
        except AttributeError:
            raise ValueError("Cannot wait on sensor %r which does not exist "
                             "on client %r" % (sensor_name, self.name))
        if sensor.strategy == 'none':
	        raise ValueError("Cannot wait on sensor %r if it has no strategy "
                             "set - see kat.%s.sensor.%s.set_strategy" %
                             (sensor_name, self.name, sensor_name))

        full_condition = lambda: sensor.status == status and (
                                 callable(condition) and condition(sensor) or
                                 sensor.value == condition)
        try:
            success = self._clock.slave_sleep(timeout, full_condition)
            if not success:
                msg = "Waiting for sensor %r %s reached timeout of %d seconds" % \
                      (sensor_name, ("condition" if callable(condition) else
                                     "== " + str(condition)), timeout)
                user_logger.warning(msg)
            return success
        except KeyboardInterrupt:
            user_logger.info("User requested interrupt of wait on sensor %r "
                             "which has value %r" % (sensor_name, sensor.value))
            raise


class GroupRequest(object):
    """The old ArrayRequest class."""
    def __init__(self, array, name, description):
        self.array = array
        self.name = name
        self.__doc__ = description

    def __call__(self, *args, **kwargs):
        for client in self.array.clients:
            method = getattr(client.req, self.name, None)
            if method:
                method(*args, **kwargs)


class ClientGroup(object):
    """The old Array class."""
    def __init__(self, name, clients, clock):
        self.name = name
        self.clients = list(clients)
        self._clock = clock
        self.req = IgnoreUnknownMethods()
        # Register requests
        for client in self.clients:
            existing_requests = vars(self.req).keys()
            for name, request in vars(client.req).iteritems():
                if name not in existing_requests:
                    setattr(self.req, name, GroupRequest(self, name, request.__doc__))

    def __iter__(self):
        """Iterate over client members of group."""
        return iter(self.clients)

    def __len__(self):
        """Number of client members in group."""
        return len(self.clients)

    def wait(self, sensor_name, condition, timeout=5, status='nominal'):
        sensor_name = escape_name(sensor_name)
        missing = [client.name for client in self.clients
                   if not hasattr(client.sensor, sensor_name)]
        if missing:
            raise ValueError("Cannot wait on sensor %r in array %r which is "
                             "not present on clients %r" %
                             (sensor_name, self.name, missing))
        sensors = [getattr(client.sensor, sensor_name) for client in self.clients]
        no_strategy = [client.name for client, sensor in zip(self.clients, sensors)
                       if sensor.strategy == 'none']
        if no_strategy:
            raise ValueError("Cannot wait on sensor %r in array %r which has "
                             "no strategy set on clients %r" %
                             (sensor_name, self.name, no_strategy))

        def sensor_condition(sensor):
             return sensor.status == status and (callable(condition) and
                    condition(sensor) or sensor.value == condition)
        full_condition = lambda: all(sensor_condition(s) for s in sensors)
        try:
            success = self._clock.slave_sleep(timeout, full_condition)
            if not success:
                non_matched = [c.name for c, s in zip(self.clients, sensors)
                               if not sensor_condition(s)]
                msg = "Waiting for sensor %r %s reached timeout of %d seconds. " \
                      "Clients %r failed." % (sensor_name,
                      ("condition" if callable(condition) else "== " + str(condition)),
                      timeout, non_matched)
                user_logger.warning(msg)
            return success
        except KeyboardInterrupt:
            user_logger.info("User requested interrupt of wait on sensor %r "
                             "which has values %s" % (sensor_name,
                             dict((c.name, s.value)
                                  for c, s in zip(self.clients, sensors))))
            raise
