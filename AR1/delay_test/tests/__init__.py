###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from testconfig import config             # nose testconfig plugin
import unittest2 as unittest
import logging
import time
import operator

import threading
import Queue

import katcp
import katconf
import katcorelib

from nosekatreport import Aqf as AqfBase
from katcorelib.katcp_client import escape_name
from katcorelib.testutils import set_restore_strategy, wait_with_strategy, \
     SensorTransitionWaiter, wait_sensor

#nosetests --nologcapture will write the output log by default to /var/kat/log/kat-tests.log

from katcorelib.utility import (sbuild, tbuild, cambuild, get_configurl_and_system,
                                set_katconf_configured)
from katmisc.utils.utils import escape_name
from katcorelib.conf import KatuilibConfig
from katuilib import obsbuild


existing_root_log_handlers = []
root_logger = logging.getLogger()

def bool_from_config(varname):
    """get a bool value from testconfig.config or set default to False if not found"""
    if varname not in config:
        config[varname] = False
    return bool(int(config[varname]))

class settings(object):
    config_url, default_system = get_configurl_and_system(katconf.sitename())
    intrusive = bool_from_config('intrusive') # should we run intrusive tests?
    include_slow = not bool_from_config('exclude-slow')        # should we run slow tests? - to be superceded with exclude-slow
    sitename = katconf.sitename()
    system = katconf.systype() #kat7, mkat or mkat_rts
    at_site = sitename.startswith("karoo")

class fixtures(object):
    cam = None                            # Should be a kat-host object
    sim = None                            # Should be a sim-host object
    obs = None                            # Should be a katuilib.KATObs object
    katuilib_config = None                # KatuilibConfig object
    tests_logger = None           # A logger instance for tests to use

tests_start_time = time.time()

def setup_package():
    pass

def teardown_package():
    #Ensure Aqf KATConn objects are disconnected
    Aqf.disconnect()

def SsetUp():
    # An evil hack to silence the 'ipython' logger. Should be resolved
    # by making future versions of s/tbuild accept a KatuilibConfig
    # object.
    def silence_ipython():
        logger = logging.getLogger('ipython')
        handlers = list(logger.handlers)   # Copy the handlers list
        for h in handlers:                 # Remove all the handlers
            logger.removeHandler(h)
        logger.addHandler(NullHandler())

    silencer = threading.Timer(0.25, silence_ipython)
    silencer.start()
    #This is a special build case with full control for integration tests
    pwd = "camcam"
    fixtures.cam = cambuild(password=pwd, full_control=True)
    silencer = threading.Timer(0.25, silence_ipython)
    silencer.start()
    fixtures.sim = sbuild()
    fixtures.katuilib_config = KatuilibConfig(fixtures.cam.system)
    #Setup obs interface
    fixtures.sysconf = katconf.SystemConfig(settings.default_system)
    db_uri = fixtures.sysconf.conf.get("katobs","db_uri")
    controlled_resources = fixtures.sysconf.conf.get("katobs", "controlled_resources")
    fixtures.obs = obsbuild(db_uri=db_uri, controlled_resources=controlled_resources)
    # set up basic Python logging
    fixtures.tests_logger = logging.getLogger('kat-integration-tests')

def TtearDown():
    fixtures.cam.disconnect()
    fixtures.sim.disconnect()

def x_intrusive(test):
    """Decorator for skipping intrusive tests"""
    return unittest.skipUnless(settings.intrusive, 'Skipping intrusive test')(test)

def x_slow(test):
    """Decorator for skipping tests that has been identified to be slow"""
    return unittest.skipUnless(settings.include_slow, 'Skipping slow test')(test)

def nodemanager_for(process):
    """
    Return nodemanager object managing a process or raises RuntimeError

    Process name is escaped to _ form. Depends on each process having
    a unique name accross all nodemanagers.
    """
    nodemanagers = [getattr(fixtures.cam, x)
                    for x in dir(fixtures.cam)
                    if x.startswith('nm_')]
    for nm in nodemanagers:
        if any(escape_name(process) == escape_name(m.arguments[0])
               for m in nm.req.process_list().messages):
            return nm
    raise RuntimeError('No nodemanager found for process %r' % process)

def nodemanager_process_req(process, request):
    """Execute `request(process)` on nodemanager managing `process`"""
    nm = nodemanager_for(process)
    # Figure out the unescaped process name using a sensor
    sens = nodemanager_process_sensor(process, 'ok')
    unescaped_process = sens.name[:-3]
    return getattr(nm.req, escape_name(request))(unescaped_process)

def nodemanager_process_sensor(process, sensor):
    """
    Get nodemanager sensor relating to process `process`

    Looks up the correct nodemanager object, and escapes the name of
    process and sensor to underscore form. Then return the sensor
    process+'_sensor'

    Example
    -------

    nodemanager_process_sensor('dbe7-dev', 'running')

    Returns the sensor cam.nm_localhost.sensor.dbe7_dev_running
    (assuming dbe7-dev is managed by nm_localhost_)
    """

    nm = nodemanager_for(process)
    return getattr(nm.sensor, escape_name(process+'_'+sensor))

# Cut-n-paste from python2.7 logging module for backwards compatibility
class NullHandler(logging.Handler):
    """
    This handler does nothing. It's intended to be used to avoid the
    "No handlers could be found for logger XXX" one-off warning. This is
    important for library code, which may contain code to log events. If a user
    of the library does not configure logging, the one-off warning might be
    produced; to avoid this, the library developer simply needs to instantiate
    a NullHandler and add it to the top-level logger of the library module or
    package.
    """
    def handle(self, record):
        pass

    def emit(self, record):
        pass

    def createLock(self):
        self.lock = None

def wait_sensor_value(kat, katsensor, look_for, timeout, exclude=False):
    """Test katsensor value every second to see if it includes/excludes the look_for string, until timeout
    katsensor can be a kat.sensors. level sensorname or a KATCPSensor object"""
    i = 0
    if isinstance(katsensor, str):
        # Get the katsensor object
        katsensor = getattr(kat.sensors, katsensor, None)
    while i < timeout:
        val = katsensor.get_value()
        if exclude:
            if str(look_for) not in str(val):
                return True
        else:
            if str(look_for) in str(val):
                return True
        i = i + 1
        time.sleep(1.0)
    return False

def wait_sensor_includes(kat, katsensor, look_for, timeout):
    """Test katsensor value every second to see if it inclues the look_for string, until timeout
    katsensor can be a kat.sensors. level sensorname or a KATCPSensor object"""
    return wait_sensor_value(kat, katsensor, look_for, timeout, exclude=False)

def wait_sensor_excludes(kat, katsensor, look_for, timeout):
    """Test katsensor value every second to see if it excludes the look_for string, until timeout
    katsensor can be a kat.sensors. level sensorname or a KATCPSensor object"""
    return wait_sensor_value(kat, katsensor, look_for, timeout, exclude=True)


class _state(object):
    """Class for storing state and progress."""
    cam_objects = {}
    sensor_handle = None

class Aqf(AqfBase):
    @classmethod
    def __ipy_sensor(cls, name):
        print "using ipython sensor: %s" % name
        # Should return a sensor object.

        import katuilib.ipython_vars as ui_ip
        return AqfSensor(name,
                         cam=ui_ip.ip_shell.user_ns['cam'],
                         sim=ui_ip.ip_shell.user_ns['sim'])

    @classmethod
    def ipython(cls):
        """Setup Aqf to work in an Ipython shell."""

        import katuilib.ipython_vars as ui_ip
        ip_shell = ui_ip.ip_shell
        if 'cam' not in ip_shell.user_ns:
            raise Exception('Could not get cam object, please create it.')
        if 'sim' not in ip_shell.user_ns:
            raise Exception('Could not get sim object, please create it.')

        print "Setup Aqf to work in an ipython shell"
        _state.sensor_handle = cls.__ipy_sensor

    @classmethod
    def _hook_sensor_handler(cls, func):
        _state.sensor_handle = func

    @classmethod
    def sensor(cls, name):
        """Return a AqfSensor Object.

        Aqf.sensor is a short hand way of getting and setting sensor values
        and for doing comparisons on the sensor value.

        A valid sensor name starting with sim or cam must be given,
        the same as would have been used in ipython. ::

            print Aqf.sensor('sim.asc.sensors.wind_speed').get()

        To set a sensor. ::

            Aqf.sensor('sim.asc.sensors.wind_speed').set(40)

        The following evaluation methods can be used:

        +------------+------------+-------------------+
        | Method     | Symbolic   | Description       |
        +============+============+===================+
        | eq         | ==         | Equals            |
        +------------+------------+-------------------+
        | ne         | !=         | Not Equals        |
        +------------+------------+-------------------+
        | ge         | >=         | Greater or Equals |
        +------------+------------+-------------------+
        | el         | <=         | Less or Equals    |
        +------------+------------+-------------------+
        | gt         | >          | Greater           |
        +------------+------------+-------------------+
        | lt         | <          | Less              |
        +------------+------------+-------------------+
        | startswith | startswith | start with        |
        +------------+------------+-------------------+
        | endswith   | endswith   | end with          |
        +------------+------------+-------------------+
        | contains   | contains   | contains          |
        +------------+------------+-------------------+

        Examples: To check that the sensor sim.asc.sensors.wind_speed
        has a value equals to 40 ::

            Aqf.sensor('sim.asc.sensors.wind_speed').eq('40')


        :return: AqfSensor.

        """
        if _state.sensor_handle:
            return _state.sensor_handle(name)

    @classmethod
    def disconnect(cls):
        """Disconnect any lazily built KATConn objects.
        """
        print "=== Aqf is disconnecting"
        for obj in ['cam', 'sim']:
            if obj in _state.cam_objects:
                try:
                    print "Disconnecting %s" % obj
                    _state.cam_objects[obj].disconnect()
                    print "Disconnected %s" % obj
                except Exception as exc:
                    print "Failure disconnecting %s (%s)" % (obj, exc)


class AqfTestCase(unittest.TestCase):

    """Base class that all integration tests should inherit from.

    self.cam, self.sim and self.obs are instantiated on first call.

    """

    def __init__(self, *args, **kwargs):
        super(AqfTestCase, self).__init__(*args, **kwargs)
        # Register the sensor method with Aqf class.
        # In the test self.sensor and Aqf.sensor will be the same reference.
        Aqf._hook_sensor_handler(self.sensor)
        self._cam = None
        self._sim = None
        self._use_sim = True
        self._obs = None

    def sensor(self, name):
        """Syntax sugar to get to the AqfSensor object."""
        return AqfSensor(name, cam=self.cam, sim=self.sim)

    @property
    def cam(self):
        if not self._cam:
            # Save a reference to the existing log-handlers (mostly to get the nosetest
            # log-capture handler) that will be removed when the cam build installs the
            # logging config
            if not existing_root_log_handlers:
                existing_root_log_handlers[:] = root_logger.handlers
            if 'cam' in _state.cam_objects:
                self._cam = _state.cam_objects['cam']
                Aqf.log_build('Use an old CAM object.')
                return self._cam
            # setup the Cam object.
            pwd = "camcam"
            Aqf.log_build('Create a new CAM object')
            self._cam = cambuild(password=pwd,
                                 full_control=True)
            _state.cam_objects['cam'] = self._cam
            # Wait a while for cam object to sync
            Aqf.log_build('Wait a while for cam object to sync')
            time.sleep(10)
            try:
                starttime = self._cam.sys.sensor.start_time.get_value()
            except:
                starttime = time.time()
            if time.time()-starttime < 120:
                #Wait some more for startup to settle down
                time.sleep(60)
            # Put back the pre-cambuild loghanders
            for h in existing_root_log_handlers:
                if h not in root_logger.handlers:
                    root_logger.addHandler(h)
            # Re-enable loggers that the logging-config disabled
            for logger in logging.root.manager.loggerDict.values():
                logger.disabled = False
        return self._cam

    @property
    def sim(self):
        if not self._sim and self._use_sim:
            if 'sim' in _state.cam_objects:
                self._sim = _state.cam_objects['sim']
                Aqf.log_build('Return an old SIM object.')
                return self._sim
            self._sim = sbuild()
            Aqf.log_build('Create a new SIM object')
            _state.cam_objects['sim'] = self._sim
            # Wait a while for sim object to sync
            time.sleep(10)
        return self._sim

    @property
    def obs(self):
        """Create a obs object."""
        if not self._obs:
            if 'obs' in _state.cam_objects:
                self._obs = _state.cam_objects['obs']
                Aqf.log_build('Return an old OBS object.')
                return self._obs
            Aqf.log_build('Create a new OBS object')
            site = katconf.sitename()
            if site is None:
                raise ValueError("Setup obs: Unable to determine site name.")

            config_url, system = get_configurl_and_system(site)
            #Verify the 'system' parameter
            if system is None:
                raise ValueError("Setup obs: System resource could not be "
                                 "automatically determined for site %r" %
                                 (site,))

            config = katconf.from_url(config_url)
            if config is None:
                raise ValueError("Setup obs: Could not retrieve configuration "
                                 "for URL %r" % (config_url,))
            katconf.set_config(config)
            set_katconf_configured(True, system)

            try:
                sysconf = katconf.SystemConfig(system)
                db_uri = sysconf.conf.get("katobs", "db_uri")
                controlled_resources = sysconf.conf.get("katobs",
                                                        "controlled_resources")
            except Exception, err:
                print "Unable to load SystemConfig ... ", err
                raise ValueError("Setup obs: Unable to load SystemConfig")
            self._obs = obsbuild(db_uri=db_uri,
                                 controlled_resources=controlled_resources)
            _state.cam_objects['obs'] = self._obs
        return self._obs

class AqfSensor(object):

    def __init__(self, sens, parent=None, cam=None, sim=None):
        if parent is not None:
            self.cam = parent.cam
            self.sim = parent.sim
        if cam is not None:
            self.cam = cam
        if sim is not None:
            self.sim = sim
        self._sensor_obj = None
        self._sensor_name = None

        self._hook_up_sensor(sens)
        #if isinstance(sens, katcorelib.katcp_client.KATSensor):
        #    self._sensor_obj = sens

    def _hook_up_sensor(self, sens):
        if isinstance(sens, katcp.resource.KATCPSensor):
            self._sensor_obj = sens
            parent_name = getattr(self._sensor_obj, 'parent_name')
            if hasattr(self.sim, parent_name):
                # If its in Sim self.set will work.
                self._base_obj = self.sim
            else:
                self._base_obj = self.cam
        elif isinstance(sens, katcorelib.katcp_client.KATSensor):
            self._sensor_obj = sens
            parent_name = getattr(self._sensor_obj, 'parent_name')
            if hasattr(self.sim, parent_name):
                # If its in Sim self.set will work.
                self._base_obj = self.sim
            else:
                self._base_obj = self.cam
        else:
            path = sens.split(".")
            base = path[0]
            if base == 'sim':
                self._base_obj = self.sim
            else:
                self._base_obj = self.cam
            obj = self._base_obj
            for n in path[1:]:
                obj = getattr(obj, n)
            self._sensor_obj = obj

        self._name = getattr(self._sensor_obj, 'name')
        self._parent_name = getattr(self._sensor_obj, 'parent_name')
        self._sensor_name = "%s.%s" % (self._parent_name, self._name)

    def approximate(self, value, inrange=1.0):
        """Pass if the sensor is with in range of the value."""
        value = float(value)
        inrange = float(inrange)
        sensor_value = float(self._get())

        if abs(value - sensor_value) <= inrange:
            Aqf.passed("Sensor {0} is close to {1}"
                       .format(self._sensor_name, value))
            return True
        else:
            Aqf.failed("Sensor {0} is {1} that is not close to {2}".
                       format(self._sensor_name, sensor_value, value))
            return False

    def status(self):
        """Get the status from the sensor.

        :return: String. The status of the sensor.

        """

        self._sensor_obj.get_value()
        status = self._sensor_obj.status
        Aqf.progress("Sensor {0} has status {1}".
                     format(self._sensor_name, status))
        return status

    def get(self):
        """Get a value from a sensor.

        :returns: Value of sensor.

        """
        val = self._get()
        Aqf.progress("Sensor {0} has value {1}".format(self._sensor_name, val))
        return val

    def set(self, val1, val2=0, val3=0):
        """Set or simulate a value on a sensor."""
        obj = getattr(self._base_obj, self._parent_name)
        obj2 = getattr(obj, 'req')
        if isinstance(val1, float) or val2 != 0 or val3 != 0:
            Aqf.progress("Sensor {0} simulate value {1} {2} {3}".
                         format(self._sensor_name, val1, val2, val3))
            res = str(obj2.simulate_value(self._name, val1, val2, val3))
            if not res.endswith("ok"):
                Aqf.log_error("Result of simulate_value({0},{1},{2},{3})"
                              " is {4}".
                              format(self._name, val1, val2, val3, res))
        else:
            Aqf.progress("Set value on %s to %s" %
                         (self._sensor_name, str(val1)))
            res = str(obj2.set_sensor_value(self._name, val1))
            if not res.endswith("ok"):
                Aqf.log_error("Result of set_sensor_value({0},{1}) is {2}".
                              format(self._name, val1, res))

    def _get(self):
        return self._sensor_obj.get_value()

    def _get_int(self):
        i = self._get()
        try:
            i = int(i)
        except ValueError:
            i = None
        return i

    def wait_until_lt(self, value, sleep=6, counter=100):
        """Wait until sensor is less than value.

        :param value: The value to wait for.
        :param sleep: Int. Time in seconds to wait in between sensor readings.
        :param counter: Int. Amount of readings before timeout.
                        Timeout = sleep x counter

        """
        sleep = float(sleep)
        counter = int(counter)
        value = float(value)
        #tmp_value = value + 1
        tmp_value = self._get()
        while counter > 0 and value <= tmp_value:
            counter -= 1
            time.sleep(sleep)
            tmp_value = self._get()
            Aqf.log_wait('Sensor {0} value is {1} waiting until it is less '
                         'than {2} - #{3}'.format(self._sensor_name,
                                                   str(tmp_value), value,
                                                   counter))

        return self.lt(value)

    def wait_until_gt(self, value, sleep=6, counter=100):
        """Wait until sensor is greater than value.

        :param value: The value to wait for.
        :param sleep: Int. Time in seconds to wait in between sensor readings.
        :param counter: Int. Amount of readings before timeout.
                        Timeout = sleep x counter

        """
        sleep = float(sleep)
        counter = int(counter)
        value = float(value)
        #tmp_value = value - 1
        tmp_value = self._get()
        while counter > 0 and value >= tmp_value:
            counter -= 1
            time.sleep(sleep)
            Aqf.log_wait('Sensor %s value is %s waiting until it is greater '
                         'than %f - #%d' %
                         (self._sensor_name, str(tmp_value), value, counter))
            tmp_value = self._get()
        return self.gt(value)

    def wait_until(self, value, sleep=6, counter=100):
        """Wait until sensor reaches this value.

        :param value: Wait until the sensor is at this value. Internaly value
                        is cast to string.
        :param sleep: Int. Time in seconds to wait in between sensor readings.
        :param counter: Int. Amount of readings before timeout.
                        Timeout = sleep x counter

        """
        sleep = float(sleep)
        counter = int(counter)
        svalue = str(value)
        tmp_value = str(self._get())
        while counter > 0 and tmp_value != svalue:
            counter -= 1
            time.sleep(sleep)
            Aqf.log_wait('Sensor %s value is "%s" waiting until it '
                         'is "%s" - #%d' %
                         (self._sensor_name, tmp_value, svalue, counter))
            tmp_value = str(self._get())
        return self.eq(value)

    def wait_until_status(self, is_in=None, is_not_in=None,
                          sleep=1, counter=100):
        if is_in is None and is_not_in is None:
            raise ValueError("wait_until_status: arguments is_in or "
                             "is_not_in must be set")
        sleep = float(sleep)
        counter = int(counter)

        if is_in is None:
            is_in = []
        elif not isinstance(is_in, list):
            is_in = [is_in]

        if is_not_in is None:
                is_not_in = []
        elif not isinstance(is_not_in, list):
            is_not_in = [is_not_in]

        message = 'Sensor {0} status is "%s"; waiting until'.format(
            self._sensor_name)
        if is_in:
            message = message + " status is in {0}".format(is_in)

        if is_not_in:
            if is_in:
                message = message + ' and '
            message = message + " status is not in {0}".format(is_not_in)

        message = message + " C:%d"
        status = self.status()
        if is_in and is_not_in:
            continue_to_wait = all([status not in is_in,
                                    status in is_not_in])
        elif is_in:
            continue_to_wait = status not in is_in
        elif is_not_in:
            continue_to_wait = status in is_not_in
        else:
            continue_to_wait = False  # should never get here.
        while counter > 0 and continue_to_wait:
            counter -= 1
            status = self.status()
            if is_in and is_not_in:
                continue_to_wait = all([status not in is_in,
                                        status in is_not_in])
            elif is_in:
                continue_to_wait = status not in is_in
            elif is_not_in:
                continue_to_wait = status in is_not_in
            else:
                continue_to_wait = False  # should never get here.
            time.sleep(sleep)
            Aqf.log_wait(message % (status, counter))

        if continue_to_wait:
            Aqf.failed("Sensor {0} did not reach expected status. Status "
                       "is {1}".format(self._sensor_name, status))
            return False
        else:
            Aqf.passed("Sensor {0} status is {1}".
                       format(self._sensor_name, status))
            return True

    def wait_until_between(self, lower, higher, sleep=6, counter=100):
        lower = float(lower)
        higher = float(higher)
        counter = int(counter)
        sleep = float(sleep)
        if higher < lower:
            higher, lower = lower, higher

        sensor_value = float(self._get())
        while counter < 0 and sensor_value > higher and sensor_value < lower:
            counter -= 1
            time.sleep(sleep)
            Aqf.log_wait('Sensor %s value is "%f" waiting until it '
                         'is between "%f" and "%f" - #%d' %
                         (self._sensor_name, sensor_value,
                          lower, higher, counter))
            sensor_value = float(self._get())

        if sensor_value < higher and sensor_value > lower:
            Aqf.passed("Sensor {0} is {1}"
                       .format(self._sensor_name, sensor_value))
            return True
        else:
            Aqf.failed("Sensor {0} is {1} that is not between {2} and {3}".
                       format(self._sensor_name, sensor_value, lower, higher))
            return False

    def wait_until_approximate(self, value, inrange=1.0, sleep=6, counter=100):
        """Pass if the sensor is with in range of the value."""
        counter = int(counter)
        sleep = float(sleep)
        value = float(value)
        inrange = float(inrange)

        sensor_value = float(self._get())
        while counter > 0 and abs(value - sensor_value) > inrange:
            counter -= 1
            time.sleep(sleep)
            Aqf.log_wait('Sensor %s value is "%f" waiting until it '
                         'is within "%f" of "%f" - #%d' %
                         (self._sensor_name, sensor_value,
                          inrange, value, counter))
            sensor_value = float(self._get())

        return self.approximate(value, inrange)

    def wait_until_includes(self, look_for, sleep=6, counter=100):
        """
        Wait until sensor value includes look_for

        Use this directly instead of wait_sensor_includes so that
        behaviours is similar to other Aqf.sensor methods
        """
        
        look_for = str(look_for)

        sensor_value = str(self._get())
        while counter < 0 and look_for not in sensor_value:
            counter -= 1
            time.sleep(sleep)
            Aqf.log_wait('Sensor %s value is "%s" waiting until it '
                         'includes "%s" #%d' %
                         (self._sensor_name, sensor_value,
                          look_for, counter))
            sensor_value = str(self._get())

        if sensor_value in sensor_value:
            Aqf.passed("Sensor {0} value is {1} - it includes {2}"
                       .format(self._sensor_name, sensor_value, look_for))
            return True
        else:
            Aqf.passed("Sensor {0} value is {1} - it does not include {2}"
                       .format(self._sensor_name, sensor_value, look_for))
            return False

    def wait_until_excludes(self, look_for, sleep=6, counter=100):
        """
        Wait until sensor value excludes look_for

        Use this directly instead of wait_sensor_excludes so that
        behaviours is similar to other Aqf.sensor methods
        """
        
        look_for = str(look_for)

        sensor_value = str(self._get())
        while counter < 0 and look_for in sensor_value:
            counter -= 1
            time.sleep(sleep)
            Aqf.log_wait('Sensor %s value is "%s" waiting until it '
                         'excludes "%s" #%d' %
                         (self._sensor_name, sensor_value,
                          look_for, counter))
            sensor_value = str(self._get())

        if sensor_value not in sensor_value:
            Aqf.passed("Sensor {0} value is {1} - it excludes {2}"
                       .format(self._sensor_name, sensor_value, look_for))
            return True
        else:
            Aqf.failed("Sensor {0} value is {1} - it does not exclude {2}"
                       .format(self._sensor_name, sensor_value, look_for))
            return False

    def startswith(self, value):
        senval = str(self._get())
        if senval is not None:
            status = senval.startswith(value)
            if status:
                Aqf.passed()
            else:
                Aqf.failed('Did not start with %s' % value)
            return status
        else:
            return False

    def endswith(self, value):
        senval = str(self._get())
        if senval is not None:
            status = senval.endswith(value)
            if status:
                Aqf.passed()
            else:
                Aqf.failed('Did not end with %s' % value)
            return status
        else:
            return False

    def contains(self, value):
        senval = str(self._get())
        if senval is not None:
            status = value in senval
            if status:
                Aqf.passed()
            else:
                Aqf.failed('Did not end with %s' % value)
            return status
        else:
            return False

    def __getattr__(self, name):
        func = getattr(operator, name, None)
        if func is not None:
            def wrap_func(arg):
                value = self._get()
                status = func(value, arg)
                if status:
                    Aqf.passed("Sensor '%s' has value '%s' it is **%s** %s" %
                               (self._name, str(value), name, str(arg)))
                else:
                    Aqf.failed("Sensor '%s' has value '%s' it is **not %s** %s" %
                               (self._name, str(value), name, str(arg)))
                return status
            return wrap_func

    def __str__(self):
        return str(self.get())


