from ConfigParser import SafeConfigParser, NoSectionError, Error
import csv

import numpy as np

from katpoint import Catalogue
from katcp import DeviceServer
from katcp.kattypes import return_reply, Str

from katsdpscripts.fake.updater import WarpClock, PeriodicUpdaterThread
from katsdpscripts.fake.sensor import FakeSensor
from katsdpscripts.fake.client import FakeClient, ClientGroup, IgnoreUnknownMethods
from katsdpscripts.fake import models


__version__ = 'dev'


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


def load_config(config_file):
    split = lambda args: csv.reader([args], skipinitialspace=True).next()
    cfg = SafeConfigParser()
    files_read = cfg.read(config_file)
    if files_read != [config_file]:
        raise Error('Could not open config file %r' % (config_file,))
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


class FakeTelescope(object):
    """Connection object for a simulated KAT system."""
    def __init__(self, config_file, dry_run=False, start_time=None):
        self._telescope = load_config(config_file)
        self.sensors = IgnoreUnknownMethods()
        self._clock = WarpClock(start_time, dry_run)
        self._clients = []
        groups = {}
        for comp_name, component in self._telescope.items():
            if component['class'] == 'Group':
                groups[comp_name] = component['attrs']['members']
                continue
            model = vars(models).get(component['class'] + 'Model')
            client = FakeClient(comp_name, model, self._telescope, self._clock)
            setattr(self, comp_name, client)
            self._clients.append(client)
            # Add component sensors to the top-level sensors group
            for sensor_args in component['sensors']:
                sensor_name = sensor_args[0]
                sensor = getattr(client.sensor, sensor_name)
                setattr(self.sensors, comp_name + '_' + sensor_name, sensor)
        for group_name, client_names in groups.items():
            group = ClientGroup(group_name, [getattr(self, client_name, None)
                                             for client_name in client_names],
                                self._clock)
            setattr(self, group_name, group)
        self.updater = PeriodicUpdaterThread(self._clients, self._clock, period=0.1)
        self.updater.start()

    def __del__(self):
        """Before deleting object, stop the system (this might not get called!)."""
        self.updater.stop()
        self.updater.join()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and stop the system."""
        self.updater.stop()
        self.updater.join()
        # Don't suppress exceptions
        return False

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

    def receptor_subset(self, rcps, name='receptors'):
        """Group a subset of receptors from a flexible specification.

        Parameters
        ----------
        rcps : :class:`ClientGroup` or :class:`FakeClient` object / list / string
            Receptors specified by a ClientGroup object containing receptor
            clients, or a single receptor client or a list of receptor clients,
            or a string of comma-separated receptor names, or the string 'all'
            for all receptors controlled by the FakeTelescope
        name : string, optional
            Name of receptor subset

        Returns
        -------
        group : :class:`ClientGroup` object
            ClientGroup object containing selected receptor clients

        Raises
        ------
        ValueError
            If receptor with a specified name is not found on FakeTelescope

        """
        if isinstance(rcps, ClientGroup):
            rcps.name = name
            rcps._clock = self._clock
            return rcps
        elif isinstance(rcps, FakeClient):
            return ClientGroup(name, [rcps], self._clock)
        elif isinstance(rcps, basestring):
            if rcps.strip() == 'all':
                return self.rcps
            try:
                rcps = [getattr(self, rcp.strip()) for rcp in rcps.split(',')]
            except AttributeError:
                raise ValueError("Receptor %r not found on Telescope" % (rcp,))
            return ClientGroup(name, rcps, self._clock)
        else:
            # The default assumes that *rcps* is a list of receptor clients
            return ClientGroup(name, rcps, self._clock)

        # self.system = system
        # self.sb_id_code = sb_id_code
        # self.connected_objects = {}
        # self.controlled_objects = []
        # self.sources = katpoint.Catalogue()
        # self.attributes = attributes if attributes is not None else {}
        # self.server = FakeCamEventServer(self.attributes, opts.sensor_list,
        #                                  host=opts.fake_cam_host,
        #                                  port=opts.fake_cam_port) if opts else None
