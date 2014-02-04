import logging

import katpoint


user_logger = logging.getLogger('user')

# List of available projections and the default one to use
PROJECTIONS = katpoint.plane_to_sphere.keys()
DEFAULT_PROJ = 'ARC'
# Move default projection to front of list
PROJECTIONS.remove(DEFAULT_PROJ)
PROJECTIONS.insert(0, DEFAULT_PROJ)


def report_compact_traceback(tb):
    """Produce a compact traceback report."""
    print '--------------------------------------------------------'
    print 'Session interrupted while doing (most recent call last):'
    print '--------------------------------------------------------'
    while tb:
        f = tb.tb_frame
        print '%s %s(), line %d' % (f.f_code.co_filename, f.f_code.co_name, f.f_lineno)
        tb = tb.tb_next
    print '--------------------------------------------------------'


class ScriptLogHandler(logging.Handler):
    """Logging handler that writes logging records to HDF5 file via ingest.

    Parameters
    ----------
    data : :class:`KATClient` object
        Data proxy device for the session

    """
    def __init__(self, data):
        logging.Handler.__init__(self)
        self.data = data

    def emit(self, record):
        """Emit a logging record."""
        try:
            msg = self.format(record)
# XXX This probably has to go to cam2spead as a req/sensor combo [YES]
#            self.data.req.k7w_script_log(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class ObsParams(dict):
    """Dictionary-ish that writes observation parameters to CAM SPEAD stream.

    Parameters
    ----------
    data : :class:`KATClient` object
        Data proxy device for the session
    product : string
        Name of data product

    """
    def __init__(self, data, product):
        dict.__init__(self)
        self.data = data
        self.product = product

    def __setitem__(self, key, value):
        # XXX Changing data product name -> ID in a hard-coded fashion
        self.data.req.set_obs_param(self.product, key, repr(value))
        dict.__setitem__(self, key, value)


class RequestSensorError(Exception):
    """Critical request failed or critical sensor could not be read."""
    pass


class CaptureSession(object):

    def __init__(self, telescope, product, **kwargs):
        
        self.telescope = telescope
        data = telescope.data_rts
        if not data.is_connected():
            raise ValueError("Data proxy %r is not connected "
                             "(is the RTS system running?)" % (data.name,))
        self.data = data

        # Default settings for session parameters (in case standard_setup is not called)
        self.receptors = None
        self.project_id = self.program_id = self.experiment_id = 'interactive'
        self.stow_when_done = False
#        self.nd_params = {'diode': 'coupler', 'on': 0., 'off': 0., 'period': -1.}
#        self.last_nd_firing = 0.
#        self.output_file = ''
#        self.dump_period = self._requested_dump_period = 0.0
        self.horizon = 3.0
#        self._end_of_previous_session = dbe.sensor.k7w_last_dump_timestamp.get_value()
        self.obs_params = {}

        # Set data product
        user_logger.info('Setting data product to %r (this may take a while...)' %
                         (product,))
        data.req.capture_init(product)

        user_logger.info('==========================')
        user_logger.info('New data capturing session')
        user_logger.info('--------------------------')
        user_logger.info('Data proxy used = %s' % (data.name,))
        user_logger.info('Data product = %s' % (product,))

        # Log details of the script to the back-end
        dbe.req.k7w_set_script_param('script-starttime', time.time())
        dbe.req.k7w_set_script_param('script-endtime', '')
        dbe.req.k7w_set_script_param('script-name', sys.argv[0])
        dbe.req.k7w_set_script_param('script-arguments', ' '.join(sys.argv[1:]))
        dbe.req.k7w_set_script_param('script-status', 'busy')

        # self.projection
        # self.coordsystem
        # self.delay_tracking
        # self.sources
        # self.dry_run
        # self.output_file
        # self.last_nd_firing

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def time(self):
        """Current time in UTC seconds since Unix epoch."""
        return self.telescope.time()

    def sleep(self, seconds):
        """Sleep for the requested duration in seconds."""
        self.telescope.sleep(seconds)

    def get_centre_freq(self, dbe_if=None):
        

    def set_centre_freq(self, centre_freq):
        pass

    def standard_setup(self, observer, description, experiment_id=None,
                       centre_freq=None, dump_rate=1.0, nd_params=None,
                       stow_when_done=None, horizon=None,
                       dbe_centre_freq=None, no_mask=False, **kwargs):
        

    def capture_start(self):
        

    def label(self, label):
        

    def set_target(self, target, component=''):
        
        
    target = property()

    def on_target(self):
        

    def target_visible(self, target=None, duration=0., timeout=300.):
        

    @dynamic_doc("', '".join(PROJECTIONS), DEFAULT_PROJ)
    def set_projection(self, projection='', coordsystem=''):
        

    def fire_noise_diode(self, diode='coupler', on=10.0, off=10.0, period=0.0,
                         align=True, announce=True):
        pass

    def track(self, duration=20.0, announce=True):

    def scan(self, duration=30.0, start=(-3.0, 0.0), end=(3.0, 0.0), index=-1,
             announce=True):
        pass

    def raster_scan(self, num_scans=3, scan_duration=30.0, scan_extent=6.0,
                    scan_spacing=0.5, announce=True):
        pass

    def radial_scan(self, num_scans=3, scan_duration=30.0, scan_extent=6.0,
                    announce=True):
        pass

    def generic_scan(self, scanfunc, duration=30.0, scale=1.0, announce=True):
        pass

    def end(self, interrupted=False):
