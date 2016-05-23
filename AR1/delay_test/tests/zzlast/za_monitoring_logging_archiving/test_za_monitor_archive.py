###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""
from tests import settings, Aqf, AqfTestCase
from nosekatreport import aqf_vr, system, slow, site_acceptance

import re
import time
import random

import katconf
from tests.sample import Sample
from katmisc.utils.timestamp import Timestamp, Duration


class TestMonitoringAndArchiving(AqfTestCase):
    """Test that devices/subsystems are monitored and archived.

    This does not test the correctness of the mechanism, but that the required
    items are being monitored and archived. The correctness of the monitoring
    and archiving mechanism is covered by test_katstore and test_monitor.
    """

    MIN_RUN_PERIOD_SEC = 60 * 21
    """Minimum period of time in seconds that the system should be running before these
    tests can be performed. Worst case 'event-rate' strategy has rate of 600 (10mins) so
    allow time for at least two of these samples to be captured and tested."""
    STARTUP_DELAY = 60 * 20
    """ Time delay after system startup time, needed to ensure that all processes are up
    and running."""
    DEFAULT_KATCP_TIMEOUT = 30
    """Default timeout value for KATCP requests."""

    def setUp(self):
        self.sitename = settings.sitename
        self.system = settings.system

    def _get_test_sensors(self, filter_regex):
        pattern = re.compile(filter_regex)
        return [s for s in self.cam.sensor.keys()
                if pattern.search(s)]

    def katstore_check_sensor_samples(self, sensors_list, start_sec,
                                      monitor_strategies):
        """TestKatstore::katstore_check_sensor_samples.

        Parameters
        ----------
        sensors_list : list
            sensor names
        start_sec : int
            time in seconds since epoch to start looking at katstore
        monitor_strategies : list of tuples
            list of monitor sensors that have specific strategies defined

        This method iterates through a list of sensors provided as a parameter,
        works out the appropriate sensor name as stored in katstore and then checks
        that each sensor is listed in the historical sensor list, that it is
        referenced in a ccsv file, and lastly that its samples can
        actually be retrieved from katstore. It also performs a more thorough
        sample check on some randomly selected sensors.
        """

        # Length of default query - need to take into account that the longest default
        # sampling strategy curently is 'event-rate' with rate 600secs, so for more than
        # one sample allow for 20+ minutes.
        DEFAULT_QUERY_PERIOD_SEC = self.MIN_RUN_PERIOD_SEC

        # Random shuffle the sensor list to ensure different sensors get picked for
        # the thorough sample checks during different test runs
        ### random.shuffle(sensors_list)

        errors = 0
        for idx,sens in enumerate(sensors_list):

            # Get the katcp sensor_name as katstore would store it
            sens_obj = getattr(self.cam.sensor, sens)
            if sens_obj.name.startswith("agg"):
                # Skip the aggregate sensors - not applicable for this test
                continue
            else:
                sens_name = sens_obj.parent_name + "." + sens_obj.name
            Aqf.log_info("Checking sensor: %s using katstore katcp requests..." %
                         (sens_name))

            # Test historical_sensor_list
            response = self.cam.katstore.req.historical_sensor_list(sens_name)
            lines = '\n'.join([msg.arguments[0]
                               for msg in response.messages[1:]])
            it_passed = len(lines.splitlines()) == 1
            if not it_passed:
                Aqf.failed("Expected sensor not found in "
                           "historical sensor list: %s" % (sens_name))
                errors += 1
            else:
                Aqf.passed("Expected sensor was found in "
                           "historical sensor list: %s" % (sens_name))

                # Set end_seconds to the default query time
                end_sec = int(start_sec + DEFAULT_QUERY_PERIOD_SEC)

                # Test historical_sensor_data

                # Exclude the cbf proxy level sensors as the cbf device-handler is only added once
                # a product is configured
                cbf_proxy_regex = re.compile('data_[0-9][_.]cbf[_.][state|connected|device_status|build_state|api_version|address]')
                cbf_proxy_sensor_found = cbf_proxy_regex.match(sens)

                # Exclude cbf product sensors in the pattern cbf_c856M4k_[channels|bandwitdh|centerfrequency]
                product_regex = re.compile('data_[0-9][_.]cbf[_.]c[0-9]+M[0-9]+k[_.][a-z._]+')
                product_found = product_regex.match(sens)
                if not product_found and not cbf_proxy_sensor_found:

                    # Do thorough sample checks - find the configured sensor strategy for
                    # the current sensor under test if it has one, and check for exact match
                    # of the expected number of samples. But instead of performing the
                    # thorough check on all we just test a random subset of the sensors (we
                    # take every 10th sensor)

                    # Get the sensor strategy settings
                    strategy, strategy_params = self._get_sensor_strategy(
                        sens_obj, monitor_strategies)
                    errors += self._check_historical_sensor_data(sens_name, start_sec,
                        end_sec, strategy, strategy_params, thorough=True)
                else:
                    # This is a dynamically_created CBF sensor
                    # - we don't know how many samples to expect
                    # Skip checking the dynamically created sensors
                    Aqf.passed("Skipping checking of dynamically created sensors "
                               "on correlator: %s" % (sens_name))

        return(errors)

    def _get_sample_stats(self, str_samples, start_sec, end_sec):
        """ Display some simple stats to aid debugging """
        samples = [Sample.from_string(s) for s in str_samples]
        if samples:
            start_dt = samples[0].update_seconds - start_sec
            end_dt =  samples[-1].update_seconds - end_sec
            diffs = [samples[j].update_seconds - samples[j-1].update_seconds
                for j in range(1, len(samples))]
            Aqf.progress("Samples[%d]: first=start%+0.3fs last=end%+0.3fs "
                % (len(samples), start_dt, end_dt))
            if diffs:
                mean_delta_t = ((samples[-1].update_seconds-samples[0].update_seconds)
                / len(diffs))
                min_delta_t = min(diffs)
                max_delta_t = max(diffs)
            else:
                mean_delta_t, min_delta_t, max_delta_t = 0, 0, 0
            Aqf.progress("Deltas: min=%0.3fs mean=%0.3fs max=%0.3fs"
                    % (min_delta_t, mean_delta_t, max_delta_t))
        else:
            Aqf.progress ("No samples returned for period from %s to %s"
                % (Timestamp(start_sec), Timestamp(end_sec)))

    def _check_historical_sensor_data(self, sens_name, start_sec, end_sec,
                                      strategy, strategy_params, thorough=False):
        """Perform a historical sensor data check on the specified sensor."""

        response = self.cam.katstore.req.historical_sensor_data(
            sens_name, start_sec, end_sec, -1,
            "stepwise", 0, timeout=self.DEFAULT_KATCP_TIMEOUT)
        result = response.messages[0]
        lines = '\n'.join([msg.arguments[1] for msg in response.messages[1:]])
        samples = lines.splitlines()
        self._get_sample_stats(samples, start_sec, end_sec)
        actual_sample_count = len(samples)
        errors = 0
        if (thorough):
            # Do more thorough sample check - exact match of expected number of samples
            if strategy == 'event-rate':
                expected_samples = int(self.MIN_RUN_PERIOD_SEC/float(strategy_params[1]))
                expected_min = expected_samples - 1
                if actual_sample_count >= expected_min:
                    Aqf.passed("Sensor %s: Actual samples %d >= expected_min samples %d"
                                % (sens_name, actual_sample_count, expected_min))
                else:
                    Aqf.failed("Sensor %s: %r Actual samples %d < expected_min samples %d "
                               "(start_sec=%s, end_sec=%s, strategy='event-rate' %s" %
                               (sens_name, result, actual_sample_count, expected_min,
                                start_sec, end_sec, strategy_params))
                    errors += 1
            elif strategy == 'period':
                expected_samples = int(self.MIN_RUN_PERIOD_SEC/float(strategy_params))
                expected_min = expected_samples - 1
                expected_max = expected_samples + 2
                if actual_sample_count in range(expected_min, expected_max + 1):
                    Aqf.passed("Sensor %s: Actual samples %d in expected "
                               "sample range (%d, %d)"
                                % (sens_name, actual_sample_count,
                                   expected_min, expected_max))
                else:
                    Aqf.failed("Sensor %s: %r Actual samples %d not in expected "
                               "sample range (%d, %d) "
                               "(start_sec=%s, end_sec=%s, strategy='period' %s)" %
                               (sens_name, result, actual_sample_count,
                                expected_min, expected_max,
                                start_sec, end_sec, strategy_params))
                    errors += 1
            else:
                # Cannot really do anything more thorough for the other types of
                # strategies, so just do the normal check - at least one sample stored
                if actual_sample_count >= 1:
                    Aqf.passed("Sensor %s: Actual samples %d >= 1"
                                % (sens_name, actual_sample_count))
                else:
                    Aqf.failed("Sensor %s: %r Actual samples %d < 1; "
                               "(start_sec=%s, end_sec=%s, strategy='%s' %s)" %
                               (sens_name, result, actual_sample_count, start_sec, end_sec,
                                strategy, strategy_params))
                    errors += 1
        else:
            # Do a less thorough sample check - just verify that at least one
            # sample has been stored
            if actual_sample_count >= 1:
                Aqf.passed("Sensor %s: Actual samples %d >= 1"
                            % (sens_name, actual_sample_count))
            else:
                Aqf.failed("Sensor %s: Actual samples %d < 1 "
                           "(start_sec=%s, end_sec=%s)"
                            % (sens_name, actual_sample_count, start_sec, end_sec))
                errors += 1

        return errors


    def _get_sensor_strategy(self, sensor, monitor_strategies):
        """Get the sensor sampling strategy and strategy parameters.

        Some sensors have specific sampling strategies defined in the MonitorConfig. This
        function checks if this is the case for the given sensor and if not returns the
        default settings as defined by the monitors.

        Parameters
        ----------
        sensor : KATSensor
            the sensor object whose strategy are being queried
        monitor_strategies: list
            list of monitor sensors that have specific strategies defined (as read from
            the config)

        Returns
        -------
        (strategy, strategy_params) : tuple
            the strategy and its parameters
        """
        DEFAULT_EVENT_RATE_PARAMS = (0.0, 600.0)
        DEFAULT_PERIOD_PARAMS = 10.0

        # Check to see if sensor has specific sampling strategy specied in MonitorConfig
        strategy = None
        for conf_strategy in monitor_strategies:
            # Get sensor name in correct format to check for it against the sensor name as
            # defined in the monitor_strategies
            cmp_sensor_name = sensor.name.replace('.','_').replace('-','_')
            # Get the strategy
            if cmp_sensor_name == conf_strategy[0]:
                # Get the specific strategy as configured for the sensor
                strategy = conf_strategy[1][0]
                # Get the strategy parameters
                if strategy == 'period':
                    strategy_params = float(conf_strategy[1][1])
                elif strategy == 'event-rate':
                    strategy_params = (float(conf_strategy[1][1]),
                                       float(conf_strategy[1][2]))
                else:
                    # Currently the config only specifies strategy types 'period' and
                    # 'event-rate'. In case there is some other strategy type added later
                    # to the config let's use the default.
                    Aqf.log_warning(
                        '!! Found monitor config strategy specified for sensor %s '
                        'BUT test only supports "period" and "event-rate" for now'
                        'so are going to use the default settings' % (sensor.name, )
                    )
                    break
                # Found it, so exit the loop
                Aqf.log_info('Found monitor config strategy specified for sensor '
                             '%s : %s' % (sensor.name, conf_strategy[1]))
                break

        # If no specific sampling strategy specified in MonitorConfig use the defaults
        if strategy is None:
            # Use the default settings depending on the type of sensor
            if sensor.type in ['boolean', 'discrete', 'lru', 'string']:
                strategy = 'event-rate'
                strategy_params = DEFAULT_EVENT_RATE_PARAMS
            else:
                strategy = 'period'
                strategy_params = DEFAULT_PERIOD_PARAMS
            Aqf.log_info('No monitor config strategy found for sensor %s, '
                         'default = %s %s' % (sensor.name, strategy, strategy_params))

        return (strategy, strategy_params)

    def _test_monitor_archive_a_device(self, proxy, device):
        """Helper function to verify monitoring and archiving for a device.

        Parameters
        ----------

        proxy: str
        device: str

        This method retrieves a list of sensors for a given proxy and device.
        Thereafter it checks to ensure that the test has been running for at
        least 15 minutes (to ensure cached data is written to disk) before
        calling katstore_check_sensor_samples for each retrieved sensor.  The
        number of errors (failed attempts to retrieve sensor data from
        katstore) is asserted to be zero.
        """

        # Refresh the sensor list from the katstore database
        # self.cam.katstore.req.refresh_sensor_list()

        Aqf.step("Extract sensor list for proxy %s device %s" %
                 (proxy, device))


        # Extract sensors for proxy_device_* from cam.sensor
        # using cam.list_sensors
        test_sensors = self._get_test_sensors("^" + proxy + "_" + device + "_")
        # Check to ensure that we have found at least one sensor
        if len(test_sensors) > 0:
            Aqf.passed("%d sensors monitored for proxy %s device %s" %
                       (len(test_sensors), proxy, device))
        else:
            Aqf.failed("Zero sensors monitored for proxy %s device %s" %
                       (proxy, device))
        # The sys 'start_time' is captured when the system is started and not when all the
        # system processes are up and running. As such we have to add a few seconds to our
        # start time to ensure by that time all processes are up and running.
        start_sec = int(self.cam.sys.sensor.start_time.get_value() + self.STARTUP_DELAY)

        Aqf.step("Verify that katstore presents these sensors for proxy %s "
                 "device %s" % (proxy, device))
        # If not running long enough for data to have been written to katstore yet,
        # then skip this step. Need at least 20mins - the default strategy for monitor
        # sensors are 'event-rate' with rate set at 600s(10mins).
        run_time = time.time() - start_sec
        if run_time >= self.MIN_RUN_PERIOD_SEC:
            Aqf.log_info("Checking retrieval of older monitoring data from disk..."
                         "\n\tStart time = %s (%s)" % (time.ctime(start_sec), start_sec))
            # Need the MonitorConfig to know what the sensor strategies are for those
            # monitor sensors that do not make use of the default settings
            monitor_conf = katconf.MonitorConfig(self.cam.katconfig.config_path)
            monitor_strategies = monitor_conf.get_strategies(proxy)
            # Perform check on all sensor samples
            errors = self.katstore_check_sensor_samples(test_sensors,
                                                        start_sec,
                                                        monitor_strategies)

            Aqf.equals(errors, 0, "Number of errors during sample retrieval "
                       "from files through katstore katcp interface: %d" % errors)
        else:
            Aqf.skipped("System has not been running long enough to "
                        "save cached monitoring data to disk (%s < %s) - "
                        "cannot check disk retrieval"
                        % (Duration(run_time), Duration(self.MIN_RUN_PERIOD_SEC)))

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.14", "VR.CM.AUTO.M.32")
    @system("all")
    @slow
    def test_monitor_archive_correlator(self):
        """Test the monitoring and archiving of Correlator."""
        Aqf.step("Select correlator")
        if self.system == "mkat":
            proxy = "mcp"
            device = "cmc"
        elif self.system == "mkat_rts":
            proxy = "mcp"
            device = "cmc"
        else:  # kat7
            proxy = "dbe7"
            device = "dbe"
        Aqf.passed("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.15", "VR.CM.AUTO.M.32")
    @system("all")
    @slow
    def test_monitor_archive_sp(self):
        """Test the monitoring and archiving of Science Processor."""
        Aqf.step("Select science processor")
        if self.system in ["mkat_rts"]:
            proxy = "data_1"
            device = "spmc"
        elif self.system in ["mkat"]:
            proxy = "data_1"
            device = "spmc"
        else:  # kat7
            proxy = "dbe7"
            device = "katcp2spead"
        Aqf.passed("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.16", "VR.CM.AUTO.M.32")
    @system("all")
    @slow
    def test_monitor_archive_receivers(self):
        """Test the monitoring and archiving of Receivers."""

        selected = []
        # Select a random receptor proxy
        proxy = random.choice(self.cam.ants).name
        if self.system in ["mkat", "mkat_rts"]:
            device = "rsc"
            selected.append((proxy, device))
        else:  # kat7
            device = "rfe3"
            selected.append((proxy, device))
            proxy = random.choice(self.cam.ants).name
            device = "rfe5"
            selected.append((proxy, device))

        for proxy, dev in selected:
            Aqf.step("Select receivers")
            Aqf.passed("Selected proxy %s and device %s" % (proxy, dev))
            self._test_monitor_archive_a_device(proxy, device)

        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.17", "VR.CM.AUTO.M.32")
    @system("mkat", "mkat_rts")
    @slow
    def test_monitor_archive_digitisers(self):
        """Test the monitoring and archiving of Digitisers."""

        Aqf.step("Select dmc")
        proxy = "mcp"
        device = "dmc"
        Aqf.step("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)

        Aqf.step("Select digitisers")
        proxy = random.choice(self.cam.ants).name
        device = "dig"
        Aqf.step("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)

        Aqf.end()


    @site_acceptance
    @aqf_vr("Monitor RFE7", "VR.CM.AUTO.M.32")
    @system("kat7")
    @slow
    def test_monitor_archive_rfe7(self):
        """Test the monitoring and archiving of RFE7."""
        Aqf.step("Select RFE7")
        proxy = "rfe7"
        device = "rfe7"
        Aqf.step("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.12", "VR.CM.AUTO.M.32")
    @system("all")
    @slow
    def test_monitor_archive_antennas_receptors(self):
        """Test the monitoring and archiving of Antennas/Receptors."""

        selected = []
        # Select a random antenna/receptor proxy
        proxy = random.choice(self.cam.ants).name
        Aqf.step("Select antenna positioner")
        if self.system in ["mkat", "mkat_rts"]:
            device = "ap"
        else:  # kat7
            device = "antenna"
        selected.append((proxy, device))

        for proxy, dev in selected:
            Aqf.passed("Selected proxy %s and device %s" % (proxy, dev))
            self._test_monitor_archive_a_device(proxy, dev)
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.19", "VR.CM.AUTO.M.32")
    @system("all")
    @slow
    def test_monitor_archive_tfr(self):
        """Test the monitoring and archiving of TFR."""
        Aqf.step("Select TFR")
        if self.system in ["mkat", "mkat_rts"]:
            proxy = "anc"
            device = "tfr"
            waived = True
        else:  # kat7
            proxy = "anc"
            device = "tfr"
            waived = False

        Aqf.passed("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)
        if waived:
            Aqf.waived("TFR implementation waived for QBL(B)")
        Aqf.end()

    @site_acceptance
    @aqf_vr("Test_Cam_Notifications")
    @system("mkat", 'kat7')
    @slow
    def test_monitor_archive_notifications(self):
        """Test the monitoring and archiving of Notifications."""
        Aqf.step("Select Notifications")
        if self.system in ["mkat", "mkat_rts"]:
            proxy = "anc"
            device = "notifications"
        else:  # kat7
            proxy = "anc"
            device = "notifications"

        Aqf.passed("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.20")
    @system("mkat")
    @slow
    def test_monitor_archive_bms(self):
        """Test the monitoring and archiving of BMS."""
        Aqf.step("Select BMS")
        if self.system in ["mkat"]:
            proxy = "anc"
            device = "bms"

        Aqf.passed("Selected proxy %s and device %s" % (proxy, device))
        self._test_monitor_archive_a_device(proxy, device)
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.13")
    @system("mkat", "kat7")
    @slow
    def test_monitor_archive_wss(self):
        """Test the monitoring and archiving of wind and weather devices."""
        Aqf.step("Select WSS devices")
        proxy = "anc"
        if self.system in ["mkat"]:
            wind_device = "wind"
            weather_device = "weather"
        elif self.system in ["kat7"]:
            wind_device = "wind"
            weather_device = "air"



        Aqf.passed("Selected proxy %s and device %s" % (proxy, wind_device))
        self._test_monitor_archive_a_device(proxy, wind_device)

        Aqf.passed("Selected proxy %s and device %s" % (proxy, weather_device))
        self._test_monitor_archive_a_device(proxy, weather_device)

        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.18")
    @system("mkat")
    @slow
    def test_monitor_archive_vds(self):
        """Test the monitoring and archiving of the vds device."""
        Aqf.step("Select vds devices")
        if self.system in ["mkat"]:
            proxy = "anc"
            vds_device = "vds"
        Aqf.passed("Selected proxy %s and device %s" % (proxy, vds_device))
        self._test_monitor_archive_a_device(proxy, vds_device)

        Aqf.end()



    @site_acceptance
    @aqf_vr("VR.CM.AUTO.M.11")
    @system("all")
    @slow
    def test_config_of_monitor_freq_startup(self):
        """Test the startup configuration of the monitoring frequency of
        sensors.

         .. Test procedure
            --------------
            Test that the CAM allows for the configuration of the monitoring frequency of
            monitoring points at startup:
                1. Select all of the sensors with a specific sampling rate specified in
                the monitor strategies configuration file.
                2. For each of the sensors:
                    2.1. Extract historical data of the sensor for a period more than 20
                    minutes (+20 minutes required to have at least more than one sample
                    for the sensors that use the default 'event-rate' strategy with rate
                    600secs).
                    2.2. Verify that the number of gathered samples for that time period
                    are as per the configured frequency for the sensor.
        """

        # The sys 'start_time' is captured when the system is started and not when all the
        # system processes are up and running. As such we have to add a few seconds to our
        # start time to ensure by that time all processes are up and running.
        system_start_time = self.cam.sys.sensor.start_time.get_value()
        start_sec = int(system_start_time + self.STARTUP_DELAY)

        # Set end_seconds
        end_sec = int(start_sec + self.MIN_RUN_PERIOD_SEC)

        Aqf.step("Select all of the sensors with a specific sampling rate specified "
                 "in the monitor strategies configuration file.")
        # If not running long enough for data to have been written to katstore yet,
        # then skip this step. Need at least 20mins - the default strategy for monitor
        # sensors are 'event-rate' with rate set at 600s(10mins).
        run_time = time.time() - start_sec
        if run_time >= self.MIN_RUN_PERIOD_SEC:
            Aqf.step("System has been running for %s => Fetching samples "
                     "for period=%s from (system_start+%s) to (now-%s)" %
                     (Duration(run_time), Duration(end_sec-start_sec),
                      Duration(start_sec-system_start_time),
                      Duration(time.time()-end_sec)))
            # Get the components that are monitored in the system
            monitor_conf = katconf.MonitorConfig(self.cam.katconfig.config_path)
            components = []
            for node in monitor_conf.monitor_nodes:
                components.extend(monitor_conf.get_components_to_monitor(node))
            # For each component get the sensors that have specific sampling strategies
            # specified in the configuration file - also fail test if a sensor is
            # encountered that does not exist in the system
            monitored_sensor_strategies = []
            errors = 0
            for comp in components:
                for s in monitor_conf.get_strategies(comp[0]):
                    # Get the full sensor name with only _ chars
                    _sens_name = '_'.join([comp[0], s[0]])
                    # Get the sensor object
                    try:
                        sens_obj = getattr(self.cam.sensor, _sens_name)
                        monitored_sensor_strategies.append(
                            ('.'.join([comp[0], sens_obj.name]), s[1])
                        )
                    except AttributeError:
                        Aqf.failed("Invalid sensor name '%s' in config file" %
                                   _sens_name)
                        errors += 1

            # Test each of the randomly selected monitored sensors
            for mss in monitored_sensor_strategies:
                query_sens_name = mss[0]
                Aqf.log_info("Retrieving historical data for sensor '%s'..." %
                             query_sens_name)

                strategy = mss[1][0]
                if strategy == 'event-rate':
                    strategy_params = (float(mss[1][1]), float(mss[1][2]))
                elif strategy == 'period':
                    strategy_params = float(mss[1][1])
                else:
                    Aqf.failed("Strategy '%s' NOT supported in current test!" % strategy)

                Aqf.log_info("\t Strategy=%s %s" % (strategy, strategy_params))
                errors += self._check_historical_sensor_data(query_sens_name, start_sec,
                    end_sec, strategy, strategy_params, thorough=True)

            Aqf.equals(errors, 0, "Number of errors during test: %d" % errors)
        else:
            Aqf.skipped("System has not been running long enough to "
                        "save cached monitoring data to disk (%s < %s) - "
                        "cannot run test_config_of_monitor_freq_startup"
                        % (Duration(run_time), Duration(self.MIN_RUN_PERIOD_SEC)))
        Aqf.end()

    @aqf_vr("VR.CM.AUTO.M.11")
    @system("all")
    @slow
    def TODO_xtest_config_of_monitor_freq_runtime(self):
        """Test the runtime configuration of the monitoring frequency of sensors.

        ..  Test procedure
            --------------
            Test that the CAM allows for the configuration of the monitoring frequency of
            monitoring points during runtime:
                1. Select a number of sensors and change their monitor sampling strategy
                to 'period' with rate equal to 0.5s.
                2. For each of the chosen sensors:
                    2.1. Extract historical data of the sensor for a period of 1 minute.
                    2.2. Verify that the number of gathered samples during that 1 minute
                    equals 120.
                2. Select a number of sensors and change their sampling strategy to
                'event-rate' with rate strategy parameters equal to (0.0, 2.0).
                3. For each of the chosen sensors:
                    3.1. Extract historical data of the sensor for a period of 1 minute.
                    3.2. Verify that the number of gathered samples during that 1 minute
                    equals 30.
        """

        NUM_SENSORS_TO_TEST = 10
        TEST_STRATEGY = ('period', 0.5)

        Aqf.step("Select a number of sensors and change their sampling strategy to "
                 "'period' with rate equal to 0.5s.")
        test_sensors = random.sample(self.cam.sensor.__dict__.items(),
                                     NUM_SENSORS_TO_TEST)
        #TODO: Complete test postRTS ... functionality still outstanding ... see Mantis
        # 3145. For #now we just leave it out (named as 'Xtest_') because it is not
        # specified for RTS.
        Aqf.end()
