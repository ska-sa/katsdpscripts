###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""
#import os
#import logging
import re
import time

import katcorelib

from katmisc.utils.timestamp import Timestamp, Duration
from tests import fixtures, Aqf, AqfTestCase

from nosekatreport import (aqf_vr, system, slow, site_acceptance)


@system("all")
class TestMKatstore(AqfTestCase):

    """Tests samples in new MeerKAT katstore.

    TBD: This test only covers the sensors in the first 60 seconds after the
        tests started. Can be expanded to cover last few minutes (to ensure
        katstore local cache is working) as well as history older than 15
        minutes (to ensure reading from file)

    """

    def setUp(self):
        pass

    def _get_test_sensors(self, filter_regex):
        pattern = re.compile(filter_regex)
        return [s for s in self.cam.sensors.keys()
                if pattern.search(s)]

    def katstore_check_sensor_samples(self, sensors_list, start_sec,
                                      run_period_sec, sampling_period_ms):
        """TestMKatstore::katstore_check_sensor_samples."""

        errors = 0
        start_sec = int(start_sec)
        run_period_sec = int(run_period_sec)
        end_sec = start_sec + run_period_sec
        timeout = max(30, run_period_sec / 5)

        for sens in sensors_list:
            #Get the katcp sensor_name as katstore would store it
            sens_obj = getattr(self.cam.sensors, sens)
            sens_name = sens_obj.parent_name + "." + sens_obj.name
            Aqf.progress("Checking sensor: %s using katstore katcp requests" %
                            (sens_name))

            #Test historical_sensor_list
            response = self.cam.katstore.req.historical_sensor_list(sens_name)
            lines = '\n'.join([msg.arguments[0] for msg in response.messages[1:]])
            it_passed = len(lines.splitlines()) == 1
            if not it_passed:
                Aqf.failed("Expected sensor not found in "
                               "historical sensor list: %s" % (sens_name))
                errors = errors + 1
            else:
                Aqf.passed("Expected sensor was found in "
                               "historical sensor list: %s" % (sens_name))

                # Exclude cbf product sensors in the patter: cbf_c856M4k_[channels|bandwitdh|centerfrequency]
                product_regex = re.compile('data_[0-9][_.]cbf[_.]c[0-9]+M[0-9]+k[_.][a-z._]+')
                product_found = product_regex.match(sens)
                #Test historical_sensor_data
                response = self.cam.katstore.req.historical_sensor_data(sens_name, int(start_sec), int(end_sec), -1, "stepwise", 0, timeout=timeout)
                # response = self.cam.katstore_query.req.historical_sensor_data(
                #     sens_name, int(start_sec), int(end_sec), -1, "stepwise", 0,
                #     timeout=timeout)
                lines = '\n'.join([msg.arguments[1] for msg in response.messages[1:]])
                expected_samples = int(run_period_sec*1000.0 / sampling_period_ms)
                actual_samples = len(lines.splitlines())
                if not product_found:
                    it_passed = actual_samples > (0.9 * expected_samples) # Expect 90% sampling
                    if not it_passed:
                        Aqf.failed("Sensor returned too little historical "
                                       "data samples: %s expected=%d actual=%d" %
                                       (sens_name, int(expected_samples), int(actual_samples)))
                        errors = errors + 1
                    else:
                        Aqf.passed("Sensor returned expected number of "
                                       "historical data samples: %s expected=%d actual=%d" %
                                       (sens_name, int(expected_samples), int(actual_samples)))
                else:
                    Aqf.passed("Dynamic sensor returned the following "
                                       "historical data samples: %s actual=%d" %
                                       (sens_name, int(actual_samples)))

        return(errors)

    @site_acceptance
    @aqf_vr('CAM_MON_storage')
    def test_monitor_storage_sensors(self):
        """TestKatstore::test_monitor_storage_sensors.

        Test the storage sensors on the monitor - this is going to need
        to be updated when the monitor exposes new katstore sensors.

        """

        errors = 0
        skipped = 0
       
        storage_sensors = self._get_test_sensors('storage')
        for sens in storage_sensors:
            it_passed = None
            sens_obj = getattr(self.cam.sensors, sens)
            sens_name = sens_obj.parent_name + "." + sens_obj.name
            Aqf.progress("Checking storage sensor: %s" % (sens_name))

            if "qrate.receive" in sens_name:
                result = sens_obj.get_value()
                it_passed = sens_obj.status == 'nominal' and result > 0
                msg = sens_name + "=" + str(result)
            elif "qsize" in sens_name:
                result = sens_obj.get_value()
                it_passed = sens_obj.status == 'nominal' and result < 50
                msg = sens_name + "=" + str(result)
            if it_passed is False:
                Aqf.failed("Storage sensor failed: "
                               "%s (%s)" % (sens_name, msg))
                errors = errors + 1
            elif not it_passed:
                Aqf.step("Storage sensor skipped: "
                               "%s" % (sens_name))
                skipped += 1
            else:
                Aqf.passed("Storage sensor passed: %s (%s)" %
                               (sens_name, msg))
        Aqf.equals(0, errors, "Monitor storage sensor tests")
        Aqf.step("Skipped %s Monitor storage sensors" % skipped)
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_MON_new_storage_sensors_exposed')
    def test_katstore_sensors_exposed(self):
        """TestMKatstore::test_katstore_sensors_exposed
        Test that at least one sensor is exposed by the anc and by each antenna proxy."""
        #self.register_requirements(["RT_temp01","RT_temp02"])
        expected_sensors = set(['%s_enviro_mean_wind_speed' % (ant.name) for ant in self.cam.ants])
        expected_sensors.add('anc_mean_wind_speed')
        diff = expected_sensors.difference(self.cam.sensors.keys())
        it_passed = len(diff) == 0
        if not it_passed:
            Aqf.failed("Expected sensors: %r were not "
                           "found in cam.sensors" % (list(diff),))
        else:
            Aqf.passed("All expected sensors: %r were "
                           "found in cam.sensors" % (list(expected_sensors),))
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_MON_new_retrieve_most_recent_data')
    def test_katstore_sample_retrieve_most_recent_data(self):
        """TestMKatstore::test_katstore_sample_retrieve_recent_data
        Test retrieval of recent samples from through katstore server."""

        #self.register_requirements(["RT_temp01","RT_temp02"])

        errors = 0
        test_sensors = []
        test_sensors = self._get_test_sensors('_enviro_mean_wind_speed')
        # test_sensors.append('anc_wind_wind_speed')
        Aqf.is_true(len(test_sensors) >= 1, 'There is at least one test sensor')
        sampling_period_ms = 1000  #ms - assume sampling rate is once per second

        Aqf.step("Checking sensors %s" % test_sensors)

        # Check the samples from 30s after kat-test for a period of 60s
        run_period_sec = 180  #Time the system must have been running
        earlier = 60  #Retrieve data from 1 minute little earlier than now
        start_sec = int(self.cam.sys.sensor.start_time.get_value() + 30) # Startiing not earlier than 30s after system has started
        waiting = 0
        while (time.time() - start_sec) < (run_period_sec + earlier + 3): #Allow 3 second leeway
            Aqf.progress("Waiting for time to elapse %d" % (waiting))
            waiting += 1
            time.sleep(1.0) # Sleep for one more seconds

        #Check the recent history to test retrieval from cached monitoring data
        start_sec = int(time.time() - run_period_sec - earlier) # Start a little earlier than now
        Aqf.progress("Checking retrieval of recent monitoring data from "
                     "cache - starting at %d (%s) for period %.f" %
                      (start_sec, Timestamp(start_sec), run_period_sec))

        errors = errors + self.katstore_check_sensor_samples(
            test_sensors, start_sec, run_period_sec, sampling_period_ms)

        Aqf.equals(0, errors, "Sample retrieval from cache through katstore "
                         "katcp interface : %s" % (test_sensors))
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_MON_new_retrieve_older_data')
    def test_katstore_sample_retrieve_older_data(self):
        """TestMKatstore::test_katstore_sample_retrieve_older_data
        Test retrieval of older samples through katstore server."""

        errors = 0
        skipped = 0
        test_sensors = []
        test_sensors = self._get_test_sensors('_enviro_mean_wind_speed')
        # test_sensors = self._get_test_sensors('(ant[1-7]_enviro|m[0-9][0-9][0-9]_enviro)_wind_speed')
        # test_sensors.append('anc_wind_wind_speed')
        Aqf.is_true(len(test_sensors) >= 1, 'There is at least one test sensor')
        sampling_period_ms = 1000  #ms - assume sampling rate is once per second
        run_period_sec = 300  #Period to retrieve

        Aqf.step("Checking sensors %s" % test_sensors)

        #If system has been running for more than the camstore
        #caching time (of 15 minutes)
        start_sec = int(self.cam.sys.sensor.start_time.get_value() + 300) # Start 5 min after system has started

        if (time.time() - self.cam.sys.sensor.start_time.get_value()) > (20*60):
            Aqf.progress("Checking retrieval of older monitoring data from "
                            "disk - starting at %d (%s) for period %.f" %
                        (start_sec, Timestamp(start_sec), run_period_sec))
            errors = errors + self.katstore_check_sensor_samples(
                test_sensors, start_sec, run_period_sec, sampling_period_ms)
        else:
            Aqf.skipped("Skipped: System has not been running long enough to "
                           "have saved cached monitoring data to disk - "
                           "cannot check disk retrieval")
            skipped = skipped + 1

        Aqf.equals(0, errors,"Sample retrieval from files through katstore "
                         "katcp interface: %s" % (test_sensors))

        if skipped > 0:
            Aqf.passed("Sample retrieval from files through katstore "
                       "katcp interface skipped: %s" % (test_sensors))
        else:
            Aqf.passed("Sample retrieval from files through katstore "
                       "katcp interface passed: %s" % (test_sensors))
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_MON_new_retrieve_through_katcorelib_from_files')
    def _test_katcorelib_sensors_older(self):
        """TestMKatstore::test_katcorelib_sensors_older
        Tests sample history through katcorelib - older data retrieved from disk."""
        start_sec = int(self.cam.sys.sensor.start_time.get_value() + 300) # Start 5 min after system has started
        self._check_katcorelib_sensor_sampling(start_sec, 60)
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_MON_new_retrieve_through_katcorelib_from_cache')
    def _test_katcorelib_sensors_recent(self):
        """TestMKatstore::test_katcorelib_sample_sensors_recent
        Tests sample history through katcorelib - recent data retrieved from cache."""
        #Check the recent history to test retrieval from cached monitoring datai
        run_period_sec = 60
        start_sec = int(time.time() - run_period_sec - 10) # Start a little earlier than now
        self._check_katcorelib_sensor_sampling(start_sec, run_period_sec)
        Aqf.end()

    def _check_katcorelib_sensor_sampling(self, start_sec, run_period_sec):
        """Function which checks the sample history through katcorelib."""

        errors = 0

        wrap_store = katcorelib.utility.KATStoreWrapper()
        wrap_store._client = self.cam.katstore

        # Check the samples from 2s after kat-test for a period of 60s
        while (time.time() - start_sec) < run_period_sec:
            Aqf.progress("Waiting for time to elapse")
            time.sleep(1.0)  # Sleep for one more second

        end_sec = start_sec + run_period_sec
        timeout = min(5, run_period_sec / 2)

        test_sensors = self._get_test_sensors('%s_antenna_acs_actual|%s_ap_actual')
        Aqf.step("Checking sensors: %s " % test_sensors)
        # Ant2 antenna.acs.actual-* sensors has a 0.5s sampling rate
        sampling_period_ms = 500  # ms
        for sens in test_sensors:
            # Get the katcp sensor_name as katstore would store it
            sens_obj = getattr(self.cam.sensors, sens)
            sens_name = sens_obj.parent_name + "." + sens_obj.name
            Aqf.progress("Checking sensor: %s using katcorelib katstore wrapper"
                         " (on katsture_query) and get_stored_history" %
                         (sens_name))

            # Test sensor list
            lines = wrap_store.get_sensor_list(sens_name)
            it_passed = len(lines.splitlines()) == 1
            if not it_passed:
                Aqf.failed("Sensor not found in "
                           "katcorelib sensor list: %s" % (sens_name))
                errors = errors + 1
            else:
                Aqf.passed("Sensor was found in "
                           "katcorelib sensor list: %s" % (sens_name))

            #Test timeranges
            lines = wrap_store.get_data_list(
                sens_name, int(start_sec), int(end_sec), timeout=timeout)
            it_passed = len(lines.splitlines()) >= 1
            if not it_passed:
                Aqf.failed("Sensor returned no "
                           "katcorelib data list: %s" % (sens_name))
                errors = errors + 1
            else:
                Aqf.passed("Sensor returned a katcorelib "
                           "data list: %s" % (sens_name))

            #Test data
            data = wrap_store.get_data(
                sens_name, int(start_sec), int(end_sec), period_seconds=-1,
                resample='stepwise', last_known=False, query_timeout=timeout)
            expected_samples = run_period_sec * 1000.0 / sampling_period_ms
            actual_samples = len(data)
            # Expect 90% sampling
            it_passed = actual_samples > (0.9 * expected_samples)
            if not it_passed:
                Aqf.failed("Sensor returned too little "
                           "katcorelib data: %s, expected=%d actual=%d" %
                           (sens_name, int(expected_samples),
                            int(actual_samples)))
                errors = errors + 1
            else:
                Aqf.passed("Sensor returned the "
                           "expected number of katcorelib data samples: "
                           "%s, expected=%d actual=%d" %
                           (sens_name, int(expected_samples),
                            int(actual_samples)))

            #Test get_stored_history against wrapper get_data
            results = sens_obj.get_stored_history(int(start_sec), int(end_sec))
            actual_katcorelib = len(results[2])
            it_passed = actual_katcorelib > (0.9 * actual_samples)
            if not it_passed:
                Aqf.failed("Sensor returned different "
                           "nr of samples in get_stored_history: %s, "
                           "katstore=%d katcorelib=%d" %
                           (sens_name, int(actual_samples),
                            int(actual_katcorelib)))
                errors = errors + 1
            else:
                Aqf.passed("Sensor returned same "
                           "nr of samples: %s, katstore wrapper=%d "
                           "get_stored_history=%d" %
                           (sens_name, int(actual_samples),
                            int(actual_katcorelib)))

            #Test get_stored_history
            ok_results = [x for x in results[2] if x != 'failure']
            it_passed = len(ok_results) > (0.9 * actual_samples)
            if not it_passed:
                Aqf.failed("Sensor returned too many "
                           "failure samples in get_stored_history: %s "
                           "non-failure=%d actual=%d" %
                           (sens_name, int(len(ok_results)),
                            int(actual_samples)))
                errors = errors + 1
            else:
                Aqf.passed("Sensor returned"
                           "non-failure samples in get_stored_history: "
                           "%s, non-failure=%d actual=%d" %
                           (sens_name, int(len(ok_results)),
                            int(actual_samples)))

        Aqf.equals(0, errors, "Tests for katstore through katcorelib")
