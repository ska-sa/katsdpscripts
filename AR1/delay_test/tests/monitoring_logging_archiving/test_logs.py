###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""

from datetime import datetime
from tests import (tests_start_time, utils, settings, Aqf, AqfTestCase)
from nosekatreport import (aqf_vr, system)
from katmisc.utils import timestamp
from katcorelib.utility import build_client  # , KATClient
from tests import utils, specifics


@system('all')
class TestLogging(AqfTestCase):

    """Test all log components of the system."""

    def setUp(self):
        #Connect to katlogserver
        self.system = settings.system
        self.logserver = build_client("katlogserver",
                                      host=self.cam.katlogserver.address[0],
                                      port=self.cam.katlogserver.address[1],
                                      controlled=True)

    @aqf_vr('CAM_LOGGING_logserver_processes_running')
    def test_kat_log_processes(self):
        """Test that kat_central_logger and katlogserver is running."""
        Aqf.step("Check kat_central_logger and katlogserver is up and running")
        ok = True
        # katlogger and katlogserver has been moved to the nm_portal node,
        # so make sure we use the sensor value to check if it is running
	sensors = self.cam.list_sensors('katlogger.running|katlogserver.running')
	for sensor in sensors:
            aqf_sensor = Aqf.sensor('cam.sensor.%s' % sensor.python_identifier)
            aqf_sensor.wait_until_status(is_in=['nominal'])
	    
        # for process in ["kat-central-logger",
        #                 "kat-log-server"]:
        #     res, s_msg = utils.check_process_in_ps(process)
        #     if not res:
        #         ok = False
        #         Aqf.failed(process + ': ' + s_msg)

        self.assertTrue(ok, msg="Some log processes not found")
        Aqf.passed("Log processes all OK")
        Aqf.end()

    @aqf_vr('VR.CM.AUTO.L.33', 'VR.CM.AUTO.L.37')
    def test_log_files(self):
        """
        Check that each katcorelib client has a log file started later than
        start of tests.
        """
        ok = True
        #msgs = []

        logserver = self.logserver
        output = logserver.req.list_logs()
        ok = True
        #msg = []
        log_dict = {}
        #Split each #list-logs inform returned
        log_infos = [line.split(" ") for line in str(output).splitlines()]

        #Build a dictionary with the log info msgs returned
        for log_info in log_infos:
            inf = log_info[0]
            if inf == "#list-logs":
                inf, filename, tstamp, size = log_info
                log_dict[filename] = (tstamp, size)
        Aqf.step("Check the logs of headnode processes")
        #Now do the checking
        #Check logs of central processes
        #prefix = "kat_" if "karoo_kat7" in self.cam.system else ""
        central_procs = ["kat.kataware", "kat.log.central", "kat.launcher",
                         "kat.sched", "kat.katpool", "kat.katsyscontroller",
                         "kat.katexecutor", "kat.kat-fileserver",
                         "kat.katstore.pulld", "kat.katstore.query",
                         "kat.katstore.hdfserver",
                         "activity", "alarms", "ils"]

        # Add monitors for each node:
        for node in self.cam.katconfig.monitor_nodes:
            central_procs.append("kat.mon_%s" % node)

        controlled_clients = []
        for client in self.cam.katconfig.single_ctl:
            controlled_clients.append("kat." + client)

        #Check logs of headnode processes and katcorelib clients
        Aqf.progress("Checking logs for : %s" %
                     (central_procs + controlled_clients))

        for which in central_procs + controlled_clients:
            Aqf.step('Check log file for %s' % which)
            ok = True
            #Check the dictionary entry and timestamp
            tstamp, size = log_dict.get(which, (0, 0))
            if tstamp == 0:
                ok = False
                stext = "No log found for %s" % which
                Aqf.failed(stext)
                continue

            #Check size
            if size == 0:
                ok = False
                stext = "File size is 0 for %s" % which

            #Check the timestamp and size
            if tstamp < tests_start_time:
                ok = False
                stext = "File timestamp too old (%s) for %s" % (
                    timestamp.seconds_to_dtstring(tstamp, output_utc=False),
                    which)

            #Check the tail of the log
            Aqf.step("Check tail of log file %s" % which)
            output = logserver.req.tail_log(which, 10)
            if len(output.messages) <= 1:
                #Only !list-logs ok 0 received
                ok = False
                stext = "No tail found for %s" % which
            else:
                Aqf.step('Verify that timestamps in log file %s are in '
                         'milliseconds by extracting the date and millisecond '
                         'portion of the timestamp' % which)
                try:
                    #Process log messages from the tail
                    found = True
                    for i in range(1, len(output.messages)):
                        log_msg = output.messages[i].arguments[0]
                        if log_msg != "" and (log_msg[0] in ('0123456789')):
                            # Log message format starts like this
                            # "2013-11-25 06:05:48.472Z name message...."
                            # But there may be lines with Tracebacks that does
                            # not start like this
                            date_str, time_str, rest = log_msg.split(" ", 2)
                            #Timestamp format is "06:05:48.472Z"
                            hr_str, min_str, sec_str = time_str.split(":")
                            #millis = int(sec_str.split(".")[1].strip("Z"))
                            found = True
                    if found:
                        Aqf.passed("Successfully extracted millisecond portion"
                                   " from log file %s (log time=%s)" %
                                   (which, time_str))
                    else:
                        ok = False
                        stext = "Could not extract milliseconds portion from" \
                                " log file %s" % (which)
                except Exception, err:
                    ok = False
                    stext = "Exception while extracting milliseconds portion" \
                            " of %s timestamp (%s)" % (which, err)

            if ok:
                Aqf.passed("Logfile ok: %s" % which)
            else:
                Aqf.failed(stext)
        Aqf.end()

    @aqf_vr("VR.CM.AUTO.L.35")
    def test_warning_logs(self):
        if settings.system == "kat7":
            sensor_name = "sim.sensors.asc_wind_speed"
            grep_for = "Activating wind stow"
            nominal_value = 9.0
            trigger_value = 15.2
        elif settings.system == "mkat_rts":
            sensor_name = "sim.sensors.asc_wind_speed"
            grep_for = "Activating wind stow"
            nominal_value = 9.0
            trigger_value = 17.0
        elif settings.system == "mkat":
            sensor_name = "sim.sensors.wind_wind_speed"
            grep_for = "Activating wind stow"
            nominal_value = 9.0
            trigger_value = 17.0

        ant = self.cam.ants[-1]
        Aqf.hop("Selecting receptor %s for test" % ant.name)
        Aqf.hop("Ensure nominal setting on sensor %s" % sensor_name)
        aqf_sensor = Aqf.sensor(sensor_name)
        aqf_sensor.set(nominal_value)
        aqf_sensor.wait_until_status(is_in=['nominal'])
        run_time = datetime.utcnow()
        Aqf.step("Induce a warning by activating gust wind stow at %s" % run_time)
        aqf_sensor.set(trigger_value)
        aqf_sensor.wait_until_status(is_in=['error'])
        #Wait for the wind gust alarm to fire
        aqf_alarm_sensor = Aqf.sensor("kat.sensors.kataware_alarm_ANC_Wind_Gust")
        aqf_alarm_sensor.wait_until_status(is_in=['error'])
        Aqf.wait(3)  # Wait for sensors to register the state and for
                     # events to appear in the log.
        Aqf.progress("Wait for windstow active on the receptor")
        Aqf.sensor("cam.%s.sensor.windstow_active" % ant.name).wait_until(True, sleep=1, counter=10)
        Aqf.step("Check that the warning is stored in the relevant receptor log file")
        log_name = 'kat.%s' % ant.name
        log_found = False
        grep = [log_name, 'WARNING', grep_for]
        for i in range(0,3):
            Aqf.wait(3, "Wait for log processing")
            log_found = utils.check_end_of_log_for(self, log_name, grep, aftertime=run_time, lines=50)
            if log_found:
                break
        Aqf.is_true(log_found,
                "Verify that the warning was recorded in the receptor log file, grep (%s)" % grep)
        Aqf.hop("Return trigger sensor to nominal")
        aqf_sensor.set(nominal_value)
        # There is a 2min delay before the wind stow is released - wait for it
        Aqf.progress("Wait for windstow release")
        Aqf.sensor("cam.%s.sensor.windstow_active" % ant.name).wait_until(False, sleep=1, counter=150)
        # Also check sys.interlock_state is back to NONE
        Aqf.sensor('cam.sensors.sys_interlock_state').wait_until('NONE', sleep=1, counter=10)
        Aqf.end()

    @aqf_vr("VR.CM.AUTO.L.35")
    def test_error_logs(self):
        run_time = datetime.utcnow()
        Aqf.step("Induce an error, by setting the antenna mode to "
                 "an invalid mode")
        ant = self.cam.ants[-1]
        ant.req.mode("this_is_a_test")
        Aqf.wait(3)  # Wait for sensors to to register the state and for
                     # events to appear in the log.
        Aqf.step("Check that the error is stored in the system logs")
        log_name = 'kat.%s' % ant.name
        grep = [log_name, 'ERROR', 'this_is_a_test']
        log_found = False
        for i in range(0,3):
            Aqf.wait(3, "Wait for log processing")
            log_found = utils.check_end_of_log_for(self, log_name, grep, aftertime=run_time, lines=50)
            if log_found:
                break
        Aqf.is_true(log_found,
                "Verify that the error was stored in the receptor log file, grep (%s)" % grep)
        Aqf.end()
    
    @system('kat7', 'mkat', all=False)
    @aqf_vr("VR.CM.AUTO.L.34")
    def test_cam_generates_alarm_logs(self):
        """Test that CAM generates alarm logs."""
        if self.system == "mkat_rts":
            Aqf.skipped("System_Fire alarm is deactivated for mkat_rts")
            return

        Aqf.step("Verify that the System_Fire alarm is not active before the test starts")
        if not utils.check_alarm_severity(self, "System_Fire", "nominal"):
            Aqf.skipped("System_Fire alarm is not nominal. Aborting test.")
            return

        alarm_name = "System_Fire"
        expected_priority = "critical"

        trigger_utctime = datetime.utcnow()
        specifics.simulate_fire(self)
        Aqf.step("Check that the fire alarm has been raised")
        utils.check_alarm_severity(self, alarm_name, expected_priority)

        found, utctime = utils.check_alarm_logged(self, alarm_name, expected_priority, aftertime=trigger_utctime, lines=50)
        Aqf.step("Checking alarms log: %s at %s - after %s" % (found, utctime, trigger_utctime))
        Aqf.is_true(found, "Alarms log was generated for %s at priority %s" % (alarm_name, expected_priority))
        
        Aqf.hop("Reset the System_Fire alarm")
        specifics.reset_fire(self)

        Aqf.step("Verify that System_Fire alarm returns to nominal")
        Aqf.sensor('cam.sensors.kataware_alarm_System_Fire').wait_until_status(is_in=['nominal'])

        Aqf.step("Clear the System_Fire alarm")
        self.cam.kataware.req.alarm_clear("System_Fire")
        Aqf.end()
