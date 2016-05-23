###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from nosekatreport import system, aqf_vr
from datetime import datetime

from tests import settings, Aqf, AqfTestCase
from tests.utils import check_and_get_utctime_of_log_for

@system('all')
class TestCommandTiming(AqfTestCase):
    """Test that CAM detects when backup link is in use and act accordingly"""

    def setUp(self):
        if settings.system == "kat7":
            self.wind_sensor_name = "sim.sensors.asc_wind_speed"
            self.nominal_value = 9.0
            self.trigger_value = 17.0
        elif settings.system == "mkat_rts":
            self.wind_sensor_name = "sim.sensors.asc_wind_speed"
            self.nominal_value = 9.0
            self.trigger_value = 17.0
        elif settings.system == "mkat":
            self.wind_sensor_name = "sim.sensors.wind_wind_speed"
            self.nominal_value = 9.0
            self.trigger_value = 17.0

        #Use the first ant for tests
        self.ant = self.cam.ants[-1]
        self.ant_id = self.ant.name

    @aqf_vr("VR.CM.AUTO.CO.42")
    def test_cam_command_executing_time(self):
        """Test that the CAM executes a command within 1s"""

        aqf_sensor = Aqf.sensor(self.wind_sensor_name)
        aqf_sensor.set(self.nominal_value)
        aqf_sensor.wait_until_status(is_in=['nominal'])

        Aqf.hop("Enable device message logging for %s" % self.ant_id)
        self.ant.req.enable_katcpmsgs_devices_logging(1)

        trigger_utctime = datetime.utcnow()
        Aqf.step("Induce an action by activating gust wind stow: using %s" % self.wind_sensor_name)
        aqf_sensor.set(self.trigger_value)
        aqf_sensor.wait_until_status(is_in=['error'])
        #Wait for the wind gust alarm to fire
        aqf_alarm_sensor = Aqf.sensor("kat.sensors.kataware_alarm_ANC_Wind_Gust")
        aqf_alarm_sensor.wait_until_status(is_in=['error'])

        Aqf.wait(3)  # Wait for action to be taken and logs to land in files

        Aqf.step("Find the time katsyscontroller reacted to windstow trigger")
        #2014-11-03 14:33:48.606Z activity INFO alarm_event(controller.py:347) Alarm event ANC_Wind_Gust set
        grep_for = ['activity', 'Alarm event ANC_Wind_Gust set']
        found_start, start_utctime = check_and_get_utctime_of_log_for(self, 'activity', grep_for, trigger_utctime, lines = 30)

        #2014-11-06 11:54:28.24Z kat.m011 INFO set_mode(mkat_receptor_proxy.py:935) Setting mode to 'STOW'
        grep_for = ['kat.%s' % self.ant_id, "set_mode", "Setting mode to 'STOW'"]
        found_proxy, proxy_utctime = check_and_get_utctime_of_log_for(self, 'kat.%s' % self.ant_id, grep_for, trigger_utctime, lines = 30)

        #2014-11-07 07:58:40.210Z katcpmsgs INFO log_device_request(proxy_base.py:483) Proxy m011 to device ap request ?stow[152]
        if "kat7" in self.cam.katconfig.site:
            #KAT7 stow request to ASC
            grep_for = ['device_request', 'Proxy %s to device' % self.ant_id, '?mode-remote stow']
        else:
            #MeerKAT stow request to AP
            grep_for = ['device_request', 'Proxy %s to device' % self.ant_id, '?stow']
        found_send, send_utctime = check_and_get_utctime_of_log_for(self, 'katcpmsgs', grep_for, trigger_utctime, lines = 2000)

        if not found_send or not found_start:
            reaction_time = 9999 #Unkown
        else:
            reaction_utctime = send_utctime - start_utctime
            reaction_time = reaction_utctime.total_seconds()

        Aqf.hop("Trigger UTC time: %s " % (trigger_utctime))
        Aqf.hop("Syscontroller UTC time: %s (%s)" % (start_utctime, found_start))
        Aqf.hop("Proxy UTC time: %s (%s)" % (proxy_utctime, found_proxy))
        Aqf.hop("Send to antenna UTC time: %s (%s)" % (send_utctime, found_send))
        Aqf.hop("===Reaction time: %s(s) (%s)" % (reaction_time, found_start and found_send))

        Aqf.is_true(reaction_time < 60, "Check that reaction time (%s) is less than 60s" % reaction_time)

        Aqf.hop("Switch off katcp device message logging for %s" % self.ant_id)
        self.ant.req.enable_katcpmsgs_devices_logging(0)
        Aqf.hop("Test cleanup")
        Aqf.hop("Reset the wind speed: %s" % self.wind_sensor_name)
        aqf_sensor.set(self.nominal_value)
        aqf_sensor.wait_until_status(is_in=['nominal'])

        Aqf.hop("Waiting for antenna %s to UNSTOW" % self.ant_id)
        aqf_sys_interlock = Aqf.sensor("kat.sensors.sys_interlock_state")
        aqf_sys_interlock.wait_until('NONE', sleep=10, counter=14)
        Aqf.end()

    @aqf_vr("VR.CM.DEMO.C.47")
    def test_cam_displays_activity_with_subsystem(self):
        """Demonstrates that the CAM displays activity with selected subsystem."""

        Aqf.step("From KatGUI open 'CAM Components' display.")
        Aqf.checkbox("Enable KATCP Logging Devices for the ANC"
                 " by using the slider on the ANC row.")
        Aqf.step("Click 'VIEW KATCP MESSAGES LOG'.")
        Aqf.step("Verify that the KATCP traffic between devices (BMS, VDS, wind, weather, ganglia and TFR)"
                 " to ANC are displayed")
        Aqf.step("Close the message log display and disable KATCP Logging Devices for the ANC")
        Aqf.checkbox("Verify that the traffic on the KATCP MESSAGES LOG display stopped")
        Aqf.checkbox("Close the KATCP MESSAGES LOG window")

        Aqf.end()


    @aqf_vr("VR.CM.AUTO.CO.60")
    def test_cam_logs_activity_with_subsystem(self):
        """Verifies that CAM can log activity with selected subsystem."""

        LIMIT_IN_SECONDS = 65

        Aqf.step('Disabling logging on all devices')
        sensors = self.cam.list_sensors('katcpmsgs')
        for sensor in sensors:
            device = getattr(self.cam, sensor[0].parent_name ,None)
            if device:
                device.req.enable_katcpmsgs_devices_logging(0)
                device.req.enable_katcpmsgs_proxy_logging(0)

        Aqf.hop('Making sure all logging was turned off')
        for sensor in sensors:
            device = getattr(self.cam.sensor, sensor[2] ,None)
            if device.get_value():
                Aqf.hop ('{} not off'.format(sensor[0].parent_name))


        Aqf.step('Turning logging on for %s' % self.ant_id)
        self.ant.req.enable_katcpmsgs_devices_logging(1)

        trigger_utctime = datetime.utcnow()
        Aqf.hop("Triggered logging at UTC time: %s " % (trigger_utctime))
        Aqf.wait(LIMIT_IN_SECONDS)  # Wait for action to be taken and logs to land in files

        Aqf.step("Searching for watchdog request issued by %s in katcpmsgs after %s" % (self.ant_id, trigger_utctime))
        #2015-07-30 13:28:54.775Z katcpmsgs INFO log_device_request(proxy_base.py:483) Proxy m022 to device rsc request ?watchdog
        grep_for = ['Proxy %s to device' % self.ant_id, 'request', '?watchdog' ]
        found_start, start_utctime = check_and_get_utctime_of_log_for(
            self, 'katcpmsgs', grep_for, trigger_utctime, lines = 10000)
        Aqf.is_true(found_start, "Verify the request was found (at %s)" % (start_utctime))


        Aqf.step("Searching for watchdog reply to %s in katcpmsgs after %s" % (self.ant_id, trigger_utctime))
        #2015-07-30 13:28:54.776Z katcpmsgs INFO log_device_reply(proxy_base.py:488) Proxy m022 from device rsc reply !watchdog ok
        grep_for = ['Proxy %s from device' % self.ant_id, 'reply', '!watchdog']
        found_start, reply_utctime = check_and_get_utctime_of_log_for(
            self, 'katcpmsgs', grep_for, trigger_utctime, lines = 10000)
        Aqf.is_true(found_start, "Verify the reply was found (at %s)" % (reply_utctime))

        self.ant.req.enable_katcpmsgs_devices_logging(0)
        Aqf.wait(2)  # Wait for action to be taken and logs to land in files

        #2015-07-30 13:28:59.339Z katcpmsgs WARNING request_enable_katcpmsgs_devices_logging(proxy_base.py:548) Proxy m022 stop katcpmsgs devices logging
        grep_for = ['%s stop katcpmsgs devices' % self.ant_id]
        found_start, start_utctime = check_and_get_utctime_of_log_for(
            self, 'katcpmsgs', grep_for, trigger_utctime, lines = 1)

        Aqf.is_true(found_start, "katcpmsgs stopped on device %s" % self.ant_id)

        Aqf.end()
