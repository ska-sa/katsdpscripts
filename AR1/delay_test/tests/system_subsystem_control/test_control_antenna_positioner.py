##############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import random

from nosekatreport import (system, aqf_vr, site_acceptance)
from tests import settings, Aqf, AqfTestCase


class TestApControl(AqfTestCase):
    """Tests that antenna positioner can be controlled over ethernet using
    katcp protocol."""

    def setUp(self):
        Aqf.step("Setup")

        #Get a connected proxy randomly.
        self.ant_proxy = random.choice(self.cam.ants)
        proxy_name = self.ant_proxy.name

        self.mode_sensor = "cam.%s.sensor.mode" % proxy_name
        if "mkat" in settings.system:
            self.ap_mode_sensor = "cam.%s.sensor.ap_mode" % proxy_name

            self.ap_indexer_sensor = "cam.%s.sensor.ap_indexer_position" % proxy_name

            self.ap_actl_azim = "cam.%s.sensor.ap_actual_azim" % proxy_name
            self.ap_actl_elev = "cam.%s.sensor.ap_actual_elev" % proxy_name
            self.ap_req_azim = "cam.%s.sensor.ap_requested_azim" % proxy_name
            self.ap_req_elev = "cam.%s.sensor.ap_requested_elev" % proxy_name

            self.ap_state = Aqf.sensor("cam.%s.sensor.ap_state" % proxy_name).get()
            self.device = "ap"
            self.az_offset = 0.0
        else: #KAT7

            self.ap_actl_azim = "cam.%s.sensor.antenna_acs_actual_azim" % proxy_name
            self.ap_actl_elev = "cam.%s.sensor.antenna_acs_actual_elev" % proxy_name
            self.ap_req_azim = "cam.%s.sensor.antenna_acs_request_azim" % proxy_name
            self.ap_req_elev = "cam.%s.sensor.antenna_acs_request_elev" % proxy_name
            self.ap_state = Aqf.sensor("cam.%s.sensor.antenna_state" % proxy_name).get()
            self.device = "antenna"
            #KAT7 azimuth has a wierd 45deg offset
            self.az_offset = 45.0

        self.initial_az_value = Aqf.sensor("cam.%s.sensor.pos_actual_pointm_azim" % proxy_name).get()
        self.initial_el_value = Aqf.sensor("cam.%s.sensor.pos_actual_pointm_elev" % proxy_name).get()
        ##self.initial_el_value = Aqf.sensor(self.el_sensor).get()
        self.addCleanup(self.restore_ap)

    def restore_ap(self):
        Aqf.hop("Return antenna positioner to known position: 70,70")
        self.ant_proxy.req.mode("STOP")
        target_azim = 70.0
        target_elev = 70.0
        self.ant_proxy.req.target_azel(target_azim,target_elev)
        self.ant_proxy.req.mode("POINT")
        # Return the ap to its initial position then stop
        Aqf.sensor(self.ap_actl_elev).wait_until_approximate(target_elev, sleep=1, counter=240)
        Aqf.sensor(self.ap_actl_azim).wait_until_approximate(target_azim-self.az_offset, sleep=1, counter=240)
        self.ant_proxy.req.mode("STOP")


    def check_if_req_ok(self,obj):
        """Helper function that check if request was successfully sent to device,
            using katcp protocol."""

        #Check if ap has requests.
        all_requests = self.ant_proxy.req.keys()
        ap_request_keys = [k for k in all_requests if k.startswith('%s_' % self.device)]
        Aqf.is_true(len(ap_request_keys) > 0,
         "Verify that antenna positioner has requests.")

        #Check if ap has sensors.
        ap_sensors=self.ant_proxy.req.sensor_list()
        msg=ap_sensors.messages[1:]
        Aqf.is_true(len([msg[i].arguments[0] for i in range(len(msg))
            if msg[i].arguments[0].startswith("ap")]) > 0 ,
                "Verify that antenna positioner has sensors.")

        #check if ap synced
        Aqf.equals(self.ap_state, "synced", "Verify that antenna positioner is synced.")

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify that request '%s' is successfully sent to antenna positioner."
                % msg.name )
        else:
            #log to the user
            Aqf.progress("Failed to send request '%s' to antenna positioner." % msg.name)
        Aqf.equals(msg.arguments[0], 'ok', ("Request '%s' is successfully sent to"
            " antenna positioner using katcp protocol.") % msg.name)

    @site_acceptance
    @aqf_vr('VR.CM.AUTO.C.22')
    @system("all")
    def test_antenna_positioner_control(self):
        """Test that antenna positioner can be controlled."""

        Aqf.step("Put %s antenna positioner into STOP" % self.ant_proxy.name)
        self.ant_proxy.req.mode("STOP")
        result = Aqf.sensor(self.mode_sensor).wait_until("STOP", sleep=1, counter=5)

        # Don't continue test if not in STOP
        if not result:
            Aqf.failed("Cannot continue this test. %s mode is not STOP" % self.ant_proxy.name)
            Aqf.end()
            return

        az = 70
        el = 58

        Aqf.step("Verify that antenna positioner can be moved to specified az, el(%s, %s)."
            % (az, el))
        Aqf.progress("Requesting antenna positioner to go to az, el %s, %s" % (az, el))

        Aqf.step("Requesting target_azel %s, %s" % (az, el))
        self.ant_proxy.req.target_azel(az, el)
        Aqf.step("Requesting mode POINT")
        self.ant_proxy.req.mode("POINT")
        result = Aqf.sensor(self.mode_sensor).wait_until("POINT", sleep=1, counter=5)
        Aqf.progress("Verify that the device 'requested' sensors show the right values")
        Aqf.sensor(self.ap_req_azim).wait_until_approximate(az-self.az_offset)
        Aqf.sensor(self.ap_req_elev).wait_until_approximate(el)
        requested_az = Aqf.sensor(self.ap_req_azim).get()
        requested_el = Aqf.sensor(self.ap_req_elev).get()
        Aqf.progress("Verify that the device 'actual' sensors approach the same values")
        Aqf.sensor(self.ap_actl_azim).wait_until_approximate(requested_az, 0.1)
        Aqf.sensor(self.ap_actl_elev).wait_until_approximate(requested_el, 0.1)
        actual_az = Aqf.sensor(self.ap_actl_azim).get()
        actual_el = Aqf.sensor(self.ap_actl_elev).get()
        Aqf.progress("antenna positioner is now at actual pos (az, el) = "
                    "(%0.1f, %0.1f): error = (%0.02f, %0.02f)"
             % (actual_az,actual_el, actual_az - az, actual_el - el))

        Aqf.step("Verify that antenna positioner can be stowed.")
        self.ant_proxy.req.mode("STOP")
        result = Aqf.sensor(self.mode_sensor).wait_until("STOP", sleep=1, counter=5)
        self.ant_proxy.req.mode("STOW")
        result = Aqf.sensor(self.mode_sensor).wait_until("STOW", sleep=1, counter=120)
        #self.ant_proxy.req.mode("POINT")

        #Aqf.step("Wait for stow position (90 ).")
        #Aqf.sensor(self.ap_actl_elev).wait_until_approximate(90,0.1)
        actual_az = Aqf.sensor(self.ap_actl_azim).get()
        actual_el = Aqf.sensor(self.ap_actl_elev).get()
        Aqf.progress("Antenna positioner is now at (az, el) = (%0.1f, %0.1f)"
             % (actual_az,actual_el))

        Aqf.step("Verify that antenna positioner can be controlled"
            " using katcp protocol.")
        self.check_if_req_ok(self.ant_proxy.req.mode("STOP"))
        result = Aqf.sensor(self.mode_sensor).wait_until("STOP", sleep=1, counter=5)
        Aqf.end()

    @site_acceptance
    @aqf_vr('VR.CM.AUTO.C.22')
    @system('mkat_rts','mkat')
    def test_indexer_position(self):
        """ Test that the AP indexer can be controlled."""
        Aqf.step("Test that the AP indexer can be controlled")
        self.ant_proxy.req.mode("STOP")
        Aqf.sensor(self.mode_sensor).wait_until("STOP", sleep=1)
        indexer_positions = ['x','l','u','s']
        initial_position = self.ant_proxy.sensor.ap_indexer_position.get_value()
        Aqf.progress("Initial indexer position is '%s'" % initial_position)
        positions_to_test = indexer_positions
        positions_to_test.remove(initial_position)
        for position in positions_to_test:
            Aqf.progress("Setting indexer position to '%s'" % (position))
        self.ant_proxy.req.ap_set_indexer_position(position)
        Aqf.sensor(self.ap_indexer_sensor).wait_until(position)
        # Now put it back the way we found it
        Aqf.progress("Returning position to '%s'" % (initial_position))
        self.ant_proxy.req.ap_set_indexer_position(initial_position)
        Aqf.sensor(self.ap_indexer_sensor).wait_until(initial_position)
        Aqf.end()


    @site_acceptance
    @aqf_vr('VR.CM.AUTO.C.22')
    @system('mkat_rts','mkat')
    def test_ap_modes(self):
        """ Test that we can put the AP into various modes, independent of the proxy """
        Aqf.step("Test that we can put the AP into maintenance mode")
        ap_mode_sensor = Aqf.sensor(self.ap_mode_sensor)

        self.ant_proxy.req.target_azel(60.0, 60.0)
        self.ant_proxy.req.mode("POINT")
        Aqf.wait(3, "Let antenna move for 3 seconds")

        Aqf.step("Stow the AP")
        self.ant_proxy.req.ap_stow()
        ap_mode_sensor.wait_until('stowing')
        ap_mode_sensor.wait_until('stowed')

        Aqf.step("Stop the AP")
        self.ant_proxy.req.ap_stop()
        ap_mode_sensor.wait_until('stop')

        Aqf.step("Set AP maintenance mode")
        self.ant_proxy.req.ap_maintenance()
        ap_mode_sensor.wait_until('going-to-maintenance')
        ap_mode_sensor.wait_until('maintenance')

        #Reset
        Aqf.step("Reset AP mode to stop")
        self.ant_proxy.req.ap_stop()
        ap_mode_sensor.wait_until('stop')

        Aqf.end()


