###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import random

from nosekatreport import system, aqf_vr
from tests import Aqf, AqfTestCase

@system('mkat')
class TestCamInfrastructureSensordata(AqfTestCase):
    """Test that cam exposes all infrastructure sensor data to the bms"""

    def setUp(self):
        Aqf.step("Setup")

        #Get a connected proxy randomly.
        self.ant_proxy = random.choice(self.cam.ants)
        self.proxy_name = self.ant_proxy.name

        self.ap_connected_sensor = "cam.%s.sensor.ap_connected" % self.proxy_name
        self.rsc_connected_sensor = "cam.%s.sensor.rsc_connected" % self.proxy_name
        self.bms_power_sensor = "cam.anc.sensor.bms_%s_power_status" % self.proxy_name
        self.proxy_number = self.proxy_name.replace("m", "")
        # The ap, rsc and rscadmin devices run in the pedestal
        # But the simulator stops the rscadmin interface together with rsc
        self.ap_stopped = getattr(self.cam,
            self.proxy_name).sensor.ap_connected.get_value()
        self.rsc_stopped = getattr(self.cam,
            self.proxy_name).sensor.rsc_connected.get_value()
        # Determine which node the simulators run on
        x = self.cam.list_sensors("ap%s_running" % self.proxy_number)
        # x[0][0] is like nm_sim_ap000_running
        node_name = x[0].python_identifier.replace("_ap%s_running" % self.proxy_number, "")
        self.the_sim_node = getattr(self.cam, node_name)

        if self.ap_stopped:
            self.the_sim_node.req.start("ap%s" % self.proxy_number)
        if self.rsc_stopped:
            self.the_sim_node.req.start("rsc%s" % self.proxy_number)

    def tearDown(self):
        # The ap and rsc devices run in the pedestal
        # But the simulator stops the rscadmin interface together with rsc
        if not self.rsc_stopped:
            self.the_sim_node.req.start("rsc%s" % self.proxy_number)
        if not self.ap_stopped:
            self.the_sim_node.req.start("ap%s" % self.proxy_number)

    @aqf_vr("VR.CM.AUTO.CO.39")
    @system("mkat")
    def test_cam_receptor_power_status_to_bms(self):
        """Test that the cam interface exposes sensor data to the bms """

        Aqf.step("Check receptor power status sensor value for configured"
            "and unconfigured receptors.")
        available_ants = [ant.name for ant in self.cam.ants]
        all_ants = ["m%03d" % i for i in range(64)]
        unavailable_ants = [a for a in all_ants if a not in available_ants]

        for s in dir(self.cam.anc.sensor):
            for avail_ant in available_ants:
                if s.endswith('_power_status') and s.startswith('bms_%s'%avail_ant):
                    value = getattr(self.cam.anc.sensor, s).get_value()
                    Aqf.is_true( value != 'unavailable',
                        "For configured receptors,"
                            "verify if %s value is either on or off." % s)
            for unavail_ant in unavailable_ants:
                if s.endswith('_power_status') and s.startswith('bms_%s'%unavail_ant):
                    value = getattr(self.cam.anc.sensor,s).get_value()
                    Aqf.equals("unavailable", value,
                        "For unconfigured receptors, verify if"
                            " %s value is unavailable." % s)

        Aqf.step("Stop rsc%s" % self.proxy_number)
        self.the_sim_node.req.stop("rsc%s" % self.proxy_number)
        Aqf.progress("Stopping rsc%s" % self.proxy_number)

        Aqf.step("Verify that rsc%s is stopped." % self.proxy_number)
        Aqf.sensor(self.rsc_connected_sensor).wait_until(False, sleep=1)

        Aqf.step("Verify that the power is 'on' for receptor %s" % self.proxy_name)
        Aqf.sensor(self.bms_power_sensor).wait_until("on", sleep=1)

        Aqf.step("Stop antenna positioner ap%s" % self.proxy_number)
        #self.the_sim_node.req.stop("ap%s" % self.proxy_number)
        self.the_sim_node.req.kill("ap%s" % self.proxy_number)
        Aqf.progress("Stopping antenna positioner ap%s" % self.proxy_number)

        Aqf.step("Verify that antenna positioner ap%s is stopped." % self.proxy_number)
        Aqf.sensor(self.ap_connected_sensor).wait_until(False, sleep = 1)

        Aqf.step("Verify that the power is now off for receptor %s" %
            self.proxy_name)
        Aqf.sensor(self.bms_power_sensor).wait_until("off", sleep=1)

        Aqf.step("Start rsc%s for receptor %s after it was stopped" %
            (self.proxy_number, self.proxy_name))
        self.the_sim_node.req.start("rsc%s" % self.proxy_number)
        Aqf.progress("Starting rsc%s for receptor %s" %
            (self.proxy_number, self.proxy_name))

        Aqf.step("Verify that rsc%s has started." % self.proxy_number)
        Aqf.sensor(self.rsc_connected_sensor).wait_until(True, sleep=1)

        Aqf.step("Verify that the power is now on for receptor %s " %
            self.proxy_name)
        Aqf.sensor(self.bms_power_sensor).wait_until("on", sleep=1)

        Aqf.step("Start ap%s after it was stopped" % self.proxy_number)
        self.the_sim_node.req.start("ap%s" % self.proxy_number)
        Aqf.progress("Starting ap for receptor %s" % self.proxy_name)

        Aqf.step("Verify that ap%s has started" % self.proxy_number)
        Aqf.sensor(self.ap_connected_sensor).wait_until(True, sleep=1)

        Aqf.step("verify that the power_status is reported to be back on for receptor %s " % self.proxy_name)
        Aqf.sensor(self.bms_power_sensor).wait_until("on", sleep=1)

        Aqf.end()
