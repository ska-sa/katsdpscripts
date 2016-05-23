###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""
import time
from tests import (fixtures, wait_sensor, specifics, settings,
                   Aqf, AqfTestCase)

from nosekatreport import (aqf_vr, system)


@system('all')
class TestMonitor(AqfTestCase):

    """Tests monitor."""

    def setUp(self):
        pass

    def check_sensor_in_failure(self, sensor_name):
        """Checks whether the sensor status is failure.

        Parameters
        ----------
        sensor_name : string

        """
        desired_status = 'failure'
        sensor = getattr(self.cam.sensors, sensor_name)
        Aqf.step("Verify that sensor %s has status %r." %
                (sensor_name, desired_status))
        reading = sensor.get_reading()
        Aqf.is_true(reading.status in [desired_status], "Checks whether sensor %s status is %s."
            % (sensor_name, desired_status))

    def check_aggregate(self):
        """Ensure that monitor calculates the value of an aggregate sensor in
        accordance to the rule of the aggregate sensor."""
        # Set device sensor value in the error range
        specifics.simulate_fire(self)
        desired_value = False
        agg_sensor = "cam.sensors.agg_system_fire_ok"
        Aqf.step("Verify that Agg sensor %s has value %r." %
                (agg_sensor, desired_value))
        result = Aqf.sensor(agg_sensor).wait_until(desired_value, sleep=1, counter=10)

        # Reset the fire failure to nominal range
        specifics.reset_fire(self)
        desired_value = True
        Aqf.step("Verify that Agg sensor %s has value %r." %
                (agg_sensor, desired_value))
        result = Aqf.sensor(agg_sensor).wait_until(desired_value, sleep=1, counter=10)

    @system("kat7", "mkat", all=False)
    @aqf_vr('CAM_MONITOR_aggregates_recalc')
    def test_device_and_proxy_down(self):
        """Test calculation of aggregate sensor.

        In the case where a device go down.

        """
        Aqf.step("Check calculation of an aggregate sensor value on device disconnect")
        self.check_aggregate()
        # Stop device and anc proxy
        if "local" in self.cam.system:
            nm_proxy = getattr(self.cam, "nm_localhost")
            nm_sim = getattr(self.cam, "nm_localhost")
        else:
            nm_monctl = getattr(self.cam, "nm_monctl")
            #If no proxy node, use monctl node
            nm_proxy = getattr(self.cam, "nm_proxy", nm_monctl)
            #If no sim node, use proxy node
            nm_sim = getattr(self.cam, "nm_sim", nm_proxy)
        if settings.system == "mkat":
            device = "bms"
            agg_sensor_name = "agg_system_fire_ok"
        else:
            device = "asc"
            agg_sensor_name = "agg_system_fire_ok"
        nm_sim.req.stop(device)
        time.sleep(1)

        try:
            desired_value = False
            running_sens_name = "cam.{0}.sensor.{1}_running".format(nm_sim.name,device)
            result = Aqf.sensor(running_sens_name).wait_until(desired_value, sleep=1, counter=20)
            ####result = Aqf.sensor("cam.nm_proxy.sensor.anc_running").wait_until(desired_value, sleep=1, counter=20)
            # Check aggregate sensor status
            self.check_sensor_in_failure(agg_sensor_name)
            # Start device
            nm_sim.req.start(device)
            # Start anc proxy
            result = Aqf.sensor(running_sens_name).wait_until(True, sleep=1, counter=20)
            result = Aqf.sensor("cam.sensors.anc_{}_state".format(device)).wait_until("synced", sleep=1, counter=20)
            # Verify that agg has recalculated
            desired_value = True
            Aqf.step("Verify that Agg sensor %s has value %r." %
                (agg_sensor_name, desired_value))
            result = Aqf.sensor("cam.sensors.{}".format(agg_sensor_name)).wait_until(desired_value, sleep=1, counter=20)
            # Check calculation of the aggregate sensor value
            self.check_aggregate()
        finally:
            # Ensure device is started
            nm_sim.req.start(device)
        
        Aqf.end()
#
