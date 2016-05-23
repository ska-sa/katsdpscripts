###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
import katcorelib

from nosekatreport import (aqf_vr, system, site_acceptance)
from tests import Aqf, AqfTestCase


@system('all')
class TestGanglia(AqfTestCase):

    """Tests reporting of Ganglia metrics by anc proxy.
    """
    def setUp(self):
        Aqf.step("Setup")
        self.sys_nodes = katcorelib.conf.KatuilibConfig(self.cam.system
                                                        ).system_nodes

    @site_acceptance
    @aqf_vr('CAM_FMECA_kat_ganglia_sensors')
    def test_sensors_exists(self):
        """Test if ganglia sensors exist on ANC with nominal status.
        """
        #ganglia_headnode_sensors = ["_kat_varkatconfig_uptodate"]
        ganglia_sensors = ["_kat_disk_part_max_used", "_memory_mem_free"]
        #desired_status = "nominal"
        status_list = []
        Aqf.step("Check some ganglia sensors (%s) for each node" %
                 str(ganglia_sensors))
        for sensor in ganglia_sensors:
            for node in self.sys_nodes:
                Aqf.step("Check ganglia sensors for node %s" % str(node))
                sensor_name = "ganglia_%s%s" % (node, sensor)
                az_req_sensor = "cam.anc.sensor.%s" % sensor_name
                status = Aqf.sensor(az_req_sensor).status()
                Aqf.equals(status, "nominal", "Check sensor %s status (%s) "
                           "is nominal" % (sensor_name, status))
                status_list.append(status)

        Aqf.equals(all(el == 'nominal' for el in status_list), True,
                   "Expected Ganglia sensors (%s) were found in cam.anc for "
                   "each node with norminal status")
        Aqf.end()
