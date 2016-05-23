###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time
import random
import logging

from nosekatreport import (aqf_vr, system)

from tests import (fixtures, wait_sensor, wait_sensor_includes, wait_sensor_excludes, Aqf,
                   AqfTestCase, specifics, utils)

logger = logging.getLogger(__name__)

@system('all')
class TestSensorSampling(AqfTestCase):

    """Tests sensor sampling and sensor listener."""

    def setUp(self):
        pass
        #self.sim.asc.req.set_sensor_value('fire.ok', 1)

    def tearDown(self):
        #self.sim.asc.req.set_sensor_value('fire.ok', 1)
        pass

    @aqf_vr('CAM_sensor_listener')
    def test_sensor_listener(self):
        """Test that subarray pool-resources sensor listening in on katpool resources-n sensor is updated."""

        #subarray_n.sensor.pool_resources listens to katpool.sensor.pool_resources_n
        ant_name = random.choice(self.cam.ants).name
        data = specifics.get_controlled_data_proxy(self)
        res = ",".join([data,ant_name])

        # Assign resources
        Aqf.step("Assigning {} to subarray_1".format(res))
        self.cam.subarray_1.req.assign_resources(res)
        ok = wait_sensor_includes(
            self.cam, "subarray_1_pool_resources",
            data, timeout=1)
        ok = wait_sensor_includes(
            self.cam, "subarray_1_pool_resources",
            ant_name, timeout=1)
        Aqf.is_true(ok, 'Verify subarray_1 includes resources {}'
                    .format(res))
        pool_resources = self.cam.katpool.sensor.pool_resources_1.get_value()
        sub_resources = self.cam.subarray_1.sensor.pool_resources.get_value()
        Aqf.equals(sub_resources, pool_resources, 'Verify subarray_1 reflects katpool resources')

        # Unassign resources
        Aqf.step("Unassigning {} to subarray_1".format(res))
        self.cam.subarray_1.req.unassign_resources(res)
        ok = wait_sensor_excludes(
            self.cam, "subarray_1_pool_resources",
            data, timeout=1)
        ok = wait_sensor_excludes(
            self.cam, "subarray_1_pool_resources",
            ant_name, timeout=1)
        pool_resources = self.cam.katpool.sensor.pool_resources_1.get_value()
        sub_resources = self.cam.subarray_1.sensor.pool_resources.get_value()
        Aqf.equals(sub_resources, pool_resources, 'Verify subarray_1 reflects katpool resources')
                    
        # Free
        Aqf.step("Teardown - free subarray_1")
        self.cam.subarray_1.req.free_subarray()
        Aqf.end()
