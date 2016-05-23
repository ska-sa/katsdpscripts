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
import katconf

import toredis
import omnijson as json
from tornado import gen
from tornado.web import Application
from sockjs.tornado import SockJSRouter
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.websocket import WebSocketHandler, websocket_connect

from tests import Aqf, AqfTestCase
from tests import settings, fixtures

from nosekatreport import system, aqf_vr

from katportal.test import redis_server_helper
from katportal.common import KATSockJSConnection
from katportal.test.jsonrpc_helper import JSONRPCRequest, JSONRPCResponse


class ILSTestConnection(KATSockJSConnection):

    @gen.coroutine
    def subscribe(self, msg):
        pass

    @gen.coroutine
    def set_sampling_strategy(self, msg):
        pass

@system('all')
class TestILS(AqfTestCase, AsyncHTTPTestCase):

    REDIS_PORT = redis_server_helper.REDIS_TEST_SERVER_PORT
    LOGGER_NAME = 'test_base_sockjs.TestILS'

    
    def setUp(self):
        Aqf.step("Setup")
        super(TestILS, self).setUp()
        fixtures.sim = self.sim
        fixtures.cam = self.cam
#        redis_server_helper.start_redis_tst_server()
	self.proxy = random.choice(self.cam.ants).name
        self.device = getattr(self.cam, '{}'.format(self.proxy))
        self.mode = self.device.sensor.mode.get_value()

        
    def tearDown(self):
        Aqf.step("Return system to initial state.")
        redis_server_helper.stop_redis_tst_server()

    def get_app(self):
        ILSTestRouter = SockJSRouter(
            ILSTestConnection, '/ils',
            dict(logger_name=self.LOGGER_NAME,
                 redis_server='localhost',
                 redis_port=self.REDIS_PORT),
            io_loop=self.io_loop
        )
        self.application = Application(
            ILSTestRouter.urls
        )
        self.logger = logging.getLogger(self.LOGGER_NAME)
        return self.application

    @aqf_vr('VR.CM.AUTO.CO.41')
    @gen_test()
    def test_cam_report_to_ils(self):
        """VR.CM.AUTO.CO.41: Test CAM data reporting to ILS server"""

	Aqf.progress("Getting initial sensor value for sensor {}_mode".format(self.proxy))
        Aqf.progress("The initial value for '{}' is '{}'".format(self.proxy, self.mode))
        Aqf.step("Setting sensor '{}' value to POINT on cam system".format(self.proxy))

        self.device.req.target_azel(10,30)
        self.device.req.mode('POINT')

        # Reading ap_mode value from the ILS server
        Aqf.step("Reading sensor value for '{}' on the ils".format(self.proxy))
        system_config_file = 'systems/{}.conf'.format(katconf.sitename())
        system_config = katconf.resource_config(system_config_file)
        katportal_ip = system_config.get('katportal', 'katportal_ip')

        ws = yield websocket_connect(
            'ws://{}:8870/ils/websocket'.format(katportal_ip), io_loop=self.io_loop)
	strat = ["{}_mode".format(self.proxy),"period 1"]
        req = JSONRPCRequest('set_sampling_strategy', strat)
        ws.write_message(req())
        response = yield ws.read_message()

	req1 = JSONRPCRequest('subscribe', ["{}_mode".format(self.proxy)])
        ws.write_message(req1())
        yield ws.read_message()
        time.sleep(0.1)
        response2 = yield ws.read_message()
        time.sleep(0.1)
        response3 = yield ws.read_message()

        a = json.loads(response3)
        b = a[u'result']
        c = b[u'msg_data']
        expected_mode = str(c[u'value'])

        Aqf.equals(
            expected_mode, 'POINT', "Verify that the value set on cam to %s "
	    "corresponds with the value received by ils" %"{}_mode".format(self.proxy)
            )

        Aqf.step("Reset sensor value to initial value recorded before testing")
	self.device.req.mode(self.mode)
	Aqf.sensor(self.device.sensor.mode).wait_until(self.mode)

        Aqf.end()


