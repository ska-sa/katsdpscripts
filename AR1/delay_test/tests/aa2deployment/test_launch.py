###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import tests.utils
import katconf

from tests import fixtures, Aqf, AqfTestCase
from nosekatreport import (aqf_vr, system, site_acceptance)

from katcorelib.katcp_client import KATClient


@system('all')
class TestLaunch(AqfTestCase):

    """
    1. Test if all expected components have been launched and are running.
    2. Look at connected status of cam.status() or cam.get_status()
    3. Look at prelaunched processes (started in kat-start, not launched through
       nodemanagers)
    4. Look at xxx.running and xxx.ok sensors on nodemanagers
    """

    def setUp(self):
        pass

    @site_acceptance
    @aqf_vr('CAM_LAUNCH_cam_connected')
    def test_kat_connected(self):
        """Test that KATClient objects in cam.connected_objects are connected.
        """
        Aqf.step("Check the KATClient objects in cam.connected_objects are actually connected.")
        ok = True
        smsg = []
        for name, katclient in self.cam.connected_objects.items():
            if isinstance(katclient, KATClient) and not katclient.is_connected():
                ok = False
                smsg.append('KATClient %r not connected' % name)
        if ok:
            Aqf.passed("All KATClients are connected")
        else:
            Aqf.failed("The following KATClient are not "
                        "connected: %s" % ",".join(smsg) )
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_LAUNCH_prelaunch')
    def test_kat_prelaunch(self):
        """Test that all prelaunch processes started by kat-launcher before syscontroller starts the
           rest of the system are running.

        """
        #NOTE: This test can be improved by reading the pre-launch processes from katconfig
        Aqf.step("Check CAM prelaunch processes are running")
        ok, msgs = True, []
        processes = ["kat-confserver", "kat-nodeman"]
        for process in processes:
            res, s_msg = tests.utils.check_process_in_ps(process)
            if not res:
                ok = False
                msgs.append('%s: %s' % (process, s_msg))
        if ok:
            Aqf.passed("All pre-launch processes are running: %s" % str(processes))
        else:
            Aqf.failed("The following pre-launch processes not "
                        "found: %s" % ",".join(msgs))
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_LAUNCH_headnode_processes_running')
    def test_kat_headnode_launch(self):
        """Test that kat system headnode specific processes are running.
        """
        #NOTE: This test can be improved by reading the processes from katconfig
        Aqf.step("Check headnode specific processes are running")
        ok, msgs = True, []
        processes = ["kat-aware",
                     "kat-monitor.py",
                     "kat-syscontroller",
                     "kat-scheduler",
                     "kat-pool",
                     ]
        for process in processes:
            res, s_msg = tests.utils.check_process_in_ps(process)
            if not res:
                ok = False
                msgs.append('%s: %s' % (process, s_msg))

        if ok:
            Aqf.passed("All headnode processes are running: %s" % str(processes))
        else:
            Aqf.failed("The following headnode processes  "
                         "were not found: "+",".join(msgs))
        Aqf.end()

    @site_acceptance
    @aqf_vr('CAM_LAUNCH_nodeman_processes_running')
    def test_kat_running(self):
        """Test that kat processes registered with each nodemanager are running.
        """
        Aqf.step("Check kat processes registered with each nodemanager are running")
        ok = True
        msgs = []

        conffile = self.cam.system
        confsvr = katconf.get_config()
        cp = confsvr.resource_config(conffile)
        nodemgrs = [x for x,y in cp.items("katconn:clients") if x[:3] == "nm_"]

        sens_filter = "running"
        for nodeman in nodemgrs:
            comp = getattr(self.cam, nodeman)
            if not comp.is_connected():
                ok = False
                msgs.append("Nodeman:%s" % nodeman)
            else:
                #Now get the tuple to process - refresh sensor first
                sens_list = comp.list_sensors(sens_filter, refresh=True)
		self.assertTrue(len(sens_list) > 0,
                            msg='No process sensors on nodemanager %r' % nodeman)
                #Exclude the cbfdata as it only starts running after an observation
                # sens_update is SensorResultTuple
                fails = [sens_update.python_identifier for sens_update in sens_list if (not bool(sens_update.reading.value)) and ('cbfdata' not in sens_update.python_identifier)]
                if len(fails) > 0:
                    ok = False
                    msgs.extend(fails)

        if ok:
            Aqf.passed("All processes on all nodemanagers are running")
        else:
            Aqf.failed("The following registered processes are not "
                        "running: "+",".join(msgs))
        Aqf.end()

