###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import katconf
import tests.utils

from tests import (fixtures, Aqf, AqfTestCase, utils)
from nosekatreport import (aqf_vr, system, site_acceptance)

from katcorelib.katcp_client import KATClient
from katcp.resource import KATCPResource


@system('all')
class TestAaFirst(AqfTestCase):

    """
    Test if all katconn clients are connected, synced and exposing sensors and requests.
    """

    def setUp(self):
        pass

    @site_acceptance
    @aqf_vr('CAM_AAFIRST_kat_connected_and_synced')
    def test_aa_first_kat_connected(self):
        """Test that KATClient objects in cam.connected_objects are connected, synced and have requests and sensors.
        """

        ok = False
        loop = 0
        while loop < 10:
            #NOTE: This test intentially keeps on tryin as something is seriously wrong and tests should not continue
            #It also ensures the system is stable before we continue

            ok = True # Assume everything is ok, until proven otherwise

            Aqf.step("Check that objects in cam.connected_objects are actually connected. Loop %d" % loop)
            smsg = []
            for name, client in self.cam.connected_objects.items():
                if isinstance(client, KATClient) or isinstance(client, KATCPResource):
                    Aqf.progress("Checking %s" % name)
                    if not client.is_connected():
                        ok = False
                        smsg.append('Client %r not connected' % name)
                else:
                    Aqf.progress("Skipping %s" % name)
            if ok:
                Aqf.passed("All clients are connected")
            else:
                Aqf.progress("The following clients are not "
                            "connected: %s" % ",".join(smsg) )

            Aqf.step("Check that objects in cam.connected_objects are synced.")
            smsg = []
            for name, client in self.cam.connected_objects.items():
                if isinstance(client, KATClient) or isinstance(client, KATCPResource):
                    Aqf.progress("Checking %s" % name)
                    if not client.synced:
                        ok = False
                        smsg.append('Client %r not synced' % name)
                else:
                    Aqf.progress("Skipping %s" % name)
            if ok:
                Aqf.passed("All clients are synced")
            else:
                Aqf.progress("The following clients are not "
                            "synced: %s" % ",".join(smsg) )

            Aqf.step("Check that objects in cam.connected_objects expose sensors and requests.")
            smsg = []
            for name, client in self.cam.connected_objects.items():
                if isinstance(client, KATClient) or isinstance(client, KATCPResource):
                    Aqf.progress("Checking sensors and requests for %s" % name)
                    sens_count = len(client.sensor)
                    req_count = len(client.req)
                    Aqf.progress("   %s has %s sensors and %s requests" % (name, sens_count, req_count))
                    if sens_count < 1:
                        ok = False
                        amsg = 'Client %r does not have sensors (sens_count=%d)' % (name, sens_count)
                        smsg.append(amsg)
                    if req_count <= 1:
                        ok = False
                        amsg = 'Client %r does not have requests (req_count=%d)' % (name, req_count)
                        smsg.append(amsg)
                else:
                    Aqf.progress("Skipping %s" % name)
            if ok:
                Aqf.passed("All clients have sensors and requests")
            else:
                Aqf.progress("The following have problems: "
                            "%s" % ",".join(smsg) )

            #Now see if we should redo all of this
            if ok:
                #Now we can stop trying as everything was OK
                break
            else:
                Aqf.progress("RETRY: as some clients have not settled yet")

            Aqf.wait(60, "Waiting for 1 minute ...")
            loop += 1

        #Now also wait for kataware to start alarms processing
        Aqf.step("Wait on kataware to start alarms processing")
        #Check the last kataware alarm status
        alarms = self.cam.kataware.list_sensors("alarm_")
        first_alarm_name = alarms[0].python_identifier
        last_alarm_name = alarms[-1].python_identifier
        Aqf.step("Using alarms %s and %s" % (first_alarm_name, last_alarm_name))

        loop = 0
        while loop < 10:
            Aqf.step("Check kataware alarms processing. Loop %d" % loop)
            try:
                first_status = Aqf.sensor("cam.kataware.sensor.%s" % first_alarm_name).status()
                last_status = Aqf.sensor("cam.kataware.sensor.%s" % last_alarm_name).status()
                Aqf.progress("Alarms status: %s is %s, %s is %s" % (first_alarm_name, first_status, last_alarm_name, last_status))
                if first_status != 'unknown' and 'last_status' != 'unknown':
                    Aqf.passed("Alarms processing by kataware has started.")
                    break
                else:
                    Aqf.progress("RETRY: some alarms are still unknown")
            except Exception, err:
                Aqf.progress("Exception: %s" % err)

            Aqf.wait(60, "Waiting for 1 minute ...")
            loop += 1

        if first_status == 'unknown' or 'last_status' == 'unknown':
            ok = False
            Aqf.failed("kataware are not yet processing alarms")

        if ok:
            Aqf.passed("All KATClients has settled and is ready")
        else:
            Aqf.failed("SYSTEM DID NOT SETTLE IN TIME !!!")
            Aqf.exit()

        Aqf.end()

    @aqf_vr('CAM_ABFIRST_set_resources')
    def test_ab_first_set_resources(self):
        """Set resources not in maintenance"""
        ok = utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        
        if ok:
            Aqf.passed("Success: clear_all_subarrays_and_schedules")
        else:
            Aqf.failed("FAILURE: clear_all_subarrays_and_schedules !!!")
            Aqf.exit()

        Aqf.end()
