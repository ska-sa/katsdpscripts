###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

from datetime import datetime

from nosekatreport import system, aqf_vr
from tests import utils, Aqf, AqfTestCase


@system('mkat', 'kat7')
class TestSmsChatNotifications(AqfTestCase):
    """Test that notifications device can send sms and chat
       to intended individuals and chat rooms"""

  
        
    @aqf_vr("Test_Sms_Notifications")
    def test_sms_notifications(self):
        """Test that sms notifications are sent through to the intended recipients"""
        Aqf.step("Send an sms message")
        trigger_utctime = datetime.utcnow()
        self.cam.anc.req.notifications_send_sms("27840957698","Testing sms")
        Aqf.step("Verify that the sms has been noted in activity logs")
        grep_for = ["sms sent to", "27840957698"]
        Aqf.wait(3, "Wait for sms log to land in activity log file")
        found, utctime = utils.check_activity_logged(self, grep_for, aftertime=trigger_utctime, lines=50000)
        Aqf.step("Checking activity log: %s at %s - after %s" % (found, utctime, trigger_utctime))
        Aqf.is_true(found, "Activity log was generated for sms notifications")
  
        Aqf.end()
 
    @aqf_vr("Test_Chat_Notifications")
    def test_chat_notifications(self):
        """Test that chat notifications are sent through to the intended recipients"""
        Aqf.step("Send a chat message")
        trigger_utctime = datetime.utcnow()
        self.cam.anc.req.notifications_send_room_chat("IRC","Testing chatrooms chats")
        Aqf.step("Verify that the chat message has been noted in activity logs")
        grep_for = ["chat sent to", "IRC"]
        Aqf.wait(3, "Wait for chat log to land in activity log file")
        found, utctime = utils.check_activity_logged(self, grep_for, aftertime=trigger_utctime, lines=5000)
        Aqf.step("Checking activity log: %s at %s - after %s" % (found, utctime, trigger_utctime))
        Aqf.is_true(found, "Activity log was generated for chat notifications")

        Aqf.end()
        
