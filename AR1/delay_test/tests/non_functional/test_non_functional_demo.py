###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

from nosekatreport import (aqf_vr, system)

from tests import settings, Aqf, AqfTestCase

@system('all')
class TestNonFunctional(AqfTestCase):

    """Tests for MKAT RTS non-functional requirements."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @aqf_vr('VR.CM.SITE.Z.32')
    def test_software_under_config(self):
        """Demonstrate that CAM software is under a recognised version control system."""
        Aqf.step("Open a browser, enter address: "
                 "https://ebserv.kat.ac.za/eBWeb/Framework/default.aspx "
                 "and login as SKASA/H@nsk0p")
        
        Aqf.step("Do a manual inspection of CAM software in "
                "eB (Configuration Management System)")
        Aqf.step("NOTE: KAT7 CAM Software is numbered K3000-0123V1 with title 'CAM AND SP SOFTWARE'")
        Aqf.checkbox("Verify that CAM Software is under configuration control in eB")
        
        Aqf.step("Verify that KATGUI provides version information")
        Aqf.step("From KATGUI open components display.")
        Aqf.checkbox("Verify that following information is displayed : System name (MKAT), CAM version")
        #            "3. CAM deployment date"
        #            "4. CAM svn revision")
        Aqf.checkbox("Verify that version and build state for CAM components are displayed")
        
        Aqf.end()

    @aqf_vr('VR.CM.SITE.Z.33')
    def test_software_release_notes(self):
        Aqf.step("Do a manual inspection to verify that the software release notes "
                        "of the latest CAM released version is in eB")
        Aqf.checkbox("Verify that CAM Software Release Notes are in eB (a Version Description Document and Deployment Process for each CAM version)")

        Aqf.step("CAM includes a VDD in the online documentation page of the system under test")
        Aqf.step("Open the CAM online documentation page")
        Aqf.step("For 'karoo_mkat' this will be at http://monctl.mkat.karoo.kat.ac.za")
        Aqf.step("Navigate to: CAM documentation, Version Descriptions")
        Aqf.checkbox("Verify the latest Version Description Document is included in the online CAM documentation")
        
        Aqf.step("Navigate to: CAM documentation, CAM Developer Documentation, Deployment Process Documents")
        Aqf.checkbox("Verify the latest Version Description Document is included in the online CAM documentation")
        
        Aqf.end()
