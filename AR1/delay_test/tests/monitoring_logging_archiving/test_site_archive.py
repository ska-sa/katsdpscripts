###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""

from nosekatreport import (aqf_vr, system, site_acceptance)
from tests import Aqf,  AqfTestCase

@system('all')
class TestArchive(AqfTestCase):

    @aqf_vr("VR.CM.SITE.M.5", "VR.CM.SITE.M.13", "VR.CM.SITE.M.16")
    def test_01_archive_data_for_12_months(self):
        """Check Cape Town Archive.

        More than 12 months of monitoring data should be in the archive.

        """
        Aqf.step("Browse the Cape Town katstore archive at /var/kat/archive/katstore on ")
        Aqf.step("    KAT-7:     http://kat-flap-cpt.kat7.control.kat.ac.za/kat/cpt_kat7_archive/katstore/")
        Aqf.step("    MKAT-RTS:  http://kat-flap-cpt.mkat-rts.control.kat.ac.za/kat/cpt_mkat_rts_archive/katstore/")
        Aqf.step("    MKAT:      http://kat-flap-cpt.mkat.control.kat.ac.za/kat/cpt_mkat_archive/katstore/")
        Aqf.checkbox("Ensure that the CAM Cape Town archive archives all "
                     "system monitoring data. If less than "
                     "12 months of data is available, inspect the cron config "
                     "on kat-flap-cpt.<systype>.control to verify "
                     "that a synchronisation process is scheduled for "
                     "mirroring the data.")
        Aqf.checkbox("Use sensorgraph to retrieve some "
                     "historic wind speed sensor data from the Cape Town archive.")
        Aqf.end()

    @aqf_vr("VR.CM.SITE.M.14", "VR.CM.SITE.M.22")
    def test_01_archive_logs_for_12_months(self):
        """Check Cape Town Archive.

        More than 12 months of logs should be in the archive.

        """
        Aqf.step("Browse the Cape Town log archive at /var/kat/archive/logs on ")
        Aqf.step("    KAT-7:     http://kat-flap-cpt.kat7.control.kat.ac.za/kat/cpt_kat7_archive/log/")
        Aqf.step("    MKAT-RTS:  http://kat-flap-cpt.mkat-rts.control.kat.ac.za/kat/cpt_mkat_rts_archive/log/")
        Aqf.step("    MKAT:      http://kat-flap-cpt.mkat.control.kat.ac.za/kat/cpt_mkat_archive/log/")
        Aqf.checkbox("Verify that the CAM Cape Town archive contains all "
                     "system log files (activity, alarms, proxies and components). If less than "
                     "12 months of files is available, inspect the cron config "
                     "on kat-flap-cpt.<systype>.control to verify "
                     "that a synchronisation process is scheduled for "
                     "mirroring the logs.")
        Aqf.end()

    @aqf_vr("VR.CM.SITE.M.15")
    def test_archive_user_logs(self):
        """Check Cape Town Archive includes user logs.
        """
        Aqf.step("Browse the Cape Town userfiles archive at /var/kat/archive/userfiles on ")
        Aqf.step("    KAT-7:     http://kat-flap-cpt.kat7.control.kat.ac.za/kat/cpt_kat7_archive/userfiles/")
        Aqf.step("    MKAT-RTS:  http://kat-flap-cpt.mkat-rts.control.kat.ac.za/kat/cpt_mkat_rts_archive/userfiles/")
        Aqf.step("    MKAT:      http://kat-flap-cpt.mkat.control.kat.ac.za/kat/cpt_mkat_archive/userfiles/")
        Aqf.checkbox("Verify that the CAM Cape Town archive contains "
                     "the user log files.")

        Aqf.step("Browse the Cape Town database dump archive at /var/kat/archive/db_dumps on ")
        Aqf.step("    KAT-7:     http://kat-flap-cpt.kat7.control.kat.ac.za/kat/cpt_kat7_archive/db_dumps/")
        Aqf.step("    MKAT-RTS:  http://kat-flap-cpt.mkat-rts.control.kat.ac.za/kat/cpt_mkat_rts_archive/db_dumps/")
        Aqf.step("    MKAT:      http://kat-flap-cpt.mkat.control.kat.ac.za/kat/cpt_mkat_archive/db_dumps/")
        Aqf.checkbox("Verify that the CAM Cape Town archive contains "
                     "the katpersist database dump file.")
        Aqf.end()

    @aqf_vr("VR.CM.SITE.ZF.1")
    def test_analyse_cam_safe_design(self):
        """Analyse CAM safe design"""
        Aqf.checkbox("A safety analysis has been performed for CAM and the output is captured "
           "in the MeerKAT CAM Design Document. CAM uses only COTS equipment that complies to "
           "industrial standards.Verify that the CAM Cape Town archive contains ")
        Aqf.end()

    @aqf_vr("VR.CM.SITE.ZF.8")
    def test_analyse_cam_lcoally_fail_safe(self):
        """Analyse CAM locally fail safe"""
        Aqf.checkbox("A safety analysis has been performed for CAM and the output is captured "
            "in the MeerKAT CAM Design Document. CAM uses only COTS equipment that complies to "
            "industrial standards and are fail safe and does not rely on external components for operations.")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.MA.50')
    def test_analyse_monitoring_data_capacity(self):
        """Analyse the CAM monitoring data capacity"""
        Aqf.step("Refer to the following document")
        Aqf.step('https://docs.google.com/document/d/1ZWkuxW_zsOWdNfRWEQqgnHEyvF1TmBOZTvlA5AUu9Fs/edit#')
        Aqf.step("The kill test reached 100000 samples written in 2-3 seconds.")
        Aqf.step("The requirement equates to 76500 samples per second when extrapolating RTS sensors")
        Aqf.step("The kill test was performed on a single node with 200Gb disk. When given multiple servers and bigger "
                 " disk space then katstore monitoring rate can be much more than 76500 samples per second.")
        Aqf.checkbox("CAM monitoring data capacity is sufficient")
        Aqf.end()
