###################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from tests import settings, Aqf, AqfTestCase
from nosekatreport import (aqf_vr, system, slow, site_acceptance)
from katcorelib.targets import AzEl

import random
import time
import math

# TBD(TA): Maybe move this function somewhere more appropriate if it proves
# useful to others


def simple_track(kat, target, duration):
    """Track a target for some duration.

    Simple tracking of specified target for some duration. First slews to
    target and then track it for duration seconds.

    Parameters
    ----------
    kat : :class:`utility.KATCoreConn` object
        KAT connection object associated with this experiment
    target : katpoint.Target
        Target to track
    duration : float
        Duration that target should be tracked in seconds
    """
    from katcorelib import start_session
    opts = {
        'projection': 'zenithal-equidistant',
        'observer': 'CAM Integration Test',
        'centre_freq': 1822.0,
        'description': 'Target track',
        'dump_rate': 0.1,
        'experiment_id': None,
        'dbe_centre_freq': None,
        'sb_id_code': None
    }
    with start_session(kat, **opts) as session:
        session.standard_setup(**opts)
        session.capture_start()
        session.label('track')
        Aqf.log_info("Initiating %g-second track on target '%s'" %
                        (duration, target.name,))
        session.track(target, duration=duration, announce=False)

class TestCatalogues(AqfTestCase):
    """Test the CAM reading and tracking of objects specified in catalogues."""

    ELEV_LIMIT_DEG = [15+10, 90-10]
    """ Elevation limit used to ensure the targets chosen to track are above the
    horizon; also allow some room to ensure target will be visible for a while to
    allow tracking it"""

    def setUp(self):
        self.system = settings.system

    def _test_tracking_of_target(self, target):
        """Convenience function used to test the tracking of a target."""

        az, el = AzEl(*target.azel())
        Aqf.log_info("Testing target '%s' at (az=%f, el=%f)" %
                     (target.name, math.degrees(az), math.degrees(el)))

        # Randomly select an antenna that will be tracking the selected target and
        # point it
        Aqf.log_info("Select a receptor proxy and point it towards the given "
                     "target")
        ###proxy = random.sample(self.cam.ants, 1)[0]
        proxy = self.cam.ants[0]
        start_az = proxy.sensor.pos_actual_pointm_azim.get_value()
        start_el = proxy.sensor.pos_actual_pointm_elev.get_value()
        Aqf.log_info("Using receptor proxy '%s' currently at (az=%f, el=%f)" %
                     (proxy.name, start_az, start_el))
        proxy.req.target(target.description)
        proxy.req.mode("POINT")
        result = Aqf.sensor("cam.%s.sensor.mode" % proxy.name).wait_until("POINT", sleep=1, counter=15)

        if not result:
            Aqf.failed("Receptor not in mode POINT. Cannot continue.")
            return False

        # Test that the selected antenna is now locked to the requested target
        # Wait with timeout = 3 x 60 = 180s
        Aqf.log_info("Test that the antenna locks to the target (timeout=180s)...")
        Aqf.sensor(proxy.sensor.lock).wait_until(True, sleep=3, counter=60)
        Aqf.is_true(True == proxy.sensor.lock.get_value(), "Verify the antenna "
            "achieved target lock before timeout")

        Aqf.log_info("Check that the antenna follows the target (track for 1min)...")
        azim_before = proxy.sensor.pos_actual_pointm_azim.get_value()
        elev_before = proxy.sensor.pos_actual_pointm_elev.get_value()
        # Track for 1min
        time.sleep(60)
        azim_after = proxy.sensor.pos_actual_pointm_azim.get_value()
        elev_after = proxy.sensor.pos_actual_pointm_elev.get_value()
        Aqf.is_false(azim_before == azim_after, "Check that the antenna is moving "
                        "relative to its target in azimuth (before=%s, after=%s)" %
                        (azim_before, azim_after))
        Aqf.is_false(elev_before == elev_after, "Check that the antenna is moving "
                        "relative to its target in elevation (before=%s, after=%s)" %
                        (elev_before, elev_after))

        # Test that the antenna is still locked to the requested target
        Aqf.is_true(True == proxy.sensor.lock.get_value(), "Verify the antenna "
                    "is still locked to the target.")

    # NOTE on VR.CM.AUTO.OBS.21 (2014/01/30)
    # SystemEngineering said that this requirement will be changed to say that
    # CAM does not have to support catalogues for Deep Space Satellites

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.OBS.21")
    @system("all")
    @slow
    def test_catalogues_tracking_meo(self):
        """Test the CAM reading and tracking of objects specified in catalogues.

        This test specifically test:
            - Medium Earth Orbit Satellites
        """
        # Test a Medium Earth Orbit Satellite that will be visible for a while
        Aqf.step("Test tracking of source catalogue Medium Earth Orbit Satellite.")
        meo_targets = self.cam.sources.filter(tags="GPS",
            el_limit_deg=self.ELEV_LIMIT_DEG)
        try:
            self._test_tracking_of_target(random.sample(meo_targets, 1)[0])
        except ValueError:
            Aqf.skipped("No visible Medium Earth Orbit Satellites to test.")
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.OBS.21")
    @system("all")
    @slow
    def test_catalogues_tracking_heo(self):
        """Test the CAM reading and tracking of objects specified in catalogues.

        This test specifically test:
            - High Earth Orbit Satellites
        """
        # Test a High Earth Orbit Satellite that will be visible for a while
        Aqf.step("Test tracking of source catalogue High Earth Orbit Satellite.")
        heo_targets = self.cam.sources.filter(tags="GEO",
            el_limit_deg=self.ELEV_LIMIT_DEG)
        try:
            self._test_tracking_of_target(random.sample(heo_targets, 1)[0])
        except ValueError:
            Aqf.skipped("No visible High Earth Orbit Satellites to test.")
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.AUTO.OBS.21")
    @system("all")
    @slow
    def test_catalogues_tracking_sso(self):
        """Test the CAM reading and tracking of objects specified in catalogues.

        This test specifically test:
            - Solar System Objects
        """
        # Test a Solar System Object that will be visible for a while - only 3 available
        # at present - first try moon, then jupiter then the sun
        Aqf.step("Test tracking of source catalogue Solar System Object.")
        solar_targets = self.cam.sources.filter(tags="special",
            el_limit_deg=self.ELEV_LIMIT_DEG)
        moon = solar_targets['moon']
        if moon is not None:
            self._test_tracking_of_target(moon)
        else:
            jupiter = solar_targets['jupiter']
            if jupiter is not None:
                self._test_tracking_of_target(jupiter)
            else:
                sun = solar_targets['sun']
                if sun is not None:
                    self._test_tracking_of_target(sun)
                else:
                    Aqf.skipped("Not the Moon nor Jupiter nor the Sun is visible to "
                                "perform test to track Solar System Object.")
        Aqf.end()

    @aqf_vr("VR.CM.DEMO.CFG.37")
    @system("all")
    def test_catalogues_config(self):
        """
        Demonstrate that the source catalogues are part of the instrumental configuration
        data.
        """
        Aqf.step("Browse to the instrumental configuration data of the system.")
        Aqf.checkbox("Verify that the instrumental configuration data is located on the "
                     "head node at /var/kat/katconfig.")
        Aqf.checkbox("Verify that the source catalogues are part of the instrumental "
                     "configuration data (/var/kat/katconfig/user/catalogues).")
        Aqf.end()

