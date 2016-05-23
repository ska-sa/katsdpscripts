

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
from tests import settings ,fixtures, specifics, Aqf, AqfTestCase
from nosekatreport import system, aqf_vr


@system('all')
class TestCriticalAlarms_Demo(AqfTestCase):

    """Tests Critical alarms."""

    def setUp(self):
        fixtures.sim = self.sim
        fixtures.cam = self.cam

    def tearDown(self):
        pass

    @aqf_vr('VR.CM.DEMO.A.10', 'VR.CM.DEMO.A.41')
    def test_demonstrate_critical_alarm_for_fire(self):
        """
        Fire - Demonstrate that a critical alarm is displayed to operators
        for fire in the processing area
        """

        Aqf.step("Login as Lead Operator on KatGUI and open 'HEALTH' window on KatGUI")
        Aqf.step("Also open 'Alarms' window on KatGUI")

        Aqf.hop("Simulating fire")
        specifics.simulate_fire(self)
        
        Aqf.checkbox("Verify that the Health display indicates the 'System Fire OK' as an error")
        Aqf.checkbox("Verify that the System_Fire alarm is indicated as a critical alarm on the Alarms display")

        Aqf.hop("Resetting fire")
        specifics.reset_fire(self)

        Aqf.checkbox("Verify that the System_Fire alarm returns to nominal on the Alarms display")

        Aqf.step("Select and Clear the System_Fire nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_Fire alarm is nominal and cleared on the Alarms display")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.A.10', 'VR.CM.DEMO.A.42')
    def test_demonstrate_critical_alarm_for_cooling_failure(self):
        """
        Cooling failure - Demonstrate that a critical alarm is displayed to operators
        for cooling failure in the processing area
        """
        Aqf.step("Login as Lead Operator on KatGUI and open 'HEALTH' window on KatGUI")
        Aqf.step("Also open 'Alarms' window on KatGUI")

        Aqf.step("NOTE: Cooling failure alarm test")
        Aqf.step("NOTE: If your verification takes longer than the alarm delay (60s) the computing will be shutdown !!!")
        Aqf.keywait("OK to continue?")

        Aqf.hop("Simulating cooling failure")
        specifics.simulate_cooling_failure(self)

        Aqf.checkbox("Verify on the Health display that the 'System Cooling OK' indicator has been set to an error value")
        Aqf.checkbox("Verify on the Alarms display that the System_Cooling_Failure alarm is indicated as a critical alarm")
       
        Aqf.hop("Resetting cooling failure")
        specifics.reset_cooling_failure(self)

        Aqf.step("Verify on the Alarms display that the System_Cooling_Failure alarm returns to nominal")
        Aqf.step("Select and Clear the System_Cooling_Failure nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_Cooling_Failure alarm is nominal and cleared on the Alarms display")

        Aqf.checkbox("Verify on the Health display that 'System Cooling OK' indicator to normal")

        Aqf.end()

    @system('mkat', 'mkat-rts', all=False)
    @aqf_vr('VR.CM.DEMO.A.10')
    def test_demonstrate_critical_alarm_for_imminent_power_failure(self):
        """
        Imminent power failure - Demonstrate that a critical alarm is displayed to operators
        for imminent power failure in the processing area
        """
        Aqf.step("Login as Lead Operator on KatGUI and open 'HEALTH' window on KatGUI")
        Aqf.step("Also open 'Alarms' window on KatGUI")

        Aqf.step("NOTE: Imminent power failure alarm test")
        Aqf.step("NOTE: If your verification takes longer than the alarm delay (30s) the computing will shutdown !!!!")
        Aqf.keywait("OK to continue?")

        Aqf.hop("Simulating imminent power failure")
        specifics.simulate_imminent_power_failure(self)

        Aqf.checkbox("Verify on the Health display that the 'System Power OK' indicator has been set to an error value")
        Aqf.checkbox("Verify on the Alarms display that the System_Imminent_Power_Failure alarm is indicated as a critical alarm")
       
        Aqf.hop("Resetting imminent power failure")
        specifics.reset_imminent_power_failure(self)

        Aqf.checkbox("Verify on the Health display that 'System Power OK' indicator returns to normal")
        Aqf.checkbox("Verify on the Alarms Display that the System_Imminent_Power_Failure alarm returns to nominal")

        Aqf.step("Clear the System_Imminent_Power_Failure nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_Imminent_Power_Failure alarm is nominal and cleared on the Alarms display")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.A.28')
    def test_demonstrate_organise_alarm(self):
        """Demonstrate that cam organise alarms"""
        Aqf.step("Open Alarm display in KatGUI.")
        ### TODO create error and warning alarm
        
        specifics.simulate_intrusion(self)  # Critical (System RFI Door open.
        
        Aqf.checkbox("Verify that there is display showing the complete list "
                     "of current alarms including 'System RFI Door'.")
        Aqf.checkbox("Verify that there are alarms with different severity.")
        
        Aqf.checkbox("Verify that Alarms are organised according to severity and time by clicking on the Severity column header.")
        specifics.reset_intrusion(self)
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.A.33')
    def test_demonstrate_critical_alarms_raised(self):
        """
        Demonstrate that fire alarm, imminenent power failure alarm and
        cooling failure alarm are triggered when the system is in fault.
        """

        Aqf.step("Open 'ALARMS' display from KatGUI")
        # Fire alarm
        Aqf.step("Simulating fire")
        specifics.simulate_fire(self)
 
        Aqf.checkbox("Verify that the System_Fire alarm is indicated as a"
                     " critical alarm on the Alarms display")
        specifics.reset_fire(self)
        Aqf.checkbox("Verify that the System_Fire alarm is back to nominal")

        # Cooling failure alarm
        Aqf.step("NOTE: Starting System_Cooling_Failure alarms test")
        Aqf.step("NOTE: If your verification takes longer than the alarm delay "
                 "(30s) the computing will shutdown !!!!")
        Aqf.keywait("OK to continue?")
        Aqf.step("Simulating cooling failure")
        specifics.simulate_cooling_failure(self)
        Aqf.checkbox("Verify on the Alarms display that the "
                     "System_Cooling_Failure alarm is indicated as a critical alarm")
        Aqf.step("Resetting cooling failure")
        specifics.reset_cooling_failure(self)
        Aqf.checkbox("Verify that the System_Cooling_Failure alarm is back to nominal")
        
	if system == ('MKAT', 'MKAT-RTS'):
        # imminent power failure alarm
            Aqf.step("NOTE: Starting System_Imminent_Power_Failure alarms test")
            Aqf.step("NOTE: If your verification takes longer than the alarm delay "
                     "(30s) the computing will shutdown !!!!")
            Aqf.keywait("OK to continue?")
            Aqf.step("Simulating imminent power failure")
            specifics.simulate_imminent_power_failure(self)
            Aqf.checkbox("Verify on the Alarms display that the "
                         "System_Imminent_Power_Failure alarm is "
                         "indicated as a critical alarm")
            Aqf.step("Resetting imminent power failure")
            specifics.reset_imminent_power_failure(self)
            Aqf.checkbox("Verify that the System_Imminent_Power_Failure alarm is back to nominal")

        # cooling failure alarm
        Aqf.step("NOTE: Starting System_KAPB_Temperature_Failure alarms test")
        Aqf.step("NOTE: If your verification takes longer than the alarm delay "
                 "(30s) the computing will shutdown !!!!")
        Aqf.keywait("OK to continue?")
        Aqf.step("Simulating cooling failure")
        specifics.simulate_cooling_failure(self)
        Aqf.checkbox("Verify on the Alarms display that the "
                     "'System_KAPB_Temperature_Failure' alarm "
                     "is indicated as a critical alarm")
        Aqf.step("Resetting cooling failure")
        specifics.reset_cooling_failure(self)
        Aqf.checkbox("Verify that the System_KAPB_Temperature_Failure alarm is back to nominal")

        Aqf.step("On KatGUI clear the System_KAPB_Temperature_Failure alarm")
        Aqf.checkbox("Verify that the System_KAPB_Temperature_Failure alarm")
        Aqf.end()

    @system('mkat', all=False)
    @aqf_vr('VR.CM.DEMO.A.43')
    def test_cam_generates_critical_alarm_for_kapb_temperature(self):
        """
        KAPB temperature out of range - Demonstrate that a critical alarm is displayed to operators
        for KAPB temperature out of range
        """

        Aqf.step("Open 'HEALTH AND STATE' window in KatGUI")
        Aqf.step("Also open 'Alarms' window on KatGUI")

        Aqf.step("NOTE: KAPB temperature failure alarm test")
        Aqf.step("NOTE: If your verification takes longer than the alarm delay (60s) the computing will be shutdown !!!")
        Aqf.keywait("OK to continue?")

        Aqf.hop("Simulating cooling failure")
        specifics.simulate_cooling_failure(self)

        Aqf.checkbox("Verify on the Health display that the 'System Cooling OK' indicator has been set to an error value")
        Aqf.checkbox("Verify on the Alarms display that the System_KAPB_Temperature_Failure alarm is indicated as a critical alarm")
       
        Aqf.hop("Resetting cooling failure")
        specifics.reset_cooling_failure(self)

        Aqf.step("Verify on the Alarms display that the System_Cooling_Failure alarm returns to nominal")
        Aqf.step("Select and Clear the System_KAPB_Temperature_Failure nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_KAPB_Temperature_Failure alarm is nominal and cleared on the Alarms display")

        Aqf.checkbox("Verify on the Health display that 'KAPB temperature' indicator returns to normal")

        Aqf.end()


    @system('kat7', all=False)
    @aqf_vr('DEMO_CAM_ALARMS_ups_low_inhibit_antennas')
    def test_asc_ups_battery_low(self):
        """Tests handling of the ASC UPS Battery Low alarm.

        A critical alarm is raised if a sensor changes. Antennas are inhibited.

        Demonstrate that a critical alarm is displayed to operators for ASC UPS Battery Low

        """

        Aqf.step("Open KatGui and verify that the ASC UPS Battery Low critical alarm is absent")
        Aqf.step("Verify that the antennas are not inhibited before continuing "
                 "(in case they are inhibited, first Click on Resume Operations)")
        Aqf.keywait("OK to continue?")

        #Trigger the alarm
        Aqf.step("Open iPython, configure_sim()")
        Aqf.step("Trigger the ASC UPS battery low alarm by issueing")
        Aqf.step("    sim.asc.req.set_sensor_value('ups.battery.not.low', 0)")
        Aqf.checkbox("Verify that ASC_UPS_Battery_Low alarm is critical")
           
        Aqf.checkbox("Verify on KatGUI that antennas are inhibited")
                                                                                                            
        Aqf.step("Reset the ASC UPS battery low alarm by issueing")
        Aqf.step("    sim.asc.req.set_sensor_value('ups.battery.not.low', 1)")
        Aqf.checkbox("Verify that ASC_UPS_Battery_Low alarm is back to normal")

        #Now clear the alarm
        Aqf.step("Acknowledge and clear ASC_UPS_Battery_Low alarm from KatGUI")
        Aqf.equals(
           self.cam.kataware.sensor.alarm_ASC_UPS_Battery_Low.get_value(),
           "nominal,cleared,anc_asc_ups_battery_not_low value = True. status = nominal.",
           '')

        Aqf.step("NOTE: Check that the antennas are still inhibited as an operator has to resume after an ASC UPS Battery low intervention")
        Aqf.step("Open KatGui and navigate to Operator Control")
        Aqf.step("Finally click on Resume Operations !!!")
        Aqf.checkbox("Verify that antennas are no longer inhibited")
        Aqf.end()

 
