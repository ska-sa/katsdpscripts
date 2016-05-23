import time
import random

from tests import settings, fixtures, specifics, utils, wait_sensor_includes, Aqf, AqfTestCase
from nosekatreport import site_only, system, aqf_vr, site_acceptance

IP_WITH_PTP = '10.97.8.1'

@system('all')
class TestControlDemo(AqfTestCase):

    """Tests Critical alarms."""

    def setUp(self):
        fixtures.sim = self.sim
        fixtures.cam = self.cam
        
        self.sub_nr = 1
        self.sub = self.cam.subarray_1
        self.sub.req.free_subarray()
        self.controlled = specifics.get_controlled_data_proxy(self)
        self.selected_ant = self.cam.ants[-1].name

    def tearDown(self):
        pass

    def _get_time(self):
        """ssh to server and check if ptp seconds offset and date"""
        cmd = ["cat /opt/ptp/status | grep Offset", "date"]
        try:
            ssh = utils.ssh_server(IP_WITH_PTP)
            Aqf.step("Executing {} on {} over ssh".format(cmd, IP_WITH_PTP))
            for _cmd in cmd:
                stdin, stdout, stderr = ssh.exec_command(_cmd)
                Aqf.step(stdout.readlines())
                if _cmd == 'date':
                    Aqf.checkbox("Verify that date and time is the same as that on KatGUI")
                else:
                    Aqf.checkbox("Verify that the seconds offset from master is small i.e less than 0.001 ")
            ssh.close()
        except Exception as err:
            Aqf.step("System failed to fetch time from the trusted time reference")

    def check_mount(aqfbase, ip):
        """Ssh to server and check if it is mounted to Sp server."""
        cmd = "mount | grep mnt"
        yr = date.today().year
        dy = date.today().day - 1
        if len(str(dy)) == 1:
            dy = "0%s" % dy
        
        month = "0%s" % date.today().month
        dir = "tasklog/%s/%s/%s" % (yr, month, dy)
        ssh = utils.ssh_server(ip)
        stdin, stdout, stderr = ssh.exec_command("ls /var/kat/%s" % dir)
        Aqf.step("Listing what is in the Cam (monctl) server")
        Aqf.step(stdout.readlines())
        Aqf.checkbox("Verify that the schedule id in Cam server will be the same as those"
                     " in SP archive server")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read()
        if output:
            try:
                sp_ip = output.split(":")[0]
                lst_cmd = 'ls %s/%s' % (output.split(":")[1].split(" ")[0], dir)
                ssh_sp = utils.ssh_server(sp_ip)
                stdin, stdout, stderr = ssh_sp.exec_command(lst_cmd)
                Aqf.step("Listing what is in the SP archive")
                Aqf.step(stdout.readlines())
                Aqf.checkbox("Verify that the schedule id in SP archive is the same"
                             " as in the mounted server")
            except Exception as err:
                Aqf.step("Could not list what is in the SP archive (%s) "  % err)
        else:
            Aqf.step("Could not list what is in the SP archive")
        ssh.close()
        ssh_sp.close()

    @aqf_vr('VR.CM.DEMO.DS.13')
    def test_demonstrate_display_date_time(self):
        """Demonstrate that CAM display date and time"""
	Aqf.step("Open KatGUI")
        Aqf.step("On the 'User Navigation' panel click 'Configure KatGUI'")
        Aqf.step("Select the checkbox for LST, local, UTC and Julian day")
        Aqf.checkbox("Verify that LST, local, UTC "
                     "and Julian day are shown on KatGUI")

        Aqf.step("Verify that system is fetching time from a trusted time reference")
        Aqf.step("Open a browser and enter ip: 10.97.64.1")
        Aqf.step("Login as spadmin / admin123")
        Aqf.checkbox("Verify that date and time displayed by KatGUI is the same "
                     "as the one displayed by spectracom(trusted time reference)")

        Aqf.step("Change the local KatGUI workstation time to be different from actual time")
        Aqf.checkbox("Verify that LST, local, UTC "
                     "and Julian day are still correct on KatGUI")

        Aqf.checkbox("Reset the local KatGUI workstation time to the correct time")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DW.15')
    def test_demonstrate_weather_data(self):
        """Demonstrate that CAM display weather data."""
        Aqf.step("Open weather display window in KatGUI")
        Aqf.checkbox("Verify that the graphs are updated")
        Aqf.step("Use the drop down box 'for the last' and choose the duration")
        Aqf.checkbox("Verify that data is displayed for the past duration selected")
        Aqf.checkbox("The system will now simulate a high gust wind")
        specifics.set_wind_speeds(self, 18)
        Aqf.checkbox("Verify that wind speed graph exceeds the gust-wind-speed line")
        Aqf.checkbox("Verify that the ANC_Wind_Gust alarm is reported")
        Aqf.checkbox("The system will now reset the wind to normal")
        specifics.reset_wind_speeds(self, 6)
        Aqf.checkbox("Verify that the ANC_Wind_Gust alarm returns to nominal")
        Aqf.checkbox("Verify that system Interlock returns to NONE - windstow release can take up to 2 min")
        Aqf.sensor("cam.sensors.sys_interlock_state").wait_until("NONE", sleep=1, counter=260)
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DO.16')
    def test_demonstrate_cam_monitoring_via_internet(self):
        """Demonstrate that cam monitoring can be accessed outside the SKA SA network."""
        Aqf.step("Configure VPN")
        Aqf.step("Unplug ethernet cable and turn off wi-fi")
        Aqf.step("Use 3g dongle (Make sure that a VPN is configured)")
        Aqf.step("Open KatGUI and login as 'Monitor Only'")
        Aqf.checkbox("Verify that you are logged in and the there is "
                     "no time syncing error")

        Aqf.end()
        
    @aqf_vr('VR.CM.DEMO.DO.18')
    @site_acceptance
    def test_demonstrate_cpt_control_centre(self):
        """Demonstrate that cam supports a control centre in cape town."""
        Aqf.step("Open KatGUI at Cape Town control centre")
        Aqf.step("Login as Lead Operator")
        Aqf.checkbox("Verify that login is successfull")
        Aqf.end()
    
    @site_acceptance
    @aqf_vr('VR.CM.DEMO.MA.20')
    def test_demonstrate_cam_mounts(self):
        """Demonstrate that the karoo archive used by CAM is mounted to storage
        location provided by SP"""
        Aqf.step("Open an SSH session to the MONCTL headnode")
        Aqf.step("Execute the following commands to verify that CAM uses mounted storage")
        Aqf.step("Verify that in the Karoo the mounted storage is on the SP storage")
        Aqf.step("sudo mount [note /mnt/nfs]")
        Aqf.step("cd /var/kat")
        Aqf.step("ls -la [note katstore, log, tasklog, postgresql, userfiles]")
        Aqf.step("Close the SSH session")
        Aqf.checkbox("CAM is using mounted storage")
        Aqf.end()
    
    @aqf_vr('VR.CM.DEMO.L.21')
    def test_demonstrate_user_log_messages_editable(self):
        """Demonstrate that CAM can edit user log messages."""
        Aqf.step("Open two KatGUI")
        Aqf.step("Login as Lead operator and as Operator respectively")
        Aqf.step("From both KatGUI open USER LOGS display")
        Aqf.step("From both select the log that you want to edit")
        Aqf.step("Edit the logs and click submit log button")
        Aqf.checkbox("Verify that your changes have been "
                     "added, by selecting the log again")
                     
        Aqf.step("Select 'MY USER LOGS' tab")
        Aqf.checkbox("Verify that you only see and edit your list")
        Aqf.end()
    
    @aqf_vr('VR.CM.DEMO.L.23')
    def test_demonstrate_user_logs_types(self):
        """Demonstrate that CAM provides input capability allowing
            a user to create logs."""
        Aqf.step("Open USER LOGS display in KatGUI")
        Aqf.step("Select the add (+) icon to add a new user log")
        Aqf.step("Use the input field and select log type 'Shift log'")
        Aqf.step("From the form Enter start time and the endtime")
        Aqf.step("Capture a note or reminder in the 'log message' text area")
        Aqf.step("To submit log click 'SUBMIT LOG' button")
        Aqf.step("Select the 'MY USER LOGS' tab in USER LOGS display")
        Aqf.checkbox("Verify that 'Shift log' has been added to the list "
                     "as 'Shift log' type")
        Aqf.checkbox("Verify that 'Shift log' timestamp is in milliseconds")
        
        Aqf.step("Select the add(+) icon to add a new userlog")
        Aqf.step("Use the input field and select log type 'Time loss log'")
        Aqf.step("From the form Enter start time and the endtime")
        Aqf.step("Capture duration and reason for loss of observation time in "
                 "the 'log message' text area")
        Aqf.step("To submit log click 'SUBMIT LOG' button")
        Aqf.step("Select the 'MY USER LOGS' tab in USER LOGS display")
        Aqf.checkbox("Verify that 'Time loss log' has been added to the "
                     "list as 'Time-loss log' type")
        
        Aqf.step("Select the add(+) icon to add a new userlog")
        Aqf.step("Use the input field and select log type 'Observation log'")
        Aqf.step("From the form Enter start time and the endtime")
        Aqf.step("Capture general observation information and information on "
                 "conditions that affect data quality, in the 'log message' text area")
        Aqf.step("To submit log click 'SUBMIT LOG' button")
        Aqf.step("Select the 'MY USER LOGS' tab in USER LOGS display")
        Aqf.checkbox("Verify that general observation information has "
                     "been added to the list as 'Observation log' type")
        
        Aqf.step("Select the add(+) icon to add a new userlog")
        Aqf.step("Use the input field and select log type 'Status log'")
        Aqf.step("From the form Enter start time and the endtime")
        Aqf.step("Capture note about health of system components in the "
                 "'log message' text area")
        Aqf.step("To submit log click 'SUBMIT LOG' button")
        Aqf.step("Select the 'MY USER LOGS' tab in USER LOGS display")
        Aqf.checkbox("Verify that note about health of system components has "
                     "been added to the list as 'Status log' type")
        
        Aqf.step( "Select the add(+) icon to add a new userlog")
        Aqf.step("Use the input field and select log type 'Maintenance log'")
        Aqf.step("From the form Enter start time and the endtime")
        Aqf.step("Capture a notes on troubleshooting and maintenance in "
                 "the 'log message' text area")
        Aqf.step("To submit log click 'SUBMIT LOG' button")
        Aqf.step("Select the 'MY USER LOGS' tab in USER LOGS display")
        Aqf.checkbox("Verify that notes of troubleshooting and maintenance have "
                     "been added to the list as 'Maintenance log' type")

        Aqf.checkbox("Confirm CAM allows the different user log types "
                     "to be entered and timestamped them with millisecond precision.")

        Aqf.end()
        
    @aqf_vr('VR.CM.DEMO.L.24')
    def test_demonstrate_user_log_attach_file(self):
        """Demonstrate that one or more file can be attached to user log."""
        Aqf.step("Open USER LOGS display in KatGUI")
        Aqf.step("Select the log that you want to edit or attach a file to")
        Aqf.step("Click Browse.. button and browse to a file to be attached")
        Aqf.step("Click 'SUBMIT LOG' button")
        Aqf.checkbox("Verify that your changes have been added, "
                     "by selecting the log again")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.MSG.27')
    def test_demonstrate_plot_monitor_points(self):
        """Demonstrate that cam plot up to 64 monitor points"""
        Aqf.step("In KatGUI open SENSOR GRAPHS")
        Aqf.checkbox("Verify that there is a mechanism "
                     "allowing a user to plot sensors")
        Aqf.step("Enter text *pos* in the search text field")
        Aqf.step("Select 64 sensors")
        Aqf.checkbox("Verify that data is plotted for all 64 sensors")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.L.29')
    def test_demonstrate_generate_shift_report(self):
        """Demonstrate that cam generate shift_report"""
        Aqf.step("Open USER LOGS display in KatGUI")
        Aqf.step("Select Uselog Reports tab")
        Aqf.step("Use the form to filter the logs and time period")
        Aqf.step("Click 'PREVIEW REPORT' button")
        Aqf.checkbox("Verify that user logs of the selected types and for the selected time period are displayed")
        Aqf.step("NB: If there are no activity logs for the time period selected, the report will be blank")
        Aqf.step("Select to include activity logs in shift report using the checkbox control")
        Aqf.step("Click 'Export PDF' button")
        Aqf.checkbox("Open the saved PDF and verify that activity logs for the selected time period are included in the report")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DS.38')
    def test_demonstrate_faults_lru(self):
        """Demonstrate that CAM display faults to LRU level"""

        Aqf.step("From KatGUI open 'SENSOR LIST DISPLAY'")
        Aqf.step("Click on one of the Receptors")
        Aqf.step("Search on 'device-status'")
        Aqf.checkbox("Verify that this display can be used to determine the LRU failures")
        Aqf.step("Click on Hide Nominal and clear the search filter")
        Aqf.checkbox("Verify that the individual sensors that are not nominal are displayed")
        Aqf.step("Click on one of the Receptors")
        Aqf.step("Search on 'dig' or 'rsc'")
        Aqf.step("Click on Hide Nominal")
        Aqf.checkbox("Verify that this display can be used to determine the failures on that device")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBR.63')
    def test_demonstrate_add_resource_to_maintenance(self):
        """Demonstrate that CAM put resource into maintenance."""

        Aqf.step("NOTE: The CA will flag a resources as FAULTY, which will prevent it from being used for normal observations once the subarray is freed.")
        Aqf.step("      The Lead Operator will investigate the FAULTY resource to determine if it should be put into maintenance.")
        Aqf.keywait("OK to continue")

        Aqf.step("In KatGUI login as Lead Operator")
        Aqf.step("Open SUBARRAY display and create SUBARRAY 1 with some resources")
        Aqf.step("Delegate SUBARRAY 1 to a Control Authority")
        
        Aqf.step("In a 2nd KatGUI login as the Control Authority")
        Aqf.checkbox("Verify that you are logged in as Control Authority")
        Aqf.step("Open the SUBARRAY 1 display")
        Aqf.step("In the Resources panel, mark a resource as FAULTY")
        Aqf.keywait("Free the subarray")
        
        Aqf.step("On the Lead Operator KatGUI verify that the resource remains FAULTY")
        Aqf.checkbox("Verify that the FAULTY resource cannot be allocated to a normal subarray")
        
        Aqf.step("On the Lead Operator KatGUI mark the FAULTY resource as in-maintenance")
        Aqf.checkbox("Verify that the resource in-maintenance "
                     " has a maintenance icon (spanner)")

        Aqf.step("In the Resources panel, take resource out of maintenance")
        Aqf.checkbox("Verify that the maintenance icon (spanner) is removed from the resource")
        Aqf.step("In the Resources panel, remove the FAULTY flag from the resource")
        Aqf.checkbox("Verify that the resource can now again be assigned to a subarray for normal observations")
        Aqf.keywait("Free all subarrays")

        Aqf.end()

    @system("mkat", all=False)
    @aqf_vr('VR.CM.DEMO.C.48')
    def test_demonstrate_receptor_power_status(self):
        """Demonstrate that CAM report the correct receptor power status to the BMS"""
        ant_proxy = random.choice(self.cam.ants)
        _proxy_name = ant_proxy.name
        proxy_number = _proxy_name.replace("m", "")
        ap_connected_sensor = "cam.{}.sensor.ap_connected".format(_proxy_name)
        rsc_connected_sensor = "cam.{}.sensor.rsc_connected".format(_proxy_name)

        Aqf.step("In KatGUI open Sensor list display")
        Aqf.step("Click ANC from Proxies")
        Aqf.step("Search for 'power-status' sensor")
        Aqf.step("Verify that power status for receptors in the configuration is 'on' or 'off'")
        Aqf.checkbox("Verify that power status for receptors that are not in configuration is 'unavailable'")
        
        Aqf.hop("Selecting {} for the test".format(_proxy_name))
        Aqf.step("Click ANC from Proxies")
        Aqf.step("Search for bms.{}-power-status sensor".format(_proxy_name))
        
        sen=self.cam.list_sensors("ap%s_running" % proxy_number)
        sim_node=sen[0][1].replace(".ap%s.running" % proxy_number, "")
        the_sim_node = getattr(self.cam, sim_node)
        Aqf.checkbox("Verify that bms.{}-power-status value is 'on'.".format(_proxy_name))

        Aqf.step("The system will now stop ap%s and rsc%s to simulate "
                 "a power failure at the pedestal" % (proxy_number, proxy_number))
        the_sim_node.req.kill("ap%s" % proxy_number)
        the_sim_node.req.stop("rsc%s" % proxy_number)
        Aqf.sensor(rsc_connected_sensor).wait_until(False, sleep = 1, counter=100)
        Aqf.sensor(ap_connected_sensor).wait_until(False, sleep = 1, counter=100)
        
        Aqf.checkbox("Verify that bms_%s_power_status value is 'off'" %  _proxy_name)

        Aqf.step("The system will now start ap%s and rsc%s again" % (proxy_number, proxy_number))
        the_sim_node.req.start("ap%s" % proxy_number)
        the_sim_node.req.start("rsc%s" % proxy_number)
        Aqf.sensor(rsc_connected_sensor).wait_until(True, sleep = 1, counter=100)
        Aqf.sensor(ap_connected_sensor).wait_until(True, sleep = 1, counter=100)
        Aqf.checkbox("Verify that bms.{}-power-status value is 'on'.".format(_proxy_name))

        Aqf.end()

    @system("mkat", all=False)
    @aqf_vr('VR.CM.SITE.C.24')
    def test_demonstrate_receptor_power_status_on_site(self):
        """Demonstrate that CAM report the correct receptor power status to the BMS"""
        ant_proxy = random.choice(self.cam.ants)
        proxy_name = ant_proxy.name

        Aqf.step("Open KatGUI")
        Aqf.step("Open Sensor list display")
        Aqf.step("Click %s from components and verify that ap and rsc are again connected" % proxy_name)
        Aqf.step("On KatGUI Sensor list display, click ANC")
        Aqf.checkbox("Verify that bms_%s_power_status sensor shows ON" % proxy_name)
        Aqf.step("Open BMS console system")
        Aqf.checkbox("Verify that tag SKA.MKT_%s_ON has value 1(on)" % proxy_name.upper())

        Aqf.step("Stop/disconnect the ap and rsc on %s" % proxy_name)
        Aqf.step("On KatGUI Sensor list display, click ANC from components")
        Aqf.checkbox("Verify that bms_%s_power_status sensor shows OFF" % proxy_name)
        Aqf.step("Open BMS console system")
        Aqf.checkbox("Verify that tag SKA.MKT_%s_ON has value 0(off)" % proxy_name.upper())

        Aqf.step("Restart/connect the ap and rsc on %s" % proxy_name)
        Aqf.checkbox("On KatGUI Sensor list display, click %s and verify that ap and rsc are again connected" % proxy_name)
        Aqf.step("On KatGUI Sensor list display, click ANC")
        Aqf.checkbox("Verify that bms_%s_power_status sensor shows ON" % proxy_name)
        Aqf.step("Open BMS console system")
        Aqf.checkbox("Verify that tag SKA.MKT_%s_ON has value 1(on)" % proxy_name.upper())

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.MSG.58')
    def test_demonstrate_view_historical_monitoring_data(self):
        """Demonstrate that cam have a display to view historical monitoring data"""
        Aqf.step("In KatGUI open SENSOR GRAPHS")
        Aqf.step("Enter text '*wind_speed*' in the search text field")
        Aqf.step("Select one of the sensor in the list)")
        Aqf.step("Select a time period")
        Aqf.checkbox("Verify that data for the time period is plotted for selected sensors")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.CBF.52')
    def test_demonstrate_trigger_next_pulser(self):
        """Demonstrate that cam trigger next pulser."""
        Aqf.waived("Next pulser trigger is USE related and waived for QBL(B)")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.CBF.55')
    def test_demonstrate_pulsar_timer_schedule(self):
        """Demonstrate pulsar timer schedule trigger."""
        Aqf.waived("Pulsar timer schedule trigger is USE related and waived for QBL(B)")
        Aqf.end()
    
    @aqf_vr('VR.CM.SITE.C.23')
    def test_demonstrate_report_ils_site(self):
        """Demonstrate that cam report to ILS """
        Aqf.step("Open the ILS system")
        Aqf.step("From the ILS system subscribe to 'anc_kapb_rfi_door_open' or any other sensor")
        Aqf.step("From ILS system set sensor strategy for the sensor to 'period,1'")
        Aqf.checkbox("Verify that you get sensor updates from ILS server")
        Aqf.step("If possible, trigger a change on the value of the sensor")
        Aqf.step("e.g. open the KAPB door in the case of 'anc_kapb_rfi_door_open'")
        Aqf.checkbox("Verify that the sensor updates reflect the new value")
        Aqf.step("From ILS system unsubscribe from the sensor")
        Aqf.checkbox("Verify that the updates stop")

        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.DEMO.V.12")
    def test_gui_interface_provides_switching_on_and_off_of_flood_lights(self):
        """Demonstrate that the CAM GUI interface provides switching on and off of floodlights."""
        Aqf.step("Login to KatGui")
        Aqf.step("Open the 'VIDEO DISPLAY'")
        Aqf.step("Locate the button to toggle the flood lights")
        Aqf.step("Toggle the flood lights off/on by clicking 'Floodlights' button")
        Aqf.checkbox("Verify that the floodlights are off/on as expected")
        Aqf.step("Open Sensor list display")
        Aqf.step("Click ANC from components and search on 'flood'")
        Aqf.checkbox("Verify that vds.flood-lights-on sensor value toggle between false/true as expected")

        Aqf.end()

