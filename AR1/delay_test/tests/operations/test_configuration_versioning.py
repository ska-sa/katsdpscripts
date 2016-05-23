###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import os
import time

from nosekatreport import (aqf_vr, system)
from tests import (Aqf, AqfTestCase)

from katcorelib.katobslib.common import (ScheduleBlockStates,
                                         ScheduleBlockTypes)
from katconf.git_tagger import get_git_bin, git


@system('all')
class TestConfigVersioning(AqfTestCase):

    """Tests for versioning of user configuration."""

    def setUp(self):
        self._git_bin = get_git_bin()
        self._git_path = None
        gp_options = ['/home/kat/svn/katconfig',
                      '/var/kat/katconfig']
        for gp in gp_options:
            if os.path.isdir(os.path.join(gp, '.git')):
                self._git_path = gp
        self._config_cleanup()

    def tearDown(self):
        self._config_cleanup()

    def _config_cleanup(self):
        Aqf.hop("Cleaning up git config after test")
        # Git fetch : Only get information about origin, dont merge it.
        self._git(['fetch'])
        # Git reset : Throw away all local changes.
        self._git(['reset', '--hard', 'origin/master'])
        # Git pull : Get into sync with remote.
        self._git(['pull'])

    @aqf_vr('VR.CM.DEMO.CFG.22')
    def test_katconfig_in_git(self):
        ret, out, err = self._git(['remote', '-v'])
        Aqf.hop("Showing that config is version controlled")
        Aqf.hop("The remote repository for {} is:".format(self._git_path))
        for line in out.splitlines():
            Aqf.hop(line)
        Aqf.step("With a registered github user that has access to ska-sa "
                     "login to github at https://github.com/ska-sa/katconfig to verify "
                     "git hosting of katconfig")
        Aqf.checkbox("Confirm katconfig is version controlled and archived in github")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.CFG.51')
    def test_select_config_label(self):
        config_labels = []
        Aqf.step("Listing config labels")
        s = self.cam.sys.req.list_config_label()
        Aqf.is_true(len(s) > 1, "Verifying that more than one config_label was recieved")
        count = 5
        for inform in s.informs:
            Aqf.hop(" ".join(inform.arguments))
            config_labels.append(inform.arguments[0])
            count -= 1
            if count < 1:
                break
        Aqf.hop("NOTE: The Control Authority has to specify the config_label to use for the subarray, together with receiver band and data product")
        Aqf.hop("      The Lead Operator configures the subarray according to these specifications before activating the subarray")
        Aqf.hop("       and delagating control of the active subarray to the CA")
        Aqf.step("On KatGUI login as Lead Operator")
        Aqf.step("Goto SET-UP SUBARRAY and click the menu option to select a config label")
        Aqf.checkbox("Verify KatGUI or iPython displays a matching list of config labels")

        Aqf.step("Clearing config_label on subarray 1")
        self.cam.subarray_1.req.set_config_label('')  # Clear config label.
        Aqf.equals(self.cam.subarray_1.sensors.config_label.get_value(), '',
                   "Verifying config_label has cleared")
        label = config_labels[-2]
        Aqf.hop("Selecting config_label '{}' for test".format(label))
        Aqf.step("In KatGUI set the config_label on subarray 1 to '{}'".format(label))
        Aqf.step("In KatGUI activate subarray 1")
        Aqf.step("In KatGUI Subarray display verify that config_label for subarray 1 is set to '{}'".format(label))
        Aqf.step("In KatGui open Sensor list, click on 'subarray_1' and verify the config_label is set to '{}'".format(label))
        Aqf.step("In KatGUI Free the subarray")
        Aqf.checkbox("The CA can specify and the Lead Operator can select the config_label to be used for observations in a subarray")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.CFG.54')
    def test_manual_update(self):
        """Demonstrate that CAM allows operators to manually add long-term
        system calibrations that become part of the Instrumental Configuration
        Data repository.
        These include:
            1. SP on-line processing configurations
            2. Antenna pointing models
            3. Tied-array reference co-ordinates
            4. Source catalogues
        """

        steps = ({'name': 'SP On-line.',
                  'path': None,
                  'description': "SP on-line configuration will be displayed "
                                 "as sensors."},
                 {'name': 'Pointing Models',
                  'path': 'user/pointing-models/systype(kat7 or mkat)/'},
                 {'name': 'Tied-Array reference',
                  'path': 'static/antennas'},
                 {'name': 'Source catalogues',
                  'path': 'user/catalogues/'})
        for step in steps:
            Aqf.hop(step.get('name'))
            path = step.get('path')
            if path is None:
                Aqf.hop(step.get('description', ''))
                continue
            Aqf.step("Inspect GitHub @ https://github.com/ska-sa/katconfig"
                         "/tree/master/{}".format(path))
            Aqf.step("Inspect {}".format(
                os.path.abspath(os.path.join(self._git_path, path))))
            Aqf.step("Inspect Instrumental Config on KatGUI for user configuration")
            Aqf.checkbox("The configuration files are stored in a "
                     "Code Versioning System ")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.CFG.62')
    def test_catalogs_are_automaticaly_updated(self):
        """Verify that CAM obtains and allows the automatic update of
        Instrumental Configuration Data for the following:
            1. UT1
            2. Leap seconds for UT1
            3. Satellite catalogues
        """
        filenames = [('./auto/catalogues/iridium.txt', 'Satellite'),
                     ('./auto/earth/finals.daily', 'UT1')]
        Aqf.hop("Performing 'git pull' for filenames {} in {}".format(filenames, self._git_path))
        self._git(['pull'])
        for filename, description in filenames:
            Aqf.hop("Getting git log for filename {}".format(filename))
            ret = self._git(['log', r'--pretty=tformat:%h %ai %ae %s',
                             '--follow', filename])
            if ret[0] != 0:
                Aqf.failed('Could not get the git log of the file ' + filename)
            Aqf.hop("{}: Showing commits for file {}".
                     format(description,
                            os.path.join(self._git_path, filename)))
            self._print_git_log(ret[1].splitlines())
            Aqf.checkbox("Verify from the commits and git log that file {} has been automaticaly updated.".
                         format(filename))
        Aqf.hop("Leap seconds for UT1: CAM do not manage leap seconds")
        Aqf.checkbox("CAM updates catalogues automatically")
        Aqf.end()

    def _print_git_log(self, log, limit=10):
        counter = int(limit)
        for n in log:
            counter -= 1
            Aqf.hop(n)
            if counter < 0:
                break

    @aqf_vr('VR.CM.AUTO.CFG.40')
    @aqf_vr('VR.CM.AUTO.CFG.55')
    @aqf_vr('VR.CM.AUTO.CFG.59')
    def test_user_select_instrumental_configuration_data_version(self):
        """VR.CM.AUTO.CFG.40: Test user select instrumental configuration data version
        Description: Verify that CAM applies the selected version of the
        Instrumental Configuration Data when observing with a subarray
        demonstrated in VR.CM.DEMO.CFG.51 R.CM.FC.23: User select instrumental
        configuration data version
        Description: CAM shall enable the Control Authority to select a version
        of Instrumental Configuration Data from the Instrumental Configuration
        Data Repository to use with a subarray.

        VR.CM.AUTO.CFG.55: Test CAM associates instrumental configuration data
        version with each SB
        Description: Verify that CAM links a schedule block with the version of
        Instrumental Configuration Data that is valid at the time of executing a
        schedule block.
        R.CM.FC.51: Associate instrumental configuration data version with each
        SB
        Description: CAM shall associate each SB with the version of the
        Instrumental Configuration Data that was valid at the time of executing
        the SB.
        """

        # Get the current config label
        start_config_label = self.cam.sys.req.list_config_label(1).informs[0]
        start_config_label = start_config_label.arguments[0]

        # Create a new head.
        readme = os.path.join(self._git_path, 'README.rst')
        Aqf.is_true(os.path.isfile(readme),
                    "Ensure katconfig has a readme file.")
        with open(readme, 'a') as fh:
            fh.write("hi")

        self._git(['add', readme])
        self._git(['commit', '-m', "mod readme to trigger a new config_label"])

        try:
            config_labels = self._test_sb_on_subarray('')
        except Exception as excpt:
            Aqf.failed(excpt)

        self.validate_list_of_config_labels(config_labels)

        Aqf.is_true(config_labels[0] != start_config_label,
                    "Confirm that a new config_label was created.")
        if config_labels:
            run_with_config_label = config_labels[0]

            try:
                config_labels = self._test_sb_on_subarray(run_with_config_label)
            except Exception as excpt:
                Aqf.failed(str(excpt))

            self.validate_list_of_config_labels(config_labels)
        else:
            Aqf.failed("could not generate config_labels")

        Aqf.end()

    def validate_list_of_config_labels(self, config_labels):
        if not config_labels:
            Aqf.failed("No config_labels.")
            return

        Aqf.hop("config_labels {}".format(config_labels))
        ref_config_label = config_labels[0]
        if not ref_config_label:
            Aqf.failed("No reference config_label.")
            return

        if not all([cl == ref_config_label for cl in config_labels if cl]):
            Aqf.failed("Mismatch in config_label. {}".format(config_labels))
            return

        # This step is a bit specific and is based on the append sequence of
        # the test.
        expected_cls = [ref_config_label, '', '', '', ref_config_label,
                        ref_config_label]

        Aqf.equals(config_labels, expected_cls, "Check the config label value "
                   "at different steps of creating an SB.")

    def _test_sb_on_subarray(self, config_label):
        """Test Running a Schedule block on a subarray using the given
        config_label."""
        sub_nr = 3
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        subarray.req.free_subarray()
        Aqf.step("Create a subarray")
        receptor = self.cam.ants[0].name
        subarray.req.assign_resources(receptor)
        subarray.req.set_band('l')
        subarray.req.set_config_label(config_label)
        Aqf.step("Activate the subarray")
        Aqf.is_true(subarray.req.activate_subarray(timeout=15).succeeded,
                    "Activation request for subarray {} successful".
                    format(sub_nr))
        Aqf.progress('Waiting for subarray {} to be active'.format(sub_nr))
        ok = Aqf.sensor(subarray.sensor.state).wait_until(
            "active", sleep=1, counter=5)
        if not ok:
            msg = "Subarray {} is not active, aborting test".format(sub_nr)
            Aqf.failed(msg)
            subarray.req.free_subarray()
            return

        config_labels = [subarray.sensor.config_label.get_value()]
        # CREATE SB
        Aqf.step("Create a basic-script")
        sb_id = self._create_obs_sb(receptor, '',
                                    "CAM_basic_script", runtime=10)
        config_labels.append(self._sb_config_label(sb_id))
        Aqf.step("Set scheduler to manual")
        Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                    "Set scheduler to Manual Scheduling Mode.")
        # ASSIGN SB
        Aqf.hop("Assign newly created sb %s to subarray %s" %
                (sb_id, sub_nr))
        subarray.req.assign_schedule_block(sb_id)
        config_labels.append(self._sb_config_label(sb_id))

        # SCHEDULE SB
        Aqf.hop('Schedule sb {}'.format(sb_id))
        self.schedule_and_check_pass(sub_nr, sb_id)
        config_labels.append(self._sb_config_label(sb_id))

        # EXECUTE SB
        Aqf.hop('Execute sb {}'.format(sb_id))
        self.execute_and_check_pass(sub_nr, sb_id)
        config_labels.append(self._sb_config_label(sb_id))
        # self.wait_sb_complete(sub_nr, sb_id)
        subarray.req.free_subarray()
        config_labels.append(self._sb_config_label(sb_id))
        return config_labels

    def _create_obs_sb(self, selected_ant, controlled,
                       program_block, runtime=180):
        """
        Parameters
        ----------
        selected_ant: str
            CSV of the selected receptors
        controlled: str
            CSV of the controlled components
        program_block: str
            Name of the program block
        runtime: int
            Duration of the SB.

        Returns
        -------
        str
            Schedule block id of the newly created schedule block.
        """
        sb_id_code = self.obs.sb.new(owner="aqf-test",
                                     antenna_spec=selected_ant,
                                     controlled_resources=controlled)
        self.obs.sb.description = "Track for %s" % selected_ant
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        self.obs.sb.instruction_set = (
            "run ~/scripts/observation/auto_attenuate.py")

        return sb_id_code

    def schedule_and_check_pass(self, sub_nr,
                                schedule_block_id_code, timeout=30):
        """Test for success of scheduling a schedule block."""
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        self.obs.sb.load(schedule_block_id_code)

        reply = subarray.req.sb_schedule(schedule_block_id_code)
        Aqf.equals(reply.succeeded, True, 'Verify schedule request succeeded '
                   'for schedule block %s.' % schedule_block_id_code)

        schedule_block = self.obs.sb
        Aqf.equals(schedule_block.type, ScheduleBlockTypes.OBSERVATION,
                   "Check SB type.")

        reply = subarray.req.sb_schedule(schedule_block_id_code)
        Aqf.equals(reply.succeeded, True, 'Verify schedule request succeeded '
                   'for schedule block %s.' % schedule_block_id_code)

        self._wait_for_sb_state(schedule_block_id_code,
                                ScheduleBlockStates.SCHEDULED)

    def execute_and_check_pass(self, sub_nr,
                               schedule_block_id_code, timeout=30):
        """Test for success of executing a schedule block."""
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        self.obs.sb.load(schedule_block_id_code, allow_load=True)

        reply = subarray.req.sb_execute(schedule_block_id_code)
        Aqf.equals(reply.succeeded, True, 'Verify execute request succeeded '
                   'for schedule block %s.' % schedule_block_id_code)

        self._wait_for_sb_state(schedule_block_id_code,
                                ScheduleBlockStates.COMPLETED)

    def _wait_for_sb_state(self, schedule_block_id_code, schedule_block_state):
        counter = 100
        while counter > 0:
            counter -= 1
            self.obs.sb.load(schedule_block_id_code, allow_load=True)
            self.obs.sb
            if self.obs.sb.state == schedule_block_state:
                counter = 0
            else:
                time.sleep(2)
                Aqf.progress("Waiting for SB {} to be completed.".
                             format(schedule_block_id_code))

        Aqf.equals(self.obs.sb.state, schedule_block_state,
                   'Verify schedule block {} is {}.'.
                   format(schedule_block_id_code, schedule_block_state.key))

    def _sb_config_label(self, schedule_block_id_code):
            self.obs.sb.load(schedule_block_id_code, allow_load=True)
            return getattr(self.obs.sb, 'config_label', None)

    def _git(self, cmd):
        if self._git_path is None:
            return 99, '', 'No Katconfig repo selected.'
        return git(self._git_bin, self._git_path, cmd)
