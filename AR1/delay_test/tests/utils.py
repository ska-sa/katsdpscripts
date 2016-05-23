#!/usr/bin/env python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################


"""Utility functions to support KAT system tests.
"""

import traceback
import StringIO
import sys
import subprocess
import paramiko
from datetime import datetime

from katmisc.utils import csv_sets
from tests import settings, fixtures, specifics, Aqf, wait_sensor
from katcorelib.katobslib.common import ScheduleBlockTypes, ScheduleBlockPriorities

def trap_cmd_output(cmd):
    # Trap output.
    stdout_trap = StringIO.StringIO()
    # manually redirect stdout
    sys_stdout = sys.stdout
    try:
        sys.stdout = stdout_trap
        exec(cmd)
    finally:
        sys.stdout = sys_stdout

    output = stdout_trap.getvalue()
    output = output.rstrip()
    return output

def trap_python_output(cmd):
    # Trap output.
    stdout_trap = StringIO.StringIO()
    # manually redirect stdout
    sys_stdout = sys.stdout
    try:
        sys.stdout = stdout_trap
        eval(cmd)
    finally:
        sys.stdout = sys_stdout

    output = stdout_trap.getvalue()
    output = output.rstrip()
    return output

def ssh_server(ip):
    """ssh to a sever"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username='kat', password='kat')

    return ssh

def execute(sparams):
    """Execute script with script parameters.

    Parameters
    ----------
    sparams: string
        Script name and parameters is a space separated string of parameters

    Return:
    -------
    res: boolean
        True or False, indicating success or not
    results: tuple
        Tuple of result records (output of process.communicate)
    """

    try:
        xparams = []
        for i in sparams.split(" "):
            xparams.append(i)
        process = subprocess.Popen(xparams, shell=False, stdout=subprocess.PIPE)
        result = process.communicate()
        return True, result
    except Exception, err:
        return False, ("Error executing (%s) \n%s" % (err, traceback.format_exc()), None)

def check_process_in_ps(process_name):
    """check that the process is running using 'ps aux'

    Parameters
    ----------
    process_name : string
        Name of process to check

    Returns:
    --------
    res, msg
    """
    script = "ps auxww"
    msg = "Process not found"
    res, results = execute(script)
    if res == True:
        for result in results:
            if result == None:
                continue
            for line in result.splitlines():
                if line.find("grep") != -1:
                    continue
                elif line.find(process_name) != -1:
                    res = True
                    msg = "Ok"
                    break
    else:
        msg = results[0]
    if msg != 'Ok':
        res = False
    return res, msg


# The functions below are are general
# utility functions that can be used by each system

def parse_timestamp(date_string):
    tformat = "%Y-%m-%d %H:%M:%S.%fZ"
    utc_dt = datetime.strptime(date_string, tformat)
    return utc_dt


def check_and_get_utctime_of_log_for(aqfbase, log_name, grep_strings,
                         aftertime=None, lines=50):
    """Find the time at which the strings in grep string occurs at the end of the log.

    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param log_name: String. Name of log file.
    :param grep_strings: List of strings.
    :param aftertime: Datetime Object. The logs must be from
                      after this time.
    :param lines: Int.
    :return: Boolean. True if grep_strings was in the end of the file.
    :return: time. The timestamp of the found log, or None if not found.

    """
    result = aqfbase.cam.katlogserver.req.tail_log(log_name, lines, timeout=60)
    output = reversed(result.informs)

    if not isinstance(grep_strings, list):
        grep_strings = [str(grep_strings)]
    found = False
    time_str = ''
    log_time = None
    for msg in output:
        line = msg.arguments[0]
        for match in grep_strings:
            if match in line:
                continue
            break
        else:
            # We will end of here on the first match
            time_str = line.split()[:2]
            log_time = parse_timestamp(' '.join(time_str))
            found = True

    if aftertime and found:
        log_time = parse_timestamp(' '.join(time_str))
        if log_time < aftertime:
            found = False

    return found, log_time


def check_end_of_log_for(aqfbase, log_name, grep_strings,
                         aftertime=None, lines=50):
    """Check that the strings in grep string occurs at the end of the log.

    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param log_name: String. Name of log file.
    :param grep_strings: List of strings.
    :param aftertime: Datetime Object. The logs must be from
                      after this time.
    :param lines: Int.
    :return: Boolean. True if grep_strings was in the end of the file.

    """
    found, log_time = check_and_get_utctime_of_log_for(aqfbase, log_name, grep_strings,
                         aftertime, lines)

    return found

def check_alarm_severity(aqfbase, alarm_name, expected_severity):
    """Wait for a while and check that alarm has specific severity.

    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param alarm_name: String. Name of alarm
    :param expected_priority: expected_priority of alarm
    :return: Boolean. True if grep_strings was in the end of the file.
    """

    Aqf.step("Verify that alarm %s severity is %s" %
             (alarm_name, expected_severity.upper()))
    try:
        count = 0
        severity = ""
        while count < 4 and not severity == expected_severity:
            Aqf.wait(3, "Waiting for kataware to process the alarm")
            sensor_obj = getattr(aqfbase.cam.kataware.sensor,
                                 "alarm_" + alarm_name)
            alarm_value = sensor_obj.get_value()
            severity, priority, message = alarm_value.split(",", 3)
            Aqf.progress("Alarm %s - severity %s" % (alarm_name, severity))
            count += 1
        if severity == expected_severity:
            Aqf.passed("Alarm %s now has expected severity %s" %
                       (alarm_name, expected_severity))
            return True
        else:
            Aqf.failed("Alarm %s does not have severity %s. Severity is %s" %
                       (alarm_name, expected_severity, severity))
            return False
    except Exception, err:
        tb = traceback.format_exc()
        Aqf.failed("EXCEPTION checking alarm severity: %s \n %s" %
                   (str(err), tb))
        return False

def check_alarm_priority(aqfbase, alarm_name, expected_priority):
    """Check that alarm has expected priority.
    
    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param alarm_name: String. Name of alarm
    :param expected_priority: expected_priority of alarm
    :return: Boolean. True if alarm has expected priority
    """

    Aqf.step("Verify that alarm %s priority is  %s" %
             (alarm_name, expected_priority.upper()))
    try:
        count = 0
        priority = ""
        while count < 4 and not priority == expected_priority:
            Aqf.wait(6, "Waiting for kataware to process the alarm")
            sensor_obj = getattr(aqfbase.cam.kataware.sensor,
                                 "alarm_" + alarm_name)
            alarm_value = sensor_obj.get_value()
            severity, priority, message = alarm_value.split(",", 3)
            Aqf.passed("Alarm %s - priority %s" % (alarm_name, priority))
            count += 1
        if priority == expected_priority:
            Aqf.passed("Alarm now %s has expected priority %s" %
                       (alarm_name, expected_priority))
            return True
        else:
            Aqf.failed("Alarm %s does not have priority %s. Priority is %s" %
                       (alarm_name, expected_priority, priority))
            return False
    except Exception, err:
        tb = traceback.format_exc()
        Aqf.failed("EXCEPTION checking alarm priority: %s \n %s" %
                   (str(err), tb))
        return False

def check_activity_logged(aqfbase, grep_for, aftertime=None, lines=50):
    """Check that the activity log occurs at the end of the activity.log file.

    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param grep_for: Strings to search for in the log
    :param aftertime: Datetime Object. The logs must be from
                      after this time.
    :param lines: Int.
    :return: Boolean. True if grep_for was found in the end of the file.
    """

    Aqf.wait(2, "Wait for activity log to be processed")
    Aqf.step("Verify that activity for %s has been logged" % grep_for)
    found, utctime = check_and_get_utctime_of_log_for(aqfbase, 'activity', grep_for,
                aftertime = aftertime, lines = lines)
    return found, utctime

def check_alarm_logged(aqfbase, alarm_name, expected_priority, aftertime = None, lines=50):
    """Check that the alarm log occurs at the end of the alarms.log file.

    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param alarm_name: String. Name of alarm
    :param expected_priority: expected_priority of alarm
    :param aftertime: Datetime Object. The logs must be from
                      after this time.
    :param lines: Int.
    :return: Boolean, time. True if grep_strings was in the end of the file, with utctime of the log
    """

    Aqf.wait(2, "Wait for alarm log to be processed")
    Aqf.step("Verify that alarm %s has been logged at priority %s" % (alarm_name, expected_priority))
    grep_for = [alarm_name, expected_priority+" alarm"]
    found, utctime = check_and_get_utctime_of_log_for(aqfbase, 'alarms', grep_for,
                aftertime = aftertime, lines = lines)
    return found, utctime

def create_basic_obs_sb(aqfbase, ant_spec, controlled, owner="aqf-test"):
    """Helper function to create an OBservation SB using track.py - always for subarray 1
    
    :param aqfbase: AqfTestCase object with cam, sim and obs attributes.
    :param ant_spec: String. Antenna spec.
    :return: String. sb_id_code
    """
    sb_id_code = aqfbase.obs.sb.new(owner=owner, antenna_spec=ant_spec,
                                    controlled_resources=controlled)
    aqfbase.obs.sb.description = "Basic script for %s" % ant_spec
    aqfbase.obs.sb.type = ScheduleBlockTypes.OBSERVATION
    aqfbase.obs.sb.instruction_set = "run-obs-script ~/scripts/cam/basic-script.py -t 5.0 -m 180 --program-block-id=CAM_AQF"
    return sb_id_code

def check_scheduler_ready(aqfbase, subnr):
    sens_obj = getattr(aqfbase.cam.sched.sensor, "mode_%s" % subnr)
    sched_mode = sens_obj.get_value()
    return sched_mode != "locked"

def check_scheduler_locked(aqfbase, subnr):
    sens_obj = getattr(aqfbase.cam.sched.sensor, "mode_%s" % subnr)
    sched_mode = sens_obj.get_value()
    return sched_mode == "locked"

def setup_subarray(aqfbase, sub_nr, controlled=None):
    Aqf.step("Setup subarray_{}".format(sub_nr))
    sub_obj = getattr(aqfbase.cam, "subarray_{}".format(sub_nr))
    ants_set = set([ant.name for ant in aqfbase.cam.ants])
    # Only use ants that are OK and not in-maintenance and free
    ok_ants_set = ants_set.intersection(
                    set(aqfbase.cam.katpool.sensor.resources_ok.get_value().split(",")))
    avail_ants_set = ok_ants_set.difference(
                    set(aqfbase.cam.katpool.sensor.resources_in_maintenance.get_value().split(",")))
    chosen_ants_set = avail_ants_set.intersection(
                    set(aqfbase.cam.katpool.sensor.pool_resources_free.get_value().split(",")))
    if controlled:
        # Add the specific controlled resource
        specific_controlled = specifics.get_specific_controlled(aqfbase, sub_nr=sub_nr, controlled=controlled)
        res_csv = ",".join(chosen_ants_set.union([specific_controlled]))
    else:
        res_csv = ",".join(chosen_ants_set)
    Aqf.step("Assign available ants and controlled resources '{}' to subarray_{}".format(res_csv, sub_nr))
    sub_obj.req.assign_resources(res_csv)
    return chosen_ants_set

def setup_single_ant_subarray(aqfbase, sub_nr, selected_ant):
    """Setup subarray"""
    controlled = specifics.get_controlled_data_proxy(aqfbase, sub_nr)
    specific_controlled = specifics.get_specific_controlled(aqfbase, sub_nr=sub_nr, controlled=controlled)
    sub = getattr(aqfbase.cam, 'subarray_%s' % sub_nr)
    
    # Free and setup subarray
    Aqf.step("Free subarray %s to start with" % sub_nr)
    sub.req.free_subarray(timeout=30)
    Aqf.sensor(sub.sensor.state).wait_until("inactive", sleep=1, counter=5)
    resource_csv = ",".join([selected_ant, specific_controlled])
    Aqf.step("Assign resources %s to subarray %s" % (resource_csv, sub_nr))
    sub.req.assign_resources(resource_csv)
    
    # Activate subarray and set manual scheduling
    Aqf.step("Activate the subarray.")
    sub.req.activate_subarray(timeout=100)
    Aqf.sensor(aqfbase.cam.sensors.subarray_1_state).wait_until("active", sleep=1, counter=5)
    sub.req.set_scheduler_mode("manual")

def teardown_subarray(aqfbase, sub_nr):
    Aqf.step("Teardown subarray - free subarray_{}".format(sub_nr))
    sub_obj = getattr(aqfbase.cam, "subarray_{}".format(sub_nr))
    sub_obj.req.free_subarray(timeout=30)

def get_next_avaivable_rec(aqfbase):
    """Get next a"""
    free = aqfbase.cam.katpool.sensor.pool_resources_free.get_value()
    faulty = aqfbase.cam.katpool.sensor.resources_faulty.get_value()
    maint = aqfbase.cam.katpool.sensor.resources_in_maintenance.get_value()
    _all = [i for i in free.split(',') if i not in faulty.split(',') and i not in maint.split(',')]
    rec = [i for i in _all if i.startswith('m0')]
    if rec:
        return rec[0]
    else:
        Aqf.step("No resources that is free")
        return None

def create_sb(aqfbase):
    """Create schedule block"""
    selected_ant = get_next_avaivable_rec(aqfbase)
    Aqf.step("Create schedule block.")
    # Get other active schedule blocks
    sub_nr = 1
    controlled = specifics.get_controlled_data_proxy(aqfbase, sub_nr)
    active_sbs = aqfbase.cam.sched.sensor.active_schedule_1.get_value()
    sb_id_code = aqfbase.obs.sb.new(owner="AQF-test_demo", antenna_spec=selected_ant,
                                    controlled_resources=controlled)
    aqfbase.obs.sb.description = "Basic script for %s" % selected_ant
    aqfbase.obs.sb.type = ScheduleBlockTypes.OBSERVATION
    aqfbase.obs.sb.instruction_set = "run-obs-script ~/scripts/cam/basic-script.py -t 5.0 -m 60 --program-block-id=CAM_AQF"
    Aqf.passed('Basic script schedule block created %s.' % sb_id_code)

    return (sb_id_code, selected_ant)

def assign_subarray(aqfbase, sub_nr, sb_id_code):
    sub = getattr(aqfbase.cam, 'subarray_%s' % sub_nr)
    sub.req.assign_schedule_block(sb_id_code)
    sub.req.sb_schedule(sb_id_code)
    
def check_sb_ready(aqfbase, sub_nr, sb_id_code):
    sensor = getattr(aqfbase.cam.sched.sensor, 'observation_schedule_%s' % sub_nr)
    wait_sensor_includes(aqfbase.cam, aqfbase.cam.sched.sensor.observation_schedule_1, sb_id_code, 300)
    time.sleep(3)
    aqfbase.obs.db_manager.expire()
    sb = aqfbase.obs.db_manager.get_schedule_block(sb_id_code)
    aqfbase.assertEqual(sb.ready, True)
    Aqf.passed('Schedule block is READY %s.' % sb_id_code)


def clear_all_subarrays_and_schedules(
        aqfbase, msg="Clear all subarrays and schedules"):
    """Clear all subarrays:
       clear in-use/maintenance/in-maintenance/faulty flags for subarrays and resources
       Clear the observation schedule for each subarray
       Also check if any allocations still exist - may need to force out the active SBs"""

    Aqf.step(msg)
    subarrays = [sub for sub in
                 aqfbase.cam.katpool.sensor.subarrays.get_value().split(",")]
    Aqf.step("Free and unflag all subarrays")
    for sub_nr in subarrays:
        sub_obj = getattr(aqfbase.cam, "subarray_%s" % sub_nr)
        sub_obj.req.free_subarray(timeout=30)
        sub_obj.req.set_subarray_maintenance(False)

    Aqf.step("Unflag all resources in katpool")
    resource_str = aqfbase.cam.katpool.sensor.pool_resources_free.get_value()
    aqfbase.cam.katpool.req.set_resources_in_maintenance(resource_str, 0)
    aqfbase.cam.katpool.req.set_resources_faulty(resource_str, 0)

    Aqf.step("Clear all Observation Schedules")
    for sub_nr in subarrays:
        clear_observation_schedule(aqfbase, sub_nr)

    Aqf.step("Set schedulers to 'idle'")
    for sub_nr in subarrays:
        aqfbase.cam.sched.req.mode(sub_nr, 'idle')

    # Wait for the actions to be processed
    all_ok = True
    ok = Aqf.sensor("cam.katpool.sensor.resources_in_maintenance").wait_until("", sleep=1, counter=5)
    if not ok:
        value = aqfbase.cam.katpool.sensor.resources_in_maintenance.get_value()
        msg = "Katpool still have resources-in-maintenance ({})".format(value)
        Aqf.progress("FAILED - "+msg)
        all_ok = False

    for sub_nr in subarrays:
        # Check subarray pool resources
        ok = Aqf.sensor("cam.katpool.sensor.pool_resources_{}".format(sub_nr)).wait_until("", sleep=1, counter=5)
        if not ok:
            sens_obj = getattr(aqfbase.cam.katpool.sensor, "pool_resources_%s" % sub_nr)
            value = sens_obj.get_value()
            msg = "Subarray {} has pool_resources {}".format(sub_nr, value)
            Aqf.progress("FAILED - "+msg)
            all_ok = False

        # Check subarray allocations
        ok = Aqf.sensor("cam.katpool.sensor.allocations_{}".format(sub_nr)).wait_until("[]", sleep=1, counter=5)
        if not ok:
            sens_obj = getattr(aqfbase.cam.katpool.sensor, "allocations_%s" % sub_nr)
            value = sens_obj.get_value()
            msg = "Subarray {} has allocations {}".format(sub_nr, value)
            Aqf.progress("FAILED - "+msg)
            all_ok = False

        # Check subarray state
        ok = Aqf.sensor("cam.subarray_{}.sensor.state".format(sub_nr)).wait_until("inactive", sleep=1, counter=15)
        if not ok:
            comp_obj = getattr(aqfbase.cam, "subarray_%s" % sub_nr)
            value = comp_obj.sensor.state.get_value()
            msg = "cam.subarray_{}.sensor.state is not inactive - it is {}".format(sub_nr, value)
            Aqf.progress("FAILED - "+msg)
            all_ok = False
 
    # Use Aqf.step here as it will mostly be called in test setup and teardown
    if all_ok:
        Aqf.step("Clear all subarrays done")
    else:
        Aqf.step("Clear all subarrays FAILED")

    return all_ok

def clear_observation_schedule(aqfbase, sub_nr):
    """Clear observation schedule for specified subarray"""
    Aqf.step("Clear observation schedule for subarray {}".format(sub_nr))
    # Get observation schedule from obs as sched sensors does not report the
    # assigned SBs unless the subarray is active
    records = aqfbase.obs.get_observation_schedule(sub_nr=sub_nr)
    for rec in records[1]:
        # Tuple with sbid the first item
        sb_id = rec[0]
        Aqf.is_true(aqfbase.cam.sched.req.sb_to_draft(sub_nr, sb_id).succeeded,
                    'Set sb {} to draft on subarray {}'.format(sb_id, sub_nr))
    sens_obj = getattr(aqfbase.cam.sched.sensor, 'observation_schedule_{}'.format(sub_nr))
    Aqf.hop("Wait for sched.observation_schedule_{} to empty".format(
        sub_nr))
    Aqf.sensor("cam.sched.sensor.observation_schedule_{}".
               format(sub_nr)).wait_until('', sleep=1, counter=30)

def create_obs_sb(aqfbase, antenna_spec, controlled,
                   program_block, runtime=180, owner="aqf-test"):
    sb_id_code = aqfbase.obs.sb.new(owner=owner,
                                 antenna_spec=antenna_spec,
                                 controlled_resources=controlled)
    aqfbase.obs.sb.description = "Track for %s" % antenna_spec
    aqfbase.obs.sb.type = ScheduleBlockTypes.OBSERVATION
    aqfbase.obs.sb.instruction_set = (
        "run-obs-script ~/scripts/cam/basic-script.py -t 3 -m {runtime} "
        "--proposal-id=CAM_AQF --program-block-id={program_block}".
        format(**locals()))
    return sb_id_code

def create_manual_sb(aqfbase, antenna_spec, controlled, owner="aqf-test"):
    sb_id_code = aqfbase.obs.sb.new(owner=owner,
                                 antenna_spec=antenna_spec,
                                 controlled_resources=controlled)
    aqfbase.obs.sb.description = "Manual SB for %s" % antenna_spec
    aqfbase.obs.sb.type = ScheduleBlockTypes.MANUAL
    return sb_id_code

def setup_default_subarray(aqfbase, sub_nr, antenna_spec='default', controlled='default', activate=False):
    """Set up a default subarray spec for specified subarray number.
    By default it will select a single antenna and
    the controlled resource (e.g. data_n or dbe7) for subarray_n.
    Optionally also activates the subarray if activate=True

    Returns ants_csv, controlled_csv
    """
    if antenna_spec=='default':
        #Select a single ant
        ants_csv = aqfbase.cam.ants[0].name
    else:
        ants_csv = antenna_spec if antenna_spec else ""
    if controlled=='default':
        controlled_csv = specifics.get_controlled_data_proxy(aqfbase, sub_nr)
    else:
        controlled_csv = controlled if controlled else ""
    sub_obj = getattr(aqfbase.cam, "subarray_{}".format(sub_nr))
    Aqf.step("Assigning {} and {} to subarray {}".format(ants_csv, controlled_csv, sub_nr))
    if controlled_csv:
        sub_obj.req.assign_resources(controlled_csv)
    if ants_csv:
        sub_obj.req.assign_resources(ants_csv)

    if activate:
        # Activating subarray
        Aqf.step("Activating subarray {} ".format(sub_nr))
        Aqf.is_true(sub_obj.req.activate_subarray(timeout=100).succeeded,
                    "Activation request for subarray {} successful".
                    format(sub_nr))

    return ants_csv, controlled_csv



