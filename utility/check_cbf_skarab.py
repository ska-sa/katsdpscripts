#!/usr/bin/python

import time
import string
import sys
from katcorelib import standard_script_options, verify_and_connect, user_logger
from katcorelib import tbuild, kat_resource
from katmisc.utils.ansi import colors, get_sensor_colour, gotoxy, clrscr, col, getKeyIf
from katmisc.utils.utils import get_time_str, escape_name

import numpy as np
import os
import katcp
import smtplib
from email.mime.text import MIMEText



email_msg = []
MIN_OP_TEMP = 16


def log_timestamp():
    # create log timestamp format
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def log_message(msg, level='info', boldtype=False, colourtext='black'):

    bold = boldtype
    colour = colourtext

    if level == 'debug':
        user_logger.debug(str(msg))
    elif level == 'info':
        user_logger.info(str(msg))
    elif level == 'warn':
        user_logger.warn(str(msg))
        colour = 'orange'
        bold = True
    elif level == 'error':
        user_logger.error(str(msg))
        colour = 'red'
        bold = True

    if not bold and colour == 'black':
        email_msg.append(log_timestamp() + level.upper() + ' ' + str(msg))
    elif colour != 'black' and not(bold):
        email_msg.append('<font color="{colour}">{timestamp} {level} {msg}</font>'.format(
            colour=colour, timestamp=log_timestamp(), level=level.upper(), msg=str(msg)))
    else:
        email_msg.append('<font color="{colour}"><b>{timestamp} {level} {msg}</b></font>'.format(
            colour=colour, timestamp=log_timestamp(), level=level.upper(), msg=str(msg)))


def fetch_archive_sensor_data(sensorname, starttime, endtime, data_type,
                              server='portal.mkat.karoo.kat.ac.za'):
    """A procedure to fetch sensor data from a server"""
    filename = '/tmp/{}_data.csv'.format(sensorname)
    res = os.system(
        'wget -q "http://{server}/katstore/samples?sensor={sensorname}'
        '&start={start}&end={end}&limit=1000000&time_type=s&format=csv" '
        '-O {filename}'.format(server=server, sensorname=sensorname,
                               start=int(starttime), end=int(endtime),
                               filename=filename))
    if res:
        log_message(
            'Error getting CSV for sensor {}, perhaps reset the HDF server process on CAM'
            .format(sensorname), 'warn')
    try:
        # read CSV
        return np.genfromtxt(filename, dtype=data_type, delimiter=',', names=True)
    except:
        log_message('Error retrieving sensor data from {}'.format(server), 'error')
        return np.array([])

def read_sensor_history(ants):
    # read through sensor history and report when vac pump last lubricated and at what elevation
    for ant in ants:
        try:
            vac_running = []
            vac_running = fetch_archive_sensor_data(
                '{}_rsc_rsc_vac_pump_running'.format(ant),
                time.time() - (60 * 60 * 24 * (opts.archive_search)),
                time.time(), data_type="S100,f8,Bool", server=server)
            if len(vac_running) != 0:
                vac_running_true = np.where(vac_running['Value'])[0]
                groups = group_consecutives(vac_running_true)
                vacpump_run_duration = []
                vacpump_run_timestamp = []
                vacpump_elevation_dict = dict()

                for x in groups:
                    if x != []:
                        vacpump_run_duration.append(
                            (vac_running['Timestamp'][x[-1] + 1] -
                             vac_running['Timestamp'][x[0]]) / 60.0)
                        vacpump_run_timestamp.append(
                            vac_running['Timestamp'][x[-1] + 1])
                if vacpump_run_timestamp:
                    log_message('{} last test completed : '.format(ant), 'info')
                    for i, x in enumerate(vacpump_run_timestamp):
                        # Only print elevation information if it's relevant because it takes time to look up
                        if ((vacpump_run_duration[i] >= opts.run_duration) and
                                (int((time.time() - x) / 3600 / 24) <= opts.lubrication_frequency)):
                            elev_vals = fetch_archive_sensor_data(
                                '{}_ap_actual_elev'.format(ant),
                                x - (vacpump_run_duration[i] * 60),
                                x, data_type="S100,f8,f8", server=server)
                            vacpump_elevation_dict[x] = np.mean(elev_vals['Value'])
                            log_message('\t{} minutes duration,\t {} days ago \t {:0.2f} Average Elevation'
                                        .format(int(vacpump_run_duration[i]),
                                                int((time.time() - x) / 3600 / 24),
                                                vacpump_elevation_dict[x]))
                        else:
                            log_message('\t{} minutes duration,\t {} days ago'
                                        .format(int(vacpump_run_duration[i]),
                                                int((time.time() - x) / 3600 / 24)))
                    vacpump_recent_runs = np.where(
                        np.array(vacpump_run_timestamp) >=
                        time.time() - (60 * 60 * 24 * (opts.lubrication_frequency)))[0]
                    need_to_run = True
                    for x in vacpump_recent_runs:
                        if ((vacpump_run_duration[x] >= opts.run_duration) and
                                (vacpump_elevation_dict[vacpump_run_timestamp[x]] <= (opts.max_elevation + 1))):
                            need_to_run = False
                            break
                    if need_to_run:
                        log_message(
                            '{} - No runs '
                            'long enough below specified elevation\n'.format(ant))
                    else:
                        log_message(
                            '{} - Vacuum Pump Lubrication not required\n'.format(ant), boldtype=True)
                else:
                    log_message(
                        '{} - No record over the last {} days\n'
                        .format(ant, opts.archive_search))
            else:
                log_message(
                    '{} - Unable to extract '
                    'sensor data from archive\n'.format(ant), 'warn')
        except:
            log_message(
                '{} - Error reading and processing sensor data.'.format(ant), 'error')


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="MeerKAT SKARAB CBF error check")
parser.add_option("--subnr", type=int, default=1,
                  help="Subarray to check. (default='%default')")

# assume basic options passed from instruction_set
parser.set_defaults(description = "CBF SKARAB error check")
(opts, args) = parser.parse_args()

if opts.subnr is None:
    raise SystemExit("Subarray number required %s" % parser.print_usage())
else:
    subnr = str(opts.subnr)

notes = """
cbfmon_1_i0_hostname_functional_mapping 
{'skarab02000-01': 'fhost00', 'skarab02007-01': 'xhost03', 'skarab02006-01': 'fhost03', 'skarab02005-01': 'xhost02', 'skarab02003-01': 'xhost01', 'skarab02002-01': 'fhost01', 'skarab02004-01': 'fhost02', 'skarab02001-01': 'xhost00'}

cbfmon_1_i0_input_labelling
[('m001h', 0, 'board020900', 0), ('m001v', 1, 'board020900', 1), ('m008h', 2, 'board020901', 0), ('m008v', 3, 'board020901', 1), ('m018h', 4, 'board020902', 0), ('m018v', 5, 'board020902', 1), ('m028h', 6, 'board020903', 0), ('m028v', 7, 'board020903', 1)]

"""

with verify_and_connect(opts) as kat:
    log_message("CBF SKARAB error check: start\n", boldtype=True,)
    log_message("Opts:\n{}\n".format(opts))
    print("=========================================")
    print("CBF SKARAB error check : start")
    print("=========================================")
    print(opts)
    print(kat.controlled_objects)
    print(kat.ants.clients)
    print("----------------------------")

    subarrays = kat.katpool.sensor.subarrays.get_value()
    subarrays_free = kat.katpool.sensor.subarrays_free.get_value()
    # Don't check this as the subarray-free and subarrays-in-use sensors on katpool is not updated correctly
    # assert subnr not in subarrays_free, "The requested subarray {} is free - no point running this script.".format(subnr)
    ants = kat.katpool.sensor.ants.get_value().split(",")
    sub = getattr(kat, "subarray_{}".format(subnr))
    sub_pool = getattr(kat.katpool.sensor, "pool_resources_{}".format(subnr)).get_value().split(',')
    # Get subarray in subarray
    sub_ants = [res for res in ants if res in sub_pool]
    sub_state = sub.sensor.state.get_value()
    print("----------------------------")
    print("Subarray: {}".format(subnr))
    print("   state: {}".format(sub_state))
    print("    ants: {}".format(sub_ants))
    print("    pool: {}".format(sub_pool))
    print(kat.ants.clients)
    print(opts)
    print("----------------------------")

    if opts.dry_run:
        log_message('Dry Run only\n')
        print("\n=========================================")
        print("CBF SKARAB error check : Dry-run")
        print("=========================================")
        exit(0)
        
    # Build the 2nd KAT object. This kat object is being used to access sensor data
    # for resources outside the subarray.
    log_message('Begin tbuild...')
    kat2 = tbuild(conn_clients='cbfmon_{}'.format(subnr))
    cbf = getattr(kat, "cbf_{}".format(subnr))
    cbfmon = getattr(kat2, "cbfmon_{}".format(subnr))
    obj = cbfmon

    log_message("Waiting for cbfmon_{} to sync\n".format(subnr))
    cbf_ok = cbf.until_synced(timeout=15)
    cbfmon_ok = cbfmon.until_synced(timeout=60)

    if not (cbf_ok and cbfmon_ok):
        log_message("Some resources did not sync \n kat.cbf_{}={} kat2.cbfmon{_{}={}\n{}\n\n"
                    .format(subnr, cbf_ok, subnr, cbfmon_ok, kat2.get_status()), 'error')
        log_message("Aborting script", 'error')
        raise RuntimeError(
            "Aborting - Some resources did not sync \n{}\n\n"
            .format(kat2.get_status()))

    try:
        running = True
        active = ['|', '/', '-', '\\']
        page = 0; c = 0; perpage=50
        cycles = 0; clrscr(); gotoxy(1, 1)
        sens_filter = ''
        sens_status = 'nominal|warn|error'
        while c != 'q' and c != 'Q':
            clrscr(); cycles += 1
            print("CBF SKARAB {} error check : <Q> to quit   {}".format(subnr, active[cycles % 4]))
            print("-------------------------------------------")
            labels = cbfmon.sensor.i0_input_labelling.get_value()
            mappings = cbfmon.sensor.i0_hostname_functional_mapping.get_value()
            log_message('sub_ants = {}'.format(sub_ants))
            log_message('i0_input_labelling = {}'.format(labels))
            log_message('i0_hostname_functional_mapping = {}'.format(mappings))
            ants_to_hosts = {}
            hosts_to_ants = {}
            for i, ant in enumerate(sub_ants):
                ants_to_hosts[ant] = "{:02}".format(i)
                hosts_to_ants["{:02}".format(i)] = ant
            print('sub_ants:')
            print('    {}'.format(sub_ants))
            print('ants_to_hosts:')
            print('    {}'.format(ants_to_hosts))
            print('hosts_to_ants:')
            print('    {}'.format(hosts_to_ants))
            print('i0_input_labelling:')
            print('    {}'.format(labels))
            print('i0_hostname_functional_mapping:')
            print('    {}'.format(mappings))
            time.sleep(1)
            sens = obj.list_sensors(sens_filter, strategy=True, status=sens_status)
            if len(sens) == 0:
                numpages, rest = 0, 0
            else:
                numpages, rest = divmod(len(sens), perpage or 1)
                numpages = numpages + (1 if rest > 0 else 0)
            for s in sens[page * perpage:page * perpage + perpage]:
                name = s.name
                python_id = s.python_identifier
                description = s.description
                units = s.units
                type = s.type
                reading = s.reading
                val = str(reading.value)
                valTime = reading.timestamp
                updateTime = reading.received_timestamp
                stat = reading.status
                strat = 'none'  # TODO
                colour = get_sensor_colour(stat)
                stratchar = " " if strat == 'none' else "*"
                #Print status with stratchar prefix - indicates strategy has been set
                val = val if len(val) <= 80 or not truncate else val[:75] + "..." # truncate value to first 75 characters
                val = r"\n".join(val.splitlines())
                print "%s %s %s %s %s %s" % (col(colour) + name.ljust(45),
                    str(units).ljust(10), (stratchar + str(stat)).ljust(10),
                    get_time_str(valTime).ljust(15), get_time_str(updateTime).ljust(15),
                    str(val).ljust(45) + col('normal'))
                sys.stdout.flush()
            

            #Get user input for display control
            if c != 'q':
                c = getKeyIf(6)
            if c == '<' or c == 'b' or c == 'p':     #Back/Prev
                page = (page - 1) % numpages
            elif c == '>' or c == 'n':   #Next
                page = (page + 1) % numpages
            elif c == '-' or c == 'l':   #Less
                perpage = max(perpage - 2, 5)
            elif c == '+' or c == 'm':   #More
                perpage = min(perpage + 2, 80)
            elif c == 0 or c == '':
                pass
            elif c == 'q' or c == 'Q':
                print "\nExiting..."
            elif c == 'r' or c == 'R':
                print "\nRefreshing not yet implemented - TBD ..."
                # TODO: Add a sensor sampling refresh here on cbfmon_n
                print "\nSet auto strategy on cbfmon sensors"
                sensor_names = obj.set_sampling_strategies(sens_filter, "auto")
                time.sleep(1)

    except KeyboardInterrupt:
        print('\nExiting...')
        pass
    except Exception as exc:
        print('EXCEPTION ({})'.format(exc))

    finally:
        print("\n=========================================")
        print("CBF SKARAB error check : Done")
        print("=========================================")


