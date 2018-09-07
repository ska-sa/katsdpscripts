#!/usr/bin/python
from __future__ import print_function

import sys
import time

from ast import literal_eval
from pprint import pprint

from katcorelib import standard_script_options, user_logger
from katcorelib import cambuild
from katmisc.utils.ansi import colors, get_sensor_colour, gotoxy, clrscr, col, getKeyIf
from katmisc.utils.utils import get_time_str


def log_timestamp():
    """Create log timestamp format."""
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def log_message(msg, level='info', colour=colors.Normal):
    if level == 'debug':
        user_logger.debug(str(msg))
    elif level == 'info':
        user_logger.info(str(msg))
    elif level == 'warn':
        user_logger.warn(str(msg))
        colour = colors.Brown
    elif level == 'error':
        user_logger.error(str(msg))
        colour = colors.Red
    print(colour + msg + colors.Normal)


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(
    usage="usage: %prog [options]",
    description="MeerKAT SKARAB CBF error check")
parser.add_option("--subnr", type=int, default=1,
                  help="Subarray to check. (default='%default')")
parser.add_option("--strategy", default="once",
                  help="Strategy for this script. 'once|period|detail' "
                       "(default='%default')")

# assume basic options passed from instruction_set
parser.set_defaults(description="CBF SKARAB error check")
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


ROACH:
Name: cbf_roach_2_i0_hostname_functional_mapping        Description: On which hostname is which functional host?
{'roach020951': 'xhost10', 'roach020954': 'xhost11', 'roach020955': 'xhost12', 'roach020A12': 'fhost7', 'roach02093A': 'fhost3', 'roach02091D': 'fhost11', 'roach02091F': 'fhost0', 'roach02091C': 'fhost10', 'roach020A03': 'xhost13', 'roach02094E': 'xhost3', 'roach020962': 'fhost12', 'roach020A0E': 'xhost7', 'roach02094C': 'fhost13', 'roach020A0F': 'fhost6', 'roach020923': 'xhost8', 'roach020922': 'fhost1', 'roach020A0B': 'xhost15', 'roach020927': 'fhost2', 'roach02092B': 'xhost9', 'roach020961': 'xhost5', 'roach020946': 'xhost2', 'roach020A01': 'xhost6', 'roach020944': 'fhost4', 'roach020A08': 'xhost14', 'roach020A0D': 'fhost5', 'roach020936': 'xhost1', 'roach02095A': 'fhost14', 'roach02095C': 'xhost4', 'roach020933': 'fhost15', 'roach020910': 'fhost8', 'roach020911': 'xhost0', 'roach020912': 'fhost9'}


Name: cbf_roach_2_i0_host_mapping                       Description: Indicates\_on\_which\_physical\_host\_a\_set\_of\_engines\_can\_be\_found.
{'roach020951': ['xeng40', 'xeng41', 'xeng42', 'xeng43'], 'roach020954': ['xeng44', 'xeng45', 'xeng46', 'xeng47'], 'roach020955': ['xeng48', 'xeng49', 'xeng50', 'xeng51'], 'roach020A12': ['feng14', 'feng15'], 'roach02093A': ['feng6', 'feng7'], 'roach02091D': ['feng22', 'feng23'], 'roach02091F': ['feng0', 'feng1'], 'roach02091C': ['feng20', 'feng21'], 'roach020A03': ['xeng52', 'xeng53', 'xeng54', 'xeng55'], 'roach02094E': ['xeng12', 'xeng13', 'xeng14', 'xeng15'], 'roach020962': ['feng24', 'feng25'], 'roach020A0E': ['xeng28', 'xeng29', 'xeng30', 'xeng31'], 'roach020922': ['feng2', 'feng3'], 'roach02094C': ['feng26', 'feng27'], 'roach020A0F': ['feng12', 'feng13'], 'roach020923': ['xeng32', 'xeng33', 'xeng34', 'xeng35'], 'roach020A0D': ['feng10', 'feng11'], 'roach020A0B': ['xeng60', 'xeng61', 'xeng62', 'xeng63'], 'roach020927': ['feng4', 'feng5'], 'roach02092B': ['xeng36', 'xeng37', 'xeng38', 'xeng39'], 'roach020961': ['xeng20', 'xeng21', 'xeng22', 'xeng23'], 'roach020946': ['xeng8', 'xeng9', 'xeng10', 'xeng11'], 'roach020A01': ['xeng24', 'xeng25', 'xeng26', 'xeng27'], 'roach020944': ['feng8', 'feng9'], 'roach020A08': ['xeng56', 'xeng57', 'xeng58', 'xeng59'], 'roach020936': ['xeng4', 'xeng5', 'xeng6', 'xeng7'], 'roach02095A': ['feng28', 'feng29'], 'roach02095C': ['xeng16', 'xeng17', 'xeng18', 'xeng19'], 'roach020933': ['feng30', 'feng31'], 'roach020910': ['feng16', 'feng17'], 'roach020911': ['xeng0', 'xeng1', 'xeng2', 'xeng3'], 'roach020912': ['feng18', 'feng19']}

Receptor to digitser:
Name: m001_dig_version_list
dig-l-60.dcpproxy KAT/1-291-g5e42b5b,dig-l-60.firmware m1130_2042sdp_rev1_23_of,dig-l-60.katcp-library v0.2.0-190-gec0e7e7 2018-02-01T09:53:40,dig-l-60.katcp-protocol 5.0-M,dig-l-60.kernel 3.13.1 #1 SMP Wed Feb 5 14:37:06 SAST 2014,katcp-device digitiser-1.b e9ec171,katcp-library Ckatcp-v0.2.0-72-g669959c,katcp-protocol 5.0-M

"""


'''
def cambuild(password=None, full_control=False, conn_object=None,
             auto_reconnect=True, log_level=logging.WARN,
             log_file="lib.log", require=[], conn_clients='all',
             sub_nr=None, config_label=None):

    corebuild(conn_object=conn,
              conn_clients=conn_clients, controlled_clients=controlled_clients,
              sb_id_code=sb_id_code, user_id=user_id, sub_nr=sub_nr,
              sim=sim, cam=cam, cam_pass=cam_pass, dry_run=dry_run,
              check_noproduct=check_noproduct)
'''
with cambuild(sub_nr=subnr) as kat:
    print("Waiting for cambuild on subarray", subnr)
    kat.until_synced(timeout=30)
    sub_ants = sorted([ant.name for ant in kat.ants.clients])
    sub_state = kat.sub.sensor.state.get_value()
    sub_band = kat.sub.sensor.band.get_value()
    sub_pool_sensor = getattr(kat.katpool.sensor, "pool_resources_{}".format(subnr))
    sub_pool = sorted(sub_pool_sensor.get_value().split(','))
    print("=========="*6)
    print("CBF SKARAB error check : start")
    print("=========="*6)
    print("Connected objects:", sorted(kat.connected_objects.keys()))
    print("Controlled objects:", sorted(kat.controlled_objects.keys()))
    print("Subarray: {}".format(subnr))
    print("   state: {}".format(sub_state))
    print("    band: {}".format(sub_band))
    print("    ants: {}".format(sub_ants))
    print("    pool: {}".format(sub_pool))
    print("----------"*5)

    if opts.strategy not in ["once", "detail", "period"]:
        print("{}>>> Unknown strategy given: '{}'', using 'once' instead{}"
              .format(colors.Red, opts.strategy, colors.Normal))
        opts.strategy = "once"

    once = opts.strategy in ["once", "detail"]
    detail = opts.strategy in ["detail"]

    if opts.dry_run:
        log_message('Dry Run only\n')
        print("\n")
        print("=========="*5)
        print("CBF SKARAB error check : Dry-run")
        print("=========="*5)
        exit(0)

    cbf = kat.cbf
    if 'cbf_dev_{}'.format(subnr) in sub_pool:
        cbfmon_prefix = "cbfmon_dev"
    else:
        cbfmon_prefix = "cbfmon"
    cbfmon = getattr(kat, "{}_{}".format(cbfmon_prefix, subnr))

    log_message("Waiting for cbf_{0} and {1}_{0} to sync\n".format(subnr, cbfmon_prefix))
    cbf_ok = cbf.until_synced(timeout=15)
    cbfmon_ok = cbfmon.until_synced(timeout=60)

    if not (cbf_ok and cbfmon_ok):
        log_message("Some resources did not sync \n kat.cbf_{}={} "
                    "kat.{}_{}={}\n{}\n\n"
                    .format(subnr, cbf_ok, cbfmon_prefix,
                            subnr, cbfmon_ok, kat.get_status()),
                    'error')
        log_message("Aborting script", 'error')
        raise RuntimeError(
            "Aborting - Some resources did not sync \n{}\n\n"
            .format(kat.get_status()))

    try:
        running = True
        active = ['|', '/', '-', '\\']
        page = 0
        numpages = 1
        c = 0
        perpage = 50
        cycles = 0
        if once:
            perpage = 9999
        else:
            clrscr()
            gotoxy(1, 1)

        truncate = False
        sens_filter = 'device-status|feng-rxtime-ok|xeng-vaccs-synchronised|fhost\d+.network.[rt]x-gbps'
        sens_status = 'warn|error|unknown|failure'
        while c != 'q' and c != 'Q':
            if once:
                # no clear screen
                print("\n\n")
                print("----------"*5)
                print("CBF SKARAB {} error check : cbf_ok {} cbfmon_ok {}"
                      .format(subnr, cbf_ok, cbfmon_ok))
                print("----------"*5)
            else:
                time.sleep(2)
                clrscr()
                cycles += 1
                print("CBF SKARAB {} error check : <Q> to quit   {}"
                      .format(subnr, active[cycles % 4]))
                print("----------"*5)

            # This is brittle because it relies on parsing the sensor value string
            # Process input labels - like
            #   [('m001h', 0, 'skarab02000-01', 0), ('m001v', 1, 'skarab02000-01', 1), ...
            labels = cbf.sensor.i0_input_labelling.get_value()
            labels_list = literal_eval(labels.strip())
            boards_to_ants = {}
            for (antpol, nr, board, pol) in labels_list:
                # item like - m001h, 0, skarab02000-01, 0
                if board not in boards_to_ants:
                    boards_to_ants[board] = antpol
                else:
                    boards_to_ants[board] = ",".join([boards_to_ants[board], antpol])
            print(type(labels_list), boards_to_ants.keys())

            # Process hostname-functional mappings - like
            #     {'skarab02000-01': 'fhost00', 'skarab02007-01': 'xhost03', ....}
            func_mappings = cbfmon.sensor.i0_hostname_functional_mapping.get_value()
            func_mappings = func_mappings.strip()
            if func_mappings:
                func_mappings_dict = literal_eval(func_mappings.strip())
            else:
                # sometimes sensor is empty on site
                func_mappings_dict = {}
            print(type(func_mappings_dict), func_mappings_dict.keys())
            hosts_to_boards = {}
            for board, host in func_mappings_dict.items():
                hosts_to_boards[host] = board

            # Process host-engine mappings - like
            #     {'skarab020804-01': ['xeng012', 'xeng013', 'xeng014', 'xeng015'], ...
            engines_to_boards = {}
            if hasattr(cbf.sensor, 'i0_host_mapping'):
                host_mappings = cbf.sensor.i0_host_mapping.get_value()
            else:
                # probably simulated system, which excludes this sensor
                host_mappings = ''
            host_mappings = host_mappings.strip()
            if host_mappings:
                host_mappings_dict = literal_eval(host_mappings.strip())
            else:
                host_mappings_dict = {}
            print(type(host_mappings_dict), host_mappings_dict.keys())
            for board, engines in host_mappings_dict.items():
                for engine in engines:
                    engines_to_boards[engine] = board

            # Workaround, for empty hostname-functional-mapping sensor on site
            if host_mappings_dict and not hosts_to_boards:
                log_message("Guesstimating hosts_to_boards", 'warn')
                # Try to build it up, making some assumptions
                for board, engines in host_mappings_dict.items():
                    # engines is a list of 4 xengs, or 2 fengs - just use first item
                    engine = engines[0]
                    # this will be like 'xeng00' or 'xeng000' or 'feng123'
                    host_type = engine[0]
                    eng_num = int(engine[4:])
                    host_index = eng_num // len(engines)
                    host = "{}host{:02}".format(host_type, host_index)
                    hosts_to_boards[host] = board

            # Use assumption of CAM mapping sorted antennas to ascending input number
            ants_to_hosts = {}
            hosts_to_ants = {}
            for i, ant in enumerate(sub_ants):
                host = "fhost{:02}".format(i)
                ants_to_hosts[ant] = host
                hosts_to_ants[host] = ant

            # Get digitiser serial number to antenna mapping
            # From DMC sensor - like
            #   dig-l-60.dcpproxy KAT/1-291-g5e42b5b,dig-l-60.firmware ...
            # From static CAM config - like
            #   ready:dig-060
            ants_to_digitisers = {}
            for ant in kat.ants:
                dig_version_list = ant.sensor.dig_version_list.get_value()
                sensor_serial = dig_version_list.split('.')[0]
                ant_config = kat.katconfig.array_conf.antennas[ant.name].rec_config_dict
                dig_key = 'digitiser_{}'.format(sub_band)
                config_serial = ant_config['installed'][dig_key].split(':')[-1]
                ants_to_digitisers[ant.name] = (sensor_serial, config_serial)

            print('\nants_to_digitisers:')
            pprint(ants_to_digitisers, indent=4)
            print('ants_to_hosts:')
            pprint(ants_to_hosts, indent=4)
            print('hosts_to_ants:')
            pprint(hosts_to_ants, indent=4)
            print('i0_input_labelling - list:')
            pprint(labels_list, indent=4)
            print('boards_to_ants:')
            pprint(boards_to_ants, indent=4)
            print('i0_hostname_functional_mapping:')
            pprint(func_mappings_dict, indent=4)
            print('hosts_to_boards:')
            pprint(hosts_to_boards, indent=4)
            print('i0_host_mapping:')
            pprint(host_mappings_dict, indent=4)
            sys.stdout.flush()

            # Get all sensor readings with a single ?sensor-value request, which is much
            # faster than get_value.
            req_sens_filter = "/"+sens_filter+"/"
            sens_statuses = sens_status.split("|")
            print("Getting sensor values for sensor filter: {}".format(req_sens_filter))
            reading_time = time.time()
            reply, informs = cbfmon.req.sensor_value(req_sens_filter)
            sens = []
            for inform in informs:
                timestamp, _count, name, status, value = inform.arguments
                special = ""
                if name.endswith('-gbps'):
                    if float(value) < 1.0:
                        special = "*** low data rate - host disabled?"
                if status in sens_statuses or special:
                    sens.append(
                        (name, float(timestamp), reading_time, status, value, special))

            print('Filter: {}, Status: {},  {}/{} sensors\n'
                  .format(sens_filter, sens_status, len(sens), reply.arguments[1]))
            xhosts_error = set()
            fhosts_error = set()
            xhosts_warn = set()
            fhosts_warn = set()
            for s in sens:
                (name, value_time, reading_time, status, value, special) = s
                if name.startswith("i0.fhost"):
                    errors = fhosts_error
                    warns = fhosts_warn
                elif name.startswith("i0.xhost"):
                    errors = xhosts_error
                    warns = xhosts_warn
                else:
                    continue
                host = name.split(".")[1]
                if status in ['error', 'failure', 'unknown']:
                    errors.add(host)
                elif status in ['warn'] or special:
                    warns.add(host)

            if fhosts_error or fhosts_warn:
                print("\nFHOST ISSUES:")
                print("level   fhost    board            ant streams   "
                      "dig S/N: sensor, config")
                print("-" * 71)

                def print_fhost_details(host, status):
                    board = hosts_to_boards.get(host, "unknown")
                    antpols = boards_to_ants.get(board, "unknown")
                    ant = hosts_to_ants.get(host, "unknown")
                    digitiser = ants_to_digitisers.get(ant, "unknown")
                    colour = col(get_sensor_colour(status.strip()))
                    print("{}{} - {}: {} - {} - {}{}".format(
                        colour, status.upper(), host, board, antpols, digitiser,
                        colors.Normal))

                for host in sorted(fhosts_error):
                    print_fhost_details(host, 'error')
                for host in sorted(fhosts_warn):
                    print_fhost_details(host, 'warn ')
            else:
                print("\n(No FHOST issues)")

            if xhosts_error or xhosts_warn:
                print("\nXHOST ISSUES:")
                print("level   xhost    board")
                print("----------------------")

                def print_xhost_details(host, status):
                    board = hosts_to_boards.get(host, "unknown")
                    colour = col(get_sensor_colour(status.strip()))
                    print("{}{} - {}: {}{}".format(
                        colour, status.upper(), host, board, colors.Normal))

                for host in sorted(xhosts_error):
                    print_xhost_details(host, 'error')
                for host in sorted(xhosts_warn):
                    print_xhost_details(host, 'warn ')
            else:
                print("\n(No XHOST issues)")

            sys.stdout.flush()

            # Get user input for display control
            if once:
                # Force the exit
                c = 'q'
            else:
                # Read a key
                c = getKeyIf(6)

            if c == '<' or c == 'b' or c == 'p':     # Back/Prev
                page = (page - 1) % numpages
            elif c == '>' or c == 'n':   # Next
                page = (page + 1) % numpages
            elif c == '-' or c == 'l':   # Less
                perpage = max(perpage - 2, 5)
            elif c == '+' or c == 'm':   # More
                perpage = min(perpage + 2, 80)
            elif c == 'r' or c == 'R':
                print("\nRefreshing not yet implemented - TBD ...")
                # TODO: Add a sensor sampling refresh here on cbfmon_n
                print("\nSet auto strategy on cbfmon sensors")
                sensor_names = cbfmon.set_sampling_strategies(sens_filter, "auto")

        # At the end print the detailed non-nominal sensors
        if detail:
            print("\n")
            print("DETAIL REPORT of all warn|error|unknown sensors")
            print('Filter: {}, Status: {}, Found sensors: {}'
                  .format(sens_filter, sens_status, len(sens)))
            print("----------"*5)
            print("%s %s %s %s %s" % ("Name".ljust(45),
                  "Status".ljust(10),
                  "Value_time".ljust(15),
                  "Reading_time".ljust(15),
                  "Value".ljust(45)))
            if len(sens) == 0:
                numpages, rest = 0, 0
            else:
                numpages, rest = divmod(len(sens), perpage or 1)
                numpages = numpages + (1 if rest > 0 else 0)
            for s in sens[page * perpage:page * perpage + perpage]:
                (name, value_time, reading_time, status, value, special) = s
                colour = get_sensor_colour(status)
                # truncate value to first 75 characters
                value = value if len(value) <= 100 or not truncate else value[:95] + "..."
                value = r"\n".join(value.splitlines())
                print("%s %s %s %s %s %s" % (col(colour) + name.ljust(45),
                      status.ljust(10),
                      get_time_str(value_time).ljust(15),
                      get_time_str(reading_time).ljust(15),
                      str(value).ljust(45),
                      special + col('normal')))
                sys.stdout.flush()

    except KeyboardInterrupt:
        print('\nExiting...')
        pass
    except Exception as exc:
        print('EXCEPTION ({})'.format(exc))

    finally:
        print("\n")
        print("=========="*6)
        print("CBF SKARAB error check : Done")
        print("=========="*6)
