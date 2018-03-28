#!/usr/bin/python

import time
import sys
from katcorelib import standard_script_options, verify_and_connect, user_logger
from katcorelib import cambuild
from katmisc.utils.ansi import colors, get_sensor_colour, gotoxy, clrscr, col, getKeyIf
from katmisc.utils.utils import get_time_str, escape_name



def log_timestamp():
    # create log timestamp format
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def log_message(msg, level='info', boldtype=False, colour=colors.Normal):

    bold = boldtype

    if level == 'debug':
        user_logger.debug(str(msg))
    elif level == 'info':
        user_logger.info(str(msg))
    elif level == 'warn':
        user_logger.warn(str(msg))
        colour = colors.Orange
    elif level == 'error':
        user_logger.error(str(msg))
        colour = colors.Red
        bold = True
    print colour+msg


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
    print("Waiting for cambuild")
    kat.until_synced(timeout=30)
    print("=========="*6)
    print("CBF SKARAB error check : start")
    print("=========="*6)
    print(opts)
    print(kat.controlled_objects)
    print(kat.ants.clients)
    print("----------"*5)

    if opts.strategy not in ["once", "detail", "period"]:
        print(">>> Unknown strategy given: '{}'', using 'once' instead"
              .format(opts.strategy))
        opts.strategy = "once"

    once = opts.strategy in ["once", "detail"]
    detail = opts.strategy in ["detail"]

    subarrays = kat.katpool.sensor.subarrays.get_value()
    subarrays_free = kat.katpool.sensor.subarrays_free.get_value()
    ants = kat.katpool.sensor.ants.get_value().split(",")
    sub = getattr(kat, "sub")
    sub_pool = getattr(kat.katpool.sensor,
                       "pool_resources_{}".format(subnr)).get_value().split(',')
    # Get subarray in subarray
    sub_ants = [res for res in ants if res in sub_pool]
    sub_state = sub.sensor.state.get_value()
    print("----------"*5)
    print("Subarray: {}".format(subnr))
    print("   state: {}".format(sub_state))
    print("    ants: {}".format(sub_ants))
    print("    pool: {}".format(sub_pool))
    print(kat.ants.clients)
    print(opts)
    print("----------"*5)

    if opts.dry_run:
        log_message('Dry Run only\n')
        print("\n")
        print("=========="*5)
        print("CBF SKARAB error check : Dry-run")
        print("=========="*5)
        exit(0)

    # Build the 2nd KAT object. This kat object is being used to access sensor data
    # for resources outside the subarray.
    cbf = getattr(kat, "cbf")
    cbfmon = getattr(kat, "cbfmon_{}".format(subnr))
    obj = cbfmon

    log_message("Waiting for cbfmon_{} to sync\n".format(subnr))
    cbf_ok = cbf.until_synced(timeout=15)
    cbfmon_ok = cbfmon.until_synced(timeout=60)

    if not (cbf_ok and cbfmon_ok):
        log_message("Some resources did not sync \n kat.cbf_{}={} "
                    "kat2.cbfmon{_{}={}\n{}\n\n"
                    .format(subnr, cbf_ok, subnr, cbfmon_ok, kat.get_status()), 'error')
        log_message("Aborting script", 'error')
        raise RuntimeError(
            "Aborting - Some resources did not sync \n{}\n\n"
            .format(kat.get_status()))

    try:
        running = True
        active = ['|', '/', '-', '\\']
        page = 0; c = 0; perpage = 50; cycles = 0;
        if once:
            perpage = 9999
        else:
            clrscr(); gotoxy(1, 1)
        truncate = False
        sens_filter = 'device_status'
        sens_status = 'warn|error|unknown|failure'
        while c != 'q' and c != 'Q':
            if once:
                # no clear screen
                print "\n\n"
                print("----------"*5)
                print("CBF SKARAB {} error check : ".format(subnr))
                print("----------"*5)
            else:
                time.sleep(2)
                clrscr(); cycles += 1
                print("CBF SKARAB {} error check : <Q> to quit   {}"
                      .format(subnr, active[cycles % 4]))
                print("----------"*5)
            # This is brittle because it relies on parsing the sensor value string
            labels = cbfmon.sensor.i0_input_labelling.get_value()

            # import ipdb; ipdb.set_trace()
            # Process input labels - like
            #    [('m001h', 0, 'board020900', 0), ('m001v', 1, 'board020900', 1), ...
            # Remove outer square brackets and single/double quotes
            cmd = "list("+labels+")"
            labels_list = eval(cmd)
            labels_dict = {}
            for (antpol, nr, board, pol) in labels_list:
                # item like - m001h, 0, board020900, 0
                if board not in labels_dict:
                    labels_dict[board] = antpol
                else:
                    labels_dict[board] = ",".join([labels_dict[board], antpol])
            print type(labels_list), labels_dict.keys()

            # Process mappings - lke  {'skarab02000-01': 'fhost00', 'skarab02007-01': 'xhost03', ....}
            mappings = cbfmon.sensor.i0_hostname_functional_mapping.get_value()
            cmd = "dict("+mappings+")"
            mappings_dict = eval(cmd)
            print type(mappings_dict), mappings_dict.keys()
            host_board_dict = {}
            for item in mappings_dict:
                host_board_dict[mappings_dict[item]] = item
            # map_dict = dict((fhost.strip(), skarab.strip()) for skarab,fhost in
            #                   (item.split(':') for item in mappings.strip("{}").split(',')))
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
            print('    {}'.format(labels_dict))
            print('i0_hostname_functional_mapping:')
            print('    {}'.format(mappings))
            print('    {}'.format(host_board_dict))
            sys.stdout.flush()
            print("\n\nGetting sensor list ...")
            sens = obj.list_sensors(sens_filter, status=sens_status, refresh=True)
            print('Filter: {}, Status: {}, Found sensors: {}\n\n'
                  .format(sens_filter, sens_status, len(sens)))
            xhosts = set()
            fhosts = set()
            for s in sens:
                name = s.name  # Here in the format i0.fhostNN.dfsdfdf.device-status
                if name.startswith("i0.fhost"):
                    host = name.split(".")[1]
                    fhosts.add(host)
                elif name.startswith("i0.shost"):
                    host = name.split(".")[1]
                    xhosts.add(host)
                else:
                    continue

            print("\nFHOST FAILURES:")
            for host in sorted(fhosts):
                board = host_board_dict.get(host, "unknown")
                print "{}: {} - {}".format(host, board, labels_dict.get(board, "unknown"))
            sys.stdout.flush()

            print("\nXHOST FAILURES:")
            for host in sorted(xhosts):
                board = host_board_dict.get(host, "unknown")
                print "{}: {}".format(host, board)
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
                print "\nRefreshing not yet implemented - TBD ..."
                # TODO: Add a sensor sampling refresh here on cbfmon_n
                print "\nSet auto strategy on cbfmon sensors"
                sensor_names = obj.set_sampling_strategies(sens_filter, "auto")

        # At the end print the detailed non-nominal sensors
        if detail:
            print("\n\n")
            print("DETAIL REPORT of all warn|error|unknown sensors")
            print('Filter: {}, Status: {}, Found sensors: {}'
                  .format(sens_filter, sens_status, len(sens)))
            print("----------"*5)
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
                # Print status with stratchar prefix - indicates strategy has been set
                # truncate value to first 75 characters
                val = val if len(val) <= 100 or not truncate else val[:95] + "..."
                val = r"\n".join(val.splitlines())
                print("%s %s %s %s %s %s" % (col(colour) + name.ljust(45),
                      str(units).ljust(10), (stratchar + str(stat)).ljust(10),
                      get_time_str(valTime).ljust(15), get_time_str(updateTime).ljust(15),
                      str(val).ljust(45) + col('normal')))
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
