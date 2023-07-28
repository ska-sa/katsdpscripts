#!/usr/bin/python
import time
import re
import os
from prettytable import PrettyTable
from colorama import Fore
import operator
from katcorelib import standard_script_options, verify_and_connect


def sync_hrs_remaining(ants, band):
    """Returns the hours left for which the digitiser remains synchronised"""
    sensor = "dig_{}_band_time_remaining".format(band)
    for ant in ants:
        try:
            time_left = ant.sensor[sensor].get_value()
            if time_left:
                break
        except Exception:
            time_left = None
    return time_left


def subarray_activity(kat):
    """Returns activity of subarray 1"""
    # return subarray_1 details for now
    obs_params = kat.subarray_1.sensor.observation_script_arguments.get_value()

    # current running observation
    if not kat.subarray_1.sensor.state.get_value() == "active":
        return None

    else:
        if kat.subarray_1.sensor.script_status.get_value() == "busy":
            obs_start_time = (
                kat.subarray_1.sensor.observation_script_starttime.get_value()
            )
            obs_duration = re.search(r"-m \d{2,5}", obs_params)
            obs_description = (
                kat.subarray_1.sensor.observation_script_description.get_value()
            )
            sub_details = " Current obs:\t {}{}".format(
                " " * 20,
                obs_description,
            )

            if obs_duration:
                obs_duration = float(obs_duration.group().split()[1])
                obs_end_time = float(obs_start_time) + obs_duration
                sub_details += "\n duration: \t {}{} - {}\n".format(
                    " " * 20,
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.gmtime(float(obs_start_time))
                    ),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(obs_end_time)),
                )
            else:
                sub_details += "\n"

            return sub_details


def get_sensors(proxy, sensors):
    """Returns sensor-value pairs for a given proxy and sensor list"""
    sensor_values = {}
    for sensor in sensors:
        try:
            sensor_values[sensor] = proxy.sensor[sensor].get_value()
        except Exception:
            sensor_values[sensor] = None
    return sensor_values


def get_ap_sensors(ant):
    """Return AP sensor values for a given antenna"""
    ap_sensors = [
        "name",
        "ap_failure_present",
        "ap_control",
        "ap_mode",
        "ap_e_stop_reason",
        "ap_local_time_synced",
        "ap_yoke_door_open",
        "ap_hatch_door_open",
        "ap_ped_door_open",
        "ap_actual_elev",
        "ap_azim_hard_limit_ccw_reached",
        "ap_azim_hard_limit_cw_reached",
        "ap_elev_hard_limit_up_reached",
        "ap_elev_hard_limit_down_reached",
        "ap_indexer_position",
    ]
    ap_values = get_sensors(ant, ap_sensors)

    hatch_door = ap_values.pop("ap_hatch_door_open")
    yoke_door = ap_values.pop("ap_yoke_door_open")
    ped_door_ = ap_values.pop("ap_ped_door_open")
    ped_door = "open" if ped_door_ else "closed"
    hatch_state = "open" if hatch_door or yoke_door else "closed"

    ap_values["hatch"] = hatch_state
    ap_values["ped_door"] = ped_door

    azim_ccw = ap_values.pop("ap_azim_hard_limit_ccw_reached")
    azim_cw = ap_values.pop("ap_azim_hard_limit_cw_reached")
    elev_up = ap_values.pop("ap_elev_hard_limit_up_reached")
    elev_dn = ap_values.pop("ap_elev_hard_limit_down_reached")

    azim_limit = True if azim_ccw or azim_cw else False
    elev_limit = True if elev_up or elev_dn else False

    ap_values["azim limit"] = azim_limit
    ap_values["elev limit"] = elev_limit

    return ap_values


def get_rsc_sensors(ant, band):
    """Returns receiver sensor values for a given antenna and band"""
    rsc_sensors = [
        "rsc_rx{}_rfe1_temperature".format(band),
        "rsc_rx{}_lna_h_power_enabled".format(band),
        "rsc_rx{}_lna_v_power_enabled".format(band),

        #Add SBAND sensor logic, temp and LNA.
        "rsc_rx{}_tempvac_temp15k".format(band),
        "rsc_rx{}_mmic_enable.mmic1".format(band),
        "rsc_rx{}_mmic_enable.mmic2".format(band),

        #Add second stage amplifier sensors.
        "rsc_rx{}_amp2_h_power_enabled".format(band),
        "rsc_rx{}_amp2_v_power_enabled".format(band),
    ]
    rsc_values = get_sensors(ant, rsc_sensors)

    if band == "s":
        lna_h = rsc_values.pop("rsc_rx{}_mmic_enable.mmic1".format(band))
        lna_v = rsc_values.pop("rsc_rx{}_mmic_enable.mmic2".format(band))
        rsc_values["lnas"] = "ON" if lna_h and lna_v else "ON"
        amp2_h = rsc_values.pop("rsc_rx{}_amp2_h_power_enabled".format(band))
        amp2_v = rsc_values.pop("rsc_rx{}_amp2_v_power_enabled".format(band))
        rsc_values["amp2"] = "ON" if amp2_h and amp2_v else "ON"

    else:
        lna_h = rsc_values.pop("rsc_rx{}_lna_h_power_enabled".format(band))
        lna_v = rsc_values.pop("rsc_rx{}_lna_v_power_enabled".format(band))
        rsc_values["lnas"] = "ON" if lna_h and lna_v else "OFF"
        amp2_h = rsc_values.pop("rsc_rx{}_amp2_h_power_enabled".format(band))
        amp2_v = rsc_values.pop("rsc_rx{}_amp2_v_power_enabled".format(band))
        rsc_values["amp2"] = "ON" if amp2_h and amp2_v else "OFF"

    return rsc_values


def get_dig_sensors(ant, band, dmc_epoch):
    """Returns digitiser sensor values for a given antenna and band"""
    dig_sensors = [
        "dig_selected_band",
        "dig_{}_band_marking".format(band),
        "dig_{}_band_time_synchronisation_epoch".format(band),
        "dig_{}_band_time_synchronisation_offset".format(band),
    ]
    dig_values = get_sensors(ant, dig_sensors)

    sync_epoch = dig_values.pop("dig_{}_band_time_synchronisation_epoch".format(band))
    sync_offset = dig_values.pop("dig_{}_band_time_synchronisation_offset".format(band))
    dig_synced = True if sync_epoch == dmc_epoch and sync_offset != 0 else False

    dig_values["dig_synced"] = dig_synced

    return dig_values


def get_ant_data(ant, band, dmc_epoch):
    """Returns a dictionary of merged AP, receiver and digitizer sensors"""
    ap = get_ap_sensors(ant)
    rsc = get_rsc_sensors(ant, band)
    dig = get_dig_sensors(ant, band, dmc_epoch)
    ap.update(rsc)
    ap.update(dig)

    return ap


def format_sensors(ant_sensors, band, full_report):
    """Returns a 2D list of antennae sensor values with text-marked anormalies"""
    data = []
    exclude = []
    revised_data = []

    # add warning text
    for sensor in ant_sensors:
        row = []
        row.append(sensor["name"])

        failure = sensor["ap_failure_present"]
        if failure == True:
            row.append(Fore.RED + str(failure) + Fore.RESET)
        else:
            row.append(failure)

        control = sensor["ap_control"]
        if control != "remote":
            row.append(Fore.RED + control + Fore.RESET)
        else:
            row.append(control)

        mode = sensor["ap_mode"]
        if mode == "stop" or mode == "track":
            row.append(mode)
        else:
            row.append(Fore.RED + mode + Fore.RESET)

        e_stop = sensor["ap_e_stop_reason"]
        if e_stop != "none":
            row.append(Fore.RED + e_stop + Fore.RESET)
        else:
            row.append(e_stop)

        acu_synced = sensor["ap_local_time_synced"]
        if acu_synced != True:
            row.append(Fore.RED + str(acu_synced) + Fore.RESET)
        else:
            row.append(acu_synced)

        hatch = sensor["hatch"]
        if hatch != "closed":
            row.append(Fore.RED + hatch + Fore.RESET)
        else:
            row.append(hatch)

        ped_door = sensor["ped_door"]
        if ped_door != "closed":
            row.append(Fore.RED + ped_door + Fore.RESET)
        else:
            row.append(ped_door)

        azim_lim = sensor["azim limit"]
        if azim_lim == True:
            row.append(Fore.RED + str(azim_lim) + Fore.RESET)
        else:
            row.append(azim_lim)

        elev_lim = sensor["elev limit"]
        if elev_lim == True:
            row.append(Fore.RED + str(elev_lim) + Fore.RESET)
        else:
            row.append(elev_lim)

        elev = "{:.2f}".format(sensor["ap_actual_elev"])
        if float(elev) < 16 or float(elev) > 88:
            row.append(Fore.RED + elev + Fore.RESET)
        else:
            row.append(elev)

        if band == "s":
            rx_temp = "{:.2f}".format(sensor["rsc_rx{}_tempvac_temp15k".format(band)])
        else:
            rx_temp = "{:.2f}".format(sensor["rsc_rx{}_rfe1_temperature".format(band)])

        if float(rx_temp) < 0 or float(rx_temp) > 30:
            row.append(Fore.RED + rx_temp + Fore.RESET)
        else:
            row.append(rx_temp)

        lnas = sensor["lnas"]
        if lnas == "ON":
            row.append(lnas)
        else:
            row.append(Fore.RED + lnas + Fore.RESET)

        amp2 = sensor["amp2"]
        if amp2 == "ON":
            row.append(amp2)
        else:
            row.append(Fore.RED + amp2 + Fore.RESET)

        ridx_pos = sensor["ap_indexer_position"]
        if ridx_pos == "undefined":
            row.append(Fore.RED + ridx_pos + Fore.RESET)
        else:
            row.append(ridx_pos)

        selected_band = sensor["dig_selected_band"]
        bands = ["u", "l","s"]
        if selected_band not in bands:
            row.append(Fore.RED + selected_band + Fore.RESET)
        else:
            row.append(selected_band)

        dig_state = sensor["dig_{}_band_marking".format(band)]
        if dig_state != "ready":
            row.append(Fore.RED + dig_state + Fore.RESET)
        else:
            row.append(dig_state)

        dig_synced = sensor["dig_synced"]
        if dig_synced != True:
            row.append(Fore.RED + str(dig_synced) + Fore.RESET)
        else:
            row.append(dig_synced)

        if (
            failure == True
            or control != "remote"
            or e_stop != "none"
            or acu_synced != True
            or hatch != "closed"
            or azim_lim == True
            or elev_lim == True
            or float(rx_temp) > 100
            or lnas != "ON"
            or amp2 != "ON"
            or selected_band not in bands
            or dig_synced != True
        ):
            revised_data.append(row)
            exclude.append(sensor["name"])

        data.append(row)

    exclude = set(exclude)
    data = sorted(data, key=operator.itemgetter(0))
    revised_data = sorted(revised_data, key=operator.itemgetter(0))

    if full_report:
        return data, exclude
    else:
        return revised_data, exclude


def print_table(data, band):
    """Prints antennae sensor values in a tabular format"""
    header = [
        "name",
        "failure",
        "control",
        "mode",
        "e-stop",
        "acu sync",
        "hatch",
        "ped open",
        "azim lim",
        "elev lim",
        "elev deg",
        "rx{} temp".format(band),
        "LNAs",
        "amp2",
        "ridx pos",
        "dig band",
        "{}-dig state".format(band),
        "{}-dig sync".format(band),
    ]

    table = PrettyTable()
    table.field_names = header
    for row in data:
        table.add_row(row)
    return table


def check_schedule(exclude):
    """Returns which observations can be scheduled by passing non-ready antennas to check_baseline script"""

    filename = "/home/kat/katsdpscripts/utility/check_baseline.py"

    if exclude:
        exclude = " ".join(exclude).strip()
    else:
        exclude = ""

    try:
        print("\n with the following out:             {}\n".format(exclude))
        os.system("python {} {}".format(filename, exclude))
    except Exception as err:
        print(err)


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(
    usage="usage: %prog --receiver-band <band> --full-report=False --check-schedule=False",
    description="meerkat sensor summary",
)
parser.add_option(
    "--receiver-band",
    type="string",
    default="l",
    help='receiver band used to query digitiser and receiver sensor.(default="rxl")',
)
parser.add_option(
    "--full-report",
    action="store_true",
    default=False,
    help="prints table for all antennas, otherwise prints antennas with bad values only (default=%default)",
)
parser.add_option(
    "--check-schedule",
    action="store_true",
    default=False,
    help="shows what can be scheduled with the current available antennas (default=%default)",
)

# assume basic options passed from instruction_set
parser.set_defaults(
    description="Meerkat Status", proposal_id="20190205-OPS1A", observer="operator"
)

(opts, args) = parser.parse_args()

if opts.receiver_band not in ["u", "l", "s"]:
    raise ValueError("Invalid receiver band. Valid bands are ['u', 'l', 's']")

with verify_and_connect(opts) as kat:
    # separate antennas by readiness
    ant_inactive = [
        ant
        for ant in kat.ants
        if ant.name in kat.katpool.sensor.resources_in_maintenance.get_value()
        or ant.name in kat.subarray_7.sensor.pool_resources.get_value()
        or ant.name in kat.katpool.sensor.resources_faulty.get_value()
    ]
    exclude = set([ant.name for ant in ant_inactive])

    ant_active = [ant for ant in kat.ants if ant not in ant_inactive]

    obs_details = subarray_activity(kat)
    dmc_epoch = kat.mcp.sensor.dmc_synchronisation_epoch.get_value()
    hours_left = sync_hrs_remaining(ant_active, opts.receiver_band)
    epoch = time.gmtime(dmc_epoch)
    epoch_time = time.strftime("%Y-%m-%d %H:%M:%S", epoch)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

    print(
        "\n {}-band MeerKAT status at:           {}".format(
            opts.receiver_band.upper(), now
        )
    )
    print(" Last global sync:                   {}".format(epoch_time))
    print(" Time left until next sync:          {} hours\n".format(hours_left))

    if obs_details:
        print(obs_details)

    # available antennas
    if ant_active:
        active_ = [get_ant_data(ant, opts.receiver_band, dmc_epoch) for ant in ant_active]
        ants = ", ".join(sorted([ant.name for ant in ant_active]))
        data, excluded = format_sensors(active_, opts.receiver_band, opts.full_report)
        exclude = exclude.union(excluded)
        print(" Available antennas:                 {}\n".format(len(active_)))
        print(" {}\n".format(ants))
        print(print_table(data, opts.receiver_band))

    if ant_inactive:
        inactive_ = [
            get_ant_data(ant, opts.receiver_band, dmc_epoch) for ant in ant_inactive
        ]
        data, excluded = format_sensors(inactive_, opts.receiver_band, True)
        print("\n unavailable antennas:               {}\n".format(len(inactive_)))
        print(print_table(data, opts.receiver_band))
        print("\n")

    if opts.check_schedule:
        exclude = sorted(list(exclude))
        check_schedule(exclude)
    print("\n")
# END
