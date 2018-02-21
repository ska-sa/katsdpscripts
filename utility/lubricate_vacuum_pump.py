#!/usr/bin/python

import time
import string
from katcorelib import standard_script_options, verify_and_connect, user_logger
from katcorelib import tbuild, kat_resource
import numpy as np
import os
import katcp
import smtplib
from email.mime.text import MIMEText

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(
    usage="usage: %prog [options]",
    description="Check receptors are in suitable position and command the RSC "
                "to lubricate vacuum pump for a specified duration. "
                "Note: AMBIENT TEMP MUST BE ABOVE 16 DEG C")
parser.add_option(
    '--max_elevation', type='int', default=20,
    help="Make sure receptor stays below this elevation for duration of vacuum "
         "pump lubrication (default=%default degrees)")
parser.add_option(
    '--run_duration', type='int', default=20,
    help="Minimum run duration of vacuum pump (default=%default minutes)")
parser.add_option(
    '--lubrication_frequency', type='int', default=14,
    help='Frequency for running vacuum pump lubrication (default=%default days)')
parser.add_option(
    '--archive_search', type='int', default=30,
    help='Search sensor archive for this many days to check when vacuum pump '
         'was last run (default=%default days)')
parser.add_option(
    '--receptors', type='str', default='all',
    help='List of receptors to run vacuum pump lubrication on (default=%default)')
parser.add_option(
    '--ideal_vac_pressure', type='float', default=5e-2,
    help='Pressure to which the vacuum must go when pump operating (default=%default)')
parser.add_option(
    '--email_to', type='str',
    default='blunsky@ska.ac.za,bjordaan@ska.ac.za,operators@ska.ac.za',
    help='Comma separated email list of people to send report to (default=%default)')

parser.set_defaults(description='Lubricate Vacuum Pumps on Receivers')
(opts, args) = parser.parse_args()

email_msg = []
MIN_OP_TEMP = 16


def timestamp():
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
        email_msg.append(timestamp() + level.upper() + ' ' + str(msg))
    elif colour != 'black' and not(bold):
        email_msg.append('<font color="{colour}">{timestamp} {level} {msg}</font>'.format(
            colour=colour, timestamp=timestamp(), level=level.upper(), msg=str(msg)))
    else:
        email_msg.append('<font color="{colour}"><b>{timestamp} {level} {msg}</b></font>'.format(
            colour=colour, timestamp=timestamp(), level=level.upper(), msg=str(msg)))


log_message("Lubricate Vacuum Pumps: start\n", boldtype=True,)


def send_email(lines, subject, messagefrom='operators@ska.ac.za'):

    body = '\n'.join(lines)
    body = string.replace(body, '\n', '<br>\n')

    html = """\
    <html>
        <body>
          <p>
              """ + body + """\
          </p>
        </body>
    </html>
    """

    if type(opts.email_to) is list:
        messageto = ', '.join((opts.email_to).replace(' ', ''))
    else:
        messageto = (opts.email_to).replace(' ', '')

    msg = MIMEText(html, 'html')
    msg['Subject'] = subject
    msg['From'] = messagefrom
    msg['To'] = messageto

    if type(opts.email_to) is list:
        sendto = (opts.email_to).replace(' ', '')
    elif (opts.email_to).find(',') >= 0:
        sendto = ((opts.email_to).replace(' ', '')).split(',')
    elif (opts.email_to).find(';') >= 0:
        sendto = ((opts.email_to).replace(' ', '')).split(';')
    else:
        sendto = (opts.email_to).replace(' ', '')

    smtp_server = smtplib.SMTP('smtp.kat.ac.za')
    smtp_server.sendmail(messagefrom, sendto, msg.as_string())
    smtp_server.quit()


def connect_to_rsc(ant_name, port):
    rsc_interface = {'controlled': True,
                     'description': 'RSC maintenance interface.'}
    rsc_interface['name'] = 'rsc_{}'.format(ant_name)
    rsc_interface['address'] = ('10.96.{}.20'.format(int(ant_name[2:])), port)
    log_message('Connecting to RSC at {ant} at IP address {ip} on port {port}'.format(
        ant=ant_name, ip=rsc_interface['address'][0], port=rsc_interface['address'][1]))
    try:
        dev_katcp = katcp.resource_client.KATCPClientResource(rsc_interface)
        rsc_device = kat_resource.make_resource_blocking(dev_katcp)
        rsc_device.until_synced()
        return rsc_device
    except:
        log_message('Failed to connect to RSC on {}'.format(ant_name), ('error'))
        return None


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


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


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


with verify_and_connect(opts) as kat:
    print "_______________________"
    print opts
    print "_______________________"
    log_message("Opts:\n{}\n".format(opts))

    MAINT_PORT = 7148
    LOCK_DELAY = 300  # seconds

    # check that lubrication frequency is less than the time we look back
    # into the sensor archive
    if opts.archive_search < opts.lubrication_frequency:
        log_message(
            "Waiting script - archive_search parameter must be > lubrication_frequency", 'error')
        raise RuntimeError(
            "Aborting - archive_search parameter must be > lubrication_frequency\n\n")

    if not(kat.dry_run):
        # Build the 2nd KAT object. This kat object is being used to access sensor data
        # for resources outside the subarray.
        log_message('Begin tbuild...', 'info')
        if opts.receptors == 'all':
            kat2 = tbuild(conn_clients='all')
        else:
            kat2 = tbuild(conn_clients=str(opts.receptors) + ',katpool,anc')

        log_message("Waiting for katpool and anc to sync\n", 'info')
        katpool_ok = kat2.katpool.until_synced(timeout=15)
        anc_ok = kat2.anc.until_synced(timeout=15)

        if not (katpool_ok and anc_ok):
            log_message("Some resources did not sync \n{}\n\n"
                        .format(kat2.get_status()), 'error')
            log_message("Aborting script", 'error')
            raise RuntimeError(
                "Aborting - Some resources did not sync \n{}\n\n"
                .format(kat2.get_status()))

    if not(kat.dry_run):
        # Ambient should be above 16 deg c.
        if (kat2.anc.sensor.air_temperature.get_value() < MIN_OP_TEMP):
            log_message(
                'Aborting script - ambient temperature is below {} deg C'
                .format(MIN_OP_TEMP), 'error')
            raise RuntimeError(
                'Aborting script - ambient temperature is below {} deg C\n\n'
                .format(MIN_OP_TEMP))
        log_message('Current Ambient temperature is {:0.2f}'.format(
                    kat2.anc.sensor.air_temperature.get_value()))

    # TODO: If the script runs on built subarray receptors only then
    #       this step is redundant.
    # Select which receptors to run the script on
    if opts.receptors == 'all':
        ant_active = sorted(
            [ant.name for ant in kat.ants])
    else:
        ant_active = sorted(
            [ant.name for ant in kat.ants if ant.name in opts.receptors])

    log_message('Active antennas : {}'.format(', '.join(ant_active)),
                'info', boldtype=False, colourtext='blue')

    # This sensor is sampled only once every ten minutes since the last
    # change.  Get sensor data from last archive_search days
    log_message('Minimum runtime for vacuum pump lubrication is: {} minutes'.format(
                opts.run_duration), boldtype=True, colourtext='green')
    log_message('Minimum runtime between vacuum pump lubrication events is: {} days'.format(
                opts.lubrication_frequency), boldtype=True, colourtext='green')
    log_message('Fetching sensor data for the last {} days.  This will take a few minutes....\n'
                .format(opts.archive_search), boldtype=True, colourtext='green')

    if 'rts' in str(kat.katconfig.site):
        server = 'portal.mkat-rts.karoo.kat.ac.za'
    else:
        server = 'portal.mkat.karoo.kat.ac.za'

    err_results = []
    # check that vacuum pumps are ready:
    log_message('Checking that receptor vacuum pumps are ready')
    if not(kat.dry_run):
        for ant in kat.ants:
            if ant.name in ant_active:
                try:
                    if not ant.sensor.rsc_rsc_vac_pump_ready.get_value():
                        ant_active.remove(ant.name)
                        err_results.append(ant.name)
                        log_message(
                            '{} - Vacuum pump not ready. '
                            '{} removed from vacuum pump lubrication run.'
                            .format(ant.name, ant.name), 'warn')
                except AttributeError:
                    ant_active.remove(ant.name)
                    err_results.append(ant.name)
                    log_message(
                        '{} - Vacuum pump not ready. '
                        '{} removed from vacuum pump lubrication run.'
                        .format(ant.name, ant.name), 'warn')

    log_message('Remaining active antennas : {}\n'
                .format(', '.join(ant_active)), boldtype=False, colourtext='blue')

    # check that receptors are below max_elevation before starting:
    log_message(
        'Checking that receptors are below elevation of {} degrees (within 1 deg)'
        .format(opts.max_elevation))
    if not(kat.dry_run):
        for ant in kat.ants:
            if ant.name in ant_active:
                if ant.sensor.ap_actual_elev.get_value() > (opts.max_elevation + 1):
                    ant_active.remove(ant.name)
                    err_results.append(ant.name)
                    log_message(
                        '{} at elevation of {:0.1f}. '
                        '{} removed from vacuum pump lubrication run.'
                        .format(ant.name, ant.sensor.ap_actual_elev.get_value(), ant.name), 'warn')
                else:
                    log_message('{} at elevation of {:0.1f}.'.format(
                        ant.name, ant.sensor.ap_actual_elev.get_value()))

    log_message('Remaining active antennas : {}\n'
                .format(', '.join(ant_active)), boldtype=False, colourtext='blue')

    if not(kat.dry_run):
        if ant_active:
            reached_pressure = []
            # begin lubrication process.  Only use Receptors that are locked on
            # target

            # log RSC L band Manifold Pressure
            log_message(
                'Capturing L band manifold pressure before starting vacuum pumps')
            for ant in ant_active:
                ant_proxy = getattr(kat, ant)
                try:
                    log_message('{} L band manifold pressure is {:0.5f} mBar'.format(
                        ant, ant_proxy.sensor.rsc_rxl_manifold_pressure.get_value()))
                except AttributeError:
                    log_message(
                        '{} - Error reading manifold pressure'.format(ant), 'warn')

            send_email(['About to start Vacuum Pump Lubrication.',
                        'Beginning of report is as follows:  \n\n\n'] + email_msg,
                       '{} - {} - PRELIMINARY EMAIL FOR VAC PUMP LUBRICATION REPORT'
                       .format(str(kat.katconfig.site), opts.sb_id_code))

            log_message('Begin vacuum pump lubrication on active receptors. Lubrication will take {} minutes'
                        .format(int(opts.run_duration)), boldtype=True,)
            try:
                for ant in ant_active:
                    rsc_device = connect_to_rsc(ant, 7148)
                    if rsc_device:
                        log_message('{} : set vacuum pump to ITC'.format(ant))
                        response = rsc_device.req.rsc_vac_pump('itc')
                        # response = rsc_device.sensor.rxl_rfe1_temp_select.get_value()
                        log_message('{} - {}'.format(ant, str(response)),
                                    boldtype=True, colourtext='blue')
                        if not kat.dry_run:
                            # provide 2 seconds for vac pump to transition to ITC before starting vac pump
                            time.sleep(2)
                        log_message('{} : Start Vacuum Pump'.format(ant))
                        # response = rsc_device.sensor.rsc_he_compressor_pcb_current.get_value()
                        response = rsc_device.req.rsc_vac_pump('start')
                        log_message('{} - {}'.format(ant, str(response)),
                                    boldtype=True, colourtext='blue')
                        rsc_device.stop()

                log_message('Confirm that vacuum pumps are running')
                for ant in ant_active:
                    ant_proxy = getattr(kat, ant)
                    try:
                        if ant_proxy.sensor.rsc_rsc_vac_pump_running:
                            log_message('{} - confirmed: vacuum pump running'.format(ant),
                                        boldtype=True)
                        else:
                            log_message('{} - Vacuum pump NOT running'.format(ant), 'warn')
                            err_results.append(ant)
                    except AttributeError:
                        log_message('{} - Unable to read vacuum pump running sensor'
                                    .format(ant), 'warn')
                        err_results.append(ant)

                # wait for run_duration minutes
                start_run_duration = time.time()
                while (time.time() - start_run_duration < (opts.run_duration * 60)):
                    if int(time.time() - start_run_duration) % 60 < 5:
                        log_message('{} out of {} minutes completed'.format(
                            int((time.time() - start_run_duration) / 60),
                            int(opts.run_duration)))
                    time.sleep(5)
                    # Check every 5 seconds that receptors remain below
                    # max_elevation during the vacuum pump run
                    for ant in ant_active:
                        if (ant not in err_results):
                            ant_proxy = getattr(kat, ant)
                            if ant_proxy.sensor.ap_actual_elev.get_value() > opts.max_elevation:
                                log_message(
                                    '{} - Test failed - receptor currently at {:0.1f} degrees elevation'
                                    .format(ant_proxy.name, ant_proxy.sensor.ap_actual_elev.get_value()),
                                    'error')
                                err_results.append(ant)
                            if (ant_proxy.name not in reached_pressure):
                                try:
                                    pressure = ant_proxy.sensor.rsc_rxl_manifold_pressure.get_value()
                                    if pressure <= opts.ideal_vac_pressure:
                                        reached_pressure.append(ant_proxy.name)
                                        log_message('{} L band manifold pressure reached {:0.5f} mBar'
                                                    .format(ant_proxy.name, pressure),
                                                    boldtype=False, colourtext='blue')
                                except:
                                    log_message('{} - Error reading manifold pressure'
                                                .format(ant_proxy.name), 'warn')

                log_message('{} out of {} minutes completed'.format(
                            int(opts.run_duration), int(opts.run_duration)))
                log_message('Vacuum pump lubrication completed\n')

            finally:
                # Enable the vacuum pump again
                final_pressure = dict()
                for ant in ant_active:
                    ant_proxy = getattr(kat, ant)
                    rsc_device = connect_to_rsc(ant, 7148)
                    if rsc_device:
                        log_message('{} - Enable vacuum pump'.format(ant))
                        # response = rsc_device.sensor.rsc_he_compressor_pcb_current.get_value()
                        response = rsc_device.req.rsc_vac_pump('enable')
                        log_message('{} - {}'.format(ant, str(response)),
                                    boldtype=True, colourtext='blue')
                        rsc_device.stop()
                    try:
                        final_pressure[
                            ant] = ant_proxy.sensor.rsc_rxl_manifold_pressure.get_value()
                    except AttributeError:
                        final_pressure[ant] = -1
                log_message('Completed enabling vacuum pumps\n\n')

                log_message('Final L band manifold pressure',
                            boldtype=True, colourtext='green')
                for ant in final_pressure.keys():
                    if final_pressure[ant] != -1:
                        if final_pressure[ant] <= opts.ideal_vac_pressure:
                            log_message('{} - Final L band manifold pressure is {:0.5f} mBar'
                                        .format(ant, final_pressure[ant]), colourtext='green')
                        else:
                            log_message('{} - Final L band manifold pressure is {:0.5f} mBar'
                                        .format(ant, final_pressure[ant]), 'warn')
                    else:
                        log_message(
                            '{} - Error reading manifold pressure'.format(ant), 'warn')

                # Wait for vac pumps to report they are on before reading their state
                time.sleep(2)

                # Read back vacuum pump state
                for ant in ant_active:
                    ant_proxy = getattr(kat, ant)
                    try:
                        log_message(
                            '{} - Vacuum pump state: {}'
                            .format(ant, ant_proxy.sensor.rsc_rsc_vac_pump_state.get_value()),
                            boldtype=False, colourtext='blue')
                    except AttributeError:
                        log_message(
                            '{} - Error reading vacuum pump state'.format(ant), 'warn')

                # Print the receptors where an error occurred (eg. went above max
                # elevation)
                for ant in err_results:
                    log_message(
                        '{} - Vacuum pump lubrication failed on Receptor {}'.format(ant, ant), 'error')

                # Check which receptors should be included in the run (based on
                # lubrication frequency and run duration during that time)
                log_message('Checking history on failed lubrication runs\n', boldtype=True)
                read_sensor_history(err_results)

                log_message('Checking history on antennas in maintenance\n', boldtype=True)
                read_sensor_history(kat2.katpool.sensor.resources_in_maintenance.get_value().split(','))

                log_message("Vacuum Pump Lubrication: stop", boldtype=True)

                send_email(email_msg, '{} - {} - Vac Pump Lubrication Report'.format(
                           str(kat.katconfig.site), opts.sb_id_code))
        else:
            log_message(
                'No receptors to run vacuum pump lubrication on.\n', 'warn')
            send_email(email_msg, 'Unsuccessful - {} - {} - Vac Pump Lubrication Report'.format(
                       str(kat.katconfig.site), opts.sb_id_code))
    else:
        if kat.dry_run:
            log_message(
                'Dry Run only\n')
