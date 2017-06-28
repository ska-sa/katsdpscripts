#!/usr/bin/python

import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger
from katcorelib import cambuild, kat_resource
import numpy as np
import os
import katcp
import smtplib
from email.mime.text import MIMEText

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
    description="Check receptors are in suitable position and command the RSC to lubricate vacuum pump for a specified duration.  Note: AMBIENT TEMP MUST BE ABOVE 16 DEG C")
parser.add_option('--max_elevation', type='int', default=20,
    help="Make sure receptor stays below this elevation for duration of vacuum pump lubrication (default=%default degrees)")
parser.add_option('--run_duration', type='int', default=20,
    help="Minimum run duration of vacuum pump (default=%default minutes)")
parser.add_option('--lubrication_frequency', type='int', default=14,
    help='Frequency for running vacuum pump lubrication (default=%default days)')
parser.add_option('--archive_search', type='int', default=30,
    help='Search sensor archive for this many days to check when vacuum pump was last run (default=%default days)')
parser.add_option('--receptors', type='str', default='all',
    help='List of receptors to run vacuum pump lubrication on (default=%default)')
parser.add_option('--ideal_vac_pressure', type='float', default=5e-2,
    help='Pressure to which the vacuum must go when pump operating (default=%default)')
parser.add_option('--email_to', type='str', default='blunsky@ska.ac.za,bjordaan@ska.ac.za,jvanstaden@emss.co.za,bjordaan@emss.co.za,operators@ska.ac.za',
    help='Comma separated email list of people to send report to (default=%default)')

parser.set_defaults(description = 'Lubricate Vacuum Pumps on Receivers')
(opts, args) = parser.parse_args()

email_msg = []

#create log timestamp format
def timestamp():
    x = time.gmtime()
    return str((str(x[0]) + '-' + str('%02d' % x[1]) + '-' + str('%02d' % x[2]) + ' ' + str('%02d' % x[3]) + ':' + str('%02d' % x[4]) + ':' + str('%02d' % x[5]) + 'Z '))

def log_message(msg, level = 'info', boldtype = False, colourtext = 'black'):
    
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

    if not(bold) and colour == 'black':
        email_msg.append(timestamp() + level.upper() + ' ' + str(msg))
    elif colour != 'black' and not(bold):
        email_msg.append(('<font color="%s">' % colour)  + timestamp() + level.upper() + ' ' + str(msg) + '</font>')
    else:
        email_msg.append(('<font color="%s"><b>' % colour)  + timestamp() + level.upper() + ' ' + str(msg) + '</b></font>')
        

log_message("Lubricate Vacuum Pumps: start\n", boldtype = True,)

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
    
def connect_to_rsc(ant, port):
    rsc_interface = {'controlled': True,
                     'description': 'RSC maintenance interface.'}
    rsc_interface['name'] = 'rsc_{}'.format(ant.name)
    rsc_interface['address'] = ('10.96.%s.20' % (int(ant.name[2:])), port)
    log_message('Connecting to RSC at %s at IP address %s on port %i' % (ant.name, rsc_interface['address'][0], rsc_interface['address'][1]))
    try:
        dev_katcp = katcp.resource_client.KATCPClientResource(rsc_interface)
        rsc_device = kat_resource.make_resource_blocking(dev_katcp, kat.ioloop_manager)
        rsc_device.until_synced()
        return rsc_device
    except:
        log_message('Failed to connect to RSC on %s' % (ant.name), ('error'))
        return []
        
def fetch_archive_sensor_data(sensorname,starttime,endtime,data_type,server='portal.mkat.karoo.kat.ac.za'):
    """A procedure to fetch sensor data from a server"""
    filename = '/tmp/%s_data.csv' % sensorname
    res = os.system('wget -q "http://%s/katstore/samples?sensor=%s&start=%i&end=%i&limit=1000000&time_type=s&format=csv" -O %s' % (server,sensorname,int(starttime),int(endtime), filename));
    if res:
        log_message('Error getting CSV for sensor %s, perhaps reset the HDF server process on CAM' % sensorname, 'warn')
    try:
        return np.genfromtxt(filename, dtype=data_type, delimiter=',', names = True)        # read CSV
        res = os.system('rm %s' % filename)
    except:
        log_message('Error retrieving sensor data from %s' % server, 'error')
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
    
    
with verify_and_connect(opts) as kat:
    print "_______________________"
    print kat.controlled_objects
    print kat.ants.clients
    print opts
    print "_______________________"
    log_message("Opts:\n{}\n".format(opts))
    try:
        cam = None
        MAINT_PORT = 7148
        LOCK_DELAY = 300  #seconds
        
        #check that lubrication frequency is less than the time we look back into the sensor archive
        if opts.archive_search < opts.lubrication_frequency:
            log_message("Waiting script - archive_search parameter must be > lubrication_frequency", 'error')
            raise RuntimeError("Aborting - archive_search parameter must be > lubrication_frequency\n\n")

        # Build the CAM object
        log_message('Begin cambuild...', 'info')
        cam = cambuild(password='camcam', conn_clients='all', full_control=True)
        log_message("Waiting for sys to sync\n", 'info')
        ok = cam.sys.until_synced(timeout=15)
        time.sleep(10)

        if not ok:
            log_message("Sys did not sync \n{}\n\n".format(cam.get_status()), 'error')
            log_message("Aborting script", 'error')
            raise RuntimeError("Aborting - Sys did not sync \n%s\n\n" % (cam.get_status()))

        #Ambient should be above 16 deg c.
        if (cam.anc.sensor.air_temperature.get_value() < 16) and not(kat.dry_run):
            log_message("Aborting script - ambient temperature is below 16 deg C", 'error')
            raise RuntimeError("Aborting script - ambient temperature is below 16 deg C\n\n")
        log_message('Current Ambient temperature is %0.2f' % cam.anc.sensor.air_temperature.get_value())
        
        #Select which receptors to run the script on
        if opts.receptors == 'all':
            ant_active = sorted([ant.name for ant in cam.ants if ant.name not in cam.katpool.sensor.resources_in_maintenance.get_value()])
        else:
            ant_active = sorted([ant.name for ant in cam.ants if ant.name in opts.receptors])
        
        log_message('Active antennas : %s' % ', '.join(ant_active), 'info', boldtype = False, colourtext = 'blue')
        
        #This sensor is sampled only once every ten minutes since the last change.  Get sensor data from last archive_search days
        log_message('Minimum runtime for vacuum pump lubrication is: %i minutes' % opts.run_duration, boldtype = True, colourtext = 'green')
        log_message('Minimum runtime between vacuum pump lubrication events is: %i days' % opts.lubrication_frequency, boldtype = True, colourtext = 'green')
        log_message('Fetching sensor data for the last %i days.  This will take a few minutes....\n' % opts.archive_search, boldtype = True, colourtext = 'green')

        if 'rts' in str(cam.katconfig.site):
            server = 'portal.mkat-rts.karoo.kat.ac.za'
        else:
            server = 'portal.mkat.karoo.kat.ac.za'

        if not(kat.dry_run):
            #Check which receptors should be included in the run (based on lubrication frequency and run duration during that time)
            temp_ant_active = [ant for ant in ant_active]
            for ant in ant_active:
                try:
                    vac_running = []
                    vac_running = fetch_archive_sensor_data('%s_rsc_rsc_vac_pump_running' % ant, time.time() - (60 * 60 * 24 * (opts.archive_search)), time.time(), data_type = "S100,f8,Bool", server = server)
                    if len(vac_running) != 0:
                        vac_running_true = np.where(vac_running['Value'])[0]
                        groups = group_consecutives(vac_running_true)
                        vacpump_run_duration = []
                        vacpump_run_timestamp = []
                        vacpump_elevation_dict = dict()
            
                        for x in groups:
                            if x != []:
                                vacpump_run_duration.append((vac_running['Timestamp'][x[-1] + 1] - vac_running['Timestamp'][x[0]]) / 60.0)
                                vacpump_run_timestamp.append(vac_running['Timestamp'][x[-1] + 1])
                        if vacpump_run_timestamp:
                            log_message('%s last test completed : ' % ant, 'info')
                            for i,x in enumerate(vacpump_run_timestamp):
                                x_gmtime = time.gmtime(x)
                                elev_vals = fetch_archive_sensor_data('%s_ap_requested_elev' % ant, x - (vacpump_run_duration[i] * 60), x, data_type="S100,f8,f8")
                                vacpump_elevation_dict[x] = np.mean(elev_vals['Value'])
                                log_message('\t%d-%02d-%02d %02d:%02d:%02d UTC, \t%i minutes duration,\t %i days ago \t %0.2f Average Elevation' % 
                                    (x_gmtime[0], x_gmtime[1], x_gmtime[2], x_gmtime[3], x_gmtime[4], x_gmtime[5], int(vacpump_run_duration[i]), int((time.time() - x) / 3600 / 24), 
                                    vacpump_elevation_dict[x]))
                            vacpump_recent_runs = np.where(np.array(vacpump_run_timestamp) >= time.time() - (60 * 60 * 24 * (opts.lubrication_frequency)))[0]
                            need_to_run = True
                            for x in vacpump_recent_runs:
                                if (vacpump_run_duration[x] >= opts.run_duration) and (vacpump_elevation_dict[vacpump_run_timestamp[x]] <= (opts.max_elevation + 1)):
                                        need_to_run = False
                                        break
                            if need_to_run:
                                log_message('%s - Scheduling for Vacuum Pump Lubrication - No runs long enough below specified elevation\n' % (ant))
                            else:
                                temp_ant_active.remove(ant)
                                log_message('%s - Vacuum Pump Lubrication not required\n' % (ant), boldtype = True)
                        else:
                            log_message('%s - Scheduling for Vacuum Pump Lubrication - no record over the last %i days\n' % (ant, opts.archive_search))
                    else:
                        log_message('%s - Scheduling for Vacuum Pump Lubrication - unable to extract sensor data from archive\n' % (ant), 'warn')

                except ValueError:
                    log_message('%s - Error reading and processing sensor data.' % ant, 'error')
            ant_active = temp_ant_active
            
            log_message('Remaining active antennas : %s\n' % ', '.join(ant_active), boldtype = False, colourtext = 'blue')

        err_results = []
        #check that vacuum pumps are ready:
        log_message('Checking that receptor vacuum pumps are ready')
        for ant in cam.ants:
            if ant.name in ant_active:
                try:
                    if not(ant.sensor.rsc_rsc_vac_pump_ready.get_value()):
                        ant_active.remove(ant.name)
                        err_results.append(ant.name)
                        log_message('%s - Vacuum pump not ready.  %s removed from vacuum pump lubrication run.' % (ant.name, ant.name), 'warn')
                except AttributeError:
                    ant_active.remove(ant.name)
                    err_results.append(ant.name)
                    log_message('%s - Vacuum pump not ready.  %s removed from vacuum pump lubrication run.' % (ant.name, ant.name), 'warn')

        log_message('Remaining active antennas : %s\n' % ', '.join(ant_active), boldtype = False, colourtext = 'blue')
        
        #check that receptors are below max_elevation before starting:
        log_message('Checking that receptors are below elevation of %i degrees (within 1 deg)' % opts.max_elevation)
        for ant in cam.ants:
            if ant.name in ant_active:
                if ant.sensor.ap_requested_elev.get_value() > (opts.max_elevation + 1):
                    ant_active.remove(ant.name)
                    log_message('%s at elevation of %0.1f.  %s removed from vacuum pump lubrication run.' % (ant.name, ant.sensor.ap_requested_elev.get_value(), ant.name), 'warn')
                else:
                    log_message('%s at elevation of %0.1f.' % (ant.name, ant.sensor.ap_requested_elev.get_value()))
        
        log_message('Remaining active antennas : %s\n' % ', '.join(ant_active), boldtype = False, colourtext = 'blue')
        
        reached_pressure = []
        #begin lubrication process.  Only use Receptors that are locked on target
        if ant_active and not(kat.dry_run):
            #log RSC L band Manifold Pressure
            log_message('Capturing L band manifold pressure before starting vacuum pumps')
            for ant in ant_active:
                ant_proxy = getattr(cam, ant)
                try:
                    log_message('%s L band manifold pressure is %0.5f mBar' % (ant, ant_proxy.sensor.rsc_rxl_manifold_pressure.get_value()))
                except AttributeError:
                    log_message('%s - Error reading manifold pressure' % ant, 'warn')

            send_email(['About to start Vacuum Pump Lubrication.', 'Beginning of report is as follows:  \n\n\n'] + email_msg, '%s - %s - PRELIMINARY EMAIL FOR VAC PUMP LUBRICATION REPORT' % (str(cam.katconfig.site), opts.sb_id_code))

            log_message('Begin vacuum pump lubrication on active receptors. Lubrication will take %i minutes' % int(opts.run_duration), boldtype = True,)
            for ant in ant_active:
                ant_proxy = getattr(cam, ant)
                rsc_device = connect_to_rsc(ant_proxy, 7148)
                if rsc_device != []:
                    log_message('%s : set vacuum pump to ITC' % ant)
                    response = rsc_device.req.rsc_vac_pump('itc')
                    #response = rsc_device.sensor.rxl_rfe1_temp_select.get_value()
                    log_message(('%s - ' % ant) + str(response), boldtype = True, colourtext = 'blue')
                    if not kat.dry_run:
                        time.sleep(2)
                    log_message('%s : Start Vacuum Pump' % ant)
                    #response = rsc_device.sensor.rsc_he_compressor_pcb_current.get_value()
                    response = rsc_device.req.rsc_vac_pump('start')
                    log_message(('%s - ' % ant) + str(response), boldtype = True, colourtext = 'blue')
                    rsc_device.stop() 

            log_message('Confirm that vacuum pumps are running')
            for ant in ant_active:
                ant_proxy = getattr(cam, ant)
                try:
                    if ant_proxy.sensor.rsc_rsc_vac_pump_running:
                        log_message('%s - confirmed: vacuum pump running' % ant, boldtype = True)
                    else:
                        log_message('%s - Vacuum pump NOT running' % ant, 'warn')
                        err_results.append(ant)
                except AttributeError:
                    log_message('%s - Unable to read vacuum pump running sensor' % ant, 'warn')
            
            #wait for run_duration minutes
            start_run_duration = time.time()
            count = 0
            while (time.time() - start_run_duration < (opts.run_duration * 60)):
                if int(time.time() - start_run_duration) % 60 == 0:
                    log_message('%i out of %i minutes completed' % (int((time.time() - start_run_duration) / 60), int(opts.run_duration)))
                time.sleep(1)
                count = count + 1
                #Check every 5 seconds that receptors remain below max_elevation during the vacuum pump run
                if count == 5:
                    for ant in ant_active:
                        if (ant not in err_results):
                            ant_proxy = getattr(cam, ant)
                            if ant_proxy.sensor.ap_requested_elev.get_value() > opts.max_elevation:
                                log_message('%s - Test failed - receptor currently at %0.1f degrees elevation' % (ant_proxy.name, ant_proxy.sensor.ap_requested_elev.get_value()), 'error')
                                err_results.append(ant)
                            if (ant_proxy.name not in reached_pressure):
                                try:
                                    pressure = ant_proxy.sensor.rsc_rxl_manifold_pressure.get_value()
                                    if pressure <= opts.ideal_vac_pressure:
                                        reached_pressure.append(ant_proxy.name)
                                        log_message('%s L band manifold pressure reached %0.5f mBar' % (ant_proxy.name, pressure), boldtype = False, colourtext = 'blue')
                                except:
                                    log_message('%s - Error reading manifold pressure' % ant_proxy.name, 'warn')
                    count = 0
                
            log_message('%i out of %i minutes completed' % (int(opts.run_duration), int(opts.run_duration)))
            log_message('Vacuum pump lubrication completed\n')
        
            #Enable the vacuum pump again
            final_pressure = dict()
            for ant in ant_active:
                ant_proxy = getattr(cam, ant)
                rsc_device = connect_to_rsc(ant_proxy, 7148)
                if rsc_device != []:
                    log_message('%s - Enable vacuum pump' % ant)
                    #response = rsc_device.sensor.rsc_he_compressor_pcb_current.get_value()
                    response = rsc_device.req.rsc_vac_pump('enable')
                    log_message(('%s - ' % ant) + str(response), boldtype = True, colourtext = 'blue')
                    rsc_device.stop()
                try:
                    final_pressure[ant] = ant_proxy.sensor.rsc_rxl_manifold_pressure.get_value()
                except AttributeError:
                    final_pressure[ant] = -1
            log_message('Completed enabling vacuum pumps\n\n')
            
            log_message('Final L band manifold pressure', boldtype = True, colourtext = 'green')
            for ant in final_pressure.keys():
                if final_pressure[ant] != -1:
                    if final_pressure[ant] <= opts.ideal_vac_pressure:
                        log_message('%s - Final L band manifold pressure is %0.5f mBar' % (ant, final_pressure[ant]), colourtext = 'green')
                    else:
                        log_message('%s - Final L band manifold pressure is %0.5f mBar' % (ant, final_pressure[ant]), 'warn')
                else:
                    log_message('%s - Error reading manifold pressure' % ant, 'warn')

            time.sleep(5)
            
            #Read back vacuum pump state
            for ant in ant_active:
                ant_proxy = getattr(cam, ant)
                try:
                    log_message('%s - Vacuum pump state: %s' % (ant, ant_proxy.sensor.rsc_rsc_vac_pump_state.get_value()), boldtype = False, colourtext = 'blue')
                except AttributeError:
                    log_message('%s - Error reading vacuum pump state' % ant, 'warn')

            #Print the receptors where an error occurred (eg. went above max elevation)
            for ant in err_results:
                log_message('%s - Vacuum pump lubrication failed on Receptor %s' % (ant, ant), 'error')
        else:
            log_message('No receptors to run vacuum pump lubrication on.\n', 'warn')
        
        log_message("Vacuum Pump Lubrication: stop", boldtype = True,)
        
        if not(kat.dry_run):
            send_email(email_msg, '%s - %s - Vac Pump Lubrication Report' % (str(cam.katconfig.site), opts.sb_id_code))

    finally:
        if kat:
            kat.disconnect()
        if cam:
            cam.disconnect()


