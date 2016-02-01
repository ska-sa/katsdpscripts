#!/usr/bin/python
# Track sources all around the sky for a few seconds each without recording data (mostly to keep tourists or antennas amused).
from __future__ import with_statement
import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger, start_session
from katcorelib import cambuild, katconf
from gmail import Gmail

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="AR1 Global Sync Script ver 2\n"+
                            "Performs a global sync,\n"+
                            "Starts data stream from digitisers,\n"+
                            "Halts AR1 array and programs the correlator")

#parser.add_option('--product', type="string", default='c856M4k',
#                  help='Specify the data product for the correlator (default="%default")')

parser.set_defaults(description = 'AR1 Global sync')
(opts, args) = parser.parse_args()
user_logger.info("global_sync_v2 script: start")

def log_info(response):
    response = str(response)
    if 'fail' in response:
        user_logger.warn(response)
    else:
        user_logger.info(response)
        
with verify_and_connect(opts) as kat:
    print "_______________________"
    print kat.controlled_objects
    print kat.ants.clients
    print opts
    print "_______________________"
    try:
        cam = None
        cont = False
        count = 1

        if not kat.dry_run:
            # pwd = time.ctime()[:10]+time.ctime()[-5:]
            # fpwd = pwd.translate(
            #         string.maketrans(string.digits, string.ascii_letters[0:10]))
            # cam = cambuild(password=pwd,
            #             full_control_password=fpwd,
            #             system="systems/"+katconf.sitename()+".conf",
            #             conn_clients="all")
            cam = cambuild(password="camcam", full_control="all")

            while not cont:
                log_info('Performing global sync on AR1 ...')
                log_info(cam.mcp.req.dmc_global_synchronise(timeout=30))
                time.sleep(5)

                # ant_active = [ant for ant in cam.ants if ant.name not in cam.katpool.sensor.maintenance_allocations.get_value()]
                ant_active = [ant for ant in cam.ants if ant.name not in cam.katpool.sensor.resources_in_maintenance.get_value()]
                for ant in ant_active:
                    response = ant.req.dig_capture_start('hv')
                    log_info(ant.name + ': ' + str(response))
                    time.sleep(1)

                log_info('Halting ar1 array...')
                # log_info(cam.mcp.req.array_halt('ar1'))
                log_info(cam.mcp.req.cmc_array_halt('1'))
                log_info('Waiting 5 seconds for things to settle')

                time.sleep(10)
                corrprod = opts.product
                if corrprod not in ('c856M4k', 'c856M32k'):
                    corrprod = 'c856M4k'
                    user_logger.warn('No correlation product specified. Using %s' % corrprod)

                log_info('Programming correlator product %s - this could take up to 5 minutes....................' % corrprod)
                # success = str(cam.data_rts.req.product_configure(corrprod, 1.0, timeout=300))
                success = str(cam.data_rts.req.product_configure(corrprod, 1.0, timeout=300))
                log_info(success)
                
                if '!product-configure ok' in success:
                    cont = True
                    user_logger.info('Programming correlator successful!')
                    time.sleep(2)
                else:
                    count = count + 1
                    user_logger.error('Failure to program correlator!!!  Trying again.....')
                    time.sleep(2)
                    
                if count % 10 == 0:
                    msg = Gmail('skadebug@gmail.com', 'skadebugpassword')
                    msg.send_message(['blunsky@ska.ac.za', 'pkotze@ska.ac.za'], 'AR1 Global Sync Failure', 
                    'Global sync and product configuration has failed %i times on AR1.  Current time is %s' % (count, time.ctime()))
            
            time.sleep(5)
            
            log_info('SPMC product deconfigure...')
            deconfig_string = 'rts_' + corrprod
            response = str(cam.data_rts.req.spmc_data_product_configure(deconfig_string,''))
            log_info(response)

            user_logger.info("Script complete")
    finally:
#        if kat:
#            kat.disconnect()
        if cam:
            cam.disconnect()
#
#with start_session(kat, **vars(opts)) as session:
