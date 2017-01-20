#!/usr/bin/python
# Global sync with all the trimmings
#
# Initial script by Benjamin for RTS
# Updated for AR1 by Ruby
# Benjamin added PPS offsets
# Added timestamp of sync output for easier use
# Tiyani replaced dig_sync_epoch with dig_l_band_time_sync

from __future__ import with_statement
import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger, start_session
from katcorelib import cambuild, katconf

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="AR1 Global Sync Script ver 2\n"+
                            "Performs a global sync,\n"+
                            "Starts data stream from digitisers,\n"+
                            "Halts AR1 array and programs the correlator")
parser.add_option('--delayfile', type="string", default='pps_delays.csv',
                  help='Specify the full path to the csv file containing receptor delays in the format m0xx, <delay> (default="%default")')
parser.add_option('--mcpsetband', type="string", default='',
                  help='If specified, script will call cam.mcp.req.set_band() with given parameter (default="%default")')
parser.add_option('--with-array', action="store_true", default=False,
                  help='Build a new array for user after sync')
parser.add_option('--all', action="store_true", default=False,
                  help='Include all antennas in the global sync')
# assume basic options passed from instruction_set
parser.set_defaults(description = 'AR1 Global sync')
(opts, args) = parser.parse_args()
print("global_sync_AR1 script: start")

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
        done = False
        count = 1

        if not kat.dry_run:
	    print('Building CAM object')
            cam = cambuild(password="camcam", full_control="all")
	    time.sleep(5)

            delay_list = {}
            try:
                for line in open(opts.delayfile):
                    x = ((line.strip('\n')).split(','))
                    if (len(x[0]) == 4 and x[0][0] == 'm'):
                        delay_list[x[0]] = int(x[1])
                        print('Receptor: %s  delay: %s' % (x[0], x[1]))
            except:
                raise RuntimeError('Failure to read pps delay file!')

#                 if (opts.mcpsetband == 'x' or opts.mcpsetband == 'l'):
#                     print('mcpsetband has been specified as %s' % opts.mcpsetband)
#                     print(str(cam.mcp.req.set_band(opts.mcpsetband)))
# RvR -- AR1 is locked into L-band for now (remove when other bands can be selected again)
                if (opts.mcpsetband == 'x'):
		    raise RuntimeError('Unavailable band: mcpsetband has been specified as %s' % opts.mcpsetband)
# RvR -- AR1 is locked into L-band for now (remove when other bands can be selected again)

            if opts.all:
                ant_active = cam.ants
            else:
                ant_active = [ant for ant in cam.ants if ant.name not in cam.katpool.sensor.resources_in_maintenance.get_value()]
            print('Set PPS delay compensation for digitisers')
            for ant in ant_active:
                #look at current delay and program in delay specified in CSV
                if ant.name in delay_list:
                    # set the delay compensations for a digitiser (assuming L band)
                    response = ant.req.dig_digitiser_offset('l')
                    curr_delay_l = int(str(response).split(' ')[2])
                    if curr_delay_l == delay_list[ant.name]:
                        print(ant.name + ': no change to PPS delay offset')
                    else:
# RvR -- AR1 not needed to check indexer -- M063 fixed
#                         # determine if we are on X or L band
#                         indexer_pos_raw = int(ant.sensor.ap_indexer_position_raw.get_value())
#                         if (indexer_pos_raw < 20) or (indexer_pos_raw > 60):
# # RvR -- AR1 is locked into L-band for now (remove when other bands can be selected again)
#                             raise RuntimeError(ant.name + " is on X band")
# # RvR -- AR1 is locked into L-band for now (remove when other bands can be selected again)
# #                                 print(ant.name + " is on X band")
# #                                 #look at current delay and program in delay specified in CSV
# # 		                  # set the delay compensations for a digitiser
# #                                 response = ant.req.dig_digitiser_offset('x')
# #                                 curr_delay_x = int(str(response).split(' ')[2])
# #                                 print(ant.name + ' X-band current delay : ' + str(curr_delay_x))
# #                                 response = ant.req.dig_digitiser_offset('x', delay_list[ant.name])
# #                                 print(ant.name + ' X-band PPS delay offset : ' + str(response))
# # RvR -- AR1 not needed to check indexer -- M063 fixed
#                         else:
                        print(ant.name + " is on L band")
                        print(ant.name + ' L-band current delay : ' + str(curr_delay_l))
                        response = ant.req.dig_digitiser_offset('l', delay_list[ant.name])
                        print(ant.name + ' L-band PPS delay offset : ' + str(response))

# RvR -- THIS IS A VERY BIG NOTE:
# Currently sync is done in serial format one digitiser at a time, which means that the timeout will have to be
# almost doubled for every 2 new antennas for the known future, until some optimization can be implemented
            print('Performing global sync on AR1 ...')
	    serial_sync_timeout=300 # seconds
            for n in xrange(2):
                # doing it twice just for good measure
                start_time = time.time()
                cam.mcp.req.dmc_global_synchronise(timeout=serial_sync_timeout)
                print("Duration of global sync: {} try number {}"
                      .format(time.time() - start_time, n))
                time.sleep(5)
# RvR -- THIS IS A VERY BIG NOTE:
# Currently sync is done in serial format one digitiser at a time, which means that the timeout will have to be
# almost doubled for every 2 new antennas for the known future, until some optimization can be implemented

            print('Reiniting all digitisers ...')
            antlist=''
            for ant in ant_active:
                if antlist: antlist=','.join((antlist,ant.name))
                else: antlist=ant.name
                response = ant.req.dig_capture_start('hv', timeout=60)
                print(ant.name + ': ' + str(response))
                time.sleep(1)

            while not done:
# RvR -- For the moment assume always subarray_1 -- need to follow up with cam about knowing which is active
		print('Halting ar1 array...')
		cam.subarray_1.req.free_subarray(timeout=30)
		print('Waiting 5 seconds for things to settle')
		time.sleep(10)

		# Do not build new array by default
		if not opts.with_array:
		    break

		corrprod = opts.product
		if corrprod not in ('c856M4k', 'c856M32k'):
		    corrprod = 'c856M4k'
		    print('No correlation product specified. Using %s' % corrprod)

		cam.subarray_1.req.free_subarray(timeout=30)
		print('Waiting 5 seconds for things to settle')
		time.sleep(10)
		print('Building new subarray, this may take a little time....')
 		cam.subarray_1.req.set_band('l')
		time.sleep(1)
                cam.subarray_1.req.set_product(corrprod)
		time.sleep(1)
 		cam.subarray_1.req.assign_resources('data_1,'+antlist)
		time.sleep(1)
		response=cam.subarray_1.req.activate_subarray(timeout=600)
# RvR -- For the moment assume always subarray_1 -- need to follow up with cam about knowing which is active

                print("The response from array activate: =={}==".format(str(response)))
		if 'ok' in str(response):
                    done = True
                    print('Programming correlator successful!')
                    print('Subarray 1 active!')
                else:
                    count = count + 1
                    print('Failure to program CBF or SP!!!  Trying again.....')
	        time.sleep(10)

                if count > 5:
		    print('Cannot auto-activate subarray, giving up.....')
		    break

            time.sleep(5)
	    etime = cam.mcp.sensor.dmc_synchronisation_epoch.get_value()
	    for ant in ant_active:
		if int(ant.sensor.dig_l_band_time_synchronisation_epoch.get_value()) != int(etime):
		    raise RuntimeError('System not synced, investigation is required...')
                else:
                   print '%s sync epoch:  %d' % (ant.name, ant.sensor.dig_l_band_time_synchronisation_epoch.get_value())
            print '\n'
 
	    import time
            print("Script complete")
    finally:
        if cam:
	    print("Cleaning up cam object")
            cam.disconnect()

    print("\nGlobal Sync Date %s" % time.ctime(etime))

# -fin-
