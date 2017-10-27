#!/usr/bin/python
# Send an intervention to syscontroller to configure a new subarray

import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger
from katcorelib import cambuild

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="Instruct sys to configure a new subarray")

# 45 core ants
core_ants = ['m000','m001','m002','m003','m004','m005','m006','m007','m008','m009',
             'm010','m011','m012','m013','m014','m015','m016','m017','m018','m019',
             'm020','m021','m022','m023','m024','m025','m026','m027','m028','m029',
             'm030','m031','m032','m033','m034','m035','m036','m037','m038','m039',
             'm040','m041','m042','m043','m047']

# 19 non core ants
non_core_ants = ['m044','m045','m046','m048','m049','m050','m051','m052','m053','m054',
                 'm055','m056','m057','m058','m059','m060','m061','m062','m063']

parser.add_option('--time-delay', type='int', default=0,
    help="Delay (in seconds) before syscontroller starts the action (default=%default)")
parser.add_option('--band', type='string', default=None,
    help="Band to configure, using current band if None (default=%default)")
parser.add_option('--user-product', type='string', default=None,
    help="User product to configure, using current user product if None (default=%default)")
parser.add_option('--dump-rate', type='float', default=None,
    help="The dump rate used by SDP for this subarray in Hz. (default=%default)")
parser.add_option('--next-subnr', type='int', default=None,
    help="Next subarray to configure, using first free subarray if None (default=%default)")
parser.add_option('--resources', type='string', default=None,
    help="List of antennas and controlled resources to assign to next subarray, using current resources if None (default=%default)")

parser.set_defaults(description = 'CAM next subarray')
(opts, args) = parser.parse_args()
user_logger.info("CAM next subarray: start")

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
    user_logger.info("Opts:\n{}".format(opts))
    try:
        cam = None
        current = {}
        new = {}
        current["subnr"] = getattr(kat.sensors,"sub_sub_nr").get_value()
        current["band"] = getattr(kat.sensors,"sub_band").get_value()
        current["product"] = getattr(kat.sensors,"sub_product").get_value()
        current["dump_rate"] = getattr(kat.sensors,"sub_dump_rate").get_value()
        current["pool_resources"] = getattr(kat.sensors,"sub_pool_resources").get_value()
        current["active_sbs"] = getattr(kat.sensors,"sub_active_sbs").get_value()
        current["allocations"] = getattr(kat.sensors,"sub_allocations").get_value()
        user_logger.info("Current subarray: {}".format(current))

        # Build the CAM object
        user_logger.info('Begin cambuild...')
        cam = cambuild('camcam', conn_clients="sys,katpool", full_control=True)
        user_logger.info("Waiting for sys to sync")
        ok = cam.sys.until_synced(timeout=15)
        if not ok:
            user_logger.error("Sys did not sync \n{}\n\n".format(cam.get_status()))
            user_logger.error("Aborting script")
            raise RuntimeError("Aborting - Sys did not sync \n%s\n\n" % (cam.get_status()))

        # Use next free subarray if not specified
        new["current_subnr"] = int(current["subnr"])
        # If next_subnr is not specified use the current subnr
        new["next_subnr"] = int(opts.next_subnr or current["subnr"])
        new["delay"] = opts.time_delay
        new["pool_resources"] = opts.resources if opts.resources else current["pool_resources"]

        # Manage the generic resources as per katcamconfig:
        #   In this step, we are looking for specific resources mapped to the current
        #   subarray, e.g. sdp_1 in subarray 1, which need new names for the next
        #   subarray, e.g. sdp_2 in subarray 2.  Specs that are generic don't need to be
        #   modified.
        # From config we have 'generic_to_specific_resources' as a dict of lists, e.g.:
        #   {'data': ['data_', 'dbe7'], 'cbf': ['cbf_roach_', 'cbf_skarab_'], ...}
        # If a prefix ends with an underscore, it means it is subarray-specific, i.e.
        # the number of the subarray gets appended.
        prefix_lists = cam.katconfig.generic_to_specific_resources.values()
        for prefix_list in prefix_lists:
            for prefix in prefix_list:
                if prefix.endswith('_'):
                    new["pool_resources"] = new["pool_resources"].replace(
                        "{}{}".format(prefix, new["current_subnr"]),
                        "{}{}".format(prefix, new["next_subnr"]))
        new["band"] = opts.band or current["band"]
        new["product"] = opts.user_product or current["product"]
        new["dump_rate"] = opts.dump_rate or current["dump_rate"]
        user_logger.info("New subarray: {}".format(new))

        user_logger.info("Sending intervention to sys using subarray next_subnr %s", new["next_subnr"])
        keys = ('current_subnr', 'next_subnr', 'delay', 'band', 'product', 'pool_resources')
        items = ["{}={}".format(k, new[k]) for k in keys]
        cmd = "cam.sys.req.change_subarray({})".format(", ".join(items))
        user_logger.info("Command: %s", cmd)

        if not kat.dry_run:
            try:
                response = cam.sys.req.change_subarray(
                    new['current_subnr'], new['next_subnr'], new['delay'],
                    new['band'], new['product'],
                    new['pool_resources'], new['dump_rate'])
                log_info(response)
            except:
                user_logger.exception("Error in sys intervention")
                raise
    finally:
        if kat:
            kat.disconnect()
        if cam:
            cam.disconnect()

user_logger.info("CAM next subarray: stop")
