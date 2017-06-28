#!/usr/bin/python

import time, string
from katcorelib import standard_script_options
from katcorelib import kat_resource
import os
import katcp

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
    description="Check receptors are in suitable position and command the RSC to lubricate vacuum pump for a specified duration.  Note: AMBIENT TEMP MUST BE ABOVE 16 DEG C")
parser.add_option('--receptors', type='str', default=None,
    help='List of receptors to run vacuum pump lubrication on (default=%default)')

parser.set_defaults(description = 'Lubricate Vacuum Pumps on Receivers')
(opts, args) = parser.parse_args()

email_msg = []

#create log timestamp format
def timestamp():
    x = time.gmtime()
    return str((str(x[0]) + '-' + str('%02d' % x[1]) + '-' + str('%02d' % x[2]) + ' ' + str('%02d' % x[3]) + ':' + str('%02d' % x[4]) + ':' + str('%02d' % x[5]) + 'Z '))

def log_message(msg, level = 'info', boldtype = False, colourtext = 'black'):
    print(timestamp() + level.upper() + ' ' + str(msg))

def connect_to_rsc(ant, port):
    rsc_interface = {'controlled': True,
                     'description': 'RSC maintenance interface.'}
    rsc_interface['name'] = 'rsc_{}'.format(ant)
    rsc_interface['address'] = ('10.96.%s.20' % (int(ant[2:])), port)
    log_message('Connecting to RSC at %s at IP address %s on port %i' % (ant, rsc_interface['address'][0], rsc_interface['address'][1]))
    try:
        dev_katcp = katcp.resource_client.KATCPClientResource(rsc_interface)
        rsc_device = kat_resource.make_resource_blocking(dev_katcp)
        rsc_device.until_synced()
        return rsc_device
    except:
        log_message('Failed to connect to RSC on %s' % (ant), ('error'))
        return []

def enable_vac_pump(ant):
    rsc_device = connect_to_rsc(ant, 7148)
    if rsc_device != []:
        log_message('%s - Enable vacuum pump' % ant)
        response = ''
        response = rsc_device.req.rsc_vac_pump('enable')
        log_message(('%s - ' % ant) + str(response))
        rsc_device.stop()
    
if __name__ == "__main__":
    if opts.receptors == None:
        print("Error. Specify which receptors to enable the vacuum pumps.")
    else:
        ants = opts.receptors.replace(' ','')
        for x in ants.split(','):
            if x[0] != 'm':
                print("Error. Illegal antenna name: %s" % x)
            else:
                enable_vac_pump(x)