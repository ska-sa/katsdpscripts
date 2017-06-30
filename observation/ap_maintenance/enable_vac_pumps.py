#!/usr/bin/python

import time
from katcorelib import standard_script_options
from katcorelib import kat_resource
import katcp

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(
    usage="usage: %prog [options]",
    description="Check receptors are in suitable position and command the RSC "
                "to lubricate vacuum pump for a specified duration. "
                "Note: AMBIENT TEMP MUST BE ABOVE 16 DEG C")
parser.add_option(
    '--receptors', type='str', default=None,
    help='List of receptors to run vacuum pump lubrication on (default=%default)')

parser.set_defaults(description='Lubricate Vacuum Pumps on Receivers')
(opts, args) = parser.parse_args()


def timestamp():
    # create log timestamp format
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def log_message(msg, level='info'):
    print('{timestamp} {level} {msg}'.format(
        timestamp=timestamp(), level=level.upper(), msg=str(msg)))


def connect_to_rsc(ant, port):
    rsc_interface = {'controlled': True,
                     'description': 'RSC maintenance interface.'}
    rsc_interface['name'] = 'rsc_{}'.format(ant)
    rsc_interface['address'] = ('10.96.{}.20'.format(int(ant[2:])), port)
    log_message('Connecting to RSC at {ant} at IP address {ip} on port {port}'.format(
        ant=ant, ip=rsc_interface['address'][0], port=rsc_interface['address'][1]))
    try:
        dev_katcp = katcp.resource_client.KATCPClientResource(rsc_interface)
        rsc_device = kat_resource.make_resource_blocking(dev_katcp)
        rsc_device.until_synced()
        return rsc_device
    except:
        log_message('Failed to connect to RSC on {}'.format(ant), ('error'))
        return []


def enable_vac_pump(ant):
    rsc_device = connect_to_rsc(ant, 7148)
    if rsc_device != []:
        log_message('{} - Enable vacuum pump'.format(ant))
        response = rsc_device.req.rsc_vac_pump('enable')
        log_message('{} - {}'.format(ant, str(response)))
        rsc_device.stop()


if __name__ == "__main__":
    if opts.receptors is None:
        print("Error. Specify which receptors to enable the vacuum pumps.")
    else:
        ants = opts.receptors.replace(' ', '')
        for x in ants.split(','):
            if x[0] != 'm':
                print("Error. Illegal antenna name: {}".format(x))
            else:
                enable_vac_pump(x)
