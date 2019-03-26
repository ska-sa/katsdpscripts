#!/usr/bin/python
import optparse
import os
parser = optparse.OptionParser(usage='%prog [options]',
                               description='This script finds and runs a reduction')

(opts, args) = parser.parse_args()
for cb_id in args:
    filename = 'http://archive-gw-1.kat.ac.za:7480/%s/%s_sdp_l0.rdb'%(cb_id,cb_id)
    params = "--aperture-efficiency=/home/kat/katconfig/user/aperture-efficiency/mkat/ --nd-models=/home/kat/katconfig/user/noise-diode-models/mkat/ --receiver-models=/home/kat/katconfig/user/receiver-models/mkat/ --spill-over-models=/home/kat/katconfig/user/spillover-models/mkat  --channel-mask=/home/kat/katsdpscripts/RTS/rfi_mask_UHF.pickle"
    scriptname = "/home/kat/katsdpscripts/RTS/2.1-Tipping_Curve/red_tipping_curve.py"
    reduction = "%s %s"%(scriptname,params)
    os.system('cd /data  ; python %s %s '%(reduction, filename ))
