#!/usr/bin/python

# Test out the DBE data capture. Needs the roach to be running and configured with IP 192.168.10.111
# k7w_server should be running on ff-dc (you can use the start_int_dc script)
# ffsocket should be running on ff-proxy (you can use the start_int_proxy script)

import ffuilib

dbe_roach = ffuilib.build_device("dbe","192.168.10.111",7147)
k7w = ffuilib.build_device("k7w","192.168.10.3",8001)

k7w.req.capture_start()
dbe_roach.req.capture_destination("dram","192.168.10.3",7010)
dbe_roach.req.poco_accumulation_length(1000)
dbe_roach.req.poco_gain("0x",10000)
dbe_roach.req.poco_gain("0y",20000)
dbe_roach.req.poco_gain("1",5000)
dbe_roach.req.capture_start("dram")

ffuilib.quitter()
