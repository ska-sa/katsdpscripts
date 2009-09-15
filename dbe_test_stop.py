#!/usr/bin/python

import ffuilib

dbe_roach = ffuilib.build_device("dbe","192.168.10.111",7147)
dbe_roach.req_capture_stop("dram")

ffuilib.quitter()
