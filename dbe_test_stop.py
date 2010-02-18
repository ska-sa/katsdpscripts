#!/usr/bin/python

import katuilib

dbe_roach = katuilib.build_device("dbe","192.168.10.111",7147)
dbe_roach.req.capture_stop("dram")

katuilib.quitter()
