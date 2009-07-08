#!/usr/bin/python
# Create a DBE data stream, capture it and send the data to the signal displays.

import ffuilib as ffui

ff = ffui.tbuild("cfg-telescope.ini", "local-simulated-ff")
 # make fringe fingder connections

ff.k7w.req_capture_start()
 # startup the k7 capture process

ff.dbe.req_dbe_packet_count(50000)
 # stream 5000 packets of data
ff.dbe.req_dbe_rate(300)
 # stream data at 300 kbps. Approx 1s per integration

ff.dbe.req_dbe_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbe.req_dbe_capture_start("stream")
 # start emitting data on stream "stream"
