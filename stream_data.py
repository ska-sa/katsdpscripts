#!/usr/bin/python
# Create a DBE data stream, capture it and send the data to the signal displays.

import ffuilib as ffui

ff = ffui.cbuild("ffuilib.local.rc")
 # make fringe fingder connections

ff.k7w.req_capture_start()
 # startup the k7 capture process

ff.dbesim.req_packet_count(5000)
 # stream 5000 packets of data
ff.dbesim.req_rate(300)
 # stream data at 300 kbps. Approx 1s per integration

ff.dbesim.req_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbesim.req_capture_start("stream")
 # start emitting data on stream "stream"
 
 # we leave the stream running and disconnect from the various devices
ff.disconnect()
