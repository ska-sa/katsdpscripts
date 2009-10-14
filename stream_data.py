#!/usr/bin/python
# Create a DBE data stream, capture it and send the data to the signal displays.

import ffuilib as ffui

ff = ffui.tbuild("cfg-user.ini", "integration_ff_client")
 # make fringe fingder connections

ff.k7w.req.capture_start()
 # startup the k7 capture process

ff.dbe.req.dbe_packet_count(900)
 # stream 10 minutes of data or until stop issued
ff.dbe.req.dbe_dump_rate(1)
 # correlator dump rate set to 1 Hz
ff.dbe.req.dbe_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
ff.dbe.req.dbe_capture_start("stream")
 # start emitting data on stream "stream"

ff.disconnect()
