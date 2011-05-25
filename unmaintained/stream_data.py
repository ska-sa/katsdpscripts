#!/usr/bin/python
# Create a DBE data stream, capture it and send the data to the signal displays.

import katuilib as katui

kat = katui.tbuild("cfg-local.ini", "local_ff")
 # make fringe fingder connections

kat.k7w.req.capture_start()
 # startup the k7 capture process

kat.dbe.req.dbe_packet_count(900)
 # stream 10 minutes of data or until stop issued
kat.dbe.req.dbe_dump_rate(1)
 # correlator dump rate set to 1 Hz
kat.dbe.req.dbe_capture_destination("stream","127.0.0.1:7010")
 # create a new data source labelled "stream". Send data to localhost on port 7010
kat.dbe.req.dbe_capture_start("stream")
 # start emitting data on stream "stream"

kat.disconnect()
