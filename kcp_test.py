from katcp import BlockingClient, MessageParser

import time
import sys

client = BlockingClient("127.0.0.1",1260)
 # connect to rf proxy
client.start()

time.sleep(1)
if not client.is_connected():
    client.stop()
    print "Failed to connect to remote"
    sys.exit()

mp = MessageParser()
msg = mp.parse("?sensor-list")

retval = client.blocking_request(msg)
print retval[0]
sensors = []
for msg in retval[1]:
    sensors.append(msg.arguments[0])

s1 = time.time()
for sensor in sensors:
    msg = mp.parse("?sensor-sampling " + sensor + " period 5000")
    s = time.time()
    retval = client.blocking_request(msg)
    if retval[0].arguments[0] == "ok":
        print "Set strategy for",sensor,"in",(time.time() - s)
    else:
        print "Failed to set strategy"

print "Total strategy time:",(time.time() - s1)
client.stop()
