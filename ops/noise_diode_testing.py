import katpoint
import ffuilib
from ffuilib import CaptureSession
import uuid
import time

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

with CaptureSession(ff, str(uuid.uuid4()), 'ffuser', 'Noise diode testing',
                    ff.ants, centre_freq=1822, dump_rate=512.0 / 1000.0):

    ff.peds.req.rfe3_rfe15_noise_source_on("coupler", 1, "now", 1, 10240, 0.5)

    ff.ants.req.target(ff.sources["Zenith"].description)
    ff.ants.req.mode("POINT")
    ff.ants.wait("lock", True, 300)

    ff.dbe.req.capture_start()

    # let the capture run for approximately 1 hour
    time.sleep(60.0 * 60)
