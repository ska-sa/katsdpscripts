import katpoint
import ffuilib
from ffuilib import CaptureSession
import uuid
import math

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

targets = [
    ff.sources["Fornax A"],
    ff.sources["J0440-4333"],
    ff.sources["Zenith"],
    katpoint.construct_azel_target(math.radians(0.0), math.radians(45.0)),
    katpoint.construct_azel_target(math.radians(0.0), math.radians(20.0)),
]

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'RFI data collection', ff.ants) as session:

    ff.dbe.req.k7w_write_raw(1)

    for target in targets:
        session.track(target, duration=300.0, drive_strategy='shortest-slew')
