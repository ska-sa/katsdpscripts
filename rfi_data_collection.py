import katpoint
import katuilib
from katuilib import CaptureSession
import uuid
import math

kat = katuilib.tbuild('cfg-local.ini', 'local_ff')

targets = [
    kat.sources["Fornax A"],
    kat.sources["J0440-4333"],
    kat.sources["Zenith"],
    katpoint.construct_azel_target(math.radians(0.0), math.radians(45.0)),
    katpoint.construct_azel_target(math.radians(0.0), math.radians(20.0)),
]

with CaptureSession(kat, str(uuid.uuid1()), 'ffuser', 'RFI data collection', kat.ants) as session:

    kat.dbe.req.k7w_write_raw(1)

    for target in targets:
        session.track(target, duration=300.0, drive_strategy='shortest-slew')
