import katpoint
import ffuilib
import ffobserve
import math

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

targets = [
    ff.sources["Fornax A"],
    ff.sources["J0440-4333"],
    ff.sources["Zenith"],
    katpoint.construct_azel_target(math.radians(0.0), math.radians(45.0)),
    katpoint.construct_azel_target(math.radians(0.0), math.radians(20.0)),
]

ffobserve.setup(ff, ff.ants)
ff.dbe.req.k7w_write_raw(1)

compscan_id = 0
for target in targets:
    ffobserve.track(ff, ff.ants, target.description, duration=300.0, compscan_id=compscan_id, drive_strategy='shortest-slew')
    compscan_id += 1

ffobserve.shutdown(ff)

ff.disconnect()
