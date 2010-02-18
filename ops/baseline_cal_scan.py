import katpoint
import katuilib
from katuilib import CaptureSession
import uuid

kat = katuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=kat.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'Baseline calibration example', kat.ants) as session:

    for target in cat.iterfilter(el_limit_deg=5):
        session.track(target, duration=60.0, drive_strategy='shortest-slew')
