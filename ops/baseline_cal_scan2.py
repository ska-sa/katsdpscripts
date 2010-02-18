import katpoint
import katuilib
from katuilib import CaptureSession
import uuid

kat = katuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=kat.sources.antenna)
good_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A']
good_cat = katpoint.Catalogue([cat[src] for src in good_sources], add_specials=False, antenna=cat.antenna)

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'Baseline calibration example', kat.ants) as session:

    for target in good_cat.iterfilter(el_limit_deg=5):
        session.track(target, duration=600.0, drive_strategy='longest-track')
