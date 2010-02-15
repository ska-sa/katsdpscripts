import katpoint
import ffuilib
from ffuilib import CaptureSession
import uuid

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
good_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A']
good_cat = katpoint.Catalogue([cat[src] for src in good_sources], add_specials=False, antenna=cat.antenna)

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'Baseline calibration example', ff.ants) as session:

    for target in good_cat.iterfilter(el_limit_deg=5):
        session.track(target, duration=600.0, drive_strategy='longest-track')
