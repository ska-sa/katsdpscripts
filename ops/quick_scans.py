import katpoint
import ffuilib
from ffuilib import CaptureSession
import uuid

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'Quick scan example', ff.ants) as session:

    for target in cat.iterfilter(el_limit_deg=5):
        session.raster_scan(target)
