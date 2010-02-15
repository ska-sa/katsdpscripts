import katpoint
import ffuilib
from ffuilib import CaptureSession
import uuid

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')
all_ants = ffuilib.Array('ants', [ff.ant1, ff.ant2])
scan_ants = ffuilib.Array('scan_ants', [ff.ant2])

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'Holography example', all_ants) as session:

    for compscan, target in enumerate(cat.iterfilter(el_limit_deg=5)):
        if not target.name.endswith('A'):
            continue
        session.holography_scan(scan_ants, target)
        if compscan >= 10:
            break
