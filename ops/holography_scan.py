# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import katpoint
import katuilib
from katuilib import CaptureSession
import uuid

kat = katuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=kat.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')
all_ants = katuilib.Array('ants', [kat.ant1, kat.ant2])
scan_ants = katuilib.Array('scan_ants', [kat.ant2])

with CaptureSession(ff, str(uuid.uuid1()), 'ffuser', 'Holography example', all_ants) as session:

    for compscan, target in enumerate(cat.iterfilter(el_limit_deg=5)):
        if not target.name.endswith('A'):
            continue
        session.holography_scan(scan_ants, target)
        if compscan >= 10:
            break
