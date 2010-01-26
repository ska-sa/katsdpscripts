import katpoint
import ffuilib
import ffobserve

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')
scan_ants = ffuilib.Array('scan_ants', [ff.ant2])
all_ants = ffuilib.Array('ants', [ff.ant1, ff.ant2])

ffobserve.setup(ff, all_ants)
compscan_id = 0
for target in cat.iterfilter(el_limit_deg=5):
    if not target.name.endswith('A'):
        continue
    ffobserve.holography_scan(ff, all_ants, scan_ants, target.description, compscan_id=compscan_id)
    compscan_id += 1
    if compscan_id >= 10:
        break
ffobserve.shutdown(ff)

ff.disconnect()
