import katpoint
import ffuilib
import ffobserve

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')
ants = [ff.ant1, ff.ant2]

ffobserve.setup(ff, ants)
compscan_id = 0
for target in cat.iterfilter(el_limit_deg=5):
    ffobserve.track(ff, ants, target.description, duration=60.0, compscan_id=compscan_id, drive_strategy='shortest-slew')
    compscan_id += 1
ffobserve.shutdown(ff)

ff.disconnect()
