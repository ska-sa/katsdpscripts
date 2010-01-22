import katpoint
import ffuilib
import ffobserve

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')

ffobserve.observation_setup(ff)
compscan_id = 0
for target in cat.iterfilter(el_limit_deg=5):
    ffobserve.raster_scan(ff, target.description, 3, compscan_id=compscan_id)
    compscan_id += 1
ffobserve.observation_shutdown(ff)

ff.disconnect()