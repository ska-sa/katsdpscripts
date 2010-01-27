import katpoint
import ffuilib
import ffobserve

ff = ffuilib.tbuild('cfg-karoo.ini', 'karoo_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
good_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A']
good_cat = katpoint.Catalogue([cat[src] for src in good_sources], add_specials=False, antenna=cat.antenna)

ffobserve.setup(ff, ff.ants)
compscan_id = 0
for target in good_cat.iterfilter(el_limit_deg=5):
    ffobserve.track(ff, ff.ants, target.description, duration=600.0, compscan_id=compscan_id, drive_strategy='longest-track')
    compscan_id += 1
ffobserve.shutdown(ff)

ff.disconnect()
