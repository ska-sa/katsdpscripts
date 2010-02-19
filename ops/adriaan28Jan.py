
#run check_defaults.py -i cfg-karoo.ini -s karoo_ff -r

import katuilib

kat = katuilib.tbuild('cfg-local.ini', 'local_ff')

lo1_freq = 6022
centre_freq = lo1_freq - 4200.0
effective_lo_freq = (centre_freq - 200.0) * 1e6
kat.rfe7.req.rfe7_lo1_frequency(lo1_freq, "MHz")
kat.ped1.req.rfe3_rfe15_noise_source_on("coupler", 1, "now", 1, 10240, 0.5)
kat.ped2.req.rfe3_rfe15_noise_source_on("coupler", 1, "now", 1, 10240, 0.5)

###kat.ant1.req.target_azel(0,90)
###kat.ant2.req.target_azel(0,90)
###kat.ant1.req.mode("POINT")
###kat.ant2.req.mode("POINT")
#### Wait until they are all in position (with 5 minute timeout)
###kat.ant1.wait("lock", True, 300)
###kat.ant2.wait("lock", True, 300)


# This is a precaution to prevent bad timestamps from the correlator
kat.dbe.req.dbe_sync_now()
kat.dbe.req.capture_setup(1000.0, effective_lo_freq)
####kat.dbe.req.capture_setup(512, 1.822)
###kat.dh.start_sdisp()
kat.dbe.req.k7w_output_directory("/var/kat/data/adriaan28Jan")
kat.dbe.req.k7w_baseline_mask(1,3)
kat.dbe.req.capture_start()

# let the capture run for approximately 1 hour

#kat.dbe.req.capture_stop()

# log on to ff-dc and then run "augment.py -f adriaan/*.h5". bring me
#the files in /var/kat/data/adriaan.


