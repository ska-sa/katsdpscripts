###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
#for use in holography
def plots():
        kat.dh.register_dbe(kat.dbe)
        kat.dbe.req.capture_stop()
        kat.dbe.req.capture_setup()
        kat.dbe.req.capture_start()
        kat.dh.stop_sdisp()
        kat.dh.start_sdisp()
        s=kat.dh.sd.plot_spectrum('mag',products=[(1,1,"HH"),(2,2,"HH"),(1,2,"HH")],start_channel=360,stop_channel=380)
        print "Spectra setup in s"
        p=kat.dh.sd.plot_time_series('mag',products=[(1,2,"HH")],start_channel=370,stop_channel=371,end_time=-1000)
        "a default plot setup in p"

def startup():
        kat.dbe.req.dbe_holo_attenuation('0x',0)
        kat.dbe.req.dbe_holo_attenuation('0y',5)
        w2m=kat.sources["EUTELSAT W2M"]
        is10=kat.sources["INTELSAT 10 (IS-10)"]
        is1r=kat.sources["INTELSAT 1R (IS-1R)"]
        print "setup satellites w2m is10 and is1r "
        print "just use kat.ant1.req.target(is10) etc."
        #kat.ant1.req.mode("POINT")

        
