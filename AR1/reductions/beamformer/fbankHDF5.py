#!/usr/bin/env/python 
# fbankHDF5.py; last mod. NJY [17/05/2016]

import os, time, h5py
import numpy as np
from optparse import OptionParser   # enables variables to be parsed from command line
import matplotlib.pyplot as p
from matplotlib.ticker import AutoMinorLocator

#################################################
# MeerKAT CBF HDF5 -> filterbank file converter.
#################################################
# Functions used in SIGPROC header creation
def _write_string(key, value):
    return "".join([struct.pack("I", len(key)), key, struct.pack("I", len(value)), value])

def _write_int(key, value):
    return "".join([struct.pack("I",len(key)), key, struct.pack("I", value)])

def _write_double(key, value):
    return "".join([struct.pack("I",len(key)), key, struct.pack("d", value)])

def _write_char(key, value):
    return "".join([struct.pack("I",len(key)), key, struct.pack("b", value)])

def get_times(adc_count,sync_time):  # calculate time info
    """Define function to get MJD time from ADC samples."""
    sf = 1712.0e6
    ts = np.float128(adc_count) / sf + sync_time
    utc = time.strftime('%Y-%m-%d-%H:%M:%S',time.gmtime(ts)) + str(ts-int(ts))[1:]
    mjd = np.float128(ts/np.float128(86400.0)+40587.0)
    return utc, mjd

def get_h5data(file):
    """Return h5py data and time stamp objects from HDF5 file."""
    data = h5py.File(file,'r')
    bfData = data['Data/bf_raw']
    tStamps = data['Data/timestamps']
    sync_time = data['TelescopeModel/cbf'].attrs['sync_time']
    return bfData, tStamps, sync_time

def check_offsets(uniqOffsets):
    """Print out locations of anomalous ADC offsets."""
    for offset in uniqOffsets:
        if ( offset != 8192) :
            indices = np.where(uniqOffsets==offset)[0]
            print ' ADC mismatch of %i samples at...' %(offset/8192.), indices  

def power_of_two(n):
  """Check if value is a power of two."""
  if n == 2:
    return True
  elif n%2 != 0:
    return False
  else:
    return power_of_two(n/2.0)

#####################
# Main routine.
#####################
if __name__=="__main__":
    parser = OptionParser(usage="%prog <options>", 
    description='Convert HDF5 data to the SIGPROC filterbank format.')
    parser.formatter.max_help_position = 50
    parser.formatter.width = 200
    parser.add_option("-c", "--chbw", action='store_false', default=True, 
    help="Channel bandwidth inversion override (def = %default).")
    parser.add_option("-d", "--dec", type='string', default="-45:10:34.8751", 
    help="DECJ override (def = %default).")   
    parser.add_option("-f", "--freq", type="float", default=1391.0, 
    help="Centre frequency override (def = %default MHz).")
    parser.add_option("-g", "--get-fig", action="store_false", default=True, 
    help="Figure output override switch (def = %default).")
    parser.add_option("--i0", type="string", default='/mnt/ramdisk0/1463411901.h5', 
    help="pol0 input file override (def = %default).")
    parser.add_option("--i1", type="string", default='/mnt/ramdisk1/1463411900.h5', 
    help="pol1 input file override (def = %default).")
    parser.add_option("-n", "--ndec", type="int", default=32, 
    help="Decimation factor override (def = %default).") 
    parser.add_option("-o", "--outfile", type="string", default=None, 
    help="Output file override (default = %default).")
    parser.add_option("-r", "--ra", type='string', default="08:35:20.61149", 
    help="DECJ override (def = %default).")   
    parser.add_option("-s", "--source", type="string", default='J0835-4510', 
    help="Source override (default = %default).")
    parser.add_option("-S", "--sync_time", type="float", default=None, 
    help="sync_time override (default = %default).")
    parser.add_option("-t", "--tot_time", type="float", default=None, 
    help="Total data read length override (default = %default (s)).")
    (opts, args) = parser.parse_args()
    t0 = time.time() # record script start time

    ################################
    # Read in HDF5 data using h5py.
    ################################
    pol0 = opts.i0
    pol1 = opts.i1
    bf0, t0s, sync_time = get_h5data(pol0)
    bf1, t1s, sync_time = get_h5data(pol1)
    nchan = bf0.shape[0]
    if opts.sync_time != None: sync_time = opts.sync_time
    if bf1.shape[0] != nchan:
        print ' %i chans (pol0) != %i chans (pol1). Exiting now...\n' %(nchan,bf1.shape[0])
        sys.exit(1)
    print bf0
    print bf1

    ##################################################
    # Determine ADC offset to synchonise pol streams
    # and poke timestamps to check for data losses.
    ##################################################
    t0s = t0s[:]
    t1s = t1s[:]
    pol0_diffs = np.diff(t0s)
    pol1_diffs = np.diff(t1s)
    p0_diffUniq = np.unique(pol0_diffs) # check where dropouts actually occur
    p1_diffUniq = np.unique(pol1_diffs)
    check_offsets(p0_diffUniq)
    check_offsets(p1_diffUniq)

    adc0 = t0s[0]
    adc1 = t1s[0]
    adcSync = adc0 if adc0 > adc1 else adc1
    index0 = np.where(t0s==adcSync)[0][0]
    index1 = np.where(t1s==adcSync)[0][0]
    Nend = Nsamp = t1s[index1:].size if t1s[index1:].size <= t0s[index0:].size else t0s[index0:].size

    Npol0 = t0s.size
    Npol1 = t1s.size
    tsamp = 1/(856e6/4096.)
    Nend0 = Nend+index0
    Nend1 = Nend+index1
    UTCstart, MJDstart = get_times(adcSync,sync_time)
    print '\n ADC sync indices for H = %i and V = %i' %(index0,index1)
    print ' Nsamp for H = %i and V = %i\n' %(Npol0,Npol1)
    print ' Nend for H = %i and V = %i\n' %(Nend0,Nend1)
    print ' Tobs for H = %i s and V = %i s\n' %(Npol0*tsamp,Npol1*tsamp) 
    print ' UTC start = %s => MJD start = %s' %(UTCstart,MJDstart)
    print ' Unique pol0 ADC offsets:', p0_diffUniq
    print ' Unique pol1 ADC offsets:', p1_diffUniq, '\n'

    #-----------------------------------------------#
    # Optionally modify Nend to reduce data read in:
    #-----------------------------------------------#  
    if ( opts.tot_time != None ):
        Nsamp_use = np.int_(opts.tot_time/tsamp)
        Nsamp_use = Nsamp_use - Nsamp_use%256 + 256 
        if ( Nsamp_use < Nend ):
            Nsamp = Nsamp_use
            Nend0 = Nsamp+index0
            Nend1 = Nsamp+index1
            t0s = t0s[index0:Nend0]
            t1s = t1s[index1:Nend1]
        print ' New Nsamp for H = %i and V = %i' %(t0s.size,t1s.size)
        print ' New Tobs = %.3f s' %(Nsamp*tsamp)    

    ####################################################
    # Check decimation factor and define header params.
    ####################################################
    if ( opts.ndec > 1 ): 
        check = power_of_two(opts.ndec)
        if not check or opts.ndec > 256:
            print ' Decimation factor must be a power of two and <= 256. Exiting now...\n'
            sys.exit(1)
    Nsamp = np.int_(float(Nsamp)/opts.ndec)
    tsamp *= opts.ndec

    chBW = 1712/8192. # MHz
    freqTop = opts.freq + (((nchan / 2) - 1) * chBW)
    freqBottom = opts.freq - (((nchan / 2)) * chBW)
    if opts.chbw: 
        fBottom = opts.freq + (((nchan / 2) - 1) * chBW)
        fTop = opts.freq - (((nchan / 2)) * chBW)
        chBW *= -1
    else:
        fTop = opts.freq + (((nchan / 2) - 1) * chBW)
        fBottom = opts.freq - (((nchan / 2)) * chBW)
    print ' fBottom = %.3f MHz, fTop = %.3f MHz, chBW = %.5f MHz' %(fBottom,fTop,chBW)
    print ' tsamp to be used = %.6f ms' %(tsamp*1e3)

    ########################################
    # Write out file header to fbank file.
    #########################################
    RAJ = opts.ra
    DECJ = opts.dec
    outfile = opts.outfile if opts.outfile != None else opts.i0.split('.h5')[0] + '.fil'
    f_handle = open(outfile, "ab+")
    headerStart = "HEADER_START"
    headerEnd = "HEADER_END"
    header = "".join([struct.pack("I", len(headerStart)), headerStart])
    header = "".join([header, _write_string("source_name", opts.source)])
    header = "".join([header, _write_int("machine_id", 64)])
    header = "".join([header, _write_int("telescope_id", 64)])
    src_raj = float(RAJ.replace(":", ""))
    header = "".join([header, _write_double("src_raj", src_raj)])
    src_dej = float(DECJ.replace(":", ""))
    header = "".join([header, _write_double("src_dej", src_dej)])
    header = "".join([header, _write_int("data_type", 1)])
    header = "".join([header, _write_double("fch1", fBottom)])
    header = "".join([header, _write_double("foff", chBW)])
    header = "".join([header, _write_int("nchans", nchan)])
    header = "".join([header, _write_int("nbits", 32)])
    header = "".join([header, _write_double("tstart", MJDstart)])
    header = "".join([header, _write_double("tsamp", tsamp)])
    header = "".join([header, _write_int("nifs", 1)])
    header = "".join([header, struct.pack("I", len(headerEnd)), headerEnd])
    f_handle.write(header)

#    hdr_cmd = '/home/kat/software/mockHeader -tel 64 -mach 64 -type 1 -raw %s -source %s -tstart %.8f -nbits 32 ' %(opts.i0,opts.source,MJDstart)
#    hdr_cmd += '-nifs 1 -tsamp %.10f -fch1 %.6f -fo %.6f -nchans %i %s' %(tsamp,fBottom,chBW,nchan,outfile)
#    os.system(hdr_cmd)

    ##############################################################
    # Power detect, re-accumulate and combine H & V pol. streams.
    ##############################################################
    start = 0
    Nblock = 256/opts.ndec
    i0s = np.int_(np.arange(index0,Nend0,256)) # 256 spectra allows fast cache access c.f. 8192
    i1s = np.int_(np.arange(index1,Nend1,256))
    bf0pows = np.zeros((nchan,Nsamp),dtype=np.float32)
    bf1pows = np.zeros((nchan,Nsamp),dtype=np.float32)
    totPows = np.zeros((nchan,Nsamp),dtype=np.float32)
    for index in np.arange(i0s.size):
        pow0 = bf0[:,i0s[index]:i0s[index]+256,:]
        pow1 = bf1[:,i1s[index]:i1s[index]+256,:]
        pow0 = pow0[...,0] + pow0[...,1]*1j
        pow1 = pow1[...,0] + pow1[...,1]*1j
        pow0 = (pow0 * pow0.conjugate()).real
        pow1 = (pow1 * pow1.conjugate()).real

        if ( opts.ndec > 1 ):
            try:
                pow0 = pow0.reshape((nchan,-1,opts.ndec)).mean(axis=2)  # re-accumulate data by a factor of ndec
                pow1 = pow1.reshape((nchan,-1,opts.ndec)).mean(axis=2)
            except ValueError: # break from loop if run out of data
                break

        totPow = pow0 + pow1
        totSpec = totPow.transpose().flatten()
        bytesSpec = totSpec.astype(np.float32).tobytes(order="C")
        f_handle.write(bytesSpec) # write sigproc data fmt to file
        f_handle.seek(0,2)
    
        bf0pows[:,start:start+Nblock] = pow0
        bf1pows[:,start:start+Nblock] = pow1
        totPows[:,start:start+Nblock] = totPow

        start += Nblock
    f_handle.close()
    t1 = time.time()

    #########################################
    # Plot average bandpass and time series.
    #########################################
    if opts.get_fig:
        p.clf()
        params = {'axes.labelsize': 18, 'text.fontsize': 14, 'legend.fontsize': 11, 
                'xtick.labelsize': 16, 'ytick.labelsize': 16, 'text.usetex': False}
        p.rcParams.update(params)

        #-------------------------#
        # Plot average bandpass:
        #-------------------------#
        ax1 = fig.add_subplot(211)
        bp0 = bf0pows.mean(axis=1)
        bp1 = bf1pows.mean(axis=1)
        bptot = totPows.mean(axis=1)

        p.plot(bp0,'b-',label='H')
        p.plot(bp1,'g-',label='V')
        p.plot(bptot,'r-',label='cmbd')
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
        p.legend(numpoints=1,loc='upper right')
        p.ylabel('Power (A.U.)')
        p.xlabel('Nchan')

        #------------------------#
        # Plot time-series data:
        #------------------------#
        ts0 = bf0pows.mean(axis=0)
        ts1 = bf0pows.mean(axis=0)
        ts_tot = totPows.mean(axis=0)
        tvals = np.arange(Nsamp)*tsamp

        p.plot(tvals,ts0,'b-',label='H')
        p.plot(tvals,ts1,'g-',label='V')
        p.plot(tvals,ts_tot,'r-',label='cmbd')
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(10))
        p.legend(numpoints=1,loc='upper right')
        p.ylabel('Power (A.U.)')
        p.xlabel('Time (s)')
        p.subplots_adjust(hspace=0.2,bottom=0.15,left=0.15,right=0.975,top=0.975)
        p.savefig('/scratch/fbank_miscPlots.png',orientation='landscape',papertype='a4', pad_inches=0)

    duration = t1 - t0
    print ' Elapsed time = %.2f s (%.3f mins)' %(duration,duration/60.)
