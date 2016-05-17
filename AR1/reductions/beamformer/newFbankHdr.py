#!/usr/bin/env python

import os, sys, time
from optparse import OptionParser   # enables variables to be parsed from command line
import numpy as np
from sigpyproc.Readers import FilReader
import matplotlib.pyplot as plt

###########################
# Define useful routines.
###########################
def power_of_two(n):
  """Check if value is a power of two."""
  if n == 2:
    return True
  elif n%2 != 0:
    return False
  else:
    return power_of_two(n/2.0)

if __name__=="__main__":
	parser = OptionParser(usage="%prog <options>", description='Write filterbank file with new header.')
	parser.formatter.max_help_position = 100
	parser.formatter.width = 250
	parser.add_option("-i", "--infile", type="string", default="1462454844_pol0_t32.fil", help="Input file override (default = %default).")
	parser.add_option("-b", "--nbits", type="int", default=32, help="Number of bits in input file (default = %default).")
	parser.add_option("-d", "--decimate", type="int", default=1, help="Decimation factor override (default = %default).")
	parser.add_option("--fch1", type="float", default=1444.4, help="First channel override (default = %default MHz).")
	parser.add_option("--fo", type="float", default=-0.208984, help="Channel bandwidth override (default = %default MHz).")
	parser.add_option("-m", "--mjd", type="float", default=57513.560696400004, help="MJD start override (default = %default).")
	parser.add_option("-M", "--mach", type="int", default=64, help="Backend override (default = 64, i.e. bfi1).")
	parser.add_option("-n", "--nchan", type="int", default=512, help="Nchan override (default = %default).")
	parser.add_option("-o", "--outfile", type="string", default=None, help="Outfile override (default = %default MHz).")
	parser.add_option("-s", "--source", type="string", default="J0835-4510", help="Source override (default = %default MHz).")
	parser.add_option("-t", "--tsamp", type="float", default=153.122e-6, help="tsamp override (default = %default s).")
	parser.add_option("-T", "--tel", type="int", default=64, help="Telescope ID override (default = 64, i.e. MeerKAT).")
	(opts, args) = parser.parse_args()
	t0 = time.time() # record script start time

    ###############################
    # Write new hdr to file.
    ###############################
	outfile = opts.infile.split('.fil')[0] + '_newhdr.fil' if not opts.outfile else opts.outfile
	hdr_cmd = 'mockHeader -tel %i -mach %i -type 1 -source %s -tstart %.8f -tsamp %s -nbits %i '\
	'-fch1 %.5f -fo %.6f -nchans %i %s' %(opts.tel,opts.mach,opts.source,opts.mjd,opts.tsamp,opts.nbits,
		opts.fch1,opts.fo,opts.nchan,outfile)
	os.system(hdr_cmd)

    ######################################
    # Read in filterbank data as mmap and
    # append to new filterbank file.
    ######################################
	fbank = FilReader(opts.infile)
	if ( opts.nbits == 8 ):
		fil = np.memmap(opts.infile,offset=fbank.header.hdrlen,dtype=np.uint8,mode='r')
	elif ( opts.nbits == 32 ):
		fil = np.memmap(opts.infile,offset=fbank.header.hdrlen,dtype=np.float32,mode='r')

	Nsamp = int(fil.size/float(opts.nchan))
	print '\n Number of samples in file = %i' %Nsamp
	
	#--------------------------------------#
	# Re-write data to file using chunker:
	#--------------------------------------#
	indices = np.arange(0,Nsamp,2048)
	if ( opts.decimate > 1 ): 
		check = power_of_two(opts.decimate)
		if not check:
			print ' Decimation factor must be a power of two. Exiting now...\n'
			sys.exit(1)
		Nsamp = int(Nsamp/opts.decimate)
		print ' Resampling data to %i samples...' %Nsamp

	print ' Writing raw data to file..\n'
	f_handle = open(outfile,"ab+")
	for i in indices:
		isamp = i*opts.nchan
		fsamp = (i+2048)*opts.nchan
		if ( opts.decimate == 1 ):
			spec = np.copy(fil[isamp:fsamp])
		else:
			spec = np.copy(fil[isamp:fsamp]).reshape((-1,opts.decimate,opts.nchan)).mean(axis=1)
			spec = spec.flatten().astype(np.float32)

		f_handle.seek(0, 2)
		bytesSpec = spec.tobytes(order="C")
		f_handle.write(bytesSpec)
	f_handle.close()

	#----------------------#
	# Print total run time:
	#----------------------#
	t1 = time.time()
	print ' Total elapsed time: ' + str(round(t1-t0,2)) + ' secs, (' + str(round((t1-t0)/60.0,2)) + ' mins)'
