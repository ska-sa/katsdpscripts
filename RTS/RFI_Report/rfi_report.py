#!/usr/bin/python
import optparse
from katsdpscripts.RTS import generate_flag_table, generate_rfi_report
import os

#command-line parameters
parser = optparse.OptionParser(usage="Please specify the input file\n\
    USAGE: python rfi_report.py <inputfile.h5> ",
    description="Produce a report detailing RFI detected in the input dataset")

parser.add_option("-a", "--antenna", type="string", default=None, help="Name of the antenna to produce the report for, default is first antenna in file")
parser.add_option("-t", "--targets", type="string", default=None, help="List of targets to produce report for, default is all targets in the file")
parser.add_option("-f", "--freq_chans", default=None, help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 90% of the bandpass.")
parser.add_option("-o", "--output_dir", default='.', help="Directory to place output .pdf report. Default is cwd")
parser.add_option("-s", "--static_flags", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', help="Location of static flags pickle file.")
opts, args = parser.parse_args()

# if no enough arguments, raise the runtimeError
if len(args) < 1:
    raise RuntimeError("No file passed as argument to script")

filename = args[0]

flags_basename=os.path.join(opts.output_dir,os.path.splitext(filename.split('/')[-1])[0]+'_flags')

generate_flag_table(filename,output_root=opts.output_dir,static_flags=opts.static_flags)
generate_rfi_report(filename,input_flags=flags_basename+'.h5',output_root=opts.output_dir,antenna=opts.antenna,targets=opts.targets,freq_chans=opts.freq_chans)


