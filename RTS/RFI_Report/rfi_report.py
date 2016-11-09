#!/usr/bin/python -W ignore
import warnings
warnings.simplefilter('ignore')
import optparse
from katsdpscripts.RTS import generate_flag_table, generate_rfi_report
import os

#command-line parameters
parser = optparse.OptionParser(usage="Please specify the input file\nUSAGE: python rfi_report.py <inputfile.h5> ",
    description="Produce a report detailing RFI detected in the input dataset")

parser.add_option("-a", "--antennas", type="string", default=None, help="Comma separated list of antennas to produce the report for, default is all antennas in file")
parser.add_option("-t", "--targets", type="string", default=None, help="List of targets to produce report for, default is all targets in the file")
parser.add_option("-f", "--freq_chans", default=None, help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 90% of the bandpass.")
parser.add_option("-o", "--output_dir", default='.', help="Directory to place output .pdf report. Default is cwd")
parser.add_option("-s", "--static_flags", default=None, help="Location of static flags pickle file.")
parser.add_option("--ku-band", action='store_true', help="Force ku-band observation")
parser.add_option("--write-input", action='store_true', help="Make a copy of the input h5 file and insert flags into it.")
parser.add_option("--flags-only", action='store_true', help="Only calculate flags (no rfi-report).")
parser.add_option("--report-only", action='store_true', help="Only generate RFI report (use flags from file).")
parser.add_option("--width-freq", type="float", default=3.0, help="Frequency width for background smoothing in MHz")
parser.add_option("--width-time", type="float", default=40.0, help="Time width for background smoothing in seconds")
parser.add_option("--freq-extend", type="int", default=3, help="Convolution width in channels to extend flags")
parser.add_option("--time-extend", type="int", default=3, help="Convolution width in dumps to extend flags")
parser.add_option("--outlier-sigma", type="float", default=4.5, help="Number of sigma to threshold for flags in single channel sumthreshold iteration")
opts, args = parser.parse_args()

# if no enough arguments, raise the runtimeError
if len(args) < 1:
    raise RuntimeError("No file passed as argument to script")

filename = args[0]

basename = filename.split('/')[-1]
flags_basename=os.path.join(opts.output_dir,os.path.splitext(basename)[0]+'_flags')

if opts.ku_band:
	opts.static_flags=None

if opts.write_input:
	input_flags=None
	report_input = os.path.join(opts.output_dir,basename)
else:
	input_flags = flags_basename+'.h5'
	report_input=filename

if opts.flags_only:
	generate_flag_table(filename,output_root=opts.output_dir,static_flags=opts.static_flags,write_into_input=opts.write_input,outlier_sigma=opts.outlier_sigma,
						width_freq=opts.width_freq,width_time=opts.width_time,freq_extend=opts.freq_extend,time_extend=opts.time_extend)
elif opts.report_only:
	generate_rfi_report(filename,input_flags=None,output_root=opts.output_dir,antenna=opts.antennas,targets=opts.targets,freq_chans=opts.freq_chans)
else:
	generate_flag_table(filename,output_root=opts.output_dir,static_flags=opts.static_flags,write_into_input=opts.write_input,outlier_sigma=opts.outlier_sigma,
						width_freq=opts.width_freq,width_time=opts.width_time,freq_extend=opts.freq_extend,time_extend=opts.time_extend)
	generate_rfi_report(report_input,input_flags=input_flags,output_root=opts.output_dir,antenna=opts.antennas,targets=opts.targets,freq_chans=opts.freq_chans)
