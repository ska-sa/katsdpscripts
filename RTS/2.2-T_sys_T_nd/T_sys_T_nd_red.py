#!/usr/bin/python
import optparse
from katsdpscripts.RTS import diodelib

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>",
                                   description=" This produces a pdf file with graphs verifying the ND model and Tsys for each antenna in the file")
    parser.add_option("-f", "--frequency-bandwidth", dest="freq_band", type="float", default='256e6',
                      help="BAndwidth of frequency channels to keep. Default = %default")
    parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
    (opts, args) = parser.parse_args()
    if len(args) ==0:
        raise RuntimeError('Please specify an h5 file to load.')
    
    return opts,args


if __name__ == "__main__":
    opts, args = parse_arguments()
    diodelib.read_and_plot_data(args[0],opts.output_dir,opts.freq_band)




