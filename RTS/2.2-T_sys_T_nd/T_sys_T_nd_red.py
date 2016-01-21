#!/usr/bin/python
import optparse
from katsdpscripts.RTS import diodelib

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>",
                                   description=" This produces a pdf file with graphs verifying the ND model and Tsys for each antenna in the file")
    parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
    parser.add_option("--pdf", action='store_true',default=True, help="Print the output to a PDF")
    parser.add_option("--Ku", action='store_true',default=False, help="The specified file is a Ku band observation")
    parser.add_option("-v","--verbose", action='store_true',default=False, help="Print some debugging information")
    parser.add_option("--error_bars", action='store_true',default=False, help="Include error bars - Still in development")
    parser.add_option("--off_target", default='off1', help="which of the two off targets to use")

    (opts, args) = parser.parse_args()
    if len(args) ==0:
        raise RuntimeError('Please specify an h5 file to load.')
    
    return opts,args


if __name__ == "__main__":
    opts, args = parse_arguments()
    diodelib.read_and_plot_data(args[0],opts.output_dir,opts.pdf,opts.Ku,opts.verbose,opts.error_bars)




