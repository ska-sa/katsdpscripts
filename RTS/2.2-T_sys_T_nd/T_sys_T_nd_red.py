#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import argparse
#from katsdpscripts.RTS import diodelib
from katsdpscripts.reduction import diodelib

def parse_arguments():
    parser = argparse.ArgumentParser(description=" This produces a pdf file with graphs verifying the ND model and Tsys for each antenna in the file")
    parser.add_argument("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
    parser.add_argument("--pdf", action='store_true',default=True, help="Print the output to a PDF")
    parser.add_argument("--Ku", action='store_true',default=False, help="The specified file is a Ku band observation")
    parser.add_argument("-v","--verbose", action='store_true',default=False, help="Print some debugging information")
    parser.add_argument("--error_bars", action='store_true',default=False, help="Include error bars - Still in development")
    parser.add_argument("--off_target", default='off1', help="which of the two off targets to use")
    parser.add_argument("--write_nd", action='store_true', default=False, help="Write the Noise Diode temp to a file")
    parser.add_argument("filename", nargs=1)
    
    args,unknown = parser.parse_known_args()

    if args.filename[0] == '':
        raise RuntimeError('Please specify an h5 file to load.')
    
    return args,unknown



if __name__ == "__main__":
    args,unknown = parse_arguments()
    print(unknown)
    kwargs = dict(list(zip(unknown[0::2],unknown[1::2])))
    diodelib.read_and_plot_data(args.filename[0],args.output_dir,args.pdf,args.Ku,args.verbose,args.error_bars,args.off_target,args.write_nd,**kwargs)




