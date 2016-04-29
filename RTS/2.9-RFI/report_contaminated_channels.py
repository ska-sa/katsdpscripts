import numpy as np
import os
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pylab as plt
import optparse
import katdal
from katdal.dataset import BrokenFile
from katsdpscripts.RTS import rfilib
import pickle

#command-line parameters
parser = optparse.OptionParser(usage="Please specify the input file\n\
    USAGE: python report_contaminated_channels.py <inputfile.h5> ",
    description="Report the frequencies of channels that have consistently been flagged.")
parser.add_option("--antenna", "-a", default=None, help="Antenna to process. Default is first ant in file")
parser.add_option("--threshold", "-t", default=0.8, help="Threshold above which to report contamination percentage. Default=0.8")
parser.add_option("--ignore_mask", "-i", default=None, help="Location of rfi mask pickle for channels to ignore in the report")
opts, args = parser.parse_args()

input_file=args[0]

try:
    #open the observation
    katdalfile=katdal.open(input_file)
    ant=opts.antenna if opts.antenna else katdalfile.ants[0].name
    katdalfile.select(ants=ant,scans='~slew')
    #Get the flag stats
    report_dict=rfilib.get_flag_stats(katdalfile)
except BrokenFile:
    #Open the rfi_report
    report_data=h5py.File(input_file)['all_data']
    report_dict=dict.fromkeys(report_data.keys())
    for key in report_dict: report_dict[key]=report_data[key].value
    #RFI reports only contain a single antenna
    ant=opts.antenna if opts.antenna else report_dict['corr_products'][0,0][:-1]
    if report_dict['corr_products'][0,0][:-1]!=ant: raise ValueError('Selected antenna (%s) not in input file.'%(ant,))

#Open a pdf
pdf = PdfPages(os.path.splitext(os.path.basename(input_file))[0]+'_chanflags.pdf')
fig = plt.figure(None,figsize = (10,16))
page_length = 90.0

#Set up the ignore mask
if opts.ignore_mask:
    ignorefile=open(opts.ignore_mask)
    ignore_mask=pickle.load(ignorefile)
else:
    ignore_mask=np.zeros(report_dict['channel_freqs'].shape,dtype=np.bool)

#Write the occupancies
for i,pol in  enumerate(["HH","VV"]):
    text=[]
    text.append("\n Flagged channels and frequencies %s, %s polarisation:"%(ant, pol))
    for j,freq in enumerate(report_dict['channel_freqs']):
        occupancy = report_dict['flagfrac'][j,i]
        if occupancy > opts.threshold and not ignore_mask[j]:
            text.append('Channel: %5d,    %f MHz , Percentage of integrations contaminated is %.3f  ' %(j+1,freq/1e6,occupancy*100))
    line=0
    for page in xrange(int(np.ceil(len(text)/page_length))):
        fig = plt.figure(None,figsize = (10,16))
        lineend = line+int(np.min((page_length,len(text[line:]))))
        factadj = 0.87*(1-(lineend-line)/page_length)
        plt.figtext(0.1 ,0.05+factadj,'\n'.join(text[line:lineend]),fontsize=10)
        line = lineend
        fig.savefig(pdf,format='pdf')
        plt.close(fig)

pdf.close()
