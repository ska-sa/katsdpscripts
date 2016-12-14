import numpy as np
import os
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pylab as plt
from matplotlib import ticker 
import optparse
import katdal
from katdal.dataset import BrokenFile
from katsdpscripts.RTS import rfilib
import pickle
import csv



def plot_RFI_mask(pltobj,rfi,channelwidth):
	if rfi:
		for this_rfi in rfi:
			start_rfi=float(this_rfi[0])*1e6-channelwidth/2
			if this_rfi[1]=='':
				end_rfi=float(this_rfi[0])*1e6+channelwidth/2
			else:
				end_rfi=float(this_rfi[1])*1e6+channelwidth/2
			pltobj.axvspan(start_rfi,end_rfi, alpha=0.3, color='grey')

def plot_flag_data(label,spectrum,flagfrac,freqs,pdf,mask=None):
    """
    Produce a plot of the average spectrum in H and V 
    after flagging and attach it to the pdf output.
    Also show fraction of times flagged per channel.
    """
    from katsdpscripts import git_info

    repo_info = git_info() 

    #Set up the figure
    fig = plt.figure(figsize=(11.7,8.3))

    #Plot the spectrum
    ax1 = fig.add_subplot(211)
    ax1.text(0.01, 0.90,repo_info, horizontalalignment='left',fontsize=10,transform=ax1.transAxes)
    ax1.set_title(label)
    plt.plot(freqs,spectrum,linewidth=.5)

    #plot_RFI_mask(ax1)
    ticklabels=ax1.get_xticklabels()
    plt.setp(ticklabels,visible=False)
    ticklabels=ax1.get_yticklabels()
    plt.setp(ticklabels,visible=False)
    plt.xlim((min(freqs),max(freqs)))
    plt.ylabel('Mean amplitude\n(arbitrary units)')
    #Plot the mask
    plot_RFI_mask(ax1,mask,freqs[1]-freqs[0])
    #Plot the flags occupancies
    ax = fig.add_subplot(212,sharex=ax1)
    plt.plot(freqs,flagfrac,'r-',linewidth=.5)
    plt.ylim((0.,1.))
    plt.axhline(0.8,color='red',linestyle='dashed',linewidth=.5)
    plot_RFI_mask(ax,mask,freqs[1]-freqs[0])
    plt.xlim((min(freqs),max(freqs)))
    minorLocator = ticker.MultipleLocator(10e6)
    plt.ylabel('Fraction flagged')
    ticklabels=ax.get_yticklabels()
    #Convert ticks to MHZ
    ticks = ticker.FuncFormatter(lambda x, pos: '{:4.0f}'.format(x/1.e6))
    ax.xaxis.set_major_formatter(ticks)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.xlabel('Frequency (MHz)')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)

#command-line parameters
parser = optparse.OptionParser(usage="Please specify the input file\n \
    USAGE: python report_contaminated_channels.py <inputfile.h5> ", \
    description="Report the frequencies of channels that have consistently been flagged.")
parser.add_option("--antenna", "-a", default=None, help="Antenna to process. Default is first ant in file")
parser.add_option("--threshold", "-t", default=0.8, help="Threshold above which to report contamination percentage. Default=0.8")
parser.add_option("--ignore-mask", "-i", default='', help="Location of rfi mask pickle for channels to ignore in the report")
parser.add_option("--known-rfi", "-k", default='/var/kat/katsdpscripts/RTS/2.9-RFI/known_satellites.csv', help="csv file containing frequencies of known RFI emitters")
opts, args = parser.parse_args()

input_file=args[0]

try:
    #open the observation
    katdalfile=katdal.open(input_file)
    ant=opts.antenna if opts.antenna else katdalfile.ants[0].name
    katdalfile.select(ants=ant,scans='~slew')
    #Get the flag stats
    report_dict=rfilib.get_flag_stats(katdalfile)['all_data']
except BrokenFile:
    #Open the rfi_report
    report_data=h5py.File(input_file)['all_data']
    report_dict=dict.fromkeys(report_data.keys())
    for key in report_dict: report_dict[key]=report_data[key].value
    #RFI reports only contain a single antenna
    ant=opts.antenna if opts.antenna else report_dict['corr_products'][0,0][:-1]
    if report_dict['corr_products'][0,0][:-1]!=ant: raise ValueError('Selected antenna (%s) not in input file.'%(ant,))

num_channels=len(report_dict['channel_freqs'])
start_chan = num_channels//20
end_chan   = num_channels - start_chan
chan_range = range(start_chan,end_chan+1)

#Open the csv file with known rfi
if opts.known_rfi is not None:
    known_rfi=[]
    known_rfi_start_freqs=[]
    with open(opts.known_rfi) as csvfile:
        data=csv.reader(csvfile,delimiter=',')
        for row in data:
            if len(row)==3:
                known_rfi.append(row)
                known_rfi_start_freqs.append(float(row[0]))
else: known_rfi=[]

#Set up a looup table for sorted known_rfi mask
known_lookup=sorted([(freq,i) for i,freq in enumerate(known_rfi_start_freqs)])

#Open a pdf
pdf = PdfPages(os.path.splitext(os.path.basename(input_file))[0]+'_'+ant+'_chanflags.pdf')
fig = plt.figure(None,figsize = (10,16))
page_length = 90.0

#Set up the ignore mask
if opts.ignore_mask:
    ignorefile=open(opts.ignore_mask)
    ignore_mask=pickle.load(ignorefile)
else:
    ignore_mask=np.zeros(report_dict['channel_freqs'].shape,dtype=np.bool)

#Show a plot summary
for i,corrprod in enumerate(report_dict['corr_products'][:2]):
	label="Flag information on baseline %s, %d records"%((','.join(corrprod), report_dict['numrecords_tot']))
	plot_flag_data(label,report_dict['spectrum'][chan_range,i],report_dict['flagfrac'][chan_range,i],report_dict['channel_freqs'][chan_range],pdf,mask=known_rfi)

#Write the occupancies
for i,pol in  enumerate(["HH","VV"]):
    text=[]
    known_iterator=iter(known_lookup)
    this_known=known_iterator.next()
    end_known=report_dict['channel_freqs'][0]
    text.append(("Flagged channels and frequencies %s, %s polarisation:"%(ant, pol),'black','bold'))
    inside_known=False
    for j,freq in enumerate(report_dict['channel_freqs']):
        if j not in chan_range: continue
        if end_known<freq:
            inside_known=False
        if this_known[0]<freq/1e6:
            inside_known=True
            known_data=known_rfi[this_known[1]]
            this_end_known=known_data[1]
            known_string='Known RFI: '
            known_string+=known_data[2]+', '
            if this_end_known is not '':
                known_string+='Start: '+known_data[0]+' MHz, End: '+known_data[1]+' MHz'
                this_end_known=float(this_end_known)*1e6
            else:
                known_string+='Frequency: '+known_data[0]+' MHz'
                this_end_known=report_dict['channel_freqs'][j+1]
            text.append((known_string,'red','bold'))
            end_known=max(end_known,this_end_known)
            try:
                this_known=known_iterator.next()
            except StopIteration:
                this_known=(max(report_dict['channel_freqs']),-1,)
        occupancy = report_dict['flagfrac'][j,i]
        if occupancy > opts.threshold and not ignore_mask[j]:
            if inside_known:
                text.append(('Channel: %5d,    %f MHz , Percentage of integrations contaminated is %.3f' %(j+1,freq/1e6,occupancy*100),'red','normal'))
            else:
                text.append(('Channel: %5d,    %f MHz , Percentage of integrations contaminated is %.3f  ' %(j+1,freq/1e6,occupancy*100),'black','normal'))
    line=0
    for page in xrange(int(np.ceil(len(text)/page_length))):
        fig = plt.figure(None,figsize = (10,16))
        lineend = line+int(np.min((page_length,len(text[line:]))))
        factadj = 0.91*(1-(lineend-line)/page_length)
        for num,pos in enumerate(np.linspace(0.95,0.05+factadj,lineend-line)):
            plt.figtext(0.1 ,pos,text[line:lineend][num][0],fontsize=10,color=text[line:lineend][num][1],fontweight=text[line:lineend][num][2])
        line = lineend
        fig.savefig(pdf,format='pdf')
        plt.close(fig)

pdf.close()
