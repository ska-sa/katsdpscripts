#!/usr/bin/env python
import h5py
import katfile
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
import os
import textwrap
import time

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from optparse import OptionParser
from pylab import axes, figure, legend, mean, plot, plt, savefig, sys, text, title, xlabel, xticks, ylabel, ylim, yticks

def get_options():
    parser = OptionParser(description='Reduction script to produce metrics on an observation katfile.')
    parser.add_option('-f', '--filename', help='Name of the hdf5 katfile')
    parser.add_option('-d', '--tempdir', default='.', help='Name of the temporary directory to use for creating output files [default: current directory]')
    parser.add_option('-k', '--keep', action='store_true', default=False, help='Keep temporary files')
    parser.add_option('--noarchive', action='store_true', default=False, help='Access the file directly, do not use the archive')
    opts, args = parser.parse_args()

    if opts.filename is None:
        print parser.format_help()
        print 
        sys.exit(2)

    return opts

def make_frontpage(instruction_set, file_ptr):
    scp='%s %s' % ('Instruction_set :', ' '.join((instruction_set[1][1], instruction_set[2][1])))
    scp='\n'.join(textwrap.wrap(scp, 126)) #add new line after every 126 charecters
    scp='\n'+scp+'\n' #add space before and after instruction set

    mystring=file_ptr.__str__()
    mystring_seperated=mystring.split("\n")

    startdate = time.strftime('%d %b %y', time.localtime(file_ptr.start_time))

    lststart=("%2.0f:%2.0f"%(np.modf(file_ptr.lst[0])[1], np.modf(file_ptr.lst[0])[0]*60))
    lststop=("%2.0f:%2.0f"%(np.modf(file_ptr.lst[len(file_ptr.lst)-1])[1], np.modf(file_ptr.lst[len(file_ptr.lst)-1])[0]*60))

    frontpage="Description: "+file_ptr.description+"\n"+"Name: "+file_ptr.name+"\n"+"Experiment ID:  "+file_ptr.experiment_id+"\n"+scp+"\n"+"Observer: "+file_ptr.observer+" \n"+mystring_seperated[5]+" \n"+"Observed on: "+startdate + " from "+lststart+" LST to "+lststop+" LST"+"\n\n"
    frontpage=frontpage+"Dump rate / period: "+str((round(1/file_ptr.dump_period,6))) +" Hz"+" / "+str(round(file_ptr.dump_period,4))+" s"+ "\n"+mystring_seperated[7]+"\n"+mystring_seperated[8]+"\n"+mystring_seperated[9]+"\n"+"Number of Dumps: "+str(file_ptr.shape[0])+"\n\n"
    frontpage=frontpage+mystring_seperated[11]+"\n"+mystring_seperated[12]+"\n"+mystring_seperated[21]+"\n"
    return frontpage

def plot_time_series(ants,pol,count,startime):
    #Time Series
    figure(figsize=(13,10), facecolor='w', edgecolor='k')
    xlabel("LST on "+starttime,fontweight="bold")
    ylabel("Amplitude",fontweight="bold")
    for ant in ants:
        print ("plotting "+ant.name+"_" +pol+pol+ " time series")
        f.select(ants=ant,corrprods='auto',pol=pol)
        if len(f.channels)<1025:
            f.select(channels=range(200,800))
        if count==0:
            plot(f.lst,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant.name+'_'+pol+pol))
            locs,labels=xticks()
            for i in range(len(locs)):
                labels[i]=("%2.0f:%2.0f"%(np.modf(locs[i])[1], np.modf(locs[i])[0]*60))
            xticks(locs,labels)
        elif count==1:
            plot(f.lst,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant.name+'_'+pol+pol))
            locs,labels=xticks()
            for i in range(len(locs)):
                labels[i]=("%2.0f:%2.0f"%(np.modf(locs[i])[1], np.modf(locs[i])[0]*60))
            xticks(locs,labels)

    legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
    savefig(pp,format='pdf')

def plot_spectrum(pol, datafile, starttime, ant):
    #Spectrum function
    fig=figure(figsize=(13,10), facecolor='w', edgecolor='k')
    ab = []
    for  count in (0,1):
        ab.append(fig.add_subplot(2,1,(count+1)))
        ab[-1].set_ylim(2,16)
        if len(f.channels)<1025:
            ab[-1].set_xlim(195,805)
        ab[-1].set_xlabel("Channels", fontweight="bold")
        ab[-1].set_ylabel("Amplitude", fontweight="bold")

        f.select(ants=ant,corrprods='auto',pol=pol[count])
        abs_vis=np.abs(f.vis[:])
        label_format = '%s_%s%s' % (ant.name, pol[count], pol[count])
        print "Starting to plot the %s spectrum." % (label_format,)
        for stat in ('mean', 'min', 'max'):
            ab[-1].plot(f.channels, 10*np.log10(getattr(abs_vis,stat)(axis=0)), label=('%s_%s' % (label_format, stat)))
        ab[-1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
        minorLocator   = AutoMinorLocator()
        ab[-1].xaxis.set_minor_locator(minorLocator)
        pl.tick_params(which='both', color='k')
        pl.tick_params(which='major', length=6,width=2)
        pl.tick_params(which='minor', width=1,length=4)

        ab.append(ab[-1].twinx())
        flag=f.flags()[:]
        # total_sum=0
        perc=[]
        for i in range(len(f.channels)):
            f_chan=flag[:,i,0].squeeze()
            suming=f_chan.sum()
            perc.append(100*(suming/float(f_chan.size)))
        ab[-1].bar(f.channels,perc)
        minorLocator   = AutoMinorLocator()
        ab[-1].xaxis.set_minor_locator(minorLocator)
        ab[-1].set_ylabel("% flagged", fontweight="bold")
        ab[-1].set_ylim(0,100)
        if len(f.channels)<1025:
            ab[-1].set_xlim(195,805)

    savefig(pp,format='pdf')

def plot_bcal_selection(f):
    fig = plt.figure(figsize=(21,15))
    try:
        for pol in ('h','v'):
            f.select(targets=f.catalogue.filter(tags='bcal'), corrprods='cross', pol=pol, scans='track')
            crosscorr = [(f.inputs.index(inpA), f.inputs.index(inpB)) for inpA, inpB in f.corr_products]
            #extract the fringes
            fringes = np.angle(f.vis[:,:,:])
            #For plotting the fringes
            fig.subplots_adjust(wspace=0., hspace=0.)
            #debug_here()
            for n, (indexA, indexB) in enumerate(crosscorr):
                subplot_index = (len(f.ants) * indexA + indexB + 1) if pol == 'h' else (indexA + len(f.ants) * indexB + 1)
                ax = fig.add_subplot(len(f.ants), len(f.ants), subplot_index)
                ax.imshow(fringes[:,:,n],aspect=fringes.shape[1]/fringes.shape[0])
                ax.set_xticks([])
                ax.set_yticks([])
                if pol == 'h':
                    if indexA == 0:
                        ax.xaxis.set_label_position('top')
                        ax.set_xlabel(f.inputs[indexB][3:],size='xx-large')
                    if indexB == len(f.ants) - 1:
                       ax.yaxis.set_label_position('right')
                       ax.set_ylabel(f.inputs[indexA][3:], rotation='horizontal',size = 'xx-large')
                else:
                    if indexA == 0:
                        ax.set_ylabel(f.inputs[indexB][3:], rotation='horizontal',size='xx-large')
                    if indexB == len(f.ants) - 1:
                        ax.set_xlabel(f.inputs[indexA][3:],size='xx-large')

    except KeyError, error:
            print 'Failed to read scans from File: %s with Key Error: %s' % (f, error)
    except ValueError, error:
            print 'Failed to read scans from File: %s with Value Error: %s' % (f, error)
    plt.savefig(pp,format='pdf')

def plot_target_selection(f):
    fig = plt.figure(figsize=(21,15))
    try:
        for pol in ('h','v'):
            f.select(targets=f.catalogue.filter(tags='target'), corrprods='cross', pol=pol, scans='track')
            crosscorr = [(f.inputs.index(inpA), f.inputs.index(inpB)) for inpA, inpB in f.corr_products]
            #extract the fringes
            power = 10 * np.log10(np.abs((f.vis[:,:,:])))
            #For plotting the fringes
            fig.subplots_adjust(wspace=0., hspace=0.)
            #debug_here()
            for n, (indexA, indexB) in enumerate(crosscorr):
                subplot_index = (num_ants * indexA + indexB + 1) if pol == 'h' else (indexA + num_ants * indexB + 1)
                ax = fig.add_subplot(num_ants, num_ants, subplot_index)
                ax.plot(f.channel_freqs,np.mean(power[:,:,n],0))
                ax.set_xticks([])
                ax.set_yticks([])
                if pol == 'h':
                    if indexA == 0:
                        ax.xaxis.set_label_position('top')
                        ax.set_xlabel(f.inputs[indexB][3:],size='xx-large')
                    if indexB == len(f.ants) - 1:
                        ax.yaxis.set_label_position('right')
                        ax.set_ylabel(f.inputs[indexA][3:], rotation='horizontal',size = 'xx-large')
                else:
                    if indexA == 0:
                        ax.set_ylabel(f.inputs[indexB][3:], rotation='horizontal',size='xx-large')
                    if indexB == len(f.ants) - 1:
                        ax.set_xlabel(f.inputs[indexA][3:],size='xx-large')
            #plt.savefig(pp,format='pdf')
        except KeyError , error:
            print 'Failed to read scans from File: ',fn,' with Key Error:',error
        except ValueError , error:
            print 'Failed to read scans from File: ',fn,' with Value Error:',error
    plt.savefig(pp,format='pdf')

################################################################################

opts = get_options()
#get data file using katarchive and open it using katfile
datafile = os.path.basename(opts.filename)

#create ouput filenames
text_log_filename = '%s.txt' % (os.path.splitext(os.path.basename(datafile))[0],)
text_log_filename = os.path.join(opts.tempdir, text_log_filename)
pdf_filename = '%s.pdf' % (os.path.splitext(os.path.basename(datafile))[0],)
pdf_filename = os.path.join(opts.tempdir, pdf_filename)

#open ouput files
text_log = open(text_log_filename, 'w')
pp = PdfPages(pdf_filename)

print 'Searching the data file from the mounted archive'
if opts.noarchive:
    d=[opts.filename]
else:
    import katarchive 
    d = katarchive.get_archived_products(datafile)
    while len(d) == 0:
        time.sleep(10)
        d=katarchive.get_archived_products(datafile)

#get instruction set using h5py
print "Opening %s using h5py" % (datafile,)
a=h5py.File(d[0],'r')
ints=a['MetaData']['Configuration']['Observation'].attrs.items()
a.close()

print "Opening %s using katfile, this might take a while" % (datafile,)
f=katfile.open(d, quicklook=True)

#start a figure
figure(figsize = (13,6))
axes(frame_on=False)
xticks([])
yticks([])
title(datafile+" Observation Report",fontsize=14, fontweight="bold")

frontpage = make_frontpage(ints, f)
text_log.write(frontpage)
text(0,0,frontpage,fontsize=12)
savefig(pp,format='pdf')
print f

count=0
pol=['h','v']
starttime = time.strftime('%d %b %y', time.localtime(f.start_time))
plot_time_series(f.ants, pol[0], count, starttime)
count=count+1
plot_time_series(f.ants, pol[1], count, starttime)

for ant in f.ants:
    plot_spectrum(pol,datafile,starttime,ant)

print "Getting wind and temperature sensors"
fig=pl.figure(figsize=(13,10))
ax1 = fig.add_subplot(211)
ax1.plot(f.lst,f.sensor['Enviro/asc.air.temperature'],'g-')
locs,labels=xticks()
for i in range(len(locs)):
    labels[i]=("%2.0f:%2.0f"%(np.modf(locs[i])[1], np.modf(locs[i])[0]*60))
xticks(locs,labels)
ax1.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ax1.set_xlabel("LST on "+starttime, fontweight="bold")
ax1.set_ylabel('Temperature (Deg C)', color='g',fontweight="bold")
for tl in ax1.get_yticklabels():
    tl.set_color('g')

#Relative to Absolute
rh=f.sensor['Enviro/asc.air.relative-humidity']
t=f.sensor['Enviro/asc.air.temperature']
Pws=[]
Pw=[]
ah=[]
for m in range(len(rh)):
    Pws.append(6.1162*(10**((7.5892*t[m])/(t[m]+240.71))))
for m in range(len(rh)):
    Pw.append(Pws[m]*(rh[m]/100))
for m in range(len(rh)):
    ah.append(2.11679*((Pw[m]*100)/(273.16+t[m])))


ax2=ax1.twinx()
ax2.plot(f.lst,ah,'c-')
locs,labels=xticks()
ax2.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ax2.set_ylabel('Absolute Humidity g/m^3', fontweight="bold",color='c')
for tl in ax2.get_yticklabels():
    tl.set_color('c')

ax3=fig.add_subplot(212)
ax3.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ax3.plot(f.lst,f.sensor['Enviro/asc.wind.speed'],'b-')
for i in range(len(locs)):
    labels[i]=("%2.0f:%2.0f"%(np.modf(locs[i])[1], np.modf(locs[i])[0]*60))
xticks(locs,labels)
ylim(ymin=-0.5)
ax3.set_xlabel("LST on "+starttime,fontweight="bold")
ax3.set_ylabel('Wind Speed (m/s)',fontweight="bold", color='b')
for tl in ax3.get_yticklabels():
    tl.set_color('b')

ax4=ax3.twinx()
ax4.plot(f.lst ,f.sensor['Enviro/asc.air.pressure'],'r-')
ax4.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ax4.set_ylabel('Air Pressure (kPa)', fontweight="bold",color='r')
for tl in ax4.get_yticklabels():
    tl.set_color('r')
savefig(pp,format='pdf')

if f.catalogue.filter(tags='bpcal'):
    print "Plotting bpcal fringes."
    plot_bpcal_selection(f)
else:
    print "No bpcal tags found in catalog, we wont plot bpcal fringes."

if f.catalogue.filter(tags='target'):
    print "Plotting %s fringes." % (tag)
    plot_target_selection(f)
else:
    print "No %s tags found in catalog, we wont plot %s fringes." % (tag, tag)

plt.close('all')
pp.close()
text_log.close()

print 'The results are save in %s and the text report in %s' % (pdf_filename, text_log_filename,)

