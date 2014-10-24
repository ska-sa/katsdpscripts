#!/usr/bin/env python
import katfile
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap
import time
import datetime as dt
import matplotlib.dates as mdates

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

def make_frontpage(file_ptr):
    if float(file_ptr.version) >= 2.0:
        instruction_set = file_ptr.file['MetaData']['Configuration']['Observation'].attrs.items()
        instruction_set = ' '.join((instruction_set[1][1], instruction_set[2][1]))
    else:
        instruction_set = 'Unknown'

    scp='Instruction_set : %s' % (instruction_set,)
    scp='\n'.join(textwrap.wrap(scp, 126)) #add new line after every 126 charecters
    scp='\n'+scp+'\n' #add space before and after instruction set
    scp1 = scp.decode('ascii', 'ignore')
    prog='http://192.168.193.8:8081/tailtask/'+file_ptr.experiment_id+'/progress'
    mystring_seperated=str(file_ptr).split('\n')

    startdate = time.strftime('%d %b %y', time.localtime(file_ptr.start_time))

    lststart=("%2.0f:%2.0f"%(np.modf(file_ptr.lst[0])[1], np.modf(file_ptr.lst[0])[0]*60))
    lststop=("%2.0f:%2.0f"%(np.modf(file_ptr.lst[len(file_ptr.lst)-1])[1], np.modf(file_ptr.lst[len(file_ptr.lst)-1])[0]*60))

    frontpage = []
    frontpage.append('Description: %s' % (file_ptr.description,))
    frontpage.append('Name: %s' % (file_ptr.name,))
    frontpage.append('Experiment ID: %s' % (file_ptr.experiment_id,))
    frontpage.append('Progess Output: %s' %(prog))
    frontpage.append(scp1)
    frontpage.append('Observer: %s' % (file_ptr.observer,))
    frontpage.append(mystring_seperated[5])
    frontpage.append('Observed on: %s from %s LST to %s LST' % (startdate, lststart, lststop))
    frontpage.append('\n')
    frontpage.append('Dump rate / period: %s Hz / %s s' % (str((round(1/file_ptr.dump_period,6))), str(round(file_ptr.dump_period,4))))
    frontpage.append(mystring_seperated[7])
    frontpage.append(mystring_seperated[8])
    frontpage.append(mystring_seperated[9])
    frontpage.append('Number of Dumps: %s' % (str(file_ptr.shape[0])))
    frontpage.append('\n')
    frontpage.append(mystring_seperated[11])
    frontpage.append(mystring_seperated[12])
    frontpage.append(mystring_seperated[21])
    frontpage.append('\n')
    return '\n'.join(frontpage)

def make_last_page():
    FName="/home/kat/svn/auto_imager/new_obs_report.py"
    rev=os.popen('svn info %s | grep "Last Changed Rev" ' % FName, "r").readline().replace("Last Changed Rev:","***\nThis report was generated using "+FName+", svn revesion: ")
    lastpage=[]

    lastpage.append("Description of the plots In the report\n==================================\n")
    lastpage.append("Time Series Plot:\n \t This plot shows the mean of the autocorrelation amplitude against time, for the duration of the observation.\
    The first time series plot\n shows HH while the second one shows VV. On primary x-axis is LTS and secondary x-axis shows SAST corresponding to \
    the LST at the \n time of observation.\n")
    lastpage.append("Antenna X Spectrum:\n\t This plot shows the power (left y-axis), against channels, with the frequency corresponding to the channels \
    plotted on the secondary\n x-axis. The mean of the autocorrelation spectrum is plotted in green, minimum in blue, and the maximum in magenta. For wide\n \
    band (1k channels) observation only channel 170 to 854 is plotted. On the same plot overlaid is the histogram of the percentage of\n data flagged per channel \
    for the duration of the observation the.\n")
    lastpage.append("Weather Data:\n\t This is the representation of the weather conditions during the period of the observation.  The plots include wind speed, \
    temperature,\n absolute humidity and air pressure. These plots are against LST as well as SAST on the secondary x-axis.\n")
    lastpage.append("Band pass calibator fringes\n\t This plot shows the spectrograms between all the baselines present during the observation. This give us an \
    idea of how \n the phase between the baselines were behaving. This plot is time(y-axis) vs channels (x-axis).\n")
    lastpage.append("Correlation Spectra\n\t This plot shows the correlation spectrum for each baseline. Common features between the crossed antenna will be amplified.\n\n\n")
    lastpage.append(rev)
    
    return '\n'.join(lastpage)


def lst_date(f):
    lststring=[]
    #changing lst to string
    for t in range(len(f.lst)):
        lststring.append(("%s:%s"%(("00" if int(np.modf(f.lst[t])[1])==0 else int(np.modf(f.lst[t])[1])), int(np.modf(f.lst[t])[0]*60))))

    elem="None"
    for a in range(len(lststring)):
        if lststring[a]=='23:59':
            elem=a


    if elem!="None":
        for a in range(0,(elem+1)):
            lststring[a]="18/4/2013 "+lststring[a]
        for b in range(elem,(len(f.lst)-1)):
            lststring[b+1]="19/4/2013 "+lststring[b+1]
    else:
        for c in range(len(f.lst)):
            lststring[c]="18/4/2013 "+lststring[c]

    # converting string to time
    lstime=[dt.datetime.strptime(d,"%d/%m/%Y %H:%M") for d in lststring]

    loc_time=[]
    for t in range(len(f.timestamps)):
        loc_time.append(time.localtime(f.timestamps[t]))
    time_string=[time.strftime("%Y/%m/%d %H:%M:%S",d) for d in loc_time]
    date_time=[dt.datetime.strptime(d,"%Y/%m/%d %H:%M:%S") for d in time_string]

    return lstime, date_time


def plot_time_series(ants,pol,startime,lst_time,loc_datetime):
    #Time Series
    fig=plt.figure(figsize=(13,10), facecolor='w', edgecolor='k')
    plt.suptitle("Time series plot",fontsize=16, fontweight="bold")
    sub1 = fig.add_subplot(111)
    sub1.set_xlabel("SAST on "+starttime,fontweight="bold")
    sub1.set_ylabel("Amplitude",fontweight="bold")
    for ant in ants:
        print ("plotting "+ant.name+"_" +pol+pol+ " time series")
        f.select(ants=ant,corrprods='auto',pol=pol)
        if len(f.channels)<1025:
            f.select(channels=range(200,800))
        sub1.plot(loc_datetime,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant.name+'_'+pol+pol))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    legend(loc='right', bbox_to_anchor=(1.13, 0.92), ncol=2, fancybox=True, shadow=False)
    #create a dummy array to plot LST on top x axis
    sub2=sub1.twiny()
    dummy=[min(sub1.get_yticks()) for i in range(len(f.lst))]
    sub2.plot(lst_time,dummy,'k')
    sub2.set_xlabel("LST on "+starttime,fontweight="bold")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for tl in sub2.get_xticklabels():
	    tl.set_color('DarkViolet')

    #legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
    savefig(pp,format='pdf')



def plot_spectrum(pol, ant):
    #Spectrum function
    fig=figure(figsize=(13,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.25)
    axes(frame_on=False)
    xticks([])
    yticks([])
    plt.suptitle("Antenna "+ant.name+" Spectrum ",fontsize=16, fontweight="bold")
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
        if (10*np.log10((abs_vis.max(axis=0)).max())) >16:
            ab[-1].set_ylim(2,ymax=0.5+(10*np.log10((abs_vis.max(axis=0)).max())))
        label_format = '%s_%s%s' % (ant.name, pol[count], pol[count])
        print "Starting to plot the %s spectrum." % (label_format,)
        plotcolours=['g','b','m']
        colours=0
        for stat in ('mean', 'min', 'max'):
            ab[-1].plot(f.channels, 10*np.log10(getattr(abs_vis,stat)(axis=0)), label=('%s_%s' % (label_format, stat)),color=plotcolours[colours] )
            colours+=1
        ab[-1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4, fancybox=True, shadow=False)
        minorLocator   = AutoMinorLocator()
        ab[-1].xaxis.set_minor_locator(minorLocator)
        plt.tick_params(which='both', color='k')
        plt.tick_params(which='major', length=6,width=2)
        plt.tick_params(which='minor', width=1,length=4)

        #plot Frequncy on the top axis
        ylocs,ylabels=yticks()
        xaxis2=ab[-1].twiny()
        dummy=[]
        for ts in range(len(f.channels)):
            dummy.append(min(ylocs))
        xaxis2.plot(f.channel_freqs/1e6, dummy,'k-')
        xaxis2.ticklabel_format(axis='x', style='plain', useOffset=False)
        xaxis2.set_ylim(min(ylocs),max(ylocs))
        xaxis2.invert_xaxis()
        xaxis2.set_xlim(xmin=(f.channel_freqs[0]-(f.channel_width*170))/1e6, xmax=(f.channel_freqs[-1]+(f.channel_width*170))/1e6)
        xaxis2.set_xlabel("Frequency MHz",fontweight="bold")

        #still to take a closer look at the calculation of the flags
        ab.append(ab[-1].twinx())
        flag=f.flags()[:]
        # total_sum=0
        perc=[]
        for i in range(len(f.channels)):
            f_chan=flag[:,i,0].squeeze()
            suming=f_chan.sum()
            perc.append(100*(suming/float(f_chan.size)))
        ab[-1].bar(f.channels,perc,color='r',edgecolor='none')
        minorLocator   = AutoMinorLocator()
        ab[-1].xaxis.set_minor_locator(minorLocator)
        ab[-1].set_ylabel("% flagged", fontweight="bold")
        ab[-1].set_ylim(0,100)
        if len(f.channels)<1025:
            ab[-1].set_xlim(195,805)
    savefig(pp,format='pdf')

def plot_envioronmental_sensors(f,starttime,lst_time,loc_datetime):
    print "Getting wind and temperature sensors"
    fig=plt.figure(figsize=(13,10))
    plt.suptitle("Weather Data",fontsize=16, fontweight="bold")
    ax1 = fig.add_subplot(211)
    
    #Plot Air Temperature
    ax1.plot(loc_datetime,f.sensor['Enviro/asc.air.temperature'],'g-')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    airtemp=f.sensor['Enviro/asc.air.temperature']
    ylim(ymin=-1,ymax=35)
    mintemp=min(airtemp)
    maxtemp=max(airtemp)
    if maxtemp>=35:
	    ylim(ymax=(maxtemp+1))
    if mintemp<=(-1.0):
	    ylim(ymin=(mintemp-1))
    
    ax1.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
    ax1.set_xlabel("LST on "+starttime, fontweight="bold")
    ax1.set_ylabel('Temperature (Deg C)', color='g',fontweight="bold")
    for tl in ax1.get_yticklabels():
        tl.set_color('g')
    #Create the twin Y to plot LST on the top axis
    ay1=ax1.twiny()
    ay1.set_xlabel("LST "+starttime,fontweight="bold")
    dummy=[min(ax1.get_yticks()) for i in range(len(f.lst))]
    ay1.plot(lst_time,dummy,'k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for tl in ay1.get_xticklabels():
	    tl.set_color('DarkViolet')

    #Convert Relative to Absolute
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
    ax2.plot(loc_datetime,ah,'c-')
    ylim(ymin=1,ymax=8)
    minah=min(ah)
    maxah=max(ah)
    if maxah>=8:
	    ylim(ymax=(maxah+1))
    if minah<=(1.0):
	    ylim(ymin=(minah-1))
    locs,labels=xticks()
    ax2.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
    ax2.set_ylabel('Absolute Humidity g/m^3', fontweight="bold",color='c')
    for tl in ax2.get_yticklabels():
        tl.set_color('c')

    ax3=fig.add_subplot(212)
    ax3.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
    ax3.plot(loc_datetime,((f.sensor['Enviro/asc.air.pressure'])/10),'r-')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    airpress=f.sensor['Enviro/asc.air.pressure']/10
    ylim(ymin=87,ymax=92)
    minairpress=min(airpress)
    maxairpress=max(airpress)
    if maxairpress>=92:
	    ylim(ymax=(maxairpress+1))
    if minairpress<=87:
	    ylim(ymin=(minairpress-1))
    
    ax3.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
    ax3.set_ylabel('Air Pressure (kPa)', fontweight="bold",color='r')
    for tl in ax3.get_yticklabels():
	    tl.set_color('r')
	
    ax4=ax3.twinx()
    ax4.plot(loc_datetime,f.sensor['Enviro/asc.wind.speed'],'b-')
    wspeed=f.sensor['Enviro/asc.wind.speed']
    ylim(ymin=-0.5,ymax=16)
    minwind=min(wspeed)
    maxwind=max(wspeed)
    if maxwind>=16:
	    ylim(ymax=(maxwind+1))
    if minwind<=-0.5:
	    ylim(ymin=(minwind-1))
    ax4.set_xlabel("SAST on "+starttime,fontweight="bold")
    ax4.set_ylabel('Wind Speed (m/s)',fontweight="bold", color='b')
    for tl in ax4.get_yticklabels():
	    tl.set_color('b')
    savefig(pp,format='pdf')

def plot_bpcal_selection(f):
    bp = np.array([t.tags.count('bpcal') for t in f.catalogue.targets]) == 1
    bp = np.arange(len(bp))[bp][0]
    fig = plt.figure(figsize=(21,15))
    plt.suptitle("Bp cal Fringes",fontsize=16, fontweight="bold")
    try:
        for pol in ('h','v'):
            f.select(targets=bp, corrprods='cross', pol=pol, scans='track')
            crosscorr = [(f.inputs.index(inpA), f.inputs.index(inpB)) for inpA, inpB in f.corr_products]
            #extract the fringes
            fringes = np.angle(f.vis[:,:,:])
            #For plotting the fringes
            fig.subplots_adjust(wspace=0., hspace=0.)
            #debug_here()
            for n, (indexA, indexB) in enumerate(crosscorr):
                subplot_index = (len(f.ants) * indexA + indexB + 1) if pol == 'h' else (indexA + len(f.ants) * indexB + 1)
                ax = fig.add_subplot(len(f.ants), len(f.ants), subplot_index)
                ax.imshow(fringes[:,:,n],aspect=float(fringes.shape[1])/fringes.shape[0])
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
    plt.suptitle("Correlation Spectra",fontsize=16, fontweight="bold")
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
                subplot_index = (len(f.ants) * indexA + indexB + 1) if pol == 'h' else (indexA + len(f.ants) * indexB + 1)
                ax = fig.add_subplot(len(f.ants), len(f.ants), subplot_index)
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
        print 'Failed to read scans from File: ',f, ' with Key Error:',error
    except ValueError , error:
        print 'Failed to read scans from File: ',f,' with Value Error:',error
    plt.savefig(pp,format='pdf')

###########################################################################################################################################

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

print "Opening %s using katfile, this might take a while" % (datafile,)
f=katfile.open(d[0], quicklook=True)
#start a figure
figure(figsize = (13.5,6.5))
axes(frame_on=False)
xticks([])
yticks([])
title(datafile+" Observation Report",fontsize=16, fontweight="bold")

frontpage = make_frontpage(f)
text_log.write(frontpage)
text(0,0,frontpage,fontsize=12)
savefig(pp,format='pdf')
print f

lst_time,loc_datetime=lst_date(f)

ants=f.ants
pol=['h','v']
starttime = time.strftime('%d %b %y', time.localtime(f.start_time))
plot_time_series(ants, pol[0], starttime,lst_time,loc_datetime)
plot_time_series(ants, pol[1], starttime,lst_time,loc_datetime)

for ant in ants:
    plot_spectrum(pol,ant)

plot_envioronmental_sensors(f,starttime,lst_time,loc_datetime)
f.select()

if f.catalogue.filter(tags='bpcal'):
    print "Plotting bpcal fringes."
    plot_bpcal_selection(f)
else:
    print "No bpcal tags found in catalog, we wont plot bpcal fringes."

if f.catalogue.filter(tags='target'):
    print "Plotting target correlation spectra."
    plot_target_selection(f)
else:
    print "No target tags found in catalog, we wont plot the target cross correlation spectra."


#creating the last page
figure(figsize = (13.5,7))
axes(frame_on=False)
xticks([])
yticks([])
lastpage=make_last_page()

text(0,0,lastpage,fontsize=12)
savefig(pp,format='pdf')
plt.close('all')
pp.close()
text_log.close()
print 'The results are save in %s and the text report in %s' % (pdf_filename, text_log_filename,)

