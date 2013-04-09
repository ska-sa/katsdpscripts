#!/usr/bin/env python
#plot data from the h5 file
import katfile
import os
import glob
import katoodt
import katarchive
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.ticker import AutoMinorLocator
from optparse import OptionParser
from pylab import *
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import textwrap
import collections

# Set up standard script options
parser = OptionParser(usage="usage: %prog [options]", 
					description="Plot the Time series and spectrum for all the antennas that were involved on during the observation\.n"+
					"The script plot the weather sensors as well for the duration of the observation.\n"+
					"The data file is specified usinp option -d (This is a non-optional option) ")
parser.add_option('-d', '--filename', help="datafile to be plotted "+ "(e.g. 1340018260.h5)")
opts, args = parser.parse_args() 

count=0
#Time Series
def times(ant,pol,count):
	figure(figsize=(13,10), facecolor='w', edgecolor='k')
	xlabel("LST on "+starttime,fontweight="bold")
	ylabel("Amplitude",fontweight="bold")
	for ant_x in ant:
		print ("plotting "+ant_x.name+"_" +pol+pol+ " time series")
       		f.select(ants=ant_x,corrprods='auto',pol=pol)
    		if len(f.channels)<1025:  		
			f.select(channels=range(200,800))
		if count==0:
			plot(f.lst,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant_x.name+'_'+pol+pol))
			locs,labels=xticks()
			for i in range(len(locs)):
				labels[i]=("%2.0f:%2.0f"%(np.modf(locs[i])[1], np.modf(locs[i])[0]*60))
			xticks(locs,labels)
		elif count==1:
			plot(f.lst,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant_x.name+'_'+pol+pol))
			locs,labels=xticks()
			for i in range(len(locs)):
				labels[i]=("%2.0f:%2.0f"%(np.modf(locs[i])[1], np.modf(locs[i])[0]*60))
			xticks(locs,labels)
	
	legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
	savefig(pp,format='pdf')


#Spectrum function
def spec(pol,datafile,starttime):
	count=0	
	fig=figure(figsize=(13,10), facecolor='w', edgecolor='k')
	ab1=fig.add_subplot(2,1,(count+1))
	ab1.set_ylim(2,16)	
	if len(f.channels)<1025:	
		ab1.set_xlim(195,805)
	ab1.set_xlabel("Channels", fontweight="bold")
	ab1.set_ylabel("Amplitude", fontweight="bold")
	

        print ("plotting "+ant_x.name+"_" +pol[count]+pol[count]+ " spectrum")
       	f.select(ants=ant_x,corrprods='auto',pol=pol[count])
        nvis=np.abs(f.vis[:])
	f_min=nvis.min(axis=0)
	f_mean=nvis.mean(axis=0)
	f_max=nvis.max(axis=0)
	ab1.plot(f.channels,10*np.log10(f_min),label=(ant_x.name+'_'+pol[count]+pol[count]+'_min'))
	ab1.plot(f.channels,10*np.log10(f_mean),label=(ant_x.name+'_'+pol[count]+pol[count]+'_mean'))
	ab1.plot(f.channels,10*np.log10(f_max),label=(ant_x.name+'_'+pol[count]+pol[count]+'_max'))
	ab1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
	minorLocator   = AutoMinorLocator()
	ab1.xaxis.set_minor_locator(minorLocator)
	pl.tick_params(which='both', color='k')
	pl.tick_params(which='major', length=6,width=2)
	pl.tick_params(which='minor', width=1,length=4)

	ab2=ab1.twinx()
	flag=f.flags()[:]
	total_sum=0
	perc=[]
	for i in range(len(f.channels)):
		f_chan=flag[:,i,0].squeeze()
		suming=f_chan.sum()
		perc.append(100*(suming/float(len(f_chan))))
	ab2.bar(f.channels,perc)
	minorLocator   = AutoMinorLocator()
	ab2.xaxis.set_minor_locator(minorLocator)
	ab2.set_ylabel("% flagged", fontweight="bold")
	ab2.set_ylim(0,100)	
	if len(f.channels)<1025:
		ab2.set_xlim(195,805)

	count=1
	ab3=fig.add_subplot(2,1,(count+1))
	ab3.set_ylim(2,16)
	if len(f.channels)<1025:
		ab3.set_xlim(195,805)
	ab3.set_xlabel("Channels", fontweight="bold")
	ab3.set_ylabel("Amplitude", fontweight="bold")
	print ("plotting "+ant_x.name+"_" +pol[count]+pol[count]+ " spectrum")
       	f.select(ants=ant_x,corrprods='auto',pol=pol[count])
        nvis=np.abs(f.vis[:])
	f_min=nvis.min(axis=0)
	f_mean=nvis.mean(axis=0)
	f_max=nvis.max(axis=0)
	plot(f.channels,10*np.log10(f_min),label=(ant_x.name+'_'+pol[count]+pol[count]+'_min'))
	plot(f.channels,10*np.log10(f_mean),label=(ant_x.name+'_'+pol[count]+pol[count]+'_mean'))
	plot(f.channels,10*np.log10(f_max),label=(ant_x.name+'_'+pol[count]+pol[count]+'_max'))
	legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
	minorLocator   = AutoMinorLocator()
	ab3.xaxis.set_minor_locator(minorLocator)
	pl.tick_params(which='both', color='k')
	pl.tick_params(which='major', length=6,width=2)
	pl.tick_params(which='minor', width=1,length=4)

	ab4=ab3.twinx()
	flag=f.flags()[:]
	total_sum=0
	perc=[]
	for i in range(len(f.channels)):
		f_chan=flag[:,i,0].squeeze()
		suming=f_chan.sum()
		perc.append(100*(suming/float(len(f_chan))))
	ab4.bar(f.channels,perc)
	minorLocator   = AutoMinorLocator()
	ab4.xaxis.set_minor_locator(minorLocator)
	ab4.set_ylabel("% flagged", fontweight="bold")
	ab4.set_ylim(0,100)	
	if len(f.channels)<1025:
		ab4.set_xlim(195,805)


	savefig(pp,format='pdf')
	#savefig("/data/siphelele/time_series_plots/time_series/"+datafile[:-3]+"_"+ant_x.name+"_Spectro.png")
	count=0

if opts.filename is None:
	print 'Specifiy a file to be plotted using option -d'
	sys.exit()



#get data file using katarchive and open it using katfile

datafile =opts.filename
#pathtofile = 'locate ' +datafile
#searched=glob.glob(datafile)                               #search the file locally 

pp = PdfPages(datafile[:-3]+'.pdf')

#if not searched:							   #If not found locally download it
print 'Searching the data file from the mounted archive'
d=katarchive.get_archived_products(datafile)
	#d=katoodt.get_archived_product(product_name=datafile)
#else: 
#	print "File found locally "
#	d=searched

#get instruction set using h5py
a=h5py.File(d[0],'r')
ints=a['MetaData']['Configuration']['Observation'].attrs.items()
scp=ints[1]+ints[2]
scp=" ".join(scp)                 #convert tuple to string
scp=scp.replace("script_arguments"," ",1) #cut out some text form the string
scp=scp.replace("script_name","Instruction_set : ",1)
scp='\n'.join(textwrap.wrap(scp, 126))         #add new line after every 126 charecters
scp='\n'+scp+'\n'                                               #add space before and after instruction set
a.close()                                                            #release the file for use by katfile


print "Opening the file using katfile, this might take a while"
f=katfile.open(d, quicklook=True)
figure(figsize = (13,6))
axes(frame_on=False)
xticks([])
yticks([])
title(datafile+" Observation Report",fontsize=14, fontweight="bold")
mystring=f.__str__()
mystring_seperated=mystring.split("\n")


epoctime=datafile[:-3]
epoctime=float(epoctime)
loctime=time.localtime(epoctime)
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
starttime=str(loctime.tm_mday) +" "+ months[loctime.tm_mon-1]+" "+ str(loctime.tm_year)

lststart=("%2.0f:%2.0f"%(np.modf(f.lst[0])[1], np.modf(f.lst[0])[0]*60))
lststop=("%2.0f:%2.0f"%(np.modf(f.lst[len(f.lst)-1])[1], np.modf(f.lst[len(f.lst)-1])[0]*60))

frontpage="Description: "+f.description+"\n"+"Name: "+f.name+"\n"+"Experiment ID:  "+f.experiment_id+"\n"+scp+"\n"+"Observer: "+f.observer+" \n"+mystring_seperated[5]+" \n"+"Observed on: "+starttime + " from "+lststart+" LST to "+lststop+" LST"+"\n\n"
frontpage=frontpage+"Dump rate / period: "+str((round(1/f.dump_period,6))) +" Hz"+" / "+str(round(f.dump_period,4))+" s"+ "\n"+mystring_seperated[7]+"\n"+mystring_seperated[8]+"\n"+mystring_seperated[9]+"\n"+"Number of Dumps: "+str(f.shape[0])+"\n\n"
frontpage=frontpage+mystring_seperated[11]+"\n"+mystring_seperated[12]+"\n"+mystring_seperated[21]+"\n"

text(0,0,frontpage,fontsize=12)
savefig(pp,format='pdf')
print f

#Get file start time from file name

ant=f.ants
pol=['h','v']
times(ant,pol[0],count)
count=count+1
times(ant,pol[1],count)

for ant_x in ant:
	count=0
	spec(pol,datafile,starttime)


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


# from kat@kat-imager:lindsay/fringes.py
f.select()
cor = f.corr_products
ants = f.ants
num_ants = len(ants)
bp = np.array([t.tags.count('bpcal') for t in f.catalogue.targets]) == 1
for z in range(len(bp)):
	if bp[z]==1:
		a=1
		break
	else:
		a=0
if a==1:
	print "Plotting fringes"
	bp = np.arange(len(bp))[bp][0]
#code to plot the cross phase ... fringes
	fig = plt.figure(figsize=(21,15))
	try:
   		j=0
#plt.figure()
   		n_channels = len(f.channels)
    		for pol in ('h','v'):
        		f.select(targets=bp,corrprods = 'cross',pol=pol,scans="track")
        		crosscorr = [(f.inputs.index(inpA), f.inputs.index(inpB)) for inpA, inpB in f.corr_products]
        		#extract the fringes
			fringes = np.angle(f.vis[:,:,:])
        		#For plotting the fringes
        		fig.subplots_adjust(wspace=0., hspace=0.)
        		#debug_here()
        		for n, (indexA, indexB) in enumerate(crosscorr):
            			subplot_index = (num_ants * indexA + indexB + 1) if pol == 'h' else (indexA + num_ants * indexB + 1)
            			ax = fig.add_subplot(num_ants, num_ants, subplot_index)
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
    #plt.savefig(pp,format='pdf')
	except KeyError , error:
    		print 'Failed to read scans from File: ',fn,' with Key Error:',error
	except ValueError , error:
    		print 'Failed to read scans from File: ',fn,' with Value Error:',error
	plt.savefig(pp,format='pdf')

else:
	print "No bandpass calibrators found, we wont plot fringes"
#Plot cross correlation spectra 
f.select()
cor = f.corr_products
ants = f.ants
num_ants = len(ants)
bp = np.array([t.tags.count('target') for t in f.catalogue.targets]) == 1
for look in range(len(bp)):
	if bp[look]==1:
		a=1
		break
	else:
		a=0
if a==1:
	bp = np.arange(len(bp))[bp]
	#code to plot the cross phase ... fringes
	print "Plotting Cross Correlation Spectra"
	fig = plt.figure(figsize=(21,15))
	try:
		j=0
		#plt.figure()
		n_channels = len(f.channels)
		for pol in ('h','v'):
        		f.select(targets=bp,corrprods = 'cross',pol=pol)
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
else:
	print "No sources are tagged as 'target', Cant plot cross correlation spectra."	
plt.close('all')
pp.close()

print 'The results are save on ~/comm/scripts/obs_report/obs_report_svnSciences/'+datafile[:-3]+'.pdf'
print "Pull the pdf file and  upload it to the elog with necessary comments"


#show()

#f=katfile.open("1345435475.h5")

#In [20]: f.select(ants="ant1",pol="v")

#In [21]: nvis=np.abs(f.vis[:])

#In [22]: imshow(nvis[:,:,0],aspect='auto',origin='lower')
#Out[22]: <matplotlib.image.AxesImage at 0x410ca90>



