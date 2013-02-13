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

def times(ant,pol,count):
	figure(figsize=(13,10), facecolor='w', edgecolor='k')
	xlabel("LST on "+starttime,fontweight="bold")
	ylabel("Amplitude",fontweight="bold")
	for ant_x in ant:
		print ("plotting "+ant_x.name+"_" +pol+pol+ " time series")
       		f.select(ants=ant_x,corrprods='auto',pol=pol)
    		f.select(channels=range(200,800))
		if count==0:
			plot(f.lst,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant_x.name+'_'+pol+pol))
		elif count==1:
			plot(f.lst,10*np.log10(mean(abs(f.vis[:]),1)),label=(ant_x.name+'_'+pol+pol))
	
	legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)
	savefig(pp,format='pdf')


#Spectrum function
def spec(pol,datafile,starttime):
	count=0	
	figure(figsize=(10,10), facecolor='w', edgecolor='k')
	subplot(2,1,(count+1))
	xlabel("Frequency", fontweight="bold")
	ylabel("Amplitude", fontweight="bold")
	title(ant_x.name+" Spetral Plots",fontsize=12, fontweight="bold")

        print ("plotting "+ant_x.name+"_" +pol[count]+pol[count]+ " spectrum")
       	f.select(ants=ant_x,corrprods='auto',pol=pol[count])
        nvis=np.abs(f.vis[:])
	f_min=nvis.min(axis=0)
	f_mean=nvis.mean(axis=0)
	f_max=nvis.max(axis=0)
	plot(f.channel_freqs,10*np.log10(f_min),label=(ant_x.name+'_'+pol[count]+pol[count]+'_min'))
	plot(f.channel_freqs,10*np.log10(f_mean),label=(ant_x.name+'_'+pol[count]+pol[count]+'_mean'))
	plot(f.channel_freqs,10*np.log10(f_max),label=(ant_x.name+'_'+pol[count]+pol[count]+'_max'))
	legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)


	count=1
	subplot(2,1,(count+1))
	xlabel("Frequency", fontweight="bold")
	ylabel("Amplitude", fontweight="bold")
	print ("plotting "+ant_x.name+"_" +pol[count]+pol[count]+ " spectrum")
       	f.select(ants=ant_x,corrprods='auto',pol=pol[count])
        nvis=np.abs(f.vis[:])
	f_min=nvis.min(axis=0)
	f_mean=nvis.mean(axis=0)
	f_max=nvis.max(axis=0)
	plot(f.channel_freqs,10*np.log10(f_min),label=(ant_x.name+'_'+pol[count]+pol[count]+'_min'))
	plot(f.channel_freqs,10*np.log10(f_mean),label=(ant_x.name+'_'+pol[count]+pol[count]+'_mean'))
	plot(f.channel_freqs,10*np.log10(f_max),label=(ant_x.name+'_'+pol[count]+pol[count]+'_max'))
	legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)

	savefig(pp,format='pdf')
	#savefig("/data/siphelele/time_series_plots/time_series/"+datafile[:-3]+"_"+ant_x.name+"_Spectro.png")
	count=0

if opts.filename is None:
	print 'Specifiy a file to be plotted using option -d'
	sys.exit()



#get data file using katarchive and open it using katfile

datafile =opts.filename
pathtofile = 'locate ' +datafile
searched=glob.glob(datafile)                               #search the file locally 

pp = PdfPages(datafile[:-3]+'.pdf')

if not searched:							   #If not found locally download it
	print 'File not found locally, Accesing the file directly from the archive'
	d=katarchive.get_archived_products(datafile)
	#d=katoodt.get_archived_product(product_name=datafile)
else: 
	print "File found locally "
	d=searched

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
figure(figsize = (13,7))
axes(frame_on=False)
xticks([])
yticks([])
title(datafile+" Observation Report",fontsize=14, fontweight="bold")
mystring=f.__str__()
mystring_seperated=mystring.split("\n")

mystr=[]
for i in range(23):
	mystr.append(mystring_seperated[i])
description=mystr[4]
mystr.append(description)
mystr[4]=scp
filestring=collections.deque(mystr)
filestring.rotate(1)
my=""
for a in range(24):
	my=my+filestring[a]+"\n"


text(0,0,my,fontsize=12)
savefig(pp,format='pdf')
print f

#Get file start time from file name
epoctime=datafile[:-3]
epoctime=float(epoctime)
loctime=time.localtime(epoctime)
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
starttime=str(loctime.tm_mday) +" "+ months[loctime.tm_mon-1]+" "+ str(loctime.tm_year)

ant=f.ants
pol=['h','v']
times(ant,pol[0],count)
count=count+1
times(ant,pol[1],count)

for ant_x in ant:
	count=0
	spec(pol,datafile,starttime)


print "Getting wind and temperature sensors"
fig=pl.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.plot(f.lst,f.sensor['Enviro/asc.air.temperature'],'g-')
ax1.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ax1.set_xlabel("LST on "+starttime, fontweight="bold")
ax1.set_ylabel('Temperature (Deg C)', color='g',fontweight="bold")
for tl in ax1.get_yticklabels():
	tl.set_color('g')

ax2=ax1.twinx()
ax2.plot(f.lst,f.sensor['Enviro/asc.air.relative-humidity'],'c-')
ax2.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ylim(10,100)
ax2.set_ylabel('Relative Humidity (%)', fontweight="bold",color='c')
for tl in ax2.get_yticklabels():
	tl.set_color('c')

ax3=fig.add_subplot(212)
ax3.grid(axis='y', linewidth=0.15, linestyle='-', color='k')
ax3.plot(f.lst,f.sensor['Enviro/asc.wind.speed'],'b-')
ylim(ymin=0)
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

pp.close()
print 'The results are save on ~/comm/scripts/obs_report/obs_report_svnSciences/'+datafile[:-3]+'.pdf' 


#show()

#f=katfile.open("1345435475.h5")

#In [20]: f.select(ants="ant1",pol="v")

#In [21]: nvis=np.abs(f.vis[:])

#In [22]: imshow(nvis[:,:,0],aspect='auto',origin='lower')
#Out[22]: <matplotlib.image.AxesImage at 0x410ca90>



