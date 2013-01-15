import katfile
import os
import glob
import katoodt
import time
import matplotlib.pyplot as pl
from optparse import OptionParser
from pylab import *
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Set up standard script options
parser = OptionParser(usage="usage: %prog [options]")
parser.add_option('-d', '--filename', help="datafile to be plotted "+ "(e.g. 1340018260.h5)")
opts, args = parser.parse_args() 

count=0

def times(ant,pol,count):
	figure(figsize=(13,10), facecolor='w', edgecolor='k')
	xlabel("Seconds from "+starttime,fontweight="bold")
	ylabel("Amplitude",fontweight="bold")
	for ant_x in ant:
		print ("plotting "+ant_x.name+"_" +pol+pol+ " time series")
       		f.select(ants=ant_x,corrprods='auto',pol=pol)
    		f.select(channels=range(200,800))
		if count==0:
			plot(f.timestamps - f.timestamps[0],10*np.log10(mean(abs(f.vis[:]),1)),label=(ant_x.name+'_'+pol+pol))
		elif count==1:
			plot(f.timestamps - f.timestamps[0],10*np.log10(mean(abs(f.vis[:]),1)),label=(ant_x.name+'_'+pol+pol))
	
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
	d=katoodt.get_archived_product(product_name=datafile)
else: 
	print "File found locally "
	d=searched
print "Opening the file using katfile, this might take a while"
f=katfile.open(d)
figure(figsize = (13,7))
axes(frame_on=False)
xticks([])
yticks([])
title(datafile+" Observation Report",fontsize=14, fontweight="bold")
mystring=f.__str__()
mystring_seperated=mystring.split("\n")
mystr=""
for i in range(23):
	mystr=mystr+mystring_seperated[i]+"\n"

text(0,0,mystr,fontsize=12)
savefig(pp,format='pdf')
print f

#Get file start time from file name
epoctime=datafile[:-3]
epoctime=float(epoctime)
loctime=time.localtime(epoctime)
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
starttime=str(loctime.tm_hour) +":"+ str(loctime.tm_min)+":"+ str(loctime.tm_sec)+" "+ str(loctime.tm_mday) +" "+ months[loctime.tm_mon-1]+" "+ str(loctime.tm_year)

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
ax1.plot(f.timestamps - f.timestamps[0],f.sensor['Enviro/asc.air.temperature'],'b-')
ax1.set_xlabel("Seconds from "+starttime, fontweight="bold")
ax1.set_ylabel('Temperature (Deg C)', color='b',fontweight="bold")
for tl in ax1.get_yticklabels():
	tl.set_color('b')

ax2=ax1.twinx()
ax2.plot(f.timestamps - f.timestamps[0],f.sensor['Enviro/asc.wind.speed'],'r-')
ax2.set_ylabel('Wind Speed (m/s)',fontweight="bold", color='r')
for tl in ax2.get_yticklabels():
	tl.set_color('r')

ax3=fig.add_subplot(212)
ax3.plot(f.timestamps - f.timestamps[0],f.sensor['Enviro/asc.air.relative-humidity'],'g-')
ax3.set_xlabel("Seconds from "+starttime,fontweight="bold")
ax3.set_ylabel('Relative Humidity (%)', fontweight="bold",color='g')
for tl in ax3.get_yticklabels():
	tl.set_color('g')

savefig(pp,format='pdf')

pp.close()



#show()

#f=katfile.open("1345435475.h5")

#In [20]: f.select(ants="ant1",pol="v")

#In [21]: nvis=np.abs(f.vis[:])

#In [22]: imshow(nvis[:,:,0],aspect='auto',origin='lower')
#Out[22]: <matplotlib.image.AxesImage at 0x410ca90>



