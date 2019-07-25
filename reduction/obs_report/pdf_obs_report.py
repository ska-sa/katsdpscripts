#! bin/usr/python
import katdal
import katpoint
import pysolr
import h5py
import optparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from katsdpscripts import git_info
from scipy import math
import numpy as np


def get_reduction_metadata(filename, reduction_name=None):
    #Get all reduction products from filename  with given reduction_name
    #(or all reduction products if reduction_name is None)
    mysolr = pysolr.Solr('http://kat-archive.kat.ac.za:8983/solr/kat_core')
    fn_search_result = mysolr.search('Filename:'+filename)
    if fn_search_result.hits < 1:
        return []
    CASProductId = fn_search_result.docs[0]['CAS.ProductId']
    reduction_products = mysolr.search('InputDataProductId:'+CASProductId)
    if reduction_name==None:
        return reduction_products.docs
    else:
        return [product for product in reduction_products.docs if product.get('ReductionName')==reduction_name]

    

def plot_flags(timestamps,freqs,flags):
    """timestamps is an array of unix timstamps
    freqs is an array of frequencys in MHz
    flags is a 2D array of boolean with [time,freq]
    it plots the percentage of the data flagged 
    as a function of time vs frequency  """
    fig=plt.figure(figsize=(10,5))
    #
    print((timestamps[:]-timestamps[0]).shape,(freqs).shape,((flags).T).shape)
    plt.pcolormesh(timestamps[:]-timestamps[0],freqs,(flags).T,rasterized=True)
    plt.title('Flags in "quiet" part of the band')
    plt.xlabel('Time (s), since %s' % (katpoint.Timestamp(timestamps[0]).local(),))
    plt.ylabel('Frequency/[MHz]')
    plt.xlim(0,timestamps[-1]-timestamps[0])
    plt.ylim(freqs[0],freqs[-1])
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    return fig 

def plot_flagtype(flag_dat,labels):
    """flag_dat is an array of length labels of percentages. 
    lables is a list of str corresponting to the bits in flags"""
    fig=plt.figure(figsize=(10,5))
    plt.xticks(list(range(len(labels))), labels, rotation=38)
    plt.ylabel('Percentage Flagged')
    plt.title('Flag Types ')
    plt.plot(flag_dat,'*',rasterized=True)
    plt.grid()
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    return fig


def plot_activity(scan_types,activity_count):
    """scan_type is a set consisiting of scans :['track', 'stop', 'slew'] and 
    activity_count is an array activities perfomed by each antenna."""
    fig=plt.figure(figsize=(12,2))
    for s,st in enumerate(scan_types):
        plt.plot(np.arange(len(data.ants)),activity_count[s,:],"o",label=st,rasterized=True)
    plt.xticks(np.arange(len(data.ants)), [ant.name for ant in data.ants])
    plt.legend(loc=5)
    plt.grid()
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    plt.xlim(-1,len(data.ants)+2)
    plt.title('Number of dumps in each scan state')
    return fig 


def plot_timeseries(num_bls,baseline_names,scan_az,scan_el,scan_timestamps,scan_vis,scan_freqinds,scan_autos):
    """num_bls is the number of baselines in the array, baseline_names is a list of all the correlation product
    in active_pols, scan_az is a list of all the scan in azemuth, scan_el is the list of all the scans in elevation
    ,scan_timestamps is a list of timestamps for each each scan,scan_autos is a list amplitude for each antenna.
    """
    #plot timeseries,spectrograph and spectrum for each antenna
    #for p,active_pol in enumerate(['H','V']):
    print('active_pol',active_pol)
    return_figs = []
    for i in range(num_ants):
        print('Antenna name', data.corr_products[i][0],i)
        ant_name = data.corr_products[i][0]
        amp = np.vstack([scan[:,:,i] for scan in scan_autos])
        az = np.hstack([a[i] for a in scan_az])
        el =np.hstack([e[i] for e in scan_el])
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Ant: %s'%(ant_name), fontsize=16)
        fig.subplots_adjust(top=0.95)
        ax1.errorbar(np.hstack(scan_timestamps),np.nanmean(amp,axis=1),np.nanstd(amp,axis=1),color='grey')

        ax1.fill_between(np.hstack(scan_timestamps), np.nanmin(amp,axis=1), np.nanmax(amp,axis=1),color='lightgrey')

        ax1.plot(np.hstack(scan_timestamps),np.nanmean(amp,axis=1),color='b',rasterized=True)
        ax1.set_title('Time series')
        ax1.set_xlabel('Time (s), since %s' % (katpoint.Timestamp(data.start_time).local(),))
        ax1.set_ylabel('Correlator counts')
        ax1.set_ylim(0,4000)
        ax1.set_xlim(scan_timestamps[0][0],scan_timestamps[-1][-1])
        for t in scan_timestamps:
            ax1.axvline(t[0],linestyle='dotted',color='mistyrose',linewidth=1)

        ma_amp = np.ma.masked_where(np.isnan(amp.T),amp.T)
        ax2.pcolormesh(np.hstack(scan_timestamps),data.channel_freqs/1e6,10*np.log10(ma_amp),rasterized=True)
        ax2.set_title('Spectrogram')
        ax2.set_xlabel('Time (s), since %s' % (katpoint.Timestamp(data.start_time).local(),))
        ax2.set_xlim(scan_timestamps[0][0],scan_timestamps[-1][-1])
        ax2.set_ylabel('Frequency/[MHz]')
        ax2.set_ylim(data.channel_freqs[0]/1e6,data.channel_freqs[-1]/1e6)

        ax3.errorbar(data.channel_freqs/1e6,np.nanmean(amp,axis=0),np.nanstd(amp,axis=0),color='grey')
        ax3.fill_between(data.channel_freqs/1e6, np.nanmin(amp,axis=0), np.nanmax(amp,axis=0),color='lightgrey')
        ax3.plot(data.channel_freqs/1e6,np.nanmean(amp,axis=0),color='b',rasterized=True)
        ax3.set_title('Spectrum')
        ax3.set_xlabel('Frequency/[MHz]')
        ax3.set_xlim(data.channel_freqs[0]/1e6,data.channel_freqs[-1]/1e6)
        ax3.set_ylabel('Correlator counts')
        ax3.set_ylim(0,4000)

        ax4.plot(az,el,"*",rasterized=True)
        ax4.set_title('Pointing')
        ax4.set_xlabel('Azimuth/[Degrees]')
        ax4.set_ylabel('Elevation/[Degrees]')
        return_figs.append(fig) # I hope the garbage collector does not get this .
    return return_figs
    
def phase_plot(phasedata,baseline_names,num_chans,scan_timestamps,num_bls):
    """phasedata is an array consiting of phase information for all correlation products,
    num_chans is the number of channels"""
    x1,x2 = scan_timestamps[0][0],scan_timestamps[-1][-1]
    scan_freqinds = [np.arange(num_bls * num_chans)] * len(scan_timestamps)
    y1,y2 = scan_freqinds[0][0]-0.5,scan_freqinds[0][-1]+0.5
       
    #num_bls = len(baseline_names)
    if num_bls > 0 :
        fig2=plt.figure(0,figsize=(12,num_bls),tight_layout=True)
    else :
        fig2=plt.figure(0,figsize=(12,1),tight_layout=True)
    #Put plots side-by-side if there are <10 scans.
    p = 0
    for p,active_pol in enumerate(['H','V']):
        if len(scan_timestamps)<10:
            plt.subplot (1,2,p+1)
        else:
            plt.subplot(2,1,p+1)

        plt.imshow(phasedata[active_pol][~np.all(np.isnan(phasedata[active_pol]),axis=1)],aspect='auto',origin='lower',extent=[x1,x2,y1,y2],interpolation='nearest',rasterized=True)
        plt.xlabel('Time (s), since %s' % (katpoint.Timestamp(data.start_time).local(),))
        if p == 1 and len(scan_timestamps) < 10:
            plt.yticks(np.arange(num_chans // 2, num_bls * num_chans, num_chans), np.repeat('',num_bls))
        else:
            plt.yticks(np.arange(num_chans // 2, num_bls * num_chans, num_chans), baseline_names)
        for yval in range(0, num_bls * num_chans, num_chans):
            plt.axhline(yval, color='k', lw=2)
        plt.title('Raw visibility phase per baseline, Pol %s'%active_pol)
    return fig2

parser = optparse.OptionParser(usage='%prog [options]',
                               description='Runs the observation report and writes on pdf')
(opts,args)=parser.parse_args()


if len(args)<1:
    raise RuntimeError('Please specify the data file to reduce')
else:
    for filename in args:
        print('Opening ', filename)
        try:
            res = get_reduction_metadata(filename.split('/')[-1],reduction_name = 'AR1 Generate RFI Flags');
            flag_results = res[0]['CAS.ReferenceDatastore']
            for fl in flag_results:
                if fl.endswith('flags.h5'):
                    flag_filename = fl.split(':')[-1]
            print(flag_filename)
            h5_flags = h5py.File(flag_filename,'r')
            cal_flags = True
        except:
            print("no calibration pipeline flags found")
            cal_flags=False

        #Loading data 
        data=katdal.open(filename)
        if cal_flags:
            data._flags = h5_flags['flags']
        
        #Nice name for pdf 
        h5name = data.name.split('/')[-1]
        obs_details = h5name +'_AR1_observation_report'
        pp =PdfPages(obs_details+'.pdf')

        N = data.shape[0]
        ext = 930
        step = max(int(math.floor(N/ext)),1)
        
        #Loading flags
        data.select()
        M = 4 * np.shape(data)[1] / 4096
        start_chan = 2200*M//4
        end_chan = 2800*M//4
        if data.receivers[list(data.receivers.keys())[0]][0] == 'u' :
            start_chan = 0#2200*M//4
            end_chan = data.shape[1]#2800*M//4
        data.select(channels=slice(start_chan,end_chan))
        time=data.timestamps-data.timestamps[0]
        freqs=data.channel_freqs/1e6
       
        flags = np.zeros((time.shape[0],freqs.shape[0])) 
        for scan in data.scans():
            flags[data.dumps,:] += data.flags[:].sum(axis=2)
            #print data.flags.shape
        #Getting antenna activity 
        activity = []
        for ant in data.ants:
            activity.append(data.sensor['Antennas/%s/activity'%ant.name])
        scan_types = set(np.hstack(activity))
        activity_count = np.zeros([len(scan_types),len(data.ants)])

        for s,st in enumerate(scan_types):
            for i in range(len(data.ants)):
                activity_count[s,i] = np.sum(np.vstack(activity)[i,:]==st)
        
        #Calling function plot flags
        fig=plot_flags(data.timestamps[:],freqs,flags)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
        del(flags)
        
        #Calling fuction plot flagtype
        flag_labels = list(data.file['/Data/flags_description'][:,0])
        flag_dat = np.zeros((len(flag_labels)))
       
        for time_index in range(data.dumps[0],data.dumps[-1]):
            tmp_data = data.file['Data/flags'][time_index,slice(start_chan,end_chan),:].flatten()
            flag_dat += np.unpackbits(tmp_data[:,np.newaxis],axis=1).sum(axis=0)
        flag_dat = flag_dat/(np.prod(data.shape))*100.
        fig= plot_flagtype(flag_dat,flag_labels)
        fig.set_size_inches(10,8)
        fig.savefig(pp,format='pdf')
        plt.close(fig)

        #Calling function plot activity 
        fig=plot_activity(scan_types,activity_count)
        fig.savefig(pp,format='pdf')
        plt.close(fig)        
        
        sp={}
        for p,active_pol in enumerate(['H','V']):
            data.select()
            data.select(pol=active_pol,scans=['scan','track'],dumps=slice(0,-1,step),channels=slice(0,-1,M))
            num_ants = len(data.ants)
            baseline_names = [('%s - %s' % (inpA[:-1], inpB[:-1])) for inpA, inpB in data.corr_products[num_ants:]]

            num_bls = len(baseline_names)
            scan_az, scan_el, scan_timestamps, scan_vis = [], [], [], []
            for scan_ind, state, target in data.scans():
                ts = data.timestamps
                vis = data.vis[:]
                flags = data.flags[:]
                vis[flags] = np.nan
                scan_vis.append(vis)
                scan_az.append(np.median(data.az,axis=0))
                scan_el.append(np.median(data.el,axis=0))
                scan_timestamps.append(ts - data.start_time.secs)
            num_chans = len(data.channel_freqs)
            scan_freqinds = [np.arange(num_bls * num_chans)] * len(scan_timestamps)
            scan_phase = [np.angle(Vis[:,:,num_ants:]).T.reshape(-1, Vis.shape[0]) for Vis in scan_vis]
            scan_autos = [np.abs(Vis[:,:,:num_ants]) for Vis in scan_vis]
            sp[active_pol]=np.hstack(scan_phase) 
                
            #Calling function timeseries plot
            fig_list=plot_timeseries(num_bls,baseline_names,scan_az,scan_el,scan_timestamps,scan_vis,scan_freqinds,scan_autos)
            for fig in fig_list :
                fig.savefig(pp,format='pdf')
                plt.close(fig)

        #Calling function timeseries plot
        fig=phase_plot(sp,baseline_names,num_chans,scan_timestamps,num_bls)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
        pp.close()
        plt.close('all')
