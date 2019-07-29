# coding: utf-8
import katdal
import numpy as np
import matplotlib
import numba
from numba import jit,njit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from katsdpscripts import git_info
import pickle
import optparse

def phase_combinations(ant_list,look_up):
    l1,l2,l3 = [],[],[]
    for a1,a2,a3 in itertools.combinations(set( ant_list), 3):
        l1.append(look_up[a1,a2])
        l2.append(look_up[a1,a3])
        l3.append(look_up[a2,a3])
    return l1,l2,l3

def amp_combinations(ant_list,look_up):
    l1,l2,l3,l4 = [],[],[],[]
    for a1,a2,a3,a4 in itertools.combinations(set( ant_list), 4):
        l1.append(look_up[a1,a2])
        l2.append(look_up[a1,a3])
        l3.append(look_up[a2,a4])
        l4.append(look_up[a3,a4])
    return l1,l2,l3,l4 

def anglemean(th,axis=None):
    """ Return the mean of angles
    Multiply angles by 2 for an directionless orentation
    eg. polorisation """
    sa = np.nansum(np.sin(th),axis=axis)
    ca = np.nansum(np.cos(th),axis=axis)
    return np.arctan2(sa,ca)

def plot_phase(channel_freqs,a123,title=''):
    """
    channel_freqs is an array of channel frequencys in Hz
    a123 is the closure quantity in radians 
    """
    fig = plt.figure(figsize=(20,10))
    plt.title('Mean Phase Closure angle(degrees)'+title) 
    plt.imshow(np.degrees(a123[:,:]),aspect='auto' )
    #plt.ylim(-5,5)
    plt.grid(True)
    #plt.ylabel('Mean Phase Closure angle(degrees)')
    plt.xlabel('Baseline number')
    plt.ylabel('Time (seconds)')
    plt.colorbar()
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    return fig 

def plot_phase_graph(channel_freqs,a123,title=''):
    """
    channel_freqs is an array of channel frequencys in Hz
    a123 is the closure quantity in radians 
    """
    fig = plt.figure(figsize=(20,10))
    plt.title('Mean Phase Closure angle(degrees)'+title) 
    plt.plot(np.degrees(a123[:,:].mean(axis=0)) )
    plt.grid(True)
    plt.ylabel('Mean Phase Closure angle(degrees)')
    plt.xlabel('Baseline number')
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    return fig 



def plot_amp(channel_freqs,a1234,title=''):
    """
    channel_freqs is an array of channel frequencys in Hz
    a1234 is the closure quantity  
    """
    fig = plt.figure(figsize=(20,10))
    plt.title('Mean Amplitude Closure ' + title)
    logthresh = np.log10(2)   
    plt.imshow(a1234,aspect='auto',vmin=float(np.min(a1234)), vmax=float(np.max(a1234)),norm=matplotlib.colors.SymLogNorm(10**-logthresh) )
    plt.grid(True)
    #plt.ylabel('Mean Amplitude Closure ')
    plt.xlabel('Baseline number')
    plt.ylabel('Time (seconds)')
    plt.colorbar()
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)    
    return fig 

def plot_amp_graph(channel_freqs,a1234,title=''):
    """
    channel_freqs is an array of channel frequencys in Hz
    a1234 is the closure quantity  
    """
    fig = plt.figure(figsize=(20,10))
    plt.title('Mean Amplitude Closure ' + title)
    plt.plot(a1234.mean(axis=0) )
    plt.grid(True)
    plt.ylabel('Mean Amplitude Closure ')
    plt.xlabel('Baseline number')
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)    
    return fig 

plot_amp_graph

@jit(nopython=True, fastmath=True,parallel=True)
def phase_closure(l1,l2,l3,full_vis):
    temp = np.zeros((len(l1)))
    for j in numba.prange(len(l1)): #for j in range(len(l1)):
        sa = np.sum(np.sin(np.angle(full_vis[:,l1[j]])-np.angle(full_vis[:,l2[j]]) +np.angle(full_vis[:,l3[j]]) ) )
        ca = np.sum(np.cos(np.angle(full_vis[:,l1[j]])-np.angle(full_vis[:,l2[j]]) +np.angle(full_vis[:,l3[j]]) ) )
        temp[j] = np.arctan2(sa,ca)
    return temp

@jit(nopython=True, fastmath=True,parallel=True)
def amplitude_closure(l1,l2,l3,l4,full_vis):
    temp = np.zeros((len(l1)))
    for j in numba.prange(len(l1)): #for j in range(len(l1)):
        temp[j] = np.nanmean((np.abs(full_vis[:,l1[j]])*np.abs(full_vis[:,l4[j]]))/(np.abs(full_vis[:,l2[j]])*np.abs(full_vis[:,l3[j]] ) ) )
    return temp


@jit(nopython=True, fastmath=False,parallel=False)
def phase_closure_baseline(l1,l2,l3,phase_closure,n):
    temp = np.zeros(n)
    for j in numba.prange(len(l1)): #for j in range(len(l1)):
        temp[np.mod(l1[j],n)] += phase_closure[j]
    for j in numba.prange(len(l2)): #for j in range(len(l1)):
        temp[np.mod(l2[j],n)] += phase_closure[j]
    for j in numba.prange(len(l3)): #for j in range(len(l1)):
        temp[np.mod(l3[j],n)] += phase_closure[j]
    return temp

@jit(nopython=True, fastmath=False,parallel=False)
def amplitude_closure_baseline(l1,l2,l3,l4,amplitude_closure,n):
    temp = np.zeros(n)
    for j in numba.prange(len(l1)): #for j in range(len(l1)):
        temp[np.mod(l1[j],n)] += amplitude_closure[j]
    for j in numba.prange(len(l2)): #for j in range(len(l1)):
        temp[np.mod(l2[j],n)] += amplitude_closure[j]
    for j in numba.prange(len(l3)): #for j in range(len(l1)):
        temp[np.mod(l3[j],n)] += amplitude_closure[j]
    for j in numba.prange(len(l3)): #for j in range(len(l1)):
        temp[np.mod(l4[j],n)] += amplitude_closure[j]
    return temp


# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a plots of the closure quantitiys in a pdf file.')
parser.add_option("-f", "--freq-chans", default=None,
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default= %default)")
parser.add_option("-d", "--print-description", action="store_true",default=False,
                  help="Add an additional page that discribes the therory of the plots, default= %default)")
parser.add_option("-t", "--token", default=None,
                  help="Archive token parameter, default= %default)")
parser.add_option("-c", "--channel-mask", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', 
                  help="Optional pickle file with boolean array specifying channels to mask (default=%default)")

(opts, args) = parser.parse_args()


if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')

if opts.token is not None:
    h5 = katdal.open(args[0],token=opts.token,s3_endpoint_url='https://archive-gw-1.kat.ac.za')
else:
    h5 = katdal.open(args[0])
h5.select()
n_chan = h5.channels.shape[0]
nice_filename =  args[0].split('/')[-1]+ '_closure_stats'
pp = PdfPages(nice_filename+'.pdf')
if not opts.freq_chans is None :
    start_freq_channel = int(opts.freq_chans.split(',')[0])
    end_freq_channel = int(opts.freq_chans.split(',')[1])
    edge = np.tile(True, n_chan)
    edge[slice(start_freq_channel, end_freq_channel)] = False
else :
    edge = np.tile(False, n_chan)
#load static flags if pickle file is given
if len(opts.channel_mask)>0:
    pickle_file = open(opts.channel_mask,mode='rb')
    rfi_static_flags = pickle.load(pickle_file)
    pickle_file.close()
else:
    rfi_static_flags = np.tile(False, n_chan)

static_flags = np.logical_or(edge,rfi_static_flags)
target='J1939-6342'
h5.select(targets=target,scans='track')
for pol in ['h','v'] :
    h5.select(pol=pol,channels=~static_flags)
    phase_r,amp_r = np.zeros((h5.shape[0],h5.shape[2])),np.zeros((h5.shape[0],h5.shape[2]))
    tmp  = np.zeros((h5.shape[2]*2))
    N_ants = len(h5.ants)
    antA = [h5.inputs.index(inpA) for inpA, inpB in h5.corr_products]
    antB = [h5.inputs.index(inpB) for inpA, inpB in h5.corr_products]
    
    #vis = h5.vis[i,:,:] #  2D  freq,baseline
    full_antA = np.r_[antA, antB]
    full_antB = np.r_[antB, antA]
    corrprods = list(zip(full_antA,full_antB))
    up = {}
    for i,(x,y)  in enumerate(zip(full_antA,full_antB)): # make lookup table
        up[x,y]=i
        up[y,x]=i
    
    for i in range(h5.vis.shape[0]):
        print('Get Data')
        full_vis = np.concatenate((h5.vis[i,:,:], h5.vis[i,:,:].conj()), axis=-1)
        print('G0t Data')
        #h5.select(scans='track',targets='PKS1934-638')
        
        l1,l2,l3 = phase_combinations(full_antA,up)
        if np.shape(l1)[0] > 0 : 
            a123 = phase_closure(l1,l2,l3,full_vis)
            phase_r[i,:] = phase_closure_baseline(l1,l2,l3,a123,full_vis.shape[1]//2)  
        l1,l2,l3,l4  = amp_combinations(full_antA,up)
        if np.shape(l1)[0] > 0 : 
            a1234 =  amplitude_closure(l1,l2,l3,l4,full_vis)
            amp_r[i,:] = amplitude_closure_baseline(l1,l2,l3,l4,a1234,full_vis.shape[1]//2)  
        print(i)
    title = "filename:%s   pol=%s  , target=%s "%(args[0].split('/')[-1],pol,target)
    fig = plot_phase(h5.channel_freqs,phase_r,title)
    fig.savefig(pp,format='pdf')
    plt.close(fig)
    title = "filename:%s   pol=%s  , target=%s "%(args[0].split('/')[-1],pol,target)
    fig = plot_phase_graph(h5.channel_freqs,phase_r,title)
    fig.savefig(pp,format='pdf')
    plt.close(fig)
    title = "filename:%s   pol=%s  , target=%s "%(args[0].split('/')[-1],pol,target)
    fig = plot_amp(h5.channel_freqs,amp_r,title)
    fig.savefig(pp,format='pdf')
    plt.close(fig)
    title = "filename:%s   pol=%s  , target=%s "%(args[0].split('/')[-1],pol,target)
    fig = plot_amp_graph(h5.channel_freqs,phase_r,title)
    fig.savefig(pp,format='pdf')
    plt.close(fig)
 
if opts.print_description :
    text = r''
    text +=r"""
     The Phase relationship equation is:
 
     $ \Phi_{12}  = \phi_{1} - \phi_{2} +  \phi_{12}  $
 
     $ \Phi_{13}  = \phi_{1} - \phi_{3} +  \phi_{13}  $
 
     $ \Phi_{23}  = \phi_{2} - \phi_{3} +  \phi_{23}  $
 
     $ \Phi_{12} - \Phi_{13} + \Phi_{23} =  \phi_{12} - \phi_{13} +\phi_{23} $
 
     For a point source , $\phi_{12} = \phi_{13} = \phi_{23} = 0 $
    """
    text +=r"""
     \n\n
     The Amplitude relationship equation is:
 
     $ A_{12}  = a_{1}a_{2}^{*}a_{12}S_{12}  $
 
     $ A_{13}  = a_{1}a_{3}^{*}a_{13}S_{13}  $
 
     $ A_{24}  = a_{2}a_{4}^{*}a_{24}S_{24}  $
 
     $ A_{34}  = a_{3}a_{4}^{*}a_{34}S_{34}  $  
 
     $ \frac{A_{12}A_{34}}{A_{13}A_{24}} =  \frac{a_{12}a_{34}}{a_{13}a_{24}} S $
 
     For a point source , $S_{12} = S_{13} = S_{24} = S_{34} = S $
    """
    fig = plt.figure(figsize=(20,10))
    plt.figtext(0.1,0.1,text,fontsize=10)
    fig.savefig(pp,format='pdf')
    plt.close(fig)
pp.close()
plt.close('all')
#TODO Produce stats per baseline. vs time . Calc stats per antenna  and per time ?