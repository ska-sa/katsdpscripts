#!/usr/bin/python
# Read data from a file and plot some stats
import numpy as np
import optparse
 
 

def histogramtxt(data,pos=None,num=10000,bit_range=6):
    """num = number of samples (int)
    pos = position in file (int)
    data is the memmap obj or the numpy array 
    bit_range = the range of bits to plot"""
    if pos == None :
        pos = np.floor((data.shape[0]-num)*np.random.random() ).astype(int)
    aaa = np.histogram(data[pos:pos+num],bins=np.arange(2**bit_range+1)-(2**(bit_range-1)-.5) )
    print("Histogram of %i samples from position %i"%(num,pos))
    for a,b in zip(aaa[1][1:]-0.5,aaa[0]):
        print('%6i %10i %s' % (a,b, 'x'.rjust(np.max([1,int(np.float(b)/aaa[0].max()*50)]),'-')  ))


def spectrumtxt(data,pos=None,channels=2**8,num_spectra=1024):
    """num_spectra = number of spectra (int)
    pos = position in file (int)
    data is the memmap obj or the numpy array 
    channels = the number of channels in the fft"""
    if pos == None :
        pos = np.floor((data.shape[0]-num_spectra*channels)*np.random.random() ).astype(int)        
    print(' Graph of average spectrum')
    print('%i x Average spectrum in %i channels from position %i' % (num_spectra,int(channels/2.),pos))
    print(' Ch    Freq     dB  Graph')
    for a,b in enumerate((np.abs(np.fft.fft(data[pos:pos+num_spectra*channels].reshape(num_spectra,channels),axis=1)[:,:channels/2])/channels).mean(axis=0)):
      print('%3i %8.1f %5.1f %s' % (a,856+1712.0*a/channels,20*np.log10(b), 'x'.rjust(np.max([1,int(40+20*np.log10(b))]),'-')  ))

def power_time_txt(data,pos=None,sample_size=32768,num_samples=100):
    """num_spectra = number of spectra (int)
    pos = position in file (int)
    data is the memmap obj or the numpy array 
    sample_size = the number of samples to average together
    the number of averaged samples to plot"""
    if pos == None :
        pos = np.floor((data.shape[0]-sample_size*num_samples)*np.random.random() ).astype(int)        
    
    # total power over time
    ts=1.0e6*sample_size/1712e6     # microseconds per dump
    print('Graph of power over time')
    print('%i Number of samples per accumulation from position %i' % (sample_size,pos))
    C = np.mean(data[pos:pos+sample_size*num_samples].reshape(num_samples,sample_size)**2.0,axis=1)
    gmax=C.max()          # graphmax
    print(' dump     us    dB  Graph')
    for a,b in enumerate(C):
      print('%4i %7.1f %6.1f %s' % (a,ts*a,10*np.log10(b), 'x'.rjust(int(80*b/gmax),'-')  ))
    #print 'peak occurs at fft bin edge frequency of %8.2f Hz' % ((abs(np.fft.fft(C))[1:todo/2].argmax()+1)/(ts*todo)*1e6)



# Set up standard script options
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script produces some text plots so that data can be examind')
# Add experiment-specific options
parser.add_option('-p', '--power-time', action="store_true", default=False,
                  help='Plot a text power vs time graph (default=%default)')
parser.add_option('-b', '--hist', action="store_true", default=False,
                  help='Plot a histogram graph (default=%default)')
parser.add_option('-s', '--spectrum', action="store_true", default=False,
                  help='Plot a text spectrum graph (default=%default)')

parser.add_option('-n','--num',  type='int', default=128,
                  help='number of lines output,  (default=%default)')


#major/minor
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='UHF signal generator track',dump_rate=1.0,nd_params='off')
# Parse the command line
opts, args = parser.parse_args()


if len(args) ==0 :
    raise RuntimeError('No file passed to the script')

data = np.load(args[0] , mmap_mode='r')
if opts.power_time :
    power_time_txt(data,sample_size=32768,num_samples=int(opts.num))
if opts.spectrum :
    spectrumtxt(data,channels=2**np.ceil(np.log2(int(opts.num)))*2,num_spectra=1024)
if opts.hist :
    histogramtxt(data,num=10000,bit_range=np.ceil(np.log2(int(opts.num)/2)).astype(int) )
    
if not( opts.hist or opts.spectrum or opts.power_time ) :
    print("Not plotting options given , look at the script help '--help' ")