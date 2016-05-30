#!/usr/bin/python
# Read data from a file and plot some stats and find pulses
import numpy as np
import optparse
import matplotlib.pyplot as plt 
 
def MAD_median(data):
    """Median absolute deviation (MAD) is a robust measure of the variability."""
    import time
    start_0 = time.time()
    med = np.median(data)
    start_1 = time.time()
    mad = np.median(np.abs(data-med) )
    end_time = time.time()
    print("Time for operations , 1st median = %f3 2nd median = %f3 "%(start_1-start_0,end_time-start_1))
    return mad,med

def join_pulses(data,pulse_gap=10,offset=0):
    """Find all the timestamps with the pulse"""
    pulse_listtmp = []
    pulse_list = []
    temp = []
    for x in xrange(1,data.shape[0]):
        if data[x] -data[x-1] > pulse_gap : # new pulse
            pulse_listtmp.append(temp)
            temp = []
            temp.append(data[x])
        else :
            temp.append(data[x])
    if len(temp) > 0 :
        pulse_listtmp.append(temp)
        temp = []
    for x in pulse_listtmp :
        pulse_list.append(np.array([np.min(x),np.max(x),offset]))  
    return np.array(pulse_list)

def map_to_raw_data(pulselist,avg_num=256,window_length=256,offset=0):
    """Find all the timestamps with the pulse"""
    #print pulselist
    for pmin,pmax in pulselist:
        pmax = pmax+window_length # rolling window forward
        pmin,pmax = pmin*avg_num,pmax*avg_num # Undo the average period
        yield slice(offset+pmin,offset+pmax) # This is because of memory problems


def rolling_window(a, window,axis=-1,pad=False,mode='reflect',**kargs):
    """
        This function produces a rolling window shaped data with the rolled data in the last col
        a      :  n-D array of data
        window : integer is the window size
        axis   : integer, axis to move the window over
        default is the last axis.
        pad    : {Boolean} Pad the array to the origanal size
        mode : {str, function} from the function numpy.pad
        One of the following string values or a user supplied function.
        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
        con  vector along each axis.
        'median'        Pads with the median value of all or part of the
        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
        the first and last values of the vector along each
        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
        The first values are used to pad the end and the
        end values are used to pad the beginning.
        <function>      of the form padding_func(vector, iaxis_pad_width, iaxis, **kwargs)
        see numpy.pad notes
        **kargs are passed to the function numpy.pad
        
        Returns:
        an array with shape = np.array(a.shape+(window,))
        and the rolled data on the last axis
        
        Example:
        import numpy as np
        data = np.random.normal(loc=1,scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1, size=10000)
        stddata = rolling_window(data, 400).std(axis=-1)
        """
    if axis == -1 : axis = len(a.shape)-1
    if pad :
        pad_width = []
        for i in xrange(len(a.shape)):
            if i == axis:
                pad_width += [(window//2,window//2 -1 +np.mod(window,2))]
            else :
                pad_width += [(0,0)]
        a = np.pad(a,pad_width=pad_width,mode=mode,**kargs)
    a1 = np.swapaxes(a,axis,-1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1],)
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2,axis) # Move original axis to

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
    raise RunTimeError('No file passed to the script')

data = np.load(args[0] , mmap_mode='r')

ts=1.0e6/1712e6     # microseconds per dump
avg_num = 256
window_length = 256
chunk_size = 1024*32768
trans = slice(0,chunk_size)
   
aaa = np.histogram(data[trans],bins=np.arange(2**6+1)-(2**5-.5) )
plt.figure()
plt.plot(aaa[1][1:]-0.5,(aaa[0]) )
plt.xlim(-32,32)

avg_data = (np.abs(data[trans]).reshape(-1,avg_num).mean(axis=-1)).astype(np.float)**2
plt.figure()
plt.plot(avg_data) 
rolled = rolling_window(avg_data, window=window_length)



#measure = (rolled.mean(axis=-1))
#measure = (rolled[...].std(axis=-1))
measure = (rolled[...].std(axis=-1)/rolled.mean(axis=-1))
mad,med =MAD_median(measure)
#print mad,med, med+8*mad,med-8*mad
plt.figure()
plt.plot(measure)
plt.hlines(med-8*mad,0,measure.shape[0])
plt.hlines(med+8*mad,0,measure.shape[0],'r')
plt.ylim(None,med+20*mad)
plt.figure()
plt.semilogy(np.abs(measure - med) / mad)
plt.grid()
plt.ylim(1,None)



pulse = (measure>med+8*mad) + (measure<med-8*mad)
pulse_list = join_pulses(pulse.nonzero()[0])
#print 'cc',pulse_list,'cc'
for selection in map_to_raw_data(pulse_list,avg_num=avg_num,window_length=window_length,offset=trans.start):
    plt.figure()
    plt.plot(ts*np.arange(data[selection].shape[0]),data[selection].astype(np.float)**2)
    a,b = plt.ylim()
    #print data[selection].shape[0]//2
    plt.vlines(ts*data[selection].shape[0]//2,a,b)
    
#plot(pulse.nonzero()[0][:-1],np.diff(pulse.nonzero()[0]),'.')
#ylim(0,5)







